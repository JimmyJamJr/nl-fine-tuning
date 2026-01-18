#!/usr/bin/env python3
"""
Pack C4 data into binary shards for pretraining.

Creates packed binary files where each block is (seq_len + 1) tokens,
ready for efficient loading during pretraining.

Usage:
  python build_pretrain.py \
    --scratch_dir /scratch/user \
    --total_steps 10000 \
    --effective_batch 32 \
    --seq_len 2048

Output structure:
  {scratch_dir}/data/c4_packed/
  ├── shard_00000.bin
  ├── shard_00001.bin
  ├── ...
  └── meta.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--scratch_dir", type=str, required=True, help="Base scratch directory")
    p.add_argument("--out_subdir", type=str, default="data/c4_packed", help="Subdirectory for output")

    p.add_argument("--model_name", default="EleutherAI/pythia-160m")
    p.add_argument("--split", default="train")

    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--total_steps", type=int, default=10000, help="Total pretraining steps")
    p.add_argument("--effective_batch", type=int, default=32, help="Effective batch size (micro_batch * grad_accum)")
    p.add_argument("--blocks_per_shard", type=int, default=10_000)

    # Important: avoid taking "first N" in dataset order
    p.add_argument("--shuffle_seed", type=int, default=1337)
    p.add_argument("--shuffle_buffer", type=int, default=100_000)

    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    # Calculate total blocks needed
    total_blocks = args.total_steps * args.effective_batch

    # Setup output directory
    out_dir = Path(args.scratch_dir) / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not args.overwrite:
        raise RuntimeError(f"Output already exists: {meta_path}. Use --overwrite to replace.")

    print(f"Packing C4 data:", flush=True)
    print(f"  total_steps={args.total_steps}", flush=True)
    print(f"  effective_batch={args.effective_batch}", flush=True)
    print(f"  total_blocks={total_blocks}", flush=True)
    print(f"  seq_len={args.seq_len}", flush=True)
    print(f"  output={out_dir}", flush=True)

    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if len(tokenizer) > 65535:
        raise ValueError("Tokenizer vocab too large for uint16 storage.")

    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer has no eos_token_id; cannot pack documents cleanly.")

    ds = load_dataset("allenai/c4", "en", split=args.split, streaming=True)
    ds = ds.shuffle(seed=args.shuffle_seed, buffer_size=args.shuffle_buffer)

    dtype = np.uint16
    block_len = args.seq_len + 1  # store +1 so training can shift labels

    shard_idx = 0
    in_shard = 0
    written = 0
    buffer = []

    shard_path = out_dir / f"shard_{shard_idx:05d}.bin"
    f = open(shard_path, "wb")

    for ex in ds:
        text = ex.get("text", "")
        if not text:
            continue

        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue

        buffer.extend(ids + [eos])

        while len(buffer) >= block_len and written < total_blocks:
            chunk = buffer[:block_len]
            buffer = buffer[block_len:]

            f.write(np.asarray(chunk, dtype=dtype).tobytes())
            written += 1
            in_shard += 1

            if written % 10000 == 0:
                print(f"  written {written}/{total_blocks} blocks ({100*written/total_blocks:.1f}%)", flush=True)

            if in_shard >= args.blocks_per_shard and written < total_blocks:
                f.close()
                shard_idx += 1
                in_shard = 0
                shard_path = out_dir / f"shard_{shard_idx:05d}.bin"
                f = open(shard_path, "wb")

        if written >= total_blocks:
            break

    f.close()

    elapsed = time.time() - t0

    meta = {
        "dataset": "allenai/c4",
        "config": "en",
        "split": args.split,
        "model_name": args.model_name,
        "seq_len": args.seq_len,
        "block_len": block_len,
        "dtype": "uint16",
        "total_blocks": written,
        "total_steps": args.total_steps,
        "effective_batch": args.effective_batch,
        "blocks_per_shard": args.blocks_per_shard,
        "num_shards": shard_idx + 1,
        "shuffle_seed": args.shuffle_seed,
        "shuffle_buffer": args.shuffle_buffer,
        "created_unix": int(time.time()),
        "created_seconds": round(elapsed, 3),
    }
    with (out_dir / "meta.json").open("w") as out:
        json.dump(meta, out, indent=2)

    print()
    print(f"Wrote {written} blocks into {meta['num_shards']} shards")
    print(f"Output: {out_dir}")
    print(f"Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
