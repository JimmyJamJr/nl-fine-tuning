import argparse
import json
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model_name", default="EleutherAI/pythia-160m")
    p.add_argument("--split", default="train")

    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--total_blocks", type=int, required=True)
    p.add_argument("--blocks_per_shard", type=int, default=10_000)

    # Important: avoid taking “first N” in dataset order
    p.add_argument("--shuffle_seed", type=int, default=1337)
    p.add_argument("--shuffle_buffer", type=int, default=100_000)

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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

    shard_path = os.path.join(args.out_dir, f"shard_{shard_idx:05d}.bin")
    f = open(shard_path, "wb")

    for ex in ds:
        text = ex.get("text", "")
        if not text:
            continue

        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue

        buffer.extend(ids + [eos])

        while len(buffer) >= block_len and written < args.total_blocks:
            chunk = buffer[:block_len]
            buffer = buffer[block_len:]

            f.write(np.asarray(chunk, dtype=dtype).tobytes())
            written += 1
            in_shard += 1

            if in_shard >= args.blocks_per_shard and written < args.total_blocks:
                f.close()
                shard_idx += 1
                in_shard = 0
                shard_path = os.path.join(args.out_dir, f"shard_{shard_idx:05d}.bin")
                f = open(shard_path, "wb")

        if written >= args.total_blocks:
            break

    f.close()

    meta = {
        "dataset": "allenai/c4",
        "config": "en",
        "split": args.split,
        "model_name": args.model_name,
        "seq_len": args.seq_len,
        "block_len": block_len,
        "dtype": "uint16",
        "total_blocks": written,
        "blocks_per_shard": args.blocks_per_shard,
        "num_shards": shard_idx + 1,
        "shuffle_seed": args.shuffle_seed,
        "shuffle_buffer": args.shuffle_buffer,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as out:
        json.dump(meta, out, indent=2)

    print(f"Wrote {written} blocks into {meta['num_shards']} shards at {args.out_dir}")


if __name__ == "__main__":
    main()
