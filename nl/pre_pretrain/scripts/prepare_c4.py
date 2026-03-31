#!/usr/bin/env python3
"""
Prepare C4 dataset for pretraining.

Downloads C4 English from HuggingFace, tokenizes with Pythia tokenizer,
packs sequences with EOS separators, and saves as memory-mapped binary shards.

Usage:
    python prepare_c4.py --output_dir /scratch/gautschi/$USER/data/c4_tokenized
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare C4 dataset for pretraining")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save tokenized shards",
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=3_500_000_000,
        help="Target number of tokens (default: 3.5B, enough for 3100 steps with headroom)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Sequence length for packing (default: 2048)",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=100_000_000,
        help="Tokens per shard (default: 100M)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="EleutherAI/pythia-160m",
        help="Tokenizer to use (default: EleutherAI/pythia-160m)",
    )
    return parser.parse_args()


def pack_blocks(token_buffer: list, block_len: int, eos_token: int) -> list:
    """Pack tokens into fixed-length blocks with EOS separators.

    Uses block_len = seq_len + 1 so that we get seq_len tokens of prediction
    (input = block[:-1], labels = block[1:]).
    """
    blocks = []
    current_block = []

    for tokens in token_buffer:
        # Add tokens to current block
        for tok in tokens:
            current_block.append(tok)
            if len(current_block) == block_len:
                blocks.append(current_block)
                current_block = []
        # Add EOS between documents
        if current_block:
            current_block.append(eos_token)
            if len(current_block) == block_len:
                blocks.append(current_block)
                current_block = []

    return blocks


def save_shard(sequences: list, shard_path: Path, dtype=np.uint16):
    """Save sequences as memory-mapped binary file."""
    arr = np.array(sequences, dtype=dtype)
    arr.tofile(shard_path)
    return arr.shape


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_token = tokenizer.eos_token_id

    block_len = args.seq_len + 1  # For causal LM: input = block[:-1], labels = block[1:]

    print(f"Target: {args.target_tokens:,} tokens")
    print(f"Sequence length: {args.seq_len}")
    print(f"Block length: {block_len}")
    print(f"Shard size: {args.shard_size:,} tokens")
    print(f"Output: {output_dir}")

    # Load C4 dataset (streaming to avoid downloading entire dataset)
    print("Loading C4 dataset (streaming)...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

    # Shuffle with seed for reproducibility
    # Large buffer_size needed because C4 is NOT pre-shuffled (maintains Common Crawl ordering)
    # buffer_size=1M provides much better global mixing at cost of ~10-20GB RAM during prep
    dataset = dataset.shuffle(seed=args.seed, buffer_size=1_000_000)

    total_tokens = 0
    shard_idx = 0
    token_buffer = []
    buffer_tokens = 0

    pbar = tqdm(total=args.target_tokens, unit="tok", unit_scale=True, desc="Tokenizing")

    for example in dataset:
        # Tokenize text
        tokens = tokenizer.encode(example["text"], add_special_tokens=False)

        if not tokens:
            continue

        token_buffer.append(tokens)
        buffer_tokens += len(tokens)

        # When buffer is full, pack and save shard
        if buffer_tokens >= args.shard_size:
            blocks = pack_blocks(token_buffer, block_len, eos_token)

            if blocks:
                shard_path = output_dir / f"shard_{shard_idx:05d}.bin"
                shape = save_shard(blocks, shard_path)
                shard_tokens = shape[0] * args.seq_len  # Predict seq_len tokens per block

                print(f"\nSaved {shard_path.name}: {shape[0]:,} blocks, {shard_tokens:,} tokens")

                total_tokens += shard_tokens
                pbar.update(shard_tokens)
                shard_idx += 1

            token_buffer = []
            buffer_tokens = 0

        if total_tokens >= args.target_tokens:
            break

    # Save any remaining tokens
    if token_buffer:
        blocks = pack_blocks(token_buffer, block_len, eos_token)
        if blocks:
            shard_path = output_dir / f"shard_{shard_idx:05d}.bin"
            shape = save_shard(blocks, shard_path)
            shard_tokens = shape[0] * args.seq_len

            print(f"\nSaved {shard_path.name}: {shape[0]:,} blocks, {shard_tokens:,} tokens")

            total_tokens += shard_tokens
            pbar.update(shard_tokens)
            shard_idx += 1

    pbar.close()

    # Calculate blocks per shard for metadata
    total_blocks = total_tokens // args.seq_len
    blocks_per_shard = args.shard_size // args.seq_len

    # Save metadata (compatible with PackedC4Dataset)
    meta = {
        "total_tokens": total_tokens,
        "total_blocks": total_blocks,
        "blocks_per_shard": blocks_per_shard,
        "num_shards": shard_idx,
        "seq_len": args.seq_len,
        "block_len": block_len,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "dtype": "uint16",
    }

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Done! Prepared {total_tokens:,} tokens in {shard_idx} shards")
    print(f"Metadata saved to {meta_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
