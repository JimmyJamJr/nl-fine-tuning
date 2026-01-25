#!/usr/bin/env python3
"""
Build all data splits (eval, mixin, test) for pre-pretraining and pretraining.

Generates three non-overlapping sets at alpha=1.0:
  - eval.jsonl:  Used for periodic evaluation during both phases
  - mixin.jsonl: Mixed into C4 during pretraining (2-5%)
  - test.jsonl:  Final evaluation only

Usage:
  python build_data_splits.py \
    --eval_size 2000 \
    --mixin_size 50000 \
    --test_size 5000 \
    --max_input_size 384 \
    --max_lookahead 64 \
    --seed 12345 \
    --scratch_dir /scratch/gautschi/mnickel \
    --out_subdir nl_data
"""

import argparse
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

from nl.pre_pretrain.synthetic import build_heldout_set


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_out_dir(
    *,
    scratch: Path,
    out_subdir: str,
    max_input_size: int,
    max_lookahead: int,
    seed: int,
) -> Path:
    name = f"mis{max_input_size}_look{max_lookahead}_seed{seed}"
    return scratch / out_subdir / name


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--eval_size", type=int, default=2000, help="Size of eval set")
    p.add_argument("--mixin_size", type=int, default=50000, help="Size of mixin set for pretraining")
    p.add_argument("--test_size", type=int, default=5000, help="Size of test set")

    p.add_argument("--max_input_size", type=int, required=True)
    p.add_argument("--max_lookahead", type=int, required=True)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--task", type=str, default="search")
    p.add_argument("--max_attempts_per_example", type=int, default=2000)

    p.add_argument("--scratch_dir", type=str, required=True, help="Base scratch directory")
    p.add_argument("--out_subdir", type=str, default="data/nl_splits", help="Subdirectory for output")

    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def generate_split(
    name: str,
    size: int,
    reserved_inputs: Set[str],
    args,
    seed_offset: int,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Generate a single data split, returning examples and new prompts."""
    print(f"Generating {name} ({size} examples)...", flush=True)
    t0 = time.time()

    examples, new_prompts = build_heldout_set(
        size=size,
        max_input_size=args.max_input_size,
        max_lookahead=args.max_lookahead,
        seed=args.seed + seed_offset,
        task=args.task,
        reserved_inputs=reserved_inputs,
        max_attempts_per_example=args.max_attempts_per_example,
    )

    elapsed = time.time() - t0
    print(f"  Generated {len(examples)} examples in {elapsed:.1f}s", flush=True)

    return examples, new_prompts


def main():
    args = parse_args()

    scratch = Path(args.scratch_dir).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    out_dir = build_out_dir(
        scratch=scratch,
        out_subdir=args.out_subdir,
        max_input_size=args.max_input_size,
        max_lookahead=args.max_lookahead,
        seed=args.seed,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_path = out_dir / "eval.jsonl"
    mixin_path = out_dir / "mixin.jsonl"
    test_path = out_dir / "test.jsonl"
    meta_path = out_dir / "meta.json"

    # Check for existing files
    existing = [p for p in [eval_path, mixin_path, test_path] if p.exists()]
    if existing and not args.overwrite:
        raise RuntimeError(
            f"Files already exist: {existing}. Use --overwrite or change args."
        )

    t0_total = time.time()
    all_prompts: Set[str] = set()

    # Generate splits in order, accumulating reserved prompts
    # Use different seed offsets to ensure different random sequences
    eval_examples, eval_prompts = generate_split(
        "eval", args.eval_size, all_prompts, args, seed_offset=0
    )
    all_prompts.update(eval_prompts)

    test_examples, test_prompts = generate_split(
        "test", args.test_size, all_prompts, args, seed_offset=100_000
    )
    all_prompts.update(test_prompts)

    mixin_examples, mixin_prompts = generate_split(
        "mixin", args.mixin_size, all_prompts, args, seed_offset=200_000
    )
    all_prompts.update(mixin_prompts)

    # Write files
    write_jsonl(eval_path, eval_examples)
    write_jsonl(test_path, test_examples)
    write_jsonl(mixin_path, mixin_examples)

    # Compute checksums
    eval_sha = sha256_file(eval_path)
    test_sha = sha256_file(test_path)
    mixin_sha = sha256_file(mixin_path)

    total_elapsed = time.time() - t0_total

    meta = {
        "task": args.task,
        "max_input_size": args.max_input_size,
        "max_lookahead": args.max_lookahead,
        "seed": args.seed,
        "alpha": 1.0,
        "max_attempts_per_example": args.max_attempts_per_example,
        "splits": {
            "eval": {
                "size": len(eval_examples),
                "path": str(eval_path),
                "sha256": eval_sha,
            },
            "test": {
                "size": len(test_examples),
                "path": str(test_path),
                "sha256": test_sha,
            },
            "mixin": {
                "size": len(mixin_examples),
                "path": str(mixin_path),
                "sha256": mixin_sha,
            },
        },
        "total_unique_prompts": len(all_prompts),
        "created_unix": int(time.time()),
        "created_seconds": round(total_elapsed, 3),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print()
    print(f"Wrote: {eval_path} ({len(eval_examples)} examples)")
    print(f"Wrote: {test_path} ({len(test_examples)} examples)")
    print(f"Wrote: {mixin_path} ({len(mixin_examples)} examples)")
    print(f"Wrote: {meta_path}")
    print()
    print(f"Total unique prompts: {len(all_prompts)}")
    print(f"Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
