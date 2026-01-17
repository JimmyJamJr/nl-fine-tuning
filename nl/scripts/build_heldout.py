#!/usr/bin/env python3
"""
Build a fixed heldout set (alpha=1.0) and write it to scratch as JSONL + meta.

Usage example:
  python build_heldout.py \
    --size 20000 \
    --max_input_size 768 \
    --max_lookahead 128 \
    --seed 999 \
    --out_subdir nl_heldout
    --scratch_dir /scratch/gautschi/mnickel
    
"""

import argparse
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List

from synthetic import build_heldout_set


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
    task: str,
    size: int,
    max_input_size: int,
    max_lookahead: int,
    seed: int,
) -> Path:
    name = f"{task}_size{size}_mis{max_input_size}_look{max_lookahead}_seed{seed}"
    return scratch / out_subdir / name


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--size", type=int, required=True)
    p.add_argument("--max_input_size", type=int, required=True)
    p.add_argument("--max_lookahead", type=int, required=True)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--task", type=str, default="search")
    p.add_argument("--max_attempts_per_example", type=int, default=2000)

    p.add_argument(
        "--scratch_dir",
        type=str,
        default='./',
    )
    p.add_argument(
        "--out_subdir",
        type=str,
        default="nl_heldout",
    )

    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    scratch = Path(args.scratch_dir).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    out_dir = build_out_dir(
        scratch=scratch,
        out_subdir=args.out_subdir,
        task=args.task,
        size=args.size,
        max_input_size=args.max_input_size,
        max_lookahead=args.max_lookahead,
        seed=args.seed,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    heldout_path = out_dir / "heldout.jsonl"
    meta_path = out_dir / "meta.json"

    if heldout_path.exists() and not args.overwrite:
        raise RuntimeError(
            f"Heldout already exists at {heldout_path}. Use --overwrite or change args."
        )

    t0 = time.time()
    heldout, heldout_inputs = build_heldout_set(
        size=args.size,
        max_input_size=args.max_input_size,
        max_lookahead=args.max_lookahead,
        seed=args.seed,
        task=args.task,
        max_attempts_per_example=args.max_attempts_per_example,
    )

    write_jsonl(heldout_path, heldout)
    digest = sha256_file(heldout_path)

    meta = {
        "task": args.task,
        "size": args.size,
        "max_input_size": args.max_input_size,
        "max_lookahead": args.max_lookahead,
        "seed": args.seed,
        "alpha": 1.0,
        "stage": args.max_lookahead,
        "max_attempts_per_example": args.max_attempts_per_example,
        "num_unique_prompts": len(heldout_inputs),
        "heldout_sha256": digest,
        "created_unix": int(time.time()),
        "created_seconds": round(time.time() - t0, 3),
        "path": str(heldout_path),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote: {heldout_path}")
    print(f"Wrote: {meta_path}")
    print(f"sha256: {digest}")
    print(f"unique_prompts: {len(heldout_inputs)}")


if __name__ == "__main__":
    main()
