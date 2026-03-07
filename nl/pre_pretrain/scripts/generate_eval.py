#!/usr/bin/env python3
"""Generate synthetic evaluation data for pptrain.

Creates eval.jsonl with examples at alpha=1.0 (full difficulty).
max_input_size is calculated as 6 * max_lookahead.
"""

import argparse
import json
from pathlib import Path

import sys
# Add nl/ directory to path (parent.parent.parent from scripts/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from synthetic import build_heldout_set


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation data")
    parser.add_argument("--max_lookahead", type=int, default=32,
                        help="Maximum lookahead for generation (default: 32)")
    parser.add_argument("--size", type=int, default=2000,
                        help="Number of eval examples to generate (default: 2000)")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/gautschi/mnickel/data/nl_splits",
                        help="Output directory for eval.jsonl")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed for generation (default: 12345)")
    args = parser.parse_args()

    # Calculate max_input_size from max_lookahead
    max_input_size = 6 * args.max_lookahead

    print(f"Generating {args.size} eval examples...")
    print(f"  max_lookahead: {args.max_lookahead}")
    print(f"  max_input_size: {max_input_size} (6 * {args.max_lookahead})")
    print(f"  seed: {args.seed}")
    print(f"  output_dir: {args.output_dir}")

    # Generate examples using build_heldout_set (alpha=1.0)
    examples, _ = build_heldout_set(
        size=args.size,
        max_input_size=max_input_size,
        max_lookahead=args.max_lookahead,
        seed=args.seed,
        task="search",
    )

    # Save to output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "eval.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
