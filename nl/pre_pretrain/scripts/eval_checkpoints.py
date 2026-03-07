#!/usr/bin/env python3
"""Evaluate pretrain checkpoints on lookahead task.

Usage:
    python eval_checkpoints.py \
        --run_dirs /scratch/gautschi/mnickel/pretrain_fresh \
                   /scratch/gautschi/mnickel/pretrain_mix \
                   /scratch/gautschi/mnickel/pretrain_from_pptrain \
        --eval_data /scratch/gautschi/mnickel/data/nl_splits/eval.jsonl \
        --output_dir /scratch/gautschi/mnickel/eval_results
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import GenerationConfig

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import load_jsonl_examples, load_model_and_tokenizer

def discover_checkpoints(run_dir: Path, filter_pattern: Optional[str] = None) -> List[Tuple[int, Path]]:
    """Find all checkpoint directories and extract step numbers.

    Returns:
        List of (step_number, checkpoint_path) sorted by step
    """
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return []

    pattern = filter_pattern or "step_*"
    checkpoints = []

    for ckpt_dir in ckpt_root.glob(pattern):
        match = re.match(r"step_(\d+)", ckpt_dir.name)
        if match:
            step = int(match.group(1))
            if (ckpt_dir / "model.safetensors").exists():
                checkpoints.append((step, ckpt_dir))

    return sorted(checkpoints, key=lambda x: x[0])


def load_checkpoint_weights(model, checkpoint_path: Path) -> None:
    """Load model weights from accelerate checkpoint."""
    safetensors_path = checkpoint_path / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"No model.safetensors in {checkpoint_path}")

    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict)


@torch.no_grad()
def evaluate_checkpoint(
    model,
    tokenizer,
    examples: List[Dict],
    batch_size: int = 32,
    max_new_tokens: int = 8,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate a single checkpoint on the lookahead task."""
    model.eval()

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    correct_count = 0
    total_count = 0

    # Process in batches
    for batch_start in range(0, len(examples), batch_size):
        batch = examples[batch_start : batch_start + batch_size]
        prompts = [ex["prompt"] for ex in batch]
        answer_lists = [ex["answers"] for ex in batch]

        # Tokenize with left-padding for generation
        tokenizer.padding_side = "left"
        encodings = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        # Get the full padded input length (same for all items in batch)
        input_len = encodings.input_ids.shape[1]

        # Generate
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                **encodings,
                generation_config=generation_config,
            )

        # Extract predictions and check correctness
        for output_ids, valid_answers in zip(outputs, answer_lists):
            # Extract only the newly generated tokens (after the full padded input)
            gen_ids = output_ids[input_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # Extract first word (split on whitespace/punctuation)
            pred_word = re.split(r"[.,\s]+", gen_text)[0] if gen_text else ""

            # Case-sensitive match against any valid answer
            is_correct = any(pred_word == ans for ans in valid_answers)

            correct_count += int(is_correct)
            total_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0.0

    return {
        "correct": correct_count,
        "total": total_count,
        "accuracy": accuracy,
    }


def evaluate_all_checkpoints(
    run_dirs: List[Path],
    eval_examples: List[Dict],
    model,
    tokenizer,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    checkpoint_filter: Optional[str] = None,
    existing_results: Optional[Dict] = None,
    save_callback=None,
) -> Dict[str, List[Dict]]:
    """Evaluate all checkpoints from multiple runs."""
    results = existing_results or {}

    for run_dir in run_dirs:
        run_name = run_dir.name

        if run_name not in results:
            results[run_name] = []

        # Get already evaluated steps
        evaluated_steps = {r["step"] for r in results[run_name]}

        checkpoints = discover_checkpoints(run_dir, checkpoint_filter)
        print(f"\n{run_name}: Found {len(checkpoints)} checkpoints")

        # Filter out already evaluated
        to_eval = [(s, p) for s, p in checkpoints if s not in evaluated_steps]
        print(f"  {len(to_eval)} remaining to evaluate")

        for step, ckpt_path in tqdm(to_eval, desc=run_name):
            # Load weights
            load_checkpoint_weights(model, ckpt_path)

            # Evaluate
            metrics = evaluate_checkpoint(
                model,
                tokenizer,
                eval_examples,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                device=device,
            )

            results[run_name].append(
                {
                    "step": step,
                    "accuracy": metrics["accuracy"],
                    "correct": metrics["correct"],
                    "total": metrics["total"],
                }
            )

            # Save incrementally
            if save_callback:
                results[run_name].sort(key=lambda x: x["step"])
                save_callback(results)

        # Sort by step
        results[run_name].sort(key=lambda x: x["step"])

    return results


def save_results(results: Dict, metadata: Dict, output_path: Path) -> None:
    """Save evaluation results to JSON."""
    output = {
        "metadata": metadata,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)


def export_to_csv(results: Dict, output_path: Path) -> None:
    """Export results to CSV for easy analysis."""
    import csv

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "step", "accuracy", "correct", "total"])

        for run_name, run_results in results.items():
            for r in run_results:
                writer.writerow([run_name, r["step"], r["accuracy"], r["correct"], r["total"]])


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrain checkpoints on lookahead task")
    parser.add_argument(
        "--run_dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to pretrain run directories",
    )
    parser.add_argument(
        "--eval_data",
        type=Path,
        required=True,
        help="Path to eval.jsonl file",
    )
    parser.add_argument(
        "--model_name",
        default="EleutherAI/pythia-160m",
        help="Base model architecture",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="Use subset of eval data for testing",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./eval_results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="checkpoint_eval",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--checkpoint_filter",
        type=str,
        default=None,
        help="Glob pattern to filter checkpoints (e.g., 'step_*00')",
    )
    parser.add_argument(
        "--resume_from",
        type=Path,
        default=None,
        help="Resume from partial results JSON",
    )

    args = parser.parse_args()

    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.output_prefix}.json"

    # Load eval data
    print(f"Loading eval data from: {args.eval_data}")
    eval_examples = load_jsonl_examples(args.eval_data, subset_size=args.subset_size)
    print(f"Loaded {len(eval_examples)} evaluation examples")

    # Create model and tokenizer
    print(f"Initializing model architecture: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model = model.to(args.device)
    model.eval()

    # Load existing results if resuming
    existing_results = None
    if args.resume_from and args.resume_from.exists():
        with args.resume_from.open() as f:
            data = json.load(f)
            existing_results = data.get("results", {})
        print(f"Resuming from {args.resume_from}")
    elif json_path.exists():
        # Auto-resume from output path
        with json_path.open() as f:
            data = json.load(f)
            existing_results = data.get("results", {})
        print(f"Auto-resuming from {json_path}")

    # Prepare metadata
    metadata = {
        "eval_data_path": str(args.eval_data),
        "eval_examples": len(eval_examples),
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "timestamp": datetime.now().isoformat(),
        "run_dirs": [str(d) for d in args.run_dirs],
    }

    # Create save callback for incremental saving
    def save_callback(results):
        save_results(results, metadata, json_path)

    # Evaluate all checkpoints
    results = evaluate_all_checkpoints(
        run_dirs=args.run_dirs,
        eval_examples=eval_examples,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        checkpoint_filter=args.checkpoint_filter,
        existing_results=existing_results,
        save_callback=save_callback,
    )

    # Final save
    save_results(results, metadata, json_path)
    print(f"Saved results to: {json_path}")

    csv_path = args.output_dir / f"{args.output_prefix}.csv"
    export_to_csv(results, csv_path)
    print(f"Saved CSV to: {csv_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
