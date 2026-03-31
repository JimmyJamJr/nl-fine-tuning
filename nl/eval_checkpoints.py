#!/usr/bin/env python3
"""
Evaluate existing checkpoints at a fixed target L.

Usage:
    # Just provide the latest job ID — it auto-discovers the full chain
    python eval_checkpoints.py \
        --job_id 8324385 \
        --target_L 96 \
        --eval_samples 500 \
        --output eval_at_L96_pythia410m.json

    # Evaluate regular checkpoints every 2000 steps (for no-curriculum runs)
    python eval_checkpoints.py \
        --job_id 8313552 \
        --target_L 96 \
        --checkpoint_mode regular --step_interval 2000 \
        --eval_samples 500 \
        --output eval_at_L96_qwen_nocurr.json
"""

import argparse
import json
import os
import re
import sys
import math
import random
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel


def resolve_job_chain(job_id: str, base_dir: str) -> List[str]:
    """Walk backwards through resume_from_job links to find the full job chain."""
    chain = []
    current = job_id

    while current:
        chain.append(current)
        meta_path = os.path.join(base_dir, f"job_{current}", "run_meta.json")
        if not os.path.exists(meta_path):
            break
        with open(meta_path) as f:
            meta = json.load(f)
        cli = meta.get("cli", "")
        m = re.search(r"--resume_from_job\s+(\d+)", cli)
        if m:
            current = m.group(1)
        else:
            break

    chain.reverse()  # oldest first
    return chain


def extract_run_config(job_id: str, base_dir: str) -> dict:
    """Extract model_name, use_lora, max_input_size, max_lookahead from run_meta."""
    meta_path = os.path.join(base_dir, f"job_{job_id}", "run_meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    cli = meta.get("cli", "")

    config = {}
    for flag, key in [
        ("--model_name", "model_name"),
        ("--max_input_size", "max_input_size"),
        ("--max_lookahead", "max_lookahead"),
    ]:
        m = re.search(rf"{flag}\s+(\S+)", cli)
        if m:
            config[key] = m.group(1)

    config["use_lora"] = "--use_lora" in cli
    # Convert numeric strings
    for k in ["max_input_size", "max_lookahead"]:
        if k in config:
            config[k] = int(config[k])

    return config


def find_checkpoints(job_ids: List[str], base_dir: str, mode: str, step_interval: int):
    """Find all checkpoints across a chain of jobs.

    Returns list of (checkpoint_path, metadata_dict) sorted by step.
    """
    checkpoints = []

    for jid in job_ids:
        job_dir = os.path.join(base_dir, f"job_{jid}")
        if not os.path.isdir(job_dir):
            print(f"Warning: {job_dir} not found, skipping")
            continue

        if mode == "stage":
            # Stage checkpoints: stage_checkpoints/stage_X_step_Y_LZ/
            stage_dir = os.path.join(job_dir, "stage_checkpoints")
            if os.path.isdir(stage_dir):
                for name in sorted(os.listdir(stage_dir)):
                    m = re.match(r"stage_(\d+)_step_(\d+)_L(\d+)", name)
                    if m:
                        ckpt_path = os.path.join(stage_dir, name)
                        checkpoints.append((ckpt_path, {
                            "job_id": jid,
                            "stage": int(m.group(1)),
                            "step": int(m.group(2)),
                            "train_L": int(m.group(3)),
                        }))

        elif mode == "regular":
            # Regular checkpoints: checkpoint-XXXXX/
            for name in sorted(os.listdir(job_dir)):
                m = re.match(r"checkpoint-(\d+)$", name)
                if m:
                    step = int(m.group(1))
                    if step % step_interval == 0:
                        ckpt_path = os.path.join(job_dir, name)
                        checkpoints.append((ckpt_path, {
                            "job_id": jid,
                            "step": step,
                        }))

    # Sort by step, deduplicate (later jobs override earlier for same step)
    checkpoints.sort(key=lambda x: x[1]["step"])
    seen_steps = {}
    for ckpt_path, meta in checkpoints:
        seen_steps[meta["step"]] = (ckpt_path, meta)
    return sorted(seen_steps.values(), key=lambda x: x[1]["step"])


def generate_eval_data(target_L, max_input_size, max_lookahead, tokenizer, n_samples, seed=42):
    """Generate eval data at a specific target L using the C++ generator."""
    from nl_generator import NaturalLanguageGraphGenerator
    from tuning_nl import alpha_for_lookahead, effective_search_L, _determine_task_type, _get_end_tokens, _tokenize_leading_space

    alpha = alpha_for_lookahead(target_L, max_input_size)
    actual_L = effective_search_L(alpha, max_input_size, max_lookahead_cap=max_lookahead)
    print(f"Target L={target_L}, computed alpha={alpha:.4f}, actual effective L={actual_L}")

    g = NaturalLanguageGraphGenerator(max_input_size, seed=seed)
    eval_inputs, eval_labels = [], []

    max_len = getattr(tokenizer, "model_max_length", 512)
    attempts = 0
    reserved = set()

    while len(eval_inputs) < n_samples and attempts < n_samples * 20:
        attempts += 1
        batch = g.generate_batch("search", batch_size=1, reserved_inputs=reserved,
                                 alpha=alpha, max_lookahead=max_lookahead)
        if not (batch and batch[0] and batch[0].output_texts):
            continue
        ex = batch[0]
        if ex.input_text in reserved:
            continue

        # Check tokenization fits
        prompt_ids = tokenizer(ex.input_text, add_special_tokens=True, truncation=False)["input_ids"]
        chosen = ex.output_texts[0]
        ans_ids = _tokenize_leading_space(tokenizer, chosen)
        task_type = _determine_task_type("search", ex.input_text)
        end_ids = tokenizer(_get_end_tokens(task_type), add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + len(ans_ids) + len(end_ids) > max_len:
            continue

        eval_inputs.append(ex.input_text)
        eval_labels.append(ex.output_texts)
        reserved.add(ex.input_text)

    print(f"Generated {len(eval_inputs)} eval samples (attempted {attempts})")
    return eval_inputs, eval_labels


def load_model(checkpoint_path, model_name, use_lora, cache_dir, device):
    """Load a model from a checkpoint."""
    if use_lora:
        # Load base model + LoRA adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = model.merge_and_unload()
    else:
        # Full FT: checkpoint has full model weights
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

    model.eval()
    model.to(device)
    return model


def evaluate_greedy(model, tokenizer, inputs, labels, device, print_mistakes=0):
    """Run greedy evaluation (single GPU version)."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    greedy_config = GenerationConfig(
        max_new_tokens=24,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    correct_first = 0
    correct_full = 0
    printed = 0

    from tuning_nl import _tokenize_leading_space

    for i, (x, ys) in enumerate(zip(inputs, labels)):
        ys = ys if isinstance(ys, list) else [ys]
        enc = tokenizer(x, return_tensors="pt").to(device)
        prompt_len = enc.input_ids.shape[1]

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen_out = model.generate(
                **enc,
                generation_config=greedy_config,
            )

        gen_ids = gen_out[0][prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        pred_word = re.split(r"[.,\s]+", gen_text)[0] if gen_text else ""

        # First token accuracy
        valid_first_ids = set()
        for y in ys:
            toks = _tokenize_leading_space(tokenizer, y)
            if toks:
                valid_first_ids.add(toks[0])
        first_ok = len(gen_ids) > 0 and gen_ids[0].item() in valid_first_ids
        correct_first += int(first_ok)

        # Full word accuracy
        full_ok = any(pred_word.lower() == y.lower() for y in ys)
        correct_full += int(full_ok)

        if print_mistakes > 0 and not full_ok and printed < print_mistakes:
            print(f"  [WRONG] Gold: {ys}, Pred: '{pred_word}', Full: '{gen_text[:60]}'")
            printed += 1

        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i+1}/{len(inputs)}: "
                  f"first={correct_first/(i+1):.1%} full={correct_full/(i+1):.1%}")

    total = len(inputs)
    return {
        "first_token_acc": correct_first / total if total else 0,
        "full_word_acc": correct_full / total if total else 0,
        "first_token_hits": correct_first,
        "full_word_hits": correct_full,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints at a fixed target L")
    parser.add_argument("--job_id", type=str, required=True,
                        help="Latest job ID (auto-discovers full chain via resume_from_job)")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Base model name (auto-detected from run_meta if omitted)")
    parser.add_argument("--target_L", type=int, required=True, help="Target lookahead to evaluate at")
    parser.add_argument("--max_input_size", type=int, default=None,
                        help="Max input size (auto-detected from run_meta if omitted)")
    parser.add_argument("--max_lookahead", type=int, default=None,
                        help="Max lookahead cap (auto-detected from run_meta if omitted)")
    parser.add_argument("--eval_samples", type=int, default=500, help="Number of eval samples")
    parser.add_argument("--eval_seed", type=int, default=99999, help="Seed for eval data generation")
    parser.add_argument("--use_lora", action="store_true", default=None,
                        help="Model uses LoRA adapters (auto-detected if omitted)")
    parser.add_argument("--cache_dir", type=str, default=None, help="HF cache dir")
    parser.add_argument("--checkpoint_mode", choices=["stage", "regular"], default="stage",
                        help="Which checkpoints to evaluate")
    parser.add_argument("--step_interval", type=int, default=2000,
                        help="Step interval for regular checkpoint mode")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--print_mistakes", type=int, default=0, help="Print N mistakes per checkpoint")
    parser.add_argument("--base_dir", type=str,
                        default="/scratch/gautschi/huan2073/nl_output/search",
                        help="Base directory for job outputs")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file, skipping already-evaluated checkpoints")

    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = os.environ.get("HF_HOME", "/scratch/gautschi/huan2073/model_cache")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve full job chain from the latest job ID
    job_ids = resolve_job_chain(args.job_id, args.base_dir)
    print(f"Resolved job chain: {' -> '.join(job_ids)}")

    # Auto-detect config from run_meta
    run_config = extract_run_config(args.job_id, args.base_dir)
    if run_config:
        print(f"Auto-detected config: {run_config}")

    if args.model_name is None:
        args.model_name = run_config.get("model_name")
        if args.model_name is None:
            print("ERROR: Could not auto-detect model_name. Provide --model_name.")
            sys.exit(1)
    if args.max_input_size is None:
        args.max_input_size = run_config.get("max_input_size")
        if args.max_input_size is None:
            print("ERROR: Could not auto-detect max_input_size. Provide --max_input_size.")
            sys.exit(1)
    if args.max_lookahead is None:
        args.max_lookahead = run_config.get("max_lookahead", 128)
    if args.use_lora is None:
        args.use_lora = run_config.get("use_lora", False)

    print(f"Config: model={args.model_name}, max_input_size={args.max_input_size}, "
          f"max_lookahead={args.max_lookahead}, use_lora={args.use_lora}")

    # Find checkpoints
    checkpoints = find_checkpoints(job_ids, args.base_dir, args.checkpoint_mode, args.step_interval)
    print(f"Found {len(checkpoints)} checkpoints to evaluate")
    for ckpt_path, meta in checkpoints:
        print(f"  step={meta['step']} {meta}")

    if not checkpoints:
        print("No checkpoints found!")
        sys.exit(1)

    # Load existing results if resuming
    existing_results = []
    evaluated_steps = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            data = json.load(f)
            existing_results = data.get("results", [])
            evaluated_steps = {r["step"] for r in existing_results}
            print(f"Resuming: {len(evaluated_steps)} checkpoints already evaluated")

    # Load tokenizer (once)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, trust_remote_code=True
    )

    # Generate eval data (once, with fixed seed for reproducibility)
    print(f"\nGenerating eval data at L={args.target_L}...")
    eval_inputs, eval_labels = generate_eval_data(
        args.target_L, args.max_input_size, args.max_lookahead,
        tokenizer, args.eval_samples, seed=args.eval_seed
    )

    # Evaluate each checkpoint
    results = list(existing_results)

    for i, (ckpt_path, meta) in enumerate(checkpoints):
        if meta["step"] in evaluated_steps:
            print(f"\n[{i+1}/{len(checkpoints)}] Skipping step {meta['step']} (already evaluated)")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(checkpoints)}] Evaluating: {meta}")
        print(f"  Checkpoint: {ckpt_path}")

        try:
            model = load_model(ckpt_path, args.model_name, args.use_lora, args.cache_dir, device)
            metrics = evaluate_greedy(model, tokenizer, eval_inputs, eval_labels, device,
                                      print_mistakes=args.print_mistakes)

            result = {**meta, **metrics, "checkpoint_path": ckpt_path}
            results.append(result)

            print(f"  Result: first={metrics['first_token_acc']:.1%} full={metrics['full_word_acc']:.1%}")

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

            # Save incrementally
            output_data = {
                "target_L": args.target_L,
                "max_input_size": args.max_input_size,
                "model_name": args.model_name,
                "use_lora": args.use_lora,
                "eval_samples": len(eval_inputs),
                "eval_seed": args.eval_seed,
                "job_ids": job_ids,
                "results": sorted(results, key=lambda r: r["step"]),
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Done! {len(results)} checkpoints evaluated. Results saved to {args.output}")

    # Print summary table
    print(f"\n{'Step':>8} {'L_train':>8} {'First':>8} {'Full':>8}")
    print("-" * 36)
    for r in sorted(results, key=lambda x: x["step"]):
        train_L = r.get("train_L", "?")
        print(f"{r['step']:>8} {train_L:>8} {r['first_token_acc']:>7.1%} {r['full_word_acc']:>7.1%}")


if __name__ == "__main__":
    main()
