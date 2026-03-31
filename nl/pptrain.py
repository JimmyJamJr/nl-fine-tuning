import argparse
import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List, Tuple, Set

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset

from accelerate import Accelerator, DataLoaderConfiguration

from common import (
    load_model_and_tokenizer,
    tokenize_leading_space,
    full_word_correct,
    checkpoint_dir,
)
from synthetic import SyntheticNL


def load_data_dir_prompts(data_dir: str) -> Set[str]:
    """Load prompts from eval.jsonl as reserved inputs for data leakage prevention."""
    data_path = Path(data_dir)
    prompts: Set[str] = set()

    filepath = data_path / "eval.jsonl"
    if not filepath.exists():
        raise RuntimeError(f"Missing required file: {filepath}")

    count = 0
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = row.get("input_text")
            if t:
                prompts.add(t)
                count += 1

    print(f"  Loaded {count} prompts from eval.jsonl", flush=True)

    return prompts


def synthetic_nl_stream(
    *,
    max_input_size: int,
    max_lookahead: int,
    seed: int,
    rank: int,
    world_size: int,
    reserved_inputs: Optional[Set[str]] = None,
    start_index: int = 0,
    stage: int = 1,
) -> Tuple[SyntheticNL, Iterator[Dict[str, Any]]]:
    ds = SyntheticNL(
        max_input_size=max_input_size,
        max_lookahead=max_lookahead,
        seed=seed,
        rank=rank,
        world_size=world_size,
        stage=stage,
        reserved_inputs=reserved_inputs,
        start_index=start_index,
    )

    def _it():
        for ex in ds:
            outs = ex.get("output_texts") or []
            if outs:
                yield {"prompt": ex["input_text"], "answers": outs}

    return ds, _it()


class PromptAnswerDataset(IterableDataset):
    def __init__(self, *, item_iter: Iterator[Dict[str, Any]]):
        self.item_iter = item_iter

    def __iter__(self):
        yield from self.item_iter


class FilteredIterator:
    def __init__(self, item_iter: Iterator[Dict[str, Any]], tokenizer, seq_len: int):
        self.item_iter = item_iter
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.skipped = 0

    def __iter__(self):
        for item in self.item_iter:
            p_ids = self.tokenizer(item["prompt"], add_special_tokens=True, truncation=False)["input_ids"]
            # Check length using first answer (all answers should be similar length)
            a_ids = tokenize_leading_space(self.tokenizer, item["answers"][0])
            total_len = len(p_ids) + len(a_ids) + 1
            if total_len <= self.seq_len:
                yield item
            else:
                self.skipped += 1
                if self.skipped <= 5:
                    print(f"WARNING: Skipping example (len={total_len} > seq_len={self.seq_len})", flush=True)
                elif self.skipped == 6:
                    print("WARNING: Further seq_len overflow warnings suppressed.", flush=True)


def build_collate_fn(*, tokenizer, seq_len: int):
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids_batch, labels_pred_batch, attn_batch = [], [], []

        for ex in batch:
            p_ids = tokenizer(ex["prompt"], add_special_tokens=True, truncation=False)["input_ids"]
            answers = ex["answers"]

            # Use first answer for training and accuracy checking
            a_ids = tokenize_leading_space(tokenizer, answers[0])
            ids = p_ids + a_ids + [eos_id]
            labels_pred = ([-100] * len(p_ids)) + a_ids + [eos_id]

            if len(ids) > seq_len:
                ids = ids[:seq_len]
                labels_pred = labels_pred[:seq_len]

            L = len(ids)

            input_ids_batch.append(torch.tensor(ids, dtype=torch.long))
            labels_pred_batch.append(torch.tensor(labels_pred, dtype=torch.long))
            attn_batch.append(torch.ones(L, dtype=torch.long))

        max_len = max(x.shape[0] for x in input_ids_batch)

        def pad_to(x: torch.Tensor, val: int, target: int) -> torch.Tensor:
            if x.shape[0] == target:
                return x
            return torch.cat([x, torch.full((target - x.shape[0],), val, dtype=x.dtype)], dim=0)

        return {
            "input_ids": torch.stack([pad_to(x, pad_id, max_len) for x in input_ids_batch]),
            "labels_pred": torch.stack([pad_to(x, -100, max_len) for x in labels_pred_batch]),
            "attention_mask": torch.stack([pad_to(x, 0, max_len) for x in attn_batch]),
        }

    return collate


def prediction_only_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Calculate loss only on prediction tokens (where labels != -100)"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )


def gather_bool_list(flags: List[bool], accelerator: Accelerator) -> List[bool]:
    """Gather boolean flags from all processes into a single list (on all ranks)."""
    t = torch.tensor([1 if f else 0 for f in flags], device=accelerator.device, dtype=torch.long)
    gathered = accelerator.gather(t)
    return [bool(x.item()) for x in gathered]


def get_warmup_scheduler(optimizer, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


def save_train_state(
    *, ckpt: Path, step: int, stage: int, curr_flags: List[bool],
    data_dir: str, micro_batch: int, curr_window: int,
    consumed_index: int, world_size: int, grad_accum: int,
) -> None:
    ckpt.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step, "synthetic_stage": stage,
        "curr_flags": [int(x) for x in curr_flags],
        "data_dir": data_dir, "micro_batch": micro_batch, "curr_window": curr_window,
        "consumed_index": consumed_index, "world_size": world_size, "grad_accum": grad_accum,
    }
    with (ckpt / "train_state.json").open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    rng_state = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng_state, ckpt / "rng_state.pt")


def load_train_state(ckpt: Path) -> Dict[str, Any]:
    p = ckpt / "train_state.json"
    if not p.exists():
        raise RuntimeError(f"Missing train_state.json in checkpoint: {ckpt}")
    with p.open("r", encoding="utf-8") as f:
        state = json.load(f)

    rng_path = ckpt / "rng_state.pt"
    if rng_path.exists():
        state["rng_state"] = torch.load(rng_path, weights_only=False)
    return state


def restore_rng_state(rng_state: Dict[str, Any]) -> None:
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "torch_cpu" in rng_state:
        torch.set_rng_state(rng_state["torch_cpu"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch_cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


def write_train_log(log_path: Path, step: int, loss: float, lr: float,
                    curr_acc: float, stage: int, alpha: float) -> None:
    entry = {
        "step": step, "loss": round(loss, 6), "lr": lr,
        "curr_acc": round(curr_acc, 4), "stage": stage,
        "alpha": round(alpha, 4), "timestamp": time.time(),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def build_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--scratch_dir", required=True)
    p.add_argument("--run_name", type=str, default="pptrain",
                   help="Name of the run directory under scratch_dir (for multiple concurrent runs)")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to data splits directory (from build_data_splits.py)")
    p.add_argument("--model_name", default="EleutherAI/pythia-160m")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--max_steps", type=int, default=None,
                   help="Optional maximum steps (curriculum runs until complete if not set)")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--micro_batch", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=500)

    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--resume_from", type=str, default=None)

    p.add_argument("--max_lookahead", type=int, default=32,
                   help="Max lookahead (max_input_size = 6 * max_lookahead)")
    p.add_argument("--start_stage", type=int, default=4,
                   help="Starting curriculum stage")
    p.add_argument("--stage_step", type=int, default=4,
                   help="Stage increment size when curriculum advances")

    p.add_argument("--curr_window", type=int, default=1000,
                   help="Number of examples to track for accuracy (deque stores per-example bools)")
    p.add_argument("--curr_threshold", type=float, default=0.98,
                   help="Accuracy threshold to advance curriculum stage")

    return p


def main():
    args = build_argparser().parse_args()
    scratch_dir = Path(args.scratch_dir)
    run_dir = scratch_dir / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    train_log_path = run_dir / "logs" / "train.jsonl"

    accelerator = Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
    )

    rank = accelerator.process_index

    if accelerator.is_main_process:
        print(f"num_processes={accelerator.num_processes} "
              f"process_index={rank} "
              f"mixed_precision={accelerator.mixed_precision}", flush=True)

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)

    start_step, start_stage = 0, args.start_stage
    start_index = 0
    curr_deque: deque = deque(maxlen=args.curr_window)
    resume_dir: Optional[Path] = None

    if args.resume_from is not None:
        resume_dir = Path(args.resume_from).expanduser().resolve()
        state = load_train_state(resume_dir)

        start_step = int(state.get("step", 0))
        start_stage = int(state.get("synthetic_stage", 4))
        start_index = int(state.get("consumed_index", start_step * args.micro_batch * args.grad_accum))

        saved_micro_batch = state.get("micro_batch")
        if saved_micro_batch is not None and saved_micro_batch != args.micro_batch:
            if accelerator.is_main_process:
                print(f"WARNING: micro_batch changed from {saved_micro_batch} to {args.micro_batch}", flush=True)

        saved_grad_accum = state.get("grad_accum")
        if saved_grad_accum is not None and saved_grad_accum != args.grad_accum:
            if accelerator.is_main_process:
                print(f"WARNING: grad_accum changed from {saved_grad_accum} to {args.grad_accum}", flush=True)

        saved_world_size = state.get("world_size")
        if saved_world_size is not None and saved_world_size != accelerator.num_processes:
            raise ValueError(f"world_size changed from {saved_world_size} to {accelerator.num_processes}")

        saved_curr_window = state.get("curr_window")
        if saved_curr_window is not None and saved_curr_window != args.curr_window:
            if accelerator.is_main_process:
                print(f"WARNING: curr_window changed from {saved_curr_window} to {args.curr_window}", flush=True)

        for x in state.get("curr_flags", [])[-args.curr_window:]:
            curr_deque.append(bool(x))

        if "rng_state" in state:
            restore_rng_state(state["rng_state"])

        if accelerator.is_main_process:
            print(f"Resuming from {resume_dir}: step={start_step} stage={start_stage} "
                  f"consumed_index={start_index} curr_window_filled={len(curr_deque)}/{args.curr_window}", flush=True)

    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Calculate max_input_size from max_lookahead
    max_input_size = 6 * args.max_lookahead

    data_dir = Path(args.data_dir).expanduser().resolve()
    if accelerator.is_main_process:
        print(f"Loading data splits from: {data_dir}", flush=True)
        print(f"max_lookahead={args.max_lookahead} max_input_size={max_input_size}", flush=True)
    reserved_prompts = load_data_dir_prompts(str(data_dir))
    if accelerator.is_main_process:
        print(f"Total reserved prompts: {len(reserved_prompts)}", flush=True)

    synth_ds, item_iter = synthetic_nl_stream(
        max_input_size=max_input_size, max_lookahead=args.max_lookahead,
        seed=args.seed, rank=accelerator.process_index, world_size=accelerator.num_processes,
        reserved_inputs=reserved_prompts, start_index=start_index, stage=start_stage,
    )
    filtered_iter = FilteredIterator(item_iter, tokenizer, args.seq_len)
    dataset = PromptAnswerDataset(item_iter=filtered_iter)
    collate_fn = build_collate_fn(tokenizer=tokenizer, seq_len=args.seq_len)

    loader = DataLoader(dataset, batch_size=args.micro_batch, collate_fn=collate_fn,
                        num_workers=0, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
                      eps=args.adam_eps, weight_decay=args.weight_decay)
    scheduler = get_warmup_scheduler(optimizer, args.warmup_steps)

    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    model.train()

    if resume_dir is not None:
        accelerator.load_state(str(resume_dir))

    loader_iter = iter(loader)

    step = start_step
    while True:
        step_loss = 0.0
        local_correct = []  # Collect local results before gathering

        for accum_idx in range(args.grad_accum):
            batch = next(loader_iter)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = prediction_only_loss(
                logits=outputs.logits, labels=batch["labels_pred"],
            ) / args.grad_accum
            accelerator.backward(loss)
            step_loss += loss.item()

            correct_list = full_word_correct(outputs.logits.detach(), batch["labels_pred"])
            local_correct.extend(correct_list)

        # Single gather after all micro-batches (1 GPU sync instead of grad_accum syncs)
        step_correct = gather_bool_list(local_correct, accelerator)

        accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Track per-example accuracy (each example is a bool in the deque)
        for correct in step_correct:
            curr_deque.append(correct)

        if accelerator.is_main_process and step % 10 == 0:
            curr_acc = sum(curr_deque) / len(curr_deque) if curr_deque else 0.0
            print(f"step={step} loss={step_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
            write_train_log(train_log_path, step=step, loss=step_loss, lr=scheduler.get_last_lr()[0],
                            curr_acc=curr_acc, stage=synth_ds.stage, alpha=synth_ds.current_alpha())

        # Check curriculum advancement when deque is full (rolling window of examples)
        if len(curr_deque) >= args.curr_window:
            acc = sum(curr_deque) / len(curr_deque)

            if acc >= args.curr_threshold:
                # Check if this is the final stage
                if synth_ds.stage >= synth_ds.max_lookahead:
                    # Save final checkpoint (same as periodic save)
                    ckpt = checkpoint_dir(run_dir, step + 1)
                    accelerator.wait_for_everyone()
                    accelerator.save_state(str(ckpt))
                    if accelerator.is_main_process:
                        save_train_state(
                            ckpt=ckpt, step=step + 1, stage=synth_ds.stage,
                            curr_flags=list(curr_deque), data_dir=args.data_dir,
                            micro_batch=args.micro_batch, curr_window=args.curr_window,
                            consumed_index=synth_ds.current_index,
                            world_size=accelerator.num_processes, grad_accum=args.grad_accum,
                        )
                        print(f"Saved checkpoint: {ckpt}", flush=True)
                        print(f"Curriculum complete at step={step+1}: stage={synth_ds.stage} "
                              f"acc={acc:.4f} >= {args.curr_threshold}", flush=True)
                    accelerator.wait_for_everyone()
                    break

                # Not at final stage - advance to next stage
                old_stage = synth_ds.stage
                synth_ds.increment_stage(args.stage_step)
                curr_deque.clear()

                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    print(f"Stage advancement at step={step+1}: acc={acc:.4f} >= {args.curr_threshold} "
                          f"stage {old_stage} -> {synth_ds.stage} (alpha={synth_ds.current_alpha():.4f})", flush=True)
                accelerator.wait_for_everyone()

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            ckpt = checkpoint_dir(run_dir, step + 1)
            accelerator.wait_for_everyone()
            accelerator.save_state(str(ckpt))
            if accelerator.is_main_process:
                save_train_state(
                    ckpt=ckpt, step=step + 1, stage=synth_ds.stage,
                    curr_flags=list(curr_deque), data_dir=args.data_dir,
                    micro_batch=args.micro_batch, curr_window=args.curr_window,
                    consumed_index=synth_ds.current_index,
                    world_size=accelerator.num_processes, grad_accum=args.grad_accum,
                )
                print(f"Saved checkpoint: {ckpt}", flush=True)

        step += 1

        # Optional max_steps limit
        if args.max_steps is not None and step >= args.max_steps:
            if accelerator.is_main_process:
                print(f"Reached max_steps={args.max_steps}, stopping.", flush=True)
            break

    accelerator.wait_for_everyone()

    if filtered_iter.skipped > 0 and accelerator.is_main_process:
        print(f"WARNING: {filtered_iter.skipped} examples were filtered due to seq_len overflow. "
              f"Resume may not be perfectly reproducible.", flush=True)


if __name__ == "__main__":
    main()
