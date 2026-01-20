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

from nl.pre_pretrain.common import (
    load_model_and_tokenizer,
    tokenize_leading_space,
    load_jsonl_examples,
    full_word_correct,
    checkpoint_dir,
)
from nl.pre_pretrain.synthetic import SyntheticNL


def load_data_dir_prompts(data_dir: str) -> Set[str]:
    """Load all prompts from data_dir (eval, test, mixin) as reserved inputs."""
    data_path = Path(data_dir)
    prompts: Set[str] = set()

    for filename in ["eval.jsonl", "test.jsonl", "mixin.jsonl"]:
        filepath = data_path / filename
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

        print(f"  Loaded {count} prompts from {filename}", flush=True)

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
                yield {"prompt": ex["input_text"], "answer": outs[0]}

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
            a_ids = tokenize_leading_space(self.tokenizer, item["answer"])
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

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_batch, labels_pred_batch, attn_batch, pred_mask_batch = [], [], [], []

        for ex in batch:
            p_ids = tokenizer(ex["prompt"], add_special_tokens=True, truncation=False)["input_ids"]
            a_ids = tokenize_leading_space(tokenizer, ex["answer"])

            ids = p_ids + a_ids + [eos_id]
            labels_pred = ([-100] * len(p_ids)) + a_ids + [eos_id]

            if len(ids) > seq_len:
                ids = ids[:seq_len]
                labels_pred = labels_pred[:seq_len]

            L = len(ids)
            T = max(L - 1, 0)
            pred_mask = [0] * T
            p_len, a_len = len(p_ids), len(a_ids)

            if T > 0:
                for i in range(max(p_len - 1, 0), min(p_len + a_len, T)):
                    pred_mask[i] = 1

            input_ids_batch.append(torch.tensor(ids, dtype=torch.long))
            labels_pred_batch.append(torch.tensor(labels_pred, dtype=torch.long))
            attn_batch.append(torch.ones(L, dtype=torch.long))
            pred_mask_batch.append(torch.tensor(pred_mask, dtype=torch.long))

        max_len = max(x.shape[0] for x in input_ids_batch)
        max_t = max(max_len - 1, 0)

        def pad_to(x: torch.Tensor, val: int, target: int) -> torch.Tensor:
            if x.shape[0] == target:
                return x
            return torch.cat([x, torch.full((target - x.shape[0],), val, dtype=x.dtype)], dim=0)

        return {
            "input_ids": torch.stack([pad_to(x, pad_id, max_len) for x in input_ids_batch]),
            "labels_pred": torch.stack([pad_to(x, -100, max_len) for x in labels_pred_batch]),
            "attention_mask": torch.stack([pad_to(x, 0, max_len) for x in attn_batch]),
            "pred_mask": torch.stack([pad_to(x, 0, max_t) for x in pred_mask_batch]),
        }

    return collate


def weighted_fullseq_loss(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pred_mask: torch.Tensor,
) -> torch.Tensor:
    """Upweights prediction tokens: w_pred = 1 + (total_tokens / pred_tokens)"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    valid = attention_mask[:, 1:].bool()

    B, T, V = shift_logits.shape
    loss_tok = F.cross_entropy(
        shift_logits.view(-1, V), shift_labels.view(-1), reduction="none"
    ).view(B, T)

    pred_mask = pred_mask.bool() & valid
    weights = valid.float()

    P = pred_mask.sum(dim=1).clamp_min(1).float()
    Tn = valid.sum(dim=1).clamp_min(1).float()
    w_pred = 1.0 + (Tn / P)

    weights = weights + pred_mask.float() * (w_pred[:, None] - 1.0)
    return (loss_tok * weights).sum() / weights.sum().clamp_min(1.0)


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
        "curr_flags": [bool(x) for x in curr_flags],
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


def write_eval_log(log_path: Path, step: int, stage: int, alpha: float,
                   heldout_acc: float, heldout_loss: float, num_examples: int) -> None:
    entry = {
        "step": step, "stage": stage, "alpha": round(alpha, 4),
        "heldout_acc": round(heldout_acc, 4), "heldout_loss": round(heldout_loss, 6),
        "num_examples": num_examples, "timestamp": time.time(),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


@torch.no_grad()
def evaluate_heldout(model, tokenizer, examples: List[Dict[str, str]],
                     seq_len: int, accelerator: Accelerator) -> Tuple[float, float]:
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()
    collate_fn = build_collate_fn(tokenizer=tokenizer, seq_len=seq_len)

    total_correct, total_loss, total_examples = 0, 0.0, 0

    for i in range(0, len(examples), 16):
        batch_examples = examples[i:i + 16]
        batch = collate_fn(batch_examples)
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}

        outputs = unwrapped(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = weighted_fullseq_loss(
            logits=outputs.logits, input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"], pred_mask=batch["pred_mask"],
        )
        correct_list = full_word_correct(outputs.logits, batch["labels_pred"])

        total_correct += sum(correct_list)
        total_loss += loss.item() * len(batch_examples)
        total_examples += len(batch_examples)

    unwrapped.train()
    return total_correct / max(total_examples, 1), total_loss / max(total_examples, 1)


def build_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--scratch_dir", required=True)
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to data splits directory (from build_data_splits.py)")
    p.add_argument("--model_name", default="EleutherAI/pythia-160m")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--max_steps", type=int, required=True)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--micro_batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=2)

    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=1000)

    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--resume_from", type=str, default=None)

    p.add_argument("--max_input_size", type=int, default=384)
    p.add_argument("--max_lookahead", type=int, default=64)

    p.add_argument("--curr_window", type=int, default=200)
    p.add_argument("--curr_check_every", type=int, default=200)
    p.add_argument("--curr_threshold", type=float, default=0.98)
    p.add_argument("--eval_subset_size", type=int, default=500)

    return p


def main():
    args = build_argparser().parse_args()
    scratch_dir = Path(args.scratch_dir)
    run_dir = scratch_dir / "pptrain"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    train_log_path = run_dir / "logs" / "train.jsonl"
    eval_log_path = run_dir / "logs" / "eval.jsonl"

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

    start_step, start_stage = 0, 1
    start_index = 0
    curr_deque: deque = deque(maxlen=args.curr_window)
    resume_dir: Optional[Path] = None

    if args.resume_from is not None:
        resume_dir = Path(args.resume_from).expanduser().resolve()
        state = load_train_state(resume_dir)

        start_step = int(state.get("step", 0))
        start_stage = int(state.get("synthetic_stage", 1))
        start_index = int(state.get("consumed_index", start_step * args.micro_batch))

        saved_micro_batch = state.get("micro_batch")
        if saved_micro_batch is not None and saved_micro_batch != args.micro_batch:
            raise ValueError(f"micro_batch changed from {saved_micro_batch} to {args.micro_batch}")

        saved_grad_accum = state.get("grad_accum")
        if saved_grad_accum is not None and saved_grad_accum != args.grad_accum:
            raise ValueError(f"grad_accum changed from {saved_grad_accum} to {args.grad_accum}")

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

    data_dir = Path(args.data_dir).expanduser().resolve()
    if accelerator.is_main_process:
        print(f"Loading data splits from: {data_dir}", flush=True)
    reserved_prompts = load_data_dir_prompts(str(data_dir))
    eval_path = data_dir / "eval.jsonl"
    if accelerator.is_main_process:
        print(f"Total reserved prompts: {len(reserved_prompts)}", flush=True)

    eval_examples = load_jsonl_examples(eval_path, args.eval_subset_size, seed=args.seed)
    if accelerator.is_main_process:
        print(f"Loaded {len(eval_examples)} eval examples", flush=True)

    synth_ds, item_iter = synthetic_nl_stream(
        max_input_size=args.max_input_size, max_lookahead=args.max_lookahead,
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

    if start_step == 0 and accelerator.is_main_process:
        heldout_acc, heldout_loss = evaluate_heldout(model, tokenizer, eval_examples, args.seq_len, accelerator)
        print(f"Initial eval: acc={heldout_acc:.4f} loss={heldout_loss:.4f}", flush=True)
        write_eval_log(eval_log_path, step=0, stage=start_stage,
                       alpha=float(start_stage) / float(args.max_lookahead),
                       heldout_acc=heldout_acc, heldout_loss=heldout_loss, num_examples=len(eval_examples))

    loader_iter = iter(loader)
    curriculum_done = False

    for step in range(start_step, args.max_steps):
        step_loss = 0.0

        for accum_idx in range(args.grad_accum):
            batch = next(loader_iter)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = weighted_fullseq_loss(
                logits=outputs.logits, input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"], pred_mask=batch["pred_mask"],
            ) / args.grad_accum
            accelerator.backward(loss)
            step_loss += loss.item()

            correct_list = full_word_correct(outputs.logits.detach(), batch["labels_pred"].detach())
            all_correct = gather_bool_list(correct_list, accelerator)
            curr_deque.extend(all_correct)

        accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if accelerator.is_main_process and step % 10 == 0:
            curr_acc = sum(curr_deque) / len(curr_deque) if curr_deque else 0.0
            print(f"step={step} loss={step_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
            write_train_log(train_log_path, step=step, loss=step_loss, lr=scheduler.get_last_lr()[0],
                            curr_acc=curr_acc, stage=synth_ds.stage, alpha=synth_ds.current_alpha())

        if (step + 1) % args.curr_check_every == 0 and len(curr_deque) == args.curr_window:
            acc = sum(curr_deque) / len(curr_deque)

            if accelerator.is_main_process:
                print(f"Curriculum check step={step+1} stage={synth_ds.stage} "
                      f"alpha={synth_ds.current_alpha():.4f} acc={acc:.4f}", flush=True)

            if acc >= args.curr_threshold:
                synth_ds.increment_stage(1)
                curr_deque.clear()

                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    print(f"Advanced curriculum -> stage={synth_ds.stage} alpha={synth_ds.current_alpha():.4f}", flush=True)
                    heldout_acc, heldout_loss = evaluate_heldout(model, tokenizer, eval_examples, args.seq_len, accelerator)
                    print(f"Stage eval: acc={heldout_acc:.4f} loss={heldout_loss:.4f}", flush=True)
                    write_eval_log(eval_log_path, step=step + 1, stage=synth_ds.stage,
                                   alpha=synth_ds.current_alpha(), heldout_acc=heldout_acc,
                                   heldout_loss=heldout_loss, num_examples=len(eval_examples))
                accelerator.wait_for_everyone()

                if synth_ds.stage > synth_ds.max_lookahead:
                    if accelerator.is_main_process:
                        print("Finished curriculum (stage > max_lookahead).", flush=True)
                    curriculum_done = True
                    break

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

    accelerator.wait_for_everyone()

    if filtered_iter.skipped > 0 and accelerator.is_main_process:
        print(f"WARNING: {filtered_iter.skipped} examples were filtered due to seq_len overflow. "
              f"Resume may not be perfectly reproducible.", flush=True)


if __name__ == "__main__":
    main()
