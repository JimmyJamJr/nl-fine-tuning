#!/usr/bin/env python3
"""
Standard pretraining on C4 with synthetic data mixed in.

Loads a pre-pretrained checkpoint and continues training on packed C4 data,
with a configurable percentage of synthetic NL examples mixed in at runtime.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from accelerate import Accelerator

from common import (
    load_model_and_tokenizer,
    tokenize_leading_space,
    load_jsonl_examples,
    full_word_correct,
    checkpoint_dir,
)


class PackedC4Dataset:
    def __init__(self, c4_dir: Path, seq_len: int):
        self.c4_dir = c4_dir
        self.seq_len = seq_len
        self.block_len = seq_len + 1

        meta_path = c4_dir / "meta.json"
        if not meta_path.exists():
            raise RuntimeError(f"Missing meta.json in {c4_dir}")
        with meta_path.open("r") as f:
            self.meta = json.load(f)

        self.total_blocks = self.meta["total_blocks"]
        self.blocks_per_shard = self.meta["blocks_per_shard"]
        self.num_shards = self.meta["num_shards"]

        self.shards: List[np.memmap] = []
        for i in range(self.num_shards):
            shard_path = c4_dir / f"shard_{i:05d}.bin"
            if not shard_path.exists():
                raise RuntimeError(f"Missing shard: {shard_path}")
            if i < self.num_shards - 1:
                blocks_in_shard = self.blocks_per_shard
            else:
                blocks_in_shard = self.total_blocks - (self.num_shards - 1) * self.blocks_per_shard
            shape = (blocks_in_shard, self.block_len)
            self.shards.append(np.memmap(shard_path, dtype=np.uint16, mode="r", shape=shape))

    def __len__(self) -> int:
        return self.total_blocks

    def get_block(self, idx: int) -> torch.Tensor:
        idx = idx % self.total_blocks
        shard_idx = idx // self.blocks_per_shard
        block_in_shard = idx % self.blocks_per_shard

        if shard_idx >= len(self.shards):
            shard_idx = len(self.shards) - 1
            block_in_shard = idx - shard_idx * self.blocks_per_shard

        return torch.from_numpy(self.shards[shard_idx][block_in_shard].astype(np.int64))


def tokenize_mixin_example(example: Dict[str, str], tokenizer, seq_len: int) -> torch.Tensor:
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    p_ids = tokenizer(example["prompt"], add_special_tokens=True, truncation=False)["input_ids"]
    a_ids = tokenize_leading_space(tokenizer, example["answer"])
    ids = p_ids + a_ids + [eos_id]

    block_len = seq_len + 1
    if len(ids) > block_len:
        ids = ids[:block_len]
    elif len(ids) < block_len:
        ids = ids + [pad_id] * (block_len - len(ids))

    return torch.tensor(ids, dtype=torch.long)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int,
                                     min_lr_ratio: float = 0.1) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)


def compute_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    B, T, V = shift_logits.shape
    loss_tok = F.cross_entropy(
        shift_logits.view(-1, V), shift_labels.view(-1), reduction="none"
    ).view(B, T)

    loss_tok = loss_tok * shift_mask.float()
    return loss_tok.sum() / shift_mask.sum().clamp_min(1.0)


def save_train_state(
    *, ckpt: Path, step: int, global_c4_idx: int, global_mixin_idx: int,
    mixin_count: int, c4_count: int, mixin_percent: float, total_steps: int,
    micro_batch: int, grad_accum: int, world_size: int,
) -> None:
    ckpt.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step, "global_c4_idx": global_c4_idx, "global_mixin_idx": global_mixin_idx,
        "mixin_count": mixin_count, "c4_count": c4_count, "mixin_percent": mixin_percent,
        "total_steps": total_steps, "micro_batch": micro_batch,
        "grad_accum": grad_accum, "world_size": world_size,
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
                    mixin_count: int, c4_count: int) -> None:
    entry = {
        "step": step, "loss": round(loss, 6), "lr": lr,
        "mixin_count": mixin_count, "c4_count": c4_count, "timestamp": time.time(),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def write_eval_log(log_path: Path, step: int, eval_acc: float, eval_loss: float,
                   num_examples: int) -> None:
    entry = {
        "step": step, "eval_acc": round(eval_acc, 4), "eval_loss": round(eval_loss, 6),
        "num_examples": num_examples, "timestamp": time.time(),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def build_eval_collate_fn(tokenizer, seq_len: int):
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_batch, labels_pred_batch, attn_batch = [], [], []

        for ex in batch:
            p_ids = tokenizer(ex["prompt"], add_special_tokens=True, truncation=False)["input_ids"]
            a_ids = tokenize_leading_space(tokenizer, ex["answer"])

            ids = p_ids + a_ids + [eos_id]
            labels_pred = ([-100] * len(p_ids)) + a_ids + [eos_id]

            if len(ids) > seq_len:
                ids = ids[:seq_len]
                labels_pred = labels_pred[:seq_len]

            input_ids_batch.append(torch.tensor(ids, dtype=torch.long))
            labels_pred_batch.append(torch.tensor(labels_pred, dtype=torch.long))
            attn_batch.append(torch.ones(len(ids), dtype=torch.long))

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


@torch.no_grad()
def evaluate(model, tokenizer, examples: List[Dict[str, str]], seq_len: int,
             accelerator: Accelerator) -> Tuple[float, float]:
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()
    collate_fn = build_eval_collate_fn(tokenizer=tokenizer, seq_len=seq_len)

    total_correct, total_loss, total_examples = 0, 0.0, 0

    for i in range(0, len(examples), 16):
        batch_examples = examples[i:i + 16]
        batch = collate_fn(batch_examples)
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}

        outputs = unwrapped(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = compute_lm_loss(outputs.logits, batch["input_ids"], batch["attention_mask"])
        correct_list = full_word_correct(outputs.logits, batch["labels_pred"])

        total_correct += sum(correct_list)
        total_loss += loss.item() * len(batch_examples)
        total_examples += len(batch_examples)

    unwrapped.train()
    return total_correct / max(total_examples, 1), total_loss / max(total_examples, 1)


def find_latest_pptrain_checkpoint(scratch_dir: Path) -> Optional[Path]:
    ckpt_root = scratch_dir / "pptrain" / "checkpoints"
    if not ckpt_root.exists():
        return None
    checkpoints = sorted(ckpt_root.glob("step_*"))
    return checkpoints[-1] if checkpoints else None


def build_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--scratch_dir", required=True)
    p.add_argument("--pptrain_checkpoint", type=str, default="latest")
    p.add_argument("--model_name", default="EleutherAI/pythia-160m")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--total_steps", type=int, default=10000)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--micro_batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=2)

    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=1000)

    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--mixin_percent", type=float, default=0.02)
    p.add_argument("--data_dir", type=str, required=True)

    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_subset_size", type=int, default=500)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--resume_from", type=str, default=None)

    return p


def main():
    args = build_argparser().parse_args()

    scratch_dir = Path(args.scratch_dir)
    run_dir = scratch_dir / "pretrain"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    train_log_path = run_dir / "logs" / "train.jsonl"
    eval_log_path = run_dir / "logs" / "eval.jsonl"
    data_dir = Path(args.data_dir).expanduser().resolve()

    accelerator = Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )

    rank = accelerator.process_index
    world_size = accelerator.num_processes

    if accelerator.is_main_process:
        print(f"num_processes={world_size} process_index={rank} "
              f"mixed_precision={accelerator.mixed_precision}", flush=True)

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)

    start_step, global_c4_idx, global_mixin_idx = 0, 0, 0
    mixin_count, c4_count = 0, 0
    resume_dir: Optional[Path] = None

    if args.resume_from is not None:
        resume_dir = Path(args.resume_from).expanduser().resolve()
        state = load_train_state(resume_dir)

        start_step = int(state.get("step", 0))
        global_c4_idx = int(state.get("global_c4_idx", 0))
        global_mixin_idx = int(state.get("global_mixin_idx", 0))
        mixin_count = int(state.get("mixin_count", 0))
        c4_count = int(state.get("c4_count", 0))

        if "rng_state" in state:
            restore_rng_state(state["rng_state"])

        saved_micro_batch = state.get("micro_batch")
        if saved_micro_batch is not None and saved_micro_batch != args.micro_batch:
            raise ValueError(f"micro_batch changed from {saved_micro_batch} to {args.micro_batch}")
        saved_grad_accum = state.get("grad_accum")
        if saved_grad_accum is not None and saved_grad_accum != args.grad_accum:
            raise ValueError(f"grad_accum changed from {saved_grad_accum} to {args.grad_accum}")
        saved_world_size = state.get("world_size")
        if saved_world_size is not None and saved_world_size != world_size:
            raise ValueError(f"world_size changed from {saved_world_size} to {world_size}")

        if accelerator.is_main_process:
            print(f"Resuming from {resume_dir}: step={start_step} "
                  f"c4_idx={global_c4_idx} mixin_idx={global_mixin_idx}", flush=True)

    model, tokenizer = load_model_and_tokenizer(args.model_name)

    if resume_dir is None:
        if args.pptrain_checkpoint == "latest":
            pptrain_ckpt = find_latest_pptrain_checkpoint(scratch_dir)
            if pptrain_ckpt is None:
                raise RuntimeError(f"No pptrain checkpoint in {scratch_dir}/pptrain/checkpoints/")
        else:
            pptrain_ckpt = Path(args.pptrain_checkpoint).expanduser().resolve()

        if not pptrain_ckpt.exists():
            raise RuntimeError(f"Checkpoint not found: {pptrain_ckpt}")

        if accelerator.is_main_process:
            print(f"Loading pre-pretrained model from: {pptrain_ckpt}", flush=True)

        model_loaded = False
        safetensors_path = pptrain_ckpt / "model.safetensors"
        if safetensors_path.exists():
            from safetensors.torch import load_file
            model.load_state_dict(load_file(safetensors_path))
            model_loaded = True
            if accelerator.is_main_process:
                print("  Loaded from model.safetensors", flush=True)

        if not model_loaded:
            pytorch_path = pptrain_ckpt / "pytorch_model.bin"
            if pytorch_path.exists():
                model.load_state_dict(torch.load(pytorch_path, map_location="cpu", weights_only=True))
                model_loaded = True
                if accelerator.is_main_process:
                    print("  Loaded from pytorch_model.bin", flush=True)

        if not model_loaded:
            raise RuntimeError(f"No model weights in {pptrain_ckpt}")

    c4_dir = scratch_dir / "data" / "c4_packed"
    if not c4_dir.exists():
        raise RuntimeError(f"C4 data not found at {c4_dir}")

    c4_dataset = PackedC4Dataset(c4_dir, args.seq_len)
    if accelerator.is_main_process:
        print(f"Loaded C4 dataset: {len(c4_dataset)} blocks", flush=True)

    mixin_path = data_dir / "mixin.jsonl"
    if not mixin_path.exists():
        raise RuntimeError(f"Mixin data not found at {mixin_path}")

    mixin_examples = load_jsonl_examples(mixin_path)
    if accelerator.is_main_process:
        print(f"Loaded mixin data: {len(mixin_examples)} examples", flush=True)

    eval_path = data_dir / "eval.jsonl"
    eval_examples = load_jsonl_examples(eval_path, args.eval_subset_size, seed=args.seed)
    if accelerator.is_main_process:
        print(f"Loaded eval data: {len(eval_examples)} examples", flush=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
                      eps=args.adam_eps, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.total_steps, args.min_lr_ratio)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    model.train()

    if resume_dir is not None:
        accelerator.load_state(str(resume_dir))

    if start_step == 0:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            eval_acc, eval_loss = evaluate(model, tokenizer, eval_examples, args.seq_len, accelerator)
            print(f"Initial eval: acc={eval_acc:.4f} loss={eval_loss:.4f}", flush=True)
            write_eval_log(eval_log_path, step=0, eval_acc=eval_acc, eval_loss=eval_loss,
                           num_examples=len(eval_examples))
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"Training from step {start_step} to {args.total_steps}", flush=True)
        print(f"  mixin_percent={args.mixin_percent} "
              f"effective_batch={args.micro_batch * args.grad_accum * world_size}", flush=True)

    for step in range(start_step, args.total_steps):
        step_loss = 0.0

        for accum_idx in range(args.grad_accum):
            decision_seed = args.seed + step * args.grad_accum + accum_idx
            use_mixin = random.Random(decision_seed).random() < args.mixin_percent

            if use_mixin and len(mixin_examples) > 0:
                batch_tensors = []
                for b in range(args.micro_batch):
                    global_idx = global_mixin_idx + b * world_size + rank
                    ex = mixin_examples[global_idx % len(mixin_examples)]
                    batch_tensors.append(tokenize_mixin_example(ex, tokenizer, args.seq_len))

                input_ids = torch.stack(batch_tensors, dim=0).to(accelerator.device)
                attention_mask = (input_ids != tokenizer.pad_token_id).long()

                if rank == 0:
                    mixin_count += 1
                global_mixin_idx += args.micro_batch * world_size
            else:
                batch_tensors = []
                for b in range(args.micro_batch):
                    global_idx = global_c4_idx + b * world_size + rank
                    batch_tensors.append(c4_dataset.get_block(global_idx))

                input_ids = torch.stack(batch_tensors, dim=0).to(accelerator.device)
                attention_mask = torch.ones_like(input_ids)

                if rank == 0:
                    c4_count += 1
                global_c4_idx += args.micro_batch * world_size

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = compute_lm_loss(outputs.logits, input_ids, attention_mask) / args.grad_accum

            accelerator.backward(loss)
            step_loss += loss.item()

        accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if accelerator.is_main_process and step % args.log_every == 0:
            print(f"step={step} loss={step_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
            write_train_log(train_log_path, step=step, loss=step_loss, lr=scheduler.get_last_lr()[0],
                            mixin_count=mixin_count, c4_count=c4_count)

        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                eval_acc, eval_loss = evaluate(model, tokenizer, eval_examples, args.seq_len, accelerator)
                print(f"step={step+1} eval: acc={eval_acc:.4f} loss={eval_loss:.4f}", flush=True)
                write_eval_log(eval_log_path, step=step + 1, eval_acc=eval_acc, eval_loss=eval_loss,
                               num_examples=len(eval_examples))
            accelerator.wait_for_everyone()

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            ckpt = checkpoint_dir(run_dir, step + 1)
            accelerator.wait_for_everyone()
            accelerator.save_state(str(ckpt))

            if accelerator.is_main_process:
                save_train_state(
                    ckpt=ckpt, step=step + 1, global_c4_idx=global_c4_idx,
                    global_mixin_idx=global_mixin_idx, mixin_count=mixin_count, c4_count=c4_count,
                    mixin_percent=args.mixin_percent, total_steps=args.total_steps,
                    micro_batch=args.micro_batch, grad_accum=args.grad_accum, world_size=world_size,
                )
                print(f"Saved checkpoint: {ckpt}", flush=True)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        eval_acc, eval_loss = evaluate(model, tokenizer, eval_examples, args.seq_len, accelerator)
        print(f"Final eval: acc={eval_acc:.4f} loss={eval_loss:.4f}", flush=True)
        write_eval_log(eval_log_path, step=args.total_steps, eval_acc=eval_acc, eval_loss=eval_loss,
                       num_examples=len(eval_examples))
        print("Training complete!", flush=True)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
