#!/usr/bin/env python3
"""
Standard pretraining on C4 with optional synthetic data mixed in.

Supports three training modes:
  1. C4 only (use_synthetic_mix=False): Fresh model + 100% C4
  2. Fresh + mixed (use_synthetic_mix=True, pptrain_checkpoint="none"): Fresh model + x% synthetic
  3. Checkpoint + mixed (use_synthetic_mix=True, pptrain_checkpoint=path): Pre-pretrained + x% synthetic

Synthetic data is generated on-the-fly using SyntheticNL at alpha=1.0 (full difficulty)
with a different seed than pre-pretraining to ensure novel examples.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from accelerate import Accelerator

from common import (
    load_model_and_tokenizer,
    tokenize_leading_space,
    checkpoint_dir,
)
from synthetic import SyntheticNL


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

        self.num_shards = self.meta["num_shards"]

        # Load shards and calculate actual sizes from file sizes (more robust than metadata)
        self.shards: List[np.memmap] = []
        self.shard_blocks: List[int] = []  # Actual blocks per shard
        self.shard_offsets: List[int] = []  # Cumulative block offsets

        total_blocks = 0
        for i in range(self.num_shards):
            shard_path = c4_dir / f"shard_{i:05d}.bin"
            if not shard_path.exists():
                raise RuntimeError(f"Missing shard: {shard_path}")

            # Calculate blocks from actual file size
            file_size = shard_path.stat().st_size
            blocks_in_shard = file_size // (2 * self.block_len)  # uint16 = 2 bytes

            self.shard_offsets.append(total_blocks)
            self.shard_blocks.append(blocks_in_shard)
            total_blocks += blocks_in_shard

            shape = (blocks_in_shard, self.block_len)
            self.shards.append(np.memmap(shard_path, dtype=np.uint16, mode="r", shape=shape))

        self.total_blocks = total_blocks

    def __len__(self) -> int:
        return self.total_blocks

    def get_block(self, idx: int) -> torch.Tensor:
        idx = idx % self.total_blocks

        # Binary search to find the right shard
        shard_idx = 0
        for i, offset in enumerate(self.shard_offsets):
            if offset <= idx:
                shard_idx = i
            else:
                break

        block_in_shard = idx - self.shard_offsets[shard_idx]
        return torch.from_numpy(self.shards[shard_idx][block_in_shard].astype(np.int64))


class SyntheticGenerator:
    """On-the-fly synthetic data generator for pretraining mix."""

    def __init__(
        self,
        *,
        seed: int,
        max_input_size: int,
        max_lookahead: int,
        reserved_inputs: Set[str],
        tokenizer,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.seed = seed
        self.max_input_size = max_input_size
        self.max_lookahead = max_lookahead
        self.reserved_inputs = reserved_inputs
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.block_len = seq_len + 1
        self.rank = rank
        self.world_size = world_size

        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id

        # Create SyntheticNL generator at alpha=1.0 (stage=max_lookahead)
        self.generator = SyntheticNL(
            max_input_size=max_input_size,
            max_lookahead=max_lookahead,
            seed=seed,
            task="search",
            stage=max_lookahead,  # alpha=1.0 for full difficulty
            reserved_inputs=reserved_inputs,
            world_size=world_size,
            rank=rank,
            start_index=0,
        )
        self._iter = iter(self.generator)

    def get_example(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a single tokenized synthetic example as a C4-like block.

        Returns:
            block: (block_len,) token IDs (prompt + answer + EOS + padding)
            attn_mask: (block_len,) attention mask (1 for real tokens, 0 for padding)
            loss_mask: (block_len,) loss mask (1 for prediction tokens only: answer + EOS)
        """
        example = next(self._iter)
        prompt = example["input_text"]
        answer = example["output_texts"][0]  # Take first valid answer

        # Tokenize
        p_ids = self.tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        a_ids = tokenize_leading_space(self.tokenizer, answer)
        ids = p_ids + a_ids + [self.eos_id]

        prompt_len = len(p_ids)
        real_len = min(len(ids), self.block_len)

        # Pad/truncate
        if len(ids) > self.block_len:
            ids = ids[:self.block_len]
        else:
            ids = ids + [self.pad_id] * (self.block_len - len(ids))

        block = torch.tensor(ids, dtype=torch.long)

        # Attention mask: all real tokens (model needs to see prompt)
        attn_mask = torch.zeros(self.block_len, dtype=torch.long)
        attn_mask[:real_len] = 1

        # Loss mask: prediction tokens only (answer + EOS)
        loss_mask = torch.zeros(self.block_len, dtype=torch.long)
        loss_mask[prompt_len:real_len] = 1

        return block, attn_mask, loss_mask


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int,
                                     min_lr_ratio: float = 0.1) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)


def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor,
                    loss_mask: torch.Tensor) -> torch.Tensor:
    """Per-sequence averaged loss. Equivalent to global avg when all seqs equal length.

    Args:
        logits: (B, T, V) model output logits
        labels: (B, T) target token IDs (already shifted relative to input)
        loss_mask: (B, T) mask (1 for tokens to include in loss, 0 otherwise)
    """
    B, T, V = logits.shape
    loss_tok = F.cross_entropy(
        logits.view(-1, V), labels.view(-1), reduction="none"
    ).view(B, T)

    loss_tok = loss_tok * loss_mask.float()

    # Average within each sequence, then across sequences
    seq_losses = loss_tok.sum(dim=1)  # (B,)
    seq_counts = loss_mask.sum(dim=1).clamp_min(1.0)  # (B,)
    per_seq_avg = seq_losses / seq_counts  # (B,)
    return per_seq_avg.mean()


def save_train_state(
    *, ckpt: Path, step: int, global_c4_idx: int, global_synthetic_idx: int,
    synthetic_count: int, c4_count: int, synthetic_percent: float, total_steps: int,
    micro_batch: int, grad_accum: int, world_size: int, use_synthetic_mix: bool,
) -> None:
    ckpt.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step, "global_c4_idx": global_c4_idx, "global_synthetic_idx": global_synthetic_idx,
        "synthetic_count": synthetic_count, "c4_count": c4_count, "synthetic_percent": synthetic_percent,
        "total_steps": total_steps, "micro_batch": micro_batch,
        "grad_accum": grad_accum, "world_size": world_size, "use_synthetic_mix": use_synthetic_mix,
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
                    synthetic_count: int, c4_count: int) -> None:
    entry = {
        "step": step, "loss": round(loss, 6), "lr": lr,
        "synthetic_count": synthetic_count, "c4_count": c4_count, "timestamp": time.time(),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def find_latest_pptrain_checkpoint(scratch_dir: Path) -> Optional[Path]:
    ckpt_root = scratch_dir / "pptrain" / "checkpoints"
    if not ckpt_root.exists():
        return None
    checkpoints = sorted(ckpt_root.glob("step_*"))
    return checkpoints[-1] if checkpoints else None


def build_argparser():
    p = argparse.ArgumentParser()

    # Run configuration
    p.add_argument("--scratch_dir", required=True)
    p.add_argument("--run_name", type=str, default="pretrain",
                   help="Run directory name (e.g., 'pretrain_mix' or 'pretrain_baseline')")
    p.add_argument("--model_name", default="EleutherAI/pythia-160m")
    p.add_argument("--seed", type=int, default=1337)

    # Pre-pretrained checkpoint (only used if use_synthetic_mix=True)
    p.add_argument("--pptrain_checkpoint", type=str, default="latest",
                   help="Path to pre-pretrained checkpoint or 'latest' (only for Condition A)")

    # Experimental condition
    p.add_argument("--use_synthetic_mix", type=lambda x: x.lower() == "true", default=True,
                   help="True=Condition A (pre-pretrained + 5%% synthetic), False=Condition B (baseline)")
    p.add_argument("--synthetic_seed", type=int, default=99999,
                   help="Seed for synthetic generation (must differ from pre-pretraining seed)")
    p.add_argument("--synthetic_percent", type=float, default=0.05,
                   help="Probability each example is synthetic (default: 5%%)")

    # Training hyperparameters
    p.add_argument("--total_steps", type=int, default=3100,
                   help="Total training steps (~3.2B tokens with default batch)")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--micro_batch", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)

    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=100)

    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")

    # Data paths
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory with eval.jsonl (for data leakage prevention)")
    p.add_argument("--c4_dir", type=str, default=None,
                   help="Path to tokenized C4 (default: $scratch_dir/data/c4_tokenized)")

    # Synthetic data config (for on-the-fly generation)
    p.add_argument("--max_lookahead", type=int, default=32,
                   help="Max lookahead (max_input_size = 6 * max_lookahead)")

    # Logging and checkpoints
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--resume_from", type=str, default=None)

    return p


def load_reserved_inputs(data_dir: Path) -> Set[str]:
    """Load prompts from eval.jsonl as reserved inputs for data leakage prevention."""
    reserved: Set[str] = set()
    filepath = data_dir / "eval.jsonl"
    if not filepath.exists():
        raise RuntimeError(f"Missing required file: {filepath}")

    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("input_text")
            if prompt:
                reserved.add(prompt)
    return reserved


def main():
    args = build_argparser().parse_args()

    scratch_dir = Path(args.scratch_dir)
    run_dir = scratch_dir / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    train_log_path = run_dir / "logs" / "train.jsonl"
    data_dir = Path(args.data_dir).expanduser().resolve()

    # Determine C4 directory
    c4_dir = Path(args.c4_dir) if args.c4_dir else scratch_dir / "data" / "c4_tokenized"

    accelerator = Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )

    rank = accelerator.process_index
    world_size = accelerator.num_processes

    if accelerator.is_main_process:
        condition = "A (pre-pretrained + 5% synthetic)" if args.use_synthetic_mix else "B (baseline)"
        print(f"=== Pretraining Condition {condition} ===", flush=True)
        print(f"num_processes={world_size} process_index={rank} "
              f"mixed_precision={accelerator.mixed_precision}", flush=True)
        print(f"run_dir={run_dir}", flush=True)

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)

    start_step, global_c4_idx, global_synthetic_idx = 0, 0, 0
    synthetic_count, c4_count = 0, 0
    resume_dir: Optional[Path] = None

    if args.resume_from is not None:
        resume_dir = Path(args.resume_from).expanduser().resolve()
        state = load_train_state(resume_dir)

        start_step = int(state.get("step", 0))
        global_c4_idx = int(state.get("global_c4_idx", 0))
        global_synthetic_idx = int(state.get("global_synthetic_idx", state.get("global_mixin_idx", 0)))
        synthetic_count = int(state.get("synthetic_count", state.get("mixin_count", 0)))
        c4_count = int(state.get("c4_count", 0))

        if "rng_state" in state:
            restore_rng_state(state["rng_state"])

        saved_micro_batch = state.get("micro_batch")
        if saved_micro_batch is not None and saved_micro_batch != args.micro_batch:
            if accelerator.is_main_process:
                print(f"WARNING: micro_batch changed from {saved_micro_batch} to {args.micro_batch}", flush=True)
        saved_grad_accum = state.get("grad_accum")
        if saved_grad_accum is not None and saved_grad_accum != args.grad_accum:
            if accelerator.is_main_process:
                print(f"WARNING: grad_accum changed from {saved_grad_accum} to {args.grad_accum}", flush=True)
        saved_world_size = state.get("world_size")
        if saved_world_size is not None and saved_world_size != world_size:
            raise ValueError(f"world_size changed from {saved_world_size} to {world_size}")

        if accelerator.is_main_process:
            print(f"Resuming from {resume_dir}: step={start_step} "
                  f"c4_idx={global_c4_idx} synthetic_idx={global_synthetic_idx}", flush=True)

    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Load model weights based on condition
    if resume_dir is None:
        if args.use_synthetic_mix:
            if args.pptrain_checkpoint == "none":
                # Condition C: Fresh model + synthetic mix (no pre-pretraining)
                if accelerator.is_main_process:
                    print(f"Using fresh {args.model_name} model with synthetic mix", flush=True)
            elif args.pptrain_checkpoint == "latest":
                # Condition A: Load latest pre-pretrained checkpoint
                pptrain_ckpt = find_latest_pptrain_checkpoint(scratch_dir)
                if pptrain_ckpt is None:
                    raise RuntimeError(f"No pptrain checkpoint in {scratch_dir}/pptrain/checkpoints/")

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
            else:
                # Condition A: Load specific pre-pretrained checkpoint
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
        else:
            # Condition B: Fresh model (already loaded)
            if accelerator.is_main_process:
                print(f"Using fresh {args.model_name} model (baseline condition)", flush=True)

    # Load C4 dataset
    if not c4_dir.exists():
        raise RuntimeError(f"C4 data not found at {c4_dir}")

    c4_dataset = PackedC4Dataset(c4_dir, args.seq_len)
    if accelerator.is_main_process:
        print(f"Loaded C4 dataset: {len(c4_dataset)} blocks", flush=True)

    # Set up synthetic generator (only for Condition A)
    synthetic_gen: Optional[SyntheticGenerator] = None
    if args.use_synthetic_mix:
        reserved_inputs = load_reserved_inputs(data_dir)
        if accelerator.is_main_process:
            print(f"Loaded {len(reserved_inputs)} reserved inputs for synthetic generation", flush=True)

        # Calculate max_input_size from max_lookahead
        max_input_size = 6 * args.max_lookahead
        if accelerator.is_main_process:
            print(f"max_lookahead={args.max_lookahead} max_input_size={max_input_size}", flush=True)

        synthetic_gen = SyntheticGenerator(
            seed=args.synthetic_seed,
            max_input_size=max_input_size,
            max_lookahead=args.max_lookahead,
            reserved_inputs=reserved_inputs,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            rank=rank,
            world_size=world_size,
        )
        if accelerator.is_main_process:
            print(f"Initialized synthetic generator (seed={args.synthetic_seed}, alpha=1.0)", flush=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
                      eps=args.adam_eps, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.total_steps, args.min_lr_ratio)

    # Don't prepare scheduler — Accelerate's AcceleratedScheduler multiplies
    # step count by grad_accum, causing multiple cosine cycles instead of one.
    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()

    if resume_dir is not None:
        accelerator.load_state(str(resume_dir))
        # Restore scheduler state manually (not managed by Accelerate)
        sched_path = resume_dir / "scheduler.pt"
        if sched_path.exists():
            scheduler.load_state_dict(torch.load(sched_path, weights_only=True))
        else:
            # Old checkpoint without scheduler.pt — fast-forward to correct step
            for _ in range(start_step):
                scheduler.step()

    effective_batch = args.micro_batch * args.grad_accum * world_size
    tokens_per_step = effective_batch * args.seq_len
    if accelerator.is_main_process:
        print(f"Training from step {start_step} to {args.total_steps}", flush=True)
        print(f"  effective_batch={effective_batch} tokens_per_step={tokens_per_step:,}", flush=True)
        if args.use_synthetic_mix:
            print(f"  synthetic_percent={args.synthetic_percent*100:.1f}%", flush=True)

    block_len = args.seq_len + 1  # 2049

    for step in range(start_step, args.total_steps):
        step_loss = 0.0

        for accum_idx in range(args.grad_accum):
            batch_blocks = []
            batch_attn_masks = []
            batch_loss_masks = []
            c4_offset = 0

            for b in range(args.micro_batch):
                # Per-example decision: synthetic or C4
                use_synthetic = False
                if args.use_synthetic_mix and synthetic_gen is not None:
                    decision_seed = (args.seed
                                     + step * args.grad_accum * args.micro_batch
                                     + accum_idx * args.micro_batch + b)
                    use_synthetic = random.Random(decision_seed).random() < args.synthetic_percent

                if use_synthetic:
                    block, attn_mask, loss_mask = synthetic_gen.get_example()
                    global_synthetic_idx += world_size
                    if rank == 0:
                        synthetic_count += 1
                else:
                    global_idx = global_c4_idx + c4_offset * world_size + rank
                    block = c4_dataset.get_block(global_idx)
                    attn_mask = torch.ones(block_len, dtype=torch.long)
                    loss_mask = torch.ones(block_len, dtype=torch.long)
                    c4_offset += 1
                    if rank == 0:
                        c4_count += 1

                batch_blocks.append(block)
                batch_attn_masks.append(attn_mask)
                batch_loss_masks.append(loss_mask)

            global_c4_idx += c4_offset * world_size

            # Stack and split into input/label pairs
            blocks = torch.stack(batch_blocks, dim=0)         # (B, 2049)
            attn_masks = torch.stack(batch_attn_masks, dim=0) # (B, 2049)
            loss_masks = torch.stack(batch_loss_masks, dim=0) # (B, 2049)

            input_ids = blocks[:, :-1].to(accelerator.device)       # (B, 2048)
            labels = blocks[:, 1:].to(accelerator.device)           # (B, 2048)
            input_mask = attn_masks[:, :-1].to(accelerator.device)  # (B, 2048) — for model forward
            label_mask = loss_masks[:, 1:].to(accelerator.device)   # (B, 2048) — for loss

            outputs = model(input_ids=input_ids, attention_mask=input_mask)
            loss = compute_lm_loss(outputs.logits, labels, label_mask) / args.grad_accum

            accelerator.backward(loss)
            step_loss += loss.item()

        accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if accelerator.is_main_process and step % args.log_every == 0:
            print(f"step={step} loss={step_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
            write_train_log(train_log_path, step=step, loss=step_loss, lr=scheduler.get_last_lr()[0],
                            synthetic_count=synthetic_count, c4_count=c4_count)

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            ckpt = checkpoint_dir(run_dir, step + 1)
            accelerator.wait_for_everyone()
            accelerator.save_state(str(ckpt))

            if accelerator.is_main_process:
                save_train_state(
                    ckpt=ckpt, step=step + 1, global_c4_idx=global_c4_idx,
                    global_synthetic_idx=global_synthetic_idx, synthetic_count=synthetic_count, c4_count=c4_count,
                    synthetic_percent=args.synthetic_percent, total_steps=args.total_steps,
                    micro_batch=args.micro_batch, grad_accum=args.grad_accum, world_size=world_size,
                    use_synthetic_mix=args.use_synthetic_mix,
                )
                torch.save(scheduler.state_dict(), ckpt / "scheduler.pt")
                print(f"Saved checkpoint: {ckpt}", flush=True)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Training complete!", flush=True)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
