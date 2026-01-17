import argparse
import os
from collections import deque
from typing import Iterator, Dict, Any, Optional, List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from synthetic import SyntheticNL


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def synthetic_nl_stream(
    *,
    max_input_size: int,
    max_lookahead: int,
    seed: int,
    rank: int,
    world_size: int,
) -> Tuple[SyntheticNL, Iterator[Dict[str, Any]]]:
    ds = SyntheticNL(
        max_input_size=max_input_size,
        max_lookahead=max_lookahead,
        seed=seed,
        rank=rank,
        world_size=world_size,
        stage=1,
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


def _tokenize_leading_space(tokenizer, s: str) -> List[int]:
    s = s if s.startswith(" ") else (" " + s)
    return tokenizer(s, add_special_tokens=False)["input_ids"]


def build_synth_collate_fn(*, tokenizer, seq_len: int):
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    seq_len = int(seq_len)

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_batch = []
        labels_pred_batch = []
        attn_batch = []
        pred_mask_batch = []

        for ex in batch:
            p_ids = tokenizer(ex["prompt"], add_special_tokens=True, truncation=False)["input_ids"]
            a_ids = _tokenize_leading_space(tokenizer, ex["answer"])
            ids = p_ids + a_ids + [eos_id]
            labels_pred = ([-100] * len(p_ids)) + a_ids + [eos_id]

            if len(ids) > seq_len:
                ids = ids[:seq_len]
                labels_pred = labels_pred[:seq_len]

            L = len(ids)
            T = max(L - 1, 0)

            pred_mask = [0] * T
            p_len = len(p_ids)
            a_len = len(a_ids)

            if T > 0:
                start = max(p_len - 1, 0)
                end = min(p_len + a_len, T)
                for i in range(start, end):
                    pred_mask[i] = 1

            input_ids_batch.append(torch.tensor(ids, dtype=torch.long))
            labels_pred_batch.append(torch.tensor(labels_pred, dtype=torch.long))
            attn_batch.append(torch.ones(L, dtype=torch.long))
            pred_mask_batch.append(torch.tensor(pred_mask, dtype=torch.long))

        max_len = max(x.shape[0] for x in input_ids_batch)
        max_t = max(max_len - 1, 0)

        def pad_to_1d(x: torch.Tensor, val: int, target: int) -> torch.Tensor:
            if x.shape[0] == target:
                return x
            pad = torch.full((target - x.shape[0],), val, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        input_ids = torch.stack([pad_to_1d(x, pad_id, max_len) for x in input_ids_batch], dim=0)
        labels_pred = torch.stack([pad_to_1d(x, -100, max_len) for x in labels_pred_batch], dim=0)
        attention_mask = torch.stack([pad_to_1d(x, 0, max_len) for x in attn_batch], dim=0)
        pred_mask = torch.stack([pad_to_1d(x, 0, max_t) for x in pred_mask_batch], dim=0)

        return {
            "input_ids": input_ids,
            "labels_pred": labels_pred,
            "attention_mask": attention_mask,
            "pred_mask": pred_mask,
        }

    return collate


@torch.no_grad()
def full_word_correct_from_logits(logits: torch.Tensor, labels_pred: torch.Tensor) -> List[bool]:
    pred_ids = logits[:, :-1, :].argmax(dim=-1)
    gold_ids = labels_pred[:, 1:]
    mask = gold_ids.ne(-100)

    out: List[bool] = []
    for b in range(gold_ids.shape[0]):
        mb = mask[b]
        if not torch.any(mb):
            out.append(False)
        else:
            out.append(bool(torch.all(pred_ids[b, mb].eq(gold_ids[b, mb])).item()))
    return out


def weighted_fullseq_loss(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pred_mask: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    valid = attention_mask[:, 1:].to(dtype=torch.bool)

    B, T, V = shift_logits.shape
    loss_tok = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
    ).view(B, T)

    pred_mask = pred_mask.to(dtype=torch.bool) & valid

    weights = valid.to(dtype=torch.float32)

    P = pred_mask.sum(dim=1).clamp_min(1).to(dtype=torch.float32)
    Tn = valid.sum(dim=1).clamp_min(1).to(dtype=torch.float32)
    w_pred = 1.0 + (Tn / P)

    weights = weights + pred_mask.to(dtype=torch.float32) * (w_pred[:, None] - 1.0)

    return (loss_tok * weights).sum() / weights.sum().clamp_min(1.0)


def reduce_mean(value: float, accelerator: Accelerator) -> float:
    t = torch.tensor([value], device=accelerator.device, dtype=torch.float32)
    t = accelerator.reduce(t, reduction="mean")
    return float(t.item())


def reduce_max_bool(flag: bool, accelerator: Accelerator) -> bool:
    t = torch.tensor([1 if flag else 0], device=accelerator.device, dtype=torch.long)
    t = accelerator.reduce(t, reduction="max")
    return bool(t.item())


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--model_name", default="EleutherAI/pythia-160m")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--max_steps", type=int, required=True)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--micro_batch", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)

    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")

    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--resume_from", default=None)

    p.add_argument("--max_input_size", type=int, default=256)
    p.add_argument("--max_lookahead", type=int, default=64)

    p.add_argument("--curr_window", type=int, default=200)
    p.add_argument("--curr_check_every", type=int, default=200)
    p.add_argument("--curr_threshold", type=float, default=0.98)
    return p


def main():
    args = build_argparser().parse_args()
    os.makedirs(os.path.join(args.run_dir, "checkpoints"), exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum,
    )

    if accelerator.is_main_process:
        print(
            f"num_processes={accelerator.num_processes} "
            f"process_index={accelerator.process_index} "
            f"local_process_index={accelerator.local_process_index} "
            f"mixed_precision={accelerator.mixed_precision}",
            flush=True,
        )

    model, tokenizer = load_model_and_tokenizer(args.model_name)

    synth_ds, item_iter = synthetic_nl_stream(
        max_input_size=args.max_input_size,
        max_lookahead=args.max_lookahead,
        seed=args.seed,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    dataset = PromptAnswerDataset(item_iter=item_iter)
    collate_fn = build_synth_collate_fn(tokenizer=tokenizer, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.micro_batch, collate_fn=collate_fn)

    curr_deque = deque(maxlen=args.curr_window)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()

    loader_iter = iter(loader)

    for step in range(args.max_steps):
        with accelerator.accumulate(model):
            batch = next(loader_iter)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = weighted_fullseq_loss(
                logits=outputs.logits,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pred_mask=batch["pred_mask"],
            )

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        correct_list = full_word_correct_from_logits(outputs.logits.detach(), batch["labels_pred"].detach())
        curr_deque.extend(correct_list)

        if accelerator.is_main_process and step % 10 == 0:
            print(f"step={step} loss={loss.item():.4f}", flush=True)

        if (step + 1) % args.curr_check_every == 0 and len(curr_deque) == args.curr_window:
            acc_local = sum(curr_deque) / len(curr_deque)
            acc = reduce_mean(acc_local, accelerator)

            if accelerator.is_main_process:
                print(
                    f"curr step={step+1} stage={synth_ds.stage} "
                    f"alpha={synth_ds.current_alpha():.4f} "
                    f"full_word_acc@{args.curr_window}={acc:.4f}",
                    flush=True,
                )

            advance_any = reduce_max_bool(acc >= args.curr_threshold, accelerator)
            if advance_any:
                synth_ds.increment_stage(1)
                curr_deque.clear()

                if accelerator.is_main_process:
                    print(
                        f"advanced curriculum -> stage={synth_ds.stage} "
                        f"alpha={synth_ds.current_alpha():.4f}",
                        flush=True,
                    )

                if synth_ds.stage > synth_ds.max_lookahead:
                    if accelerator.is_main_process:
                        print("finished curriculum (stage > max_lookahead).", flush=True)
                    break

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
