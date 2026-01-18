import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def tokenize_leading_space(tokenizer, s: str) -> List[int]:
    s = s if s.startswith(" ") else (" " + s)
    return tokenizer(s, add_special_tokens=False)["input_ids"]


def load_jsonl_examples(
    path: Path,
    subset_size: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Load examples from JSONL with input_text/output_texts format."""
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("input_text")
            outputs = row.get("output_texts", [])
            if prompt and outputs:
                examples.append({"prompt": prompt, "answer": outputs[0]})

    if subset_size and len(examples) > subset_size:
        examples = random.Random(seed).sample(examples, subset_size)

    return examples


@torch.no_grad()
def full_word_correct(logits: torch.Tensor, labels_pred: torch.Tensor) -> List[bool]:
    """Check if full answer was predicted correctly for each example in batch."""
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


def checkpoint_dir(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"step_{step:08d}"
