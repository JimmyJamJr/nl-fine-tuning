import os
import re
import gc
import sys
import json
import random
import warnings
import datetime
import math
from collections import deque
from typing import List, Tuple, Dict, Any, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging as hf_logging

from peft import LoraConfig, get_peft_model, TaskType

from nl_generator import NaturalLanguageGraphGenerator

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# --- Runtime env sanity for Accelerate / FSDP / NCCL (must be set before Trainer builds Accelerator) ---
os.environ.setdefault("ACCELERATE_DISPATCH_BATCHES", "false")
os.environ.setdefault("ACCELERATE_SPLIT_BATCHES", "true")
os.environ.setdefault("ACCELERATE_USE_DATA_LOADER_SHARDING", "false")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
if os.environ.get("ACCELERATE_FSDP_MIN_NUM_PARAMS", "").endswith(".0"):
    os.environ["ACCELERATE_FSDP_MIN_NUM_PARAMS"] = os.environ["ACCELERATE_FSDP_MIN_NUM_PARAMS"].split(".")[0]
if "MASTER_PORT" not in os.environ:
    try:
        _jid = int(os.environ.get("SLURM_JOB_ID", "0") or 0)
    except Exception:
        _jid = 0
    os.environ["MASTER_PORT"] = str(10000 + (_jid % 50000))


# Util functions for multi-GPU

def dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    if dist_is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    if dist_is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    return get_rank() == 0


def rank_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def barrier():
    if dist_is_initialized():
        torch.distributed.barrier()


def broadcast_object(obj, src: int = 0):
    if not dist_is_initialized():
        return obj
    obj_list = [obj]
    torch.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


# Function for setting seed across libraries and GPUs
def set_all_seeds(seed: int):
    r = get_rank()
    true_seed = (seed or 0) + r * 9973
    rank_print(f"[SEED] Setting all random seeds to {true_seed}")
    random.seed(true_seed)
    np.random.seed(true_seed)
    torch.manual_seed(true_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(true_seed)
        torch.cuda.manual_seed_all(true_seed)
    os.environ["PYTHONHASHSEED"] = str(true_seed)
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(true_seed)
    except Exception:
        pass


# ================== Curriculum state persistence ==================

def _save_curriculum_state(dirpath: str, stage: int, stage_start_step: int) -> None:
    try:
        os.makedirs(dirpath, exist_ok=True)
        fp = os.path.join(dirpath, "curriculum_state.json")
        with open(fp, "w") as f:
            json.dump({"stage": int(stage), "stage_start_step": int(stage_start_step)}, f)
    except Exception as e:
        rank_print(f"[CURRICULUM][WARN] Failed to save state: {e}")


def _try_restore_curriculum_state(path: Optional[str], dataset, curriculum) -> bool:
    if not path:
        return False
    fp = os.path.join(path, "curriculum_state.json")
    if not os.path.isfile(fp):
        return False
    try:
        with open(fp) as f:
            cs = json.load(f)
        dataset.stage = int(cs.get("stage", dataset.stage))
        curriculum.stage_start_step = int(cs.get("stage_start_step", 0))
        rank_print(
            f"[CURRICULUM] Restored stage={dataset.stage}, stage_start_step={curriculum.stage_start_step} from {fp}")
        return True
    except Exception as e:
        rank_print(f"[CURRICULUM][WARN] Failed to restore from {fp}: {e}")
        return False


# ================== Task helpers ==================

def _determine_task_type(task: str, input_text: str) -> str:
    if task == "si":
        text = input_text.strip()
        if text.endswith(" is"):
            if ", then" in text.split(".")[-1]:
                return "inference"
            else:
                return "selection"
    return task


def _get_end_tokens(task_type: str) -> str:
    return ", then" if task_type == "selection" else ". "


def _tokenize_leading_space(tokenizer, s: str) -> List[int]:
    return tokenizer(" " + s, add_special_tokens=False)["input_ids"]


# ================== Effective lookahead (SEARCH) ==================

def effective_search_L(alpha: float,
                       n: int,
                       max_lookahead_cap: Optional[int] = None,
                       tokens_per_edge: int = 3,
                       fixed_tokens: int = 4,
                       reserve_edges: int = 1) -> int:
    edges_unscaled = max(0, (n - fixed_tokens) // tokens_per_edge)
    max_edges = int(alpha * edges_unscaled)
    safe_edges = max(0, max_edges - reserve_edges)

    L_edges = safe_edges // 2
    L_tokens = max(0, (n - fixed_tokens) // (2 * tokens_per_edge))

    L_eff = min(L_edges, L_tokens)
    if max_lookahead_cap:
        L_eff = min(L_eff, int(max_lookahead_cap))
    return L_eff


# ================== Redaction helpers ==================
_name_given_re = re.compile(
    r"(Given that\s+([A-Z][a-z]+)\s+is\s+)([a-z]+)(\s*,?\s+and we want to prove\s+\2\s+is\s+[a-z]+\.?)"
)
_proof_first_sentence_re_tpl = r"Proof:\s*{name}\s+is\s+([a-z]+)(\.)"


def _extract_name_and_given_word(text: str) -> Tuple[Optional[str], Optional[str]]:
    m = _name_given_re.search(text)
    if m:
        return m.group(2), m.group(3)
    m2 = re.search(r"Proof:\s*([A-Z][a-z]+)\s+is\s+([a-z]+)\.", text)
    if m2:
        return m2.group(1), m2.group(2)
    return None, None


def _redact_given_and_first_proof_line(text: str, token: str = "_____") -> Tuple[
    Optional[str], Optional[str], Optional[str]]:
    name, word = _extract_name_and_given_word(text)
    if not (name and word):
        return None, None, None

    out = text
    out = _name_given_re.sub(lambda m: m.group(1) + token + m.group(4), out, count=1)
    proof_re = re.compile(_proof_first_sentence_re_tpl.format(name=re.escape(name)))
    out = proof_re.sub(lambda m: f"Proof: {name} is {token}{m.group(2)}", out, count=1)
    return out, name, word


def _redact_first_k_successors(text: str,
                               name: str,
                               given_word: str,
                               token: str = "_____",
                               k: int = 2) -> str:
    if k <= 0:
        return text

    patterns = [
        rf"If\s+(?:{re.escape(name)}|[A-Z][a-z]+|someone|Someone|a person|A person)\s+is\s+{re.escape(given_word)},\s+then\s+(?:they|{re.escape(name)}|[A-Z][a-z]+)\s+are\s+([a-z]+)\.",
        rf"Everyone\s+that\s+is\s+{re.escape(given_word)}\s+is\s+([a-z]+)\."
    ]

    s = text
    matches: List[Tuple[int, int, int]] = []
    for pat in patterns:
        pat_re = re.compile(pat)
        for m in pat_re.finditer(s):
            matches.append((m.start(1), m.end(1), m.start()))

    matches.sort(key=lambda t: t[2])

    offset = 0
    replaced = 0
    for g_start, g_end, _ in matches:
        if replaced >= k:
            break
        g_start += offset
        g_end += offset
        s = s[:g_start] + token + s[g_end:]
        offset += len(token) - (g_end - g_start)
        replaced += 1

    return s


# Build a evaluation dataset where some of the attributes are redacted
# as sanity check (should expect low accuracy)
def build_redacted_eval_set(
        inputs: List[str],
        labels: List[List[str]],
        token: str = "_____",
        max_n: Optional[int] = None,
) -> Tuple[List[str], List[List[str]]]:
    red_inputs, red_labels = [], []
    for x, y in zip(inputs, labels):
        red, name, word = _redact_given_and_first_proof_line(x, token=token)
        if not red or red == x or not (name and word):
            continue
        red2 = _redact_first_k_successors(red, name, word, token=token, k=2)
        red_inputs.append(red2)
        red_labels.append(y)
        if max_n is not None and len(red_inputs) >= max_n:
            break
    return red_inputs, red_labels


class SinglePathARDataset(Dataset):
    def __init__(
            self,
            task: str,
            tokenizer,
            stage: int = 1,
            n_stages: int = 10,
            base_alpha: float = 0.15,
            max_input_size: int = 256,
            reserved_inputs: Optional[Set[str]] = None,
            num_shots: int = 0,
            seed: Optional[int] = None,
            store_examples: bool = False,
            store_cap: int = 1000,
            **task_kwargs,
    ):
        self.task = task
        self.tokenizer = tokenizer
        self.stage = stage
        self.n_stages = n_stages
        self.base_alpha = base_alpha
        self.max_input_size = max_input_size
        self.reserved_inputs = reserved_inputs or set()
        self.num_shots = num_shots
        self.seed = seed
        self.task_kwargs = task_kwargs

        self._store = bool(store_examples)
        self._store_cap = int(store_cap)
        self._seen_inputs: deque = deque(maxlen=self._store_cap)
        self._seen_labels: deque = deque(maxlen=self._store_cap)

        self.few_shot_examples = self._build_few_shots(num_shots, seed)
        self.trunc_warns = 0
        self.trunc_warns_max = 10

        # Epoch size - number of samples per "epoch"
        self.epoch_size = 50000

        rank_print(
            f"[DATASET] Map-style dataset | epoch_size={self.epoch_size} | store_seen={self._store} cap={self._store_cap}")

    def __len__(self):
        """Return epoch size for PyTorch DataLoader"""
        return self.epoch_size

    def get_seen_samples(self) -> Tuple[List[str], List[List[str]]]:
        """Return stored training samples"""
        return list(self._seen_inputs), list(self._seen_labels)

    def _build_few_shots(self, k: int, seed: Optional[int]):
        """Build few-shot examples for prompting"""
        if k <= 0:
            return []
        fs_seed = (seed or 0) + 12345
        g = NaturalLanguageGraphGenerator(self.max_input_size, seed=fs_seed)
        batch = g.generate_batch(self.task, batch_size=max(k, 3), alpha=0.5, **self.task_kwargs)
        out = []
        for ex in batch:
            if ex and ex.output_texts:
                out.append({"input": ex.input_text, "output": ex.output_texts[0]})
                if len(out) >= k:
                    break
        return out[:k]

    def _stage_alpha(self) -> float:
        """Calculate alpha for current curriculum stage"""
        if self.stage >= self.n_stages:
            return 1.0
        return self.base_alpha + (1.0 - self.base_alpha) * (self.stage - 1) / max(self.n_stages - 1, 1)

    def _shots_prefix(self) -> str:
        """Build few-shot prefix for prompts"""
        if not self.few_shot_examples:
            return ""
        parts = []
        for ex in self.few_shot_examples:
            tt = _determine_task_type(self.task, ex["input"])
            parts.append(f"{ex['input']} {ex['output']}{_get_end_tokens(tt)}")
        return "\n\n".join(parts) + ("\n\n" if parts else "")

    def _warn_trunc(self, cur_len, max_len):
        """Warn about truncated samples (limited warnings)"""
        if self.trunc_warns < self.trunc_warns_max:
            rank_print(f"[WARN] Skipping long sample ({cur_len} > {max_len})")
            self.trunc_warns += 1
            if self.trunc_warns == self.trunc_warns_max:
                rank_print("[WARN] Suppressing further truncation warnings")

    def __getitem__(self, idx):
        """
        Generate sample on-demand for given index.
        Index is used to vary the seed for diversity across samples.
        """
        # Build unique seed for this sample
        rank = get_rank()
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0

        # Unique seed per sample: base_seed + rank_offset + worker_offset + index
        sample_seed = (self.seed or 0) + rank * 9973 + worker_id * 997 + idx
        rng = random.Random(sample_seed)

        # Create generator for this sample
        gen = NaturalLanguageGraphGenerator(self.max_input_size, seed=sample_seed)
        max_len = getattr(self.tokenizer, "model_max_length", 512)

        # Try to generate a valid sample (up to 100 attempts)
        max_attempts = 100
        for attempt in range(max_attempts):
            batch = gen.generate_batch(
                self.task,
                batch_size=1,
                reserved_inputs=self.reserved_inputs,
                alpha=self._stage_alpha(),
                **self.task_kwargs
            )

            if not (batch and batch[0] and batch[0].output_texts):
                continue

            ex = batch[0]
            if ex.input_text in self.reserved_inputs or not ex.output_texts:
                continue

            # Build prompt with few-shot examples
            shots = self._shots_prefix()
            prompt_text = (shots + ex.input_text) if shots else ex.input_text

            # Choose random answer from valid outputs
            chosen = rng.choice(ex.output_texts)
            task_type = _determine_task_type(self.task, ex.input_text)

            # Tokenize
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=True, truncation=False)["input_ids"]
            ans_ids = _tokenize_leading_space(self.tokenizer, chosen)
            end_ids = self.tokenizer(_get_end_tokens(task_type), add_special_tokens=False)["input_ids"]
            full_len = len(prompt_ids) + len(ans_ids) + len(end_ids)

            # Skip if too long
            if full_len > max_len:
                self._warn_trunc(full_len, max_len)
                continue

            # Build final sequences
            input_ids = prompt_ids + ans_ids + end_ids
            labels = [-100] * len(prompt_ids) + ans_ids + end_ids
            attention_mask = [1] * len(input_ids)

            # Store if tracking seen samples
            if self._store and len(self._seen_inputs) < self._store_cap:
                self._seen_inputs.append(ex.input_text)
                self._seen_labels.append(list(ex.output_texts))

            # Build valid first token targets
            first_union = sorted({
                _tokenize_leading_space(self.tokenizer, a)[0]
                for a in ex.output_texts
                if _tokenize_leading_space(self.tokenizer, a)
            })

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "prompt_len": len(prompt_ids),
                "valid_first_targets": first_union,
            }

        # Failed to generate valid sample after max_attempts
        raise RuntimeError(
            f"[DATASET] Failed to generate valid sample after {max_attempts} attempts. "
            f"Stage {self.stage}, alpha={self._stage_alpha():.2f}, idx={idx}. "
            f"This may indicate curriculum is too aggressive or max_input_size is too small."
        )


# ================== Collator ==================

def make_collate(tokenizer):
    def collate(features):
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # Dynamic sequence length (batch-specific)
        max_len = max(len(f["input_ids"]) for f in features)

        # Dynamic first_k size (batch-specific, minimum 16 for safety)
        dyn_max_k = max(len(f["valid_first_targets"]) for f in features)
        max_k = max(dyn_max_k, 16)  # Keep minimum of 16 as safety margin

        input_ids, attn, labels, prompt_lens = [], [], [], []
        vtargets, vmask = [], []

        for f in features:
            L = len(f["input_ids"])
            pad = max_len - L
            input_ids.append(f["input_ids"] + [pad_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
            prompt_lens.append(f["prompt_len"])

            K = len(f["valid_first_targets"])
            pad_k = max_k - K
            vtargets.append(f["valid_first_targets"] + [pad_id] * pad_k)
            vmask.append([1] * K + [0] * pad_k)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_len": torch.tensor(prompt_lens, dtype=torch.long),
            "valid_first_targets": torch.tensor(vtargets, dtype=torch.long),
            "valid_first_mask": torch.tensor(vmask, dtype=torch.bool),
        }

    return collate


# ================== Trainer (soft first-token CE) ==================
class SinglePathARTrainer(Trainer):
    def __init__(self, *args, first_token_soft_weight=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_token_soft_weight = first_token_soft_weight
        self.recent_losses = deque(maxlen=100)
        self.first_token_correct = deque(maxlen=200)
        self.full_word_correct = deque(maxlen=200)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompt_len = inputs.pop("prompt_len")
        valid_first = inputs.pop("valid_first_targets")
        valid_first_mask = inputs.pop("valid_first_mask")

        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        ce_all = F.cross_entropy(
            shift_logits.permute(0, 2, 1),
            torch.clamp(shift_labels, min=0),
            reduction="none",
        )
        valid_mask = (shift_labels != -100)
        ce_all = ce_all * valid_mask.float()
        B, Tm1, _ = shift_logits.shape

        # Soft-CE at first answer position
        for i in range(B):
            first_idx = prompt_len[i].item() - 1
            if first_idx < 0 or first_idx >= Tm1:
                continue
            if not valid_mask[i, first_idx]:
                continue
            mask = valid_first_mask[i]
            if mask.any():
                ids = valid_first[i][mask]
                logp = F.log_softmax(shift_logits[i, first_idx, :], dim=-1)
                soft = torch.zeros_like(logp)
                soft[ids] = 1.0 / ids.numel()
                soft_ce = -(soft * logp).sum()
                ce_all[i, first_idx] = (
                        self.first_token_soft_weight * soft_ce +
                        (1.0 - self.first_token_soft_weight) * ce_all[i, first_idx]
                )

        denom = valid_mask.sum().clamp_min(1)
        loss = ce_all.sum() / denom
        self.recent_losses.append(loss.item())

        # quick train-side tracking (TF-style)
        with torch.no_grad():
            for i in range(B):
                first_idx = prompt_len[i].item() - 1
                if 0 <= first_idx < Tm1 and valid_mask[i, first_idx]:
                    pred_first = torch.argmax(shift_logits[i, first_idx, :]).item()
                    valid_ids = set(valid_first[i][valid_first_mask[i]].tolist())
                    self.first_token_correct.append(pred_first in valid_ids)

                ok = True
                idxs = torch.nonzero(valid_mask[i], as_tuple=False).squeeze(-1)
                for j in idxs.tolist():
                    gold = shift_labels[i, j].item()
                    pred = torch.argmax(shift_logits[i, j, :]).item()
                    if gold != pred:
                        ok = False
                        break
                self.full_word_correct.append(ok)

        return (loss, outputs) if return_outputs else loss

    def get_first_token_acc(self):
        return (sum(self.first_token_correct) / len(self.first_token_correct)) if self.first_token_correct else 0.0

    def get_full_word_acc(self):
        return (sum(self.full_word_correct) / len(self.full_word_correct)) if self.full_word_correct else 0.0


# ================== Eval helpers ==================
@torch.inference_mode()
def run_eval_teacher_forced_parity(
        model,
        tokenizer,
        task: str,
        eval_inputs: List[str],
        eval_labels: List[List[str]],
        num_shots: int,
        max_input_size: int,
        seed: Optional[int],
        print_mistakes: int = 0,
        gc_every: int = 0,
        **task_kwargs,
) -> Dict[str, Any]:
    # CRITICAL: Synchronize all ranks before starting eval
    barrier()
    rank_print(f"[EVAL-TF] Rank {get_rank()} starting eval on {len(eval_inputs)} samples")

    device = next(model.parameters()).device

    few_shots = []
    if num_shots > 0:
        fs_seed = (seed or 0) + 12345
        gfs = NaturalLanguageGraphGenerator(max_input_size, seed=fs_seed)
        batch = gfs.generate_batch(task, batch_size=max(num_shots, 3), alpha=0.5, **task_kwargs)
        for ex in batch:
            if ex and ex.output_texts:
                few_shots.append({"input": ex.input_text, "output": ex.output_texts[0]})
                if len(few_shots) >= num_shots:
                    break

    def shots_prefix():
        if not few_shots:
            return ""
        parts = [f"{ex['input']} {ex['output']}{_get_end_tokens(_determine_task_type(task, ex['input']))}" for ex in
                 few_shots]
        return "\n\n".join(parts) + "\n\n"

    first_ok, full_ok, total = 0, 0, 0
    mistakes_printed = 0
    allow_print = is_main_process() and print_mistakes > 0

    for idx, (x, ys) in enumerate(zip(eval_inputs, eval_labels), 1):
        ys_list = ys if isinstance(ys, list) else [ys]
        prompt = (shots_prefix() + x) if few_shots else x
        enc = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        prompt_len = enc["input_ids"].shape[1]
        ttype = _determine_task_type(task, x)
        end_ids = tokenizer(_get_end_tokens(ttype), add_special_tokens=False)["input_ids"]

        union = set()
        candidates = []
        for gold in ys_list:
            ids = _tokenize_leading_space(tokenizer, gold)
            if ids:
                union.add(ids[0])
                candidates.append(ids + end_ids)

        if not candidates:
            total += 1
            continue

        out_prompt = model(**enc)
        fidx = prompt_len - 1
        pred_first_id = torch.argmax(out_prompt.logits[0, fidx, :]).item()
        first_ok += int(pred_first_id in union)

        matched = False
        pred_word = None
        for cand in candidates:
            full = torch.tensor(
                enc["input_ids"].tolist()[0] + cand,
                dtype=torch.long, device=device
            ).unsqueeze(0)
            attn = torch.ones_like(full)
            out = model(input_ids=full, attention_mask=attn)
            slogits = out.logits[:, :-1, :]
            slabels = full[:, 1:]
            start = prompt_len - 1
            ok = True
            for j in range(start, slabels.shape[1]):
                gold_tok = slabels[0, j].item()
                pred_tok = torch.argmax(slogits[0, j, :]).item()
                if pred_tok != gold_tok:
                    ok = False
                    if pred_word is None and j == start:
                        pred_word = tokenizer.decode([pred_tok], skip_special_tokens=True).strip()
                    break
            if ok:
                matched = True
                break

        if pred_word is None:
            pred_word = tokenizer.decode([pred_first_id], skip_special_tokens=True).strip()

        if not matched and allow_print and mistakes_printed < print_mistakes:
            print("\n" + "=" * 60)
            print(f"[TF MISTAKE #{mistakes_printed + 1}] (rank {get_rank()})")
            print("Prompt fed to model:\n" + prompt)
            print("Expected any of:", ys_list)
            print(f"Predicted first-token id: {pred_first_id} (word guess: '{pred_word}')")
            mistakes_printed += 1

        full_ok += int(matched)
        total += 1

        if gc_every > 0 and (idx % gc_every == 0):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # CRITICAL: Synchronize all ranks before returning
    barrier()
    rank_print(f"[EVAL-TF] Rank {get_rank()} completed eval")

    return {
        "first_token_acc": (first_ok / total) if total else 0.0,
        "full_word_acc": (full_ok / total) if total else 0.0,
        "first_token_hits": first_ok,
        "full_word_hits": full_ok,
        "total": total,
    }


@torch.inference_mode()
def run_eval_greedy_readable(
        model,
        tokenizer,
        task: str,
        inputs: List[str],
        labels: List[List[str]],
        print_examples: int = 0,
        **kwargs,
) -> Dict[str, Any]:
    barrier()
    rank_print(f"[EVAL-GREEDY] Rank {get_rank()} starting generation on {len(inputs)} samples")
    device = next(model.parameters()).device

    correct_first, correct_full, total = 0, 0, 0

    # Ensure pad token exists for generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for i, (x, ys) in enumerate(zip(inputs, labels)):
        ys = ys if isinstance(ys, list) else [ys]

        # 1. Tokenize Prompt
        enc = tokenizer(x, return_tensors="pt").to(device)
        prompt_len = enc.input_ids.shape[1]

        # 2. Optimized Generation (Greedy)
        # do_sample=False ensures deterministic greedy decoding
        gen_out = model.generate(
            **enc,
            max_new_tokens=24,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False
        )

        # 3. Extract Answer
        # We slice [prompt_len:] to get only the new tokens
        gen_ids = gen_out[0][prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Split at punctuation or space to isolate the "answer word"
        pred_word = re.split(r"[.,\s]+", gen_text)[0] if gen_text else ""

        # 4. Calculate Metrics
        # Metric A: First Token Accuracy
        valid_first_ids = {
            _tokenize_leading_space(tokenizer, y)[0]
            for y in ys
            if _tokenize_leading_space(tokenizer, y)
        }
        first_ok = (len(gen_ids) > 0 and gen_ids[0].item() in valid_first_ids)
        correct_first += int(first_ok)

        # Metric B: Full Word Accuracy
        full_ok = any(pred_word.lower() == y.lower() for y in ys)
        correct_full += int(full_ok)
        total += 1

        # 5. Print Mistakes (Optional)
        if is_main_process() and (not first_ok or not full_ok) and print_examples > 0:
            print("\n" + "=" * 60)
            print(f"[GREEDY MISTAKE] (rank {get_rank()})")
            print(f"Prompt (tail): ...{x[-100:]}")
            print(f"Gold Options: {ys}")
            print(f"Predicted Word: '{pred_word}'")
            print(f"Full Generation: '{gen_text}'")
            print_examples -= 1

    barrier()
    rank_print(f"[EVAL-GREEDY] Rank {get_rank()} completed eval")

    return {
        "first_token_acc": (correct_first / total) if total else 0.0,
        "full_word_acc": (correct_full / total) if total else 0.0,
        "first_token_hits": correct_first,
        "full_word_hits": correct_full,
        "total": total,
    }


# ================== Curriculum callback (FULL-word metric only) ==================
class FirstTokenCurriculum(TrainerCallback):
    def __init__(
            self,
            dataset: SinglePathARDataset,
            n_stages: int,
            accuracy_threshold: float,
            min_steps_per_stage: int,
            check_every: int,
    ):
        self.dataset = dataset
        self.n_stages = n_stages
        self.acc_thr = accuracy_threshold
        self.min_steps = min_steps_per_stage
        self.check_every = check_every
        self.trainer: Optional[SinglePathARTrainer] = None
        self.stage_start_step = 0
        self._last_log = -1
        self.finished = False

        # Speed tracking
        self.samples_processed = 0
        self.last_time = None
        self.speed_history = deque(maxlen=100)  # Moving average
        self.stage_start_time = None

        # Resume cooldown
        self.resume_cooldown_steps = 50
        self.last_resume_step = -1

        # NEW: Capture grad_norm from logs
        self.last_grad_norm = 0.0
        self.last_lr = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture grad_norm and lr from Transformers' logs"""
        if logs:
            self.last_grad_norm = logs.get('grad_norm', self.last_grad_norm)
            self.last_lr = logs.get('learning_rate', self.last_lr)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        """Track time at step start"""
        if self.last_time is None:
            self.last_time = datetime.datetime.now()
            self.stage_start_time = self.last_time
        return control

    def _current_metric(self) -> float:
        if self.trainer is None:
            return 0.0
        return self.trainer.get_full_word_acc()

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        _save_curriculum_state(checkpoint_dir, self.dataset.stage, self.stage_start_step)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is None or self.finished or state.global_step == 0:
            return control

        # Calculate training speed
        current_time = datetime.datetime.now()
        if self.last_time is not None:
            time_delta = (current_time - self.last_time).total_seconds()
            if time_delta > 0:
                # Samples per step = batch_size * gradient_accumulation_steps * world_size
                samples_per_step = (
                        self.trainer.args.per_device_train_batch_size *
                        self.trainer.args.gradient_accumulation_steps *
                        get_world_size()
                )
                samples_per_sec = samples_per_step / time_delta
                self.speed_history.append(samples_per_sec)
                self.samples_processed += samples_per_step

        self.last_time = current_time

        # Log every 10 steps
        if state.global_step % 10 == 0 and state.global_step != self._last_log and is_main_process():
            loss = np.mean(self.trainer.recent_losses) if self.trainer.recent_losses else 0.0
            f1 = self.trainer.get_first_token_acc()
            fw = self.trainer.get_full_word_acc()

            # Calculate speed metrics
            instant_speed = self.speed_history[-1] if self.speed_history else 0
            avg_speed = np.mean(self.speed_history) if self.speed_history else 0

            # Time in current stage
            if self.stage_start_time:
                stage_time = (current_time - self.stage_start_time).total_seconds()
                stage_time_str = f"{stage_time / 60:.1f}m"
            else:
                stage_time_str = "0m"

            # Use captured values
            lr = self.last_lr
            grad_norm = self.last_grad_norm

            # Show warmup status
            warmup_msg = ""
            if self.last_resume_step >= 0 and (state.global_step - self.last_resume_step) < self.resume_cooldown_steps:
                remaining = self.resume_cooldown_steps - (state.global_step - self.last_resume_step)
                warmup_msg = f" | Warmup={remaining}"

            # Show loss window size
            loss_window = len(self.trainer.recent_losses)

            print(f"[Stage {self.dataset.stage}/{self.n_stages}] step {state.global_step} | "
                  f"loss_avg={loss:.4f}({loss_window}) | First={f1:.2%} | Full={fw:.2%} | "
                  f"lr={lr:.2e} | grad_norm={grad_norm:.2f} | "
                  f"Speed={instant_speed:.1f} samples/s (avg={avg_speed:.1f}) | "
                  f"Stage time={stage_time_str}{warmup_msg}")
            self._last_log = state.global_step

        # Skip curriculum checks during warmup period after resume
        if self.last_resume_step >= 0 and (state.global_step - self.last_resume_step) < self.resume_cooldown_steps:
            if is_main_process() and state.global_step % 10 == 0:
                remaining = self.resume_cooldown_steps - (state.global_step - self.last_resume_step)
                print(f"[CURRICULUM] Skipping checks during warmup ({remaining} steps remaining)")
            return control

        # Check for stage advancement
        if state.global_step % self.check_every == 0 and (state.global_step - self.stage_start_step) >= self.min_steps:
            m = self._current_metric()
            if is_main_process():
                print(f"[CHECK] Stage {self.dataset.stage} full={m:.2%} target>={self.acc_thr:.2%}")
            if m >= self.acc_thr:
                if is_main_process():
                    # Summary for completed stage
                    if self.stage_start_time:
                        total_stage_time = (current_time - self.stage_start_time).total_seconds()
                        steps_in_stage = state.global_step - self.stage_start_step
                        print(f"[COMPLETE] Stage {self.dataset.stage} complete in {steps_in_stage} steps, "
                              f"{total_stage_time / 60:.1f} minutes")
                    else:
                        print(f"[COMPLETE] Stage {self.dataset.stage} complete")

                if self.dataset.stage < self.n_stages:
                    self.dataset.stage += 1
                    self.stage_start_step = state.global_step
                    self.stage_start_time = current_time
                    self.trainer.first_token_correct.clear()
                    self.trainer.full_word_correct.clear()

                    new_alpha = self.dataset._stage_alpha()
                    if is_main_process():
                        msg = f" -> Advanced to stage {self.dataset.stage} | alpha={new_alpha:.3f}"
                        if getattr(self.dataset, "task", None) == "search":
                            n = getattr(self.dataset, "max_input_size", 256)
                            cap = None
                            try:
                                cap = int(self.dataset.task_kwargs.get("max_lookahead", 0)) or None
                            except Exception:
                                cap = None
                            L_eff = effective_search_L(new_alpha, n, max_lookahead_cap=cap)
                            msg += f" | effective max_lookahead={L_eff} (n={n}, cap={cap})"
                        print(msg)
                else:
                    if is_main_process():
                        print("[FINISHED] Curriculum complete")
                        # Final summary
                        print(f"[SUMMARY] Total samples: {self.samples_processed:,} | "
                              f"Avg speed: {np.mean(self.speed_history):.1f} samples/s")
                    control.should_training_stop = True
                    self.finished = True

        # Periodic memory cleanup (every 100 steps)
        if state.global_step > 0 and state.global_step % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if is_main_process() and state.global_step % 1000 == 0:
                    mem_allocated = torch.cuda.memory_allocated() / 1e9
                    mem_reserved = torch.cuda.memory_reserved() / 1e9
                    print(
                        f"[MEM] Step {state.global_step} | Allocated: {mem_allocated:.2f}GB | Reserved: {mem_reserved:.2f}GB")

        return control


# out-of-memory auto-scaling batch size and gradient accumulation training loop
def train_with_oom_autoscale(
        trainer,
        init_bs: int,
        init_gas: int,
        resume_from_checkpoint: Optional[str]
) -> None:
    bs = max(1, int(init_bs))
    gas = max(1, int(init_gas))
    current_resume_ckpt = resume_from_checkpoint

    max_retries = max(int(math.ceil(math.log2(bs))) + 2, 5)
    min_bs = 1

    for attempt in range(1, max_retries + 1):
        if is_main_process():
            print(f"\n[OOM-AUTO] Attempt {attempt}/{max_retries} | BS={bs} GAS={gas}")
            if current_resume_ckpt:
                print(f"[OOM-AUTO] Resuming from: {current_resume_ckpt}")

        trainer.args.per_device_train_batch_size = bs
        trainer.args.gradient_accumulation_steps = gas

        try:
            trainer.train(resume_from_checkpoint=current_resume_ckpt)
            return

        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise  # Not OOM, re-raise

            if is_main_process():
                print(f"[OOM-AUTO] OOM on attempt {attempt}")

            # Clear cache immediately
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Find last good checkpoint
            if current_resume_ckpt is None:
                last_ckpt = get_last_checkpoint(trainer.args.output_dir)
                if last_ckpt:
                    current_resume_ckpt = last_ckpt
                    if is_main_process():
                        print(f"[OOM-AUTO] Found checkpoint: {current_resume_ckpt}")

            if bs > min_bs:
                old_bs = bs
                bs = max(min_bs, bs // 2)
                gas = gas * 2
                if is_main_process():
                    print(f"[OOM-AUTO] Reducing: BS {old_bs}->{bs}, GAS {gas // 2}→{gas}")
            else:
                if is_main_process():
                    print("[OOM-AUTO] Already at min batch size, retrying...")

    # Final attempt without catching
    if is_main_process():
        print("[OOM-AUTO] Final attempt without error handling")
    trainer.train(resume_from_checkpoint=current_resume_ckpt)


# Generate eval set in a similar way to the training data
@torch.inference_mode()
def generate_eval_like_training(
        n_samples: int,
        task: str,
        tokenizer,
        max_input_size: int,
        alpha: float,
        num_shots: int,
        reserved_inputs: Set[str],
        seed: Optional[int],
        **task_kwargs,
) -> Tuple[List[str], List[List[str]], List[str]]:
    g = NaturalLanguageGraphGenerator(max_input_size, seed=seed)
    eval_inputs, eval_labels, picked_answers = [], [], []

    def build_shots():
        if num_shots <= 0:
            return []
        fs_seed = (seed or 0) + 12345
        gfs = NaturalLanguageGraphGenerator(max_input_size, seed=fs_seed)
        batch = gfs.generate_batch(task, batch_size=max(num_shots, 3), alpha=0.5, **task_kwargs)
        out = []
        for ex in batch:
            if ex and ex.output_texts:
                out.append({"input": ex.input_text, "output": ex.output_texts[0]})
                if len(out) >= num_shots:
                    break
        return out[:num_shots]

    few_shots = build_shots()
    shots_prefix = ""
    if few_shots:
        parts = [f"{ex['input']} {ex['output']}{_get_end_tokens(_determine_task_type(task, ex['input']))}" for ex in
                 few_shots]
        shots_prefix = "\n\n".join(parts) + "\n\n"

    max_len = getattr(tokenizer, "model_max_length", 512)
    rng = random.Random((seed or 0) + 424242)

    attempts = 0
    while len(eval_inputs) < n_samples and attempts < n_samples * 10:
        attempts += 1
        batch = g.generate_batch(task, batch_size=1, reserved_inputs=reserved_inputs, alpha=alpha, **task_kwargs)
        if not (batch and batch[0] and batch[0].output_texts):
            continue
        ex = batch[0]
        if ex.input_text in reserved_inputs:
            continue

        chosen = rng.choice(ex.output_texts)
        task_type = _determine_task_type(task, ex.input_text)
        prompt_text = (shots_prefix + ex.input_text) if shots_prefix else ex.input_text

        prompt_ids = tokenizer(prompt_text, add_special_tokens=True, truncation=False)["input_ids"]
        ans_ids = _tokenize_leading_space(tokenizer, chosen)
        end_ids = tokenizer(_get_end_tokens(task_type), add_special_tokens=False)["input_ids"]
        full_len = len(prompt_ids) + len(ans_ids) + len(end_ids)
        if full_len > max_len:
            continue

        eval_inputs.append(ex.input_text)
        eval_labels.append(ex.output_texts)
        picked_answers.append(chosen)

    return eval_inputs, eval_labels, picked_answers


def main():
    rank_print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())
    rank_print("PyTorch version:", torch.__version__)

    import argparse

    p = argparse.ArgumentParser()

    # Task/model
    p.add_argument("--task", type=str, choices=["search", "dfs", "si"], default="si")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./nl_output")  # redirected to scratch

    # FSDP
    p.add_argument("--fsdp_enable", action="store_true")
    p.add_argument("--fsdp_min_num_params", type=int,
                   default=int(os.environ.get("ACCELERATE_FSDP_MIN_NUM_PARAMS", "100000000")))

    # LoRA
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)

    # Seed
    p.add_argument("--seed", type=int, default=1234)

    # Training hyperparams
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--first_token_soft_weight", type=float, default=0.3)

    # Few-shot in prompt
    p.add_argument("--num_shots", type=int, default=0, choices=[0, 1, 2])

    # Curriculum
    p.add_argument("--n_stages", type=int, default=10)
    p.add_argument("--base_alpha", type=float, default=0.15)
    p.add_argument("--accuracy_threshold", type=float, default=0.98)
    p.add_argument("--min_steps_per_stage", type=int, default=500)
    p.add_argument("--check_every", type=int, default=50)

    # Task params
    p.add_argument("--max_input_size", type=int, default=256)
    p.add_argument("--max_lookahead", type=int, default=12)
    p.add_argument("--max_frontier_size", type=int, default=12)
    p.add_argument("--max_branch_size", type=int, default=12)
    p.add_argument("--requested_backtrack", type=int, default=3)

    # Eval sizes / printing
    p.add_argument("--eval_samples", type=int, default=1000)
    p.add_argument("--print_eval_examples", type=int, default=0)
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--skip_final_eval", action="store_true")

    # Redacted sanity eval
    p.add_argument("--eval_redacted_samples", type=int, default=None)
    p.add_argument("--redaction_token", type=str, default="_____")

    # Memory control
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--oom_autoscale", action="store_true")

    # Store & sanity-eval on seen train items
    p.add_argument("--store_seen_train", action="store_true")
    p.add_argument("--store_seen_cap", type=int, default=1000)
    p.add_argument("--eval_seen_train", action="store_true")

    # Scratch / resume
    p.add_argument("--scratch_dir", type=str,
                   default=os.environ.get("SCRATCH") or os.path.join("/scratch", os.environ.get("USER", "user")))
    p.add_argument("--job_id", type=str, default=os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID"))
    p.add_argument("--resume_from_job", type=str, default=None)

    args = p.parse_args()

    # Initialize distributed if needed
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        # Set device rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # Initialize PyTorch distributed
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        rank_print(f"[DISTRIBUTED] Initialized rank {get_rank()}/{get_world_size()} on device cuda:{local_rank}")

    # Seeds
    if args.seed is not None:
        set_all_seeds(args.seed)

    # Scratch dir names
    if args.job_id:
        run_dir_name = f"job_{args.job_id}"
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir_name = f"local_{ts}"

    base_out = os.path.join(args.scratch_dir, "nl_output", args.task)
    args.output_dir = os.path.join(base_out, run_dir_name)
    os.makedirs(args.output_dir, exist_ok=True)

    rank_print("\n" + "=" * 60)
    for k, v in sorted(vars(args).items()):
        rank_print(f"{k:>28}: {v}")
    rank_print("=" * 60 + "\n")
    rank_print("[CKPT] Scratch base      :", args.scratch_dir)
    rank_print("[CKPT] Task output base  :", base_out)
    rank_print("[CKPT] This run dir      :", args.output_dir, "\n")

    # Resume from checkpoint logic
    resume_ckpt = None
    if args.resume_from_job:
        # Explicit resume from a different job
        prev_dir = os.path.join(args.scratch_dir, "nl_output", args.task, f"job_{args.resume_from_job}")
        if os.path.isdir(prev_dir):
            resume_ckpt = get_last_checkpoint(prev_dir)
            rank_print(f"[CKPT] Resuming from job {args.resume_from_job}: {resume_ckpt}")
        else:
            rank_print(f"[CKPT][ERROR] No run dir found for job {args.resume_from_job}")
            rank_print(f"[CKPT]         Expected: {prev_dir}")
            sys.exit(1)
    else:
        if os.path.isdir(args.output_dir):
            resume_ckpt = get_last_checkpoint(args.output_dir)
            if resume_ckpt:
                rank_print(f"[CKPT] Auto-resuming from: {resume_ckpt}")
            else:
                rank_print(f"[CKPT] Fresh start")

    # Extract job directory for curriculum state restoration
    resume_job_dir = None
    if resume_ckpt:
        if "checkpoint-" in os.path.basename(resume_ckpt):
            # This is a checkpoint subdirectory like .../job_4771708/checkpoint-300
            resume_job_dir = os.path.dirname(resume_ckpt)  # Get parent: .../job_4771708
            rank_print(f"[CKPT] Extracted job directory for curriculum: {resume_job_dir}")
        else:
            # This is already a job directory
            resume_job_dir = resume_ckpt

    # Task kwargs
    task_kwargs = {}
    if args.task == "search":
        task_kwargs = {"max_lookahead": args.max_lookahead}
    elif args.task == "dfs":
        task_kwargs = {"requested_backtrack": args.requested_backtrack}
    elif args.task == "si":
        task_kwargs = {"max_frontier_size": args.max_frontier_size, "max_branch_size": args.max_branch_size}

    # Tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, trust_remote_code=True)
    rank_print(f"[TOKENIZER] model_max_length: {tokenizer.model_max_length}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "cache_dir": args.cache_dir,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": "flash_attention_2",
    }

    # Load the pretrained model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # Check what attention implementation is being used
    if is_main_process():
        attn_impl = getattr(model.config, "_attn_implementation", "unknown")
        print(f"[ATTENTION] Using: {attn_impl}")

    # Gradient checkpointing
    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            rank_print("[MEM] Gradient checkpointing enabled")
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
            rank_print("[MEM] model.config.use_cache = False (training)")

    # LoRA
    if args.use_lora:
        rank_print("Applying LoRA...")
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(model, lora)
        if is_main_process():
            model.print_trainable_parameters()

        # FIX: Convert all parameters to uniform dtype for FSDP
        if args.fsdp_enable:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model = model.to(dtype)
            rank_print(f"[FSDP] Converted all parameters to {dtype} for FSDP compatibility")

    # Reserved inputs for deduplication
    reserved_inputs: Set[str] = set()

    # ---------------- Training dataset ----------------
    dataset = SinglePathARDataset(
        task=args.task,
        tokenizer=tokenizer,
        stage=1,
        n_stages=args.n_stages,
        base_alpha=args.base_alpha,
        max_input_size=args.max_input_size,
        reserved_inputs=reserved_inputs,
        num_shots=args.num_shots,
        seed=args.seed,
        store_examples=args.store_seen_train,
        store_cap=args.store_seen_cap,
        **task_kwargs,
    )

    curriculum = FirstTokenCurriculum(
        dataset=dataset,
        n_stages=args.n_stages,
        accuracy_threshold=args.accuracy_threshold,
        min_steps_per_stage=args.min_steps_per_stage,
        check_every=args.check_every,
    )

    if resume_ckpt:
        # Attempt to load curriculum_state.json from inside checkpoint-XXX/
        if _try_restore_curriculum_state(resume_ckpt, dataset, curriculum):
            rank_print(f"[CURRICULUM] Synced state with checkpoint: {resume_ckpt}")
        else:
            rank_print(f"[CURRICULUM] WARNING: Resuming model from {resume_ckpt} but no curriculum state found there.")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=1000,
        logging_steps=10,
        logging_first_step=False,  # supress duplicate logs
        report_to="none",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        save_safetensors=True,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        seed=args.seed if args.seed is not None else 42,
        fsdp=(['full_shard', 'auto_wrap'] if args.fsdp_enable else []),
        fsdp_config=({"min_num_params": int(args.fsdp_min_num_params)} if args.fsdp_enable else None),
        load_best_model_at_end=False,
    )

    callbacks = [curriculum]
    # Create trainer (handles GPU placement)
    trainer = SinglePathARTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=make_collate(tokenizer),
        callbacks=callbacks,
        first_token_soft_weight=args.first_token_soft_weight,
    )
    curriculum.trainer = trainer

    eval_inputs_1, eval_labels_1 = None, None
    eval_inputs_2, eval_labels_2 = None, None

    # Baseline eval
    if not args.skip_baseline:
        rank_print("[BASELINE] Starting baseline evaluation...")

        # For FSDP, we need to use the model through trainer's infrastructure
        # For DDP, we can prepare the model manually
        if not args.fsdp_enable and torch.cuda.is_available():
            rank_print("[BASELINE] Preparing model for DDP baseline eval...")
            trainer.model = trainer.accelerator.prepare_model(trainer.model)
            device = next(trainer.model.parameters()).device
            rank_print(f"[BASELINE] Model is on device: {device}")
        else:
            # For FSDP, the model will be wrapped automatically when needed
            # Just move to device for now
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                device = torch.device(f"cuda:{local_rank}")
                trainer.model = trainer.model.to(device)
                rank_print(f"[BASELINE] Model moved to device: {device} (FSDP will wrap during forward)")

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            rank_print(f"[BASELINE] GPU memory after preparation: {mem:.2f}GB")

        trainer.model.eval()

        # Only rank 0 generates eval data
        if is_main_process():
            rank_print("[BASELINE] Generating eval data...")
            # Samples with full complexity (alpha=1.0)
            eval_inputs_1, eval_labels_1, _ = generate_eval_like_training(
                n_samples=args.eval_samples,
                task=args.task,
                tokenizer=tokenizer,
                max_input_size=args.max_input_size,
                alpha=1.0,
                num_shots=args.num_shots,
                reserved_inputs=reserved_inputs,
                seed=(args.seed or 0) + 42,
                **task_kwargs,
            )

            # Samples with starting complexity (alpha=base_alpha)
            eval_inputs_2, eval_labels_2, _ = generate_eval_like_training(
                n_samples=args.eval_samples,
                task=args.task,
                tokenizer=tokenizer,
                max_input_size=args.max_input_size,
                alpha=args.base_alpha,
                num_shots=args.num_shots,
                reserved_inputs=set(eval_inputs_1),
                seed=(args.seed or 0) + 99,
                **task_kwargs,
            )
            rank_print(f"[BASELINE] Generated {len(eval_inputs_1)} + {len(eval_inputs_2)} eval samples")

            # Synchronize and broadcast to all GPUs
        barrier()
        eval_inputs_1 = broadcast_object(eval_inputs_1, src=0)
        eval_labels_1 = broadcast_object(eval_labels_1, src=0)
        eval_inputs_2 = broadcast_object(eval_inputs_2, src=0)
        eval_labels_2 = broadcast_object(eval_labels_2, src=0)

        # Update reserved inputs
        if eval_inputs_1:
            reserved_inputs.update(eval_inputs_1)
        if eval_inputs_2:
            reserved_inputs.update(eval_inputs_2)
        dataset.reserved_inputs = reserved_inputs

        rank_print("[BASELINE] Running evaluation...")

        # For FSDP, wrap the evaluation in a try-catch to handle potential issues
        try:
            base_tf_1 = run_eval_teacher_forced_parity(
                trainer.model,
                tokenizer, args.task, eval_inputs_1, eval_labels_1,
                num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
                print_mistakes=args.print_eval_examples, gc_every=50, **task_kwargs
            )
            base_tf_2 = run_eval_teacher_forced_parity(
                trainer.model,
                tokenizer, args.task, eval_inputs_2, eval_labels_2,
                num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
                print_mistakes=args.print_eval_examples, gc_every=50, **task_kwargs
            )

            rank_print(
                f"\n[BASELINE-TF-α1.0] α={1.0:.2f} | First={base_tf_1['first_token_acc']:.2%} | Full={base_tf_1['full_word_acc']:.2%} | N={base_tf_1['total']}")
            rank_print(
                f"[BASELINE-TF-αbase] α={args.base_alpha:.2f} | First={base_tf_2['first_token_acc']:.2%} | Full={base_tf_2['full_word_acc']:.2%} | N={base_tf_2['total']}\n")

        except RuntimeError as e:
            if "size mismatch" in str(e) and args.fsdp_enable:
                rank_print("[BASELINE][WARNING] FSDP evaluation failed - model needs proper FSDP context")
                rank_print("[BASELINE] Skipping baseline eval for FSDP (will work after training starts)")
            else:
                raise

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Put model back in train mode
        trainer.model.train()
    else:
        eval_inputs_1, eval_labels_1 = None, None
        eval_inputs_2, eval_labels_2 = None, None

    # Save run metadata
    if is_main_process():
        try:
            with open(os.path.join(args.output_dir, "run_meta.json"), "w") as f:
                meta = {
                    "job_id": args.job_id,
                    "scratch_dir": args.scratch_dir,
                    "created_at": datetime.datetime.now().isoformat(),
                    "resume_from": resume_ckpt,
                    "cli": " ".join(sys.argv),
                    "world_size": get_world_size(),
                }
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"[META][WARN] Could not write run_meta.json: {e}")

    # Save initial curriculum state
    _save_curriculum_state(args.output_dir, dataset.stage, curriculum.stage_start_step)

    # ----- TRAINING STARTS HERE -----
    rank_print("\n[TRAIN] Starting training...\n")

    # Mark curriculum warmup if resuming
    if resume_ckpt:
        match = re.search(r'checkpoint-(\d+)', resume_ckpt)
        if match:
            resume_step = int(match.group(1))
            curriculum.last_resume_step = resume_step
            rank_print(f"[CURRICULUM] Resuming from step {resume_step}")
            rank_print(f"[CURRICULUM] Will skip checks for {curriculum.resume_cooldown_steps} steps (warmup period)")

    # Start training
    if args.oom_autoscale:
        train_with_oom_autoscale(
            trainer,
            init_bs=trainer.args.per_device_train_batch_size,
            init_gas=trainer.args.gradient_accumulation_steps,
            resume_from_checkpoint=resume_ckpt,
        )
    else:
        trainer.train(resume_from_checkpoint=resume_ckpt)

    rank_print("\n[TRAIN] Training complete.\n")

    if args.skip_final_eval:
        return 0

    # Final eval 1: seen training samples for sanity check (should be near 100%)
    if args.eval_seen_train and args.store_seen_train:
        # Step 1: Each rank gets its own seen samples
        local_inputs, local_labels = dataset.get_seen_samples()

        # Step 2: Gather all samples to rank 0 (ALL ranks participate in each broadcast)
        all_inputs_list = []
        all_labels_list = []

        for rank_id in range(get_world_size()):
            if rank_id == get_rank():
                # This rank sends its data
                broadcast_object(local_inputs, src=rank_id)
                broadcast_object(local_labels, src=rank_id)
                if is_main_process():
                    all_inputs_list.append(local_inputs)
                    all_labels_list.append(local_labels)
            else:
                # All other ranks receive (even if they don't store it)
                rank_inputs = broadcast_object(None, src=rank_id)
                rank_labels = broadcast_object(None, src=rank_id)
                if is_main_process():
                    all_inputs_list.append(rank_inputs)
                    all_labels_list.append(rank_labels)

        # Step 3: Deduplicate on rank 0
        if is_main_process():
            seen_set = set()
            seen_inputs = []
            seen_labels = []

            for inputs, labels in zip(all_inputs_list, all_labels_list):
                for inp, lab in zip(inputs, labels):
                    if inp not in seen_set:
                        seen_inputs.append(inp)
                        seen_labels.append(lab)
                        seen_set.add(inp)

            rank_print(f"[SEEN-TF] Gathered {len(seen_inputs)} unique samples from {get_world_size()} ranks")
        else:
            seen_inputs = None
            seen_labels = None

        # Step 4: Broadcast the combined deduplicated set to all ranks
        barrier()
        seen_inputs = broadcast_object(seen_inputs, src=0)
        seen_labels = broadcast_object(seen_labels, src=0)

        if seen_inputs:
            rank_print(f"[SEEN-TF] Evaluating on {len(seen_inputs)} seen samples from all ranks")
            trainer.model.eval()
            seen_tf = run_eval_teacher_forced_parity(
                trainer.model,
                tokenizer, args.task, seen_inputs, seen_labels,
                num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
                print_mistakes=min(5, args.print_eval_examples), gc_every=50, **task_kwargs
            )
            rank_print(
                f"[SEEN-TF] First={seen_tf['first_token_acc']:.2%} | Full={seen_tf['full_word_acc']:.2%} | N={seen_tf['total']}")

        barrier()

    # Final eval 2: eval at final curriculum alpha, give teacher forced acc and greedy acc
    trainer.model.eval()

    # Reuse eval sets from baseline (or generate if baseline was skipped)
    if eval_inputs_1 is None:
        rank_print(f"[FINAL EVAL] Generating eval data...")

        if is_main_process():
            eval_inputs_1, eval_labels_1, _ = generate_eval_like_training(
                n_samples=args.eval_samples,
                task=args.task,
                tokenizer=tokenizer,
                max_input_size=args.max_input_size,
                alpha=1.0,
                num_shots=args.num_shots,
                reserved_inputs=reserved_inputs,
                seed=(args.seed or 0) + 42,  # Same seed as baseline
                **task_kwargs,
            )

            eval_inputs_2, eval_labels_2, _ = generate_eval_like_training(
                n_samples=args.eval_samples,
                task=args.task,
                tokenizer=tokenizer,
                max_input_size=args.max_input_size,
                alpha=args.base_alpha,
                num_shots=args.num_shots,
                reserved_inputs=set(eval_inputs_1),
                seed=(args.seed or 0) + 99,  # Same seed as baseline
                **task_kwargs,
            )
        else:
            eval_inputs_1, eval_labels_1 = None, None
            eval_inputs_2, eval_labels_2 = None, None

        barrier()
        eval_inputs_1 = broadcast_object(eval_inputs_1, src=0)
        eval_labels_1 = broadcast_object(eval_labels_1, src=0)
        eval_inputs_2 = broadcast_object(eval_inputs_2, src=0)
        eval_labels_2 = broadcast_object(eval_labels_2, src=0)
    else:
        rank_print(f"[FINAL EVAL] Reusing eval data from baseline")

    # Use eval_inputs_1 for main eval (alpha=1.0 is hardest)
    final_tf = run_eval_teacher_forced_parity(
        trainer.model, tokenizer, args.task, eval_inputs_1, eval_labels_1,
        num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
        print_mistakes=args.print_eval_examples, gc_every=50, **task_kwargs
    )
    rank_print(
        f"[FINAL-TF-α1.0] α={1.0:.2f} | First={final_tf['first_token_acc']:.2%} | Full={final_tf['full_word_acc']:.2%} | N={final_tf['total']}")

    greedy_read = run_eval_greedy_readable(
        trainer.model, tokenizer, args.task,
        eval_inputs_1, eval_labels_1,
        num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
        print_examples=min(3, args.print_eval_examples), **task_kwargs
    )
    rank_print(
        f"[FINAL-GREEDY-α1.0] α={1.0:.2f} | First={greedy_read['first_token_acc']:.2%} | Full={greedy_read['full_word_acc']:.2%} | N={greedy_read['total']}")

    # Optionally evaluate on base alpha set too
    if eval_inputs_2 is not None:  # Only if we have it
        base_final_tf = run_eval_teacher_forced_parity(
            trainer.model, tokenizer, args.task, eval_inputs_2, eval_labels_2,
            num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
            print_mistakes=0, gc_every=50, **task_kwargs
        )
        rank_print(
            f"[FINAL-TF-αbase] α={args.base_alpha:.2f} | First={base_final_tf['first_token_acc']:.2%} | Full={base_final_tf['full_word_acc']:.2%} | N={base_final_tf['total']}")

    # Final eval 3: eval with redacted data for sanity check (should be low acc)
    red_n = args.eval_redacted_samples
    if red_n is None or red_n <= 0:
        red_n = args.eval_samples
    red_inputs, red_labels = build_redacted_eval_set(
        eval_inputs_1, eval_labels_1,
        token=args.redaction_token,
        max_n=red_n
    )
    if red_inputs:
        red_tf = run_eval_teacher_forced_parity(
            trainer.model,
            tokenizer, args.task, red_inputs, red_labels,
            num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
            print_mistakes=min(2, args.print_eval_examples), gc_every=50, **task_kwargs
        )
        rank_print(
            f"[FINAL-TF-REDACTED] First={red_tf['first_token_acc']:.2%} | Full={red_tf['full_word_acc']:.2%} | N={red_tf['total']}")

        red_greedy = run_eval_greedy_readable(
            trainer.model,
            tokenizer, args.task,
            red_inputs, red_labels,
            num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
            print_examples=min(3, args.print_eval_examples), **task_kwargs
        )
        rank_print(
            f"[FINAL-GREEDY-REDACTED] First={red_greedy['first_token_acc']:.2%} | Full={red_greedy['full_word_acc']:.2%} | N={red_greedy['total']}")
    else:
        rank_print("[FINAL-REDACTED] Skipped (could not redact any eval items cleanly)")

    # Clean up
    del eval_inputs_1, eval_labels_1
    if eval_inputs_2 is not None:
        del eval_inputs_2, eval_labels_2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save artifacts
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "final_tf_metrics.json"), "w") as f:
            json.dump(final_tf, f, indent=2)

    # Save model and tokenizer (all ranks participate for FSDP)
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

    rank_print("\n[DONE] Training/evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())