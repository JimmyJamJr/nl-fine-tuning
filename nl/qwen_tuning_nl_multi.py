import os
import re
import gc
import sys
import json
import random
import warnings
import datetime
import math
import time
from collections import deque
from typing import List, Tuple, Dict, Any, Optional, Set
from contextlib import nullcontext
import multiprocessing

import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging as hf_logging
from transformers import GenerationConfig

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


def _save_run_config(dirpath: str, bs: int, gas: int, port: int = None) -> None:
    """Atomic save of run config. Includes PORT to enable hopping."""
    try:
        os.makedirs(dirpath, exist_ok=True)
        pid = os.getpid()
        tmp_path = os.path.join(dirpath, f"run_config.json.tmp.{pid}")
        final_path = os.path.join(dirpath, "run_config.json")

        # If port isn't provided, try to grab current
        if port is None:
            port = int(os.environ.get("MASTER_PORT", 29500))

        data = {
            "batch_size": int(bs),
            "grad_acc": int(gas),
            "master_port": int(port)  # Save the port so next run can increment it
        }

        with open(tmp_path, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())

        os.rename(tmp_path, final_path)  # Atomic move
        print(f"[RANK {get_rank()}] Saved config to {final_path}: {data}")
        sys.stdout.flush()
    except Exception as e:
        print(f"[CONFIG][WARN] Failed to save run config: {e}")
        sys.stdout.flush()


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
            base_alpha: float = 0.1,
            max_alpha: float = 1.0,
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
        # self.stage = stage
        self._stage = multiprocessing.Value('i', stage)
        self.n_stages = n_stages
        self.base_alpha = base_alpha
        self.max_alpha = max_alpha
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

        # Epoch size - number of samples per "epoch"
        self.epoch_size = 50000

        rank_print(
            f"[DATASET] Map-style dataset | epoch_size={self.epoch_size} | store_seen={self._store} cap={self._store_cap}")

    @property
    def stage(self):
        return self._stage.value

    @stage.setter
    def stage(self, value):
        self._stage.value = value

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
            return self.max_alpha
        return self.base_alpha + (self.max_alpha - self.base_alpha) * (self.stage - 1) / max(self.n_stages - 1, 1)

    def _shots_prefix(self) -> str:
        """Build few-shot prefix for prompts"""
        if not self.few_shot_examples:
            return ""
        parts = []
        for ex in self.few_shot_examples:
            tt = _determine_task_type(self.task, ex["input"])
            parts.append(f"{ex['input']} {ex['output']}{_get_end_tokens(tt)}")
        return "\n\n".join(parts) + ("\n\n" if parts else "")

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

        base_seed = (self.seed or 0) + rank * 9973 + worker_id * 997 + idx

        # Try to generate a valid sample (up to 100 attempts)
        max_attempts = 500
        for attempt in range(max_attempts):
            # Different seed each attempt
            sample_seed = base_seed + attempt * 104729  # Large prime
            rng = random.Random(sample_seed)
            gen = NaturalLanguageGraphGenerator(self.max_input_size, seed=sample_seed)

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

def make_collate(tokenizer, pad_to_multiple_of=64):
    def collate(features):
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # # Dynamic sequence length (batch-specific)
        # max_len = max(len(f["input_ids"]) for f in features)

        # # Dynamic first_k size (batch-specific, minimum 16 for safety)
        # dyn_max_k = max(len(f["valid_first_targets"]) for f in features)
        # max_k = max(dyn_max_k, 16)  # Keep minimum of 16 as safety margin

        # input_ids, attn, labels, prompt_lens = [], [], [], []
        # vtargets, vmask = [], []

        # for f in features:
        #     L = len(f["input_ids"])
        #     pad = max_len - L
        #     input_ids.append(f["input_ids"] + [pad_id] * pad)
        #     attn.append(f["attention_mask"] + [0] * pad)
        #     labels.append(f["labels"] + [-100] * pad)
        #     prompt_lens.append(f["prompt_len"])

        #     K = len(f["valid_first_targets"])
        #     pad_k = max_k - K
        #     vtargets.append(f["valid_first_targets"] + [pad_id] * pad_k)
        #     vmask.append([1] * K + [0] * pad_k)

        raw_max_len = max(len(f["input_ids"]) for f in features)

        # Round up to the nearest multiple of 64 to stabilize shapes for torch.compile
        # e.g., 100 -> 128, 129 -> 192
        if pad_to_multiple_of is not None and pad_to_multiple_of > 0:
            remainder = raw_max_len % pad_to_multiple_of
            if remainder == 0:
                target_len = raw_max_len
            else:
                target_len = raw_max_len + (pad_to_multiple_of - remainder)
        else:
            target_len = raw_max_len

        # 2. Determine Max Candidates (Bucketing optional, but usually small enough to ignore)
        dyn_max_k = max(len(f["valid_first_targets"]) for f in features)
        max_k = max(dyn_max_k, 16)  # Minimum 16 safety

        input_ids, attn, labels, prompt_lens = [], [], [], []
        vtargets, vmask = [], []

        for f in features:
            # Pad Input Sequences to target_len
            curr_len = len(f["input_ids"])
            pad = target_len - curr_len

            input_ids.append(f["input_ids"] + [pad_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
            prompt_lens.append(f["prompt_len"])

            # Pad Candidate Targets to max_k
            k_len = len(f["valid_first_targets"])
            pad_k = max_k - k_len

            vtargets.append(f["valid_first_targets"] + [pad_id] * pad_k)
            vmask.append([1] * k_len + [0] * pad_k)

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
    def __init__(self, *args, first_token_soft_weight=0.3, use_chunked_ce=False, ce_chunk_size=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_token_soft_weight = first_token_soft_weight
        self.recent_losses = deque(maxlen=100)
        self.first_token_correct = deque(maxlen=200)
        self.full_word_correct = deque(maxlen=200)

        self.use_chunked_ce = use_chunked_ce
        self.ce_chunk_size = ce_chunk_size

        if self.use_chunked_ce:
            print(f"[TRAINER] Using chunked cross-entropy (chunk_size={ce_chunk_size})")

    def _get_lm_head_weight(self, model):
        """Extract lm_head weight from potentially wrapped model."""
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        if hasattr(unwrapped, "base_model"):  # LoRA
            unwrapped = unwrapped.base_model
        if hasattr(unwrapped, "model"):
            unwrapped = unwrapped.model
        return unwrapped.lm_head.weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompt_len = inputs.pop("prompt_len")
        valid_first = inputs.pop("valid_first_targets")
        valid_first_mask = inputs.pop("valid_first_mask")
        labels = inputs["labels"]

        if self.use_chunked_ce:
            return self._compute_loss_chunked(
                model, inputs, labels, prompt_len,
                valid_first, valid_first_mask, return_outputs
            )
        else:
            return self._compute_loss_standard(
                model, inputs, labels, prompt_len,
                valid_first, valid_first_mask, return_outputs
            )

    def _compute_loss_standard(self, model, inputs, labels, prompt_len,
                               valid_first, valid_first_mask, return_outputs):
        """Original implementation using full logits tensor."""
        outputs = model(**inputs)
        logits = outputs.logits

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

        # Training-time tracking
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

    def _compute_loss_chunked(self, model, inputs, labels, prompt_len,
                              valid_first, valid_first_mask, return_outputs):
        """
        Memory-efficient loss using chunked cross-entropy.
        Bypasses Accelerate's fp32 conversion by accessing transformer directly.
        """
        # Unwrap model to get the actual transformer (avoids Accelerate fp32 conversion)
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        if hasattr(unwrapped, "base_model"):  # LoRA
            unwrapped = unwrapped.base_model
        if hasattr(unwrapped, "model"):
            unwrapped = unwrapped.model

        # Get the inner transformer (e.g., Qwen2Model) and lm_head separately
        transformer = unwrapped.model  # The transformer layers
        lm_head_weight = unwrapped.lm_head.weight

        # Forward through transformer only - returns last_hidden_state directly
        # This avoids storing all 36 layers of hidden states
        transformer_outputs = transformer(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        hidden_states = transformer_outputs.last_hidden_state  # [B, T, H] - only last layer!

        # Match dtype with lm_head
        hidden_states = hidden_states.to(lm_head_weight.dtype)

        # Shift for autoregressive loss
        shift_hidden = hidden_states[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        B, Tm1, H = shift_hidden.shape
        V = lm_head_weight.shape[0]
        valid_mask = (shift_labels != -100)

        # === CHUNKED MAIN LOSS ===
        flat_hidden = shift_hidden.view(-1, H)
        flat_labels = shift_labels.view(-1)
        total_tokens = flat_hidden.shape[0]

        total_loss = flat_hidden.new_zeros(())
        total_valid = 0

        for start in range(0, total_tokens, self.ce_chunk_size):
            end = min(start + self.ce_chunk_size, total_tokens)

            chunk_hidden = flat_hidden[start:end]
            chunk_labels = flat_labels[start:end]
            chunk_logits = F.linear(chunk_hidden, lm_head_weight)

            chunk_valid = (chunk_labels != -100)
            n_valid = chunk_valid.sum().item()

            if n_valid > 0:
                chunk_loss = F.cross_entropy(
                    chunk_logits[chunk_valid],
                    chunk_labels[chunk_valid],
                    reduction='sum'
                )
                total_loss = total_loss + chunk_loss
                total_valid += n_valid

        main_loss = total_loss / max(total_valid, 1)

        # === SOFT-CE ADJUSTMENT AT FIRST POSITION ===
        total_adjustment = hidden_states.new_zeros(())

        for i in range(B):
            first_idx = prompt_len[i].item() - 1
            if first_idx < 0 or first_idx >= Tm1:
                continue
            if not valid_mask[i, first_idx]:
                continue

            mask = valid_first_mask[i]
            if not mask.any():
                continue

            ids = valid_first[i][mask]
            first_hidden = shift_hidden[i, first_idx, :]
            first_logits = F.linear(first_hidden, lm_head_weight)

            gold_label = shift_labels[i, first_idx]
            hard_ce = F.cross_entropy(
                first_logits.unsqueeze(0),
                gold_label.unsqueeze(0)
            )

            logp = F.log_softmax(first_logits, dim=-1)
            soft = torch.zeros_like(logp)
            soft[ids] = 1.0 / ids.numel()
            soft_ce = -(soft * logp).sum()

            delta = self.first_token_soft_weight * (soft_ce - hard_ce)
            total_adjustment = total_adjustment + delta

        loss = main_loss + total_adjustment / max(total_valid, 1)

        # === TRACKING ===
        with torch.no_grad():
            # Batch first-token accuracy
            first_indices = (prompt_len - 1).clamp(0, Tm1 - 1)
            first_valid = (first_indices >= 0) & (first_indices < Tm1)
            for i in range(B):
                if first_valid[i] and valid_mask[i, first_indices[i]]:
                    first_logits = F.linear(shift_hidden[i, first_indices[i]], lm_head_weight)
                    pred_first = torch.argmax(first_logits).item()
                    valid_ids = set(valid_first[i][valid_first_mask[i]].tolist())
                    self.first_token_correct.append(pred_first in valid_ids)

            # Full word accuracy
            for i in range(B):
                idxs = torch.nonzero(valid_mask[i], as_tuple=False).squeeze(-1)
                if idxs.numel() > 0:
                    if idxs.dim() == 0:
                        idxs = idxs.unsqueeze(0)
                    valid_hidden = shift_hidden[i, idxs]
                    valid_logits = F.linear(valid_hidden, lm_head_weight)
                    preds = torch.argmax(valid_logits, dim=-1)
                    golds = shift_labels[i, idxs]
                    ok = (preds == golds).all().item()
                else:
                    ok = True
                self.full_word_correct.append(ok)

        self.recent_losses.append(loss.item())

        # Create a mock outputs object for return_outputs
        if return_outputs:
            class MockOutputs:
                def __init__(self, loss):
                    self.loss = loss

            return (loss, MockOutputs(loss))

        return loss

    def get_first_token_acc(self):
        return (sum(self.first_token_correct) / len(self.first_token_correct)) if self.first_token_correct else 0.0

    def get_full_word_acc(self):
        return (sum(self.full_word_correct) / len(self.full_word_correct)) if self.full_word_correct else 0.0


# ================== Eval helpers ==================
@torch.no_grad()
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
    barrier()
    rank = get_rank()
    world_size = get_world_size()
    device = next(model.parameters()).device

    # --- CONSTANTS ---
    # CRITICAL for FSDP: Every rank must run exactly the same number of model steps.
    # We cap the number of candidates we check per sample.
    # If a sample has fewer candidates, we run dummy passes to fill the quota.
    MAX_CANDIDATES_PER_SAMPLE = 5

    # 1. Data Sharding & Padding (Sample Level)
    total_len = len(eval_inputs)
    chunk_size = math.ceil(total_len / world_size)

    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, total_len)

    my_inputs = eval_inputs[start_idx:end_idx]
    my_labels = eval_labels[start_idx:end_idx]

    actual_count = len(my_inputs)

    # Pad samples to ensure every rank processes the same number of "rows"
    pad_needed = chunk_size - actual_count
    if pad_needed > 0:
        dummy_in = my_inputs[-1] if my_inputs else ""
        dummy_la = my_labels[-1] if my_labels else []
        my_inputs.extend([dummy_in] * pad_needed)
        my_labels.extend([dummy_la] * pad_needed)

    rank_print(f"[EVAL-TF] Starting eval on {total_len} samples (Local: {actual_count}, Pad: {pad_needed})")

    # 2. Setup Few-Shot
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

    # 3. Inference Loop
    local_first_ok = 0
    local_full_ok = 0
    local_total = 0
    mistakes_printed = 0
    allow_print = is_main_process() and (print_mistakes > 0)

    for idx, (x, ys) in enumerate(zip(my_inputs, my_labels)):
        is_padding_sample = (idx >= actual_count)

        ys_list = ys if isinstance(ys, list) else [ys]
        prompt = (shots_prefix() + x) if few_shots else x
        enc = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        prompt_len = enc["input_ids"].shape[1]
        ttype = _determine_task_type(task, x)
        end_ids = tokenizer(_get_end_tokens(ttype), add_special_tokens=False)["input_ids"]

        # --- First Token Check (1 Forward Pass) ---
        # This is safe because it runs exactly once per sample on all ranks.
        out_prompt = model(**enc)  # <--- Forward Pass 1 (Sync Point)

        if is_padding_sample:
            # We still had to run the forward pass above to keep FSDP sync,
            # but we stop processing this sample now.
            # However, we MUST still run the "Candidate Loop" below as dummies
            # so we stay in sync with ranks that are processing real samples.
            pass
        else:
            # Calculate First Token Acc
            union = set()
            candidates = []
            for gold in ys_list:
                ids = _tokenize_leading_space(tokenizer, gold)
                if ids:
                    union.add(ids[0])
                    candidates.append(ids + end_ids)

            if not candidates:
                local_total += 1
                # We still fall through to the loop to run dummy passes
            else:
                fidx = prompt_len - 1
                pred_first_id = torch.argmax(out_prompt.logits[0, fidx, :]).item()
                local_first_ok += int(pred_first_id in union)

        # --- Candidate Loop (Variable number of Forward Passes) ---
        # CRITICAL FIX: Pad this loop so every rank runs exactly MAX_CANDIDATES steps.

        # 1. Get real candidates (if any, and not padding sample)
        real_candidates = []
        if not is_padding_sample and 'candidates' in locals():
            real_candidates = candidates[:MAX_CANDIDATES_PER_SAMPLE]

        matched = False
        pred_word = None

        # 2. Run fixed number of steps
        for i in range(MAX_CANDIDATES_PER_SAMPLE):
            if i < len(real_candidates):
                # === REAL CANDIDATE CHECK ===
                cand = real_candidates[i]
                full = torch.tensor(
                    enc["input_ids"].tolist()[0] + cand,
                    dtype=torch.long, device=device
                ).unsqueeze(0)
                attn = torch.ones_like(full)

                out = model(input_ids=full, attention_mask=attn)  # <--- Forward Pass (Sync Point)

                # Check match
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
            else:
                # === DUMMY PASS (Prevent Deadlock) ===
                # Run the model on the prompt again just to participate in FSDP comms
                _ = model(**enc)  # <--- Forward Pass (Sync Point)

        # --- Metrics Update ---
        if not is_padding_sample:
            if pred_word is None and 'pred_first_id' in locals():
                pred_word = tokenizer.decode([pred_first_id], skip_special_tokens=True).strip()

            if not matched and allow_print and mistakes_printed < print_mistakes:
                print("\n" + "=" * 60)
                print(f"[TF MISTAKE #{mistakes_printed + 1}] (Rank {rank})")
                print("Prompt fed to model:\n" + prompt)
                print("Expected any of:", ys_list)
                print(f"Predicted first-token id: {locals().get('pred_first_id', '?')} (word guess: '{pred_word}')")
                mistakes_printed += 1

            local_full_ok += int(matched)
            if 'candidates' in locals() and candidates:
                # Only increment total if valid candidates existed
                # (If candidates was empty, we already incremented total above)
                pass
            if not ('candidates' in locals() and not candidates):
                local_total += 1

        if gc_every > 0 and (idx % gc_every == 0):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 4. Global Aggregation
    metrics = torch.tensor([local_first_ok, local_full_ok, local_total], dtype=torch.long, device=device)
    if dist_is_initialized():
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)

    global_first = metrics[0].item()
    global_full = metrics[1].item()
    global_total = metrics[2].item()

    barrier()
    rank_print(f"[EVAL-TF] Completed. Global samples: {global_total}")

    return {
        "first_token_acc": (global_first / global_total) if global_total else 0.0,
        "full_word_acc": (global_full / global_total) if global_total else 0.0,
        "first_token_hits": global_first,
        "full_word_hits": global_full,
        "total": global_total,
    }


@torch.no_grad()
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
    rank = get_rank()
    world_size = get_world_size()
    device = next(model.parameters()).device

    # 1. Data Sharding & Padding
    total_len = len(inputs)
    chunk_size = math.ceil(total_len / world_size)
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, total_len)

    my_inputs = inputs[start_idx:end_idx]
    my_labels = labels[start_idx:end_idx]
    actual_count = len(my_inputs)

    pad_needed = chunk_size - actual_count
    if pad_needed > 0:
        dummy_in = my_inputs[-1] if my_inputs else ""
        dummy_la = my_labels[-1] if my_labels else []
        my_inputs.extend([dummy_in] * pad_needed)
        my_labels.extend([dummy_la] * pad_needed)

    rank_print(f"[EVAL-GREEDY] Starting generation on {total_len} samples (Local: {actual_count}, Pad: {pad_needed})")

    local_correct_first = 0
    local_correct_full = 0
    local_total = 0
    local_printed = 0

    allow_print = is_main_process() and (print_examples > 0)

    # Ensure pad token exists for generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    greedy_config = GenerationConfig(
        max_new_tokens=24,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True
    )

    is_fsdp = isinstance(model, FSDP)

    # 2. Inference Loop
    for i, (x, ys) in enumerate(zip(my_inputs, my_labels)):
        is_padding = (i >= actual_count)
        ys = ys if isinstance(ys, list) else [ys]

        # Tokenize
        enc = tokenizer(x, return_tensors="pt").to(device)
        prompt_len = enc.input_ids.shape[1]

        # Generate (Must run for everyone)
        if is_fsdp:
            with FSDP.summon_full_params(model, writeback=False, rank0_only=False):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    gen_out = model.generate(
                        **enc,
                        max_new_tokens=24,
                        generation_config=greedy_config,
                        synced_gpus=True
                    )
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                unwrapped = model.module if hasattr(model, "module") else model
                gen_out = unwrapped.generate(
                    **enc,
                    max_new_tokens=24,
                    generation_config=greedy_config,
                    synced_gpus=True
                )

        if is_padding:
            continue

        # Extract
        gen_ids = gen_out[0][prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        pred_word = re.split(r"[.,\s]+", gen_text)[0] if gen_text else ""

        # Metrics
        valid_first_ids = {
            _tokenize_leading_space(tokenizer, y)[0]
            for y in ys
            if _tokenize_leading_space(tokenizer, y)
        }
        first_ok = (len(gen_ids) > 0 and gen_ids[0].item() in valid_first_ids)
        local_correct_first += int(first_ok)

        full_ok = any(pred_word.lower() == y.lower() for y in ys)
        local_correct_full += int(full_ok)
        local_total += 1

        # Print (Rank 0 only)
        if allow_print and (not first_ok or not full_ok) and local_printed < print_examples:
            print("\n" + "=" * 60)
            print(f"[GREEDY MISTAKE] (Rank {rank})")
            print(f"Prompt (tail): ...{x[-100:]}")
            print(f"Gold Options: {ys}")
            print(f"Predicted Word: '{pred_word}'")
            print(f"Full Generation: '{gen_text}'")
            local_printed += 1

    # 3. Aggregation
    metrics = torch.tensor([local_correct_first, local_correct_full, local_total], dtype=torch.long, device=device)
    if dist_is_initialized():
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)

    global_first = metrics[0].item()
    global_full = metrics[1].item()
    global_total = metrics[2].item()

    barrier()
    rank_print(f"[EVAL-GREEDY] Completed. Global samples: {global_total}")

    return {
        "first_token_acc": (global_first / global_total) if global_total else 0.0,
        "full_word_acc": (global_full / global_total) if global_total else 0.0,
        "first_token_hits": global_first,
        "full_word_hits": global_full,
        "total": global_total,
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
            do_stage_eval: bool = False,
            eval_inputs_hard: List[str] = None,
            eval_labels_hard: List[List[str]] = None,
            eval_fingerprint_hard: str = None,
            tokenizer=None,
            task: str = None,
            task_kwargs: dict = None,
            num_shots: int = 0,
            max_input_size: int = 256,
            seed: int = None,
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

        # Track stage eval results
        self.stage_eval_history = []
        self.do_stage_eval = do_stage_eval
        self.eval_inputs_hard = eval_inputs_hard
        self.eval_labels_hard = eval_labels_hard
        self.eval_fingerprint_hard = eval_fingerprint_hard
        self.tokenizer = tokenizer
        self.task = task
        self.task_kwargs = task_kwargs or {}
        self.num_shots = num_shots
        self.max_input_size = max_input_size
        self.seed = seed

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
        # if self.trainer is None:
        #     return 0.0
        # return self.trainer.get_full_word_acc()

        # Get Local Accuracy
        local_acc = self.trainer.get_full_word_acc()

        # Synchronize across all GPUs (Average the accuracy)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = self.trainer.args.device
            # Move scalar to tensor on device
            metric_tensor = torch.tensor([local_acc], dtype=torch.float32, device=device)
            # Sum across all ranks
            torch.distributed.all_reduce(metric_tensor, op=torch.distributed.ReduceOp.SUM)
            # Divide by world size to get average
            global_acc = metric_tensor.item() / torch.distributed.get_world_size()
            return global_acc

        return local_acc

    def on_save(self, args, state, control, **kwargs):
        if is_main_process():
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            _save_curriculum_state(checkpoint_dir, self.dataset.stage, self.stage_start_step)
        return control

    def _run_stage_eval(self, stage: int, global_step: int):
        """Run TF + greedy eval at α=1.0 after stage advancement."""
        if not self.do_stage_eval:
            return
        if self.eval_inputs_hard is None or self.tokenizer is None:
            rank_print("[STAGE-EVAL] Skipped (missing eval data or tokenizer)")
            return

        # Verify fingerprint
        fp = _eval_data_fingerprint(self.eval_inputs_hard, self.eval_labels_hard)
        if fp != self.eval_fingerprint_hard:
            rank_print(f"[STAGE-EVAL] WARNING: Fingerprint mismatch! {fp} != {self.eval_fingerprint_hard}")

        rank_print(f"\n[STAGE-EVAL] Stage {stage} complete (step {global_step}), evaluating at α=1.0...")

        # Switch to eval mode
        model = self.trainer.model
        was_training = model.training
        model.eval()

        with torch.no_grad():
            # Teacher-forced eval
            tf_result = run_eval_teacher_forced_parity(
                model, self.tokenizer, self.task,
                self.eval_inputs_hard, self.eval_labels_hard,
                num_shots=self.num_shots, max_input_size=self.max_input_size,
                seed=self.seed, print_mistakes=0, gc_every=50,
                **self.task_kwargs
            )

            # Greedy eval
            greedy_result = run_eval_greedy_readable(
                model, self.tokenizer, self.task,
                self.eval_inputs_hard, self.eval_labels_hard,
                num_shots=self.num_shots, max_input_size=self.max_input_size,
                seed=self.seed, print_examples=0,
                **self.task_kwargs
            )

        # Restore training mode
        if was_training:
            model.train()

        # Log results
        rank_print(
            f"[STAGE-EVAL] Stage {stage} | Step {global_step} | "
            f"TF: First={tf_result['first_token_acc']:.2%}, Full={tf_result['full_word_acc']:.2%} | "
            f"Greedy: First={greedy_result['first_token_acc']:.2%}, Full={greedy_result['full_word_acc']:.2%}"
        )

        # Store history
        self.stage_eval_history.append({
            "stage": stage,
            "step": global_step,
            "alpha_training": self.dataset._stage_alpha(),
            "tf_first": tf_result['first_token_acc'],
            "tf_full": tf_result['full_word_acc'],
            "greedy_first": greedy_result['first_token_acc'],
            "greedy_full": greedy_result['full_word_acc'],
        })

        # Save to file
        if is_main_process():
            try:
                out_path = os.path.join(self.trainer.args.output_dir, "stage_eval_history.json")
                with open(out_path, "w") as f:
                    json.dump(self.stage_eval_history, f, indent=2)
            except Exception as e:
                rank_print(f"[STAGE-EVAL] Warning: Could not save history: {e}")

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        step = state.global_step

        # Skip checks during cooldown after resume
        if self.last_resume_step > 0:
            steps_since_resume = step - self.last_resume_step
            if steps_since_resume < self.resume_cooldown_steps:
                return control

        # Only check periodically
        if step % self.check_every != 0:
            return control

        # Don't advance past final stage
        if self.dataset.stage >= self.n_stages:
            return control

        # Check minimum steps
        steps_in_stage = step - self.stage_start_step
        if steps_in_stage < self.min_steps:
            return control

        # Check accuracy
        acc = self._current_metric()
        if acc >= self.acc_thr:
            old_stage = self.dataset.stage
            self.dataset.stage = old_stage + 1
            self.stage_start_step = step

            new_alpha = self.dataset._stage_alpha()
            rank_print(
                f"\n[CURRICULUM] Stage {old_stage} → {self.dataset.stage} "
                f"(α={new_alpha:.3f}) at step {step} (acc={acc:.2%})\n"
            )

            # Save curriculum state
            _save_curriculum_state(self.trainer.args.output_dir, self.dataset.stage, self.stage_start_step)

            # Run stage eval after advancement
            self._run_stage_eval(old_stage, step)

        return control


# ================== Baseline eval callback ==================
class BaselineEvalCallback(TrainerCallback):
    def __init__(self, baseline_runner, skip_baseline: bool):
        self.baseline_runner = baseline_runner
        self.skip_baseline = skip_baseline
        self._ran = False
        self.trainer = None  # will be set after Trainer is created

    def on_train_begin(self, args, state, control, **kwargs):
        if self._ran or self.skip_baseline:
            return control

        # Trainer isn't passed in kwargs; use stored reference.
        trainer = self.trainer
        if trainer is None:
            raise RuntimeError("BaselineEvalCallback.trainer was not set")

        # model in kwargs (if present) is already wrapped/prepared.
        model = kwargs.get("model", trainer.model_wrapped)

        trainer.accelerator.wait_for_everyone()
        model.eval()

        try:
            with torch.no_grad():
                self.baseline_runner(model, trainer)
            # mark as ran only if baseline succeeded
            self._ran = True
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            trainer.model.train()

        return control


# out-of-memory auto-scaling batch size and gradient accumulation training loop
def train_with_oom_autoscale(trainer, init_bs, init_gas, resume_from_checkpoint):
    # This function now acts as a wrapper that simply runs training,
    # but handles the OOM by saving state and CRASHING all ranks.

    # Save the initial config as "stable" when we start.
    if is_main_process():
        _save_run_config(
            trainer.args.output_dir,
            trainer.args.per_device_train_batch_size,
            trainer.args.gradient_accumulation_steps
        )

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
            print(f"[RANK {get_rank()}] OOM Detected! Preparing to restart...")

            # Calculate new parameters
            current_bs = trainer.args.per_device_train_batch_size
            current_gas = trainer.args.gradient_accumulation_steps

            new_bs = max(1, current_bs // 2)
            new_gas = current_gas * 2

            if new_bs == current_bs:
                print(f"[RANK {get_rank()}] Already at BS=1. Cannot reduce further. Exiting.")
                raise e

            # Write state for future resumes
            _save_run_config(trainer.args.output_dir, new_bs, new_gas)

            # Create a restart flag
            try:
                flag_path = os.path.join(trainer.args.output_dir, "RESTART_FLAG")
                with open(flag_path, "w") as f:
                    f.write(f"oom_{datetime.datetime.now().isoformat()}")
                print(f"[RANK {get_rank()}] Created restart flag at {flag_path}")
            except Exception as e:
                print(f"[RANK {get_rank()}] WARNING: Failed to create restart flag: {e}")

            # Exit with non-zero code to trigger torchrun restart
            print(f"[RANK {get_rank()}] Exiting process to trigger torchrun restart with new params.")
            sys.stdout.flush()
            os._exit(42)
        else:
            # Not an OOM, re-raise normal errors
            raise e


# Generate eval set in a similar way to the training data
@torch.no_grad()
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


def estimate_worst_case_length(args, tokenizer, safety_margin=1.25):
    """
    Generates 'hardest possible' samples to calculate a safe upper bound
    for token length. Returns a HARD LIMIT that the dataset must respect.
    """
    if is_main_process():
        print("[INIT] Estimating worst-case token length (Alpha=1.0)...")

    g = NaturalLanguageGraphGenerator(args.max_input_size, seed=42)

    # Generate a larger batch (50) to catch outliers
    batch = g.generate_batch(
        task=args.task,
        batch_size=100,
        alpha=1.0,
        **{
            "max_lookahead": args.max_lookahead,
            "max_frontier_size": args.max_frontier_size,
            "max_branch_size": args.max_branch_size,
            "requested_backtrack": args.requested_backtrack
        }
    )

    max_seen_len = 0
    for ex in batch:
        if not ex or not ex.output_texts: continue
        task_type = _determine_task_type(args.task, ex.input_text)

        # 1. Prompt
        prompt_ids = tokenizer(ex.input_text, add_special_tokens=True, truncation=False)["input_ids"]
        # 2. Answer (Max length one)
        longest_ans = max(ex.output_texts, key=len)
        ans_ids = _tokenize_leading_space(tokenizer, longest_ans)
        # 3. End tokens
        end_ids = tokenizer(_get_end_tokens(task_type), add_special_tokens=False)["input_ids"]
        total_len = len(prompt_ids) + len(ans_ids) + len(end_ids)
        max_seen_len = max(max_seen_len, total_len)

    # Apply Safety Margin
    safe_len = int(max_seen_len * safety_margin)

    # Clamp to model max
    model_max = getattr(tokenizer, "model_max_length", 100000)
    if model_max > 32768: model_max = 32768

    final_len = min(safe_len, model_max)

    if is_main_process():
        print(f"[INIT] Max observed: {max_seen_len}. Hard Limit set to: {final_len} (+25%)")

    return final_len


def _eval_data_fingerprint(inputs: List[str], labels: List[List[str]]) -> str:
    """Create a fingerprint to verify eval data identity."""
    if not inputs:
        return "empty"
    import hashlib
    content = f"{len(inputs)}|{inputs[0][:100]}|{inputs[-1][:100]}|{len(labels)}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    rank_print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())
    rank_print("PyTorch version:", torch.__version__)

    import argparse

    p = argparse.ArgumentParser()

    # Task/model
    p.add_argument("--task", type=str, choices=["search", "dfs", "si"], default="si")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./nl_output")

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
    p.add_argument("--base_alpha", type=float, default=0.1)
    p.add_argument("--max_alpha", type=float, default=1.0, help="Maximum alpha during training (eval always uses 1.0)")
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

    # Eval flags
    p.add_argument("--do_baseline", action="store_true", help="Run pre-training baseline eval")
    p.add_argument("--do_final_eval", action="store_true", help="Run post-training TF + greedy eval")
    p.add_argument("--do_redacted_eval", action="store_true", help="Run redacted sanity check (should be low)")
    p.add_argument("--do_seen_eval", action="store_true", help="Run seen-samples sanity check (should be ~100%)")
    p.add_argument("--do_stage_eval", action="store_true",
                   help="Run TF+greedy eval at α=1.0 after each stage advancement")

    # Redacted eval config
    p.add_argument("--eval_redacted_samples", type=int, default=None)
    p.add_argument("--redaction_token", type=str, default="_____")

    # Memory control
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--oom_autoscale", action="store_true")

    # Scratch / resume
    p.add_argument("--scratch_dir", type=str,
                   default=os.environ.get("SCRATCH") or os.path.join("/scratch", os.environ.get("USER", "user")))
    p.add_argument("--job_id", type=str, default=os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID"))
    p.add_argument("--resume_from_job", type=str, default=None)

    # Liger kernels
    p.add_argument("--use_liger", action="store_true", help="Use Liger kernel for memory-efficient training")

    # Chunked cross-entropy
    p.add_argument("--use_chunked_ce", action="store_true", help="Use chunked cross-entropy for memory efficiency")
    p.add_argument("--ce_chunk_size", type=int, default=1024, help="Chunk size for chunked cross-entropy")

    global args
    args = p.parse_args()

    # Check local config first to override args
    cfg_candidate = None
    if os.path.isfile(os.path.join(args.output_dir, "run_config.json")):
        cfg_candidate = os.path.join(args.output_dir, "run_config.json")
    elif args.resume_from_job:
        prev = os.path.join(args.scratch_dir, "nl_output", args.task, f"job_{args.resume_from_job}", "run_config.json")
        if os.path.isfile(prev): cfg_candidate = prev

    if cfg_candidate:
        try:
            with open(cfg_candidate) as f:
                rc = json.load(f)
                args.batch_size = int(rc["batch_size"])
                args.gradient_accumulation_steps = int(rc["grad_acc"])
                if is_main_process(): print(f"[CONFIG] Loaded override from {cfg_candidate}: BS={args.batch_size}")
        except:
            pass

    # Initialize distributed if needed
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=datetime.timedelta(minutes=30)
        )
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

    # --- CONFIGURATION OVERRIDE LOGIC (OOM / RESUME) ---
    cfg_candidate = None
    if os.path.isfile(os.path.join(args.output_dir, "run_config.json")):
        cfg_candidate = os.path.join(args.output_dir, "run_config.json")
    elif args.resume_from_job:
        prev_dir = os.path.join(args.scratch_dir, "nl_output", args.task, f"job_{args.resume_from_job}")
        if os.path.isfile(os.path.join(prev_dir, "run_config.json")):
            cfg_candidate = os.path.join(prev_dir, "run_config.json")

    if cfg_candidate:
        try:
            with open(cfg_candidate, "r") as f:
                rc = json.load(f)
                if is_main_process():
                    print(f"[RESUME-CFG] Found config at {cfg_candidate}")
                    print(
                        f"[RESUME-CFG] Overriding CLI args: BS {args.batch_size}->{rc['batch_size']}, GAS {args.gradient_accumulation_steps}->{rc['grad_acc']}")
                args.batch_size = int(rc["batch_size"])
                args.gradient_accumulation_steps = int(rc["grad_acc"])
        except Exception as e:
            if is_main_process():
                print(f"[RESUME-CFG][WARN] Failed to load config from {cfg_candidate}: {e}")

    # Resume from checkpoint logic
    resume_ckpt = None

    if os.path.isdir(args.output_dir):
        last_local_ckpt = get_last_checkpoint(args.output_dir)
        if last_local_ckpt:
            resume_ckpt = last_local_ckpt
            rank_print(f"[CKPT] Auto-resuming from local run: {resume_ckpt}")

    if resume_ckpt is None and args.resume_from_job:
        prev_dir = os.path.join(args.scratch_dir, "nl_output", args.task, f"job_{args.resume_from_job}")
        if os.path.isdir(prev_dir):
            resume_ckpt = get_last_checkpoint(prev_dir)
            rank_print(f"[CKPT] Resuming from previous job {args.resume_from_job}: {resume_ckpt}")
        else:
            rank_print(f"[CKPT][ERROR] No run dir found for job {args.resume_from_job}")
            rank_print(f"[CKPT]          Expected: {prev_dir}")
            sys.exit(1)

    if resume_ckpt is None:
        rank_print(f"[CKPT] Fresh start")

    # Tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "cache_dir": args.cache_dir,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": "flash_attention_2",
    }

    # Apply Liger fused ops (NOT cross entropy)
    if getattr(args, 'use_liger', False):
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3
            apply_liger_kernel_to_qwen3(
                rope=True,
                rms_norm=True,
                swiglu=True,
                cross_entropy=False,
                fused_linear_cross_entropy=False,
            )
            rank_print("[LIGER] Applied fused RoPE/RMSNorm/SwiGLU kernels")
        except ImportError:
            rank_print("[LIGER] liger-kernel not installed, continuing without")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # Disable tie word embeddings for FSDP/LoRA compatibility
    if args.fsdp_enable and args.use_lora:
        model.config.tie_word_embeddings = False

    if is_main_process():
        attn_impl = getattr(model.config, "_attn_implementation", "unknown")
        print(f"[ATTENTION] Using: {attn_impl}")

    # Gradient checkpointing
    if args.gradient_checkpointing:
        if hasattr(model, "config"):
            model.config.use_cache = False
            rank_print("[MEM] model.config.use_cache = False (training)")

    # LoRA
    if args.use_lora:
        rank_print("[INIT] Applying LoRA...")
        from peft import PeftModel

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if args.fsdp_enable:
            model = model.to(torch.bfloat16)
            rank_print("[FSDP] Cast model to bfloat16 for LoRA compatibility")

        if args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
                rank_print("[MEM] Enabled input grads for Gradient Checkpointing + LoRA")

    if is_main_process():
        model.print_trainable_parameters()

    # Find hard limit to avoid OOM
    _ = estimate_worst_case_length(args, tokenizer)

    # Reserved inputs for deduplication
    reserved_inputs: Set[str] = set()

    # Task kwargs
    task_kwargs = {}
    if args.task == "search":
        task_kwargs = {"max_lookahead": args.max_lookahead}
    elif args.task == "dfs":
        task_kwargs = {"requested_backtrack": args.requested_backtrack}
    elif args.task == "si":
        task_kwargs = {"max_frontier_size": args.max_frontier_size, "max_branch_size": args.max_branch_size}

    # ==================== GENERATE EVAL DATA ONCE ====================
    eval_inputs_hard, eval_labels_hard = None, None  # α=1.0 (hardest)
    eval_inputs_easy, eval_labels_easy = None, None  # α=base_alpha (easiest)
    eval_fingerprint_hard, eval_fingerprint_easy = None, None

    need_eval_data = args.do_baseline or args.do_final_eval or args.do_redacted_eval or args.do_stage_eval

    if need_eval_data:
        rank_print("[EVAL-DATA] Generating eval sets (once for all evals)...")

        if is_main_process():
            # Hard eval set: α=1.0
            eval_inputs_hard, eval_labels_hard, _ = generate_eval_like_training(
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

            # Easy eval set: α=base_alpha
            eval_inputs_easy, eval_labels_easy, _ = generate_eval_like_training(
                n_samples=args.eval_samples,
                task=args.task,
                tokenizer=tokenizer,
                max_input_size=args.max_input_size,
                alpha=args.base_alpha,
                num_shots=args.num_shots,
                reserved_inputs=set(eval_inputs_hard),  # Dedupe from hard set
                seed=(args.seed or 0) + 99,
                **task_kwargs,
            )

        barrier()
        eval_inputs_hard = broadcast_object(eval_inputs_hard, src=0)
        eval_labels_hard = broadcast_object(eval_labels_hard, src=0)
        eval_inputs_easy = broadcast_object(eval_inputs_easy, src=0)
        eval_labels_easy = broadcast_object(eval_labels_easy, src=0)

        # Create fingerprints for verification
        eval_fingerprint_hard = _eval_data_fingerprint(eval_inputs_hard, eval_labels_hard)
        eval_fingerprint_easy = _eval_data_fingerprint(eval_inputs_easy, eval_labels_easy)

        rank_print(f"[EVAL-DATA] Hard (α=1.0): n={len(eval_inputs_hard)}, fingerprint={eval_fingerprint_hard}")
        rank_print(
            f"[EVAL-DATA] Easy (α={args.base_alpha}): n={len(eval_inputs_easy)}, fingerprint={eval_fingerprint_easy}")

        # Reserve these inputs so training doesn't generate duplicates
        if eval_inputs_hard: reserved_inputs.update(eval_inputs_hard)
        if eval_inputs_easy: reserved_inputs.update(eval_inputs_easy)

    # ---------------- Training dataset ----------------
    dataset = SinglePathARDataset(
        task=args.task,
        tokenizer=tokenizer,
        stage=1,
        n_stages=args.n_stages,
        base_alpha=args.base_alpha,
        max_alpha=args.max_alpha,
        max_input_size=args.max_input_size,
        reserved_inputs=reserved_inputs,
        num_shots=args.num_shots,
        seed=args.seed,
        store_examples=args.do_seen_eval,
        store_cap=1000,
        **task_kwargs,
    )

    curriculum = FirstTokenCurriculum(
        dataset=dataset,
        n_stages=args.n_stages,
        accuracy_threshold=args.accuracy_threshold,
        min_steps_per_stage=args.min_steps_per_stage,
        check_every=args.check_every,
        # Stage eval config
        do_stage_eval=args.do_stage_eval,
        eval_inputs_hard=eval_inputs_hard,
        eval_labels_hard=eval_labels_hard,
        eval_fingerprint_hard=eval_fingerprint_hard,
        tokenizer=tokenizer,
        task=args.task,
        task_kwargs=task_kwargs,
        num_shots=args.num_shots,
        max_input_size=args.max_input_size,
        seed=args.seed,
    )

    if resume_ckpt:
        if _try_restore_curriculum_state(resume_ckpt, dataset, curriculum):
            rank_print(f"[CURRICULUM] Synced state with checkpoint: {resume_ckpt}")
            match = re.search(r'checkpoint-(\d+)', resume_ckpt)
            if match:
                curriculum.last_resume_step = int(match.group(1))

    # FSDP must use original parameters when LoRA is applied
    fsdp_use_orig = True if args.use_lora else False
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=1000,
        logging_steps=10,
        logging_first_step=False,
        report_to="none",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        save_safetensors=True,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        optim="adamw_torch_fused",
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        seed=args.seed if args.seed is not None else 42,
        fsdp=(['full_shard', 'auto_wrap'] if args.fsdp_enable else []),
        fsdp_config=(
            {
                "min_num_params": int(args.fsdp_min_num_params),
                "use_orig_params": fsdp_use_orig,
                "limit_all_gathers": True,
                "sync_module_states": True,
                "offload_to_cpu": False,
                "state_dict_type": "SHARDED_STATE_DICT",
                "auto_wrap_policy": "transformer_based_wrap",
                "backward_prefetch": "backward_pre",
                "forward_prefetch": True,
            } if args.fsdp_enable else None
        ),
        torch_compile=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
    )

    # Baseline runner uses pre-generated eval data
    def baseline_runner(eval_model, trainer):
        rank_print(f"[BASELINE] Model prepared (Backend: {type(eval_model).__name__})")

        # Verify we're using the same data
        fp_hard = _eval_data_fingerprint(eval_inputs_hard, eval_labels_hard)
        fp_easy = _eval_data_fingerprint(eval_inputs_easy, eval_labels_easy)
        rank_print(f"[BASELINE] Verifying eval data: hard={fp_hard}, easy={fp_easy}")
        assert fp_hard == eval_fingerprint_hard, f"Hard eval fingerprint mismatch! {fp_hard} != {eval_fingerprint_hard}"
        assert fp_easy == eval_fingerprint_easy, f"Easy eval fingerprint mismatch! {fp_easy} != {eval_fingerprint_easy}"
        rank_print(f"[BASELINE] Fingerprints verified ✓")

        # Hard eval: α=1.0
        base_tf_hard = run_eval_teacher_forced_parity(
            eval_model, tokenizer, args.task, eval_inputs_hard, eval_labels_hard,
            num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
            print_mistakes=args.print_eval_examples, gc_every=50, **task_kwargs
        )

        # Easy eval: α=base_alpha
        base_tf_easy = run_eval_teacher_forced_parity(
            eval_model, tokenizer, args.task, eval_inputs_easy, eval_labels_easy,
            num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
            print_mistakes=args.print_eval_examples, gc_every=50, **task_kwargs
        )

        rank_print(
            f"\n[BASELINE-TF-α1.0] First={base_tf_hard['first_token_acc']:.2%} "
            f"| Full={base_tf_hard['full_word_acc']:.2%} | N={base_tf_hard['total']}"
        )
        rank_print(
            f"[BASELINE-TF-α{args.base_alpha}] First={base_tf_easy['first_token_acc']:.2%} "
            f"| Full={base_tf_easy['full_word_acc']:.2%} | N={base_tf_easy['total']}\n"
        )

    # Skip baseline if not requested OR if resuming
    baseline_cb = BaselineEvalCallback(
        baseline_runner,
        skip_baseline=(not args.do_baseline) or (resume_ckpt is not None)
    )

    trainer = SinglePathARTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=make_collate(tokenizer, pad_to_multiple_of=64),
        callbacks=[curriculum, baseline_cb],
        first_token_soft_weight=args.first_token_soft_weight,
        use_chunked_ce=args.use_chunked_ce,
        ce_chunk_size=args.ce_chunk_size,
    )
    curriculum.trainer = trainer
    baseline_cb.trainer = trainer

    if not resume_ckpt:
        if is_main_process():
            _save_run_config(args.output_dir, trainer.args.per_device_train_batch_size,
                             trainer.args.gradient_accumulation_steps)

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
                    "eval_fingerprint_hard": eval_fingerprint_hard,
                    "eval_fingerprint_easy": eval_fingerprint_easy,
                }
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"[META][WARN] Could not write run_meta.json: {e}")

    # Save initial curriculum state
    _save_curriculum_state(args.output_dir, dataset.stage, curriculum.stage_start_step)

    # ----- TRAINING STARTS HERE -----
    rank_print("\n[TRAIN] Starting training...\n")

    if resume_ckpt:
        match = re.search(r'checkpoint-(\d+)', resume_ckpt)
        if match:
            resume_step = int(match.group(1))
            curriculum.last_resume_step = resume_step
            rank_print(f"[CURRICULUM] Resuming from step {resume_step}")
            rank_print(f"[CURRICULUM] Will skip checks for {curriculum.resume_cooldown_steps} steps (warmup period)")

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

    # ==================== FINAL EVALUATIONS ====================

    # ----- Seen samples sanity check (should be ~100%) -----
    if args.do_seen_eval:
        local_inputs, local_labels = dataset.get_seen_samples()

        all_inputs_list = []
        all_labels_list = []

        for rank_id in range(get_world_size()):
            if rank_id == get_rank():
                broadcast_object(local_inputs, src=rank_id)
                broadcast_object(local_labels, src=rank_id)
                if is_main_process():
                    all_inputs_list.append(local_inputs)
                    all_labels_list.append(local_labels)
            else:
                rank_inputs = broadcast_object(None, src=rank_id)
                rank_labels = broadcast_object(None, src=rank_id)
                if is_main_process():
                    all_inputs_list.append(rank_inputs)
                    all_labels_list.append(rank_labels)

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

            rank_print(f"[SEEN-EVAL] Gathered {len(seen_inputs)} unique samples from {get_world_size()} ranks")
        else:
            seen_inputs = None
            seen_labels = None

        barrier()
        seen_inputs = broadcast_object(seen_inputs, src=0)
        seen_labels = broadcast_object(seen_labels, src=0)

        if seen_inputs:
            rank_print(f"[SEEN-EVAL] Evaluating on {len(seen_inputs)} seen samples")
            trainer.model.eval()
            seen_tf = run_eval_teacher_forced_parity(
                trainer.model,
                tokenizer, args.task, seen_inputs, seen_labels,
                num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
                print_mistakes=min(5, args.print_eval_examples), gc_every=50, **task_kwargs
            )
            rank_print(
                f"[SEEN-EVAL] First={seen_tf['first_token_acc']:.2%} | Full={seen_tf['full_word_acc']:.2%} | N={seen_tf['total']}")

        barrier()

    # ----- Final eval: TF + greedy at α=1.0 and α=base_alpha -----
    if args.do_final_eval:
        trainer.model.eval()

        # Verify we're using the same data
        fp_hard = _eval_data_fingerprint(eval_inputs_hard, eval_labels_hard)
        fp_easy = _eval_data_fingerprint(eval_inputs_easy, eval_labels_easy)
        rank_print(f"[FINAL-EVAL] Verifying eval data: hard={fp_hard}, easy={fp_easy}")
        assert fp_hard == eval_fingerprint_hard, f"Hard eval fingerprint mismatch! {fp_hard} != {eval_fingerprint_hard}"
        assert fp_easy == eval_fingerprint_easy, f"Easy eval fingerprint mismatch! {fp_easy} != {eval_fingerprint_easy}"
        rank_print(f"[FINAL-EVAL] Fingerprints verified ✓")

        # Hard eval: α=1.0
        final_tf_hard = run_eval_teacher_forced_parity(
            trainer.model, tokenizer, args.task, eval_inputs_hard, eval_labels_hard,
            num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
            print_mistakes=args.print_eval_examples, gc_every=50, **task_kwargs
        )
        rank_print(
            f"[FINAL-TF-α1.0] First={final_tf_hard['first_token_acc']:.2%} | Full={final_tf_hard['full_word_acc']:.2%} | N={final_tf_hard['total']}")

        greedy_hard = run_eval_greedy_readable(
            trainer.model, tokenizer, args.task,
            eval_inputs_hard, eval_labels_hard,
            num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
            print_examples=min(3, args.print_eval_examples), **task_kwargs
        )
        rank_print(
            f"[FINAL-GREEDY-α1.0] First={greedy_hard['first_token_acc']:.2%} | Full={greedy_hard['full_word_acc']:.2%} | N={greedy_hard['total']}")

        # Easy eval: α=base_alpha
        final_tf_easy = run_eval_teacher_forced_parity(
            trainer.model, tokenizer, args.task, eval_inputs_easy, eval_labels_easy,
            num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
            print_mistakes=0, gc_every=50, **task_kwargs
        )
        rank_print(
            f"[FINAL-TF-α{args.base_alpha}] First={final_tf_easy['first_token_acc']:.2%} | Full={final_tf_easy['full_word_acc']:.2%} | N={final_tf_easy['total']}")

        greedy_easy = run_eval_greedy_readable(
            trainer.model, tokenizer, args.task,
            eval_inputs_easy, eval_labels_easy,
            num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
            print_examples=0, **task_kwargs
        )
        rank_print(
            f"[FINAL-GREEDY-α{args.base_alpha}] First={greedy_easy['first_token_acc']:.2%} | Full={greedy_easy['full_word_acc']:.2%} | N={greedy_easy['total']}")

        # Save metrics
        if is_main_process():
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
                json.dump({
                    "eval_fingerprint_hard": eval_fingerprint_hard,
                    "eval_fingerprint_easy": eval_fingerprint_easy,
                    "tf_hard": final_tf_hard,
                    "greedy_hard": greedy_hard,
                    "tf_easy": final_tf_easy,
                    "greedy_easy": greedy_easy,
                }, f, indent=2)

    # ----- Redacted eval: sanity check (should be low) -----
    if args.do_redacted_eval:
        trainer.model.eval()

        # Verify we're using the hard eval data
        fp_hard = _eval_data_fingerprint(eval_inputs_hard, eval_labels_hard)
        rank_print(f"[REDACTED-EVAL] Verifying source data: hard={fp_hard}")
        assert fp_hard == eval_fingerprint_hard, f"Hard eval fingerprint mismatch! {fp_hard} != {eval_fingerprint_hard}"
        rank_print(f"[REDACTED-EVAL] Fingerprint verified ✓")

        red_n = args.eval_redacted_samples
        if red_n is None or red_n <= 0:
            red_n = args.eval_samples

        # Redact the hard eval set (α=1.0)
        red_inputs, red_labels = build_redacted_eval_set(
            eval_inputs_hard, eval_labels_hard,
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
                f"[REDACTED-TF] First={red_tf['first_token_acc']:.2%} | Full={red_tf['full_word_acc']:.2%} | N={red_tf['total']}")

            red_greedy = run_eval_greedy_readable(
                trainer.model,
                tokenizer, args.task,
                red_inputs, red_labels,
                num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
                print_examples=min(3, args.print_eval_examples), **task_kwargs
            )
            rank_print(
                f"[REDACTED-GREEDY] First={red_greedy['first_token_acc']:.2%} | Full={red_greedy['full_word_acc']:.2%} | N={red_greedy['total']}")
        else:
            rank_print("[REDACTED-EVAL] Skipped (could not redact any eval items cleanly)")

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save model and tokenizer
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

    rank_print("\n[DONE] Training/evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())