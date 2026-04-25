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
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Any, Optional, Set
import multiprocessing

import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

# PyTorch 2.6+ defaults torch.load to weights_only=True, which rejects the
# numpy RNG state pickled by HF Trainer in rng_state_*.pth. We trust our own
# checkpoints, so override the default to keep resume working.
_torch_load_orig = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load_orig(*args, **kwargs)
torch.load = _torch_load_compat

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
from transformers import GenerationConfig
import torch.utils.checkpoint as checkpoint

from peft import LoraConfig, get_peft_model, TaskType

from nl_generator import NaturalLanguageGraphGenerator

# Check for flash-attn availability
try:
    from flash_attn import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_varlen_func = None

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# --- Runtime env sanity for Accelerate / NCCL (must be set before Trainer builds Accelerator) ---
os.environ.setdefault("ACCELERATE_DISPATCH_BATCHES", "false")
os.environ.setdefault("ACCELERATE_SPLIT_BATCHES", "true")
os.environ.setdefault("ACCELERATE_USE_DATA_LOADER_SHARDING", "false")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

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


def estimate_flops_per_token(model) -> int:
    """Estimate FLOPs per token (6N approximation per Kaplan et al.)"""
    total_params = sum(p.numel() for p in model.parameters())
    return 6 * total_params


# Function for setting seed across libraries and GPUs
def set_all_seeds(seed: int, deterministic: bool = False):
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
    if deterministic:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        try:
            torch.use_deterministic_algorithms(True)
            rank_print("[SEED] Deterministic algorithms enabled")
        except Exception as e:
            rank_print(f"[SEED] Could not enable deterministic algorithms: {e}")
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(true_seed)
    except Exception:
        pass


# ================== Curriculum state persistence ==================

def _save_curriculum_state(dirpath: str, stage: int, stage_start_step: int, wall_time_offset: float = 0.0,
                           first_token_correct=None, full_word_correct=None,
                           recent_losses=None, samples_this_stage: int = 0,
                           tokens_this_stage: int = 0,
                           lr_reset_step: int = None,
                           batch_increase_count: int = None,
                           plateau_last_spike_step: int = None) -> None:
    try:
        os.makedirs(dirpath, exist_ok=True)
        fp = os.path.join(dirpath, "curriculum_state.json")
        data = {
            "stage": int(stage),
            "stage_start_step": int(stage_start_step),
            "wall_time_offset": float(wall_time_offset),
            "samples_this_stage": int(samples_this_stage),
            "tokens_this_stage": int(tokens_this_stage),
        }
        if lr_reset_step is not None:
            data["lr_reset_step"] = int(lr_reset_step)
        if batch_increase_count is not None:
            data["batch_increase_count"] = int(batch_increase_count)
        if plateau_last_spike_step is not None:
            data["plateau_last_spike_step"] = int(plateau_last_spike_step)
        if first_token_correct is not None:
            data["first_token_correct"] = list(first_token_correct)
        if full_word_correct is not None:
            data["full_word_correct"] = list(full_word_correct)
        if recent_losses is not None:
            data["recent_losses"] = list(recent_losses)
        with open(fp, "w") as f:
            json.dump(data, f)
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


# ================== Loss history persistence (JSONL) ==================

def _atomic_write_json(path, data):
    """Write JSON atomically: write to tmp, fsync, rename."""
    tmp_path = path + f".tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _append_to_jsonl(dirpath, entries):
    """Append loss entries to loss_history.jsonl (append-only persistence)."""
    path = os.path.join(dirpath, "loss_history.jsonl")
    with open(path, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry, separators=(',', ':')) + "\n")
        f.flush()


def _load_from_jsonl(dirpath, max_step=None):
    """Load loss entries from JSONL file, optionally truncating to max_step.

    Returns list of entries, or None if file doesn't exist.
    Gracefully handles corrupted lines (from crashes mid-write).
    """
    path = os.path.join(dirpath, "loss_history.jsonl")
    if not os.path.isfile(path):
        return None
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if max_step is None or entry.get("step", 0) <= max_step:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue  # Skip corrupted trailing line from crash
    return entries if entries else None


def _rewrite_jsonl(dirpath, entries):
    """Rewrite JSONL file with given entries (used after truncation on resume)."""
    path = os.path.join(dirpath, "loss_history.jsonl")
    tmp_path = path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, separators=(',', ':')) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp_path, path)


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
        curriculum.wall_time_offset = float(cs.get("wall_time_offset", 0.0))
        curriculum._restored_trainer_state = {
            "first_token_correct": cs.get("first_token_correct"),
            "full_word_correct": cs.get("full_word_correct"),
            "recent_losses": cs.get("recent_losses"),
        }
        curriculum.samples_this_stage = int(cs.get("samples_this_stage", 0))
        curriculum.tokens_this_stage = int(cs.get("tokens_this_stage", 0))
        if cs.get("lr_reset_step") is not None:
            curriculum.lr_reset_step = int(cs["lr_reset_step"])
        if cs.get("batch_increase_count") is not None:
            curriculum.batch_increase_count = int(cs["batch_increase_count"])
        if cs.get("plateau_last_spike_step") is not None:
            curriculum.plateau_last_spike_step = int(cs["plateau_last_spike_step"])
        rank_print(
            f"[CURRICULUM] Restored stage={dataset.stage}, stage_start_step={curriculum.stage_start_step}, wall_time_offset={curriculum.wall_time_offset:.1f}s from {fp}")
    except Exception as e:
        rank_print(f"[CURRICULUM][WARN] Failed to restore from {fp}: {e}")
        return False

    # Restore loss history — prefer JSONL (has entries between checkpoint saves)
    output_dir = os.path.dirname(path)
    resume_step_from_ckpt = 0
    ckpt_match = re.search(r'checkpoint-(\d+)', path)
    if ckpt_match:
        resume_step_from_ckpt = int(ckpt_match.group(1))
    else:
        # Fallback: try to extract step from trainer_state.json in checkpoint dir
        ts_path = os.path.join(path, "trainer_state.json")
        if os.path.isfile(ts_path):
            try:
                with open(ts_path) as f:
                    ts = json.load(f)
                resume_step_from_ckpt = int(ts.get("global_step", 0))
                rank_print(f"[CURRICULUM] Extracted step {resume_step_from_ckpt} from trainer_state.json")
            except Exception:
                pass
        if resume_step_from_ckpt == 0:
            # Last resort: use step from curriculum_state.json that we just loaded
            resume_step_from_ckpt = curriculum.stage_start_step or 0
            rank_print(f"[CURRICULUM][WARN] Could not extract step from checkpoint path, using {resume_step_from_ckpt}")

    # Load from JSONL first (has entries between checkpoint saves), fall back to JSON
    # NOTE: We only READ here — JSONL is written to the actual output_dir by the caller
    jsonl_entries = _load_from_jsonl(output_dir, max_step=resume_step_from_ckpt)
    if jsonl_entries:
        # Filter out legacy fake resume entries (tokens=0, resume=True)
        curriculum.loss_history = [
            h for h in jsonl_entries
            if not (h.get("resume") and h.get("tokens", -1) == 0)
        ]
        rank_print(f"[CURRICULUM] Restored {len(curriculum.loss_history)} loss records from JSONL (truncated to step {resume_step_from_ckpt})")
    else:
        # Fall back to loss_history.json
        loss_path = os.path.join(output_dir, "loss_history.json")
        if os.path.isfile(loss_path):
            try:
                with open(loss_path) as f:
                    loaded = json.load(f)
                # Filter out legacy fake resume entries and entries beyond checkpoint
                curriculum.loss_history = [
                    h for h in loaded
                    if h.get("step", 0) <= resume_step_from_ckpt
                    and not (h.get("resume") and h.get("tokens", -1) == 0)
                ]
                rank_print(f"[CURRICULUM] Restored {len(curriculum.loss_history)} loss records from JSON")
            except Exception as e:
                rank_print(f"[CURRICULUM][WARN] Failed to restore loss history: {e}")

    if curriculum.loss_history:
        # Update wall_time_offset from last recorded entry
        last_wall_time = curriculum.loss_history[-1].get("wall_time", 0.0)
        if last_wall_time > curriculum.wall_time_offset:
            curriculum.wall_time_offset = last_wall_time
            rank_print(f"[CURRICULUM] Updated wall_time_offset to {last_wall_time:.1f}s from loss history")

        # Flag to mark the first new entry as a resume point (real data, not fake entry)
        curriculum._mark_next_as_resume = True
        curriculum.resume_points.append(resume_step_from_ckpt)
        rank_print(f"[CURRICULUM] Will mark next entry as resume point (step ~{resume_step_from_ckpt})")

        # Set _last_persist_step so we don't re-persist old steps on resume
        curriculum._last_persist_step = curriculum.loss_history[-1].get("step", 0)

    # Restore stage eval history
    stage_eval_path = os.path.join(os.path.dirname(path), "stage_eval_history.json")
    if os.path.isfile(stage_eval_path):
        try:
            with open(stage_eval_path) as f:
                curriculum.stage_eval_history = json.load(f)
            rank_print(f"[CURRICULUM] Restored {len(curriculum.stage_eval_history)} stage eval records")
        except Exception as e:
            rank_print(f"[CURRICULUM][WARN] Failed to restore stage eval history: {e}")

    return True


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


def alpha_for_lookahead(L_target: int,
                        n: int,
                        tokens_per_edge: int = 3,
                        fixed_tokens: int = 4,
                        reserve_edges: int = 1) -> float:
    """
    Calculate the minimum alpha needed to achieve target effective lookahead L.
    Inverse of effective_search_L().
    """
    if L_target <= 0:
        return 0.0

    edges_unscaled = max(1, (n - fixed_tokens) // tokens_per_edge)

    # From effective_search_L:
    # L_edges = (alpha * edges_unscaled - reserve_edges) // 2
    # To get L_target, we need: (alpha * edges_unscaled - reserve_edges) >= 2 * L_target
    # alpha >= (2 * L_target + reserve_edges) / edges_unscaled

    needed_alpha = (2 * L_target + reserve_edges) / edges_unscaled

    return min(max(needed_alpha, 0.0), 1.0)


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


def build_scrambled_eval_set(
        inputs: List[str],
        labels: List[List[str]],
        max_n: Optional[int] = None,
        seed: int = 12345,
) -> Tuple[List[str], List[List[str]]]:
    """Scramble rule consequents to destroy graph structure.

    For each eval example, randomly permutes the successor words across all
    rules while keeping the exact same format, vocabulary, and question.
    The graph edges become nonsensical so no valid path from given -> target
    exists. Expected accuracy should be near baseline (~3%) if the model is
    truly doing graph traversal rather than pattern matching.
    """
    rng = random.Random(seed)
    # Regex: matches a rule sentence and captures (prefix, consequent_word)
    # Handles: "If X is A, then X/they are B" / "Everyone that is A is B"
    #          "If someone is A, then they are B" / "If a person is A, they are B"
    _rule_re = re.compile(
        r'('
        r'(?:If\s+(?:[A-Z][a-z]+|someone|a person)\s+is\s+[a-z]+,?\s*then\s+(?:[A-Z][a-z]+|they)\s+are\s+)'
        r'|(?:Everyone\s+that\s+is\s+[a-z]+\s+is\s+)'
        r')'
        r'([a-z]+)'
        r'(\.\s*)'
    )

    scr_inputs, scr_labels = [], []
    for x, y in zip(inputs, labels):
        # Only scramble the rules section (before "Given that")
        given_idx = x.find("Given that")
        if given_idx == -1:
            continue

        rules_part = x[:given_idx]
        question_part = x[given_idx:]

        matches = list(_rule_re.finditer(rules_part))
        if len(matches) < 3:
            continue

        # Extract and shuffle consequents
        consequents = [m.group(2) for m in matches]
        shuffled = consequents[:]
        for _ in range(20):
            rng.shuffle(shuffled)
            if shuffled != consequents:
                break

        # Rebuild rules_part with shuffled consequents (replace from end)
        new_rules = list(rules_part)
        for m, new_word in reversed(list(zip(matches, shuffled))):
            s, e = m.start(2), m.end(2)
            new_rules[s:e] = list(new_word)
        new_rules = ''.join(new_rules)

        scr_inputs.append(new_rules + question_part)
        scr_labels.append(y)
        if max_n is not None and len(scr_inputs) >= max_n:
            break

    return scr_inputs, scr_labels


# --------------- Plotting (extracted to plot_training.py) ---------------
from plot_training import (
    fit_exponential_decay,
    plot_stage_loss,
    plot_overall_loss,
    plot_loss_vs_flops,
    plot_loss_vs_walltime,
    plot_achieved_tflops,
    plot_stage_eval,
    plot_eval_acc_vs_step,
    plot_eval_acc_vs_flops,
    generate_all_plots,
    save_plot_data,
)
# Wire up rank_print so plot functions use it during training
import plot_training as _plot_mod
_plot_mod._print = rank_print


class PackedSequenceDataset(Dataset):
    """
    Map-style dataset for packed sequences.
    Each __getitem__ returns one fully packed batch.
    Enables multi-worker prefetching for better performance.
    """

    def __init__(
            self,
            task: str,
            tokenizer,
            batch_size: int = 64,
            stage: int = 1,
            n_stages: int = 10,
            base_alpha: float = 0.1,
            max_alpha: float = 1.0,
            max_input_size: int = 256,
            linear_lookahead: bool = False,
            base_lookahead: int = 1,
            lookahead_step: int = 1,
            reserved_inputs: Optional[Set[str]] = None,
            num_shots: int = 0,
            seed: Optional[int] = None,
            resume_step: int = 0,
            store_examples: bool = False,
            store_cap: int = 1000,
            epoch_size: int = 10_000_000,  # Large enough to never cycle
            mix_pretrain_data: Optional[str] = None,
            mix_pretrain_subset: Optional[str] = "en",
            mix_pretrain_ratio: float = 0.1,
            mix_pretrain_max_len: int = 512,
            mix_pretrain_cache_dir: Optional[str] = None,
            use_chat_template: bool = False,
            **task_kwargs,
    ):
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError(
                "flash-attn required for PackedSequenceDataset. "
                "Install with: pip install flash-attn --no-build-isolation"
            )

        self.task = task
        self.tokenizer = tokenizer
        self.target_samples_per_batch = batch_size
        self._stage = multiprocessing.Value('i', stage)
        self.n_stages = n_stages
        self.base_alpha = base_alpha
        self.max_alpha = max_alpha
        self.linear_lookahead = linear_lookahead
        self.base_lookahead = base_lookahead
        self.lookahead_step = lookahead_step
        self.max_input_size = max_input_size
        self.reserved_inputs = reserved_inputs or set()
        self.num_shots = num_shots
        self.seed = seed
        self.resume_step = resume_step
        self.task_kwargs = task_kwargs
        self.epoch_size = epoch_size

        # Pretraining data mixing (anti-catastrophic-forgetting)
        self.mix_pretrain_data = mix_pretrain_data
        self.mix_pretrain_subset = mix_pretrain_subset
        self.mix_pretrain_ratio = mix_pretrain_ratio
        self.mix_pretrain_max_len = mix_pretrain_max_len
        self.mix_pretrain_cache_dir = mix_pretrain_cache_dir
        self.use_chat_template = use_chat_template
        if mix_pretrain_data:
            print(f"[DATASET] Pretraining mix: {mix_pretrain_data} ({mix_pretrain_subset}), "
                  f"ratio={mix_pretrain_ratio:.0%}, max_len={mix_pretrain_max_len}")
        if use_chat_template:
            print(f"[DATASET] Chat template enabled for search data (enable_thinking=False)")

        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # Storage for seen samples (thread-safe for multi-worker)
        self._store = bool(store_examples)
        self._store_cap = int(store_cap)
        self._seen_inputs: deque = deque(maxlen=self._store_cap)
        self._seen_labels: deque = deque(maxlen=self._store_cap)
        self._seen_lock = multiprocessing.Lock() if store_examples else None

        # Per-worker state (lazily initialized)
        self._worker_generator = None
        self._worker_rng = None
        self._pretrain_iterator = None
        self._worker_id = None

        self.few_shot_examples = self._build_few_shots(num_shots, seed)

        rank_print(f"[DATASET] Packed Map-style | batch_size={batch_size} | epoch_size={epoch_size}")

    @property
    def stage(self):
        return self._stage.value

    @stage.setter
    def stage(self, value):
        self._stage.value = value

    def __len__(self):
        return self.epoch_size

    def get_seen_samples(self) -> Tuple[List[str], List[List[str]]]:
        """Return stored training samples (thread-safe)."""
        if self._seen_lock:
            with self._seen_lock:
                return list(self._seen_inputs), list(self._seen_labels)
        return list(self._seen_inputs), list(self._seen_labels)

    def _stage_target_lookahead(self) -> Optional[int]:
        """Get target lookahead for current stage (search task with linear_lookahead only)."""
        if not (self.linear_lookahead and self.task == "search"):
            return None
        max_L = self.task_kwargs.get("max_lookahead", 12)
        target_L = self.base_lookahead + (self.stage - 1) * self.lookahead_step
        return min(target_L, max_L)

    def _stage_alpha(self) -> float:
        """Calculate alpha for current curriculum stage."""
        if self.linear_lookahead and self.task == "search":
            target_L = self._stage_target_lookahead()
            return alpha_for_lookahead(target_L, self.max_input_size)
        else:
            if self.stage >= self.n_stages:
                return self.max_alpha
            return self.base_alpha + (self.max_alpha - self.base_alpha) * (self.stage - 1) / max(self.n_stages - 1, 1)

    def _is_final_stage(self) -> bool:
        """Check if current stage is the final stage."""
        if self.linear_lookahead and self.task == "search":
            max_L = self.task_kwargs.get("max_lookahead", 12)
            current_L = self._stage_target_lookahead()
            return current_L >= max_L
        else:
            return self.stage >= self.n_stages

    def _build_few_shots(self, k: int, seed: Optional[int]):
        """Build few-shot examples for prompting."""
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

    def _shots_prefix(self) -> str:
        """Build few-shot prefix for prompts (cached after first call)."""
        if not hasattr(self, '_cached_shots_prefix'):
            if not self.few_shot_examples:
                self._cached_shots_prefix = ""
            else:
                parts = []
                for ex in self.few_shot_examples:
                    tt = _determine_task_type(self.task, ex["input"])
                    parts.append(f"{ex['input']} {ex['output']}{_get_end_tokens(tt)}")
                self._cached_shots_prefix = "\n\n".join(parts) + ("\n\n" if parts else "")
        return self._cached_shots_prefix

    def _get_worker_state(self, idx: int):
        """idx should already be offset by resume_step from __getitem__"""
        worker_info = torch.utils.data.get_worker_info()
        current_worker_id = worker_info.id if worker_info else 0

        if self._worker_generator is None or self._worker_id != current_worker_id:
            rank = get_rank()
            worker_seed = (self.seed or 0) + rank * 9973 + current_worker_id * 7919

            self._worker_generator = NaturalLanguageGraphGenerator(
                self.max_input_size,
                seed=worker_seed
            )
            self._worker_rng = random.Random(worker_seed)
            self._worker_id = current_worker_id

        # Reseed ALL RNGs per batch so data is deterministic by index, not call order.
        # This prevents data cycling on resume (where workers are recreated with fresh RNG state).
        batch_seed = ((self.seed or 0) +
                      idx * 104729 +
                      self._worker_id * 7919 +
                      get_rank() * 999983)
        self._worker_rng.seed(batch_seed)
        import generator as _gen_module
        _gen_module.set_seed(batch_seed & 0x7FFFFFFF)  # C++ minstd_rand uses unsigned 31-bit seed
        random.seed(batch_seed)  # NL text generation uses global random

        return self._worker_generator, self._worker_rng

    def _generate_one_sample(self, rng: random.Random) -> Optional[Dict[str, Any]]:
        """Generate a single tokenized sample."""
        gen = self._worker_generator
        alpha = self._stage_alpha()

        ex = None
        for _ in range(100):
            batch = gen.generate_batch(
                self.task, batch_size=1,
                reserved_inputs=self.reserved_inputs,
                alpha=alpha, **self.task_kwargs
            )
            if batch and batch[0] and batch[0].output_texts:
                candidate = batch[0]
                if candidate.input_text not in self.reserved_inputs:
                    ex = candidate
                    break

        if ex is None:
            return None

        shots = self._shots_prefix()
        prompt_text = (shots + ex.input_text) if shots else ex.input_text
        chosen = rng.choice(ex.output_texts)
        task_type = _determine_task_type(self.task, ex.input_text)

        if self.use_chat_template:
            # Wrap search data in chat template with enable_thinking=False
            msgs = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": chosen},
            ]
            full_ids = self.tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
                enable_thinking=False,
            )
            # Find the boundary: tokenize prefix up to assistant response
            prefix_ids = self.tokenizer.apply_chat_template(
                msgs[:1], tokenize=True, add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = full_ids[:len(prefix_ids)]
            ans_ids = full_ids[len(prefix_ids):]
            input_ids = full_ids
            labels = [-100] * len(prompt_ids) + ans_ids
        else:
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=True, truncation=False)["input_ids"]
            ans_ids = _tokenize_leading_space(self.tokenizer, chosen)
            end_ids = self.tokenizer(_get_end_tokens(task_type), add_special_tokens=False)["input_ids"]

            input_ids = prompt_ids + ans_ids + end_ids
            labels = [-100] * len(prompt_ids) + ans_ids + end_ids

        # Build valid first token targets (tokenize each alternative once)
        first_union = sorted({
            tokens[0]
            for tokens in (_tokenize_leading_space(self.tokenizer, a) for a in ex.output_texts)
            if tokens
        })

        # Store for seen eval (thread-safe)
        if self._store and len(self._seen_inputs) < self._store_cap:
            if self._seen_lock:
                with self._seen_lock:
                    if len(self._seen_inputs) < self._store_cap:
                        self._seen_inputs.append(ex.input_text)
                        self._seen_labels.append(list(ex.output_texts))
            else:
                self._seen_inputs.append(ex.input_text)
                self._seen_labels.append(list(ex.output_texts))

        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": len(prompt_ids),
            "valid_first_targets": first_union,
            "seq_len": len(input_ids),
        }

    def _pack_batch(self, samples: List[Dict]) -> Dict[str, Any]:
        """
        Pack all samples into a single flat sequence for flash_attn_varlen_func.
        No padding, no fixed pack_length — size is determined by actual content.
        """
        all_input_ids = []
        all_labels = []
        all_position_ids = []
        cu_seqlens = [0]
        max_seqlen = 0
        all_sequence_info = []

        offset = 0
        for sample in samples:
            seq_len = sample["seq_len"]

            all_input_ids.extend(sample["input_ids"])
            all_labels.extend(sample["labels"])
            all_position_ids.extend(range(seq_len))

            offset += seq_len
            cu_seqlens.append(offset)

            max_seqlen = max(max_seqlen, seq_len)

            all_sequence_info.append({
                "row_idx": 0,
                "start_idx": cu_seqlens[-2],
                "end_idx": cu_seqlens[-1],
                "prompt_len": sample["prompt_len"],
                "valid_first_targets": sample["valid_first_targets"],
            })

        total_tokens = len(all_input_ids)

        return {
            "input_ids": torch.tensor([all_input_ids], dtype=torch.long),       # [1, total_tokens]
            "labels": torch.tensor([all_labels], dtype=torch.long),             # [1, total_tokens]
            "position_ids": torch.tensor([all_position_ids], dtype=torch.long), # [1, total_tokens]
            "cu_seqlens_list": [cu_seqlens],
            "max_seqlen_list": [max_seqlen],
            "sequence_info": all_sequence_info,
            "num_sequences": len(samples),
            "_efficiency": 100.0,
        }

    def _get_pretrain_iterator(self, rng):
        """Lazily load streaming pretraining dataset in each worker."""
        if self._pretrain_iterator is None:
            import os
            os.environ.setdefault("HF_DATASETS_DOWNLOAD_TIMEOUT", "120")
            from datasets import load_dataset
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else 0
            load_kwargs = dict(
                path=self.mix_pretrain_data,
                split="train",
                streaming=True,
                cache_dir=self.mix_pretrain_cache_dir,
                trust_remote_code=True,
                download_config=__import__('datasets').DownloadConfig(
                    num_proc=1, max_retries=5,
                ),
            )
            if self.mix_pretrain_subset:
                load_kwargs["name"] = self.mix_pretrain_subset
            ds = load_dataset(**load_kwargs)
            # Shuffle with worker-specific seed
            seed = (self.seed or 0) + 77777 + worker_id
            ds = ds.shuffle(seed=seed, buffer_size=1000)
            self._pretrain_iterator = iter(ds)
        return self._pretrain_iterator

    def _generate_one_pretrain_sample(self, rng):
        """Generate a single pretraining/instruction sample.

        Supports two dataset formats:
          - Raw text (e.g. C4): uses 'text' field, standard causal LM loss on all tokens
          - Chat/instruction (e.g. Dolci-Instruct-SFT): uses 'messages' field,
            formats with chat template, loss only on assistant turns
        """
        pretrain_iter = self._get_pretrain_iterator(rng)
        max_len = self.mix_pretrain_max_len

        for _ in range(20):  # try up to 20 texts to find a valid one
            try:
                example = next(pretrain_iter)
            except StopIteration:
                self._pretrain_iterator = None
                pretrain_iter = self._get_pretrain_iterator(rng)
                example = next(pretrain_iter)
            except Exception:
                import time
                time.sleep(1)
                continue

            # Chat/instruction format (e.g. Dolci-Instruct-SFT)
            if "messages" in example and example["messages"]:
                messages = example["messages"]
                # Need at least one user + one assistant message
                if len(messages) < 2:
                    continue
                # Filter to role/content only
                msgs = [{"role": m["role"], "content": m["content"]} for m in messages
                        if m.get("role") and m.get("content")]
                if len(msgs) < 2:
                    continue

                # Use chat template with enable_thinking=False to properly format
                # instruction data with empty <think> blocks (matches Qwen3 non-thinking mode)
                try:
                    # Tokenize full conversation
                    full_ids = self.tokenizer.apply_chat_template(
                        msgs, tokenize=True, add_generation_prompt=False,
                        enable_thinking=False,
                        truncation=True, max_length=max_len,
                    )
                except Exception:
                    continue

                if len(full_ids) < 4:
                    continue

                # Build labels: only train on assistant response tokens
                # Tokenize prefixes to find assistant response boundaries
                labels = [-100] * len(full_ids)
                for i in range(len(msgs)):
                    if msgs[i]["role"] != "assistant":
                        continue
                    # Prefix: everything up to this assistant turn (with generation prompt)
                    if i > 0:
                        prefix_ids = self.tokenizer.apply_chat_template(
                            msgs[:i], tokenize=True, add_generation_prompt=True,
                            enable_thinking=False,
                            truncation=True, max_length=max_len,
                        )
                    else:
                        prefix_ids = []
                    # Through: everything up to and including this assistant turn
                    through_ids = self.tokenizer.apply_chat_template(
                        msgs[:i + 1], tokenize=True, add_generation_prompt=False,
                        enable_thinking=False,
                        truncation=True, max_length=max_len,
                    )
                    tok_start = len(prefix_ids)
                    tok_end = min(len(through_ids), len(full_ids))
                    # Labels: compute_loss already shifts (shift_labels = all_labels[1:]),
                    # so labels[t] = full_ids[t] (same position, NOT t+1)
                    for t in range(tok_start, tok_end):
                        labels[t] = full_ids[t]

                if all(l == -100 for l in labels):
                    continue

                return {
                    "input_ids": full_ids,
                    "labels": labels,
                    "seq_len": len(full_ids),
                    "prompt_len": 0,
                    "valid_first_targets": [],
                }

            # Raw text format (e.g. C4)
            text = example.get("text", "")
            if not text or len(text) < 10:
                continue

            token_ids = self.tokenizer(
                text, add_special_tokens=True, truncation=True,
                max_length=max_len, return_attention_mask=False,
            )["input_ids"]

            if len(token_ids) < 4:
                continue

            # Standard causal LM: labels = input_ids, with -100 at position 0
            labels = [-100] + token_ids[1:]

            return {
                "input_ids": token_ids,
                "labels": labels,
                "seq_len": len(token_ids),
                "prompt_len": 0,
                "valid_first_targets": [],
            }

        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Generate one fully packed batch.

        Each call generates target_samples_per_batch samples and packs them
        into a single flat sequence. With mix_pretrain_data enabled, a fraction
        of samples within each batch are pretraining data (within-batch mixing).
        """
        effective_idx = idx + self.resume_step
        gen, rng = self._get_worker_state(effective_idx)

        target_samples = self.target_samples_per_batch

        # Determine how many pretrain vs search samples in this batch
        if self.mix_pretrain_data and self.mix_pretrain_ratio > 0:
            n_pretrain = int(target_samples * self.mix_pretrain_ratio)
            n_search = target_samples - n_pretrain
        else:
            n_pretrain = 0
            n_search = target_samples

        all_samples = []

        # Generate search samples
        max_attempts = n_search * 10
        attempts = 0
        while len(all_samples) < n_search and attempts < max_attempts:
            attempts += 1
            sample = self._generate_one_sample(rng)
            if sample is None:
                continue
            all_samples.append(sample)

        # Generate pretrain samples (within same batch)
        for _ in range(n_pretrain):
            pt_sample = self._generate_one_pretrain_sample(rng)
            if pt_sample is not None:
                all_samples.append(pt_sample)

        # Handle edge case: no samples generated
        if not all_samples:
            raise RuntimeError(
                f"[DATASET] Failed to generate any samples for idx={idx}. "
                f"Stage={self.stage}, alpha={self._stage_alpha():.3f}"
            )

        return self._pack_batch(all_samples)


class PackedSequenceTrainer(Trainer):
    """
    Trainer for packed sequences using Flash Attention varlen.
    Supports Qwen and GPT-NeoX (Pythia) architectures.
    """

    def __init__(self, *args, first_token_soft_weight=0.3, accuracy_window=200, ce_chunk_size=4096, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_token_soft_weight = first_token_soft_weight
        self.ce_chunk_size = ce_chunk_size
        self.recent_losses = deque(maxlen=100)
        self.first_token_correct = deque(maxlen=accuracy_window)
        self.full_word_correct = deque(maxlen=accuracy_window)
        self._last_search_loss = None
        self._last_pretrain_loss = None

        self._last_batch_samples = 0
        self._last_batch_tokens = 0
        self._last_efficiency = None

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Override to handle scheduler state mismatches gracefully (e.g. when
        switching from constant to stage_schedule between runs)."""
        try:
            super()._load_optimizer_and_scheduler(checkpoint)
        except (KeyError, TypeError, ValueError) as e:
            # Scheduler state from checkpoint is incompatible — load optimizer only
            if is_main_process():
                rank_print(f"[WARN] Scheduler state incompatible ({e}), loading optimizer only. "
                           f"Scheduler will be recreated by stage_schedule.")
            import os
            if checkpoint:
                sched_path = os.path.join(checkpoint, "scheduler.pt")
                sched_bak = sched_path + ".bak"
                if os.path.exists(sched_path):
                    os.rename(sched_path, sched_bak)
                    try:
                        super()._load_optimizer_and_scheduler(checkpoint)
                    finally:
                        os.rename(sched_bak, sched_path)

        # Accumulate across micro-batches within one optimizer step (for grad_acc > 1)
        self._step_samples = 0
        self._step_tokens = 0

        # Training timing
        self._train_timing = {
            "data_wait": 0.0,
            "total_step": 0.0,
            "steps": 0,
        }
        self._step_start_time: Optional[float] = None
        self._step_end_time: Optional[float] = None

        # Import RoPE implementation (try multiple model backends)
        self._apply_rope = None
        for rope_module in [
            'transformers.models.qwen3.modeling_qwen3',
            'transformers.models.gpt_neox.modeling_gpt_neox',
            'transformers.models.llama.modeling_llama',
        ]:
            try:
                mod = __import__(rope_module, fromlist=['apply_rotary_pos_emb'])
                self._apply_rope = mod.apply_rotary_pos_emb
                break
            except (ImportError, AttributeError):
                continue

        # Cached model internals (populated on first compute_loss call)
        self._cached_model_parts = None

    def get_train_dataloader(self):
        """Bypass Accelerate's dataloader wrapping --use TrainingArguments settings."""
        from torch.utils.data import DataLoader
        nw = self.args.dataloader_num_workers
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=False,  # idx provides randomness via seeding
            collate_fn=lambda x: x[0],  # Unwrap single-item batch
            num_workers=nw,
            prefetch_factor=self.args.dataloader_prefetch_factor if nw > 0 else None,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=nw > 0,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        t0 = time.perf_counter()
        # Measure data wait: time between end of last step and start of this one
        if self._step_end_time is not None:
            self._train_timing["data_wait"] += t0 - self._step_end_time
        loss = super().training_step(model, inputs, num_items_in_batch)
        t1 = time.perf_counter()
        self._train_timing["total_step"] += t1 - t0
        self._train_timing["steps"] += 1
        self._step_end_time = t1
        return loss

    def _report_train_timing(self):
        n = self._train_timing["steps"]
        if n == 0 or not is_main_process():
            return

        data_ms = (self._train_timing["data_wait"] / n) * 1000   # Idle: waiting for dataloader
        compute_ms = (self._train_timing["total_step"] / n) * 1000  # Active: fwd + bwd + optim
        wall_ms = data_ms + compute_ms  # Total wall time per step

        data_pct = (data_ms / wall_ms) * 100 if wall_ms > 0 else 0

        print(
            f"\n[TRAIN-TIMING] Steps={n} | "
            f"DataWait={data_ms:.1f}ms ({data_pct:.1f}%) | "
            f"Compute={compute_ms:.1f}ms | "
            f"Wall={wall_ms:.1f}ms | "
            f"Throughput={1000 / wall_ms:.1f} steps/s\n"
        )

    def reset_timing(self):
        self._train_timing = {
            "data_wait": 0.0,
            "total_step": 0.0,
            "steps": 0,
        }
        self._step_start_time = None
        self._step_end_time = None

    def _forward_layer_varlen(self, layer, hidden_states, cu_seqlens, max_seqlen,
                              num_heads, num_kv_heads, head_dim, cos, sin,
                              arch='qwen', parallel_residual=False, rotary_ndims=None):
        """Forward one layer using flash_attn_varlen_func. Supports Qwen and GPT-NeoX."""

        seq_len = hidden_states.shape[0]

        residual = hidden_states
        ln1_out = layer.input_layernorm(hidden_states)

        # QKV projection
        if arch == 'gpt_neox':
            # GPT-NeoX uses interleaved QKV: [q1|k1|v1 | q2|k2|v2 | ...] per head
            qkv = layer.attention.query_key_value(ln1_out)  # [seq, 3 * H]
            qkv = qkv.view(seq_len, num_heads, 3 * head_dim)
            q = qkv[..., :head_dim]              # [seq, num_heads, head_dim]
            k = qkv[..., head_dim:2*head_dim]
            v = qkv[..., 2*head_dim:]
        else:
            q = layer.self_attn.q_proj(ln1_out)
            k = layer.self_attn.k_proj(ln1_out)
            v = layer.self_attn.v_proj(ln1_out)
            # Reshape: [seq_len, num_heads, head_dim]
            q = q.view(seq_len, num_heads, head_dim)
            k = k.view(seq_len, num_kv_heads, head_dim)
            v = v.view(seq_len, num_kv_heads, head_dim)

        # QK Normalization (Qwen3 specific)
        if arch == 'qwen':
            if hasattr(layer.self_attn, 'q_norm') and layer.self_attn.q_norm is not None:
                q = layer.self_attn.q_norm(q)
                k = layer.self_attn.k_norm(k)

        # Add batch dim and transpose for RoPE: [1, num_heads, seq_len, head_dim]
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)

        # Handle partial RoPE (e.g. Pythia rotary_pct=0.25)
        partial_rope = rotary_ndims is not None and rotary_ndims < head_dim
        if partial_rope:
            q_rot, q_pass = q[..., :rotary_ndims], q[..., rotary_ndims:]
            k_rot, k_pass = k[..., :rotary_ndims], k[..., rotary_ndims:]
        else:
            q_rot, k_rot = q, k

        # Apply pre-computed RoPE cos/sin
        if self._apply_rope is not None:
            q_rot, k_rot = self._apply_rope(q_rot, k_rot, cos, sin)
        else:
            cos_u = cos.unsqueeze(1)
            sin_u = sin.unsqueeze(1)
            q_rot = (q_rot * cos_u) + (self._rotate_half(q_rot) * sin_u)
            k_rot = (k_rot * cos_u) + (self._rotate_half(k_rot) * sin_u)

        if partial_rope:
            q = torch.cat((q_rot, q_pass), dim=-1)
            k = torch.cat((k_rot, k_pass), dim=-1)
        else:
            q, k = q_rot, k_rot

        # Reshape for flash_attn_varlen: [seq_len, num_heads, head_dim]
        q = q.squeeze(0).transpose(0, 1).contiguous()
        k = k.squeeze(0).transpose(0, 1).contiguous()
        v = v.contiguous()

        # Flash attention
        attn_output = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )

        # Output projection
        attn_output = attn_output.reshape(seq_len, num_heads * head_dim)
        if arch == 'gpt_neox':
            attn_output = layer.attention.dense(attn_output)
        else:
            attn_output = layer.self_attn.o_proj(attn_output)

        # Residual + MLP
        if parallel_residual:
            # GPT-NeoX parallel: x = x + attn(ln1(x)) + mlp(ln2(x))
            ln2_out = layer.post_attention_layernorm(residual)
            mlp_output = layer.mlp(ln2_out)
            hidden_states = residual + attn_output + mlp_output
        else:
            # Sequential (Qwen/Llama): x = x + attn(ln1(x)); x = x + mlp(ln2(x))
            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _get_model_parts(self, model):
        """Cache model internals on first call to avoid repeated unwrapping."""
        if self._cached_model_parts is not None:
            return self._cached_model_parts
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        try:
            from peft import PeftModel
            if isinstance(unwrapped, PeftModel):
                unwrapped = unwrapped.base_model.model
        except ImportError:
            pass
        config = unwrapped.config

        # Auto-detect architecture
        if hasattr(unwrapped, 'gpt_neox'):
            # GPT-NeoX (Pythia)
            inner = unwrapped.gpt_neox
            arch = 'gpt_neox'
            embed = inner.embed_in
            norm = inner.final_layer_norm
            lm_head = unwrapped.embed_out
            parallel_residual = getattr(config, 'use_parallel_residual', True)
        else:
            # Qwen / Llama-style
            inner = unwrapped.model
            if not hasattr(inner, 'embed_tokens') and hasattr(inner, 'model'):
                inner = inner.model
            arch = 'qwen'
            embed = inner.embed_tokens
            norm = inner.norm
            lm_head = unwrapped.lm_head
            parallel_residual = False

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rotary_pct = getattr(config, 'rotary_pct', 1.0)
        rotary_ndims = int(head_dim * rotary_pct) if rotary_pct < 1.0 else head_dim

        self._cached_model_parts = {
            'arch': arch,
            'embed': embed,
            'layers': inner.layers,
            'norm': norm,
            'rotary_emb': inner.rotary_emb,
            'lm_head': lm_head,
            'num_heads': config.num_attention_heads,
            'num_kv_heads': getattr(config, "num_key_value_heads", config.num_attention_heads),
            'head_dim': head_dim,
            'rotary_ndims': rotary_ndims,
            'parallel_residual': parallel_residual,
        }
        return self._cached_model_parts

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract metadata
        sequence_info = inputs.pop("sequence_info")
        num_sequences = inputs.pop("num_sequences")
        cu_seqlens_list = inputs.pop("cu_seqlens_list")
        max_seqlen_list = inputs.pop("max_seqlen_list")
        efficiency = inputs.pop("_efficiency", None)

        self._last_efficiency = efficiency
        self._last_batch_samples = num_sequences
        self._last_batch_tokens = sum(cu[-1] for cu in cu_seqlens_list)

        self._step_samples += self._last_batch_samples
        self._step_tokens += self._last_batch_tokens

        input_ids = inputs["input_ids"]   # [1, total_tokens]
        labels_pad = inputs["labels"]     # [1, total_tokens]
        position_ids = inputs["position_ids"]  # [1, total_tokens]

        B, T = input_ids.shape
        device = input_ids.device

        use_checkpoint = getattr(self.args, 'gradient_checkpointing', False)

        # Cached model internals (avoids unwrapping every step)
        mp = self._get_model_parts(model)
        embed = mp['embed']
        layers = mp['layers']
        norm = mp['norm']
        rotary_emb = mp['rotary_emb']
        lm_head = mp['lm_head']
        num_heads = mp['num_heads']
        num_kv_heads = mp['num_kv_heads']
        head_dim = mp['head_dim']
        arch = mp['arch']
        parallel_residual = mp['parallel_residual']
        rotary_ndims = mp['rotary_ndims']

        # Single row, no padding — directly use the tensors
        flat_ids = input_ids[0]           # [total_tokens]
        all_labels = labels_pad[0]        # [total_tokens]
        all_pos = position_ids[0:1]       # [1, total_tokens]
        merged_cu = torch.tensor(cu_seqlens_list[0], dtype=torch.int32, device=device)
        global_max_seqlen = max_seqlen_list[0]

        # Custom layer-by-layer forward with flash_attn_varlen_func
        # Embed tokens
        h = embed(flat_ids)  # [total_tokens, H]

        # Compute RoPE cos/sin once (same for all layers)
        cos, sin = rotary_emb(h, all_pos)

        for layer in layers:
            if use_checkpoint and model.training:
                h = checkpoint.checkpoint(
                    self._forward_layer_varlen,
                    layer, h, merged_cu, global_max_seqlen,
                    num_heads, num_kv_heads, head_dim,
                    cos, sin, arch, parallel_residual, rotary_ndims,
                    use_reentrant=False,
                )
            else:
                h = self._forward_layer_varlen(
                    layer, h, merged_cu, global_max_seqlen,
                    num_heads, num_kv_heads, head_dim,
                    cos, sin, arch, parallel_residual, rotary_ndims,
                )

        h = norm(h)          # [total_tokens, H]

        # Shift for autoregressive loss (per-sequence boundaries are masked by -100 labels)
        shift_h = h[:-1, :]           # [Tm1, H]
        shift_labels = all_labels[1:]  # [Tm1]
        Tm1 = shift_h.size(0)

        valid_mask = (shift_labels != -100)
        n_valid = valid_mask.sum().item()

        if n_valid > 0:
            # Pre-compute global first_idx and answer spans for all sequences
            S = len(sequence_info)
            first_indices = torch.zeros(S, dtype=torch.long, device=device)
            seq_starts = torch.zeros(S, dtype=torch.long, device=device)
            seq_ends = torch.zeros(S, dtype=torch.long, device=device)
            is_search = torch.zeros(S, dtype=torch.bool, device=device)
            for si, seq in enumerate(sequence_info):
                pl = seq["prompt_len"]
                is_search[si] = pl > 0  # pretrain samples have prompt_len=0
                first_indices[si] = seq["start_idx"] + max(pl, 1) - 1
                seq_starts[si] = seq["start_idx"] + max(pl, 1) - 1
                seq_ends[si] = min(seq["end_idx"] - 1, Tm1)

            first_in_range = (first_indices >= 0) & (first_indices < Tm1) & is_search
            first_valid = first_in_range & valid_mask[first_indices.clamp(0, Tm1 - 1)]

            # Chunked lm_head to avoid OOM (full [Tm1, V] logits would be ~65 GiB)
            chunk_size = self.ce_chunk_size or 4096
            ce_parts = []
            preds = torch.empty(Tm1, dtype=torch.long, device=device)

            for cs in range(0, Tm1, chunk_size):
                ce_end = min(cs + chunk_size, Tm1)
                chunk_logits = lm_head(shift_h[cs:ce_end])  # [chunk, V]
                preds[cs:ce_end] = chunk_logits.detach().argmax(dim=-1)
                chunk_vm = valid_mask[cs:ce_end]
                if chunk_vm.any():
                    ce_parts.append(F.cross_entropy(
                        chunk_logits[chunk_vm],
                        shift_labels[cs:ce_end][chunk_vm],
                        reduction='none'
                    ))
                del chunk_logits

            ce = torch.cat(ce_parts)

            # Apply soft first-token CE adjustments
            if self.first_token_soft_weight > 0 and first_valid.any():
                ce_indices = torch.cumsum(valid_mask.int(), dim=0) - 1  # [Tm1]
                valid_fi = first_indices[first_valid]
                valid_ce = ce_indices[valid_fi]
                fi_logits = lm_head(shift_h[valid_fi])  # [N_first, V]
                batch_logp = F.log_softmax(fi_logits, dim=-1)
                del fi_logits

                w = self.first_token_soft_weight
                ce_list = valid_ce.tolist()
                valid_seq_indices = torch.nonzero(first_valid, as_tuple=True)[0].tolist()
                for j, si in enumerate(valid_seq_indices):
                    vf = sequence_info[si]["valid_first_targets"]
                    if not vf:
                        continue
                    ids = torch.tensor(vf, device=device, dtype=torch.long)
                    soft_ce = -batch_logp[j, ids].mean()
                    ci = ce_list[j]
                    ce[ci] = w * soft_ce + (1.0 - w) * ce[ci]

            loss = ce.sum() / n_valid

            # Track separate search vs pretrain loss for logging
            with torch.no_grad():
                # Build per-token is_search mask (shifted by 1 for autoregressive)
                token_is_search = torch.zeros(Tm1, dtype=torch.bool, device=device)
                for si in range(S):
                    if is_search[si]:
                        s, e = int(seq_starts[si]), int(seq_ends[si])
                        if s < e:
                            token_is_search[s:e] = True
                search_valid = valid_mask & token_is_search
                pretrain_valid = valid_mask & ~token_is_search
                ce_idx = torch.cumsum(valid_mask.int(), dim=0) - 1
                n_search = search_valid.sum().item()
                n_pretrain = pretrain_valid.sum().item()
                self._last_search_loss = ce[ce_idx[search_valid]].mean().item() if n_search > 0 else None
                self._last_pretrain_loss = ce[ce_idx[pretrain_valid]].mean().item() if n_pretrain > 0 else None

            # Track accuracy
            # Batch GPU->CPU transfers to minimize sync points
            with torch.no_grad():
                pred_matches = (preds == shift_labels)
                # Single GPU->CPU transfers (4 syncs instead of ~384)
                first_pred_list = preds[first_indices[first_valid].clamp(0, Tm1 - 1)].tolist()
                first_si_list = torch.nonzero(first_valid, as_tuple=True)[0].tolist()
                starts = seq_starts.tolist()
                ends = seq_ends.tolist()
                pred_matches_cpu = pred_matches.cpu()
                valid_mask_cpu = valid_mask.cpu()

                # First-token accuracy (pure Python loop, no GPU sync)
                for j, si in enumerate(first_si_list):
                    self.first_token_correct.append(first_pred_list[j] in sequence_info[si]["valid_first_targets"])

                # Full-word accuracy (pure Python loop on CPU tensors)
                is_search_cpu = is_search.cpu()
                for si in range(S):
                    if not is_search_cpu[si]:
                        continue  # skip pretrain samples
                    s, e = starts[si], ends[si]
                    if s >= e:
                        continue
                    span_valid = valid_mask_cpu[s:e]
                    if span_valid.any():
                        self.full_word_correct.append(pred_matches_cpu[s:e][span_valid].all().item())
                    # If no valid labels in span, skip (don't inflate accuracy)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        self.recent_losses.append(loss.item())

        return loss

    def get_first_token_acc(self):
        return (sum(self.first_token_correct) / len(self.first_token_correct)) if self.first_token_correct else 0.0

    def get_full_word_acc(self):
        return (sum(self.full_word_correct) / len(self.full_word_correct)) if self.full_word_correct else 0.0
@torch.no_grad()
def run_eval_tf_loss(
        model,
        tokenizer,
        task: str,
        inputs: List[str],
        labels: List[List[str]],
        **kwargs,
) -> float:
    """Compute average teacher-forced CE loss on eval set (distributed)."""
    barrier()
    rank = get_rank()
    world_size = get_world_size()
    device = next(model.parameters()).device

    total_len = len(inputs)
    chunk_size = math.ceil(total_len / world_size)
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, total_len)

    my_inputs = inputs[start_idx:end_idx]
    my_labels = labels[start_idx:end_idx]
    actual_count = len(my_inputs)

    rng = random.Random((kwargs.get("seed", 0) or 0) + 777)
    num_shots = kwargs.get("num_shots", 0)
    max_input_size = kwargs.get("max_input_size", 256)

    local_loss_sum = 0.0
    local_count = 0
    unwrapped = model.module if hasattr(model, "module") else model

    use_chat_template = kwargs.get("use_chat_template", False)

    for x, ys in zip(my_inputs, my_labels):
        ys = ys if isinstance(ys, list) else [ys]
        chosen = rng.choice(ys)
        task_type = _determine_task_type(task, x)

        if use_chat_template:
            msgs = [
                {"role": "user", "content": x},
                {"role": "assistant", "content": chosen},
            ]
            full_ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
                enable_thinking=False,
            )
            prefix_ids = tokenizer.apply_chat_template(
                msgs[:1], tokenize=True, add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = full_ids[:len(prefix_ids)]
            ans_ids = full_ids[len(prefix_ids):]
            input_ids = torch.tensor([full_ids], device=device)
            label_ids = torch.tensor([[-100] * len(prompt_ids) + ans_ids], device=device)
        else:
            prompt_ids = tokenizer(x, add_special_tokens=True, truncation=False)["input_ids"]
            ans_ids = _tokenize_leading_space(tokenizer, chosen)
            end_ids = tokenizer(_get_end_tokens(task_type), add_special_tokens=False)["input_ids"]

            input_ids = torch.tensor([prompt_ids + ans_ids + end_ids], device=device)
            label_ids = torch.tensor([[-100] * len(prompt_ids) + ans_ids + end_ids], device=device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = unwrapped(input_ids=input_ids, labels=label_ids)

        local_loss_sum += out.loss.item()
        local_count += 1

    metrics = torch.tensor([local_loss_sum, float(local_count)], dtype=torch.float64, device=device)
    if dist_is_initialized():
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)

    total_loss = metrics[0].item()
    total_count = int(metrics[1].item())

    barrier()
    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    rank_print(f"[EVAL-TF] Loss={avg_loss:.4f} (n={total_count})")
    return avg_loss


@torch.no_grad()
def run_eval_greedy_readable(
        model,
        tokenizer,
        task: str,
        inputs: List[str],
        labels: List[List[str]],
        print_examples: int = 0,
        use_chat_template: bool = False,
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

    # 2. Inference Loop
    for i, (x, ys) in enumerate(zip(my_inputs, my_labels)):
        is_padding = (i >= actual_count)
        ys = ys if isinstance(ys, list) else [ys]

        # Tokenize
        if use_chat_template:
            msgs = [{"role": "user", "content": x}]
            input_ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_tensors="pt",
            )
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor([input_ids], dtype=torch.long)
            enc = {"input_ids": input_ids.to(device), "attention_mask": torch.ones_like(input_ids).to(device)}
        else:
            enc = tokenizer(x, return_tensors="pt").to(device)
        prompt_len = enc["input_ids"].shape[1]

        # Generate (Must run for everyone)
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


class FirstTokenCurriculum(TrainerCallback):
    def __init__(
            self,
            dataset,
            n_stages: int,
            accuracy_threshold: float,
            min_steps_per_stage: int,
            check_every: int,
            use_packing: bool = False,  # NEW
            # Stage eval config
            do_stage_eval: bool = False,
            stage_eval_every: int = 1,  # Run stage eval every N stage advancements (1=every stage)
            eval_every_steps: int = 0,
            eval_inputs_hard: List[str] = None,
            eval_labels_hard: List[List[str]] = None,
            eval_fingerprint_hard: str = None,
            tokenizer=None,
            task: str = None,
            task_kwargs: dict = None,
            num_shots: int = 0,
            max_input_size: int = 256,
            seed: int = None,
            persist_every: int = 2000,
            print_examples: int = 0,
            lr_reset_on_stage: bool = False,
            lr_reset_warmup: int = 50,
            peak_lr: float = 1e-4,
            stage_schedule: str = "none",
            cosine_t_max: int = 3000,
            cosine_t0: int = 10000,
            cosine_t_mult: int = 2,
            cosine_eta_min_ratio: float = 0.01,
            batch_increase_factor: float = 2.0,
            lr_spike_factor: float = 5.0,
            lr_spike_steps: int = 200,
            plateau_spike: bool = False,
            plateau_action: str = "lr_spike",  # "lr_spike" or "batch_increase"
            plateau_window: int = 5000,
            plateau_threshold: float = 0.02,
            plateau_cooldown: int = 10000,
    ):
        self.dataset = dataset
        self.n_stages = n_stages
        self.acc_thr = accuracy_threshold
        self.min_steps = min_steps_per_stage
        self.check_every = check_every
        self.use_packing = use_packing
        self.trainer: Optional[PackedSequenceTrainer] = None
        self.stage_start_step = 0
        self._last_log = -1
        self.finished = False
        # Backward compat: --lr_reset_on_stage maps to stage_schedule=warmup_reset
        if lr_reset_on_stage and stage_schedule == "none":
            stage_schedule = "warmup_reset"
        self.stage_schedule = stage_schedule
        self.lr_reset_warmup = lr_reset_warmup
        self.peak_lr = peak_lr
        self.lr_reset_step = None  # global_step when LR was last reset/changed
        # Cosine decay params
        self.cosine_t_max = cosine_t_max
        self.cosine_t0 = cosine_t0
        self.cosine_t_mult = cosine_t_mult
        self.cosine_eta_min_ratio = cosine_eta_min_ratio
        # Batch increase params
        self.batch_increase_factor = batch_increase_factor
        self.batch_increase_count = 0  # how many times we've increased
        # LR spike params
        self.lr_spike_factor = lr_spike_factor
        self.lr_spike_steps = lr_spike_steps
        # Plateau-triggered params
        self.plateau_spike = plateau_spike
        self.plateau_action = plateau_action
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.plateau_cooldown = plateau_cooldown
        self.plateau_last_spike_step = -plateau_cooldown  # allow spike from the start
        self.plateau_acc_history = []  # list of (step, full_word_acc)

        # Speed tracking
        self.stage_start_time = None
        self.samples_this_stage = 0  # Track actual samples for token budget mode
        self.tokens_this_stage = 0  # Track actual tokens

        # Capture grad_norm and lr from Trainer logs
        self.last_grad_norm = 0.0
        self.last_lr = 0.0

        # Stage eval config
        self.do_stage_eval = do_stage_eval
        self.stage_eval_every = stage_eval_every
        self.eval_every_steps = eval_every_steps
        self.eval_inputs_hard = eval_inputs_hard
        self.eval_labels_hard = eval_labels_hard
        self.eval_fingerprint_hard = eval_fingerprint_hard
        self.tokenizer = tokenizer
        self.task = task
        self.task_kwargs = task_kwargs or {}
        self.num_shots = num_shots
        self.max_input_size = max_input_size
        self.seed = seed
        self.print_examples = print_examples
        self.stage_eval_history = []

        # Loss tracking
        self.loss_history = []
        # Flops tracking
        self.flops_per_token = None
        self.cumulative_wall_time = 0.0  # Total wall time across resumes (seconds)
        self.wall_time_offset = 0.0  # Offset from previous runs
        self.training_start_time = None  # Set when training starts
        self.plot_metadata = None  # Set after trainer creation
        self.persist_every = persist_every
        self._last_persist_step = 0

        # Resume tracking (resume markers are real entries with "resume" flag, not fake data)
        self.resume_points = []  # List of step numbers where resumes occurred
        self._mark_next_as_resume = False  # Flag to mark next entry as resume point

        # JSONL incremental persistence (survives preemption between checkpoint saves)
        self._jsonl_buffer = []
        self._jsonl_flush_every = 10  # Flush to disk every N entries

    def _check_plateau_spike(self, state):
        """Check if accuracy has plateaued, and if so spike LR temporarily."""
        if not self.plateau_spike:
            return
        # Don't spike if we're already in a spike (lr_spike schedule active)
        if self.lr_reset_step is not None and self.stage_schedule == "lr_spike":
            steps_since = state.global_step - self.lr_reset_step
            if steps_since < self.lr_spike_steps:
                return
        # Cooldown check
        if state.global_step - self.plateau_last_spike_step < self.plateau_cooldown:
            return
        # Record current accuracy
        fw = self.trainer.get_full_word_acc()
        self.plateau_acc_history.append((state.global_step, fw))
        # Need enough history
        if len(self.plateau_acc_history) < 2:
            return
        # Check if we have data spanning plateau_window steps
        oldest_step = self.plateau_acc_history[0][0]
        if state.global_step - oldest_step < self.plateau_window:
            return
        # Trim history older than plateau_window
        cutoff = state.global_step - self.plateau_window
        while self.plateau_acc_history and self.plateau_acc_history[0][0] < cutoff:
            self.plateau_acc_history.pop(0)
        if len(self.plateau_acc_history) < 2:
            return
        # Compare: best acc in first half vs best acc in second half
        mid_step = self.plateau_acc_history[0][0] + self.plateau_window // 2
        first_half = [a for s, a in self.plateau_acc_history if s < mid_step]
        second_half = [a for s, a in self.plateau_acc_history if s >= mid_step]
        if not first_half or not second_half:
            return
        best_first = max(first_half)
        best_second = max(second_half)
        improvement = best_second - best_first
        if improvement < self.plateau_threshold:
            # Plateau detected
            self.plateau_last_spike_step = state.global_step
            self.plateau_acc_history.clear()  # reset history after action

            if is_main_process():
                print(f"[PLATEAU] Plateau detected! acc improvement={improvement:.4f} < {self.plateau_threshold} "
                      f"over {self.plateau_window} steps (best_first={best_first:.2%}, best_second={best_second:.2%})")

            if self.plateau_action == "lr_spike":
                peak_lr = self.peak_lr
                spike_factor = self.lr_spike_factor
                spike_steps = self.lr_spike_steps
                from torch.optim.lr_scheduler import LambdaLR

                def lr_lambda(current_step):
                    if current_step < spike_steps // 2:
                        t = float(current_step) / float(max(1, spike_steps // 2))
                        return 1.0 + (spike_factor - 1.0) * t
                    elif current_step < spike_steps:
                        t = float(current_step - spike_steps // 2) / float(max(1, spike_steps - spike_steps // 2))
                        return spike_factor - (spike_factor - 1.0) * t
                    else:
                        return 1.0

                for pg in self.trainer.optimizer.param_groups:
                    pg['lr'] = peak_lr
                    pg['initial_lr'] = peak_lr
                self.trainer.lr_scheduler = LambdaLR(self.trainer.optimizer, lr_lambda, last_epoch=-1)
                self.lr_reset_step = state.global_step
                if is_main_process():
                    max_lr = peak_lr * spike_factor
                    print(f"[PLATEAU] Action: LR spike {peak_lr:.2e}→{max_lr:.2e}→{peak_lr:.2e} "
                          f"over {spike_steps} steps at step {state.global_step}")

            elif self.plateau_action == "batch_increase":
                old_ga = self.trainer.args.gradient_accumulation_steps
                new_ga = max(1, int(old_ga * self.batch_increase_factor))
                if new_ga != old_ga:
                    self.trainer.args.gradient_accumulation_steps = new_ga
                    self.batch_increase_count += 1
                    if is_main_process():
                        eff_batch = self.trainer.args.per_device_train_batch_size * new_ga * max(1, torch.cuda.device_count())
                        print(f"[PLATEAU] Action: batch increase grad_acc {old_ga}→{new_ga} "
                              f"(eff_batch≈{eff_batch}, increase #{self.batch_increase_count}) "
                              f"at step {state.global_step}")
                elif is_main_process():
                    print(f"[PLATEAU] Action: batch_increase requested but grad_acc unchanged at {old_ga}")

    def _apply_stage_schedule(self, state):
        """Apply LR/batch schedule strategy on stage advance."""
        strategy = self.stage_schedule
        peak_lr = self.peak_lr

        if strategy == "warmup_reset":
            # Warmup from 0 to peak_lr, then hold constant
            from torch.optim.lr_scheduler import LambdaLR
            warmup = self.lr_reset_warmup

            def lr_lambda(current_step):
                if current_step < warmup:
                    return float(current_step) / float(max(1, warmup))
                return 1.0

            for pg in self.trainer.optimizer.param_groups:
                pg['lr'] = peak_lr
                pg['initial_lr'] = peak_lr
            self.trainer.lr_scheduler = LambdaLR(self.trainer.optimizer, lr_lambda, last_epoch=-1)
            self.lr_reset_step = state.global_step
            if is_main_process():
                print(f"[STAGE-SCHED] warmup_reset: LR→{peak_lr} with {warmup}-step warmup at step {state.global_step}")

        elif strategy == "cosine_restart":
            # CosineAnnealingLR — single cosine decay per stage, reset on advance
            from torch.optim.lr_scheduler import CosineAnnealingLR
            eta_min = peak_lr * self.cosine_eta_min_ratio

            for pg in self.trainer.optimizer.param_groups:
                pg['lr'] = peak_lr
                pg['initial_lr'] = peak_lr
            self.trainer.lr_scheduler = CosineAnnealingLR(
                self.trainer.optimizer,
                T_max=self.cosine_t_max,
                eta_min=eta_min,
            )
            self.lr_reset_step = state.global_step
            if is_main_process():
                print(f"[STAGE-SCHED] cosine_decay: T_max={self.cosine_t_max}, "
                      f"eta_min={eta_min:.2e}, peak={peak_lr:.2e} at step {state.global_step}")

        elif strategy == "cosine_sgdr":
            # CosineAnnealingWarmRestarts (SGDR) — cyclic cosine with periodic LR resets
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            eta_min = peak_lr * self.cosine_eta_min_ratio

            for pg in self.trainer.optimizer.param_groups:
                pg['lr'] = peak_lr
                pg['initial_lr'] = peak_lr
            self.trainer.lr_scheduler = CosineAnnealingWarmRestarts(
                self.trainer.optimizer,
                T_0=self.cosine_t0,
                T_mult=self.cosine_t_mult,
                eta_min=eta_min,
            )
            self.lr_reset_step = state.global_step
            if is_main_process():
                print(f"[STAGE-SCHED] cosine_sgdr: T_0={self.cosine_t0}, T_mult={self.cosine_t_mult}, "
                      f"eta_min={eta_min:.2e}, peak={peak_lr:.2e} at step {state.global_step}")

        elif strategy == "batch_increase":
            # Increase gradient accumulation steps (simulates larger batch)
            old_ga = self.trainer.args.gradient_accumulation_steps
            new_ga = max(1, int(old_ga * self.batch_increase_factor))
            if new_ga != old_ga:
                self.trainer.args.gradient_accumulation_steps = new_ga
                self.batch_increase_count += 1
                if is_main_process():
                    eff_batch = self.trainer.args.per_device_train_batch_size * new_ga * max(1, torch.cuda.device_count())
                    print(f"[STAGE-SCHED] batch_increase: grad_acc {old_ga}→{new_ga} "
                          f"(eff_batch≈{eff_batch}) at step {state.global_step}")

        elif strategy == "lr_spike":
            # Mini 1-cycle: spike LR up then decay back to peak
            from torch.optim.lr_scheduler import LambdaLR
            spike_factor = self.lr_spike_factor
            spike_steps = self.lr_spike_steps

            def lr_lambda(current_step):
                if current_step < spike_steps // 2:
                    # Phase 1: ramp up from 1.0 to spike_factor
                    t = float(current_step) / float(max(1, spike_steps // 2))
                    return 1.0 + (spike_factor - 1.0) * t
                elif current_step < spike_steps:
                    # Phase 2: ramp down from spike_factor to 1.0
                    t = float(current_step - spike_steps // 2) / float(max(1, spike_steps - spike_steps // 2))
                    return spike_factor - (spike_factor - 1.0) * t
                else:
                    # After spike: constant at peak
                    return 1.0

            for pg in self.trainer.optimizer.param_groups:
                pg['lr'] = peak_lr
                pg['initial_lr'] = peak_lr
            self.trainer.lr_scheduler = LambdaLR(self.trainer.optimizer, lr_lambda, last_epoch=-1)
            self.lr_reset_step = state.global_step
            if is_main_process():
                max_lr = peak_lr * spike_factor
                print(f"[STAGE-SCHED] lr_spike: {peak_lr:.2e}→{max_lr:.2e}→{peak_lr:.2e} "
                      f"over {spike_steps} steps at step {state.global_step}")

    def _restore_stage_schedule(self, state):
        """Restore LR scheduler after preemption resume (lambda not serialized)."""
        strategy = self.stage_schedule
        if strategy in ("warmup_reset", "cosine_restart", "cosine_sgdr", "lr_spike") and self.lr_reset_step is not None:
            peak_lr = self.peak_lr
            steps_since_reset = max(state.global_step - self.lr_reset_step, 0)

            for pg in self.trainer.optimizer.param_groups:
                pg['initial_lr'] = peak_lr

            if strategy == "warmup_reset":
                from torch.optim.lr_scheduler import LambdaLR
                warmup = self.lr_reset_warmup

                def lr_lambda(current_step):
                    if current_step < warmup:
                        return float(current_step) / float(max(1, warmup))
                    return 1.0

                new_sched = LambdaLR(self.trainer.optimizer, lr_lambda,
                                     last_epoch=max(steps_since_reset - 1, -1))
                self.trainer.lr_scheduler = new_sched

            elif strategy == "cosine_restart":
                from torch.optim.lr_scheduler import CosineAnnealingLR
                eta_min = peak_lr * self.cosine_eta_min_ratio
                new_sched = CosineAnnealingLR(
                    self.trainer.optimizer,
                    T_max=self.cosine_t_max,
                    eta_min=eta_min,
                    last_epoch=max(steps_since_reset - 1, -1),
                )
                self.trainer.lr_scheduler = new_sched

            elif strategy == "cosine_sgdr":
                from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
                eta_min = peak_lr * self.cosine_eta_min_ratio
                new_sched = CosineAnnealingWarmRestarts(
                    self.trainer.optimizer,
                    T_0=self.cosine_t0,
                    T_mult=self.cosine_t_mult,
                    eta_min=eta_min,
                    last_epoch=max(steps_since_reset - 1, -1),
                )
                self.trainer.lr_scheduler = new_sched

            elif strategy == "lr_spike":
                from torch.optim.lr_scheduler import LambdaLR
                spike_factor = self.lr_spike_factor
                spike_steps = self.lr_spike_steps

                def lr_lambda(current_step):
                    if current_step < spike_steps // 2:
                        t = float(current_step) / float(max(1, spike_steps // 2))
                        return 1.0 + (spike_factor - 1.0) * t
                    elif current_step < spike_steps:
                        t = float(current_step - spike_steps // 2) / float(max(1, spike_steps - spike_steps // 2))
                        return spike_factor - (spike_factor - 1.0) * t
                    else:
                        return 1.0

                new_sched = LambdaLR(self.trainer.optimizer, lr_lambda,
                                     last_epoch=max(steps_since_reset - 1, -1))
                self.trainer.lr_scheduler = new_sched

            current_lr = self.trainer.optimizer.param_groups[0]['lr']
            if is_main_process():
                rank_print(f"[STAGE-SCHED] Restored {strategy} from step {self.lr_reset_step} "
                          f"({steps_since_reset} steps ago), current lr={current_lr:.6e}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture grad_norm and lr from Transformers' logs"""
        if logs:
            self.last_grad_norm = logs.get('grad_norm', self.last_grad_norm)
            self.last_lr = logs.get('learning_rate', self.last_lr)
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize stage timing at training start.
        Note: samples_this_stage/tokens_this_stage are NOT reset here --
        they may have been restored from checkpoint by _try_restore_curriculum_state."""
        if self.stage_start_time is None:
            self.stage_start_time = datetime.datetime.now()

        # Track wall time
        if self.training_start_time is None:
            self.training_start_time = time.time()

        # Apply or restore stage schedule
        if self.stage_schedule != "none" and self.trainer.lr_scheduler is not None:
            if self.lr_reset_step is not None:
                # Resume: recreate scheduler at correct position
                self._restore_stage_schedule(state)
            else:
                # Fresh start: apply schedule from Stage 1
                self._apply_stage_schedule(state)

        # Restore batch_increase: reapply accumulated gradient_accumulation increases
        if self.stage_schedule == "batch_increase" and self.batch_increase_count > 0:
            base_ga = self.trainer.args.gradient_accumulation_steps
            new_ga = max(1, int(base_ga * (self.batch_increase_factor ** self.batch_increase_count)))
            if new_ga != base_ga:
                self.trainer.args.gradient_accumulation_steps = new_ga
                if is_main_process():
                    eff_batch = self.trainer.args.per_device_train_batch_size * new_ga * max(1, torch.cuda.device_count())
                    rank_print(f"[STAGE-SCHED] Restored batch_increase: grad_acc={new_ga} "
                              f"(increased {self.batch_increase_count}x, eff_batch≈{eff_batch})")

        return control

    def _current_metric(self) -> float:
        """Get synchronized full-word accuracy across all GPUs"""
        local_acc = self.trainer.get_full_word_acc()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = self.trainer.args.device
            metric_tensor = torch.tensor([local_acc], dtype=torch.float32, device=device)
            torch.distributed.all_reduce(metric_tensor, op=torch.distributed.ReduceOp.SUM)
            global_acc = metric_tensor.item() / torch.distributed.get_world_size()
            return global_acc

        return local_acc

    def on_save(self, args, state, control, **kwargs):
        """Save curriculum state alongside checkpoint"""
        barrier()
        current_wall_time = self.wall_time_offset
        if self.training_start_time is not None:
            current_wall_time += (time.time() - self.training_start_time)

        if is_main_process():
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            _save_curriculum_state(checkpoint_dir, self.dataset.stage, self.stage_start_step, current_wall_time,
                                   first_token_correct=self.trainer.first_token_correct,
                                   full_word_correct=self.trainer.full_word_correct,
                                   recent_losses=self.trainer.recent_losses,
                                   samples_this_stage=self.samples_this_stage,
                                   tokens_this_stage=self.tokens_this_stage,
                                   lr_reset_step=self.lr_reset_step,
                                   batch_increase_count=self.batch_increase_count if self.batch_increase_count else None,
                                   plateau_last_spike_step=self.plateau_last_spike_step if self.plateau_spike else None)

        if is_main_process():
            # Flush any buffered JSONL entries to disk
            if self._jsonl_buffer:
                try:
                    _append_to_jsonl(args.output_dir, self._jsonl_buffer)
                    self._jsonl_buffer = []
                except Exception as e:
                    rank_print(f"[LOSS][WARN] JSONL flush on save failed: {e}")

            # Atomic write of full loss_history.json (for backward compat / plotting)
            try:
                loss_path = os.path.join(args.output_dir, "loss_history.json")
                _atomic_write_json(loss_path, self.loss_history)
            except Exception as e:
                rank_print(f"[LOSS] Warning: Could not save loss history: {e}")

            # Save resume points
            if self.resume_points:
                try:
                    rp_path = os.path.join(args.output_dir, "resume_points.json")
                    _atomic_write_json(rp_path, self.resume_points)
                except Exception:
                    pass

        # Persistent checkpoint (not rotated by save_total_limit)
        if (self.persist_every > 0
                and state.global_step >= self._last_persist_step + self.persist_every):
            try:
                persist_dir = os.path.join(args.output_dir, "persistent_checkpoints",
                                           f"step_{state.global_step}")
                if is_main_process():
                    rank_print(f"[PERSIST] Saving persistent checkpoint to {persist_dir}")
                self.trainer.save_model(persist_dir)
                barrier()
                if is_main_process():
                    _save_curriculum_state(persist_dir, self.dataset.stage,
                                           self.stage_start_step, current_wall_time,
                                           first_token_correct=self.trainer.first_token_correct,
                                           full_word_correct=self.trainer.full_word_correct,
                                           recent_losses=self.trainer.recent_losses,
                                           samples_this_stage=self.samples_this_stage,
                                           tokens_this_stage=self.tokens_this_stage,
                                           lr_reset_step=self.lr_reset_step,
                                   batch_increase_count=self.batch_increase_count if self.batch_increase_count else None,
                                   plateau_last_spike_step=self.plateau_last_spike_step if self.plateau_spike else None)
                self._last_persist_step = state.global_step
            except Exception as e:
                rank_print(f"[PERSIST] Warning: Could not save persistent checkpoint: {e}")

        barrier()
        return control

    def _run_stage_eval(self, stage: int, global_step: int, is_periodic: bool = False):
        """Run greedy eval + TF loss at alpha=1.0 after stage advancement.
        If curriculum is enabled and this is not a periodic eval, also evaluate
        at the current stage's alpha to measure accuracy on stage-difficulty data."""
        if not self.do_stage_eval:
            return
        if self.eval_inputs_hard is None or self.tokenizer is None:
            rank_print("[STAGE-EVAL] Skipped (missing eval data or tokenizer)")
            return

        # Verify fingerprint
        fp = _eval_data_fingerprint(self.eval_inputs_hard, self.eval_labels_hard)
        if fp != self.eval_fingerprint_hard:
            rank_print(f"[STAGE-EVAL] WARNING: Fingerprint mismatch! {fp} != {self.eval_fingerprint_hard}")

        # Get effective L for this stage
        target_L = self.dataset._stage_target_lookahead()
        if target_L is not None:
            effective_L = target_L
        else:
            alpha = self.dataset._stage_alpha()
            n = getattr(self.dataset, "max_input_size", 256)
            cap = self.dataset.task_kwargs.get("max_lookahead")
            effective_L = effective_search_L(alpha, n, max_lookahead_cap=cap)

        rank_print(
            f"\n[STAGE-EVAL] Stage {stage} complete (step {global_step}, L={effective_L}), evaluating at alpha=1.0...")

        # Switch to eval mode
        model = self.trainer.model
        was_training = model.training
        model.eval()

        _use_chat = getattr(self.dataset, 'use_chat_template', False)

        with torch.no_grad():
            # Teacher-forced loss
            tf_loss = run_eval_tf_loss(
                model, self.tokenizer, self.task,
                self.eval_inputs_hard, self.eval_labels_hard,
                num_shots=self.num_shots, max_input_size=self.max_input_size,
                seed=self.seed, use_chat_template=_use_chat, **self.task_kwargs
            )

            # Greedy accuracy
            greedy_result = run_eval_greedy_readable(
                model, self.tokenizer, self.task,
                self.eval_inputs_hard, self.eval_labels_hard,
                num_shots=self.num_shots, max_input_size=self.max_input_size,
                seed=self.seed, print_examples=self.print_examples,
                use_chat_template=_use_chat, **self.task_kwargs
            )

            # Stage-alpha eval: generate and evaluate on stage-difficulty data
            stage_greedy_result = None
            stage_alpha = self.dataset._stage_alpha()
            max_alpha_L = self.dataset.task_kwargs.get("max_lookahead")
            is_full_difficulty = (target_L is not None and target_L >= (max_alpha_L or 256))
            if not is_full_difficulty and stage_alpha < 1.0:
                rank_print(f"[STAGE-EVAL] Also evaluating at stage alpha={stage_alpha:.4f} (L={effective_L})...")
                # Generate stage-difficulty eval data on the fly
                stage_eval_inputs, stage_eval_labels = None, None
                if is_main_process():
                    n_stage_samples = min(len(self.eval_inputs_hard), 500)
                    stage_eval_inputs, stage_eval_labels, _ = generate_eval_like_training(
                        n_samples=n_stage_samples,
                        task=self.task,
                        tokenizer=self.tokenizer,
                        max_input_size=self.max_input_size,
                        alpha=stage_alpha,
                        num_shots=self.num_shots,
                        reserved_inputs=set(),
                        seed=(self.seed or 0) + global_step,  # Vary seed per eval
                        **self.task_kwargs,
                    )
                barrier()
                stage_eval_inputs = broadcast_object(stage_eval_inputs, src=0)
                stage_eval_labels = broadcast_object(stage_eval_labels, src=0)

                if stage_eval_inputs:
                    stage_greedy_result = run_eval_greedy_readable(
                        model, self.tokenizer, self.task,
                        stage_eval_inputs, stage_eval_labels,
                        num_shots=self.num_shots, max_input_size=self.max_input_size,
                        seed=self.seed, print_examples=0,
                        use_chat_template=_use_chat, **self.task_kwargs
                    )

        # Restore training mode
        if was_training:
            model.train()

        # Log results
        log_msg = (
            f"[STAGE-EVAL] Stage {stage} | Step {global_step} | L={effective_L} | "
            f"TF Loss={tf_loss:.4f} | "
            f"Greedy: First={greedy_result['first_token_acc']:.2%}, Full={greedy_result['full_word_acc']:.2%}"
        )
        if stage_greedy_result:
            log_msg += (
                f"\n[STAGE-EVAL] Stage {stage} | Step {global_step} | L={effective_L} (stage-alpha) | "
                f"Greedy: First={stage_greedy_result['first_token_acc']:.2%}, Full={stage_greedy_result['full_word_acc']:.2%}"
            )
        rank_print(log_msg)

        # Store history
        entry = {
            "stage": stage,
            "step": global_step,
            "effective_L": effective_L,
            "alpha_training": stage_alpha,
            "tf_loss": tf_loss,
            "greedy_first": greedy_result['first_token_acc'],
            "greedy_full": greedy_result['full_word_acc'],
        }
        if stage_greedy_result:
            entry["stage_greedy_first"] = stage_greedy_result['first_token_acc']
            entry["stage_greedy_full"] = stage_greedy_result['full_word_acc']
        self.stage_eval_history.append(entry)

        # Save to file and plot
        if is_main_process():
            try:
                out_path = os.path.join(self.trainer.args.output_dir, "stage_eval_history.json")
                with open(out_path, "w") as f:
                    json.dump(self.stage_eval_history, f, indent=2)
                plot_stage_eval(self.stage_eval_history, self.trainer.args.output_dir, self.plot_metadata)
                plot_eval_acc_vs_step(self.stage_eval_history, self.trainer.args.output_dir, self.plot_metadata,
                                     loss_history=self.loss_history)
                plot_eval_acc_vs_flops(self.stage_eval_history, self.loss_history,
                                      self.trainer.args.output_dir, self.flops_per_token, self.plot_metadata)
            except Exception as e:
                rank_print(f"[STAGE-EVAL] Warning: Could not save history/plot: {e}")

    def on_step_end(self, args, state, control, **kwargs):
        # Early exit conditions
        if self.trainer is None or self.finished or state.global_step == 0:
            return control

        # Track loss history
        if self.trainer.recent_losses:
            current_loss = self.trainer.recent_losses[-1]

            # Get effective lookahead for search task
            effective_L = None
            if getattr(self.dataset, "task", None) == "search":
                target_L = self.dataset._stage_target_lookahead()
                if target_L is not None:
                    effective_L = target_L
                else:
                    alpha = self.dataset._stage_alpha()
                    n = getattr(self.dataset, "max_input_size", 256)
                    cap = self.dataset.task_kwargs.get("max_lookahead")
                    effective_L = effective_search_L(alpha, n, max_lookahead_cap=cap)

            # Calculate current wall time
            current_wall_time = self.wall_time_offset
            if self.training_start_time is not None:
                current_wall_time += (time.time() - self.training_start_time)

            # Calculate achieved TFLOPs/s (use accumulated tokens for full optimizer step)
            tokens_this_step = self.trainer._step_tokens * get_world_size() if hasattr(self.trainer,
                                                                                       '_step_tokens') else 0
            achieved_tflops = 0.0
            if hasattr(self.trainer, '_train_timing') and self.trainer._train_timing["steps"] > 0:
                recent_step_time = self.trainer._train_timing["total_step"] / self.trainer._train_timing["steps"]
                if recent_step_time > 0 and self.flops_per_token and tokens_this_step > 0:
                    flops_this_step = tokens_this_step * self.flops_per_token
                    achieved_tflops = flops_this_step / recent_step_time / 1e12

            entry = {
                "step": state.global_step,
                "loss": current_loss,
                "stage": self.dataset.stage,
                "alpha": self.dataset._stage_alpha(),
                "effective_L": effective_L,
                "tokens": tokens_this_step,
                "wall_time": current_wall_time,
                "achieved_tflops": achieved_tflops,
                "n_gpus": get_world_size(),
            }
            # Add separate search/pretrain losses if available
            if hasattr(self.trainer, '_last_search_loss') and self.trainer._last_search_loss is not None:
                entry["search_loss"] = self.trainer._last_search_loss
            if hasattr(self.trainer, '_last_pretrain_loss') and self.trainer._last_pretrain_loss is not None:
                entry["pretrain_loss"] = self.trainer._last_pretrain_loss

            # Mark as resume point if this is the first entry after a resume
            if self._mark_next_as_resume:
                entry["resume"] = True
                self._mark_next_as_resume = False

            self.loss_history.append(entry)

            # JSONL incremental persistence (survives preemption between checkpoints)
            if is_main_process():
                self._jsonl_buffer.append(entry)
                if len(self._jsonl_buffer) >= self._jsonl_flush_every:
                    try:
                        _append_to_jsonl(args.output_dir, self._jsonl_buffer)
                        self._jsonl_buffer = []
                    except Exception as e:
                        rank_print(f"[LOSS][WARN] JSONL flush failed: {e}")

        current_time = datetime.datetime.now()

        # ==================== Track Samples for Packing mode
        # Use accumulated values (correct with gradient_accumulation_steps > 1)
        if self.use_packing:
            if hasattr(self.trainer, '_step_samples'):
                self.samples_this_stage += self.trainer._step_samples * get_world_size()
            if hasattr(self.trainer, '_step_tokens'):
                self.tokens_this_stage += self.trainer._step_tokens * get_world_size()
            # Reset accumulators for next optimizer step
            self.trainer._step_samples = 0
            self.trainer._step_tokens = 0

        # ==================== Logging (every 10 steps) ====================
        if state.global_step % 10 == 0 and state.global_step != self._last_log and is_main_process():
            loss = np.mean(self.trainer.recent_losses) if self.trainer.recent_losses else 0.0
            f1 = self.trainer.get_first_token_acc()
            fw = self.trainer.get_full_word_acc()

            # Time in current stage
            stage_time_str = "0m"
            samples_per_sec = 0.0
            tokens_per_sec = 0.0

            if self.stage_start_time:
                stage_time = (current_time - self.stage_start_time).total_seconds()
                stage_time_str = f"{stage_time / 60:.1f}m"

                if self.use_packing:
                    # Use actual tracked samples
                    if stage_time > 0 and self.samples_this_stage > 0:
                        samples_per_sec = self.samples_this_stage / stage_time
                    if stage_time > 0 and self.tokens_this_stage > 0:
                        tokens_per_sec = self.tokens_this_stage / stage_time
                else:
                    # Fixed batch size mode
                    steps_in_stage = state.global_step - self.stage_start_step
                    if steps_in_stage > 0 and stage_time > 0:
                        batch_size = self.trainer.args.per_device_train_batch_size
                        world_size = get_world_size()
                        samples_per_sec = (steps_in_stage * batch_size * world_size) / stage_time

            # Extra info for packing mode
            extra_info = ""
            if self.use_packing and self.samples_this_stage > 0:
                steps_in_stage = state.global_step - self.stage_start_step
                if steps_in_stage > 0:
                    avg_seqs_per_step = self.samples_this_stage / steps_in_stage / get_world_size()
                    avg_tokens_per_step = self.tokens_this_stage / steps_in_stage / get_world_size() if self.tokens_this_stage > 0 else 0
                    extra_info = f" | seqs/step={avg_seqs_per_step:.1f} | toks/step={avg_tokens_per_step:.0f}"

            if self.use_packing:
                eff = getattr(self.trainer, '_last_efficiency', None)
                if eff is not None:
                    extra_info += f" | eff={eff:.1f}%"

            # Format tokens/s with K suffix for readability
            tokens_per_sec_str = f"{tokens_per_sec / 1000:.1f}K" if tokens_per_sec >= 1000 else f"{tokens_per_sec:.0f}"

            achieved_tflops_str = ""
            if self.flops_per_token and tokens_per_sec > 0:
                achieved_tflops = tokens_per_sec * self.flops_per_token / 1e12
                achieved_tflops_str = f" | {achieved_tflops:.1f} TFLOPs/s"

            # Calculate proper stage denominator
            if getattr(self.dataset, 'linear_lookahead', False) and self.dataset.task == "search":
                max_L = self.dataset.task_kwargs.get("max_lookahead", 12)
                if self.dataset.lookahead_step > 0:
                    expected_stages = math.ceil((max_L - self.dataset.base_lookahead) / self.dataset.lookahead_step) + 1
                else:
                    expected_stages = 1
                current_L = self.dataset._stage_target_lookahead()
                stage_str = f"[Stage {self.dataset.stage}/{expected_stages} L={current_L}]"
            else:
                stage_str = f"[Stage {self.dataset.stage}/{self.n_stages}]"

            print(f"{stage_str} step {state.global_step} | "
                  f"loss_avg={loss:.4f}({len(self.trainer.recent_losses)}) | First={f1:.2%} | Full={fw:.2%} | "
                  f"lr={self.last_lr:.2e} | grad_norm={self.last_grad_norm:.2f} | "
                  f"Speed={samples_per_sec:.1f} samples/s ({tokens_per_sec_str} toks/s){achieved_tflops_str} | "
                  f"Stage time={stage_time_str}{extra_info}")
            self._last_log = state.global_step

        # ==================== Periodic Eval (independent of stage advancement) ====================
        if self.eval_every_steps > 0 and state.global_step % self.eval_every_steps == 0:
            self._run_stage_eval(self.dataset.stage, state.global_step, is_periodic=True)
            if is_main_process():
                plot_overall_loss(self.loss_history, self.trainer.args.output_dir, self.n_stages, self.plot_metadata)
                plot_loss_vs_flops(self.loss_history, self.trainer.args.output_dir, self.flops_per_token, self.n_stages,
                                   self.plot_metadata)
                plot_loss_vs_walltime(self.loss_history, self.trainer.args.output_dir, self.n_stages,
                                      self.plot_metadata)
                plot_achieved_tflops(self.loss_history, self.trainer.args.output_dir, self.n_stages, self.plot_metadata)

        # ==================== Plateau Spike Check ====================
        if self.plateau_spike and state.global_step % self.check_every == 0:
            self._check_plateau_spike(state)

        # ==================== Stage Advancement Check ====================
        if state.global_step % self.check_every == 0 and (state.global_step - self.stage_start_step) >= self.min_steps:

            m = self._current_metric()

            if is_main_process():
                print(f"[CHECK] Stage {self.dataset.stage} full={m:.2%} target>={self.acc_thr:.2%}")

            barrier()

            if m >= self.acc_thr:
                old_stage = self.dataset.stage

                if is_main_process():
                    # Summary for completed stage
                    if self.stage_start_time:
                        total_stage_time = (current_time - self.stage_start_time).total_seconds()
                        steps_in_stage = state.global_step - self.stage_start_step

                        if self.use_packing:
                            samples_per_sec = self.samples_this_stage / total_stage_time if total_stage_time > 0 else 0
                            print(f"[COMPLETE] Stage {old_stage} complete in {steps_in_stage} steps, "
                                  f"{total_stage_time / 60:.1f} minutes | "
                                  f"{self.samples_this_stage:,} samples | "
                                  f"{samples_per_sec:.1f} samples/s")
                        else:
                            print(f"[COMPLETE] Stage {old_stage} complete in {steps_in_stage} steps, "
                                  f"{total_stage_time / 60:.1f} minutes")
                    else:
                        print(f"[COMPLETE] Stage {old_stage} complete")

                # Run stage eval BEFORE advancing (skip if not a multiple of stage_eval_every)
                target_L_for_eval = self.dataset._stage_target_lookahead()
                if self.stage_eval_every <= 1 or (target_L_for_eval and target_L_for_eval % self.stage_eval_every == 0):
                    self._run_stage_eval(old_stage, state.global_step)

                # Save persistent stage checkpoint (won't be rotated by save_total_limit)
                try:
                    target_L = self.dataset._stage_target_lookahead()
                    ckpt_L = target_L if target_L is not None else "?"
                    stage_ckpt_dir = os.path.join(
                        self.trainer.args.output_dir,
                        "stage_checkpoints",
                        f"stage_{old_stage}_step_{state.global_step}_L{ckpt_L}"
                    )
                    rank_print(f"[STAGE-CKPT] Saving persistent checkpoint to {stage_ckpt_dir}")
                    self.trainer.save_model(stage_ckpt_dir)
                    barrier()
                    if is_main_process():
                        _save_curriculum_state(stage_ckpt_dir, old_stage, self.stage_start_step,
                                               self.wall_time_offset + (
                                                           time.time() - self.training_start_time) if self.training_start_time else self.wall_time_offset,
                                               first_token_correct=self.trainer.first_token_correct,
                                               full_word_correct=self.trainer.full_word_correct,
                                               recent_losses=self.trainer.recent_losses,
                                               samples_this_stage=self.samples_this_stage,
                                               tokens_this_stage=self.tokens_this_stage)
                    rank_print(f"[STAGE-CKPT] Saved stage {old_stage} checkpoint")
                except Exception as e:
                    rank_print(f"[STAGE-CKPT] Warning: Could not save stage checkpoint: {e}")

                if is_main_process():
                    plot_stage_loss(self.loss_history, old_stage, self.trainer.args.output_dir, self.plot_metadata)
                    plot_overall_loss(self.loss_history, self.trainer.args.output_dir, self.n_stages,
                                      self.plot_metadata)
                    plot_loss_vs_flops(self.loss_history, self.trainer.args.output_dir, self.flops_per_token,
                                       self.n_stages, self.plot_metadata)
                    plot_loss_vs_walltime(self.loss_history, self.trainer.args.output_dir, self.n_stages,
                                          self.plot_metadata)
                    plot_achieved_tflops(self.loss_history, self.trainer.args.output_dir, self.n_stages,
                                         self.plot_metadata)

                # Check if this was the final stage
                if self.dataset._is_final_stage():
                    if is_main_process():
                        print("[FINISHED] Curriculum complete (reached max_lookahead)" if getattr(self.dataset,
                                                                                                  'linear_lookahead',
                                                                                                  False)
                              else "[FINISHED] Curriculum complete")
                    control.should_training_stop = True
                    self.finished = True
                else:
                    # Advance to next stage
                    self.dataset.stage += 1
                    self.stage_start_step = state.global_step
                    self.stage_start_time = datetime.datetime.now()
                    self.samples_this_stage = 0
                    self.tokens_this_stage = 0

                    # Clear accuracy tracking for fresh measurement
                    self.trainer.first_token_correct.clear()
                    self.trainer.full_word_correct.clear()
                    if self.plateau_spike:
                        self.plateau_acc_history.clear()

                    # Report and reset timing for new stage
                    if hasattr(self.trainer, '_report_train_timing'):
                        self.trainer._report_train_timing()
                    if hasattr(self.trainer, 'reset_timing'):
                        self.trainer.reset_timing()

                    # Apply stage schedule strategy
                    if self.stage_schedule != "none":
                        self._apply_stage_schedule(state)

                    new_alpha = self.dataset._stage_alpha()
                    if is_main_process():
                        msg = f" -> Advanced to stage {self.dataset.stage} | alpha={new_alpha:.3f}"

                        if getattr(self.dataset, "task", None) == "search":
                            target_L = self.dataset._stage_target_lookahead()
                            if target_L is not None:
                                max_L = self.dataset.task_kwargs.get("max_lookahead", 12)
                                msg += f" | L={target_L}/{max_L}"
                            else:
                                n = getattr(self.dataset, "max_input_size", 256)
                                cap = self.dataset.task_kwargs.get("max_lookahead")
                                L_eff = effective_search_L(new_alpha, n, max_lookahead_cap=cap)
                                msg += f" | effective_L={L_eff} (n={n}, cap={cap})"
                        print(msg)

                        current_wall_time = self.wall_time_offset
                        if self.training_start_time is not None:
                            current_wall_time += (time.time() - self.training_start_time)

                        _save_curriculum_state(self.trainer.args.output_dir, self.dataset.stage, self.stage_start_step,
                                               current_wall_time,
                                               first_token_correct=self.trainer.first_token_correct,
                                               full_word_correct=self.trainer.full_word_correct,
                                               recent_losses=self.trainer.recent_losses,
                                               samples_this_stage=self.samples_this_stage,
                                               tokens_this_stage=self.tokens_this_stage,
                                               lr_reset_step=self.lr_reset_step,
                                   batch_increase_count=self.batch_increase_count if self.batch_increase_count else None,
                                   plateau_last_spike_step=self.plateau_last_spike_step if self.plateau_spike else None)

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
    p.add_argument("--model_name", type=str, default="EleutherAI/pythia-160m")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./nl_output")

    # Model initialization
    p.add_argument("--reinit_weights", action="store_true", default=False,
                   help="Re-initialize model weights randomly instead of using pretrained weights")

    # LoRA
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)

    # Seed
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--deterministic", action="store_true",
                   help="Enable CUDA deterministic algorithms (may be slower)")

    # Training hyperparams
    p.add_argument("--batch_size", type=int, default=16,
                   help="Samples per training step. With packing: target sequences packed per step. Without: per_device_train_batch_size.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    # LR scheduling is handled by --stage_schedule (per-stage cosine/warmup/SGDR).
    # Global lr_scheduler_type is always "constant" — do not change.
    p.add_argument("--first_token_soft_weight", type=float, default=0.3)

    # Few-shot in prompt
    p.add_argument("--num_shots", type=int, default=0, choices=[0, 1, 2])

    # Curriculum
    p.add_argument("--n_stages", type=int, default=10)
    p.add_argument("--base_alpha", type=float, default=0.1)
    p.add_argument("--max_alpha", type=float, default=1.0, help="Maximum alpha during training (eval always uses 1.0)")
    p.add_argument("--linear_lookahead", action="store_true",
                   help="Use linear lookahead curriculum (search task only)")
    p.add_argument("--base_lookahead", type=int, default=1,
                   help="Starting lookahead at stage 1 (linear_lookahead mode)")
    p.add_argument("--lookahead_step", type=int, default=1,
                   help="Lookahead increase per stage (linear_lookahead mode)")
    p.add_argument("--accuracy_threshold", type=float, default=0.98)
    p.add_argument("--min_steps_per_stage", type=int, default=500)
    p.add_argument("--check_every", type=int, default=50)
    p.add_argument("--accuracy_window", type=int, default=200,
                   help="Rolling window size (per GPU) for advancement accuracy check")

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
    p.add_argument("--stage_eval_every", type=int, default=1,
                   help="Run stage eval every N lookahead units (e.g. 8 = eval only when L is multiple of 8)")
    p.add_argument("--do_stage_eval", action="store_true",
                   help="Run TF+greedy eval at alpha=1.0 after each stage advancement")
    p.add_argument("--eval_every_steps", type=int, default=0,
                   help="Run greedy eval every N steps (0=disabled, useful for no-curriculum runs)")
    p.add_argument("--persist_every", type=int, default=2000,
                   help="Save persistent checkpoint every N steps (not rotated by save_total_limit). 0=disabled.")
    p.add_argument("--save_total_limit", type=int, default=20,
                   help="Number of rolling checkpoint-* dirs to keep. stage_checkpoints/ and persistent_checkpoints/ are unaffected.")
    p.add_argument("--lr_reset_on_stage", action="store_true",
                   help="[DEPRECATED: use --stage_schedule warmup_reset] Reset LR scheduler on stage advance")
    p.add_argument("--lr_reset_warmup", type=int, default=50,
                   help="Warmup steps after LR reset on stage advance (default 50)")
    p.add_argument("--stage_schedule", type=str, default="none",
                   choices=["none", "warmup_reset", "cosine_restart", "cosine_sgdr", "batch_increase", "lr_spike"],
                   help="LR/batch schedule strategy on stage advance")
    p.add_argument("--cosine_t_max", type=int, default=3000,
                   help="Steps for cosine decay per stage (cosine_restart)")
    p.add_argument("--cosine_t0", type=int, default=10000,
                   help="Initial restart period for SGDR (cosine_sgdr)")
    p.add_argument("--cosine_t_mult", type=int, default=2,
                   help="Period multiplier for SGDR warm restarts (cosine_sgdr)")
    p.add_argument("--cosine_eta_min_ratio", type=float, default=0.01,
                   help="Min LR as fraction of peak for cosine_restart")
    p.add_argument("--batch_increase_factor", type=float, default=2.0,
                   help="Multiply grad_accumulation_steps by this on stage advance (batch_increase)")
    p.add_argument("--lr_spike_factor", type=float, default=5.0,
                   help="Spike LR to peak*factor, then decay back (lr_spike/plateau_spike)")
    p.add_argument("--lr_spike_steps", type=int, default=200,
                   help="Duration of LR spike cycle in steps (lr_spike/plateau_spike)")
    p.add_argument("--plateau_spike", action="store_true",
                   help="Take action when accuracy plateaus (independent of stage_schedule)")
    p.add_argument("--plateau_action", type=str, default="lr_spike",
                   choices=["lr_spike", "batch_increase"],
                   help="Action to take on plateau: spike LR or increase batch size")
    p.add_argument("--plateau_window", type=int, default=5000,
                   help="Steps to look back for plateau detection")
    p.add_argument("--plateau_threshold", type=float, default=0.02,
                   help="Min accuracy improvement over window to not count as plateau")
    p.add_argument("--plateau_cooldown", type=int, default=10000,
                   help="Min steps between plateau spikes")

    # Redacted eval config
    p.add_argument("--eval_redacted_samples", type=int, default=None)
    p.add_argument("--redaction_token", type=str, default="_____")

    # Memory control
    p.add_argument("--gradient_checkpointing", action="store_true")

    # Scratch / resume
    p.add_argument("--scratch_dir", type=str,
                   default=os.environ.get("SCRATCH") or os.path.join("/scratch", os.environ.get("USER", "user")))
    p.add_argument("--job_id", type=str, default=os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID"))
    p.add_argument("--resume_from_job", type=str, default=None)
    p.add_argument("--resume_weights_path", type=str, default=None,
                   help="Load model weights from this path (no optimizer/scheduler). Use with --resume_stage.")
    p.add_argument("--resume_stage", type=int, default=None,
                   help="Start curriculum at this stage (use with --resume_weights_path)")

    # Liger kernels
    p.add_argument("--use_liger", action="store_true",
                   help="Use Liger kernel for memory-efficient training")

    # Chunked cross-entropy
    p.add_argument("--use_chunked_ce", action="store_true", help="Use chunked cross-entropy for memory efficiency")
    p.add_argument("--ce_chunk_size", type=int, default=1024, help="Chunk size for chunked cross-entropy")

    # Packing
    p.add_argument("--use_packing", action="store_true",
                   help="Use sequence packing for efficiency")

    # Pretraining data mixing (anti-catastrophic-forgetting)
    p.add_argument("--mix_pretrain_data", type=str, default=None,
                   help="HuggingFace dataset for pretraining mix (e.g. 'allenai/c4')")
    p.add_argument("--mix_pretrain_subset", type=str, default=None,
                   help="Dataset subset/config (e.g. 'en' for C4, omit for datasets without subsets)")
    p.add_argument("--mix_pretrain_ratio", type=float, default=0.1,
                   help="Fraction of batches that are pretraining data (default: 0.1 = 10%%)")
    p.add_argument("--mix_pretrain_max_len", type=int, default=2048,
                   help="Max sequence length for pretraining samples (default: 2048)")
    p.add_argument("--use_chat_template", action="store_true",
                   help="Wrap search data in chat template (enable_thinking=False)")

    global args
    args = p.parse_args()

    # Validate linear_lookahead
    if args.linear_lookahead:
        if args.task != "search":
            rank_print("[WARN] --linear_lookahead only applies to search task, ignoring")
            args.linear_lookahead = False
        else:
            # Calculate expected number of stages
            if args.lookahead_step > 0:
                expected_stages = math.ceil((args.max_lookahead - args.base_lookahead) / args.lookahead_step) + 1
            else:
                expected_stages = 1  # No curriculum progression (fixed lookahead)
            rank_print(
                f"[CURRICULUM] Linear lookahead mode: L={args.base_lookahead} to {args.max_lookahead}, step={args.lookahead_step}")
            rank_print(f"[CURRICULUM] Expected stages: {expected_stages}")

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
        set_all_seeds(args.seed, deterministic=getattr(args, 'deterministic', False))

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

    resume_step = 0
    if resume_ckpt:
        match = re.search(r'checkpoint-(\d+)', resume_ckpt)
        if match:
            resume_step = int(match.group(1))

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

    # Apply Liger fused ops (NOT cross entropy) — architecture-dependent
    # Only apply to compatible architectures (Qwen uses RMSNorm+SwiGLU; Pythia uses LayerNorm+GELU)
    if getattr(args, 'use_liger', False):
        is_qwen = 'qwen' in args.model_name.lower()
        if is_qwen:
            try:
                from liger_kernel.transformers import apply_liger_kernel_to_qwen3
                apply_liger_kernel_to_qwen3(
                    rope=True, rms_norm=True, swiglu=True,
                    cross_entropy=False, fused_linear_cross_entropy=False,
                )
                rank_print("[LIGER] Applied fused RoPE/RMSNorm/SwiGLU kernels (Qwen3)")
            except ImportError:
                rank_print("[LIGER] liger-kernel not installed, continuing without")
        else:
            rank_print("[LIGER] Skipping — no compatible Liger kernels for this architecture")

    if getattr(args, 'reinit_weights', False):
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, **{k: v for k, v in model_kwargs.items() if k != "cache_dir"})
        rank_print("[INIT] Randomly initialized model weights (--reinit_weights)")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # Force-tie lm_head.weight to embed_tokens.weight as the same Parameter.
    # HF's tie_weights() is broken for Qwen3 + Trainer.resume_from_checkpoint,
    # leaving lm_head at pretrained init after resume (loss explodes to ~30).
    # Direct aliasing ensures both params point to the same tensor so
    # load_state_dict updates both simultaneously.
    if hasattr(model, "lm_head") and hasattr(model, "get_input_embeddings"):
        embed = model.get_input_embeddings()
        if embed is not None and model.lm_head.weight is not embed.weight:
            model.lm_head.weight = embed.weight
            rank_print("[TIE] Force-tied lm_head.weight to embed_tokens.weight")

    if is_main_process():
        attn_impl = getattr(model.config, "_attn_implementation", "unknown")
        print(f"[ATTENTION] Using: {attn_impl}")
        print(f"[FLASH] flash_sdp_enabled: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"[FLASH] mem_efficient_sdp_enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
        print(f"[FLASH] math_sdp_enabled: {torch.backends.cuda.math_sdp_enabled()}")

    # Gradient checkpointing
    if args.gradient_checkpointing:
        if hasattr(model, "config"):
            model.config.use_cache = False
            rank_print("[MEM] model.config.use_cache = False (training)")

    # LoRA
    if args.use_lora:
        rank_print("[INIT] Applying LoRA...")
        from peft import PeftModel

        # Auto-detect LoRA target modules based on architecture
        if hasattr(model, 'gpt_neox'):
            lora_targets = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        else:
            lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        rank_print(f"[INIT] LoRA targets: {lora_targets}")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=args.lora_dropout,
            target_modules=lora_targets,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
                rank_print("[MEM] Enabled input grads for Gradient Checkpointing + LoRA")

    if is_main_process():
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()
        else:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            rank_print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100*trainable/total:.4f}")

    # Load weights from a specific path (no optimizer state)
    if getattr(args, 'resume_weights_path', None):
        from safetensors.torch import load_file as load_safetensors
        weights_path = args.resume_weights_path
        sf_path = os.path.join(weights_path, "model.safetensors")
        if os.path.exists(sf_path):
            state_dict = load_safetensors(sf_path)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            rank_print(f"[WEIGHTS] Loaded weights from {weights_path}")
            if missing:
                rank_print(f"[WEIGHTS] Missing keys: {len(missing)}")
            if unexpected:
                rank_print(f"[WEIGHTS] Unexpected keys: {len(unexpected)}")
        else:
            rank_print(f"[WEIGHTS][ERROR] No model.safetensors found at {weights_path}")
            sys.exit(1)

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
    eval_inputs_hard, eval_labels_hard = None, None  # alpha=1.0 (hardest)
    eval_inputs_easy, eval_labels_easy = None, None  # alpha=base_alpha (easiest)
    eval_fingerprint_hard, eval_fingerprint_easy = None, None

    need_eval_data = (args.do_baseline or args.do_final_eval or args.do_redacted_eval or args.do_stage_eval)

    if need_eval_data:
        rank_print("[EVAL-DATA] Generating eval sets (once for all evals)...")

        if is_main_process():
            # Hard eval set: alpha=1.0
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

            # Easy eval set: alpha=base_alpha
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

        rank_print(f"[EVAL-DATA] Hard (alpha=1.0): n={len(eval_inputs_hard)}, fingerprint={eval_fingerprint_hard}")
        rank_print(
            f"[EVAL-DATA] Easy (alpha={args.base_alpha}): n={len(eval_inputs_easy)}, fingerprint={eval_fingerprint_easy}")

        # Reserve these inputs so training doesn't generate duplicates
        if eval_inputs_hard: reserved_inputs.update(eval_inputs_hard)
        if eval_inputs_easy: reserved_inputs.update(eval_inputs_easy)

    # ---------------- Training dataset ----------------
    rank_print(
        f"[DATASET] Using PackedSequenceDataset (batch_size={args.batch_size})")
    if args.do_seen_eval:
        rank_print("[WARN] --do_seen_eval disabled for packed dataset (incompatible with multi-worker)")
        args.do_seen_eval = False

    dataset = PackedSequenceDataset(
        task=args.task,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        stage=1,
        n_stages=args.n_stages,
        base_alpha=args.base_alpha,
        max_alpha=args.max_alpha,
        max_input_size=args.max_input_size,
        reserved_inputs=reserved_inputs,
        num_shots=args.num_shots,
        seed=args.seed,
        resume_step=resume_step * args.gradient_accumulation_steps,
        store_examples=args.do_seen_eval,
        store_cap=1000,
        linear_lookahead=args.linear_lookahead,
        base_lookahead=args.base_lookahead,
        lookahead_step=args.lookahead_step,
        epoch_size=1_000_000_000,  # Must exceed max_steps * gradient_accumulation_steps
        mix_pretrain_data=getattr(args, 'mix_pretrain_data', None),
        mix_pretrain_subset=getattr(args, 'mix_pretrain_subset', None),
        mix_pretrain_ratio=getattr(args, 'mix_pretrain_ratio', 0.1),
        mix_pretrain_max_len=getattr(args, 'mix_pretrain_max_len', 512),
        mix_pretrain_cache_dir=os.path.join(os.environ.get("SCRATCH", "/tmp"), "pretrain_cache"),
        use_chat_template=getattr(args, 'use_chat_template', False),
        **task_kwargs,
    )
    data_collator = lambda x: x[0]  # Identity
    effective_batch_size = 1
    num_workers = int(os.environ.get("BENCH_NUM_WORKERS", 4))
    curriculum = FirstTokenCurriculum(
        dataset=dataset,
        n_stages=args.n_stages,
        accuracy_threshold=args.accuracy_threshold,
        min_steps_per_stage=args.min_steps_per_stage,
        check_every=args.check_every,
        use_packing=args.use_packing,
        # Stage eval config
        do_stage_eval=args.do_stage_eval,
        stage_eval_every=getattr(args, 'stage_eval_every', 1),
        eval_every_steps=args.eval_every_steps,
        eval_inputs_hard=eval_inputs_hard,
        eval_labels_hard=eval_labels_hard,
        eval_fingerprint_hard=eval_fingerprint_hard,
        tokenizer=tokenizer,
        task=args.task,
        task_kwargs=task_kwargs,
        num_shots=args.num_shots,
        max_input_size=args.max_input_size,
        seed=args.seed,
        persist_every=getattr(args, 'persist_every', 2000),
        print_examples=min(5, args.print_eval_examples),
        lr_reset_on_stage=getattr(args, 'lr_reset_on_stage', False),
        lr_reset_warmup=getattr(args, 'lr_reset_warmup', 50),
        peak_lr=args.learning_rate,
        stage_schedule=getattr(args, 'stage_schedule', 'none'),
        cosine_t_max=getattr(args, 'cosine_t_max', 3000),
        cosine_t0=getattr(args, 'cosine_t0', 10000),
        cosine_t_mult=getattr(args, 'cosine_t_mult', 2),
        cosine_eta_min_ratio=getattr(args, 'cosine_eta_min_ratio', 0.01),
        batch_increase_factor=getattr(args, 'batch_increase_factor', 2.0),
        lr_spike_factor=getattr(args, 'lr_spike_factor', 5.0),
        lr_spike_steps=getattr(args, 'lr_spike_steps', 200),
        plateau_spike=getattr(args, 'plateau_spike', False),
        plateau_action=getattr(args, 'plateau_action', 'lr_spike'),
        plateau_window=getattr(args, 'plateau_window', 5000),
        plateau_threshold=getattr(args, 'plateau_threshold', 0.02),
        plateau_cooldown=getattr(args, 'plateau_cooldown', 10000),
    )

    if resume_ckpt:
        if _try_restore_curriculum_state(resume_ckpt, dataset, curriculum):
            rank_print(f"[CURRICULUM] Synced state with checkpoint: {resume_ckpt}")

        # Load existing resume_points from current output_dir (persists across restarts)
        rp_path = os.path.join(args.output_dir, "resume_points.json")
        if os.path.isfile(rp_path):
            try:
                with open(rp_path) as f:
                    existing_rps = json.load(f)
                # Merge: existing points first, then any new ones from _try_restore
                merged_rps = existing_rps
                for rp in curriculum.resume_points:
                    if rp not in merged_rps:
                        merged_rps.append(rp)
                curriculum.resume_points = merged_rps
                rank_print(f"[CURRICULUM] Loaded {len(existing_rps)} existing resume points")
            except Exception:
                pass

    # Ensure loss history includes predecessor job's data when auto-resuming locally
    if args.resume_from_job and curriculum.loss_history:
        first_step = curriculum.loss_history[0].get("step", 0)
        if first_step > 1:
            prev_dir = os.path.join(args.scratch_dir, "nl_output", args.task, f"job_{args.resume_from_job}")
            # Try JSONL first (has entries between checkpoint saves), fall back to JSON
            prev_history = _load_from_jsonl(prev_dir, max_step=first_step - 1)
            if prev_history is None:
                prev_loss_path = os.path.join(prev_dir, "loss_history.json")
                if os.path.isfile(prev_loss_path):
                    with open(prev_loss_path) as f:
                        raw = json.load(f)
                    prev_history = [
                        h for h in raw
                        if h.get("step", 0) < first_step
                        and not (h.get("resume") and h.get("tokens", -1) == 0)
                    ]
            if prev_history:
                curriculum.loss_history = prev_history + curriculum.loss_history
                rank_print(f"[CURRICULUM] Prepended {len(prev_history)} loss records from job {args.resume_from_job}")

                # Also merge resume_points from predecessor
                prev_rp_path = os.path.join(prev_dir, "resume_points.json")
                if os.path.isfile(prev_rp_path):
                    try:
                        with open(prev_rp_path) as f:
                            prev_rps = json.load(f)
                        curriculum.resume_points = prev_rps + curriculum.resume_points
                    except Exception:
                        pass

                # Rewrite JSONL with merged data
                if is_main_process():
                    try:
                        _rewrite_jsonl(args.output_dir, curriculum.loss_history)
                    except Exception as e:
                        rank_print(f"[CURRICULUM][WARN] Failed to rewrite JSONL after merge: {e}")

            prev_eval_path = os.path.join(prev_dir, "stage_eval_history.json")
            if os.path.isfile(prev_eval_path) and curriculum.stage_eval_history:
                first_eval_step = curriculum.stage_eval_history[0].get("step", 0)
                with open(prev_eval_path) as f:
                    prev_eval = json.load(f)
                prev_eval = [h for h in prev_eval if h.get("step", 0) < first_eval_step]
                if prev_eval:
                    curriculum.stage_eval_history = prev_eval + curriculum.stage_eval_history
                    rank_print(f"[CURRICULUM] Prepended {len(prev_eval)} eval records from job {args.resume_from_job}")

    # Write JSONL to current output_dir (handles both within-job and cross-job resume)
    if is_main_process() and curriculum.loss_history:
        try:
            _rewrite_jsonl(args.output_dir, curriculum.loss_history)
            rank_print(f"[LOSS] Wrote {len(curriculum.loss_history)} records to JSONL in {args.output_dir}")
        except Exception as e:
            rank_print(f"[LOSS][WARN] Failed to write JSONL: {e}")

    # Override curriculum stage if --resume_stage is set (for weights-only resume)
    if getattr(args, 'resume_stage', None) is not None:
        dataset.stage = args.resume_stage
        curriculum.stage_start_step = 0
        rank_print(f"[CURRICULUM] Overriding stage to {args.resume_stage} (--resume_stage)")

    # ==================== RETROACTIVE PLOT GENERATION / EXTENSION CHECK ====================
    # If resuming a completed job, either generate plots and exit, or continue with extended params
    if resume_ckpt and curriculum.loss_history:
        # Check if curriculum would be finished under CURRENT parameters
        # Only consider finished if we're at the final stage AND training actually completed
        # (has a completion marker). Being at the final stage alone just means we were
        # preempted mid-training at the last stage.
        is_finished = False

        # Check for completion marker (indicates previous run finished)
        # Look in both current output_dir and source checkpoint directory
        source_dir = os.path.dirname(resume_ckpt)
        completion_markers = [
            os.path.join(args.output_dir, "final", "config.json"),
            os.path.join(source_dir, "final", "config.json"),
        ]
        was_previously_completed = any(os.path.exists(m) for m in completion_markers)

        if was_previously_completed:
            if dataset._is_final_stage():
                # Current params also consider it finished
                is_finished = True
            elif getattr(dataset, 'linear_lookahead', False) and dataset.task == "search":
                # Previous run completed, but we're extending with higher max_lookahead
                current_L = dataset._stage_target_lookahead()
                max_L = dataset.task_kwargs.get("max_lookahead")
                old_max_L = current_L  # The old max was whatever stage we're at

                rank_print(f"\n{'=' * 60}")
                rank_print(f"[CURRICULUM] EXTENDING from completed job")
                rank_print(f"[CURRICULUM] Resumed at stage {dataset.stage} (L={current_L})")
                rank_print(f"[CURRICULUM] New max_lookahead: {max_L}")
                rank_print(f"[CURRICULUM] Will continue training to L={max_L}")
                rank_print(f"{'=' * 60}\n")
                # is_finished stays False - continue training

        if is_finished:
            rank_print(f"[RETROACTIVE] Detected completed job with {len(curriculum.loss_history)} loss records")

            # Compute flops_per_token
            flops_per_token = estimate_flops_per_token(model)
            rank_print(
                f"[FLOPS] Model has {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, ~{flops_per_token / 1e9:.1f}B FLOPs/token")

            # Check if loss_history has tokens field, backfill if missing
            has_tokens = any(h.get("tokens", 0) > 0 for h in curriculum.loss_history)
            if not has_tokens:
                rank_print("[RETROACTIVE] Loss history missing 'tokens' field - estimating from steps")
                if args.use_packing:
                    avg_seq_len = 300
                    est_tokens_per_step = args.batch_size * avg_seq_len * get_world_size()
                else:
                    avg_seq_len = 300
                    est_tokens_per_step = args.batch_size * avg_seq_len * get_world_size()

                rank_print(f"[RETROACTIVE] Estimating ~{est_tokens_per_step} tokens/step")
                for h in curriculum.loss_history:
                    h["tokens"] = est_tokens_per_step

            # Build metadata for retroactive plots
            plot_metadata = {
                "model_name": args.model_name.split("/")[-1],
                "model_params_b": sum(p.numel() for p in model.parameters()) / 1e9,
                "learning_rate": args.learning_rate,
                "use_packing": args.use_packing,
                "batch_size": args.batch_size,
                "target_samples": args.batch_size,
                "accuracy_threshold": args.accuracy_threshold,
            }

            if is_main_process():
                rank_print("[RETROACTIVE] Generating plots for completed job...")
                save_plot_data(args.output_dir, plot_metadata, flops_per_token, args.n_stages,
                              n_gpus=get_world_size())

                plot_overall_loss(curriculum.loss_history, args.output_dir, args.n_stages, plot_metadata)
                plot_loss_vs_flops(curriculum.loss_history, args.output_dir, flops_per_token, args.n_stages,
                                   plot_metadata)
                plot_loss_vs_walltime(curriculum.loss_history, args.output_dir, args.n_stages, plot_metadata)
                plot_achieved_tflops(curriculum.loss_history, args.output_dir, args.n_stages, plot_metadata)

                # Per-stage plots
                stages_seen = set(h["stage"] for h in curriculum.loss_history)
                for stage in sorted(stages_seen):
                    plot_stage_loss(curriculum.loss_history, stage, args.output_dir, plot_metadata)

                # Stage eval plot
                if curriculum.stage_eval_history:
                    plot_stage_eval(curriculum.stage_eval_history, args.output_dir, plot_metadata)
                    plot_eval_acc_vs_step(curriculum.stage_eval_history, args.output_dir, plot_metadata,
                                         loss_history=curriculum.loss_history)
                    plot_eval_acc_vs_flops(curriculum.stage_eval_history, curriculum.loss_history,
                                          args.output_dir, flops_per_token, plot_metadata)

                rank_print(f"[RETROACTIVE] Generated plots for {len(stages_seen)} stages")
                rank_print("[RETROACTIVE] Done. Exiting without training.")

            barrier()
            return 0

    # Disable Accelerate batch dispatching for token-budget training
    os.environ["ACCELERATE_DISPATCH_BATCHES"] = "false"
    os.environ["ACCELERATE_SPLIT_BATCHES"] = "false"
    os.environ["ACCELERATE_EVEN_BATCHES"] = "false"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=effective_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="constant",  # per-stage decay handled by --stage_schedule
        max_steps=100000000,
        num_train_epochs=1000,
        logging_steps=10,
        logging_first_step=False,
        report_to="none",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        optim="adamw_torch_fused",
        dataloader_num_workers=num_workers,
        dataloader_prefetch_factor=2 if num_workers > 0 else None,
        dataloader_persistent_workers=num_workers > 0,
        dataloader_pin_memory=True,

        seed=args.seed if args.seed is not None else 42,

        ddp_find_unused_parameters=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,

        ignore_data_skip=True,
    )

    # Baseline runner uses pre-generated eval data
    def baseline_runner(eval_model, trainer):
        rank_print(f"[BASELINE] Model prepared (Backend: {type(eval_model).__name__})")

        # Verify eval data
        fp_hard = _eval_data_fingerprint(eval_inputs_hard, eval_labels_hard)
        fp_easy = _eval_data_fingerprint(eval_inputs_easy, eval_labels_easy)
        rank_print(f"[BASELINE] Verifying eval data: hard={fp_hard}, easy={fp_easy}")
        assert fp_hard == eval_fingerprint_hard, f"Hard eval fingerprint mismatch!"
        assert fp_easy == eval_fingerprint_easy, f"Easy eval fingerprint mismatch!"

        # Hard eval: alpha=1.0
        base_hard = run_eval_greedy_readable(
            eval_model, tokenizer, args.task,
            eval_inputs_hard, eval_labels_hard,
            num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
            print_examples=args.print_eval_examples, **task_kwargs
        )

        # Easy eval: alpha=base_alpha
        base_easy = run_eval_greedy_readable(
            eval_model, tokenizer, args.task,
            eval_inputs_easy, eval_labels_easy,
            num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
            print_examples=args.print_eval_examples, **task_kwargs
        )

        rank_print(
            f"\n[BASELINE-alpha1.0] First={base_hard['first_token_acc']:.2%} "
            f"| Full={base_hard['full_word_acc']:.2%} | N={base_hard['total']}"
        )
        rank_print(
            f"[BASELINE-alpha{args.base_alpha}] First={base_easy['first_token_acc']:.2%} "
            f"| Full={base_easy['full_word_acc']:.2%} | N={base_easy['total']}\n"
        )

    # Skip baseline if not requested OR if resuming
    baseline_cb = BaselineEvalCallback(
        baseline_runner,
        skip_baseline=(not args.do_baseline) or (resume_ckpt is not None)
    )

    trainer = PackedSequenceTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[curriculum, baseline_cb],
        first_token_soft_weight=args.first_token_soft_weight,
        accuracy_window=args.accuracy_window,
        ce_chunk_size=args.ce_chunk_size,
    )
    curriculum.trainer = trainer
    baseline_cb.trainer = trainer

    # Restore trainer deques from checkpoint if available
    if hasattr(curriculum, '_restored_trainer_state') and curriculum._restored_trainer_state:
        restored = curriculum._restored_trainer_state
        if restored.get("first_token_correct") is not None:
            trainer.first_token_correct.extend(restored["first_token_correct"])
            rank_print(f"[CURRICULUM] Restored {len(restored['first_token_correct'])} first_token_correct entries")
        if restored.get("full_word_correct") is not None:
            trainer.full_word_correct.extend(restored["full_word_correct"])
            rank_print(f"[CURRICULUM] Restored {len(restored['full_word_correct'])} full_word_correct entries")
        if restored.get("recent_losses") is not None:
            trainer.recent_losses.extend(restored["recent_losses"])
            rank_print(f"[CURRICULUM] Restored {len(restored['recent_losses'])} recent_losses entries")
        del curriculum._restored_trainer_state

    curriculum.flops_per_token = estimate_flops_per_token(model)
    rank_print(
        f"[FLOPS] Model has {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, ~{curriculum.flops_per_token / 1e9:.1f}B FLOPs/token")

    # Build plot metadata
    curriculum.plot_metadata = {
        "model_name": args.model_name.split("/")[-1],  # Just the model name, not full path
        "model_params_b": sum(p.numel() for p in model.parameters()) / 1e9,
        "learning_rate": args.learning_rate,
        "use_packing": args.use_packing,
        "batch_size": args.batch_size,
        "target_samples": args.batch_size,
        "accuracy_threshold": args.accuracy_threshold,
    }

    if is_main_process():
        save_plot_data(args.output_dir, curriculum.plot_metadata,
                       curriculum.flops_per_token, args.n_stages,
                       n_gpus=get_world_size())

    if not resume_ckpt:
        if is_main_process():
            _save_run_config(args.output_dir, args.batch_size,
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
    if is_main_process():
        _save_curriculum_state(args.output_dir, dataset.stage, curriculum.stage_start_step)

    # ----- TRAINING STARTS HERE -----
    rank_print("\n[TRAIN] Starting training...\n")

    if resume_ckpt:
        match = re.search(r'checkpoint-(\d+)', resume_ckpt)
        if match:
            resume_step = int(match.group(1))
            rank_print(f"[CURRICULUM] Resuming from step {resume_step}")

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
            seen_result = run_eval_greedy_readable(
                trainer.model, tokenizer, args.task,
                seen_inputs, seen_labels,
                num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
                print_examples=min(5, args.print_eval_examples), **task_kwargs
            )
            rank_print(
                f"[SEEN-EVAL] First={seen_result['first_token_acc']:.2%} | Full={seen_result['full_word_acc']:.2%} | N={seen_result['total']}")

        barrier()

    # ----- Final eval: TF + greedy at alpha=1.0 and alpha=base_alpha -----
    if args.do_final_eval:
        trainer.model.eval()

        # Verify we're using the same data
        fp_hard = _eval_data_fingerprint(eval_inputs_hard, eval_labels_hard)
        fp_easy = _eval_data_fingerprint(eval_inputs_easy, eval_labels_easy)
        rank_print(f"[FINAL-EVAL] Verifying eval data: hard={fp_hard}, easy={fp_easy}")
        assert fp_hard == eval_fingerprint_hard, f"Hard eval fingerprint mismatch! {fp_hard} != {eval_fingerprint_hard}"
        assert fp_easy == eval_fingerprint_easy, f"Easy eval fingerprint mismatch! {fp_easy} != {eval_fingerprint_easy}"
        rank_print(f"[FINAL-EVAL] Fingerprints verified OK")

        greedy_hard = run_eval_greedy_readable(
            trainer.model, tokenizer, args.task,
            eval_inputs_hard, eval_labels_hard,
            num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
            print_examples=min(3, args.print_eval_examples), **task_kwargs
        )
        rank_print(
            f"[FINAL-GREEDY-alpha1.0] First={greedy_hard['first_token_acc']:.2%} | Full={greedy_hard['full_word_acc']:.2%} | N={greedy_hard['total']}")

        greedy_easy = run_eval_greedy_readable(
            trainer.model, tokenizer, args.task,
            eval_inputs_easy, eval_labels_easy,
            num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
            print_examples=0, **task_kwargs
        )
        rank_print(
            f"[FINAL-GREEDY-alpha{args.base_alpha}] First={greedy_easy['first_token_acc']:.2%} | Full={greedy_easy['full_word_acc']:.2%} | N={greedy_easy['total']}")

        # Save metrics
        if is_main_process():
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
                json.dump({
                    "eval_fingerprint_hard": eval_fingerprint_hard,
                    "eval_fingerprint_easy": eval_fingerprint_easy,
                    # "tf_hard": final_tf_hard,
                    "greedy_hard": greedy_hard,
                    # "tf_easy": final_tf_easy,
                    "greedy_easy": greedy_easy,
                }, f, indent=2)

    # ----- Scrambled eval: sanity check (should be near baseline ~3%) -----
    # Randomly permutes rule consequents so graph edges are nonsensical.
    # If model truly does graph traversal, accuracy drops to baseline.
    if args.do_redacted_eval:
        trainer.model.eval()

        fp_hard = _eval_data_fingerprint(eval_inputs_hard, eval_labels_hard)
        rank_print(f"[SCRAMBLED-EVAL] Verifying source data: hard={fp_hard}")
        assert fp_hard == eval_fingerprint_hard, f"Hard eval fingerprint mismatch! {fp_hard} != {eval_fingerprint_hard}"

        red_n = args.eval_redacted_samples
        if red_n is None or red_n <= 0:
            red_n = args.eval_samples

        scr_inputs, scr_labels = build_scrambled_eval_set(
            eval_inputs_hard, eval_labels_hard, max_n=red_n
        )

        if scr_inputs:
            scr_result = run_eval_greedy_readable(
                trainer.model, tokenizer, args.task,
                scr_inputs, scr_labels,
                num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
                print_examples=min(3, args.print_eval_examples), **task_kwargs
            )
            rank_print(
                f"[SCRAMBLED] First={scr_result['first_token_acc']:.2%} | Full={scr_result['full_word_acc']:.2%} | N={scr_result['total']}")
        else:
            rank_print("[SCRAMBLED-EVAL] Skipped (could not scramble any eval items)")

    if is_main_process():
        # Flush any remaining JSONL buffer
        if curriculum._jsonl_buffer:
            try:
                _append_to_jsonl(args.output_dir, curriculum._jsonl_buffer)
                curriculum._jsonl_buffer = []
            except Exception:
                pass

        # Save final loss history (atomic write)
        try:
            _atomic_write_json(
                os.path.join(args.output_dir, "loss_history.json"),
                curriculum.loss_history
            )
            rank_print(f"[LOSS] Saved {len(curriculum.loss_history)} records")
        except Exception as e:
            rank_print(f"[LOSS] Warning: {e}")

        # Save resume points
        if curriculum.resume_points:
            try:
                _atomic_write_json(
                    os.path.join(args.output_dir, "resume_points.json"),
                    curriculum.resume_points
                )
            except Exception:
                pass

        # Final overall plot
        plot_overall_loss(curriculum.loss_history, args.output_dir, args.n_stages, curriculum.plot_metadata)
        plot_loss_vs_flops(curriculum.loss_history, args.output_dir, curriculum.flops_per_token, args.n_stages,
                           curriculum.plot_metadata)
        plot_loss_vs_walltime(curriculum.loss_history, args.output_dir, args.n_stages, curriculum.plot_metadata)
        plot_achieved_tflops(curriculum.loss_history, args.output_dir, args.n_stages, curriculum.plot_metadata)
        if curriculum.stage_eval_history:
            plot_stage_eval(curriculum.stage_eval_history, args.output_dir, curriculum.plot_metadata)
            plot_eval_acc_vs_step(curriculum.stage_eval_history, args.output_dir, curriculum.plot_metadata,
                                 loss_history=curriculum.loss_history)
            plot_eval_acc_vs_flops(curriculum.stage_eval_history, curriculum.loss_history,
                                  args.output_dir, curriculum.flops_per_token, curriculum.plot_metadata)

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