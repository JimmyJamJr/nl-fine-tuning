"""Curriculum fine-tuning on packed natural-language graph-search data (nl/tuning_nl.py).

Training forward (refactored 2026-10-09). Every micro-batch is one flat packed sequence [1, T] with per-sequence
position_ids restarting at 0 (PackedSequenceDataset._pack_batch). PackedSequenceTrainer.compute_loss runs the
transformer BODY only (Qwen3Model / GPTNeoXModel, i.e. unwrapped.model / unwrapped.gpt_neox after stripping the DDP
`.module` and PeftModel wrappers) through transformers' built-in packed flash-attention path, passing
attention_mask=None, use_cache=False and explicit int32 cu_seq_lens_q/k plus python-int max_length_q/k, and then
applies the lm_head (sparse: labelled rows only, see NL_HEAD), the cross-entropy, the soft first-token blend,
loss = ce.sum()/n_valid, the loss split and the gate accuracies itself (PackedSequenceTrainer._hf_forward_body,
compute_loss, _head_ce_and_preds). Two status-quo properties are preserved on purpose: DistributedDataParallel.forward
is bypassed, so gradients are NOT averaged across ranks unless NL_DDP_GRAD_AVERAGE=1, and no autocast is active (the
model runs in native bf16).

Environment flags
  NL_PACKING=hf|custom          hf (default): the HF body path above. custom: the pre-refactor hand-rolled layer loop
                                in legacy_varlen_forward.py (parity runs only); the model then loads with transformers'
                                default attention (SDPA) exactly as the paper runs did.
  NL_HEAD=sparse|full           sparse (default): the lm_head, argmax and cross-entropy run only on the labelled rows
                                of the shifted sequence (valid_mask), which the loss and both gate metrics are the
                                only consumers of; the ~99% unlabelled rows (zero gradient) are skipped, i.e. about a
                                quarter of all FLOPs per step at 156M head / 596M params. full: the pre-2026-10-09
                                chunked loop over every row. Forced to full under NL_PACKING=custom so the legacy
                                path stays bit-identical to the archived behaviour. Same loss up to GEMM rounding.
                                Every loss_history entry carries head_rows and a "head" tag; achieved_tflops counts
                                the FLOPs executed (executed_flops), the cumulative PFLOPs accounting stays
                                6N * tokens (estimate_flops_per_token). bench/run_arm.sh does not set NL_HEAD, so
                                re-running the pre-sparse full-head hf arms (hf_fa3_det, hf_fa2_det, hf_fa3, hf_fa2)
                                needs NL_HEAD=full exported; bench/sparse_arms.sh exports NL_HEAD=sparse itself.
  NL_ATTN_KERNEL=auto|fa3|fa2   flash-attention family. auto = FA3 only on Hopper (sm90), FA2 elsewhere; fa3/fa2
                                force. hf mode: selects attn_implementation=flash_attention_3|flash_attention_2 at
                                model load. custom mode: selects the kernel legacy_varlen_forward imports. In-run
                                evals (teacher-forced loss, greedy) always run with SDPA, the transformers default and
                                what eval_checkpoints.py uses (eval_attn_sdpa).
  NL_DDP_GRAD_AVERAGE=1         opt-in: all-reduce (AVG) every trainable gradient across ranks after backward and
                                before clipping, i.e. what DistributedDataParallel would do. Default off (status quo).
  NL_CKPT_EVERY_N_LAYERS=N      selective gradient checkpointing: recompute only every N-th decoder layer (default 1).
  NL_CKPT_RELAX_MAX_TOKENS=M    fall back to every layer when a micro-batch exceeds M tokens (default 0 = never).
  FLASH_ATTENTION_DETERMINISTIC=1  deterministic flash-attention backward in both paths (transformers reads it for
                                the HF path, FA2 and FA3 alike; legacy_varlen_forward passes it to the kernel).
                                Determinism for parity runs is env-only: this flag plus CUBLAS_WORKSPACE_CONFIG,
                                both exported by bench/run_arm.sh under DETERMINISTIC=1. The old --deterministic
                                CLI flag (cudnn flags, use_deterministic_algorithms) was removed 2026-10-09.
  NL_DEBUG_PARAM_SYNC=1         [PARAM-SYNC] per-rank parameter checksum after every optimizer step, plus a one-time
                                [DEBUG] compute_loss line reporting DDP-wrapped / is_autocast_enabled.
  NL_PARITY_DUMP=1              parity hooks: first-micro-batch dump, step-1 gradient dump, per-parameter sums per
                                step (all under <output_dir>/parity_dump/) and peak memory in the 10-step log line.
  NL_LEGACY_NEOX_RESIDUAL_ORDER=hf  custom mode only: HF's GPT-NeoX residual summation order (see legacy module).
"""
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
import contextlib
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

from peft import LoraConfig, get_peft_model, TaskType

from nl_generator import NaturalLanguageGraphGenerator

# ---------------------------------------------------------------------------------------------------------------
# Packed-forward path and attention-kernel selection (2026-10-09; the module docstring lists every env flag).
# ---------------------------------------------------------------------------------------------------------------
NL_PACKING = os.environ.get("NL_PACKING", "hf").lower()
if NL_PACKING not in ("hf", "custom"):
    raise ValueError(f"NL_PACKING must be 'hf' or 'custom', got {NL_PACKING!r}")
_ATTN_KERNEL_REQ = os.environ.get("NL_ATTN_KERNEL", "auto").lower()
if _ATTN_KERNEL_REQ not in ("auto", "fa3", "fa2"):
    raise ValueError(f"NL_ATTN_KERNEL must be 'auto', 'fa3' or 'fa2', got {_ATTN_KERNEL_REQ!r}")
# Output head over labelled rows only (sparse, default) or over every packed row (full, the pre-2026-10-09 loop).
# custom mode is the archived legacy path and is pinned to full regardless of the request (see _head_ce_and_preds).
_NL_HEAD_REQ = os.environ.get("NL_HEAD", "sparse").lower()
if _NL_HEAD_REQ not in ("sparse", "full"):
    raise ValueError(f"NL_HEAD must be 'sparse' or 'full', got {_NL_HEAD_REQ!r}")
NL_HEAD = "full" if NL_PACKING == "custom" else _NL_HEAD_REQ


def _is_hopper() -> bool:
    # FA3 is built around Hopper (sm90a: H100/H200). The `search` env's build (flash-attn-3 3.0.0b1) also ships
    # sm_80 kernels, but every non-Hopper run so far used FA2 (and older FA3 builds fail on those GPUs at the
    # first forward), so auto mode uses FA3 only on Hopper. NL_ATTN_KERNEL=fa3 forces it elsewhere.
    try:
        return bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 9)
    except Exception:
        return False


def select_attn_implementation(req: str = _ATTN_KERNEL_REQ) -> str:
    """Map NL_ATTN_KERNEL onto the transformers attn_implementation string used in hf mode. The Qwen paper runs used
    FA2 and the Pythia runs FA3; NL_ATTN_KERNEL=fa2 reproduces the Qwen kernel exactly. transformers accepts
    flash_attention_3 wherever the flash_attn_3 package imports (utils/import_utils.py:1214-1229, no capability
    check), so the Hopper gate is kept here."""
    if req == "fa3" or (req == "auto" and _is_hopper()):
        return "flash_attention_3"
    return "flash_attention_2"


# Startup check that this transformers build takes the explicit varlen kwargs the hf path passes to the body:
# cu_seq_lens_q/k and max_length_q/k reach the kernel through **kwargs chains only (modeling_qwen3.py:255-268,
# 188-226; modeling_flash_attention_utils.py:543-546). Written against transformers==4.57.5.
try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs as _FlashAttentionKwargs
except ImportError as _e:
    raise ImportError("transformers.modeling_flash_attention_utils.FlashAttentionKwargs is missing; the packed "
                      "forward in tuning_nl.py was written against transformers==4.57.5") from _e
_FA_KW_FIELDS = (set(getattr(_FlashAttentionKwargs, "__annotations__", {}))
                 | set(getattr(_FlashAttentionKwargs, "__dataclass_fields__", {})))
if not {"cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"} <= _FA_KW_FIELDS:
    raise ImportError(f"FlashAttentionKwargs does not accept cu_seq_lens_q/cu_seq_lens_k/max_length_q/max_length_k "
                      f"(fields: {sorted(_FA_KW_FIELDS)}); the packed forward in tuning_nl.py was written against "
                      f"transformers==4.57.5")

if NL_PACKING == "custom":
    # Pre-refactor hand-rolled varlen loop; imports flash_attn_varlen_func (FA3/FA2) with the same auto rule.
    import legacy_varlen_forward as _legacy
else:
    _legacy = None

# config._attn_implementation the model was loaded with (set in main()); eval_attn_sdpa restores and asserts it.
NL_TRAIN_ATTN_IMPL = None

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# --- Runtime env sanity for NCCL ---
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


def _dataloader_worker_init(worker_id: int) -> None:
    """Pin each dataloader worker to a single CPU thread.

    Workers inherit OMP_NUM_THREADS from the trainer process, which the launch scripts size for
    training (CPUs / GPUs, e.g. 14 on a 56-core 4-GPU node). Every rank then forks num_workers of
    them, so one node can ask for 4 ranks x (1 main + 4 workers) x 14 = 280 threads on 56 cores, a
    five-fold oversubscription. Sample generation is Python graph sampling plus tokenisation and
    gains nothing from OpenMP, so it pays only the contention.

    Measured symptom: raising workers from 4 to 12 made DataWait WORSE, 17.1% -> 33.7%, instead of
    better (util_probe, 2026-09-08). torch.set_num_threads is the part that actually takes effect
    after the fork; the environment variables are belt-and-braces for libraries that read them
    lazily, since OpenMP itself latches its pool size at first use.
    """
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def estimate_flops_per_token(model) -> int:
    """Estimate FLOPs per token (6N approximation per Kaplan et al.)

    Left at 6N on purpose, for comparability with every past run: the cumulative PFLOPs accounting (milestones,
    max_total_pflops, the loss-vs-PFLOPs plots) is 6N * tokens everywhere. With NL_HEAD=sparse (default since
    2026-10-09) the lm_head only runs on the labelled rows, so the FLOPs actually executed in a step are
        6 * (N - N_head) * tokens + 6 * N_head * head_rows
    with N_head the output-head parameter count (estimate_head_params: lm_head, which shares the embedding weight,
    for Qwen3; embed_out, a separate weight, for Pythia) and head_rows the per-step count written to loss_history
    next to "tokens" (FirstTokenCurriculum.on_step_end). 6N * tokens therefore OVERSTATES the sparse-head cost by
    about 6 * N_head * (tokens - head_rows), roughly a quarter of the total at 156M / 596M. Only the per-step
    throughput diagnostic achieved_tflops (loss_history and the [Stage] log line) uses the executed count
    (executed_flops); the cumulative accounting does not, pending the paper-side decision.

    loss_history conventions: entries without a head_rows key predate the sparse head and ran the full head, treat
    head_rows = tokens; entries since 2026-10-09 carry head_rows, and since the same-day fix a "head" tag
    (sparse|full) as well (_entry_head_mode resolves the mode of any entry).
    """
    total_params = sum(p.numel() for p in model.parameters())
    return 6 * total_params


def estimate_head_params(model) -> int:
    """N_head for the sparse-head cost formula in estimate_flops_per_token: the parameter count of the output head
    that _head_ce_and_preds runs (resolve_model_parts: lm_head for Qwen3, where it shares the embedding weight and so
    enters N once; embed_out for Pythia, a separate weight that enters N next to embed_in). 0 when the head cannot
    be resolved (then achieved_tflops falls back to 6N * tokens and the caller warns)."""
    try:
        return int(sum(p.numel() for p in resolve_model_parts(model)['lm_head'].parameters()))
    except Exception:
        return 0


def executed_flops(tokens: int, head_rows: int, flops_per_token: int, flops_per_head_row, head: str) -> float:
    """FLOPs actually executed for `tokens` packed tokens of which `head_rows` went through the lm_head, for the
    achieved_tflops diagnostic only (see estimate_flops_per_token). head='full' (or N_head unknown): 6N * tokens,
    exactly as always. head='sparse': 6N * tokens - 6 * N_head * (tokens - head_rows); head_rows counts the
    labelled rows plus the soft blend's first rows, never more than tokens on that path (clamped anyway)."""
    flops = tokens * flops_per_token
    if head == "sparse" and flops_per_head_row:
        flops -= flops_per_head_row * max(0, tokens - head_rows)
    return flops


def _entry_head_mode(h: Dict[str, Any]) -> str:
    """Head mode ('sparse'|'full') of one loss_history entry: its "head" tag when present; otherwise inferred from
    head_rows / tokens (the sparse head logs about 1% of tokens, the full head at least tokens - 1 per micro-batch);
    entries without head_rows predate the sparse head and ran the full head (estimate_flops_per_token)."""
    head = h.get("head")
    if head in ("sparse", "full"):
        return head
    tokens = int(h.get("tokens", 0) or 0)
    if tokens <= 0 or h.get("head_rows") is None:
        return "full"
    return "sparse" if int(h["head_rows"]) < 0.5 * tokens else "full"


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


def _dist_version(*names) -> Optional[str]:
    """Installed version of the first distribution name that resolves (liger ships as liger-kernel-nightly)."""
    import importlib.metadata as _md
    for n in names:
        try:
            return _md.version(n)
        except Exception:
            continue
    return None


def _run_provenance(model=None) -> Dict[str, Any]:
    """Which forward path, kernel, flags and library versions a run used; written to run_meta.json and
    run_config.json (2026-10-09) so OLD/NEW parity runs and later re-evaluations are attributable."""
    cfg = getattr(model, "config", None) if model is not None else None
    return {
        "forward_path": NL_PACKING,
        "head": NL_HEAD,
        "attn_implementation": getattr(cfg, "_attn_implementation", None) if cfg is not None else NL_TRAIN_ATTN_IMPL,
        "NL_ATTN_KERNEL": _ATTN_KERNEL_REQ,
        "NL_CKPT_EVERY_N_LAYERS": os.environ.get("NL_CKPT_EVERY_N_LAYERS", "1"),
        "NL_CKPT_RELAX_MAX_TOKENS": os.environ.get("NL_CKPT_RELAX_MAX_TOKENS", "0"),
        "NL_DDP_GRAD_AVERAGE": os.environ.get("NL_DDP_GRAD_AVERAGE", "0"),
        "FLASH_ATTENTION_DETERMINISTIC": os.environ.get("FLASH_ATTENTION_DETERMINISTIC"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "NL_LEGACY_NEOX_RESIDUAL_ORDER": os.environ.get("NL_LEGACY_NEOX_RESIDUAL_ORDER"),
        "versions": {
            "torch": torch.__version__,
            "transformers": _dist_version("transformers"),
            "flash_attn": _dist_version("flash_attn"),
            "flash_attn_3": _dist_version("flash_attn_3"),
            "liger_kernel": _dist_version("liger-kernel-nightly", "liger-kernel"),
            "peft": _dist_version("peft"),
        },
    }


def _save_run_config(dirpath: str, bs: int, gas: int, port: int = None, extra: Optional[Dict[str, Any]] = None) -> None:
    """Atomic save of run config. Includes PORT to enable hopping. `extra` (run provenance) is stored under
    "provenance" so the three historical keys keep their meaning."""
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
        if extra:
            data["provenance"] = dict(extra)

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
        # Restore cumulative_tokens for PFLOPS-milestone tracking
        curriculum._cumulative_tokens = sum(int(h.get("tokens", 0)) for h in curriculum.loss_history)
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
                curriculum._cumulative_tokens = sum(int(h.get("tokens", 0)) for h in curriculum.loss_history)
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

        # NL_HEAD provenance of the resumed lineage (2026-10-09): NL_HEAD defaults to sparse for every launch, so a
        # pre-sparse-head lineage resumed by an unchanged job script silently switches head mode here. Print only.
        last = curriculum.loss_history[-1]
        if last.get("head") in ("sparse", "full") or int(last.get("tokens", 0) or 0) > 0:
            prev_head = _entry_head_mode(last)
            if prev_head != NL_HEAD:
                rank_print(f"[HEAD][WARN] resumed lineage last ran NL_HEAD={prev_head} (step {last.get('step')}); "
                           f"this process runs NL_HEAD={NL_HEAD}: per-step compute changes from here (loss differs "
                           f"at rounding level; head_rows and achieved_tflops in loss_history change scale). "
                           f"Export NL_HEAD={prev_head} to keep the lineage's head mode.")

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
    # search is the only task (the dfs/si branches were removed 2026-10-09)
    return task


def _get_end_tokens(task_type: str) -> str:
    return ". "


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


def _parse_lookahead_mixture(spec: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    """Parse --shuffled_mixture 'L:steps,L:steps,...' into [(lookahead, weight_in_steps), ...]."""
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        L, n = part.split(":")
        out.append((int(L), int(n)))
    if not out:
        return None
    return out


def alpha_for_lookahead(L_target: int,
                        n: int,
                        tokens_per_edge: int = 3,
                        fixed_tokens: int = 4,
                        reserve_edges: int = 1,
                        breadth: int = 2) -> float:
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

    # breadth=2 (default) is the minimum budget for a depth-L search: one goal
    # path plus one equal-length distractor. Larger values widen the deepest
    # instances to `breadth` branches (and scale the vertex-ID pool with them),
    # at the cost of a lower reachable L for a given context size.
    needed_alpha = (breadth * L_target + reserve_edges) / edges_unscaled

    return min(max(needed_alpha, 0.0), 1.0)


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
            breadth: int = 2,
            shuffled_mixture: Optional[List[Tuple[int, int]]] = None,
            reserved_inputs: Optional[Set[str]] = None,
            seed: Optional[int] = None,
            resume_step: int = 0,
            epoch_size: int = 10_000_000,  # Large enough to never cycle
            mix_pretrain_data: Optional[str] = None,
            mix_pretrain_subset: Optional[str] = "en",
            mix_pretrain_ratio: float = 0.1,
            mix_pretrain_max_len: int = 512,
            mix_pretrain_cache_dir: Optional[str] = None,
            use_chat_template: bool = False,
            **task_kwargs,
    ):
        if NL_PACKING == "custom" and not _legacy.FLASH_ATTN_AVAILABLE:
            raise ImportError(
                "flash-attn required for PackedSequenceDataset (NL_PACKING=custom). "
                "Install with: pip install flash-attn --no-build-isolation"
            )
        # NL_PACKING=hf: the model load with attn_implementation=flash_attention_{2,3} has already failed if
        # neither flash_attn nor flash_attn_3 is importable.

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
        self.breadth = breadth
        # --shuffled_mixture (exposure-matched shuffled control, 2026-09-25): each search example draws its
        # lookahead L independently from the curriculum's realized stage mix, weighted by the number of
        # steps the curriculum spent at each L (every step is the same number of examples). Drawn from
        # the per-batch-reseeded worker RNG, so data stays deterministic by batch index. None = off.
        self.shuffled_mixture = shuffled_mixture
        if shuffled_mixture:
            self._mix_L = [int(L) for L, _ in shuffled_mixture]
            self._mix_w = [int(w) for _, w in shuffled_mixture]
        self.max_input_size = max_input_size
        self.reserved_inputs = reserved_inputs or set()
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

        # Per-worker state (lazily initialized)
        self._worker_generator = None
        self._worker_rng = None
        self._pretrain_iterator = None
        self._worker_id = None

        rank_print(f"[DATASET] Packed Map-style | batch_size={batch_size} | epoch_size={epoch_size}")

    @property
    def stage(self):
        return self._stage.value

    @stage.setter
    def stage(self, value):
        self._stage.value = value

    def __len__(self):
        return self.epoch_size

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
            return alpha_for_lookahead(target_L, self.max_input_size,
                                       breadth=getattr(self, "breadth", 2))
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
        if self.shuffled_mixture:
            L_mix = rng.choices(self._mix_L, weights=self._mix_w)[0]
            alpha = alpha_for_lookahead(L_mix, self.max_input_size, breadth=self.breadth)
        else:
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

        prompt_text = ex.input_text
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


def unwrap_model_for_body(model):
    """Strip the DistributedDataParallel (.module) and PeftModel wrappers; returns the HF *ForCausalLM."""
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    try:
        from peft import PeftModel
        if isinstance(unwrapped, PeftModel):
            unwrapped = unwrapped.base_model.model
    except ImportError:
        pass
    return unwrapped


def resolve_model_parts(model) -> Dict[str, Any]:
    """{'arch', 'unwrapped', 'inner', 'lm_head'}: the transformer body (unwrapped.model for Qwen/Llama-style,
    unwrapped.gpt_neox for GPT-NeoX) and the output head. Under LoRA the body's nn.Linear modules were replaced in
    place by peft, so the adapters apply when the body is called directly (PeftModel.forward is bypassed, as before)."""
    unwrapped = unwrap_model_for_body(model)
    if hasattr(unwrapped, 'gpt_neox'):
        # GPT-NeoX (Pythia)
        inner, arch, lm_head = unwrapped.gpt_neox, 'gpt_neox', unwrapped.embed_out
    else:
        # Qwen / Llama-style
        inner = unwrapped.model
        if not hasattr(inner, 'embed_tokens') and hasattr(inner, 'model'):
            inner = inner.model
        arch, lm_head = 'qwen', unwrapped.lm_head
    return {'arch': arch, 'unwrapped': unwrapped, 'inner': inner, 'lm_head': lm_head}


def set_layer_checkpointing(layers, use_ckpt: bool, ckpt_every: int) -> None:
    """Per-micro-batch selective gradient checkpointing for the HF body. GradientCheckpointingLayer.__call__ reads
    the per-instance `gradient_checkpointing` flag at call time (transformers/modeling_layers.py:58-61, 93) and
    Trainer set it True on every layer at train start (trainer.py:2447 -> modeling_utils.py:3716-3730), so writing
    `use_ckpt and (li % ckpt_every == 0)` before the forward recomputes exactly the layers the legacy loop did."""
    for li, layer in enumerate(layers):
        layer.gradient_checkpointing = bool(use_ckpt and (li % ckpt_every == 0))


@contextlib.contextmanager
def eval_attn_sdpa(model):
    """Run an in-run eval (teacher-forced loss, generate) with SDPA, the transformers default and what
    eval_checkpoints.py uses, and restore the training attention implementation afterwards.

    Operates on the unwrapped PreTrainedModel (DistributedDataParallel does not forward attribute access).
    set_attn_implementation (transformers/modeling_utils.py:2748-2782) validates the request and rewrites
    config._attn_implementation_internal on the config object shared by every attention module, which reads it at
    call time (modeling_qwen3.py:212-214, modeling_gpt_neox.py:173-175), so the switch is atomic across layers.
    The restore is asserted: a model left on SDPA would turn the packed position_ids with attention_mask=None into
    a dense [1, 1, T, T] mask (masking_utils.py:734-740, 795-832), about 13.7 GB of bool at T = 117,040.
    No-op when the model already runs SDPA (NL_PACKING=custom loads it that way)."""
    pm = unwrap_model_for_body(model)
    train_impl = pm.config._attn_implementation
    if train_impl == "sdpa":
        yield
        return
    pm.set_attn_implementation("sdpa")
    assert pm.config._attn_implementation == "sdpa", pm.config._attn_implementation
    try:
        yield
    finally:
        pm.set_attn_implementation(train_impl)
        assert pm.config._attn_implementation == train_impl, \
            f"attention implementation not restored after eval: {pm.config._attn_implementation!r} != {train_impl!r}"
        if NL_TRAIN_ATTN_IMPL is not None:
            assert pm.config._attn_implementation == NL_TRAIN_ATTN_IMPL, \
                f"training attention implementation drifted: {pm.config._attn_implementation!r} != {NL_TRAIN_ATTN_IMPL!r}"


def _head_ce_and_preds(lm_head, shift_h, shift_labels, valid_mask, mode: str = NL_HEAD, chunk_size: int = 4096):
    """Output head + per-row cross-entropy of one packed micro-batch (shared by both forward paths; CPU-testable).

    shift_h [Tm1, H], shift_labels [Tm1] (-100 = unlabelled), valid_mask = shift_labels != -100. Returns
    (ce [n_valid] in valid-row order, preds [Tm1] long, head_rows) where head_rows is how many rows went through
    lm_head. ce is F.cross_entropy(reduction='none') on the lm_head output with no explicit dtype conversion here,
    exactly as before: the training forward runs with autocast off (asserted in _hf_forward_body; transformers'
    compute_loss_context_manager enters autocast only for CPU AMP), so with the bf16 model the CE runs on bf16
    logits; under an active torch.autocast it would run in fp32 instead (cross_entropy is on the autocast fp32
    list). Either way the dtype path is the caller's and identical in both modes.

    mode='full': the pre-2026-10-09 loop, verbatim: lm_head and argmax over EVERY row in chunk_size slices, CE at
    the valid rows of each slice; head_rows = Tm1.
    mode='sparse': lm_head only at rows = valid_mask.nonzero() (one call when n_valid <= chunk_size, otherwise
    chunk_size row pieces so the transient [rows, V] logits never exceed the full path's); preds is -1 at every
    unlabelled row and the argmax at labelled rows, so every consumer (first-token and full-word gate metrics, the
    parity dump) is untouched since they only ever read labelled rows (first_valid, span_valid); head_rows = n_valid.
    The two modes differ only by GEMM rounding (different row counts per cuBLAS call)."""
    Tm1 = shift_h.size(0)
    device = shift_h.device
    chunk_size = chunk_size or 4096
    ce_parts = []
    if mode == "full":
        # Chunked lm_head to avoid OOM (full [Tm1, V] logits would be ~65 GiB)
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
        head_rows = Tm1
    elif mode == "sparse":
        rows = valid_mask.nonzero(as_tuple=True)[0]  # long, ascending, on device
        preds = torch.full((Tm1,), -1, dtype=torch.long, device=device)
        for cs in range(0, rows.numel(), chunk_size):
            rows_c = rows[cs:cs + chunk_size]
            logits_v = lm_head(shift_h[rows_c])  # [n_valid_chunk, V]
            preds[rows_c] = logits_v.detach().argmax(dim=-1)
            ce_parts.append(F.cross_entropy(logits_v, shift_labels[rows_c], reduction='none'))
            del logits_v
        head_rows = int(rows.numel())
    else:
        raise ValueError(f"NL_HEAD mode must be 'sparse' or 'full', got {mode!r}")
    ce = torch.cat(ce_parts) if ce_parts else shift_h.new_zeros(0)
    return ce, preds, head_rows


def _blend_first_token_ce(ce, lm_head, shift_h, valid_mask, first_indices, first_valid, targets_by_seq, w: float):
    """Soft first-token CE blend, in place on ce (the per-valid-row CE from _head_ce_and_preds), unchanged from the
    pre-sparse-head code: for every sequence si with first_valid[si], the CE at its first answer row
    first_indices[si] becomes w * soft_ce + (1 - w) * ce, soft_ce = -mean log p over targets_by_seq[si] (skipped when
    that list is empty). The first rows' logits come from a SECOND lm_head call on exactly those rows (as before),
    not from a slice of the main head's logits: that keeps the soft term bit-identical to the old code under both
    NL_HEAD modes (same GEMM shape); the first rows are a subset of the valid rows. Returns the number of rows that
    second call ran through lm_head (0 when the blend is off), for the head_rows accounting."""
    if w <= 0 or not first_valid.any():
        return 0
    device = shift_h.device
    ce_indices = torch.cumsum(valid_mask.int(), dim=0) - 1  # [Tm1]
    valid_fi = first_indices[first_valid]
    valid_ce = ce_indices[valid_fi]
    fi_logits = lm_head(shift_h[valid_fi])  # [N_first, V]
    batch_logp = F.log_softmax(fi_logits, dim=-1)
    del fi_logits

    ce_list = valid_ce.tolist()
    valid_seq_indices = torch.nonzero(first_valid, as_tuple=True)[0].tolist()
    for j, si in enumerate(valid_seq_indices):
        vf = targets_by_seq[si]
        if not vf:
            continue
        ids = torch.tensor(vf, device=device, dtype=torch.long)
        soft_ce = -batch_logp[j, ids].mean()
        ci = ce_list[j]
        ce[ci] = w * soft_ce + (1.0 - w) * ce[ci]
    return int(valid_fi.numel())


class PackedSequenceTrainer(Trainer):
    """
    Trainer for packed sequences: one flat [1, T] micro-batch with per-sequence position_ids and cu_seqlens.
    Supports Qwen and GPT-NeoX (Pythia) architectures. NL_PACKING=hf runs the HF body through transformers' built-in
    varlen flash-attention path (_hf_forward_body); NL_PACKING=custom runs legacy_varlen_forward.forward_body. The
    head, loss and gate metrics are computed here in both modes (compute_loss -> _head_ce_and_preds, which under
    NL_HEAD=sparse runs the lm_head on the labelled rows only; custom mode is pinned to the full per-row loop).
    """

    def __init__(self, *args, first_token_soft_weight=0.3, accuracy_window=1000, ce_chunk_size=4096, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_token_soft_weight = first_token_soft_weight
        self.ce_chunk_size = ce_chunk_size
        self.recent_losses = deque(maxlen=100)
        # accuracy_window is the *global* sample count (total across ranks).
        # Predictions are all-gathered across ranks each microbatch so all ranks
        # see the same deque state, eliminating GPU-count-dependent staleness.
        self.first_token_correct = deque(maxlen=accuracy_window)
        self.full_word_correct = deque(maxlen=accuracy_window)
        self._last_search_loss = None
        self._last_pretrain_loss = None

        self._last_batch_samples = 0
        self._last_batch_tokens = 0
        self._last_efficiency = None

        # Accumulate across micro-batches within one optimizer step (for grad_acc > 1).
        # (Until 2026-10-09 this and the blocks below were initialised inside _load_optimizer_and_scheduler, which
        # only worked because Trainer calls that hook unconditionally at train start.)
        self._step_samples = 0
        self._step_tokens = 0
        self._step_head_rows = 0   # rows the lm_head processed this optimizer step (reset with _step_tokens)

        # Training timing
        self._train_timing = {
            "data_wait": 0.0,
            "total_step": 0.0,
            "steps": 0,
        }
        self._step_start_time: Optional[float] = None
        self._step_end_time: Optional[float] = None

        # Cached model internals (populated on first compute_loss call)
        self._cached_model_parts = None
        self._first_batch_checked = False   # hf mode: one-time cu_seqlens dtype/device, no-cache, no-autocast check

        # NL_DDP_GRAD_AVERAGE=1: average trainable gradients across ranks after backward (see training_step).
        self._ddp_grad_average = os.environ.get("NL_DDP_GRAD_AVERAGE", "0") == "1"
        if self._ddp_grad_average:
            rank_print("[DDP] cross-rank gradient averaging: ON (NL_DDP_GRAD_AVERAGE=1; all-reduce AVG of trainable "
                       "grads after backward, before clipping)")
        else:
            rank_print("[DDP] cross-rank gradient averaging: OFF (status quo: compute_loss bypasses "
                       "DistributedDataParallel.forward, so each rank trains its own replica between checkpoint "
                       "reloads; set NL_DDP_GRAD_AVERAGE=1 to all-reduce)")
        # NL_HEAD: output head over labelled rows only (sparse) or over every packed row (full); see _head_ce_and_preds.
        if NL_HEAD == "sparse":
            rank_print(f"[HEAD] NL_HEAD=sparse: lm_head/argmax/CE on labelled rows only (valid_mask), in "
                       f"--ce_chunk_size={self.ce_chunk_size} row pieces; per-step rows logged as head_rows next to "
                       f"tokens in loss_history (achieved_tflops counts executed FLOPs; cumulative PFLOPs stay "
                       f"6N*tokens, which overstates this path)")
        else:
            _why = ("forced by NL_PACKING=custom (legacy path kept bit-identical)" if NL_PACKING == "custom"
                    else "NL_HEAD=full")
            rank_print(f"[HEAD] NL_HEAD=full ({_why}): chunked lm_head over every packed row, "
                       f"--ce_chunk_size={self.ce_chunk_size}")

        # NL_PARITY_DUMP=1: one-time dumps for the old-vs-new parity test (see _write_parity_dump).
        self._parity_dump = os.environ.get("NL_PARITY_DUMP", "0") == "1"
        self._parity_first_batch_done = False
        self._parity_grad_done = False

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
            worker_init_fn=_dataloader_worker_init if nw > 0 else None,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        t0 = time.perf_counter()
        # Measure data wait: time between end of last step and start of this one
        if self._step_end_time is not None:
            self._train_timing["data_wait"] += t0 - self._step_end_time
        loss = super().training_step(model, inputs, num_items_in_batch)
        # Trainer.training_step has run backward (accelerator.backward, trainer.py:4071). sync_gradients is set per
        # micro-batch before the call (trainer.py:2626) and is True on the last micro-batch of the accumulation
        # window; clipping and the optimizer step follow in _inner_training_loop (trainer.py:2698-2718), so this is
        # the point where DistributedDataParallel would have reduced.
        if self.accelerator.sync_gradients:
            if self._parity_dump and not self._parity_grad_done:
                self._parity_grad_done = True
                self._parity_dump_grads(model)
            if self._ddp_grad_average and dist_is_initialized():
                self._all_reduce_grads(model)
        t1 = time.perf_counter()
        self._train_timing["total_step"] += t1 - t0
        self._train_timing["steps"] += 1
        self._step_end_time = t1
        return loss

    def _all_reduce_grads(self, model):
        """NL_DDP_GRAD_AVERAGE=1: average every trainable parameter's gradient across ranks (ReduceOp.AVG), i.e. what
        DistributedDataParallel would do if compute_loss went through its forward."""
        with torch.no_grad():
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                if p.grad is None:
                    # A rank whose whole accumulation window produced no valid labels has no grads; it must still
                    # take part in every collective (as DDP's reducer would with a zero bucket), or the ranks'
                    # all-reduce sequences desynchronise. Zero is the correct contribution to the average.
                    p.grad = torch.zeros_like(p)
                torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

    def _write_parity_dump(self, name: str, payload: Dict[str, Any]) -> None:
        """NL_PARITY_DUMP=1: write <output_dir>/parity_dump/<name>_rank<r>.json (one file per rank)."""
        try:
            d = os.path.join(self.args.output_dir, "parity_dump")
            os.makedirs(d, exist_ok=True)
            payload = dict(payload, rank=get_rank(), world_size=get_world_size(),
                           global_step=int(self.state.global_step), packing=NL_PACKING, head=NL_HEAD,
                           attn_implementation=NL_TRAIN_ATTN_IMPL)
            with open(os.path.join(d, f"{name}_rank{get_rank()}.json"), "w") as f:
                json.dump(payload, f, indent=1)
            rank_print(f"[PARITY] wrote {name} dump(s) under {d}")
        except Exception as e:
            print(f"[PARITY][WARN] rank {get_rank()}: dump {name} failed: {e}", flush=True)

    def _parity_dump_grads(self, model):
        """NL_PARITY_DUMP=1: per trainable parameter, float64 sum / sum of squares of .grad and a sha256 of its raw
        bytes, per rank, once the first optimizer step's backward is complete (before any NL_DDP_GRAD_AVERAGE
        all-reduce and before clipping). Under the DDP bypass grads are per-rank quantities and the per-rank data
        streams are index-seeded, so rank-by-rank comparison between the two paths is well defined."""
        import hashlib
        rec = {}
        with torch.no_grad():
            # Unwrapped names (no DDP 'module.' prefix), so they join with ParamSyncDebugCallback's per-parameter sums.
            for n, p in unwrap_model_for_body(model).named_parameters():
                if not p.requires_grad or p.grad is None:
                    continue
                g = p.grad.detach()
                g64 = g.double()
                raw = g.contiguous().cpu().view(torch.uint8).numpy().tobytes()
                rec[n] = {"sum": g64.sum().item(), "sumsq": (g64 * g64).sum().item(),
                          "dtype": str(g.dtype), "sha256": hashlib.sha256(raw).hexdigest()}
        self._write_parity_dump(f"grads_step{int(self.state.global_step) + 1}", {"params": rec})

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

    def _get_model_parts(self, model):
        """Cache model internals on first call to avoid repeated unwrapping. hf mode: {arch, unwrapped, inner,
        lm_head} from resolve_model_parts; custom mode: the legacy loop's full dict from
        legacy_varlen_forward.get_model_parts (embed, layers, norm, rotary_emb, head counts, rotary dims, ...)."""
        if self._cached_model_parts is None:
            if NL_PACKING == "custom":
                self._cached_model_parts = _legacy.get_model_parts(model)
            else:
                self._cached_model_parts = resolve_model_parts(model)
        return self._cached_model_parts

    def _resolve_ckpt_every(self, tokens: int) -> int:
        """Selective-checkpointing stride for this micro-batch (NL_CKPT_EVERY_N_LAYERS, gated by
        NL_CKPT_RELAX_MAX_TOKENS on `tokens`) with the [CKPT-GATE] logging; shared by both forward paths."""
        # Selective checkpointing: recompute only every Nth layer. N=1 (default) is the historical
        # all-layers behaviour; larger N trades memory for a cheaper backward. Purely a
        # memory/compute dial -- gradients are identical either way.
        #
        # The activation cost tracks tokens per micro-batch, which grows with the curriculum stage
        # (denser graphs pack more tokens into the same buffers), so a setting that fits early runs
        # out of memory later. NL_CKPT_RELAX_MAX_TOKENS therefore gates the relaxed setting on the
        # actual micro-batch size and falls back to every layer above it. Keyed on tokens rather
        # than on a stage number so it responds to the real driver of the memory.
        #
        # Measured on 4xH100 80GB, Qwen3-0.6B, batch 48 x grad-accum 4 x 4 ranks (eff_batch 768):
        #     L    tokens/micro-batch   peak of 81,559 MiB   every-2nd-layer
        #     80        80,095               67,923          fits
        #     88        87,273               68,503          fits
        #    104        95,966               80,837          OOM after ~60 steps
        #    128       117,040               74,127          OOM (74,127 is the every-layer figure)
        # Hence 88,000 for THIS configuration. The threshold is not portable: activation memory per
        # token scales with model width and depth, so a different model, batch size or rank count
        # needs its own measurement before the relaxed setting is enabled.
        ckpt_every = max(1, int(os.environ.get("NL_CKPT_EVERY_N_LAYERS", "1")))
        _relax_max = int(os.environ.get("NL_CKPT_RELAX_MAX_TOKENS", "0"))
        _gated = ckpt_every > 1 and _relax_max > 0 and tokens > _relax_max
        if _gated:
            ckpt_every = 1
        # Report the gate, rate-limited. Packing makes tokens per micro-batch vary by about 40%
        # within a stage (measured on probe 7: 4,793 to 6,728 at a nominal 5,900), so near the
        # threshold the setting flips continually and logging every change approaches one line per
        # step. Log the first few transitions in full, then one line per _LOG_EVERY flips carrying
        # the running tally. That is what a multi-day run actually needs: proof the gate is live,
        # and how often it is switching. The tag is deliberately distinct from the plain [CKPT]
        # used by resume/checkpoint logging, which would otherwise make the log un-greppable.
        if ckpt_every != getattr(self, "_ckpt_gate_last", None):
            self._ckpt_gate_last = ckpt_every
            self._ckpt_gate_flips = getattr(self, "_ckpt_gate_flips", 0) + 1
            _LOG_FIRST, _LOG_EVERY = 5, 500
            if is_main_process() and (self._ckpt_gate_flips <= _LOG_FIRST
                                      or self._ckpt_gate_flips % _LOG_EVERY == 0):
                _tail = (f"  [flip {self._ckpt_gate_flips:,}]"
                         if self._ckpt_gate_flips > _LOG_FIRST else "")
                print(f"[CKPT-GATE] recomputing every {ckpt_every} layer(s) at "
                      f"{tokens:,} tokens/micro-batch "
                      f"(every_n={os.environ.get('NL_CKPT_EVERY_N_LAYERS', '1')}, "
                      f"relax_max={_relax_max:,}"
                      f"{', gate ACTIVE' if _gated else ''}){_tail}", flush=True)
        return ckpt_every

    def _hf_forward_body(self, model, input_ids, position_ids, cu_seqlens_list, max_seqlen_list,
                         use_ckpt: bool, ckpt_every: int):
        """NL_PACKING=hf: run the HF transformer body (Qwen3Model / GPTNeoXModel) on one packed micro-batch through
        transformers' built-in varlen flash-attention path; returns (h [T, H] after the final norm, lm_head).

        The body is called directly, not the CausalLM and not the DDP wrapper: the lm_head and loss stay ours
        (chunked, with the soft first-token blend), DistributedDataParallel.forward is bypassed exactly as before (no
        gradient averaging unless NL_DDP_GRAD_AVERAGE=1) and no autocast is entered (native bf16). With
        attention_mask=None the FA mask builder returns None (masking_utils.py:525-561, 627-628) and
        _flash_attention_forward takes the varlen branch with our cu_seqlens (modeling_flash_attention_utils.py:
        600-603, 632-656); RoPE cos/sin come from the same rotary_emb module the legacy loop used
        (modeling_qwen3.py:407), and Liger's module-level patches (RoPE, RMSNorm, SwiGLU) apply unchanged."""
        mp = self._get_model_parts(model)
        inner, lm_head = mp['inner'], mp['lm_head']
        # Selective gradient checkpointing, per micro-batch (same gate as the legacy loop).
        set_layer_checkpointing(inner.layers, use_ckpt, ckpt_every)
        device = input_ids.device
        # int32 on purpose: FlashAttentionKwargs annotates cu_seq_lens_* as LongTensor (transformers 4.57.5,
        # modeling_flash_attention_utils.py:447-450) but flash_attn 2.8.3 / flash_attn_3 3.0.0b1 take int32, and
        # HF's own position-ids path builds int32 (ibid. :337-342); max_length_* must be python ints
        # (ibid. :352-354). Do not "fix" this to int64 to match the annotation.
        cu = torch.tensor(cu_seqlens_list[0], dtype=torch.int32, device=device)
        max_len = int(max_seqlen_list[0])
        if not self._first_batch_checked:
            assert cu.dtype == torch.int32 and cu.device == device and isinstance(max_len, int), \
                f"cu_seq_lens must be int32 on {device} with a python-int max_length, got {cu.dtype} {cu.device} {type(max_len)}"
            assert not torch.is_autocast_enabled(), "autocast must stay off in the training forward (status quo)"
        out = inner(
            input_ids=input_ids,          # [1, T]
            position_ids=position_ids,    # [1, T], restarts at 0 per sequence
            attention_mask=None,
            use_cache=False,
            cu_seq_lens_q=cu, cu_seq_lens_k=cu,
            max_length_q=max_len, max_length_k=max_len,
        )
        if not self._first_batch_checked:
            assert getattr(out, "past_key_values", None) is None, "body allocated a KV cache despite use_cache=False"
            self._first_batch_checked = True
        return out.last_hidden_state[0], lm_head   # [T, H], after the final norm

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if os.environ.get("NL_DEBUG_PARAM_SYNC") == "1" and not getattr(self, "_fwd_ctx_reported", False):
            self._fwd_ctx_reported = True   # one-time report for the DDP / autocast audit (2026-10-09)
            rank_print(f"[DEBUG] compute_loss: model is DDP-wrapped={hasattr(model, 'module')}, "
                       f"torch.is_autocast_enabled()={torch.is_autocast_enabled()}")
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
        ckpt_every = self._resolve_ckpt_every(self._last_batch_tokens)
        use_ckpt = bool(use_checkpoint) and model.training   # a layer is recomputed when use_ckpt and li % ckpt_every == 0

        all_labels = labels_pad[0]        # [total_tokens]

        # Transformer body -> h [total_tokens, H] after the final norm, plus the output head. Both paths bypass the
        # DDP wrapper and run without autocast; everything from the shift onwards is shared and unchanged.
        if NL_PACKING == "custom":
            h, lm_head = _legacy.forward_body(self._get_model_parts(model), input_ids, position_ids,
                                              cu_seqlens_list, max_seqlen_list, use_ckpt, ckpt_every)
        else:
            h, lm_head = self._hf_forward_body(model, input_ids, position_ids, cu_seqlens_list, max_seqlen_list,
                                               use_ckpt, ckpt_every)

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

            # Output head + per-valid-row CE. NL_HEAD=sparse (default): lm_head only on the labelled rows, preds = -1
            # elsewhere (every consumer below reads labelled rows only). NL_HEAD=full / custom mode: the chunked loop
            # over every row (full [Tm1, V] logits would be ~65 GiB). ce is ordered by ascending valid row either way.
            ce, preds, head_rows = _head_ce_and_preds(lm_head, shift_h, shift_labels, valid_mask,
                                                      mode=NL_HEAD, chunk_size=self.ce_chunk_size)

            # Apply soft first-token CE adjustments (second lm_head call on the first rows, as before)
            head_rows += _blend_first_token_ce(ce, lm_head, shift_h, valid_mask, first_indices, first_valid,
                                               [seq["valid_first_targets"] for seq in sequence_info],
                                               self.first_token_soft_weight)
            self._step_head_rows += head_rows

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

                # First-token accuracy: build local per-sequence boolean list
                local_first = [first_pred_list[j] in sequence_info[si]["valid_first_targets"]
                               for j, si in enumerate(first_si_list)]

                # Full-word accuracy: build local per-sequence boolean list
                is_search_cpu = is_search.cpu()
                local_full = []
                for si in range(S):
                    if not is_search_cpu[si]:
                        continue  # skip pretrain samples
                    s, e = starts[si], ends[si]
                    if s >= e:
                        continue
                    span_valid = valid_mask_cpu[s:e]
                    if span_valid.any():
                        local_full.append(pred_matches_cpu[s:e][span_valid].all().item())
                    # If no valid labels in span, skip (don't inflate accuracy)

                # All-gather predictions across ranks so the rolling buffers are
                # GPU-count independent (full eff_batch's predictions per microbatch).
                # Tiny payload (a few hundred bools), negligible overhead.
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    ws = torch.distributed.get_world_size()
                    g_first = [None] * ws
                    g_full = [None] * ws
                    torch.distributed.all_gather_object(g_first, local_first)
                    torch.distributed.all_gather_object(g_full, local_full)
                    self.first_token_correct.extend(b for sl in g_first for b in sl)
                    self.full_word_correct.extend(b for sl in g_full for b in sl)
                else:
                    self.first_token_correct.extend(local_first)
                    self.full_word_correct.extend(local_full)

            # NL_PARITY_DUMP=1: first micro-batch of each rank, for the old-vs-new parity test.
            if self._parity_dump and not self._parity_first_batch_done:
                self._parity_first_batch_done = True
                self._write_parity_dump("first_microbatch", {
                    "loss": float(loss.item()), "n_valid": int(n_valid), "tokens": int(self._last_batch_tokens),
                    "num_sequences": int(num_sequences), "h_sum": float(h.float().sum().item()),
                    "first_preds": [int(x) for x in first_pred_list],
                    "local_first": [bool(b) for b in local_first], "local_full": [bool(b) for b in local_full],
                    "search_loss": self._last_search_loss, "pretrain_loss": self._last_pretrain_loss,
                    "autocast": bool(torch.is_autocast_enabled()), "ckpt_every": int(ckpt_every),
                    "head_rows": int(head_rows), "n_preds_set": int((preds >= 0).sum().item()),
                })
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        self.recent_losses.append(loss.item())

        return loss

    def get_first_token_acc(self):
        return (sum(self.first_token_correct) / len(self.first_token_correct)) if self.first_token_correct else 0.0

    def get_full_word_acc(self):
        return (sum(self.full_word_correct) / len(self.full_word_correct)) if self.full_word_correct else 0.0
def run_eval_tf_loss(model, tokenizer, task: str, inputs: List[str], labels: List[List[str]], **kwargs) -> float:
    """Teacher-forced eval loss, run with SDPA (eval_attn_sdpa) whatever the training attention implementation is.
    Wrapping here covers every caller (stage/periodic evals); the body is unchanged in _run_eval_tf_loss_impl."""
    with eval_attn_sdpa(model):
        return _run_eval_tf_loss_impl(model, tokenizer, task, inputs, labels, **kwargs)


@torch.no_grad()
def _run_eval_tf_loss_impl(
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


def run_eval_greedy_readable(model, *args, **kwargs) -> Dict[str, Any]:
    """Greedy eval, run with SDPA (eval_attn_sdpa) whatever the training attention implementation is. Wrapping here
    covers every caller (stage/periodic, baseline, final evals); the body is unchanged in
    _run_eval_greedy_readable_impl."""
    with eval_attn_sdpa(model):
        return _run_eval_greedy_readable_impl(model, *args, **kwargs)


@torch.no_grad()
def _run_eval_greedy_readable_impl(
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
            skip_stage_alpha_eval: bool = False,
            eval_every_steps: int = 0,
            eval_inputs_hard: List[str] = None,
            eval_labels_hard: List[List[str]] = None,
            eval_fingerprint_hard: str = None,
            tokenizer=None,
            task: str = None,
            task_kwargs: dict = None,
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
        self.skip_stage_alpha_eval = skip_stage_alpha_eval
        self.eval_every_steps = eval_every_steps
        self.eval_inputs_hard = eval_inputs_hard
        self.eval_labels_hard = eval_labels_hard
        self.eval_fingerprint_hard = eval_fingerprint_hard
        self.tokenizer = tokenizer
        self.task = task
        self.task_kwargs = task_kwargs or {}
        self.max_input_size = max_input_size
        self.seed = seed
        self.print_examples = print_examples
        self.stage_eval_history = []

        # Loss tracking
        self.loss_history = []
        # Flops tracking
        self.flops_per_token = None
        self.flops_per_head_row = None  # 6 * N_head, set with flops_per_token; sparse-head achieved_tflops only
        self.cumulative_wall_time = 0.0  # Total wall time across resumes (seconds)
        self.wall_time_offset = 0.0  # Offset from previous runs
        self.training_start_time = None  # Set when training starts
        self.plot_metadata = None  # Set after trainer creation
        self.persist_every = persist_every
        self._last_persist_step = 0

        # Cumulative PFLOPS milestones: save checkpoint at each, then optionally stop
        self.pflops_milestones = []  # list of float PFLOPS thresholds (sorted)
        self.pflops_milestones_done = set()  # set of milestones already saved
        self.max_total_pflops = 0.0  # 0 = disabled
        self._cumulative_tokens = 0  # restored from loss_history on resume

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

        # PFLOPS milestone checkpoints + max_total_pflops stop
        if (self.pflops_milestones or self.max_total_pflops > 0) and self.flops_per_token:
            cum_pflops = self._cumulative_tokens * self.flops_per_token / 1e15
            for m in sorted(self.pflops_milestones):
                if m not in self.pflops_milestones_done and cum_pflops >= m:
                    m_dir = os.path.join(args.output_dir, "pflops_checkpoints", f"pflops_{int(m)}")
                    # If checkpoint already saved (e.g., from a prior run that crossed this milestone), skip.
                    if os.path.isdir(m_dir) and os.path.isfile(os.path.join(m_dir, "model.safetensors.index.json")) or \
                       os.path.isdir(m_dir) and os.path.isfile(os.path.join(m_dir, "model.safetensors")):
                        if is_main_process():
                            rank_print(f"[PFLOPS] Milestone {int(m)} already saved at {m_dir} — skipping")
                        self.pflops_milestones_done.add(m)
                        continue
                    try:
                        if is_main_process():
                            rank_print(f"[PFLOPS] Reached {cum_pflops:.0f} PFLOPS — saving checkpoint at milestone {int(m)} → {m_dir}")
                        self.trainer.save_model(m_dir)
                        barrier()
                        if is_main_process():
                            _save_curriculum_state(m_dir, self.dataset.stage,
                                                   self.stage_start_step, current_wall_time,
                                                   first_token_correct=self.trainer.first_token_correct,
                                                   full_word_correct=self.trainer.full_word_correct,
                                                   recent_losses=self.trainer.recent_losses,
                                                   samples_this_stage=self.samples_this_stage,
                                                   tokens_this_stage=self.tokens_this_stage)
                    except Exception as e:
                        rank_print(f"[PFLOPS] Warning: could not save milestone {int(m)}: {e}")
                    self.pflops_milestones_done.add(m)
            if self.max_total_pflops > 0 and cum_pflops >= self.max_total_pflops:
                if is_main_process():
                    rank_print(f"[PFLOPS] Reached {cum_pflops:.0f} PFLOPS ≥ max {self.max_total_pflops:.0f} — stopping training")
                # Force a FULL HF Trainer checkpoint so resume can extend max_total_pflops cleanly
                # without losing the last save_steps worth of training.
                control.should_save = True
                control.should_training_stop = True
                self.finished = True

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
                max_input_size=self.max_input_size,
                seed=self.seed, use_chat_template=_use_chat, **self.task_kwargs
            )

            # Greedy accuracy
            greedy_result = run_eval_greedy_readable(
                model, self.tokenizer, self.task,
                self.eval_inputs_hard, self.eval_labels_hard,
                max_input_size=self.max_input_size,
                seed=self.seed, print_examples=self.print_examples,
                use_chat_template=_use_chat, **self.task_kwargs
            )

            # Stage-alpha eval: generate and evaluate on stage-difficulty data
            stage_greedy_result = None
            stage_alpha = self.dataset._stage_alpha()
            max_alpha_L = self.dataset.task_kwargs.get("max_lookahead")
            is_full_difficulty = (target_L is not None and target_L >= (max_alpha_L or 256))
            if not is_full_difficulty and stage_alpha < 1.0 and not self.skip_stage_alpha_eval:
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
                        max_input_size=self.max_input_size,
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
            # Rows the lm_head actually processed this step (NL_HEAD=sparse: labelled rows + first rows of the soft
            # blend; full: every row + first rows). Summed over ranks like tokens; see estimate_flops_per_token.
            head_rows_this_step = (self.trainer._step_head_rows * get_world_size()
                                   if hasattr(self.trainer, '_step_head_rows') else 0)
            achieved_tflops = 0.0
            if hasattr(self.trainer, '_train_timing') and self.trainer._train_timing["steps"] > 0:
                recent_step_time = self.trainer._train_timing["total_step"] / self.trainer._train_timing["steps"]
                if recent_step_time > 0 and self.flops_per_token and tokens_this_step > 0:
                    # FLOPs executed this step: 6N * tokens for the full head (as always); under NL_HEAD=sparse the
                    # head ran on head_rows_this_step rows only (executed_flops). The cumulative PFLOPs accounting
                    # (_cumulative_tokens * flops_per_token) stays 6N * tokens, see estimate_flops_per_token.
                    flops_this_step = executed_flops(tokens_this_step, head_rows_this_step, self.flops_per_token,
                                                     self.flops_per_head_row, NL_HEAD)
                    achieved_tflops = flops_this_step / recent_step_time / 1e12

            entry = {
                "step": state.global_step,
                "loss": current_loss,
                "stage": self.dataset.stage,
                "alpha": self.dataset._stage_alpha(),
                "effective_L": effective_L,
                "tokens": tokens_this_step,
                "head_rows": head_rows_this_step,
                "head": NL_HEAD,
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
            self._cumulative_tokens += int(entry.get("tokens", 0))

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
            self.trainer._step_head_rows = 0

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
            if os.environ.get("NL_PARITY_DUMP", "0") == "1" and torch.cuda.is_available():
                # Parity-regime only (keeps production logs byte-identical): peak allocated memory so far.
                extra_info += f" | peak_mem={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB"

            # Format tokens/s with K suffix for readability
            tokens_per_sec_str = f"{tokens_per_sec / 1000:.1f}K" if tokens_per_sec >= 1000 else f"{tokens_per_sec:.0f}"

            achieved_tflops_str = ""
            if self.flops_per_token and tokens_per_sec > 0:
                # Same executed-FLOPs count as the loss_history achieved_tflops: per-token rate over this stage's
                # logged steps (sparse-head correction per entry via its head tag), times the stage's tokens/s.
                # Full-head entries contribute 6N exactly. Walks loss_history backwards to the stage start only.
                st_tokens, st_flops = 0, 0.0
                for h in reversed(self.loss_history):
                    if h.get("step", 0) <= self.stage_start_step:
                        break
                    t = int(h.get("tokens", 0) or 0)
                    st_tokens += t
                    st_flops += executed_flops(t, int(h.get("head_rows", t) or 0), self.flops_per_token,
                                               self.flops_per_head_row, _entry_head_mode(h))
                eff_flops_per_token = st_flops / st_tokens if st_tokens > 0 else self.flops_per_token
                achieved_tflops = tokens_per_sec * eff_flops_per_token / 1e12
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
                    # Force a FULL HF Trainer checkpoint (model + optimizer + scheduler + rng + trainer_state)
                    # so future runs can resume training cleanly without losing optimizer state.
                    # Stage_checkpoints/ above is model-only (via save_model); this gives a full resume target.
                    control.should_save = True
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
        reserved_inputs: Set[str],
        seed: Optional[int],
        **task_kwargs,
) -> Tuple[List[str], List[List[str]], List[str]]:
    g = NaturalLanguageGraphGenerator(max_input_size, seed=seed)
    eval_inputs, eval_labels, picked_answers = [], [], []

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

        prompt_ids = tokenizer(ex.input_text, add_special_tokens=True, truncation=False)["input_ids"]
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


class ParamSyncDebugCallback(TrainerCallback):
    """NL_DEBUG_PARAM_SYNC=1 (2026-10-09): after every optimizer step, all-gather a float64 checksum of the
    model parameters and report the spread across ranks. Zero spread = the ranks hold identical weights, i.e.
    DDP gradient averaging is in effect; nonzero = each rank is training its own replica (the custom forward
    bypasses DistributedDataParallel.forward, so the reducer's hooks are never armed)."""
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not dist_is_initialized() or model is None:
            return
        m = model.module if hasattr(model, "module") else model
        # NL_PARITY_DUMP=1 additionally records the per-parameter float64 sums per rank (same terms, same order).
        per_param = {} if os.environ.get("NL_PARITY_DUMP", "0") == "1" else None
        with torch.no_grad():
            dev = next(m.parameters()).device
            s = torch.zeros((), dtype=torch.float64, device=dev)
            for n, p in m.named_parameters():
                ps = p.detach().double().sum()
                s += ps
                if per_param is not None:
                    per_param[n] = ps.item()
        gathered = [torch.zeros_like(s) for _ in range(get_world_size())]
        torch.distributed.all_gather(gathered, s)
        vals = [g.item() for g in gathered]
        if is_main_process():
            spread = max(vals) - min(vals)
            # Line format is parsed by bench/compare_packing_bench.py (param_sync_lines); keep it unchanged.
            print(f"[PARAM-SYNC] step {state.global_step}: per-rank param sums "
                  f"{['%.6f' % v for v in vals]} | spread {spread:.3e} -> "
                  f"{'IN SYNC' if spread == 0 else 'DIVERGED'}", flush=True)
        if per_param is not None:
            try:
                d = os.path.join(args.output_dir, "parity_dump")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f"param_sums_rank{get_rank()}.jsonl"), "a") as f:
                    f.write(json.dumps({"step": int(state.global_step), "sums": per_param}) + "\n")
            except Exception as e:
                print(f"[PARITY][WARN] rank {get_rank()}: param sums dump failed: {e}", flush=True)


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    rank_print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())
    rank_print("PyTorch version:", torch.__version__)

    import argparse

    p = argparse.ArgumentParser()

    # Task/model
    p.add_argument("--task", type=str, choices=["search"], default="search")
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

    # Training hyperparams
    p.add_argument("--batch_size", type=int, default=16,
                   help="Samples per training step. With packing: target sequences packed per step. Without: per_device_train_batch_size.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    # LR scheduling is handled by --stage_schedule (per-stage cosine/warmup_reset/SGDR). The global
    # lr_scheduler_type is always "constant", with no global warmup — do not change.
    p.add_argument("--first_token_soft_weight", type=float, default=0.3)

    # Curriculum
    p.add_argument("--n_stages", type=int, default=10)
    p.add_argument("--base_alpha", type=float, default=0.1)
    p.add_argument("--max_alpha", type=float, default=1.0, help="Maximum alpha during training (eval always uses 1.0)")
    p.add_argument("--linear_lookahead", action="store_true",
                   help="Use linear lookahead curriculum (search task only)")
    p.add_argument("--base_lookahead", type=int, default=1,
                   help="Starting lookahead at stage 1 (linear_lookahead mode)")
    p.add_argument("--breadth", type=int, default=2,
                   help="Edge-budget multiplier per unit lookahead. 2 (default) = one goal "
                        "path + one equal-length distractor at max depth; 3/4/5 widen the "
                        "deepest instances to that many branches and scale the vertex-ID "
                        "pool to breadth*L+2. Caps reachable L at ((n-5)//3-1)//breadth.")
    p.add_argument("--shuffled_mixture", type=str, default=None,
                   help="Exposure-matched shuffled control: 'L:steps,L:steps,...' from a curriculum run. "
                        "Every search example draws its lookahead L at random, weighted by the steps the "
                        "curriculum spent at L, so the run sees the curriculum's multiset in random order. "
                        "Combine with an accuracy_threshold above 1 so the stage never advances.")
    p.add_argument("--max_train_steps", type=int, default=0,
                   help="Stop after this many optimizer steps (0 = no step limit, the default).")
    p.add_argument("--lookahead_step", type=int, default=1,
                   help="Lookahead increase per stage (linear_lookahead mode)")
    p.add_argument("--accuracy_threshold", type=float, default=0.98)
    p.add_argument("--min_steps_per_stage", type=int, default=500)
    p.add_argument("--check_every", type=int, default=50)
    p.add_argument("--accuracy_window", type=int, default=1000,
                   help="Rolling window size (TOTAL across ranks, after all-gather) for advancement accuracy check. Default 1000 gives ~±0.9pp noise at p=0.98.")

    # Task params
    p.add_argument("--max_input_size", type=int, default=256)
    p.add_argument("--max_lookahead", type=int, default=12)
    p.add_argument("--vocab_pool", type=str, default="none", choices=["none", "grow", "fixed"],
                   help="Entity-vocabulary curriculum: map each symbolic vertex ID to one fixed attribute name. grow = the ID range follows the stage (2(L+1) names at stage L; ID 0 is reserved); fixed = the ID range is pinned at the context maximum ((n-5)//3+1 names) throughout; none = fresh random names per instance (historical behaviour).")
    p.add_argument("--fixed_vocab", action="store_true",
                   help="Search ablation: keep max_vertex_id at its un-alpha-scaled max so the vertex-ID vocab does not scale with curriculum (graph size still scales). Search task only.")

    # Eval sizes / printing
    p.add_argument("--eval_samples", type=int, default=1000)
    p.add_argument("--print_eval_examples", type=int, default=0)

    # Eval flags
    p.add_argument("--do_baseline", action="store_true", help="Run pre-training baseline eval")
    p.add_argument("--do_final_eval", action="store_true", help="Run post-training TF + greedy eval")
    p.add_argument("--stage_eval_every", type=int, default=1,
                   help="Run stage eval every N lookahead units (e.g. 8 = eval only when L is multiple of 8)")
    p.add_argument("--do_stage_eval", action="store_true",
                   help="Run TF+greedy eval at alpha=1.0 after each stage advancement")
    p.add_argument("--skip_stage_alpha_eval", action="store_true",
                   help="Skip the secondary eval at the current stage's alpha (only run alpha=1.0 eval)")
    p.add_argument("--eval_every_steps", type=int, default=0,
                   help="Run greedy eval every N steps (0=disabled, useful for no-curriculum runs)")
    p.add_argument("--save_steps", type=int, default=500,
                   help="HF Trainer checkpoint interval (rolling checkpoint-* dirs); also the granularity at which --max_total_pflops is checked.")
    p.add_argument("--persist_every", type=int, default=2000,
                   help="Save persistent checkpoint every N steps (not rotated by save_total_limit). 0=disabled.")
    p.add_argument("--save_pflops_milestones", type=str, default="",
                   help="Comma-separated PFLOPS values; save persistent checkpoint when cumulative compute first crosses each. "
                        "Saved to pflops_checkpoints/pflops_<int>/. e.g. '5362,25273,83770'.")
    p.add_argument("--max_total_pflops", type=float, default=0.0,
                   help="Stop training when cumulative compute reaches this many PFLOPS. 0=disabled.")
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

    # Chunked cross-entropy: always on. compute_loss applies the lm_head in --ce_chunk_size row slices
    # (NL_HEAD=full: over every packed row; NL_HEAD=sparse: over the labelled rows, usually one slice; see
    # _head_ce_and_preds). The old no-op --use_chunked_ce flag was removed 2026-10-09; job scripts outside
    # archive/ were updated to stop passing it.
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
    p.add_argument("--optim", type=str, default="adamw_torch_fused",
                   help="Optimizer name passed to HF TrainingArguments (e.g. adamw_torch_fused, adamw_bnb_8bit, paged_adamw_8bit)")

    global args, NL_TRAIN_ATTN_IMPL
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
            timeout=datetime.timedelta(minutes=int(os.environ.get("NL_DDP_TIMEOUT_MIN", "30")))
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
    }
    if NL_PACKING == "hf":
        # The training forward runs the HF body through transformers' packed flash-attention path, so the kernel is
        # chosen at load time (NL_ATTN_KERNEL: auto = FA3 on Hopper, else FA2). In-run evals are pinned to SDPA by
        # eval_attn_sdpa, i.e. they stay on the transformers default that eval_checkpoints.py uses as well.
        attn_impl_req = select_attn_implementation()
        model_kwargs["attn_implementation"] = attn_impl_req
        rank_print(f"[ATTN-KERNEL] NL_PACKING=hf NL_ATTN_KERNEL={_ATTN_KERNEL_REQ} -> "
                   f"attn_implementation={attn_impl_req} (hopper={_is_hopper()}); in-run evals use sdpa")
    else:
        # NL_PACKING=custom: no attn_implementation, i.e. Hugging Face's default (SDPA) wherever the HF forward runs
        # (the in-run evals), exactly as the paper runs loaded the model. Training never goes through it:
        # compute_loss runs legacy_varlen_forward with its own FA3/FA2 kernel. (Until 2026-10-09 this was
        # flash_attention_2 here and sdpa in tuning_nl_fa3.py.)
        attn_impl_req = None
        rank_print(f"[ATTN-KERNEL] NL_PACKING=custom NL_ATTN_KERNEL={_ATTN_KERNEL_REQ}: model loads with the "
                   f"transformers default attention (sdpa); training uses legacy_varlen_forward "
                   f"FA{_legacy.FLASH_ATTN_VERSION}")

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

    # Record the training attention implementation (eval_attn_sdpa restores and asserts it) and, in hf mode, check
    # that the load honoured the request and that per-layer selective checkpointing can be applied.
    NL_TRAIN_ATTN_IMPL = getattr(model.config, "_attn_implementation", None)
    if NL_PACKING == "hf":
        assert NL_TRAIN_ATTN_IMPL == attn_impl_req, \
            f"model loaded with attn_implementation={NL_TRAIN_ATTN_IMPL!r}, requested {attn_impl_req!r}"
        from transformers.modeling_layers import GradientCheckpointingLayer
        _body = resolve_model_parts(model)
        _not_gcl = [type(l).__name__ for l in _body['inner'].layers if not isinstance(l, GradientCheckpointingLayer)]
        assert not _not_gcl, ("per-layer selective checkpointing needs every decoder layer to be a "
                              f"transformers GradientCheckpointingLayer; found {_not_gcl}")
        # The config string says what was requested; check which varlen kernel transformers actually imports for it
        # (FA3 lives in the top-level `flash_attn_interface` module, FA2 in `flash_attn.flash_attn_interface`).
        import transformers.modeling_flash_attention_utils as _mfa
        _mfa.lazy_import_flash_attention(NL_TRAIN_ATTN_IMPL, force_import=True)
        _vfn = _mfa._flash_varlen_fn
        _vmod = getattr(_vfn, "__module__", "") or ""
        _want = "flash_attn_interface" if NL_TRAIN_ATTN_IMPL == "flash_attention_3" else "flash_attn"
        assert _vfn is not None and _vmod.split(".")[0] == _want, \
            f"transformers imported the varlen kernel from {_vmod!r}; expected package {_want!r} for {NL_TRAIN_ATTN_IMPL}"
        rank_print(f"[ATTN-KERNEL] verified: config._attn_implementation={NL_TRAIN_ATTN_IMPL}; varlen kernel from "
                   f"{_vmod}; {len(_body['inner'].layers)} {_body['arch']} decoder layers are GradientCheckpointingLayer")

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

    # Task kwargs (search is the only task; the dfs/si branches were removed 2026-10-09)
    task_kwargs = {"max_lookahead": args.max_lookahead, "fixed_vocab": args.fixed_vocab,
                   "vocab_pool": getattr(args, "vocab_pool", "none")}

    # ==================== GENERATE EVAL DATA ONCE ====================
    eval_inputs_hard, eval_labels_hard = None, None  # alpha=1.0 (hardest)
    eval_inputs_easy, eval_labels_easy = None, None  # alpha=base_alpha (easiest)
    eval_fingerprint_hard, eval_fingerprint_easy = None, None

    need_eval_data = (args.do_baseline or args.do_final_eval or args.do_stage_eval)

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
        seed=args.seed,
        resume_step=resume_step * args.gradient_accumulation_steps,
        linear_lookahead=args.linear_lookahead,
        base_lookahead=args.base_lookahead,
        lookahead_step=args.lookahead_step,
        breadth=getattr(args, "breadth", 2),
        shuffled_mixture=_parse_lookahead_mixture(getattr(args, "shuffled_mixture", None)),
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
        skip_stage_alpha_eval=args.skip_stage_alpha_eval,
        stage_eval_every=getattr(args, 'stage_eval_every', 1),
        eval_every_steps=args.eval_every_steps,
        eval_inputs_hard=eval_inputs_hard,
        eval_labels_hard=eval_labels_hard,
        eval_fingerprint_hard=eval_fingerprint_hard,
        tokenizer=tokenizer,
        task=args.task,
        task_kwargs=task_kwargs,
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

    # Wire up PFLOPS milestones + max_total_pflops from CLI args
    if getattr(args, 'save_pflops_milestones', '').strip():
        try:
            curriculum.pflops_milestones = sorted({float(x) for x in args.save_pflops_milestones.split(',') if x.strip()})
            rank_print(f"[PFLOPS] Will save milestones at: {curriculum.pflops_milestones}")
        except Exception as e:
            rank_print(f"[PFLOPS] Failed to parse --save_pflops_milestones={args.save_pflops_milestones!r}: {e}")
    if getattr(args, 'max_total_pflops', 0.0) > 0:
        curriculum.max_total_pflops = float(args.max_total_pflops)
        rank_print(f"[PFLOPS] Will stop training at {curriculum.max_total_pflops} PFLOPS")

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

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=effective_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",  # per-stage decay handled by --stage_schedule
        max_steps=(args.max_train_steps if getattr(args, "max_train_steps", 0) > 0 else 100000000),
        num_train_epochs=1000,
        logging_steps=10,
        logging_first_step=False,
        report_to="none",
        save_strategy="steps",
        save_steps=int(getattr(args, 'save_steps', 500)),
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        optim=args.optim,
        dataloader_num_workers=num_workers,
        # One __getitem__ produces an ENTIRE packed micro-batch, and generation time varies a lot
        # because sampling is rejection-based (up to 100 inner retries per sample). A depth of 2 is
        # therefore a thin buffer against that variance. 4 costs only host RAM. NL_PREFETCH_FACTOR
        # overrides it for probing.
        dataloader_prefetch_factor=(int(os.environ.get("NL_PREFETCH_FACTOR", "4"))
                                    if num_workers > 0 else None),
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
            max_input_size=args.max_input_size, seed=args.seed,
            print_examples=args.print_eval_examples, **task_kwargs
        )

        # Easy eval: alpha=base_alpha
        base_easy = run_eval_greedy_readable(
            eval_model, tokenizer, args.task,
            eval_inputs_easy, eval_labels_easy,
            max_input_size=args.max_input_size, seed=args.seed,
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
        callbacks=[curriculum, baseline_cb] + ([ParamSyncDebugCallback()] if os.environ.get("NL_DEBUG_PARAM_SYNC") == "1" else []),
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
    # N_head for the sparse-head achieved_tflops correction (executed_flops); cumulative PFLOPs stay 6N * tokens.
    _n_head = estimate_head_params(model)
    curriculum.flops_per_head_row = 6 * _n_head if _n_head > 0 else None
    if _n_head > 0:
        rank_print(f"[FLOPS] Output head: N_head={_n_head / 1e6:.1f}M params, 6*N_head={6 * _n_head / 1e9:.2f}B FLOPs per "
                   f"head row; NL_HEAD={NL_HEAD}: achieved_tflops counts executed FLOPs "
                   f"({'6(N-N_head)*tokens + 6*N_head*head_rows' if NL_HEAD == 'sparse' else '6N*tokens, as always'})")
    else:
        rank_print("[FLOPS][WARN] could not resolve the output head; achieved_tflops falls back to 6N*tokens "
                   f"(overstates NL_HEAD={NL_HEAD} if sparse)")

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
                             trainer.args.gradient_accumulation_steps,
                             extra=_run_provenance(model))

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
                    "provenance": _run_provenance(model),
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
            max_input_size=args.max_input_size, seed=args.seed,
            print_examples=min(3, args.print_eval_examples), **task_kwargs
        )
        rank_print(
            f"[FINAL-GREEDY-alpha1.0] First={greedy_hard['first_token_acc']:.2%} | Full={greedy_hard['full_word_acc']:.2%} | N={greedy_hard['total']}")

        greedy_easy = run_eval_greedy_readable(
            trainer.model, tokenizer, args.task,
            eval_inputs_easy, eval_labels_easy,
            max_input_size=args.max_input_size, seed=args.seed,
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