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

import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset

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

def _save_curriculum_state(dirpath: str, stage: int, stage_start_step: int, wall_time_offset: float = 0.0) -> None:
    try:
        os.makedirs(dirpath, exist_ok=True)
        fp = os.path.join(dirpath, "curriculum_state.json")
        with open(fp, "w") as f:
            json.dump({
                "stage": int(stage),
                "stage_start_step": int(stage_start_step),
                "wall_time_offset": float(wall_time_offset),
            }, f)
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
        curriculum.wall_time_offset = float(cs.get("wall_time_offset", 0.0))
        rank_print(
            f"[CURRICULUM] Restored stage={dataset.stage}, stage_start_step={curriculum.stage_start_step}, wall_time_offset={curriculum.wall_time_offset:.1f}s from {fp}")
    except Exception as e:
        rank_print(f"[CURRICULUM][WARN] Failed to restore from {fp}: {e}")
        return False

    # Restore loss history
    loss_path = os.path.join(os.path.dirname(path), "loss_history.json")
    if os.path.isfile(loss_path):
        try:
            with open(loss_path) as f:
                curriculum.loss_history = json.load(f)
            rank_print(f"[CURRICULUM] Restored {len(curriculum.loss_history)} loss history records")

            # Update wall_time_offset from last recorded entry if available
            if curriculum.loss_history:
                last_wall_time = curriculum.loss_history[-1].get("wall_time", 0.0)
                if last_wall_time > curriculum.wall_time_offset:
                    curriculum.wall_time_offset = last_wall_time
                    rank_print(f"[CURRICULUM] Updated wall_time_offset to {last_wall_time:.1f}s from loss history")
        except Exception as e:
            rank_print(f"[CURRICULUM][WARN] Failed to restore loss history: {e}")

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


def fit_exponential_decay(steps: List[int], losses: List[float], window: int = 50, skip_initial_spike: bool = True) -> \
Optional[Dict[str, float]]:
    """
    Fit exponential decay: L(t) = a * exp(-b * t) + c

    Args:
        skip_initial_spike: If True, start fitting from the peak loss (skips initial rise)

    Returns dict with:
        - a: initial amplitude above asymptote
        - b: decay rate
        - c: asymptote (estimated minimum loss)
        - r_squared: goodness of fit
        - half_life: steps to decay halfway to asymptote
    Returns None if fit fails.
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        rank_print("[FIT] scipy not installed, skipping exponential fit")
        return None

    if len(losses) < 20:
        return None

    fit_steps = list(steps)
    fit_losses = list(losses)

    # Skip initial spike - find peak and start from there
    if skip_initial_spike and len(fit_losses) > 30:
        # Look for peak in first 30% of stage
        search_range = max(10, len(fit_losses) // 3)
        peak_idx = np.argmax(fit_losses[:search_range])

        # Only skip if peak isn't at the very start (i.e., there was a spike)
        if peak_idx > 2:
            fit_steps = fit_steps[peak_idx:]
            fit_losses = fit_losses[peak_idx:]

    if len(fit_losses) < 15:
        return None

    # Smooth the data
    smooth_window = min(window, len(fit_losses) // 3) or 1
    if len(fit_losses) >= smooth_window:
        smoothed = np.convolve(fit_losses, np.ones(smooth_window) / smooth_window, mode='valid')
        smooth_steps = fit_steps[smooth_window - 1:]
    else:
        smoothed = np.array(fit_losses)
        smooth_steps = fit_steps

    if len(smoothed) < 10:
        return None

    # Normalize steps to start at 0
    t = np.array(smooth_steps, dtype=float) - smooth_steps[0]
    y = np.array(smoothed, dtype=float)

    def exp_decay(t, a, b, c):
        return a * np.exp(-b * t) + c

    # Initial guesses
    a0 = max(y[0] - y[-1], 0.01)
    c0 = max(y[-1], 0.001)
    b0 = 1.0 / max(len(t) / 3, 1)

    try:
        popt, _ = curve_fit(
            exp_decay, t, y,
            p0=[a0, b0, c0],
            bounds=([0, 1e-10, 0], [np.inf, 1.0, np.inf]),
            maxfev=5000
        )
        a, b, c = popt

        # Calculate R-squared
        y_pred = exp_decay(t, a, b, c)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Half-life: time to decay halfway to asymptote
        half_life = np.log(2) / b if b > 0 else float('inf')

        return {
            "a": a,
            "b": b,
            "c": c,
            "r_squared": r_squared,
            "half_life": half_life,
            "fit_start_step": smooth_steps[0],  # Where fit starts (after spike)
            "fit_steps": smooth_steps,
            "fit_values": exp_decay(t, a, b, c).tolist(),
        }
    except Exception as e:
        rank_print(f"[FIT] Exponential fit failed: {e}")
        return None


def _build_plot_subtitle(metadata: Dict) -> str:
    """Build a subtitle string from plot metadata."""
    parts = []

    if metadata.get("model_name"):
        parts.append(f"Model: {metadata['model_name']}")

    if metadata.get("model_params_b"):
        parts.append(f"{metadata['model_params_b']:.2f}B params")

    if metadata.get("use_packing"):
        parts.append(f"Packed (target={metadata.get('target_samples', '?')})")
    else:
        parts.append(f"BS={metadata.get('batch_size', '?')}")

    if metadata.get("learning_rate"):
        parts.append(f"LR={metadata['learning_rate']:.1e}")

    if metadata.get("accuracy_threshold"):
        parts.append(f"Acc≥{metadata['accuracy_threshold']:.0%}")

    return " | ".join(parts)


def plot_stage_loss(loss_history: List[Dict], stage: int, output_dir: str, metadata: Dict = None):
    """Plot loss curve for a single completed stage with exponential fit."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        rank_print("[PLOT] matplotlib not installed, skipping plot")
        return

    stage_data = [h for h in loss_history if h["stage"] == stage]
    if not stage_data:
        return

    steps = [h["step"] for h in stage_data]
    losses = [h["loss"] for h in stage_data]

    alpha = stage_data[0].get("alpha", None)
    effective_L = stage_data[0].get("effective_L", None)

    start_step = steps[0]
    rel_steps = [s - start_step for s in steps]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rel_steps, losses, alpha=0.3, linewidth=0.5, color='blue', label='Raw')

    # Smoothed line
    window = min(200, len(losses) // 3) or 1
    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(losses)), smoothed, linewidth=2, color='blue', label=f'Smoothed (w={window})')

    # Exponential fit
    fit_result = fit_exponential_decay(rel_steps, losses, window=window)
    fit_text = ""
    if fit_result and fit_result["r_squared"] > 0.8:
        # Show where fit starts (after spike)
        fit_start = fit_result["fit_start_step"]  # Already relative since we passed rel_steps
        ax.axvline(x=fit_start, color='orange', linestyle=':', alpha=0.5, label='Fit start')

        # Draw fit curve
        ax.plot(fit_result["fit_steps"], fit_result["fit_values"], '--', linewidth=2, color='red',
                label=f'Exp fit (R²={fit_result["r_squared"]:.3f})')

        # Asymptote line
        ax.axhline(y=fit_result["c"], color='green', linestyle=':', alpha=0.7,
                   label=f'Asymptote={fit_result["c"]:.4f}')

        fit_text = (f'Fit: L(t) = {fit_result["a"]:.3f}·e^(-{fit_result["b"]:.2e}·t) + {fit_result["c"]:.4f}\n'
                    f'Est. minimum: {fit_result["c"]:.4f} | Half-life: {fit_result["half_life"]:.0f} steps | R²={fit_result["r_squared"]:.3f}')
    elif fit_result:
        fit_text = f'Exp fit poor (R²={fit_result["r_squared"]:.3f})'

    ax.set_xlabel('Steps in Stage')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')

    title = f'Stage {stage} Loss ({len(stage_data)} steps, final={losses[-1]:.4f})'
    if effective_L is not None:
        title += f' | L={effective_L}'
    if alpha is not None:
        title += f' | α={alpha:.3f}'
    ax.set_title(title)

    if fit_text:
        ax.text(0.02, 0.02, fit_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    if metadata:
        subtitle = _build_plot_subtitle(metadata)
        fig.suptitle(subtitle, fontsize=9, color='gray', y=0.02)

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(output_dir, f"loss_stage_{stage}.png"), dpi=150)
    plt.close()

    if fit_result and fit_result["r_squared"] > 0.8:
        rank_print(
            f"[PLOT] Saved loss_stage_{stage}.png | Est. min={fit_result['c']:.4f}, half-life={fit_result['half_life']:.0f} steps")
    else:
        rank_print(f"[PLOT] Saved loss_stage_{stage}.png")


def plot_overall_loss(loss_history: List[Dict], output_dir: str, n_stages: int, metadata: Dict = None):
    """Plot full training loss with stage markers and exponential fits."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not loss_history:
        return

    steps = [h["step"] for h in loss_history]
    losses = [h["loss"] for h in loss_history]
    stages = [h["stage"] for h in loss_history]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, losses, alpha=0.3, linewidth=0.5, color='blue')

    # Smoothed
    window = min(100, len(losses) // 10) or 1
    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
        ax.plot(steps[window - 1:], smoothed, linewidth=2, color='blue', label=f'Smoothed (w={window})')

    # Find stage transitions and their effective_L values
    stage_info = {}  # stage -> (first_step, effective_L, alpha)
    for h in loss_history:
        s = h["stage"]
        if s not in stage_info:
            stage_info[s] = (h["step"], h.get("effective_L"), h.get("alpha"))

    # Stage transitions with lookahead labels
    max_stage = max(stages)
    colors = plt.cm.tab10(np.linspace(0, 1, max(max_stage, n_stages)))

    prev_stage = stages[0]
    for i, (step, stage) in enumerate(zip(steps, stages)):
        if stage != prev_stage:
            ax.axvline(x=step, color=colors[(stage - 1) % 10], linestyle='--', alpha=0.7)

            # Get effective_L for this stage
            effective_L = stage_info.get(stage, (None, None, None))[1]
            if effective_L is not None:
                label = f'S{stage}\nL={effective_L}'
            else:
                label = f'S{stage}'

            ax.text(step, ax.get_ylim()[1] * 0.95, label, fontsize=8, ha='left', va='top')
            prev_stage = stage

    # Fit exponential to each stage and collect asymptotes
    stage_fits = {}
    stages_seen = sorted(set(stages))
    for stg in stages_seen:
        stage_data = [h for h in loss_history if h["stage"] == stg]
        if len(stage_data) < 20:
            continue
        stg_steps = [h["step"] for h in stage_data]
        stg_losses = [h["loss"] for h in stage_data]
        fit = fit_exponential_decay(stg_steps, stg_losses, window=min(50, len(stg_losses) // 5) or 1)
        if fit and fit["r_squared"] > 0.8:
            stage_fits[stg] = fit
            # Draw asymptote segment for this stage
            start_step = stg_steps[0]
            end_step = stg_steps[-1]
            ax.hlines(y=fit["c"], xmin=start_step, xmax=end_step,
                      colors='green', linestyles=':', alpha=0.6, linewidth=1.5)

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_title('Training Loss (Overall)')

    # Build fit summary text
    if stage_fits:
        fit_lines = ["Stage asymptotes:"]
        for stg in sorted(stage_fits.keys()):
            f = stage_fits[stg]
            L = stage_info.get(stg, (None, None))[1]
            L_str = f"L={L}" if L else ""
            fit_lines.append(f"  S{stg} {L_str}: min≈{f['c']:.4f}")
        fit_text = "\n".join(fit_lines)
        ax.text(0.98, 0.98, fit_text, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', horizontalalignment='right', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add subtitle with metadata
    if metadata:
        subtitle = _build_plot_subtitle(metadata)
        fig.suptitle(subtitle, fontsize=9, color='gray', y=0.02)

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, "loss_overall.png"), dpi=150)
    plt.close()

    rank_print(f"[PLOT] Saved loss_overall.png ({len(stage_fits)} stages with exp fits)")


def plot_loss_vs_flops(loss_history: List[Dict], output_dir: str, flops_per_token: int, n_stages: int,
                       metadata: Dict = None):
    """Plot loss vs cumulative FLOPs."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not loss_history or flops_per_token is None:
        return

    # Calculate cumulative FLOPs
    cumulative_flops = []
    total_flops = 0
    for h in loss_history:
        tokens = h.get("tokens", 0)
        total_flops += tokens * flops_per_token
        cumulative_flops.append(total_flops)

    losses = [h["loss"] for h in loss_history]
    stages = [h["stage"] for h in loss_history]

    # Convert to PetaFLOPs for readability
    cumulative_pflops = [f / 1e15 for f in cumulative_flops]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cumulative_pflops, losses, alpha=0.3, linewidth=0.5, color='blue')

    # Smoothed
    window = min(100, len(losses) // 10) or 1
    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
        ax.plot(cumulative_pflops[window - 1:], smoothed, linewidth=2, color='blue', label=f'Smoothed (w={window})')

    # Stage transitions
    stage_info = {}
    for i, h in enumerate(loss_history):
        s = h["stage"]
        if s not in stage_info:
            stage_info[s] = (cumulative_pflops[i], h.get("effective_L"))

    prev_stage = stages[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(max(stages), n_stages)))

    for i, (pflops, stage) in enumerate(zip(cumulative_pflops, stages)):
        if stage != prev_stage:
            ax.axvline(x=pflops, color=colors[(stage - 1) % 10], linestyle='--', alpha=0.7)
            effective_L = stage_info.get(stage, (None, None))[1]
            label = f'S{stage}\nL={effective_L}' if effective_L else f'S{stage}'
            ax.text(pflops, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else losses[0],
                    label, fontsize=8, ha='left', va='top')
            prev_stage = stage

    ax.set_xlabel('Cumulative PFLOPs')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_title(f'Training Loss vs Compute (6N approx, {flops_per_token / 1e9:.1f}B FLOPs/token)')

    # Add subtitle with metadata
    if metadata:
        subtitle = _build_plot_subtitle(metadata)
        fig.suptitle(subtitle, fontsize=9, color='gray', y=0.02)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, "loss_vs_flops.png"), dpi=150)
    plt.close()

    rank_print(f"[PLOT] Saved loss_vs_flops.png")


def plot_loss_vs_walltime(loss_history: List[Dict], output_dir: str, n_stages: int, metadata: Dict = None):
    """Plot loss vs wall clock time."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not loss_history:
        return

    # Check if wall_time field exists
    if not any(h.get("wall_time", 0) > 0 for h in loss_history):
        rank_print("[PLOT] No wall_time data available, skipping wall clock plot")
        return

    wall_times = [h.get("wall_time", 0) / 3600 for h in loss_history]  # Convert to hours
    losses = [h["loss"] for h in loss_history]
    stages = [h["stage"] for h in loss_history]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wall_times, losses, alpha=0.3, linewidth=0.5, color='blue')

    # Smoothed
    window = min(100, len(losses) // 10) or 1
    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
        ax.plot(wall_times[window - 1:], smoothed, linewidth=2, color='blue', label=f'Smoothed (w={window})')

    # Stage transitions
    stage_info = {}
    for i, h in enumerate(loss_history):
        s = h["stage"]
        if s not in stage_info:
            stage_info[s] = (wall_times[i], h.get("effective_L"))

    prev_stage = stages[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(max(stages), n_stages)))

    for i, (wt, stage) in enumerate(zip(wall_times, stages)):
        if stage != prev_stage:
            ax.axvline(x=wt, color=colors[(stage - 1) % 10], linestyle='--', alpha=0.7)
            effective_L = stage_info.get(stage, (None, None))[1]
            label = f'S{stage}\nL={effective_L}' if effective_L else f'S{stage}'
            ax.text(wt, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else losses[0],
                    label, fontsize=8, ha='left', va='top')
            prev_stage = stage

    ax.set_xlabel('Wall Clock Time (hours)')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_title('Training Loss vs Wall Time')

    # Add subtitle with metadata
    if metadata:
        subtitle = _build_plot_subtitle(metadata)
        fig.suptitle(subtitle, fontsize=9, color='gray', y=0.02)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, "loss_vs_walltime.png"), dpi=150)
    plt.close()

    rank_print(f"[PLOT] Saved loss_vs_walltime.png")


def plot_achieved_tflops(loss_history: List[Dict], output_dir: str, n_stages: int, metadata: Dict = None):
    """Plot achieved TFLOPs/s over training."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not loss_history:
        return

    # Check if achieved_tflops field exists
    if not any(h.get("achieved_tflops", 0) > 0 for h in loss_history):
        rank_print("[PLOT] No achieved_tflops data available, skipping TFLOPs plot")
        return

    steps = [h["step"] for h in loss_history]
    tflops = [h.get("achieved_tflops", 0) for h in loss_history]
    stages = [h["stage"] for h in loss_history]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, tflops, alpha=0.3, linewidth=0.5, color='green')

    # Smoothed
    window = min(100, len(tflops) // 10) or 1
    if len(tflops) >= window:
        smoothed = np.convolve(tflops, np.ones(window) / window, mode='valid')
        ax.plot(steps[window - 1:], smoothed, linewidth=2, color='green', label=f'Smoothed (w={window})')

    # Stage transitions
    prev_stage = stages[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(max(stages), n_stages)))

    for i, (step, stage) in enumerate(zip(steps, stages)):
        if stage != prev_stage:
            ax.axvline(x=step, color=colors[(stage - 1) % 10], linestyle='--', alpha=0.7)
            prev_stage = stage

    ax.set_xlabel('Step')
    ax.set_ylabel('Achieved TFLOPs/s')
    ax.set_title('GPU Compute Throughput Over Training')

    # Add average line
    avg_tflops = np.mean([t for t in tflops if t > 0])
    ax.axhline(y=avg_tflops, color='red', linestyle=':', alpha=0.7, label=f'Avg: {avg_tflops:.1f}')

    # Add subtitle with metadata
    if metadata:
        subtitle = _build_plot_subtitle(metadata)
        fig.suptitle(subtitle, fontsize=9, color='gray', y=0.02)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, "achieved_tflops.png"), dpi=150)
    plt.close()

    rank_print(f"[PLOT] Saved achieved_tflops.png (avg: {avg_tflops:.1f} TFLOPs/s)")


def plot_stage_eval(stage_eval_history: List[Dict], output_dir: str, metadata: Dict = None):
    """Plot held-out eval loss and accuracy vs effective L after each stage."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not stage_eval_history or len(stage_eval_history) < 2:
        return

    stages = [h["stage"] for h in stage_eval_history]
    greedy_first = [h["greedy_first"] for h in stage_eval_history]
    greedy_full = [h["greedy_full"] for h in stage_eval_history]
    has_loss = "tf_loss" in stage_eval_history[0]
    has_L = "effective_L" in stage_eval_history[0]

    # Use effective_L for x-axis if available, else stage number
    if has_L:
        x_vals = [h["effective_L"] for h in stage_eval_history]
        x_label = "Effective Lookahead (L)"
    else:
        x_vals = stages
        x_label = "Stage"

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left axis: accuracy
    color_first = '#2196F3'
    color_full = '#4CAF50'
    ax1.plot(x_vals, greedy_first, 'o-', color=color_first, label='Greedy First Token', markersize=5)
    ax1.plot(x_vals, greedy_full, 's-', color=color_full, label='Greedy Full Word', markersize=5)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    if has_loss:
        # Right axis: TF loss
        ax2 = ax1.twinx()
        tf_losses = [h["tf_loss"] for h in stage_eval_history]
        color_loss = '#F44336'
        ax2.plot(x_vals, tf_losses, '^--', color=color_loss, label='TF Loss (alpha=1.0)', markersize=5)
        ax2.set_ylabel('Teacher-Forced Loss', color=color_loss)
        ax2.tick_params(axis='y', labelcolor=color_loss)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    else:
        ax1.legend(loc='lower right')

    # Annotate each point with stage number
    for i, h in enumerate(stage_eval_history):
        ax1.annotate(f'S{h["stage"]}', (x_vals[i], greedy_full[i]),
                     textcoords="offset points", xytext=(0, 8), fontsize=7, ha='center')

    ax1.set_title('Held-Out Eval (alpha=1.0) After Each Stage')

    if metadata:
        subtitle = _build_plot_subtitle(metadata)
        fig.suptitle(subtitle, fontsize=9, color='gray', y=0.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, "stage_eval.png"), dpi=150)
    plt.close()

    rank_print(f"[PLOT] Saved stage_eval.png ({len(stage_eval_history)} stages)")


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
            linear_lookahead: bool = False,
            base_lookahead: int = 1,
            lookahead_step: int = 1,
            reserved_inputs: Optional[Set[str]] = None,
            num_shots: int = 0,
            seed: Optional[int] = None,
            store_examples: bool = False,
            store_cap: int = 1000,
            resume_step: int = 0,
            **task_kwargs,
    ):
        self.task = task
        self.tokenizer = tokenizer
        # self.stage = stage
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
        self.task_kwargs = task_kwargs

        self.resume_step = resume_step

        self._store = bool(store_examples)
        self._store_cap = int(store_cap)
        self._seen_inputs: deque = deque(maxlen=self._store_cap)
        self._seen_labels: deque = deque(maxlen=self._store_cap)

        self.few_shot_examples = self._build_few_shots(num_shots, seed)

        # Epoch size - number of samples per "epoch"
        self.epoch_size = 50000

        # Per-worker state (initialized lazily)
        self._worker_generator: Optional[NaturalLanguageGraphGenerator] = None
        self._worker_rng: Optional[random.Random] = None
        self._worker_id: Optional[int] = None

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
            # Original alpha-based curriculum
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

    def _get_worker_state(self, idx: int) -> Tuple[NaturalLanguageGraphGenerator, random.Random]:
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

        # Reseed RNG per batch
        batch_seed = ((self.seed or 0) +
                      idx * 104729 +
                      self._worker_id * 7919 +
                      get_rank() * 999983)
        self._worker_rng.seed(batch_seed)

        return self._worker_generator, self._worker_rng

    def __getitem__(self, idx):
        """
        Generate sample on-demand for given index.
        Index is used to vary the seed for diversity across samples.
        """

        effective_idx = idx + self.resume_step
        gen, rng = self._get_worker_state(effective_idx)

        # max_len = getattr(self.tokenizer, "model_max_length", 512)
        alpha = self._stage_alpha()

        # base_seed = (self.seed or 0) + rank * 9973 + worker_id * 997 + idx

        # Try to generate a valid sample (up to 500 attempts)
        max_attempts = 500
        attempts = 0
        ex = None
        while ex is None and attempts < max_attempts:
            attempts += 1

            batch = gen.generate_batch(
                self.task,
                batch_size=1,
                reserved_inputs=self.reserved_inputs,
                alpha=alpha,
                **self.task_kwargs
            )

            if batch and batch[0] and batch[0].output_texts:
                candidate = batch[0]
                if candidate.input_text not in self.reserved_inputs:
                    ex = candidate

        if ex is None:
            raise RuntimeError(
                f"[DATASET] Failed to generate valid sample after {max_attempts} attempts. "
                f"Stage {self.stage}, alpha={alpha:.2f}, idx={idx}."
            )

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
        # full_len = len(prompt_ids) + len(ans_ids) + len(end_ids)

        # Build final sequences
        input_ids = prompt_ids + ans_ids + end_ids
        labels = [-100] * len(prompt_ids) + ans_ids + end_ids
        attention_mask = [1] * len(input_ids)

        # Store if tracking seen samples
        if self._store and len(self._seen_inputs) < self._store_cap:
            self._seen_inputs.append(ex.input_text)
            self._seen_labels.append(list(ex.output_texts))

        # Build valid first token targets (tokenize each alternative once)
        first_union = sorted({
            tokens[0]
            for tokens in (_tokenize_leading_space(self.tokenizer, a) for a in ex.output_texts)
            if tokens
        })

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_len": len(prompt_ids),
            "valid_first_targets": first_union,
        }


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
            pack_length: int = 4096,
            target_samples_per_batch: int = 64,
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
            **task_kwargs,
    ):
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError(
                "flash-attn required for PackedSequenceDataset. "
                "Install with: pip install flash-attn --no-build-isolation"
            )

        self.task = task
        self.tokenizer = tokenizer
        self.pack_length = pack_length
        self.target_samples_per_batch = target_samples_per_batch
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
        self._worker_id = None

        self.few_shot_examples = self._build_few_shots(num_shots, seed)

        rank_print(f"[DATASET] Packed Map-style | pack_length={pack_length} | "
                   f"target_samples={target_samples_per_batch} | epoch_size={epoch_size}")

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

        # idx already includes resume_step, so batch_seed naturally differs after resume
        batch_seed = ((self.seed or 0) +
                      idx * 104729 +
                      self._worker_id * 7919 +
                      get_rank() * 999983)
        self._worker_rng.seed(batch_seed)

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

    def _pack_batch(self, all_samples: List[List[Dict]]) -> Dict[str, Any]:
        """
        Pack samples into batch format for flash_attn_varlen_func.

        Args:
            all_samples: List of rows, each row is a list of samples to pack

        Returns:
            Batch dict with concatenated tokens and cu_seqlens
        """
        batch_input_ids = []
        batch_labels = []
        batch_position_ids = []
        batch_cu_seqlens = []
        batch_max_seqlen = []
        all_sequence_info = []

        total_actual_tokens = 0
        total_tokens = 0

        for row_idx, row_samples in enumerate(all_samples):
            row_input_ids = []
            row_labels = []
            row_position_ids = []
            cu_seqlens = [0]

            current_pos = 0
            for sample in row_samples:
                seq_len = sample["seq_len"]

                row_input_ids.extend(sample["input_ids"])
                row_labels.extend(sample["labels"])
                row_position_ids.extend(range(seq_len))

                current_pos += seq_len
                cu_seqlens.append(current_pos)

                all_sequence_info.append({
                    "row_idx": row_idx,
                    "start_idx": cu_seqlens[-2],
                    "end_idx": cu_seqlens[-1],
                    "prompt_len": sample["prompt_len"],
                    "valid_first_targets": sample["valid_first_targets"],
                })

            actual_len = len(row_input_ids)
            total_actual_tokens += actual_len

            # Pad row to pack_length
            pad_len = self.pack_length - actual_len
            if pad_len > 0:
                row_input_ids.extend([self.pad_token_id] * pad_len)
                row_labels.extend([-100] * pad_len)
                row_position_ids.extend([0] * pad_len)
            elif pad_len < 0:
                # Truncate if somehow too long
                row_input_ids = row_input_ids[:self.pack_length]
                row_labels = row_labels[:self.pack_length]
                row_position_ids = row_position_ids[:self.pack_length]

            total_tokens += self.pack_length

            batch_input_ids.append(row_input_ids)
            batch_labels.append(row_labels)
            batch_position_ids.append(row_position_ids)
            batch_cu_seqlens.append(cu_seqlens)
            batch_max_seqlen.append(max(s["seq_len"] for s in row_samples) if row_samples else 0)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "position_ids": torch.tensor(batch_position_ids, dtype=torch.long),
            "cu_seqlens_list": batch_cu_seqlens,
            "max_seqlen_list": batch_max_seqlen,
            "sequence_info": all_sequence_info,
            "num_sequences": sum(len(row) for row in all_samples),
            "_efficiency": (total_actual_tokens / total_tokens * 100) if total_tokens > 0 else 100,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Generate one fully packed batch.

        Each call generates target_samples_per_batch samples and packs them
        into rows of pack_length tokens.
        """
        effective_idx = idx + self.resume_step
        gen, rng = self._get_worker_state(effective_idx)

        target_samples = self.target_samples_per_batch

        all_rows = []
        total_samples = 0
        row_samples = []
        row_len = 0

        # Generate samples until we hit target count
        max_attempts = target_samples * 10
        attempts = 0

        while total_samples < target_samples and attempts < max_attempts:
            attempts += 1
            sample = self._generate_one_sample(rng)

            if sample is None:
                continue

            # Skip sequences too long for pack_length
            if sample["seq_len"] > self.pack_length:
                continue

            # Check if sample fits in current row
            if row_len + sample["seq_len"] > self.pack_length:
                # Finalize current row
                if row_samples:
                    all_rows.append(row_samples)
                    row_samples = []
                    row_len = 0

            # Add sample to current row
            row_samples.append(sample)
            row_len += sample["seq_len"]
            total_samples += 1

        # Finalize last row
        if row_samples:
            all_rows.append(row_samples)

        # Handle edge case: no samples generated
        if not all_rows:
            raise RuntimeError(
                f"[DATASET] Failed to generate any samples for idx={idx}. "
                f"Stage={self.stage}, alpha={self._stage_alpha():.3f}"
            )

        return self._pack_batch(all_rows)


class PackedSequenceTrainer(Trainer):
    """
    Trainer for packed sequences using Flash Attention varlen.
    Properly handles Qwen3's RoPE implementation.
    """

    def __init__(self, *args, first_token_soft_weight=0.3, accuracy_window=200, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_token_soft_weight = first_token_soft_weight
        self.recent_losses = deque(maxlen=100)
        self.first_token_correct = deque(maxlen=accuracy_window)
        self.full_word_correct = deque(maxlen=accuracy_window)

        self._last_batch_samples = 0
        self._last_batch_tokens = 0
        self._last_efficiency = None

        # Training timing
        self._train_timing = {
            "data_wait": 0.0,
            "forward": 0.0,
            "total_step": 0.0,
            "steps": 0,
        }
        self._step_start_time: Optional[float] = None

        # Import Qwen3's RoPE implementation
        try:
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as qwen_apply_rope
            self._qwen_apply_rope = qwen_apply_rope
        except ImportError:
            # Fallback for older transformers versions
            self._qwen_apply_rope = None

    def get_train_dataloader(self):
        """Bypass Accelerate's dataloader wrapping."""
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=False,  # idx provides randomness via seeding
            collate_fn=lambda x: x[0],  # Unwrap single-item batch
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=True,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        t0 = time.perf_counter()
        loss = super().training_step(model, inputs, num_items_in_batch)
        dt = time.perf_counter() - t0
        self._train_timing["total_step"] += dt
        self._train_timing["steps"] += 1
        return loss

    def _report_train_timing(self):
        n = self._train_timing["steps"]
        if n == 0 or not is_main_process():
            return

        data_ms = (self._train_timing["data_wait"] / n) * 1000
        fwd_ms = (self._train_timing["forward"] / n) * 1000
        total_ms = (self._train_timing["total_step"] / n) * 1000

        data_pct = (data_ms / (data_ms + total_ms)) * 100 if (data_ms + total_ms) > 0 else 0

        print(
            f"\n[TRAIN-TIMING] Steps={n} | "
            f"DataWait={data_ms:.1f}ms ({data_pct:.1f}%) | "
            f"Forward+Loss={fwd_ms:.1f}ms | "
            f"StepTotal={total_ms:.1f}ms | "
            f"Throughput={1000 / total_ms:.1f} steps/s\n"
        )

    def reset_timing(self):
        self._train_timing = {
            "data_wait": 0.0,
            "forward": 0.0,
            "total_step": 0.0,
            "steps": 0,
        }
        self._step_start_time = None

    def _forward_layer_varlen(self, layer, hidden_states, cu_seqlens, max_seqlen,
                              num_heads, num_kv_heads, head_dim, position_ids, rotary_emb):
        """Forward one layer using flash_attn_varlen_func with Qwen3 compatibility."""

        seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        # QKV projection - hidden_states is [seq_len, H]
        q = layer.self_attn.q_proj(hidden_states)  # [seq_len, num_heads * head_dim]
        k = layer.self_attn.k_proj(hidden_states)  # [seq_len, num_kv_heads * head_dim]
        v = layer.self_attn.v_proj(hidden_states)  # [seq_len, num_kv_heads * head_dim]

        # Reshape: [seq_len, num_heads, head_dim]
        q = q.view(seq_len, num_heads, head_dim)
        k = k.view(seq_len, num_kv_heads, head_dim)
        v = v.view(seq_len, num_kv_heads, head_dim)

        # === QK Normalization (Qwen3 specific) ===
        if hasattr(layer.self_attn, 'q_norm') and layer.self_attn.q_norm is not None:
            q = layer.self_attn.q_norm(q)
            k = layer.self_attn.k_norm(k)

        # Add batch dim and transpose for RoPE: [1, num_heads, seq_len, head_dim]
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)

        # Get cos/sin from rotary_emb (uses first arg only for dtype/device inference)
        cos, sin = rotary_emb(v, position_ids)

        # Apply RoPE (only to Q and K)
        if self._qwen_apply_rope is not None:
            q, k = self._qwen_apply_rope(q, k, cos, sin)
        else:
            cos = cos.unsqueeze(1)  # [1, 1, seq_len, head_dim]
            sin = sin.unsqueeze(1)
            q = (q * cos) + (self._rotate_half(q) * sin)
            k = (k * cos) + (self._rotate_half(k) * sin)

        # Reshape for flash_attn_varlen: [seq_len, num_heads, head_dim]
        q = q.squeeze(0).transpose(0, 1).contiguous()
        k = k.squeeze(0).transpose(0, 1).contiguous()
        # V is already [seq_len, num_kv_heads, head_dim] from the view() above
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
        attn_output = layer.self_attn.o_proj(attn_output)

        hidden_states = residual + attn_output

        # MLP
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

        input_ids = inputs["input_ids"]  # [B, pack_length]
        labels = inputs["labels"]  # [B, pack_length]
        position_ids = inputs["position_ids"]  # [B, pack_length]

        B, T = input_ids.shape
        device = input_ids.device

        use_checkpoint = getattr(self.args, 'gradient_checkpointing', False)

        # Unwrap model to get components
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        if hasattr(unwrapped, "base_model"):
            unwrapped = unwrapped.base_model
        if hasattr(unwrapped, "model"):
            unwrapped = unwrapped.model

        # Get model components
        embed_tokens = unwrapped.model.embed_tokens
        layers = unwrapped.model.layers
        norm = unwrapped.model.norm
        lm_head = unwrapped.lm_head
        rotary_emb = unwrapped.model.rotary_emb  # Model-level RoPE

        # Get config
        config = unwrapped.config
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)

        # Embed all tokens
        hidden_states = embed_tokens(input_ids)  # [B, T, H]

        # Process each row with its own cu_seqlens
        all_row_losses = []
        all_row_valid = []

        # Pre-group sequence_info by row_idx (avoid repeated linear scans)
        seqs_by_row = defaultdict(list)
        for s in sequence_info:
            seqs_by_row[s["row_idx"]].append(s)

        for row_idx in range(B):
            cu_seqlens = torch.tensor(cu_seqlens_list[row_idx], dtype=torch.int32, device=device)
            max_seqlen = max_seqlen_list[row_idx]
            actual_len = cu_seqlens[-1].item()

            if actual_len == 0:
                continue

            # Trim to actual length
            row_hidden = hidden_states[row_idx, :actual_len, :]  # [actual_len, H]
            row_labels = labels[row_idx, :actual_len]  # [actual_len]
            row_pos = position_ids[row_idx:row_idx + 1, :actual_len]  # [1, actual_len]

            h = row_hidden  # [actual_len, H]

            # Forward through layers
            for layer in layers:
                if use_checkpoint and model.training:
                    h = checkpoint.checkpoint(
                        self._forward_layer_varlen,
                        layer, h, cu_seqlens, max_seqlen,
                        num_heads, num_kv_heads, head_dim,
                        row_pos, rotary_emb,
                        use_reentrant=False,
                    )
                else:
                    h = self._forward_layer_varlen(
                        layer, h, cu_seqlens, max_seqlen,
                        num_heads, num_kv_heads, head_dim,
                        row_pos, rotary_emb,
                    )

            h = norm(h)  # [actual_len, H]
            logits = lm_head(h)  # [actual_len, V]

            # Compute loss for this row
            shift_logits = logits[:-1, :]  # [actual_len-1, V]
            shift_labels = row_labels[1:]  # [actual_len-1]

            valid_mask = (shift_labels != -100)
            n_valid = valid_mask.sum().item()

            if n_valid > 0:
                ce = F.cross_entropy(
                    shift_logits[valid_mask],
                    shift_labels[valid_mask],
                    reduction='none'
                )

                # Apply soft first-token CE for sequences in this row
                row_seqs = seqs_by_row[row_idx]

                # Cache valid_positions and build position->ce_idx lookup once per row
                valid_positions = torch.nonzero(valid_mask, as_tuple=True)[0]
                pos_to_ce_idx = {}
                for ci, pos in enumerate(valid_positions.tolist()):
                    pos_to_ce_idx[pos] = ci

                for seq in row_seqs:
                    first_idx = seq["start_idx"] + seq["prompt_len"] - 1
                    if first_idx < 0 or first_idx >= shift_logits.size(0):
                        continue
                    if not valid_mask[first_idx]:
                        continue
                    valid_first = seq["valid_first_targets"]
                    if not valid_first:
                        continue

                    ce_idx = pos_to_ce_idx.get(first_idx)
                    if ce_idx is None:
                        continue

                    # Soft CE (valid_first is already sorted unique from __getitem__)
                    logp = F.log_softmax(shift_logits[first_idx], dim=-1)
                    ids = torch.tensor(valid_first, device=device, dtype=torch.long)
                    soft_ce = -logp[ids].mean()

                    # Blend
                    ce[ce_idx] = (
                            self.first_token_soft_weight * soft_ce +
                            (1.0 - self.first_token_soft_weight) * ce[ce_idx]
                    )

                all_row_losses.append(ce.sum())
                all_row_valid.append(n_valid)

                # Track accuracy (vectorized)
                with torch.no_grad():
                    preds = shift_logits.argmax(dim=-1)  # [actual_len-1]
                    for seq in row_seqs:
                        first_idx = seq["start_idx"] + seq["prompt_len"] - 1
                        if 0 <= first_idx < shift_logits.size(0) and valid_mask[first_idx]:
                            self.first_token_correct.append(preds[first_idx].item() in seq["valid_first_targets"])

                        # Full word accuracy
                        seq_start = seq["start_idx"] + seq["prompt_len"] - 1
                        seq_end = min(seq["end_idx"] - 1, shift_logits.size(0))
                        seq_valid = valid_mask[seq_start:seq_end]
                        if seq_valid.any():
                            ok = (preds[seq_start:seq_end][seq_valid] == shift_labels[seq_start:seq_end][
                                seq_valid]).all().item()
                        else:
                            ok = True
                        self.full_word_correct.append(ok)

        # Aggregate loss
        if all_row_losses:
            total_loss = sum(all_row_losses)
            total_valid = sum(all_row_valid)
            loss = total_loss / max(total_valid, 1)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        self.recent_losses.append(loss.item())

        return loss

    def get_first_token_acc(self):
        return (sum(self.first_token_correct) / len(self.first_token_correct)) if self.first_token_correct else 0.0

    def get_full_word_acc(self):
        return (sum(self.full_word_correct) / len(self.full_word_correct)) if self.full_word_correct else 0.0


# ================== Collator ==================
def make_collate(tokenizer, pad_to_multiple_of=64):
    def collate(features):
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
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
    def __init__(self, *args, first_token_soft_weight=0.3, use_chunked_ce=False, ce_chunk_size=1024, use_packing=False,
                 accuracy_window=200, **kwargs):
        self.use_packing = use_packing
        super().__init__(*args, **kwargs)
        self.first_token_soft_weight = first_token_soft_weight
        self.recent_losses = deque(maxlen=100)
        self.first_token_correct = deque(maxlen=accuracy_window)
        self.full_word_correct = deque(maxlen=accuracy_window)

        self.use_chunked_ce = use_chunked_ce
        self.ce_chunk_size = ce_chunk_size

        if self.use_chunked_ce:
            print(f"[TRAINER] Using chunked cross-entropy (chunk_size={ce_chunk_size})")

        # Training timing
        self._train_timing = {
            "data_wait": 0.0,  # Time waiting for DataLoader
            "forward": 0.0,  # Forward pass
            "total_step": 0.0,  # Total step time
            "steps": 0,
        }
        self._step_start_time: Optional[float] = None
        self._data_ready_time: Optional[float] = None

        # Track actual batch sizes for token budget mode
        self._last_batch_samples = 0
        self._last_batch_tokens = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        t0 = time.perf_counter()
        loss = super().training_step(model, inputs, num_items_in_batch)
        dt = time.perf_counter() - t0
        self._train_timing["total_step"] += dt
        self._train_timing["steps"] += 1
        return loss

    def _report_train_timing(self):
        """Print training timing breakdown"""
        n = self._train_timing["steps"]
        if n == 0 or not is_main_process():
            return

        data_ms = (self._train_timing["data_wait"] / n) * 1000
        fwd_ms = (self._train_timing["forward"] / n) * 1000
        total_ms = (self._train_timing["total_step"] / n) * 1000

        # Calculate percentages
        if total_ms > 0:
            data_pct = (data_ms / (data_ms + total_ms)) * 100
            fwd_pct = (fwd_ms / total_ms) * 100
        else:
            data_pct = fwd_pct = 0

        print(
            f"\n[TRAIN-TIMING] Steps={n} | "
            f"DataWait={data_ms:.1f}ms ({data_pct:.1f}%) | "
            f"Forward+Loss={fwd_ms:.1f}ms ({fwd_pct:.1f}%) | "
            f"StepTotal={total_ms:.1f}ms | "
            f"Throughput={1000 / total_ms:.1f} steps/s (excl. data)\n"
        )

    def reset_timing(self):
        """Reset timing stats (call on stage change)"""
        self._train_timing = {
            "data_wait": 0.0,
            "forward": 0.0,
            "total_step": 0.0,
            "steps": 0,
        }
        self._step_start_time = None
        self._data_ready_time = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Track batch size from inputs (for token budget mode)
        if "_batch_size" in inputs:
            self._last_batch_samples = inputs.pop("_batch_size")
        else:
            self._last_batch_samples = inputs["input_ids"].shape[0]

        if "_tokens_actual" in inputs:
            self._last_batch_tokens = inputs.pop("_tokens_actual")
        else:
            self._last_batch_tokens = inputs["input_ids"].numel()

        # Remove other metadata fields
        if "_tokens_padded" in inputs:
            inputs.pop("_tokens_padded")

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
                soft_ce = -logp[ids].mean()

                ce_all[i, first_idx] = (
                        self.first_token_soft_weight * soft_ce +
                        (1.0 - self.first_token_soft_weight) * ce_all[i, first_idx]
                )

        denom = valid_mask.sum().clamp_min(1)
        loss = ce_all.sum() / denom
        self.recent_losses.append(loss.item())

        # Training-time tracking (vectorized)
        with torch.no_grad():
            preds = shift_logits.argmax(dim=-1)  # [B, Tm1]
            for i in range(B):
                first_idx = prompt_len[i].item() - 1
                if 0 <= first_idx < Tm1 and valid_mask[i, first_idx]:
                    valid_ids = set(valid_first[i][valid_first_mask[i]].tolist())
                    self.first_token_correct.append(preds[i, first_idx].item() in valid_ids)

                row_valid = valid_mask[i]
                if row_valid.any():
                    ok = (preds[i][row_valid] == shift_labels[i][row_valid]).all().item()
                else:
                    ok = True
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
            soft_ce = -logp[ids].mean()

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

    def get_train_dataloader(self):
        """Override to bypass Accelerate's dataloader wrapping for token budget mode."""
        if self.use_packing:
            from torch.utils.data import DataLoader

            # Return a simple DataLoader that Accelerate won't mess with
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=1,  # Dataset yields pre-batched data
                collate_fn=self.data_collator,
                num_workers=0,
                pin_memory=True,
            )

            # Skip Accelerate's prepare() - just move batches to device manually
            return dataloader
        else:
            return super().get_train_dataloader()


# ================== Eval helpers ==================
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

    for x, ys in zip(my_inputs, my_labels):
        ys = ys if isinstance(ys, list) else [ys]
        chosen = rng.choice(ys)
        task_type = _determine_task_type(task, x)

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
        enc = tokenizer(x, return_tensors="pt").to(device)
        prompt_len = enc.input_ids.shape[1]

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
    ):
        self.dataset = dataset
        self.n_stages = n_stages
        self.acc_thr = accuracy_threshold
        self.min_steps = min_steps_per_stage
        self.check_every = check_every
        self.use_packing = use_packing
        self.trainer: Optional[SinglePathARTrainer] = None
        self.stage_start_step = 0
        self._last_log = -1
        self.finished = False

        # Speed tracking
        self.stage_start_time = None
        self.samples_this_stage = 0  # Track actual samples for token budget mode
        self.tokens_this_stage = 0  # Track actual tokens

        # Resume cooldown
        self.resume_cooldown_steps = 50
        self.last_resume_step = -1

        # Capture grad_norm and lr from Trainer logs
        self.last_grad_norm = 0.0
        self.last_lr = 0.0

        # Stage eval config
        self.do_stage_eval = do_stage_eval
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
        self.stage_eval_history = []

        # Loss tracking
        self.loss_history = []
        # Flops tracking
        self.flops_per_token = None
        self.cumulative_wall_time = 0.0  # Total wall time across resumes (seconds)
        self.wall_time_offset = 0.0  # Offset from previous runs
        self.training_start_time = None  # Set when training starts
        self.plot_metadata = None  # Set after trainer creation

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture grad_norm and lr from Transformers' logs"""
        if logs:
            self.last_grad_norm = logs.get('grad_norm', self.last_grad_norm)
            self.last_lr = logs.get('learning_rate', self.last_lr)
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize stage timing at training start"""
        if self.stage_start_time is None:
            self.stage_start_time = datetime.datetime.now()
            self.samples_this_stage = 0
            self.tokens_this_stage = 0

        # Track wall time
        if self.training_start_time is None:
            self.training_start_time = time.time()

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
            _save_curriculum_state(checkpoint_dir, self.dataset.stage, self.stage_start_step, current_wall_time)

        if is_main_process():
            try:
                loss_path = os.path.join(args.output_dir, "loss_history.json")
                with open(loss_path, "w") as f:
                    json.dump(self.loss_history, f)
            except Exception as e:
                rank_print(f"[LOSS] Warning: Could not save loss history: {e}")
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

        with torch.no_grad():
            # Teacher-forced loss
            tf_loss = run_eval_tf_loss(
                model, self.tokenizer, self.task,
                self.eval_inputs_hard, self.eval_labels_hard,
                num_shots=self.num_shots, max_input_size=self.max_input_size,
                seed=self.seed, **self.task_kwargs
            )

            # Greedy accuracy
            greedy_result = run_eval_greedy_readable(
                model, self.tokenizer, self.task,
                self.eval_inputs_hard, self.eval_labels_hard,
                num_shots=self.num_shots, max_input_size=self.max_input_size,
                seed=self.seed, print_examples=0,
                **self.task_kwargs
            )

            # Stage-alpha eval: generate and evaluate on stage-difficulty data
            # Skip for periodic evals (no stage transition) and single-stage runs
            stage_greedy_result = None
            stage_alpha = self.dataset._stage_alpha()
            max_alpha_L = self.dataset.task_kwargs.get("max_lookahead")
            is_full_difficulty = (target_L is not None and target_L >= (max_alpha_L or 256))
            if not is_periodic and not is_full_difficulty and stage_alpha < 1.0:
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
                        **self.task_kwargs
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

            # Calculate achieved TFLOPs/s
            tokens_this_step = self.trainer._last_batch_tokens * get_world_size() if hasattr(self.trainer,
                                                                                             '_last_batch_tokens') else 0
            achieved_tflops = 0.0
            if hasattr(self.trainer, '_train_timing') and self.trainer._train_timing["steps"] > 0:
                recent_step_time = self.trainer._train_timing["total_step"] / self.trainer._train_timing["steps"]
                if recent_step_time > 0 and self.flops_per_token and tokens_this_step > 0:
                    flops_this_step = tokens_this_step * self.flops_per_token
                    achieved_tflops = flops_this_step / recent_step_time / 1e12

            self.loss_history.append({
                "step": state.global_step,
                "loss": current_loss,
                "stage": self.dataset.stage,
                "alpha": self.dataset._stage_alpha(),
                "effective_L": effective_L,
                "tokens": tokens_this_step,
                "wall_time": current_wall_time,
                "achieved_tflops": achieved_tflops,
            })

        current_time = datetime.datetime.now()

        # ==================== Track Samples for Packing mode
        if self.use_packing:
            if hasattr(self.trainer, '_last_batch_samples'):
                self.samples_this_stage += self.trainer._last_batch_samples * get_world_size()
            if hasattr(self.trainer, '_last_batch_tokens'):
                self.tokens_this_stage += self.trainer._last_batch_tokens * get_world_size()

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

            # Warmup indicator
            warmup_msg = ""
            if self.last_resume_step >= 0 and (state.global_step - self.last_resume_step) < self.resume_cooldown_steps:
                remaining = self.resume_cooldown_steps - (state.global_step - self.last_resume_step)
                warmup_msg = f" | Warmup={remaining}"

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
                expected_stages = math.ceil((max_L - self.dataset.base_lookahead) / self.dataset.lookahead_step) + 1
                current_L = self.dataset._stage_target_lookahead()
                stage_str = f"[Stage {self.dataset.stage}/{expected_stages} L={current_L}]"
            else:
                stage_str = f"[Stage {self.dataset.stage}/{self.n_stages}]"

            print(f"{stage_str} step {state.global_step} | "
                  f"loss_avg={loss:.4f}({len(self.trainer.recent_losses)}) | First={f1:.2%} | Full={fw:.2%} | "
                  f"lr={self.last_lr:.2e} | grad_norm={self.last_grad_norm:.2f} | "
                  f"Speed={samples_per_sec:.1f} samples/s ({tokens_per_sec_str} toks/s){achieved_tflops_str} | "
                  f"Stage time={stage_time_str}{extra_info}{warmup_msg}")
            self._last_log = state.global_step

        # ==================== Resume Warmup ====================
        if self.last_resume_step >= 0 and (state.global_step - self.last_resume_step) < self.resume_cooldown_steps:
            if is_main_process() and state.global_step % 10 == 0:
                remaining = self.resume_cooldown_steps - (state.global_step - self.last_resume_step)
                print(f"[CURRICULUM] Skipping checks during warmup ({remaining} steps remaining)")
            return control

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

                # Run stage eval BEFORE advancing
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
                                                           time.time() - self.training_start_time) if self.training_start_time else self.wall_time_offset)
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

                    # Reset timing for new stage
                    if hasattr(self.trainer, 'reset_timing'):
                        self.trainer.reset_timing()

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
                                               current_wall_time)

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

    # LoRA
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)

    # Seed
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--deterministic", action="store_true",
                   help="Enable CUDA deterministic algorithms (may be slower)")

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
    p.add_argument("--do_stage_eval", action="store_true",
                   help="Run TF+greedy eval at alpha=1.0 after each stage advancement")
    p.add_argument("--eval_every_steps", type=int, default=0,
                   help="Run greedy eval every N steps (0=disabled, useful for no-curriculum runs)")

    # Redacted eval config
    p.add_argument("--eval_redacted_samples", type=int, default=None)
    p.add_argument("--redaction_token", type=str, default="_____")

    # Memory control
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--oom_autoscale", action="store_true")
    p.add_argument("--torch_compile", action="store_true", help="Enable torch.compile for the model")

    # Scratch / resume
    p.add_argument("--scratch_dir", type=str,
                   default=os.environ.get("SCRATCH") or os.path.join("/scratch", os.environ.get("USER", "user")))
    p.add_argument("--job_id", type=str, default=os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID"))
    p.add_argument("--resume_from_job", type=str, default=None)

    # Liger kernels
    p.add_argument("--use_liger", default=True, action="store_true",
                   help="Use Liger kernel for memory-efficient training")

    # Chunked cross-entropy
    p.add_argument("--use_chunked_ce", action="store_true", help="Use chunked cross-entropy for memory efficiency")
    p.add_argument("--ce_chunk_size", type=int, default=1024, help="Chunk size for chunked cross-entropy")

    # Packing
    p.add_argument("--use_packing", action="store_true",
                   help="Use sequence packing for efficiency")
    p.add_argument("--pack_length", type=int, default=4096,
                   help="Fixed length for packed rows")
    p.add_argument("--target_samples_per_batch", type=int, default=64,
                   help="Fixed number of samples per training step")

    global args
    args = p.parse_args()

    # Validate linear_lookahead
    if args.linear_lookahead:
        if args.task != "search":
            rank_print("[WARN] --linear_lookahead only applies to search task, ignoring")
            args.linear_lookahead = False
        else:
            # Calculate expected number of stages
            expected_stages = math.ceil((args.max_lookahead - args.base_lookahead) / args.lookahead_step) + 1
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
    if not args.use_packing:
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

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
                rank_print("[MEM] Enabled input grads for Gradient Checkpointing + LoRA")

    if is_main_process():
        model.print_trainable_parameters()

    # Find hard limit to avoid OOM (no longer needed but kept for debug)
    # _ = estimate_worst_case_length(args, tokenizer)

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

    need_eval_data = args.do_baseline or args.do_final_eval or args.do_redacted_eval or args.do_stage_eval

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
    if args.use_packing:
        rank_print(
            f"[DATASET] Using PackedSequenceDataset (pack_length={args.pack_length}, target_samples={args.target_samples_per_batch})")
        if args.do_seen_eval:
            rank_print("[WARN] --do_seen_eval disabled for packed dataset (incompatible with multi-worker)")
            args.do_seen_eval = False

        dataset = PackedSequenceDataset(
            task=args.task,
            tokenizer=tokenizer,
            pack_length=args.pack_length,
            target_samples_per_batch=args.target_samples_per_batch,
            stage=1,
            n_stages=args.n_stages,
            base_alpha=args.base_alpha,
            max_alpha=args.max_alpha,
            max_input_size=args.max_input_size,
            reserved_inputs=reserved_inputs,
            num_shots=args.num_shots,
            seed=args.seed,
            resume_step=resume_step,
            store_examples=args.do_seen_eval,
            store_cap=1000,
            linear_lookahead=args.linear_lookahead,
            base_lookahead=args.base_lookahead,
            lookahead_step=args.lookahead_step,
            epoch_size=100_000_000,
            **task_kwargs,
        )
        data_collator = lambda x: x[0]  # Identity
        effective_batch_size = 1
        num_workers = 4
    else:
        rank_print(f"[DATASET] Using SinglePathARDataset (batch_size={args.batch_size})")
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
            resume_step=resume_step,
            linear_lookahead=args.linear_lookahead,
            base_lookahead=args.base_lookahead,
            lookahead_step=args.lookahead_step,
            **task_kwargs,
        )
        data_collator = make_collate(tokenizer, pad_to_multiple_of=None)
        effective_batch_size = args.batch_size
        num_workers = 8

    curriculum = FirstTokenCurriculum(
        dataset=dataset,
        n_stages=args.n_stages,
        accuracy_threshold=args.accuracy_threshold,
        min_steps_per_stage=args.min_steps_per_stage,
        check_every=args.check_every,
        use_packing=args.use_packing,
        # Stage eval config
        do_stage_eval=args.do_stage_eval,
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
    )

    if resume_ckpt:
        if _try_restore_curriculum_state(resume_ckpt, dataset, curriculum):
            rank_print(f"[CURRICULUM] Synced state with checkpoint: {resume_ckpt}")
            match = re.search(r'checkpoint-(\d+)', resume_ckpt)
            if match:
                curriculum.last_resume_step = int(match.group(1))

    # ==================== RETROACTIVE PLOT GENERATION / EXTENSION CHECK ====================
    # If resuming a completed job, either generate plots and exit, or continue with extended params
    if resume_ckpt and curriculum.loss_history:
        # Check if curriculum would be finished under CURRENT parameters
        is_finished = dataset._is_final_stage() and curriculum.stage_start_step > 0

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
                    est_tokens_per_step = args.target_samples_per_batch * avg_seq_len * get_world_size()
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
                "target_samples": args.target_samples_per_batch,
                "accuracy_threshold": args.accuracy_threshold,
            }

            if is_main_process():
                rank_print("[RETROACTIVE] Generating plots for completed job...")

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

                rank_print(f"[RETROACTIVE] Generated plots for {len(stages_seen)} stages")
                rank_print("[RETROACTIVE] Done. Exiting without training.")

            barrier()
            return 0

    # Disable Accelerate batch dispatching for token-budget training
    if args.use_packing:
        os.environ["ACCELERATE_DISPATCH_BATCHES"] = "false"
        os.environ["ACCELERATE_SPLIT_BATCHES"] = "false"
        os.environ["ACCELERATE_EVEN_BATCHES"] = "false"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=effective_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=100000000 if args.use_packing else -1,
        num_train_epochs=1000,
        logging_steps=10,
        logging_first_step=False,
        report_to="none",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
        save_safetensors=True,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        optim="adamw_torch_fused",
        # dataloader_num_workers=0,

        dataloader_num_workers=num_workers,
        dataloader_prefetch_factor=4 if num_workers > 0 else None,
        dataloader_persistent_workers=num_workers > 0,
        dataloader_pin_memory=True,

        seed=args.seed if args.seed is not None else 42,

        torch_compile=args.torch_compile,
        torch_compile_backend="inductor",
        torch_compile_mode="reduce-overhead",

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

    if args.use_packing:
        trainer = PackedSequenceTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[curriculum, baseline_cb],
            first_token_soft_weight=args.first_token_soft_weight,
            accuracy_window=args.accuracy_window,
        )
    else:
        trainer = SinglePathARTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[curriculum, baseline_cb],
            first_token_soft_weight=args.first_token_soft_weight,
            use_chunked_ce=args.use_chunked_ce,
            ce_chunk_size=args.ce_chunk_size,
            use_packing=False,
            accuracy_window=args.accuracy_window,
        )
    curriculum.trainer = trainer
    baseline_cb.trainer = trainer

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
        "target_samples": args.target_samples_per_batch,
        "accuracy_threshold": args.accuracy_threshold,
    }

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
    if is_main_process():
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
        rank_print(f"[FINAL-EVAL] Fingerprints verified ✓")

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

        # Redact the hard eval set (alpha=1.0)
        red_inputs, red_labels = build_redacted_eval_set(
            eval_inputs_hard, eval_labels_hard,
            token=args.redaction_token,
            max_n=red_n
        )

        if red_inputs:
            red_result = run_eval_greedy_readable(
                trainer.model, tokenizer, args.task,
                red_inputs, red_labels,
                num_shots=args.num_shots, max_input_size=args.max_input_size, seed=args.seed,
                print_examples=min(3, args.print_eval_examples), **task_kwargs
            )
            rank_print(
                f"[REDACTED] First={red_result['first_token_acc']:.2%} | Full={red_result['full_word_acc']:.2%} | N={red_result['total']}")
        else:
            rank_print("[REDACTED-EVAL] Skipped (could not redact any eval items cleanly)")

    if is_main_process():
        # Save final loss history
        try:
            with open(os.path.join(args.output_dir, "loss_history.json"), "w") as f:
                json.dump(curriculum.loss_history, f)
            rank_print(f"[LOSS] Saved {len(curriculum.loss_history)} records")
        except Exception as e:
            rank_print(f"[LOSS] Warning: {e}")

        # Final overall plot
        plot_overall_loss(curriculum.loss_history, args.output_dir, args.n_stages, curriculum.plot_metadata)
        plot_loss_vs_flops(curriculum.loss_history, args.output_dir, curriculum.flops_per_token, args.n_stages,
                           curriculum.plot_metadata)
        plot_loss_vs_walltime(curriculum.loss_history, args.output_dir, args.n_stages, curriculum.plot_metadata)
        plot_achieved_tflops(curriculum.loss_history, args.output_dir, args.n_stages, curriculum.plot_metadata)
        if curriculum.stage_eval_history:
            plot_stage_eval(curriculum.stage_eval_history, args.output_dir, curriculum.plot_metadata)

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