#!/usr/bin/env python3
"""
Standalone plotting script for NL fine-tuning training runs.

Can be used in two ways:
  1. Imported by qwen_tuning_nl_multi.py during training
  2. Run standalone to regenerate plots for past runs:
       python plot_training.py /path/to/run/output_dir

Data files expected in output_dir:
  - loss_history.json        (required)
  - stage_eval_history.json  (optional)
  - plot_data.json           (required for standalone — contains metadata, flops_per_token, n_stages)
"""

import os
import json
import sys
import argparse
from typing import List, Dict, Optional

import numpy as np


# --------------- Helpers ---------------

def _print(msg):
    """Print helper — uses rank_print if available, else plain print."""
    print(msg)


def _smooth_step_based(steps, values, window=500):
    """Step-based smoothing: average values within a step window."""
    smoothed = np.empty(len(values))
    smoothed[:] = np.nan
    left = 0
    for i in range(len(values)):
        while steps[left] < steps[i] - window:
            left += 1
        smoothed[i] = np.mean(values[left:i + 1])
    return smoothed


def _overlay_split_losses(ax, valid_history, x_values, steps, step_window=500):
    """Overlay search_loss and pretrain_loss lines if available in loss history."""
    has_search = any("search_loss" in h for h in valid_history)
    has_pretrain = any("pretrain_loss" in h for h in valid_history)
    if not has_search and not has_pretrain:
        return

    x = np.array(x_values)

    if has_search:
        search_losses = np.array([h.get("search_loss", np.nan) for h in valid_history])
        mask = ~np.isnan(search_losses)
        if mask.any():
            sm = _smooth_step_based(steps[mask], search_losses[mask], step_window)
            sm_valid = ~np.isnan(sm)
            ax.plot(x[mask][sm_valid], sm[sm_valid], linewidth=2, color='green',
                    label=f'Search Loss (w={step_window})', alpha=0.9)

    if has_pretrain:
        pt_losses = np.array([h.get("pretrain_loss", np.nan) for h in valid_history])
        mask = ~np.isnan(pt_losses)
        if mask.any():
            sm = _smooth_step_based(steps[mask], pt_losses[mask], step_window)
            sm_valid = ~np.isnan(sm)
            ax.plot(x[mask][sm_valid], sm[sm_valid], linewidth=2, color='orange',
                    label=f'Instruct Loss (w={step_window})', alpha=0.9)


def fit_exponential_decay(steps: List[int], losses: List[float], window: int = 50,
                          skip_initial_spike: bool = True) -> Optional[Dict[str, float]]:
    """
    Fit exponential decay: L(t) = a * exp(-b * t) + c

    Returns dict with a, b, c, r_squared, half_life, fit_start_step, fit_steps, fit_values.
    Returns None if fit fails.
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        _print("[FIT] scipy not installed, skipping exponential fit")
        return None

    if len(losses) < 20:
        return None

    fit_steps = list(steps)
    fit_losses = list(losses)

    if skip_initial_spike and len(fit_losses) > 30:
        search_range = max(10, len(fit_losses) // 3)
        peak_idx = np.argmax(fit_losses[:search_range])
        if peak_idx > 2:
            fit_steps = fit_steps[peak_idx:]
            fit_losses = fit_losses[peak_idx:]

    if len(fit_losses) < 15:
        return None

    smooth_window = min(window, len(fit_losses) // 3) or 1
    if len(fit_losses) >= smooth_window:
        smoothed = np.convolve(fit_losses, np.ones(smooth_window) / smooth_window, mode='valid')
        smooth_steps = fit_steps[smooth_window - 1:]
    else:
        smoothed = np.array(fit_losses)
        smooth_steps = fit_steps

    if len(smoothed) < 10:
        return None

    t = np.array(smooth_steps, dtype=float) - smooth_steps[0]
    y = np.array(smoothed, dtype=float)

    def exp_decay(t, a, b, c):
        return a * np.exp(-b * t) + c

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

        y_pred = exp_decay(t, a, b, c)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        half_life = np.log(2) / b if b > 0 else float('inf')

        return {
            "a": a,
            "b": b,
            "c": c,
            "r_squared": r_squared,
            "half_life": half_life,
            "fit_start_step": smooth_steps[0],
            "fit_steps": smooth_steps,
            "fit_values": exp_decay(t, a, b, c).tolist(),
        }
    except Exception as e:
        _print(f"[FIT] Exponential fit failed: {e}")
        return None


def _draw_resume_markers(ax, loss_history: List[Dict], x_key: str = "step", x_values=None, x_scale: float = 1.0):
    """Draw vertical dashed red lines at resume points in loss_history."""
    for i, h in enumerate(loss_history):
        if h.get("resume"):
            if x_values is not None:
                x = x_values[i]
            else:
                x = h.get(x_key, 0) * x_scale
            ax.axvline(x=x, color='red', linestyle=':', alpha=0.5, linewidth=1.5, zorder=5)


def _stage_alpha_series(xs, stage_eval_history: List[Dict], full_alpha_tol: float = 1e-3):
    """Return (xs, ys) for the stage-α greedy line.

    The trainer skips the redundant stage-α evaluation when training α already
    equals 1.0 (e.g. final stage). For those entries we fall back to ``greedy_full``
    (identical by definition); entries that legitimately lack the data are dropped.
    """
    out_xs, out_ys = [], []
    for x, h in zip(xs, stage_eval_history):
        v = h.get("stage_greedy_full")
        if v is None and abs(h.get("alpha_training", 0.0) - 1.0) <= full_alpha_tol:
            v = h.get("greedy_full")
        if v is None:
            continue
        out_xs.append(x)
        out_ys.append(v * 100)
    return out_xs, out_ys


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
        parts.append(f"Acc\u2265{metadata['accuracy_threshold']:.0%}")

    return " | ".join(parts)


# --------------- Plot functions ---------------

def plot_stage_loss(loss_history: List[Dict], stage: int, output_dir: str, metadata: Dict = None):
    """Plot loss curve for a single completed stage with exponential fit."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        _print("[PLOT] matplotlib not installed, skipping plot")
        return

    stage_data = [h for h in loss_history if h["stage"] == stage]
    if not stage_data:
        return

    steps = np.array([h["step"] for h in stage_data])
    losses = np.array([h["loss"] for h in stage_data])

    alpha = stage_data[0].get("alpha", None)
    effective_L = stage_data[0].get("effective_L", None)

    start_step = steps[0]
    rel_steps = steps - start_step

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rel_steps, losses, alpha=0.3, linewidth=0.5, color='blue', label='Raw')

    # Step-based smoothing (200 steps) to handle variable entry density
    step_window = 200
    smoothed = np.empty(len(losses))
    smoothed[:] = np.nan
    left = 0
    for i in range(len(losses)):
        while steps[left] < steps[i] - step_window:
            left += 1
        smoothed[i] = np.mean(losses[left:i + 1])
    valid = ~np.isnan(smoothed)
    ax.plot(rel_steps[valid], smoothed[valid], linewidth=2, color='blue', label=f'Smoothed (w={step_window} steps)')
    window = step_window  # for fit_exponential_decay below

    fit_result = fit_exponential_decay(rel_steps, losses, window=window)
    fit_text = ""
    if fit_result and fit_result["r_squared"] > 0.8:
        fit_start = fit_result["fit_start_step"]
        ax.axvline(x=fit_start, color='orange', linestyle=':', alpha=0.5, label='Fit start')

        ax.plot(fit_result["fit_steps"], fit_result["fit_values"], '--', linewidth=2, color='red',
                label=f'Exp fit (R\u00b2={fit_result["r_squared"]:.3f})')

        ax.axhline(y=fit_result["c"], color='green', linestyle=':', alpha=0.7,
                   label=f'Asymptote={fit_result["c"]:.4f}')

        fit_text = (f'Fit: L(t) = {fit_result["a"]:.3f}\u00b7e^(-{fit_result["b"]:.2e}\u00b7t) + {fit_result["c"]:.4f}\n'
                    f'Est. minimum: {fit_result["c"]:.4f} | Half-life: {fit_result["half_life"]:.0f} steps | R\u00b2={fit_result["r_squared"]:.3f}')
    elif fit_result:
        fit_text = f'Exp fit poor (R\u00b2={fit_result["r_squared"]:.3f})'

    ax.set_xlabel('Steps in Stage')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')

    step_range = int(steps[-1] - steps[0])
    title = f'Stage {stage} Loss ({step_range} steps, final={losses[-1]:.4f})'
    if effective_L is not None:
        title += f' | L={effective_L}'
    if alpha is not None:
        title += f' | \u03b1={alpha:.3f}'
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
        _print(
            f"[PLOT] Saved loss_stage_{stage}.png | Est. min={fit_result['c']:.4f}, half-life={fit_result['half_life']:.0f} steps")
    else:
        _print(f"[PLOT] Saved loss_stage_{stage}.png")


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

    steps = np.array([h["step"] for h in loss_history])
    losses = np.array([h["loss"] for h in loss_history])
    stages = [h["stage"] for h in loss_history]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, losses, alpha=0.3, linewidth=0.5, color='blue')

    # Use step-based smoothing window (500 steps) to handle variable entry density
    step_window = 500
    smoothed = np.empty(len(losses))
    smoothed[:] = np.nan
    left = 0
    for i in range(len(losses)):
        while steps[left] < steps[i] - step_window:
            left += 1
        smoothed[i] = np.mean(losses[left:i + 1])
    valid = ~np.isnan(smoothed)
    ax.plot(steps[valid], smoothed[valid], linewidth=2, color='blue', label=f'Smoothed (w={step_window} steps)')

    stage_info = {}
    for h in loss_history:
        s = h["stage"]
        if s not in stage_info:
            stage_info[s] = (h["step"], h.get("effective_L"), h.get("alpha"))

    max_stage = max(stages)
    colors = plt.cm.tab10(np.linspace(0, 1, max(max_stage, n_stages)))

    prev_stage = stages[0]
    for i, (step, stage) in enumerate(zip(steps, stages)):
        if stage != prev_stage:
            ax.axvline(x=step, color=colors[(stage - 1) % 10], linestyle='--', alpha=0.7)

            effective_L = stage_info.get(stage, (None, None, None))[1]
            if effective_L is not None:
                label = f'S{stage}\nL={effective_L}'
            else:
                label = f'S{stage}'

            ax.text(step, ax.get_ylim()[1] * 0.95, label, fontsize=8, ha='left', va='top')
            prev_stage = stage

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
            start_step = stg_steps[0]
            end_step = stg_steps[-1]
            ax.hlines(y=fit["c"], xmin=start_step, xmax=end_step,
                      colors='green', linestyles=':', alpha=0.6, linewidth=1.5)

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_title('Training Loss (Overall)')

    if stage_fits:
        fit_lines = ["Stage asymptotes:"]
        for stg in sorted(stage_fits.keys()):
            f = stage_fits[stg]
            L = stage_info.get(stg, (None, None))[1]
            L_str = f"L={L}" if L else ""
            fit_lines.append(f"  S{stg} {L_str}: min\u2248{f['c']:.4f}")
        fit_text = "\n".join(fit_lines)
        ax.text(0.98, 0.98, fit_text, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', horizontalalignment='right', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    if metadata:
        subtitle = _build_plot_subtitle(metadata)
        fig.suptitle(subtitle, fontsize=9, color='gray', y=0.02)

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    _draw_resume_markers(ax, loss_history, x_key="step")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, "loss_overall.png"), dpi=150)
    plt.close()

    _print(f"[PLOT] Saved loss_overall.png ({len(stage_fits)} stages with exp fits)")


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

    # Filter out entries with no token data (e.g. from trainer_state gap recovery)
    valid_history = [h for h in loss_history if h.get("tokens", 0) > 0]
    if not valid_history:
        _print("[PLOT] No token data available, skipping FLOPs plot")
        return

    cumulative_flops = []
    total_flops = 0
    for h in valid_history:
        total_flops += h["tokens"] * flops_per_token
        cumulative_flops.append(total_flops)

    losses = np.array([h["loss"] for h in valid_history])
    stages = [h["stage"] for h in valid_history]
    steps = np.array([h["step"] for h in valid_history])

    cumulative_pflops = np.array([f / 1e15 for f in cumulative_flops])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cumulative_pflops, losses, alpha=0.3, linewidth=0.5, color='blue')

    # Step-based smoothing (500 steps) to handle variable entry density
    step_window = 500
    smoothed = np.empty(len(losses))
    smoothed[:] = np.nan
    left = 0
    for i in range(len(losses)):
        while steps[left] < steps[i] - step_window:
            left += 1
        smoothed[i] = np.mean(losses[left:i + 1])
    valid_mask = ~np.isnan(smoothed)
    ax.plot(cumulative_pflops[valid_mask], smoothed[valid_mask], linewidth=2, color='blue', label=f'Total Loss (w={step_window} steps)')

    # Overlay search/instruct loss lines if available
    _overlay_split_losses(ax, valid_history, cumulative_pflops.tolist(), steps, step_window)

    stage_info = {}
    for i, h in enumerate(valid_history):
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

    if metadata:
        subtitle = _build_plot_subtitle(metadata)
        fig.suptitle(subtitle, fontsize=9, color='gray', y=0.02)

    ax.legend()
    ax.grid(True, alpha=0.3)
    _draw_resume_markers(ax, valid_history, x_values=cumulative_pflops.tolist())

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, "loss_vs_flops.png"), dpi=150)
    plt.close()

    _print(f"[PLOT] Saved loss_vs_flops.png")


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

    if not any(h.get("wall_time", 0) > 0 for h in loss_history):
        _print("[PLOT] No wall_time data available, skipping wall clock plot")
        return

    # Filter out entries with no wall_time data (e.g. from trainer_state gap recovery)
    valid_history = [h for h in loss_history if h.get("wall_time", 0) > 0]
    if not valid_history:
        _print("[PLOT] No wall_time data available, skipping wall clock plot")
        return

    wall_times = np.array([h["wall_time"] / 3600 for h in valid_history])
    losses = np.array([h["loss"] for h in valid_history])
    stages = [h["stage"] for h in valid_history]
    steps = np.array([h["step"] for h in valid_history])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wall_times, losses, alpha=0.3, linewidth=0.5, color='blue')

    # Step-based smoothing (500 steps) to handle variable entry density
    step_window = 500
    smoothed = np.empty(len(losses))
    smoothed[:] = np.nan
    left = 0
    for i in range(len(losses)):
        while steps[left] < steps[i] - step_window:
            left += 1
        smoothed[i] = np.mean(losses[left:i + 1])
    valid_mask = ~np.isnan(smoothed)
    ax.plot(wall_times[valid_mask], smoothed[valid_mask], linewidth=2, color='blue', label=f'Total Loss (w={step_window} steps)')

    # Overlay search/instruct loss lines if available
    _overlay_split_losses(ax, valid_history, wall_times.tolist(), steps, step_window)

    stage_info = {}
    for i, h in enumerate(valid_history):
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

    if metadata:
        subtitle = _build_plot_subtitle(metadata)
        fig.suptitle(subtitle, fontsize=9, color='gray', y=0.02)

    ax.legend()
    ax.grid(True, alpha=0.3)
    _draw_resume_markers(ax, valid_history, x_key="wall_time", x_scale=1/3600)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, "loss_vs_walltime.png"), dpi=150)
    plt.close()

    _print(f"[PLOT] Saved loss_vs_walltime.png")


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

    if not any(h.get("achieved_tflops", 0) > 0 for h in loss_history):
        _print("[PLOT] No achieved_tflops data available, skipping TFLOPs plot")
        return

    steps = [h["step"] for h in loss_history]
    tflops = [h.get("achieved_tflops", 0) for h in loss_history]
    stages = [h["stage"] for h in loss_history]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, tflops, alpha=0.3, linewidth=0.5, color='green')

    window = min(500, len(tflops) // 5) or 1
    if len(tflops) >= window:
        smoothed = np.convolve(tflops, np.ones(window) / window, mode='valid')
        ax.plot(steps[window - 1:], smoothed, linewidth=2, color='green', label=f'Smoothed (w={window})')

    prev_stage = stages[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(max(stages), n_stages)))

    for i, (step, stage) in enumerate(zip(steps, stages)):
        if stage != prev_stage:
            ax.axvline(x=step, color=colors[(stage - 1) % 10], linestyle='--', alpha=0.7)
            prev_stage = stage

    ax.set_xlabel('Step')
    ax.set_ylabel('Achieved TFLOPs/s')
    ax.set_title('GPU Compute Throughput Over Training')

    avg_tflops = np.mean([t for t in tflops if t > 0])
    ax.axhline(y=avg_tflops, color='red', linestyle=':', alpha=0.7, label=f'Avg: {avg_tflops:.1f}')

    if metadata:
        subtitle = _build_plot_subtitle(metadata)
        fig.suptitle(subtitle, fontsize=9, color='gray', y=0.02)

    ax.legend()
    ax.grid(True, alpha=0.3)
    _draw_resume_markers(ax, loss_history, x_key="step")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, "achieved_tflops.png"), dpi=150)
    plt.close()

    _print(f"[PLOT] Saved achieved_tflops.png (avg: {avg_tflops:.1f} TFLOPs/s)")


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

    if has_L:
        x_vals = [h["effective_L"] for h in stage_eval_history]
        x_label = "Effective Lookahead (L)"
    else:
        x_vals = stages
        x_label = "Stage"

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_first = '#2196F3'
    color_full = '#4CAF50'
    ax1.plot(x_vals, greedy_first, 'o-', color=color_first, label='Greedy First Token', markersize=5)
    ax1.plot(x_vals, greedy_full, 's-', color=color_full, label='Greedy Full Word', markersize=5)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    if has_loss:
        ax2 = ax1.twinx()
        tf_losses = [h["tf_loss"] for h in stage_eval_history]
        color_loss = '#F44336'
        ax2.plot(x_vals, tf_losses, '^--', color=color_loss, label='TF Loss (alpha=1.0)', markersize=5)
        ax2.set_ylabel('Teacher-Forced Loss', color=color_loss)
        ax2.tick_params(axis='y', labelcolor=color_loss)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    else:
        ax1.legend(loc='lower right')

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

    _print(f"[PLOT] Saved stage_eval.png ({len(stage_eval_history)} stages)")


def plot_eval_acc_vs_step(stage_eval_history: List[Dict], output_dir: str, metadata: Dict = None,
                          loss_history: List[Dict] = None):
    """Plot full-alpha and stage-alpha greedy accuracy vs training step."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not stage_eval_history or len(stage_eval_history) < 2:
        return

    steps = [h["step"] for h in stage_eval_history]
    greedy_full = [h["greedy_full"] * 100 for h in stage_eval_history]
    greedy_first = [h["greedy_first"] * 100 for h in stage_eval_history]

    has_stage_alpha = any("stage_greedy_full" in h for h in stage_eval_history)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, greedy_full, '-', color='#4CAF50', label='Eval (\u03b1=1.0)', linewidth=1.5, markersize=2, marker='.')

    if has_stage_alpha:
        # When stage-α eval is skipped (e.g. final stage where training α already == 1.0),
        # fall back to greedy_full (identical by definition); otherwise drop the point.
        stage_xs, stage_ys = _stage_alpha_series(steps, stage_eval_history)
        ax.plot(stage_xs, stage_ys, '-', color='#FF9800', label='Train (stage \u03b1)', linewidth=1.5, markersize=2, marker='.')

    # Stage divider lines
    prev_stage = stage_eval_history[0]["stage"]
    n_stages_seen = max(h["stage"] for h in stage_eval_history)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_stages_seen, 10)))
    for i, h in enumerate(stage_eval_history):
        if h["stage"] != prev_stage:
            ax.axvline(x=steps[i], color=colors[(h["stage"] - 1) % 10], linestyle='--', alpha=0.5, linewidth=1)
            effective_L = h.get("effective_L", None)
            label = f'S{h["stage"]} L={effective_L}' if effective_L else f'S{h["stage"]}'
            ax.text(steps[i], ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] > 0 else 100,
                    label, fontsize=7, ha='left', va='top', rotation=90, alpha=0.7)
            prev_stage = h["stage"]

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Greedy Full Word Accuracy (%)', fontsize=12)
    ax.set_title('Eval Accuracy vs Step')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if loss_history:
        _draw_resume_markers(ax, loss_history, x_key="step")

    if metadata:
        subtitle = _build_plot_subtitle(metadata)
        fig.suptitle(subtitle, fontsize=9, color='gray', y=0.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, "eval_acc_vs_step.png"), dpi=150)
    plt.close()

    _print(f"[PLOT] Saved eval_acc_vs_step.png ({len(stage_eval_history)} evals)")


def plot_eval_acc_vs_flops(stage_eval_history: List[Dict], loss_history: List[Dict],
                           output_dir: str, flops_per_token: int, metadata: Dict = None):
    """Plot full-alpha and stage-alpha greedy accuracy vs cumulative FLOPs."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not stage_eval_history or len(stage_eval_history) < 2:
        return
    if not loss_history or not flops_per_token:
        return

    step_to_flops = {}
    total_flops = 0
    for h in loss_history:
        total_flops += h.get("tokens", 0) * flops_per_token
        step_to_flops[h["step"]] = total_flops

    eval_flops = []
    for h in stage_eval_history:
        s = h["step"]
        if s in step_to_flops:
            eval_flops.append(step_to_flops[s])
        else:
            closest = max((k for k in step_to_flops if k <= s), default=None)
            if closest is not None:
                eval_flops.append(step_to_flops[closest])
            else:
                eval_flops.append(0)

    eval_exaflops = [f / 1e18 for f in eval_flops]

    greedy_full = [h["greedy_full"] * 100 for h in stage_eval_history]
    has_stage_alpha = any("stage_greedy_full" in h for h in stage_eval_history)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(eval_exaflops, greedy_full, '-', color='#4CAF50', label='Eval (\u03b1=1.0)', linewidth=1.5, markersize=2, marker='.')

    if has_stage_alpha:
        stage_xs, stage_ys = _stage_alpha_series(eval_exaflops, stage_eval_history)
        ax.plot(stage_xs, stage_ys, '-', color='#FF9800', label='Train (stage \u03b1)', linewidth=1.5, markersize=2, marker='.')

    # Stage divider lines
    prev_stage = stage_eval_history[0]["stage"]
    n_stages_seen = max(h["stage"] for h in stage_eval_history)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_stages_seen, 10)))
    for i, h in enumerate(stage_eval_history):
        if h["stage"] != prev_stage:
            xpos = eval_exaflops[i]
            ax.axvline(x=xpos, color=colors[(h["stage"] - 1) % 10], linestyle='--', alpha=0.5, linewidth=1)
            effective_L = h.get("effective_L", None)
            label = f'S{h["stage"]} L={effective_L}' if effective_L else f'S{h["stage"]}'
            ax.text(xpos, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] > 0 else 100,
                    label, fontsize=7, ha='left', va='top', rotation=90, alpha=0.7)
            prev_stage = h["stage"]

    ax.set_xlabel('Cumulative FLOPs (ExaFLOPs)', fontsize=12)
    ax.set_ylabel('Greedy Full Word Accuracy (%)', fontsize=12)
    ax.set_title('Eval Accuracy vs Compute')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if loss_history:
        cum_flops_all = []
        total = 0
        for h in loss_history:
            total += h.get("tokens", 0) * flops_per_token
            cum_flops_all.append(total / 1e18)
        _draw_resume_markers(ax, loss_history, x_values=cum_flops_all)

    if metadata:
        subtitle = _build_plot_subtitle(metadata)
        fig.suptitle(subtitle, fontsize=9, color='gray', y=0.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, "eval_acc_vs_flops.png"), dpi=150)
    plt.close()

    _print(f"[PLOT] Saved eval_acc_vs_flops.png ({len(stage_eval_history)} evals)")


def plot_loss_vs_gpu_hours(loss_history: List[Dict], output_dir: str, n_gpus: int, n_stages: int,
                           metadata: Dict = None):
    """Plot loss vs GPU-hours (wall_time * n_gpus)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not loss_history or n_gpus is None:
        return

    if not any(h.get("wall_time", 0) > 0 for h in loss_history):
        _print("[PLOT] No wall_time data available, skipping GPU-hours plot")
        return

    # Filter out entries with no wall_time data (e.g. from trainer_state gap recovery)
    valid_history = [h for h in loss_history if h.get("wall_time", 0) > 0]
    if not valid_history:
        _print("[PLOT] No wall_time data available, skipping GPU-hours plot")
        return

    # Use per-entry n_gpus if available (handles GPU count changes across resumes)
    gpu_hours = np.array([h["wall_time"] / 3600 * h.get("n_gpus", n_gpus) for h in valid_history])
    losses = np.array([h["loss"] for h in valid_history])
    stages = [h["stage"] for h in valid_history]
    steps = np.array([h["step"] for h in valid_history])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(gpu_hours, losses, alpha=0.3, linewidth=0.5, color='blue')

    # Step-based smoothing (500 steps) to handle variable entry density
    step_window = 500
    smoothed = np.empty(len(losses))
    smoothed[:] = np.nan
    left = 0
    for i in range(len(losses)):
        while steps[left] < steps[i] - step_window:
            left += 1
        smoothed[i] = np.mean(losses[left:i + 1])
    valid_mask = ~np.isnan(smoothed)
    ax.plot(gpu_hours[valid_mask], smoothed[valid_mask], linewidth=2, color='blue', label=f'Total Loss (w={step_window} steps)')

    # Overlay search/instruct loss lines if available
    _overlay_split_losses(ax, valid_history, gpu_hours.tolist(), steps, step_window)

    stage_info = {}
    for i, h in enumerate(valid_history):
        s = h["stage"]
        if s not in stage_info:
            stage_info[s] = (gpu_hours[i], h.get("effective_L"))

    prev_stage = stages[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(max(stages), n_stages)))

    for i, (gh, stage) in enumerate(zip(gpu_hours, stages)):
        if stage != prev_stage:
            ax.axvline(x=gh, color=colors[(stage - 1) % 10], linestyle='--', alpha=0.7)
            effective_L = stage_info.get(stage, (None, None))[1]
            label = f'S{stage}\nL={effective_L}' if effective_L else f'S{stage}'
            ax.text(gh, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else losses[0],
                    label, fontsize=8, ha='left', va='top')
            prev_stage = stage

    ax.set_xlabel('GPU-Hours')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_title(f'Training Loss vs GPU-Hours ({n_gpus} GPUs)')

    if metadata:
        subtitle = _build_plot_subtitle(metadata)
        fig.suptitle(subtitle, fontsize=9, color='gray', y=0.02)

    ax.legend()
    ax.grid(True, alpha=0.3)
    _draw_resume_markers(ax, valid_history, x_values=gpu_hours.tolist())

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, "loss_vs_gpu_hours.png"), dpi=150)
    plt.close()

    _print(f"[PLOT] Saved loss_vs_gpu_hours.png")


def plot_eval_acc_vs_gpu_hours(stage_eval_history: List[Dict], loss_history: List[Dict],
                                output_dir: str, n_gpus: int, metadata: Dict = None):
    """Plot greedy Full Word accuracy vs GPU-hours with stage dividers."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not stage_eval_history or len(stage_eval_history) < 2:
        return
    if not loss_history or n_gpus is None:
        return

    if not any(h.get("wall_time", 0) > 0 for h in loss_history):
        return

    step_to_gpu_hours = {}
    for h in loss_history:
        step_to_gpu_hours[h["step"]] = h.get("wall_time", 0) / 3600 * h.get("n_gpus", n_gpus)

    eval_gpu_hours = []
    for h in stage_eval_history:
        s = h["step"]
        if s in step_to_gpu_hours:
            eval_gpu_hours.append(step_to_gpu_hours[s])
        else:
            closest = max((k for k in step_to_gpu_hours if k <= s), default=None)
            if closest is not None:
                eval_gpu_hours.append(step_to_gpu_hours[closest])
            else:
                eval_gpu_hours.append(0)

    greedy_full = [h["greedy_full"] * 100 for h in stage_eval_history]
    has_stage_alpha = any("stage_greedy_full" in h for h in stage_eval_history)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Eval accuracy (full alpha) — line with small markers
    ax.plot(eval_gpu_hours, greedy_full, '-', color='#4CAF50', label='Eval (\u03b1=1.0)', linewidth=1.5, markersize=2, marker='.')

    # Training accuracy (stage alpha) — the metric that drives curriculum advancement
    if has_stage_alpha:
        stage_xs, stage_ys = _stage_alpha_series(eval_gpu_hours, stage_eval_history)
        ax.plot(stage_xs, stage_ys, '-', color='#FF9800', label='Train (stage \u03b1)', linewidth=1.5, markersize=2, marker='.')

    # Draw vertical stage divider lines with labels
    n_stages_seen = max(h["stage"] for h in stage_eval_history)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_stages_seen, 10)))
    prev_stage = stage_eval_history[0]["stage"]
    for i, h in enumerate(stage_eval_history):
        if h["stage"] != prev_stage:
            gh = eval_gpu_hours[i]
            ax.axvline(x=gh, color=colors[(h["stage"] - 1) % 10], linestyle='--', alpha=0.5, linewidth=1)
            effective_L = h.get("effective_L", None)
            label = f'S{h["stage"]} L={effective_L}' if effective_L else f'S{h["stage"]}'
            ax.text(gh, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] > 0 else 100,
                    label, fontsize=7, ha='left', va='top', rotation=90, alpha=0.7)
            prev_stage = h["stage"]

    ax.set_xlabel('GPU-Hours', fontsize=12)
    ax.set_ylabel('Greedy Full Word Accuracy (%)', fontsize=12)
    ax.set_title(f'Eval Accuracy vs GPU-Hours ({n_gpus} GPUs)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if loss_history:
        gpu_hours_all = [h.get("wall_time", 0) / 3600 * h.get("n_gpus", n_gpus) for h in loss_history]
        _draw_resume_markers(ax, loss_history, x_values=gpu_hours_all)

    if metadata:
        subtitle = _build_plot_subtitle(metadata)
        fig.suptitle(subtitle, fontsize=9, color='gray', y=0.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, "eval_acc_vs_gpu_hours.png"), dpi=150)
    plt.close()

    _print(f"[PLOT] Saved eval_acc_vs_gpu_hours.png ({len(stage_eval_history)} evals)")


# --------------- Convenience: generate all plots ---------------

def generate_all_plots(output_dir: str, loss_history: List[Dict], n_stages: int,
                       metadata: Dict = None, flops_per_token: int = None,
                       stage_eval_history: List[Dict] = None, n_gpus: int = None):
    """Generate all available plots for a training run."""
    if not loss_history:
        _print("[PLOT] No loss history, skipping all plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Loss plots
    plot_overall_loss(loss_history, output_dir, n_stages, metadata)
    plot_loss_vs_flops(loss_history, output_dir, flops_per_token, n_stages, metadata)
    plot_loss_vs_walltime(loss_history, output_dir, n_stages, metadata)
    plot_loss_vs_gpu_hours(loss_history, output_dir, n_gpus, n_stages, metadata)
    plot_achieved_tflops(loss_history, output_dir, n_stages, metadata)

    # Per-stage loss plots
    stages_seen = sorted(set(h["stage"] for h in loss_history))
    for stage in stages_seen:
        plot_stage_loss(loss_history, stage, output_dir, metadata)

    # Eval plots
    if stage_eval_history and len(stage_eval_history) >= 2:
        plot_stage_eval(stage_eval_history, output_dir, metadata)
        plot_eval_acc_vs_step(stage_eval_history, output_dir, metadata, loss_history)
        plot_eval_acc_vs_flops(stage_eval_history, loss_history, output_dir, flops_per_token, metadata)
        plot_eval_acc_vs_gpu_hours(stage_eval_history, loss_history, output_dir, n_gpus, metadata)


def save_plot_data(output_dir: str, metadata: Dict, flops_per_token: int, n_stages: int,
                   n_gpus: int = None):
    """Save plot metadata so plots can be regenerated standalone."""
    os.makedirs(output_dir, exist_ok=True)
    data = {
        "metadata": metadata,
        "flops_per_token": flops_per_token,
        "n_stages": n_stages,
        "n_gpus": n_gpus,
    }
    path = os.path.join(output_dir, "plot_data.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# --------------- CLI entry point ---------------

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate training plots from saved data",
        epilog="Example: python plot_training.py /scratch/.../nl_output/search/job_8254807"
    )
    parser.add_argument("output_dir", help="Path to training run output directory")
    parser.add_argument("--flops_per_token", type=int, default=None,
                        help="Override FLOPs per token (6*N_params). E.g. 3636854784 for Qwen3-0.6B")
    parser.add_argument("--n_stages", type=int, default=None,
                        help="Override number of stages")
    parser.add_argument("--n_gpus", type=int, default=None,
                        help="Number of GPUs used (for GPU-hours plots)")
    parser.add_argument("--prev_jobs", type=str, default=None,
                        help="Comma-separated list of predecessor job dirs to merge history from (in order)")
    parser.add_argument("--loss_history", type=str, default=None,
                        help="Path to a pre-merged loss_history JSON file (overrides --prev_jobs and live file)")
    parser.add_argument("--eval_history", type=str, default=None,
                        help="Path to a pre-merged stage_eval_history JSON file")
    args = parser.parse_args()

    output_dir = args.output_dir

    # If pre-merged files provided, use them directly (safe for running jobs)
    if args.loss_history:
        with open(args.loss_history) as f:
            loss_history = json.load(f)
        print(f"Loaded {len(loss_history)} records from {args.loss_history}")

        stage_eval_history = None
        eval_src = args.eval_history or os.path.join(output_dir, "stage_eval_history_complete.json")
        if os.path.isfile(eval_src):
            with open(eval_src) as f:
                stage_eval_history = json.load(f)
            print(f"Loaded {len(stage_eval_history)} eval records from {eval_src}")
        else:
            eval_path = os.path.join(output_dir, "stage_eval_history.json")
            if os.path.isfile(eval_path):
                with open(eval_path) as f:
                    stage_eval_history = json.load(f)

        # Skip the normal loading, jump to plot_data
        # Load plot data (saved by training script)
        metadata = None
        flops_per_token = args.flops_per_token
        n_stages = args.n_stages

        plot_data_path = os.path.join(output_dir, "plot_data.json")
        if os.path.isfile(plot_data_path):
            with open(plot_data_path) as f:
                plot_data = json.load(f)
            metadata = plot_data.get("metadata")
            if flops_per_token is None:
                flops_per_token = plot_data.get("flops_per_token")
            if n_stages is None:
                n_stages = plot_data.get("n_stages", 1)
            if args.n_gpus is None:
                args.n_gpus = plot_data.get("n_gpus")
            print(f"Loaded plot_data.json (flops_per_token={flops_per_token}, n_stages={n_stages}, n_gpus={args.n_gpus})")

        if n_stages is None:
            n_stages = max(h["stage"] for h in loss_history) if loss_history else 1
        if flops_per_token is None:
            print("Warning: flops_per_token unknown, FLOPs-based plots will be skipped")

        generate_all_plots(
            output_dir=output_dir,
            loss_history=loss_history,
            n_stages=n_stages,
            metadata=metadata,
            flops_per_token=flops_per_token,
            stage_eval_history=stage_eval_history,
            n_gpus=args.n_gpus,
        )
        print("Done.")
        return

    # Load and merge loss history from predecessor jobs
    loss_history = []
    if args.prev_jobs:
        for prev_dir in args.prev_jobs.split(","):
            prev_dir = prev_dir.strip()
            prev_loss_path = os.path.join(prev_dir, "loss_history.json")
            if os.path.isfile(prev_loss_path):
                with open(prev_loss_path) as f:
                    prev_history = json.load(f)
                # Only include records with steps before the current history starts
                if loss_history:
                    max_step = loss_history[-1]["step"]
                    prev_history = [h for h in prev_history if h["step"] > max_step]
                # Mark boundary between jobs as a resume point
                if prev_history and loss_history:
                    prev_history[0] = {**prev_history[0], "resume": True}
                loss_history.extend(prev_history)
                print(f"Merged {len(prev_history)} records from {prev_dir}")

            prev_eval_path = os.path.join(prev_dir, "stage_eval_history.json")
            # stage eval merging handled below

    loss_path = os.path.join(output_dir, "loss_history.json")
    if not os.path.isfile(loss_path):
        print(f"Error: {loss_path} not found")
        sys.exit(1)
    with open(loss_path) as f:
        current_history = json.load(f)
    if loss_history:
        max_step = loss_history[-1]["step"]
        current_history = [h for h in current_history if h["step"] > max_step]
        # Mark boundary between merged jobs as a resume point
        if current_history:
            current_history[0] = {**current_history[0], "resume": True}
    loss_history.extend(current_history)
    print(f"Total: {len(loss_history)} loss history records")

    # Load and merge stage eval history (optional)
    stage_eval_history = []
    if args.prev_jobs:
        for prev_dir in args.prev_jobs.split(","):
            prev_dir = prev_dir.strip()
            prev_eval_path = os.path.join(prev_dir, "stage_eval_history.json")
            if os.path.isfile(prev_eval_path):
                with open(prev_eval_path) as f:
                    prev_eval = json.load(f)
                if stage_eval_history:
                    max_step = stage_eval_history[-1]["step"]
                    prev_eval = [h for h in prev_eval if h["step"] > max_step]
                stage_eval_history.extend(prev_eval)
                print(f"Merged {len(prev_eval)} eval records from {prev_dir}")

    eval_path = os.path.join(output_dir, "stage_eval_history.json")
    if os.path.isfile(eval_path):
        with open(eval_path) as f:
            current_eval = json.load(f)
        if stage_eval_history:
            max_step = stage_eval_history[-1]["step"]
            current_eval = [h for h in current_eval if h["step"] > max_step]
        stage_eval_history.extend(current_eval)
    if not stage_eval_history:
        stage_eval_history = None
    if stage_eval_history:
        print(f"Total: {len(stage_eval_history)} stage eval records")

    # Load plot data (saved by training script)
    metadata = None
    flops_per_token = args.flops_per_token
    n_stages = args.n_stages

    plot_data_path = os.path.join(output_dir, "plot_data.json")
    if os.path.isfile(plot_data_path):
        with open(plot_data_path) as f:
            plot_data = json.load(f)
        metadata = plot_data.get("metadata")
        if flops_per_token is None:
            flops_per_token = plot_data.get("flops_per_token")
        if n_stages is None:
            n_stages = plot_data.get("n_stages", 1)
        if args.n_gpus is None:
            args.n_gpus = plot_data.get("n_gpus")
        print(f"Loaded plot_data.json (flops_per_token={flops_per_token}, n_stages={n_stages}, n_gpus={args.n_gpus})")
    else:
        print(f"Note: {plot_data_path} not found (old run?), inferring from data")

    # Infer n_stages from data if still not set
    if n_stages is None:
        n_stages = max(h["stage"] for h in loss_history) if loss_history else 1

    # Try to estimate flops_per_token from run_meta.json CLI args if still missing
    if flops_per_token is None:
        meta_path = os.path.join(output_dir, "run_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                run_meta = json.load(f)
            cli = run_meta.get("cli", "")
            # Try to extract model name and estimate params
            if "Qwen3-0.6B" in cli:
                flops_per_token = 6 * 606_142_464
                print(f"Inferred flops_per_token={flops_per_token} from Qwen3-0.6B")

    if flops_per_token is None:
        print("Warning: flops_per_token unknown, FLOPs-based plots will be skipped")
        print("  Use --flops_per_token to specify (e.g. --flops_per_token 3636854784 for Qwen3-0.6B)")

    generate_all_plots(
        output_dir=output_dir,
        loss_history=loss_history,
        n_stages=n_stages,
        metadata=metadata,
        flops_per_token=flops_per_token,
        stage_eval_history=stage_eval_history,
        n_gpus=args.n_gpus,
    )
    print("Done.")


def generate_combined_plots(job_dirs, labels, colors, out_dir, title_prefix="Combined"):
    """Generate combined plots for multiple training runs on the same axes.

    Args:
        job_dirs: list of job output directory paths
        labels: list of legend labels (with lr/bs info)
        colors: list of matplotlib colors
        out_dir: directory to save combined plots
        title_prefix: prefix for plot titles
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Load all data
    runs = {}
    for job_dir, label, color in zip(job_dirs, labels, colors):
        # Follow job chain: check run_meta.json for resume_from_job
        import re as _re
        chain_dirs = [job_dir]
        cur = job_dir
        while True:
            meta_path = os.path.join(cur, "run_meta.json")
            if not os.path.isfile(meta_path):
                break
            with open(meta_path) as f:
                meta = json.load(f)
            rfj = _re.search(r'--resume_from_job (\d+)', meta.get('cli', ''))
            if not rfj:
                break
            parent_dir = os.path.join(os.path.dirname(cur), f"job_{rfj.group(1)}")
            if not os.path.isdir(parent_dir) or parent_dir in chain_dirs:
                break
            chain_dirs.append(parent_dir)
            cur = parent_dir

        # Load loss history — use the latest job's data (it includes parent data if resumed properly)
        # Only merge from parent if child doesn't already include it
        chain_dirs.reverse()  # oldest first
        h = []
        for d in chain_dirs:
            lp = os.path.join(d, "loss_history.json")
            if not os.path.isfile(lp):
                continue
            with open(lp) as f:
                part = json.load(f)
            if not part:
                continue
            if not h:
                h = part
            elif part[0]["step"] > h[-1]["step"]:
                # Child continues from where parent left off — append
                h.extend(part)
            else:
                # Child includes parent data (starts at step 1) — use child's data
                # which is more complete (has both parent history + new training)
                if len(part) > len(h):
                    h = part

        if not h:
            _print(f"Skipping {label}: no loss_history.json")
            continue

        # Merge stage_eval_history from chain
        all_evals = []
        for d in chain_dirs:
            ep = os.path.join(d, "stage_eval_history.json")
            if os.path.isfile(ep):
                with open(ep) as f:
                    all_evals.extend(json.load(f))
        # Deduplicate by step
        seen_steps = set()
        merged_evals = []
        for ev in all_evals:
            if ev["step"] not in seen_steps:
                seen_steps.add(ev["step"])
                merged_evals.append(ev)
        merged_evals.sort(key=lambda e: e["step"])

        # Load plot_data for flops_per_token (from latest job)
        fpt = None
        n_gpus = 4
        for d in reversed(chain_dirs):
            pd_path = os.path.join(d, "plot_data.json")
            if os.path.isfile(pd_path):
                with open(pd_path) as f:
                    pd = json.load(f)
                fpt = pd.get("flops_per_token")
                n_gpus = pd.get("n_gpus") or 4
                if fpt:
                    break

        if fpt is None:
            _print(f"Skipping {label}: no flops_per_token in plot_data.json")
            continue

        if len(chain_dirs) > 1:
            _print(f"  (merged {len(chain_dirs)} jobs: {' -> '.join(os.path.basename(d) for d in chain_dirs)})")

        tokens = np.array([e.get("tokens", 0) for e in h], dtype=np.float64)
        steps = np.array([e["step"] for e in h])
        # Use search_loss if available (for pretrain-mix runs), fall back to total loss
        has_search_loss = any("search_loss" in e for e in h)
        losses = np.array([e.get("search_loss", e["loss"]) for e in h]) if has_search_loss else np.array([e["loss"] for e in h])
        cum_pflops = np.cumsum(tokens * fpt) / 1e15
        gpu_hours = np.array([e.get("wall_time", 0) / 3600 * n_gpus for e in h])

        # Smoothed loss
        step_window = 500
        smoothed = np.empty(len(losses))
        smoothed[:] = np.nan
        left = 0
        for i in range(len(losses)):
            while steps[left] < steps[i] - step_window:
                left += 1
            smoothed[i] = np.mean(losses[left:i + 1])

        # Stage transitions
        prev = h[0]["stage"]
        transitions = []
        for i, e in enumerate(h):
            if e["stage"] != prev:
                transitions.append((i, e.get("effective_L", "?")))
                prev = e["stage"]

        runs[label] = {
            "h": h, "steps": steps, "losses": losses, "smoothed": smoothed,
            "cum_pflops": cum_pflops, "gpu_hours": gpu_hours,
            "transitions": transitions, "evals": merged_evals, "color": color,
        }
        _print(f"Loaded {label}: {len(h):,} entries, L={h[-1].get('effective_L')}")

    if not runs:
        _print("No valid runs to plot")
        return

    sfn = lambda h: max(len(h) // 5000, 1)

    # ── Loss plots (plot_step_comparison style: inline text labels, density-controlled) ──
    for x_key, xlabel, fn in [
        ("pflops", "Cumulative Compute (PFLOPs)", "loss_vs_flops.png"),
        ("gpu_hours", "GPU-hours", "loss_vs_gpu_hours.png"),
        ("steps", "Training Steps", "loss_vs_steps.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        # Pick label vertical-offset direction per run to avoid overlap (alternate above/below)
        for r_idx, (label, md) in enumerate(runs.items()):
            s = sfn(md["h"])
            x = md["steps"] if x_key == "steps" else md["cum_pflops"] if x_key == "pflops" else md["gpu_hours"]
            # Subsample raw trace
            ax.plot(x[::s], md["losses"][::s], alpha=0.15, linewidth=0.5, color=md["color"])
            # For smoothed line, ensure x is monotonically increasing (fix wall_time resets)
            x_sm = x.copy()
            for i in range(1, len(x_sm)):
                if x_sm[i] < x_sm[i-1]:
                    x_sm[i] = x_sm[i-1]
            sm_stride = max(len(md["h"]) // 10000, 1)
            ax.plot(x_sm[::sm_stride], md["smoothed"][::sm_stride], linewidth=1.5,
                    color=md["color"], label=label)
            # Auto-pick label density: ≤16 stages = label all, more = downsample to ~16
            n_stages = len(md["transitions"])
            labeled_every = max(1, n_stages // 16)
            # Alternate above/below per run so labels from different runs don't collide
            place_above = (r_idx % 2 == 0)
            for idx, L in md["transitions"]:
                if labeled_every > 1 and L % labeled_every != 0:
                    continue
                y_text = md["smoothed"][idx] * (1.3 if place_above else 0.75)
                ax.text(x_sm[idx], y_text, f"L={L}", fontsize=6, color=md["color"],
                        ha="center", va="bottom" if place_above else "top")
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.set_title(f'Training Loss vs {xlabel} — {title_prefix}')
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fn), dpi=150)
        plt.close()
        _print(f"Saved {fn}")

    # ── Lookahead vs (FLOPs | GPU-hours) ──
    # Build per-run stage-transition (x, L) lists for each x-axis at the same time so
    # we can reuse them later for the ratio plot.
    per_run_stages = {}  # label -> {"pflops": [(pflop, L), ...], "gpu_hours": [(gh, L), ...]}
    for label, md in runs.items():
        stages_pflops = [(0, md["h"][0].get("effective_L", 0))]
        stages_gpuh = [(0, md["h"][0].get("effective_L", 0))]
        prev = md["h"][0]["stage"]
        for i, e in enumerate(md["h"]):
            if e["stage"] != prev:
                L = e.get("effective_L", 0)
                stages_pflops.append((md["cum_pflops"][i], L))
                stages_gpuh.append((md["gpu_hours"][i], L))
                prev = e["stage"]
        stages_pflops.append((md["cum_pflops"][-1], md["h"][-1].get("effective_L", 0)))
        stages_gpuh.append((md["gpu_hours"][-1], md["h"][-1].get("effective_L", 0)))
        per_run_stages[label] = {"pflops": stages_pflops, "gpu_hours": stages_gpuh}

    def _annotate_stages_boxed(ax, stages, color, labeled_every=1, y_offset=0.5):
        """plot_step_comparison-style boxed L=N labels with white bg + colored border."""
        Ls = sorted({L for _, L in stages if L > 0})
        for L in Ls:
            if labeled_every > 1 and L % labeled_every != 0:
                continue
            # Find x for first occurrence of this L
            x = next(x for x, l in stages if l == L)
            ax.text(x, L + y_offset, f"L={L}", fontsize=7, color=color,
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color,
                              lw=0.5, alpha=0.9))

    for axis_key, xlabel, fname in [
        ("pflops",    "Cumulative Compute (PFLOPs)", "lookahead_vs_flops.png"),
        ("gpu_hours", "GPU-hours",                   "lookahead_vs_gpu_hours.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        for label, md in runs.items():
            stages = per_run_stages[label][axis_key]
            xs, ys = zip(*stages)
            ax.plot(xs, ys, "-", linewidth=2, color=md["color"], label=label)
            # Auto-pick label density: if >20 stages, label every Nth so it's not crowded
            n_stages = len({L for _, L in stages if L > 0})
            labeled_every = max(1, n_stages // 16)
            _annotate_stages_boxed(ax, stages, md["color"], labeled_every=labeled_every)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Achieved Lookahead (L)')
        ax.set_title(f'Achieved Lookahead vs {xlabel} — {title_prefix}')
        ax.legend(loc="lower right", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()
        _print(f"Saved {fname}")

    # ── Compute ratio per L (variant/baseline) ──
    # Only meaningful if there are ≥2 runs; uses the FIRST run as baseline.
    if len(runs) >= 2:
        labels_list = list(runs.keys())
        baseline_label = labels_list[0]
        # baseline_L -> first_pflops dict for quick lookup
        baseline_stages = {L: pf for pf, L in per_run_stages[baseline_label]["pflops"]
                           if L > 0}
        fig, ax = plt.subplots(figsize=(14, 7))
        bar_width = 0
        # Collect all variant L values to size bars proportionally
        for v_idx, label in enumerate(labels_list[1:], start=1):
            md = runs[label]
            v_stages = {L: pf for pf, L in per_run_stages[label]["pflops"] if L > 0}
            shared_Ls = sorted(set(v_stages.keys()) & set(baseline_stages.keys()))
            if not shared_Ls:
                continue
            ratios = [v_stages[L] / baseline_stages[L] for L in shared_Ls]
            # Offset bars per variant to avoid overlap
            n_variants = len(labels_list) - 1
            if bar_width == 0:
                bar_width = (min(np.diff(shared_Ls)) if len(shared_Ls) > 1 else 4) * 0.8 / n_variants
            offset = (v_idx - 1 - (n_variants - 1) / 2) * bar_width
            ax.bar([L + offset for L in shared_Ls], ratios, width=bar_width,
                   color=md["color"], edgecolor="black", linewidth=0.5,
                   label=f"{label} / {baseline_label}")
            # Annotate
            for L, r in zip(shared_Ls, ratios):
                if r < 100:  # skip absurd noise from tiny baseline at very early L
                    ax.text(L + offset, r * 1.02, f"{r:.1f}x",
                            ha="center", va="bottom", fontsize=7, color="black")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1,
                   label=f"{baseline_label} baseline")
        ax.set_xlabel('Lookahead Depth L', fontsize=12)
        ax.set_ylabel(f'Compute Ratio (variant / {baseline_label})', fontsize=12)
        ax.set_title(f'Per-L Compute Ratio vs Baseline — {title_prefix}', fontsize=13)
        # Cap y-axis at 95th percentile of finite ratios to avoid early-stage noise dominating
        ax.set_ylim(0, 8)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "compute_ratio_per_L.png"), dpi=150)
        plt.close()
        _print("Saved compute_ratio_per_L.png")

    # ── Eval accuracy plots ──
    for x_key, xlabel, suffix in [
        ("pflops", "Cumulative Compute (PFLOPs)", "flops"),
        ("gpu_hours", "GPU-hours", "gpu_hours"),
        ("steps", "Training Steps", "steps"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 7))
        for label, md in runs.items():
            if not md["evals"]:
                continue
            # Build monotonic x-axis arrays
            x_arr = md["steps"] if x_key == "steps" else md["cum_pflops"] if x_key == "pflops" else md["gpu_hours"]
            x_mono = x_arr.copy()
            for i in range(1, len(x_mono)):
                if x_mono[i] < x_mono[i-1]:
                    x_mono[i] = x_mono[i-1]
            xs, ys = [], []
            for ev in md["evals"]:
                idx = min(np.searchsorted(md["steps"], ev["step"]), len(md["steps"]) - 1)
                xs.append(x_mono[idx])
                ys.append(ev.get("greedy_full", 0) * 100)
            # Sort by x to prevent backwards lines
            pairs = sorted(zip(xs, ys))
            xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
            ax.plot(xs, ys, 'o-', color=md["color"], label=label, markersize=3, linewidth=1.5)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Greedy Full Accuracy (%)', fontsize=12)
        ax.set_title(f'Eval Accuracy vs {xlabel} — {title_prefix}', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"eval_acc_vs_{suffix}.png"), dpi=150)
        plt.close()
        _print(f"Saved eval_acc_vs_{suffix}.png")


def combined_main():
    """CLI for combined plots: python plot_training.py --combined dir1:label1:color dir2:label2:color ..."""
    parser = argparse.ArgumentParser(description="Generate combined plots for multiple runs")
    parser.add_argument("runs", nargs="+",
                        help="Each run as dir:label:color (e.g. /path/to/job:'Pythia-410M (lr=5e-5, bs=192)':#FF9800)")
    parser.add_argument("--out_dir", required=True, help="Output directory for combined plots")
    parser.add_argument("--title", default="Combined", help="Title prefix for plots")
    args = parser.parse_args()

    job_dirs, labels, colors = [], [], []
    for run in args.runs:
        parts = run.split(":", 2)
        if len(parts) != 3:
            print(f"Error: each run must be dir:label:color, got: {run}")
            sys.exit(1)
        job_dirs.append(parts[0])
        labels.append(parts[1])
        colors.append(parts[2])

    generate_combined_plots(job_dirs, labels, colors, args.out_dir, args.title)
    print("Done.")


if __name__ == "__main__":
    # Check if --combined mode
    if "--combined" in sys.argv or any(a.startswith("--out_dir") for a in sys.argv):
        sys.argv = [a for a in sys.argv if a != "--combined"]
        combined_main()
    else:
        main()
