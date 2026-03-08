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


def _draw_resume_markers(ax, loss_history: List[Dict], x_key: str = "step", x_values=None):
    """Draw vertical dashed red lines at resume points in loss_history."""
    for i, h in enumerate(loss_history):
        if h.get("resume"):
            if x_values is not None:
                x = x_values[i]
            else:
                x = h.get(x_key, 0)
            ax.axvline(x=x, color='red', linestyle=':', alpha=0.5, linewidth=1.5, zorder=5)


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

    steps = [h["step"] for h in stage_data]
    losses = [h["loss"] for h in stage_data]

    alpha = stage_data[0].get("alpha", None)
    effective_L = stage_data[0].get("effective_L", None)

    start_step = steps[0]
    rel_steps = [s - start_step for s in steps]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rel_steps, losses, alpha=0.3, linewidth=0.5, color='blue', label='Raw')

    window = min(200, len(losses) // 3) or 1
    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(losses)), smoothed, linewidth=2, color='blue', label=f'Smoothed (w={window})')

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

    title = f'Stage {stage} Loss ({len(stage_data)} steps, final={losses[-1]:.4f})'
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

    steps = [h["step"] for h in loss_history]
    losses = [h["loss"] for h in loss_history]
    stages = [h["stage"] for h in loss_history]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, losses, alpha=0.3, linewidth=0.5, color='blue')

    window = min(100, len(losses) // 10) or 1
    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
        ax.plot(steps[window - 1:], smoothed, linewidth=2, color='blue', label=f'Smoothed (w={window})')

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

    cumulative_flops = []
    total_flops = 0
    for h in loss_history:
        tokens = h.get("tokens", 0)
        total_flops += tokens * flops_per_token
        cumulative_flops.append(total_flops)

    losses = [h["loss"] for h in loss_history]
    stages = [h["stage"] for h in loss_history]

    cumulative_pflops = [f / 1e15 for f in cumulative_flops]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cumulative_pflops, losses, alpha=0.3, linewidth=0.5, color='blue')

    window = min(100, len(losses) // 10) or 1
    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
        ax.plot(cumulative_pflops[window - 1:], smoothed, linewidth=2, color='blue', label=f'Smoothed (w={window})')

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

    if metadata:
        subtitle = _build_plot_subtitle(metadata)
        fig.suptitle(subtitle, fontsize=9, color='gray', y=0.02)

    ax.legend()
    ax.grid(True, alpha=0.3)
    _draw_resume_markers(ax, loss_history, x_values=cumulative_pflops)

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

    wall_times = [h.get("wall_time", 0) / 3600 for h in loss_history]
    losses = [h["loss"] for h in loss_history]
    stages = [h["stage"] for h in loss_history]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wall_times, losses, alpha=0.3, linewidth=0.5, color='blue')

    window = min(100, len(losses) // 10) or 1
    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
        ax.plot(wall_times[window - 1:], smoothed, linewidth=2, color='blue', label=f'Smoothed (w={window})')

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

    if metadata:
        subtitle = _build_plot_subtitle(metadata)
        fig.suptitle(subtitle, fontsize=9, color='gray', y=0.02)

    ax.legend()
    ax.grid(True, alpha=0.3)
    _draw_resume_markers(ax, loss_history, x_key="wall_time")

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

    window = min(100, len(tflops) // 10) or 1
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

    has_stage_alpha = "stage_greedy_full" in stage_eval_history[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, greedy_full, 's-', color='#4CAF50', label='Full Alpha (\u03b1=1.0) Full Word', markersize=5)
    ax.plot(steps, greedy_first, 'o-', color='#2196F3', label='Full Alpha (\u03b1=1.0) First Token', markersize=5, alpha=0.6)

    if has_stage_alpha:
        stage_full = [h.get("stage_greedy_full", 0) * 100 for h in stage_eval_history]
        stage_first = [h.get("stage_greedy_first", 0) * 100 for h in stage_eval_history]
        ax.plot(steps, stage_full, 's--', color='#FF9800', label='Stage Alpha Full Word', markersize=5)
        ax.plot(steps, stage_first, 'o--', color='#F44336', label='Stage Alpha First Token', markersize=5, alpha=0.6)

    for i, h in enumerate(stage_eval_history):
        if "effective_L" in h:
            ax.annotate(f'L={h["effective_L"]}', (steps[i], greedy_full[i]),
                        textcoords="offset points", xytext=(0, 8), fontsize=7, ha='center')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Greedy Accuracy (%)', fontsize=12)
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
    greedy_first = [h["greedy_first"] * 100 for h in stage_eval_history]
    has_stage_alpha = "stage_greedy_full" in stage_eval_history[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(eval_exaflops, greedy_full, 's-', color='#4CAF50', label='Full Alpha (\u03b1=1.0) Full Word', markersize=5)
    ax.plot(eval_exaflops, greedy_first, 'o-', color='#2196F3', label='Full Alpha (\u03b1=1.0) First Token', markersize=5, alpha=0.6)

    if has_stage_alpha:
        stage_full = [h.get("stage_greedy_full", 0) * 100 for h in stage_eval_history]
        stage_first = [h.get("stage_greedy_first", 0) * 100 for h in stage_eval_history]
        ax.plot(eval_exaflops, stage_full, 's--', color='#FF9800', label='Stage Alpha Full Word', markersize=5)
        ax.plot(eval_exaflops, stage_first, 'o--', color='#F44336', label='Stage Alpha First Token', markersize=5, alpha=0.6)

    for i, h in enumerate(stage_eval_history):
        if "effective_L" in h:
            ax.annotate(f'L={h["effective_L"]}', (eval_exaflops[i], greedy_full[i]),
                        textcoords="offset points", xytext=(0, 8), fontsize=7, ha='center')

    ax.set_xlabel('Cumulative FLOPs (ExaFLOPs)', fontsize=12)
    ax.set_ylabel('Greedy Accuracy (%)', fontsize=12)
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


# --------------- Convenience: generate all plots ---------------

def generate_all_plots(output_dir: str, loss_history: List[Dict], n_stages: int,
                       metadata: Dict = None, flops_per_token: int = None,
                       stage_eval_history: List[Dict] = None):
    """Generate all available plots for a training run."""
    if not loss_history:
        _print("[PLOT] No loss history, skipping all plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Loss plots
    plot_overall_loss(loss_history, output_dir, n_stages, metadata)
    plot_loss_vs_flops(loss_history, output_dir, flops_per_token, n_stages, metadata)
    plot_loss_vs_walltime(loss_history, output_dir, n_stages, metadata)
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


def save_plot_data(output_dir: str, metadata: Dict, flops_per_token: int, n_stages: int):
    """Save plot metadata so plots can be regenerated standalone."""
    os.makedirs(output_dir, exist_ok=True)
    data = {
        "metadata": metadata,
        "flops_per_token": flops_per_token,
        "n_stages": n_stages,
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
    args = parser.parse_args()

    output_dir = args.output_dir

    # Load loss history
    loss_path = os.path.join(output_dir, "loss_history.json")
    if not os.path.isfile(loss_path):
        print(f"Error: {loss_path} not found")
        sys.exit(1)
    with open(loss_path) as f:
        loss_history = json.load(f)
    print(f"Loaded {len(loss_history)} loss history records")

    # Load stage eval history (optional)
    stage_eval_history = None
    eval_path = os.path.join(output_dir, "stage_eval_history.json")
    if os.path.isfile(eval_path):
        with open(eval_path) as f:
            stage_eval_history = json.load(f)
        print(f"Loaded {len(stage_eval_history)} stage eval records")

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
        print(f"Loaded plot_data.json (flops_per_token={flops_per_token}, n_stages={n_stages})")
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
    )
    print("Done.")


if __name__ == "__main__":
    main()
