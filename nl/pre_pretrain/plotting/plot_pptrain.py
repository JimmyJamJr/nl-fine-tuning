#!/usr/bin/env python3
"""Plot pptrain loss with curriculum advancement markers.

Usage:
    python plot_pptrain.py /scratch/gautschi/mnickel/pptrain
    python plot_pptrain.py /scratch/gautschi/mnickel/pptrain --output loss_plot.png
    python plot_pptrain.py /scratch/gautschi/mnickel/pptrain --smoothing 100
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_train_log(run_dir: Path) -> List[dict]:
    """Load training log from run directory."""
    log_path = run_dir / "logs" / "train.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found: {log_path}")

    entries = []
    with log_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Apply exponential moving average smoothing."""
    if window <= 1:
        return values

    alpha = 2.0 / (window + 1)
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]

    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]

    return smoothed


def find_stage_changes(entries: List[dict]) -> List[Tuple[int, int]]:
    """Find steps where curriculum stage changes.

    Returns list of (step, new_stage) tuples.
    """
    changes = []
    prev_stage = None

    for entry in entries:
        stage = entry.get("stage")
        step = entry.get("step")

        if stage is not None and stage != prev_stage:
            if prev_stage is not None:  # Skip the initial stage
                changes.append((step, stage))
            prev_stage = stage

    return changes


def plot_pptrain_loss(
    run_dir: Path,
    output_path: Path = None,
    smoothing: int = 50,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """Plot pptrain loss with curriculum markers."""

    entries = load_train_log(run_dir)

    if not entries:
        print("No log entries found.")
        return

    # Extract data
    steps = np.array([e["step"] for e in entries])
    losses = np.array([e["loss"] for e in entries])

    # Find curriculum stage changes
    stage_changes = find_stage_changes(entries)

    # Create smoothed loss
    smoothed_losses = smooth(losses, smoothing)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot raw loss (semi-transparent)
    ax.semilogy(steps, losses, alpha=0.3, color="steelblue", linewidth=0.8, label="Raw loss")

    # Plot smoothed loss
    ax.semilogy(steps, smoothed_losses, color="darkblue", linewidth=2,
                label=f"Smoothed (EMA {smoothing})")

    # Add vertical lines at curriculum advancements
    for step, stage in stage_changes:
        ax.axvline(x=step, color="red", linestyle="--", alpha=0.7, linewidth=1)
        # Add label at top of plot
        y_pos = ax.get_ylim()[1] * 0.9
        ax.text(step, y_pos, f"L={stage}", rotation=90, va="top", ha="right",
                fontsize=9, color="red", alpha=0.8)

    # Labels and formatting
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss (log scale)", fontsize=12)
    ax.set_title(f"Pre-Pretraining Loss: {run_dir.name}", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    # Add info text
    final_stage = entries[-1].get("stage", "?")
    final_step = entries[-1].get("step", "?")
    info_text = f"Final: step={final_step}, stage={final_stage}"
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot pptrain loss with curriculum markers")
    parser.add_argument("run_dir", type=Path, help="Path to pptrain run directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output path for plot (if not specified, displays interactively)")
    parser.add_argument("--smoothing", "-s", type=int, default=50,
                        help="Smoothing window for EMA (default: 50)")
    parser.add_argument("--figsize", type=str, default="12,6",
                        help="Figure size as 'width,height' (default: 12,6)")

    args = parser.parse_args()

    # Parse figsize
    figsize = tuple(map(int, args.figsize.split(",")))

    plot_pptrain_loss(
        run_dir=args.run_dir,
        output_path=args.output,
        smoothing=args.smoothing,
        figsize=figsize,
    )


if __name__ == "__main__":
    main()
