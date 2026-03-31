#!/usr/bin/env python3
"""Plot pptrain curriculum accuracy with curriculum advancement markers.

Usage:
    python plot_pptrain_curr.py /scratch/gautschi/mnickel/pptrain
    python plot_pptrain_curr.py /scratch/gautschi/mnickel/pptrain --output curr_plot.png
    python plot_pptrain_curr.py /scratch/gautschi/mnickel/pptrain --smoothing 100
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


def plot_pptrain_curr(
    run_dir: Path,
    output_path: Path = None,
    smoothing: int = 50,
    figsize: Tuple[int, int] = (12, 6),
    ymin: float = 0.5,
) -> None:
    """Plot pptrain curriculum accuracy with curriculum markers."""

    entries = load_train_log(run_dir)

    if not entries:
        print("No log entries found.")
        return

    # Extract data
    steps = np.array([e["step"] for e in entries])
    curr_acc = np.array([e["curr_acc"] for e in entries])

    # Find curriculum stage changes
    stage_changes = find_stage_changes(entries)

    # Create smoothed accuracy
    smoothed_curr_acc = smooth(curr_acc, smoothing)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot raw accuracy (semi-transparent)
    ax.plot(steps, curr_acc, alpha=0.3, color="forestgreen", linewidth=0.8,
            label="Raw curr_acc")

    # Plot smoothed accuracy
    ax.plot(steps, smoothed_curr_acc, color="darkgreen", linewidth=2,
            label=f"Smoothed (EMA {smoothing})")

    # Add horizontal line at 98% threshold
    ax.axhline(y=0.98, color="orange", linestyle="-", alpha=0.7, linewidth=1.5,
               label="Threshold (98%)")

    # Add vertical lines at curriculum advancements
    for step, stage in stage_changes:
        ax.axvline(x=step, color="red", linestyle="--", alpha=0.7, linewidth=1)
        # Add label at top of plot
        ax.text(step, 1.01, f"L={stage}", rotation=90, va="bottom", ha="right",
                fontsize=9, color="red", alpha=0.8)

    # Labels and formatting
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Curriculum Accuracy", fontsize=12)
    ax.set_title(f"Pre-Pretraining Curriculum Accuracy: {run_dir.name}", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Set y-axis limits - focus on high accuracy range
    ax.set_ylim(ymin, 1.02)

    # Add info text
    final_stage = entries[-1].get("stage", "?")
    final_step = entries[-1].get("step", "?")
    final_acc = entries[-1].get("curr_acc", "?")
    info_text = f"Final: step={final_step}, stage={final_stage}, acc={final_acc:.3f}"
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
    parser = argparse.ArgumentParser(description="Plot pptrain curriculum accuracy with markers")
    parser.add_argument("run_dir", type=Path, help="Path to pptrain run directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output path for plot (if not specified, displays interactively)")
    parser.add_argument("--smoothing", "-s", type=int, default=50,
                        help="Smoothing window for EMA (default: 50)")
    parser.add_argument("--figsize", type=str, default="12,6",
                        help="Figure size as 'width,height' (default: 12,6)")
    parser.add_argument("--ymin", type=float, default=0.5,
                        help="Minimum y-axis value (default: 0.5, use 0 to see full range)")

    args = parser.parse_args()

    # Parse figsize
    figsize = tuple(map(int, args.figsize.split(",")))

    plot_pptrain_curr(
        run_dir=args.run_dir,
        output_path=args.output,
        smoothing=args.smoothing,
        figsize=figsize,
        ymin=args.ymin,
    )


if __name__ == "__main__":
    main()
