#!/usr/bin/env python3
"""Plot training loss curves from JSONL log files.

Supports log-log scale for comparing loss across different runs.

Usage:
    python plot_loss.py /path/to/run1 /path/to/run2 --output loss.png
    python plot_loss.py run_dir --labels "Run 1" "Run 2"
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


COLORS = ["steelblue", "darkorange", "forestgreen", "crimson", "purple", "brown", "teal", "olive"]


def load_train_log(run_dir: Path) -> List[Dict]:
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


def plot_loss(
    run_dirs: List[Path],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Training Loss",
    labels: Optional[List[str]] = None,
    log_log: bool = True,
) -> None:
    """Plot loss curves for multiple runs.

    Args:
        run_dirs: List of run directory paths
        output_path: Path to save plot (displays if None)
        figsize: Figure size (width, height)
        title: Plot title
        labels: Optional list of labels (one per run_dir)
        log_log: Use log-log scale (default True)
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, run_dir in enumerate(run_dirs):
        try:
            entries = load_train_log(run_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

        if not entries:
            print(f"No log entries found for {run_dir.name}, skipping.")
            continue

        steps = np.array([e["step"] for e in entries])
        losses = np.array([e["loss"] for e in entries])

        label = labels[i] if labels and i < len(labels) else run_dir.name
        color = COLORS[i % len(COLORS)]

        if log_log:
            ax.loglog(steps, losses, alpha=0.7, color=color, linewidth=1.0, label=label)
        else:
            ax.plot(steps, losses, alpha=0.7, color=color, linewidth=1.0, label=label)

    ax.set_xlabel("Steps" + (" (log)" if log_log else ""), fontsize=11)
    ax.set_ylabel("Loss" + (" (log)" if log_log else ""), fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both" if log_log else "major")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training loss curves")
    parser.add_argument("run_dirs", type=Path, nargs="+", help="Run directories")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output path")
    parser.add_argument("--title", type=str, default="Training Loss")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels for each run")
    parser.add_argument("--no_log", action="store_true", help="Use linear scale")
    parser.add_argument("--figsize", type=str, default="10,6", help="Figure size as w,h")

    args = parser.parse_args()

    figsize = tuple(map(int, args.figsize.split(",")))

    plot_loss(
        run_dirs=args.run_dirs,
        output_path=args.output,
        figsize=figsize,
        title=args.title,
        labels=args.labels,
        log_log=not args.no_log,
    )


if __name__ == "__main__":
    main()
