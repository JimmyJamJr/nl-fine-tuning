#!/usr/bin/env python3
"""Plot accuracy curves from evaluation CSV files.

Supports log-scale y-axis to emphasize low-accuracy regions where
catastrophic forgetting comparisons are most relevant.

Usage:
    python plot_accuracy.py results.csv --output plot.png
    python plot_accuracy.py results.csv --runs run1 run2 --log_scale
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


COLORS = ["steelblue", "darkorange", "forestgreen", "crimson", "purple", "brown", "teal", "olive"]


def load_csv(csv_path: Path) -> Dict[str, List[Dict]]:
    """Load evaluation results from CSV.

    Returns:
        Dict mapping run_name to list of {step, accuracy, correct, total}
    """
    results = defaultdict(list)

    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["run_name"]].append({
                "step": int(row["step"]),
                "accuracy": float(row["accuracy"]),
                "correct": int(row["correct"]),
                "total": int(row["total"]),
            })

    for run_name in results:
        results[run_name].sort(key=lambda x: x["step"])

    return dict(results)


def plot_accuracy(
    results: Dict[str, List[Dict]],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Search Task Accuracy vs. Training Steps",
    labels: Optional[Dict[str, str]] = None,
    log_scale: bool = True,
    runs: Optional[List[str]] = None,
) -> None:
    """Plot accuracy curves.

    Args:
        results: Dict mapping run_name to list of result dicts
        output_path: Path to save plot (displays if None)
        figsize: Figure size (width, height)
        title: Plot title
        labels: Optional dict mapping run_name to display label
        log_scale: Use log scale for y-axis (emphasizes low values)
        runs: Optional list of run names to include (all if None)
    """
    labels = labels or {}
    fig, ax = plt.subplots(figsize=figsize)

    # Filter runs if specified
    if runs:
        results = {k: v for k, v in results.items() if k in runs}

    for i, (run_name, run_results) in enumerate(results.items()):
        if not run_results:
            continue

        steps = np.array([r["step"] for r in run_results])
        accuracies = np.array([r["accuracy"] for r in run_results])

        # For log scale, clamp zeros to small value
        if log_scale:
            accuracies = np.maximum(accuracies, 0.001)

        label = labels.get(run_name, run_name)
        color = COLORS[i % len(COLORS)]

        ax.plot(steps, accuracies, color=color, linewidth=1.5, alpha=0.8, label=label)

    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=12)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(0.001, 1.0)
    else:
        ax.set_ylim(0, 1.0)

    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both" if log_scale else "major")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot accuracy from evaluation CSV")
    parser.add_argument("csv_path", type=Path, help="Path to checkpoint_eval.csv")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output path")
    parser.add_argument("--title", type=str, default="Search Task Accuracy vs. Training Steps")
    parser.add_argument("--runs", nargs="+", default=None, help="Specific runs to include")
    parser.add_argument("--no_log", action="store_true", help="Use linear scale instead of log")
    parser.add_argument("--figsize", type=str, default="10,6", help="Figure size as w,h")

    args = parser.parse_args()

    results = load_csv(args.csv_path)
    figsize = tuple(map(int, args.figsize.split(",")))

    plot_accuracy(
        results=results,
        output_path=args.output,
        figsize=figsize,
        title=args.title,
        log_scale=not args.no_log,
        runs=args.runs,
    )


if __name__ == "__main__":
    main()
