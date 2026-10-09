#!/usr/bin/env python3
"""Appendix figure: mean training compute per completed stage for Pythia-1.4B, in consecutive
five-stage groups.

Pythia-1.4B is the chain that sits closest to its fitted ceiling (last completed L = 75 against
Lmax = 81.9), so it is the natural case for showing how the cost of a stage grows with depth.
Groups are equal sized by construction, five stages each, 1-5 through 71-75, so no group is
compared against one covering a different number of stages.  The vertical axis is linear so that
the growth is read directly rather than through a logarithm.

Data source: paper/dm_heldout/per_stage_compute_Pythia-1p4B.csv, the same per-stage table kept in
the supplement.  Compute in PFLOPs = cumulative training tokens x 6N, completed-stage convention.
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Drawn at printed size (full text width) and included at 100% scale: ticks and bar values 7pt,
# axis labels 8pt (print_style).
sys.path.insert(0, "/home/huan2073/nl-fine-tuning/nl/figure_src")
import print_style as PS
PS.apply()

CSV = "/home/huan2073/nl-fine-tuning/nl/paper/dm_heldout/per_stage_compute_Pythia-1p4B.csv"
OUT = "/home/huan2073/nl-fine-tuning/nl/paper/figures"
os.makedirs(OUT, exist_ok=True)

# The Pythia-1.4B colour used elsewhere in the paper: inferno(0.15 + 0.65*2/3) in the family-scaling
# panel of the merged Figure 7+8, and the same value hardcoded in the Lmax scaling figure.  Do not
# take colours from the RUNS dict in plot_fig7_8_house.py: those are overridden by the colormap at
# render time and never reach a figure.
COLOR = "#d84c3e"
EDGE = "#9b372d"
GROUP = 5

rows = list(csv.DictReader(open(CSV)))
L = np.array([int(r["L_completed"]) for r in rows])
dC = np.array([float(r["stage_pflops"]) for r in rows])
assert np.array_equal(L, np.arange(1, L.max() + 1)), "stages are not contiguous 1..L_end"
assert L.max() % GROUP == 0, f"L_end={L.max()} is not a whole number of {GROUP}-stage groups"

edges = np.arange(0, L.max(), GROUP)
means = np.array([dC[(L >= a + 1) & (L <= a + GROUP)].mean() for a in edges])
labels = [f"{a + 1}–{a + GROUP}" for a in edges]
assert all(((L >= a + 1) & (L <= a + GROUP)).sum() == GROUP for a in edges), "unequal group sizes"


def compact(v):
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    if v >= 1e5:
        return f"{v / 1e3:.0f}k"
    if v >= 1e3:
        return f"{v / 1e3:.1f}k"
    return f"{v:.0f}"


fig, ax = plt.subplots(figsize=(5.5, 2.2))
x = np.arange(len(means))
ax.bar(x, means, width=0.74, color=COLOR, edgecolor=EDGE, linewidth=0.5, zorder=3)
ax.grid(True, axis="y", color="0.6", alpha=0.35, linewidth=0.4, zorder=0)

for xi, m in zip(x, means):
    ax.annotate(compact(m), (xi, m), xytext=(0, 1.5), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, color="0.25")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Lookahead stages completed")
ax.set_ylabel("Mean compute per\ncompleted stage (PFLOPs)")
ax.yaxis.set_major_formatter(FuncFormatter(
    lambda v, _: "0" if v == 0 else (f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k")))
ax.set_xlim(-0.7, len(means) - 0.3)
ax.set_ylim(0, means.max() * 1.13)
ax.margins(x=0)

fig.tight_layout(pad=0.1)
PS.save(fig, f"{OUT}/per_stage_cost_pythia14b")

print(f"\n{'group':>8}{'mean PFLOPs/stage':>20}{'group total':>16}{'x previous group':>19}")
tot = dC.sum()
for i, (lab, m) in enumerate(zip(labels, means)):
    xp = "   -  " if i == 0 else f"{m / means[i - 1]:5.2f}x"
    print(f"{lab:>8}{m:>20,.0f}{m * GROUP:>16,.0f}{xp:>19}")
print(f"\nchain total across the 75 completed stages: {tot:,.0f} PFLOPs")
print(f"first group / last group: {means[0]:,.0f} vs {means[-1]:,.0f}  ({means[-1] / means[0]:,.0f}x)")
