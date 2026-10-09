#!/usr/bin/env python3
"""Pretrained versus random initialization, both models in ONE panel, capped at 1.5M PFLOPs.

Requested 2026-09-25 as a one-figure replacement for the two-panel pretrained_vs_randominit_combined.
Encoding: colour is the model, linestyle is the initialization (solid pretrained, dotted random).
(The older plot_pretrained_vs_reinit_single.py used the opposite encoding and predates print size.)

Data, chains, loader and completed-stage convention are taken verbatim from
plot_pretrained_vs_reinit_combined.py (exec of its head), so the curves are identical to the two-panel
figure. Every chain is cut at 1.5M PFLOPs. What the cap hides (see that script's sibling docstring):
Pythia-1.4B pretrained runs on to 5.0M (L=75) and Pythia random to 6.0M (L=36); Qwen3-0.6B pretrained
ends at 1,504,595 (L=245); Qwen random runs to 4.8M (L=62). At 1.5M Pythia has not reached its
plateau, so do not read Pythia ceilings off this figure.

Standalone, drawn at printed size (half text width, print_style.HALF) and included at 100%.
Writes paper/figures/pretrained_vs_randominit_onepanel.{png,pdf}.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SRC = "/home/huan2073/nl-fine-tuning/nl/figure_src/plot_pretrained_vs_reinit_combined.py"
src = open(SRC).read()
ns = {"__file__": SRC}
exec(compile(src[:src.index("fig, axes = plt.subplots")], SRC, "exec"), ns)
PANELS, load, fmt, PS, OUT = ns["PANELS"], ns["load"], ns["fmt"], ns["PS"], ns["OUT"]

CAP = 1_500_000
YMAX = 260
# LOGLOG=1 draws both axes on log scales and writes a separate *_loglog output. Log axes cannot show the
# (0, 0) origin, so points with zero compute or L=0 are dropped: every chain starts at its first completed
# stage (L=1, at 25 to 1,500 PFLOPs). x runs 10..1.5M, y 0.8..300.
LOGLOG = os.environ.get("LOGLOG") == "1"
COL = {"Qwen3-0.6B": "#1f77b4", "Pythia-1.4B": "#ff7f0e"}   # model colours; init is the linestyle

fig, ax = plt.subplots(figsize=PS.HALF)
for key in ("qwen06b", "1.4B"):                 # Qwen first so its steep pretrained curve sits underneath
    model, N, arms, _ = PANELS[key]
    for name, _, dirs in arms:
        pf, Ls = load(dirs, N)
        m = (pf <= CAP) & ((pf > 0) & (Ls > 0) if LOGLOG else True)
        x = np.append(pf[m], CAP); y = np.append(Ls[m], Ls[m][-1])
        ls = ":" if "random" in name else "-"
        ax.plot(x, y, ls, color=COL[model], linewidth=PS.LW, alpha=0.9, drawstyle="steps-post")
        print(f"{name}: chain end {pf[-1]:,.0f} PFLOPs, L at 1.5M cap = {int(y[-1])}")

if LOGLOG:
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(10, CAP)
    ax.set_ylim(0.8, 300)   # below 1 so the L=1 stretches are not hidden in the x-axis spine
    ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    ax.set_yticks([1, 10, 100])
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v/1e6:g}M" if v >= 1e6 else (f"{v/1e3:g}k" if v >= 1e3 else f"{v:g}")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
else:
    ax.set_xlim(-CAP * 0.02, CAP)
    ax.set_ylim(0, YMAX)
    ax.set_yticks(np.arange(0, YMAX + 1, 50))
    ax.set_xticks([0, 500_000, 1_000_000, 1_500_000])
    ax.xaxis.set_major_formatter(fmt)
ax.set_xlabel("Cumulative Compute (PFLOPs)")
ax.set_ylabel("Achieved Lookahead $L$")
PS.grid(ax)

# Two keys in one legend: colour = model, linestyle = initialization. The band between the Qwen pretrained
# curve (above ~180 past 0.5M) and the Pythia pretrained curve (below ~70) is empty on the right.
handles = [Line2D([0], [0], color=COL["Qwen3-0.6B"], lw=PS.LW),
           Line2D([0], [0], color=COL["Pythia-1.4B"], lw=PS.LW),
           Line2D([0], [0], color="0.3", lw=PS.LW, ls="-"),
           Line2D([0], [0], color="0.3", lw=PS.LW, ls=":")]
labels = ["Qwen3-0.6B", "Pythia-1.4B", "Pretrained", "Random"]   # short: the caption says "initialization"
if LOGLOG:   # on log-log every curve starts low on the left, so the upper left is empty
    ax.legend(handles, labels, loc="upper left", ncol=2, handlelength=1.4, columnspacing=0.8)
else:
    ax.legend(handles, labels, loc="lower right", bbox_to_anchor=(1.0, 0.25), ncol=2,
              handlelength=1.4, columnspacing=0.8)

fig.tight_layout(pad=0.1)
PS.save(fig, f"{OUT}/pretrained_vs_randominit_onepanel{'_loglog' if LOGLOG else ''}")
