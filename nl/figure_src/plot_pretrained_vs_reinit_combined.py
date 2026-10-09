#!/usr/bin/env python3
"""Pretrained versus random initialization, one two-panel figure: Pythia-1.4B left, Qwen3-0.6B right.
Panels carry no letter labels (removed 2026-09-21 at the author's request, paper-wide).

Shared y-scale (0..260), completed-stage convention, steps-post lines at 72% opacity so the two arms
stay visible where they overlap, k/M tick labels, x-axis ending at the matched-compute cap.

The legend sits INSIDE panel (a) rather than in a band under the figure, which removes the reserved
vertical strip and makes the whole figure shorter. It carries the generic arm names because it
describes both panels.

Standalone: this sets the house style itself (print_style) instead of relying on render_house.py.
Drawn at its printed size, two panels at 0.8 text width (print_style.FULL2_80), and included at
100% scale. Each panel keeps a one-line label naming its model; the two-line titles that restated
the caption are gone.
Writes paper/figures/pretrained_vs_randominit_combined.{png,pdf}.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

sys.path.insert(0, "/home/huan2073/nl-fine-tuning/nl/figure_src")
import print_style as PS
PS.apply()

SCRATCH = "/scratch/gautschi/huan2073"
OUT = "/home/huan2073/nl-fine-tuning/nl/paper/figures"
os.makedirs(OUT, exist_ok=True)
PRE, RND = "#1f77b4", "#d62728"

PANELS = {
    "1.4B": ("Pythia-1.4B", 1_414_647_808, [
        ("Pythia-1.4B, pretrained init", PRE,
         ["10579766", "10682439", "10730893", "11696279", "15232695"]),
        ("Pythia-1.4B, random init", RND,
         ["local_20260501_125955_pythia14b_step1_REINIT_lr1e4_eff768", "11427690", "11649285",
          "12969578", "13385183", "13607583", "13654529", "15233497", "11650849"])], 85),
    "qwen06b": ("Qwen3-0.6B", 596_049_408, [
        ("Qwen3-0.6B, pretrained init", PRE, ["10696449", "10730891", "11426006"]),
        ("Qwen3-0.6B, random init", RND,
         ["11427737", "11649286", "12982257", "13364120", "13542862", "13607950", "14110834",
          "15233496", "11590956"])], 260),
}


def load(dirs, N):
    f = 6 * N / 1e15
    seen, cum, prev, prevL = -1, 0, None, None
    pf, Ls = [0.0], [0]
    for d in dirs:
        p = f"{SCRATCH}/nl_output/search/job_{d}/loss_history.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            try:
                e = json.loads(line)
            except Exception:
                continue
            s = e["step"]
            if s <= seen:
                continue
            seen = s
            cum += e.get("tokens", 0)
            st, L = e.get("stage"), e.get("effective_L")
            if st is None or L is None:
                continue
            if prev is not None and st != prev:
                pf.append(cum * f); Ls.append(prevL)      # COMPLETED stage
            prev, prevL = st, L
    pf.append(cum * f); Ls.append(Ls[-1])
    return np.array(pf), np.array(Ls)


sys.path.insert(0, "/scratch/gautschi/huan2073/audit_tmp/completed")
from chain_cache import cache_wrap
load = cache_wrap(load, "reinit")

fmt = FuncFormatter(lambda v, _: "0" if v == 0 else (f"{v/1e6:g}M" if v >= 1e6 else f"{v/1e3:g}k"))
YMAX = max(v[3] for v in PANELS.values())     # shared scale: both panels compare the same quantity

# Height 1.35in instead of FULL2_80's 1.7in (2026-09-26, user: "make this graph less tall"); width unchanged.
fig, axes = plt.subplots(1, 2, figsize=(PS.FULL2_80[0], 1.35), sharey=True)
for ax, (key, (model, N, arms, _)) in zip(axes, PANELS.items()):
    data = [(name, c, *load(dirs, N)) for name, c, dirs in arms]
    CAP = min(pf[-1] for _, _, pf, _ in data)          # matched-compute cap = shorter chain's end
    for name, c, pf, Ls in data:
        m = pf <= CAP
        x = np.append(pf[m], CAP); y = np.append(Ls[m], Ls[m][-1])
        ax.plot(x, y, "-", color=c, linewidth=PS.LW, alpha=0.72, label=name, drawstyle="steps-post")
        print(f"{name}: L at cap ({CAP:,.0f} PFLOPs) = {int(y[-1])}")
    ax.set_xlim(-CAP * 0.02, CAP)
    ax.set_ylim(0, YMAX)
    ax.set_yticks(np.arange(0, YMAX + 1, 50))
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel("Cumulative Compute (PFLOPs)")
    ax.tick_params(labelleft=True)
    ax.set_title(model)
    PS.grid(ax)
    print(f"{key}: x-axis ends at {CAP:,.0f} PFLOPs")
axes[0].set_ylabel("Achieved Lookahead $L$")     # shared y-scale: label once

# One legend, inside the left panel. Both panels use the same two colours, so generic names are
# correct and the second panel needs no legend of its own. Upper left is clear there: the
# pretrained arm plateaus near L=75 against a 260 scale, so the top of that panel is empty.
axes[0].legend([Line2D([0], [0], color=PRE, lw=PS.LW), Line2D([0], [0], color=RND, lw=PS.LW)],
               ["Pretrained initialization", "Random initialization"], loc="upper left")

fig.tight_layout(pad=0.1, w_pad=1.0)
PS.save(fig, f"{OUT}/pretrained_vs_randominit_combined")
