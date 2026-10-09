#!/usr/bin/env python3
"""Pretrained versus random initialization, all four chains in ONE panel.

Single-panel alternative to plot_pretrained_vs_reinit_combined.py, which draws the same data as
two side-by-side panels. That file is unchanged and still writes
paper/figures/pretrained_vs_randominit_combined.{png,pdf}; this one adds a separate set of
outputs, so both layouts are available and neither overwrites the other.

Encoding: colour is the initialization (blue pretrained, red random), linestyle is the model
(solid Pythia-1.4B, dashed Qwen3-0.6B). The colours therefore keep the meaning they carry in the
two-panel figure, and the claim the figure makes reads off the colours alone.

Same data, same loader and same completed-stage convention as the two-panel script. Shared y-scale
0..260, steps-post at 72% opacity, k/M tick labels, each model drawn to its own matched-compute cap
(the shorter of its two chains). The legend sits centre-right, which is empty in this layout.

READ THE CLIPPING BEFORE QUOTING ANY ENDPOINT. The matched-compute cap hides a large part of both
random-init chains, because within each model pair the longer chain is cut at the shorter one's
end. Raw chain ends against plotted ends:
    Pythia-1.4B pretrained   5,000,829 PFLOPs  L= 75    sets the Pythia cap
    Pythia-1.4B random       6,021,679 PFLOPs  L= 36    plotted to 5.0M, shown at L=33
    Qwen3-0.6B  pretrained   1,504,595 PFLOPs  L=245    sets the Qwen cap, no continuation exists
    Qwen3-0.6B  random       4,815,706 PFLOPs  L= 62    plotted to 1.5M, shown at L=35
So the apparent result that both random-init arms flatten in the same band just above L=30 is an
artifact of the clip. Run far enough, Qwen random reaches 62 against Pythia random's 36, which is
the opposite reading: the smaller model is ahead per FLOP from random init too. Any claim about
random-init ceilings belongs to the full-extent variant, not to the matched-cap one. See also
memory project_reinit_ceiling.md, which holds the ceiling confidence intervals.

Four outputs:
    pretrained_vs_randominit_single.{png,pdf}            recommended, linear x, matched caps
    pretrained_vs_randominit_single_alt_sharedcap.png    both models truncated at 1.5M
    pretrained_vs_randominit_single_alt_logx.png         log x
    pretrained_vs_randominit_single_alt_fullextent.png   every chain to its own true end, no clip
The three alternatives are PNG only and are for comparison, not for the manuscript.

Standalone, like its two-panel sibling: it sets the hybrid house style itself rather than relying
on render_house.py's monkeypatching. Line width 2.6 is the old 2.0 times that renderer's 1.3 scale.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans", "sans-serif"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.6,
    "legend.frameon": False, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
    "xtick.major.size": 5, "ytick.major.size": 5, "pdf.fonttype": 42,
    "axes.axisbelow": True,
})

SCRATCH = "/scratch/gautschi/huan2073"
OUT = "/home/huan2073/nl-fine-tuning/nl/paper/figures"
os.makedirs(OUT, exist_ok=True)
PRE, RND = "#1f77b4", "#d62728"

# (label, colour, linestyle, params, job chain).  Identical chains to the two-panel script.
ARMS = [
    ("Pythia-1.4B, pretrained init", PRE, "-", 1_414_647_808,
     ["10579766", "10682439", "10730893", "11696279", "15232695"]),
    ("Pythia-1.4B, random init", RND, "-", 1_414_647_808,
     ["local_20260501_125955_pythia14b_step1_REINIT_lr1e4_eff768", "11427690", "11649285",
      "12969578", "13385183", "13607583", "13654529", "15233497", "11650849"]),
    ("Qwen3-0.6B, pretrained init", PRE, "--", 596_049_408,
     ["10696449", "10730891", "11426006"]),
    ("Qwen3-0.6B, random init", RND, "--", 596_049_408,
     ["11427737", "11649286", "12982257", "13364120", "13542862", "13607950", "14110834",
      "15233496", "11590956"]),
]


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

DATA = [(name, c, ls, *load(dirs, N)) for name, c, ls, N, dirs in ARMS]
CAPS = {fam: min(pf[-1] for n, _, _, pf, _ in DATA if n.startswith(fam))
        for fam in ("Pythia", "Qwen")}
print(f"caps: Pythia {CAPS['Pythia']:,.0f} PFLOPs   Qwen {CAPS['Qwen']:,.0f} PFLOPs")
for name, _, _, pf, Ls in DATA:
    own = CAPS["Pythia" if name.startswith("Pythia") else "Qwen"]
    print(f"  {name:32s} L={int(Ls[pf <= own][-1]):3d} at own cap, "
          f"L={int(Ls[pf <= min(CAPS.values())][-1]):3d} at the shared {min(CAPS.values())/1e6:.1f}M cap")

fmt = FuncFormatter(lambda v, _: "0" if v == 0 else (f"{v/1e6:g}M" if v >= 1e6 else f"{v/1e3:g}k"))
YMAX = 260                       # shared with the two-panel figure so the two are comparable
HANDLES = [Line2D([0], [0], color=PRE, lw=2.6), Line2D([0], [0], color=RND, lw=2.6),
           Line2D([0], [0], color="0.35", lw=2.6, ls="-"),
           Line2D([0], [0], color="0.35", lw=2.6, ls="--")]
LABELS = ["Pretrained initialization", "Random initialization", "Pythia-1.4B", "Qwen3-0.6B"]


RAW_END = max(pf[-1] for _, _, _, pf, _ in DATA)


def render(cap_mode, logx, loc, stem, pdf):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for name, c, ls, pf, Ls in DATA:
        # "full" stops each curve at ITS OWN last logged point, so no arm is padded with a
        # flat tail it did not train. "own" and "shared" clip to a cap shared by the pair.
        cap = (min(CAPS.values()) if cap_mode == "shared"
               else pf[-1] if cap_mode == "full"
               else CAPS["Pythia" if name.startswith("Pythia") else "Qwen"])
        m = pf <= cap
        x = np.append(pf[m], cap); y = np.append(Ls[m], Ls[m][-1])
        if logx:
            k = x > 0
            x, y = x[k], y[k]
        ax.plot(x, y, ls, color=c, linewidth=2.6, alpha=0.72, drawstyle="steps-post")
    hi = (min(CAPS.values()) if cap_mode == "shared"
          else RAW_END if cap_mode == "full" else max(CAPS.values()))
    if logx:
        ax.set_xscale("log"); ax.set_xlim(3e3, hi)
    else:
        ax.set_xlim(-hi * 0.02, hi)
    ax.set_ylim(0, YMAX)
    ax.set_yticks(np.arange(0, YMAX + 1, 50))
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel("Cumulative Compute (PFLOPs)", fontsize=11)
    ax.set_ylabel("Achieved Lookahead $L$", fontsize=11)
    ax.tick_params(labelsize=9.5)
    ax.set_title("Achieved Lookahead vs Cumulative Compute ($s{=}1$)", fontsize=11)
    ax.grid(True, axis="both", color="0.6", alpha=0.35, linewidth=0.8, linestyle="-")
    ax.legend(HANDLES, LABELS, loc=loc, fontsize=9.5, frameon=False,
              handlelength=2.2, borderaxespad=0.8, labelspacing=0.4)
    fig.tight_layout()
    for ext in (("png", "pdf") if pdf else ("png",)):
        p = f"{OUT}/{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", **({"dpi": 300} if ext == "png" else {}))
        print("saved", p)
    plt.close(fig)


render("own",    False, "center right", "pretrained_vs_randominit_single", True)
render("shared", False, "upper left",   "pretrained_vs_randominit_single_alt_sharedcap", False)
render("own",    True,  "upper left",   "pretrained_vs_randominit_single_alt_logx", False)
render("full",   False, "center right", "pretrained_vs_randominit_single_alt_fullextent", False)
