"""Paper Figure 6: Pythia family scaling at s=1, completed-stage convention.

Drawn at its printed size (half text width, print_style.HALF) and included at 100% scale, so
ticks and legend print at 7pt and axis labels at 8pt. No title: the caption says what it shows.
The published variant is written with OUT_SUFFIX=_short.

The colours come from the inferno colormap over [0.15, 0.80], matching the step-size figure.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, "/home/huan2073/nl-fine-tuning/nl/figure_src")
import print_style as PS
PS.apply()

SCRATCH = "/scratch/gautschi/huan2073"
OUT = "/home/huan2073/nl-fine-tuning/nl/paper/figures"
os.makedirs(OUT, exist_ok=True)
CAP = 1_000_000

RUNS = {
    "Pythia-160M": (162_322_944, ["jackie_pythia160m_step1_lr1e4_resumed", "12587988", "12854034",
                                  "12969577", "13364117", "13542861", "13606486", "13930099",
                                  "14249842", "11590957"]),
    "Pythia-410M": (405_334_016, ["10696605", "10730890", "11114263", "11426007", "12494239",
                                  "13930098", "14110833"]),
    "Pythia-1.4B": (1_414_647_808, ["10579766", "10682439", "10730893", "11696279", "15232695",
                                    "11650850"]),
    "Pythia-2.8B": (2_775_208_960, ["10580483", "10694584", "10730892", "11426089", "11647896"]),
}


def load_f(dirs, N):
    """Completed stage carried forward at every logged step."""
    factor = 6 * N / 1e15
    seen, cum, prev, prevL, done = -1, 0, None, None, 0
    pf, Ls = [], []
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
            L, st = e.get("effective_L"), e.get("stage")
            if L is None or st is None:
                continue
            if prev is not None and st != prev:
                done = prevL
            prev, prevL = st, L
            pf.append(cum * factor); Ls.append(done)
    return np.array(pf), np.array(Ls)


sys.path.insert(0, "/scratch/gautschi/huan2073/audit_tmp/completed")
from chain_cache import cache_wrap
load_f = cache_wrap(load_f, "family")

fmt = FuncFormatter(lambda v, _: "0" if v == 0 else (f"{v/1e6:g}M" if v >= 1e6 else f"{v/1e3:g}k"))
CM = plt.get_cmap("inferno")
NAMES = list(RUNS)
COL = {n: CM(0.15 + 0.65 * i / (len(NAMES) - 1)) for i, n in enumerate(NAMES)}

# FIG_W / FIG_H override the printed size (default print_style.HALF, same as the step-size
# figure). OUT_SUFFIX keeps a variant from overwriting paper/figures/pythia_family_scaling.*.
FIG_W = float(os.environ.get("FIG_W", PS.HALF[0]))
FIG_H = float(os.environ.get("FIG_H", PS.HALF[1]))
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
for name, (N, dirs) in RUNS.items():
    pf, Ls = load_f(dirs, N)
    m = (pf <= CAP) & (Ls >= 1)
    pf, Ls = pf[m], Ls[m]
    if len(pf) > 4000:
        idx = np.linspace(0, len(pf) - 1, 4000).astype(int)
        pf, Ls = pf[idx], Ls[idx]
    # Hold the last completed stage out to the cap: Pythia-160M stops at 995,723 PFLOPs with stage
    # 10 in progress, having already spent ~74K PFLOPs on it, so the remaining ~4K cannot change
    # the level either.
    if pf[-1] < CAP:
        pf = np.append(pf, CAP); Ls = np.append(Ls, Ls[-1])
    ax.plot(pf, Ls, "-", color=COL[name], lw=PS.LW, alpha=0.85, label=name.replace("Pythia-", ""),
            drawstyle="steps-post")
    print(f"{name}: L at end/cap = {int(Ls[-1])} at {pf[-1]:,.0f}")

ax.set_xlim(-CAP * 0.02, CAP)
ax.set_ylim(0, 112)     # headroom above 100 keeps the legend clear of the 2.8B curve
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_xticks(np.arange(0, CAP + 1, 250_000))
ax.xaxis.set_major_formatter(fmt)
ax.set_xlabel("Cumulative Compute (PFLOPs)")
ax.set_ylabel("Achieved Lookahead $L$")
# Hybrid house style: faint grid on BOTH axes (the published figure carries vertical gridlines
# at the x ticks as well as horizontal ones).
PS.grid(ax)
# Short labels under a "Pythia" title keep the legend narrow enough to sit above the curves at
# half width; two columns keep it short.
ax.legend(loc="upper left", title="Pythia", ncol=2, alignment="left")
fig.tight_layout(pad=0.1)
PS.save(fig, f"{OUT}/pythia_family_scaling{OUT_SUFFIX}")
