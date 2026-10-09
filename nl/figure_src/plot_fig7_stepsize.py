"""Paper Figure 5: Qwen3-0.6B step-size sweep, completed-stage convention.

Drawn at its printed size (half text width, print_style.HALF) and included at 100% scale, so
ticks and legend print at 7pt and axis labels at 8pt. No title: the caption says what it shows.
The published variant is
  STEP_KEEP=8,16,32,64 X_MAX=300000 X_RIGHT_MARGIN=0.03 PAD_TO_AXIS=1 CMA_NAME=inferno \
  CMA_LO=0.15 CMA_SPAN=0.65 OUT_SUFFIX=_8_16_32_64_padded
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

OUT = "/home/huan2073/nl-fine-tuning/nl/paper/figures"
os.makedirs(OUT, exist_ok=True)

# Reuse the paper script's chains and completed-stage loader, as the merged figure did.
src = open("/home/huan2073/nl-fine-tuning/nl/plot_stepsize_sweep.py").read()
ns = {}
exec(compile(src.split("kfmt=FuncFormatter")[0], "stepsize_head", "exec"), ns)
CHAINS, load_s = ns["CHAINS"], ns["load"]
CAP = ns.get("CAP", 300_000)
TARGET = ns.get("TARGET", 256)
sys.path.insert(0, "/scratch/gautschi/huan2073/audit_tmp/completed")
from chain_cache import cache_wrap
load_s = cache_wrap(load_s, "stepsize")

fmt = FuncFormatter(lambda v, _: "0" if v == 0 else (f"{v/1e6:g}M" if v >= 1e6 else f"{v/1e3:g}k"))
# Colour ramp is overridable so the figure can be matched to another one without editing code.
# Defaults reproduce the published viridis rendering. The Pythia family figure
# (plot_fig8_family.py) uses inferno over [0.15, 0.80]: CMA_NAME=inferno CMA_LO=0.15 CMA_SPAN=0.65.
# OUT_SUFFIX keeps a variant from overwriting paper/figures/stepsize_sweep.{png,pdf}.
CM = plt.get_cmap(os.environ.get("CMA_NAME", "viridis"))
CMA_LO = float(os.environ.get("CMA_LO", "0.06"))
CMA_SPAN = float(os.environ.get("CMA_SPAN", "0.86"))
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")
# STEP_KEEP="1,4,16,64" plots only those step sizes. Seven lines on one sequential ramp leave
# neighbours (s=2/s=4, s=8/s=16) indistinguishable; four log-spaced sizes keep the full range
# and every line readable. Colours are then spread over the kept sizes only.
if os.environ.get("STEP_KEEP"):
    keep = {int(x) for x in os.environ["STEP_KEEP"].split(",")}
    CHAINS = {s: v for s, v in CHAINS.items() if s in keep}
# S1_CHAIN=fit draws s=1 from the chain used for the Qwen L-vs-compute curve fit
# (plot_curve_fit_families.py / goodness_of_fit_table.py): 10696449 -> 10730891 -> 11426006.
# It is the SAME run as the default s=1 chain, branched at checkpoint 68,000; the two agree
# exactly to 300K PFLOPs. The branch continues to 1,504,595 PFLOPs and L=244, where the
# default chain stops at 413,746 and L=166. Each file carries the full history from step 1,
# so the fit chain alone reproduces the pre-branch trajectory. Same effective batch (768);
# after the branch the gate window is 800 rather than 200.
if os.environ.get("S1_CHAIN") == "fit" and 1 in CHAINS:
    CHAINS = dict(CHAINS)
    CHAINS[1] = (CHAINS[1][0], ["10696449", "10730891", "11426006"])
# CHAIN_OVERRIDE="16=qwen06b_step16_rerun" (";"-separated entries, ","-separated job dirs)
# swaps a step size's chain. DIAGNOSTIC ONLY: the *_rerun jobs use gate window 800 and warmup
# 500, the original chains use window 200 or 1000 and warmup 100, so mixing them puts different
# training configurations on one figure. Also note the s=16 rerun and original diverge after 75K
# PFLOPs (rerun stuck at L=144, original reaches 240), i.e. single runs are not reproducible.
if os.environ.get("CHAIN_OVERRIDE"):
    CHAINS = dict(CHAINS)
    for entry in os.environ["CHAIN_OVERRIDE"].split(";"):
        k, v = entry.split("=")
        if int(k) in CHAINS:
            CHAINS[int(k)] = (CHAINS[int(k)][0], v.split(","))
SVALS = sorted(CHAINS)
COL = {s: CM(CMA_LO + CMA_SPAN * i / (len(SVALS) - 1)) for i, s in enumerate(SVALS)}

# X_MAX: "full" sets the axis to the longest kept run's end (s=1, 413,746 PFLOPs); a number
# sets it explicitly; unset keeps the published 300K. Beyond 300K the other runs END between
# 282K and 333K, and the default padding would draw flat plateaus they never trained, so any
# X_MAX also switches to stopping each line at its own run end, marked with a dot.
# FIG_H sets the figure height (default 4.9, the published panel).
DATA = {s: load_s(dirs, s) for s, (col, dirs) in CHAINS.items()}
X_MAX = os.environ.get("X_MAX")
# PAD_TO_AXIS=1 keeps every line running flat to the right edge even with X_MAX set. Appropriate
# when the short runs ended by COMPLETING their curriculum (old s=8 and s=16 reached L=256, their
# n_stages ceiling), so the flat tail means "done" rather than an untrained plateau.
STOP_AT_END = bool(X_MAX) and os.environ.get("PAD_TO_AXIS") != "1"
# PAD_STEPS="32,64" extends only those step sizes flat to the right edge while every other line stops at
# its own run end (marked with a dot). For runs whose flat tail is a real plateau (s=32, s=64 had stopped
# advancing), as opposed to runs that simply ended.
PAD_STEPS = {int(x) for x in os.environ.get("PAD_STEPS", "").split(",") if x.strip()}
# Y_MAX truncates the y-axis (default 268, i.e. room for L=256). Lines above it are cut at the edge.
Y_MAX_ENV = float(os.environ.get("Y_MAX", "0"))
if X_MAX == "full":
    CAP = max(end for _, _, end in DATA.values())
elif X_MAX:
    CAP = float(X_MAX)
FIG_W = float(os.environ.get("FIG_W", PS.HALF[0]))
FIG_H = float(os.environ.get("FIG_H", PS.HALF[1]))

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
for s in CHAINS:
    pf, Ls, end_cum = DATA[s]
    cleared = int(Ls.max())
    if cleared + s >= TARGET and cleared < TARGET:
        pf = np.append(pf, end_cum); Ls = np.append(Ls, TARGET)
    stop = min(CAP, end_cum) if (STOP_AT_END and s not in PAD_STEPS) else CAP
    m = pf <= stop
    pfc, Lsc = pf[m], Ls[m]
    if pfc[-1] < stop:
        pfc = np.append(pfc, stop); Lsc = np.append(Lsc, Lsc[-1])
    # solid_capstyle="butt": matplotlib's default "projecting" cap extends each line by half its
    # width past the last data point (~6 px at 300 dpi), which reads as running past the final tick.
    # LINE_ALPHA (default 0.85): lower values make overlapping lines and crossovers easier to see.
    ax.plot(pfc, Lsc, "-", color=COL[s], lw=PS.LW, alpha=float(os.environ.get("LINE_ALPHA", "0.85")), label=f"$s={s}$",
            solid_capstyle="butt",
            drawstyle="steps-post")
    if STOP_AT_END and stop < CAP and os.environ.get("END_DOTS", "1") != "0":   # END_DOTS=0: no end markers
        # With Y_MAX, a dot sitting exactly on the truncated top edge is drawn unclipped so it stays whole;
        # dots above the edge (lines cut off by the truncation) stay clipped.
        ax.plot([stop], [Lsc[-1]], "o", color=COL[s], ms=PS.MS, alpha=0.95, zorder=5,
                clip_on=not (Y_MAX_ENV and Lsc[-1] <= Y_MAX_ENV))

# LOG_X=1 puts compute on a log axis, for when one run is far longer than the rest (s=1 on the
# fit chain runs 4.5x past the others). Otherwise linear, with a coarser tick step past 500K.
ax.set_ylim(0, Y_MAX_ENV or 268)
if os.environ.get("LOG_X") == "1":
    ax.set_xscale("log")
    ax.set_xlim(float(os.environ.get("X_MIN", "2000")), CAP)
    ax.set_xticks([t for t in (1e4, 3e4, 1e5, 3e5, 1e6) if t <= CAP])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
else:
    # X_RIGHT_MARGIN (fraction, default 0) leaves blank space past CAP. With no right spine, lines
    # that end exactly at the axis edge read as running off the plot; a small margin makes them
    # visibly stop on the last tick. Data still stops at CAP; only the view is wider.
    ax.set_xlim(0, CAP * (1 + float(os.environ.get("X_RIGHT_MARGIN", "0"))))
    ax.set_xticks(np.arange(0, CAP + 1, 100_000 if CAP <= 500_000 else 250_000))
ax.xaxis.set_major_formatter(fmt)
if os.environ.get("LOG_X") == "1":
    # Decade ticks from X_MIN up; below 1k plain numbers (fmt would print "0.1k").
    _x0 = float(os.environ.get("X_MIN", "2000"))
    ax.set_xticks([t for t in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6) if _x0 <= t <= CAP])
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v/1e6:g}M" if v >= 1e6 else (f"{v/1e3:g}k" if v >= 1e3 else f"{v:g}")))
if os.environ.get("LOG_Y") == "1":
    # LOG_Y=1: log lookahead axis from Y_MIN (default 1, the first s=1 stage) to Y_MAX (or 268), with
    # power-of-two ticks and a tick at a truncated cap. Points at L=0 are not drawable and drop out.
    ax.set_yscale("log")
    _ytop = Y_MAX_ENV or 268
    ax.set_ylim(float(os.environ.get("Y_MIN", "1")), _ytop)
    _yt = [2 ** k for k in range(0, 9) if float(os.environ.get("Y_MIN", "1")) <= 2 ** k <= _ytop]
    if Y_MAX_ENV and _yt[-1] != int(Y_MAX_ENV):
        _yt.append(int(Y_MAX_ENV))
    ax.set_yticks(_yt)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
elif Y_MAX_ENV:   # truncated axis: 64-step ticks below the cap plus a tick at the cap itself
    _yt = list(range(0, int(Y_MAX_ENV) + 1, 64))
    ax.set_yticks(_yt + ([int(Y_MAX_ENV)] if _yt[-1] != int(Y_MAX_ENV) else []))
else:
    ax.set_yticks(np.arange(0, 257, 64))
ax.set_xlabel("Cumulative Compute (PFLOPs)")
ax.set_ylabel("Achieved Lookahead $L$")
# Hybrid house style: faint grid on BOTH axes (the published figure carries vertical gridlines
# at the x ticks as well as horizontal ones).
PS.grid(ax)
# LEGEND_LOC overrides the corner. On a linear axis to 1.5M the short runs crowd the upper left,
# so that variant uses "lower right", which only the long s=1 tail approaches.
ax.legend(loc=os.environ.get("LEGEND_LOC", "upper left"), ncol=2)
fig.tight_layout(pad=0.1)
PS.save(fig, f"{OUT}/stepsize_sweep{OUT_SUFFIX}")
