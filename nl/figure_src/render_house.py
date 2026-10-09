"""Re-render every paper figure in a hybrid of the figures4papers house style (ChenLiu-1996/figures4papers):
no top/right spines, thick remaining spines, faint horizontal grid (HOUSE_GRID=none for no grid), frameless legends, thicker lines, Helvetica-like
sans font, 300-dpi PNG + PDF. Data, curves, titles and axis text are untouched; only styling changes.
Outputs go to paper/figures_house/ under the canonical paper/figures/ basenames."""
import os, sys, runpy, traceback, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

HOUSE_DIR = os.environ.get("HOUSE_DIR", "/home/huan2073/nl-fine-tuning/nl/paper/figures"); os.makedirs(HOUSE_DIR, exist_ok=True)
GRID = os.environ.get("HOUSE_GRID", "y")   # "y" = hybrid (faint horizontal grid), "none" = full house style
# Since 2026-09-23 every figure is drawn at its PRINTED size and included at 100% scale, so the
# style comes from print_style.py (7pt ticks/legends, 8pt labels, page-weight lines). The old
# values here (1.6pt spines, lines x1.3) assumed the figure would be shrunk to 30-45%.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import print_style as PS
HOUSE_RC = dict(PS.RC, **{"axes.grid": GRID != "none", "axes.grid.axis": "y"})
LW_SCALE = 1.0   # scripts now pass their final (printed) line widths

# --- monkeypatches (styling only) ---
_grid = Axes.grid
def _house_grid(self, *a, **k):
    if GRID == "none": return _grid(self, False)
    return _grid(self, True, axis="y", color="0.6", alpha=0.35, linewidth=0.4, linestyle="-")
Axes.grid = _house_grid
_legend = Axes.legend
def _house_legend(self, *a, **k):
    k.pop("framealpha", None); k.pop("fancybox", None); k["frameon"] = False
    return _legend(self, *a, **k)
Axes.legend = _house_legend
_flegend = Figure.legend
def _house_flegend(self, *a, **k):
    k.pop("framealpha", None); k["frameon"] = False
    return _flegend(self, *a, **k)
Figure.legend = _house_flegend
_plot = Axes.plot
def _house_plot(self, *a, **k):
    lw = k.pop("lw", None); lw = k.pop("linewidth", lw)
    if lw is not None and lw > 0: k["linewidth"] = lw * LW_SCALE
    elif lw is None and not (k.get("linestyle") == "" or k.get("ls") == ""): k["linewidth"] = PS.LW
    else: k["linewidth"] = lw if lw is not None else 0
    return _plot(self, *a, **k)
Axes.plot = _house_plot
_savefig = Figure.savefig
def _house_savefig(self, fname, *a, **k):
    if isinstance(fname, str):
        base = os.path.basename(fname).replace("_completed", "").replace("_house", "")
        fname = os.path.join(HOUSE_DIR, base)
        k.setdefault("bbox_inches", "tight"); k.setdefault("pad_inches", 0.01)
        if base.endswith(".png"): k["dpi"] = 300
        print("   ->", fname)
    return _savefig(self, fname, *a, **k)
Figure.savefig = _house_savefig

C = "/scratch/gautschi/huan2073/audit_tmp/completed"; NL = "/home/huan2073/nl-fine-tuning/nl"
SCRIPTS = [
    (f"{NL}/plot_curr_vs_nocurr.py",                          {}),   # Fig: curr vs nocurr per-L acc (+3panel)
    (f"{C}/plot_fig5_overlay_completed.py",                   {}),   # Fig 5 overlay per model
    (f"{NL}/plot_vocab_ablation.py",                          {}),   # Fig 4 lookahead_and_acc_vs_flops_kticks
    (f"{C}/plot_pythia_lmax_scaling_completed.py",            {"METRIC": "lmax"}),   # legend = name + ceiling only   # pythia_lmax_scaling
    # plot_pretrained_vs_reinit_combined.py is STANDALONE since 2026-09-14 (sets the house style itself,
    # line width already 2.6 = 2.0 x LW_SCALE). Running it here would scale its lines again. Run it
    # directly: python figure_src/plot_pretrained_vs_reinit_combined.py
    # singles superseded by merged figures (stepsize+family linear -> plot_fig7_8_house; 1.4B+qwen06b -> reinit_combined)
    # plot_fig7_8_house.py (merged step-size + family figure) is no longer in the paper and was not
    # moved to print size; the paper uses figure_src/plot_fig7_stepsize.py and plot_fig8_family.py,
    # both standalone. Re-enable only after converting it.
    # (f"{C}/plot_fig7_8_house.py", {"CMA_NAME": "viridis", "CMA_LO": "0.06", "CMA_SPAN": "0.86", "OUT_SUFFIX": "", "ALPHA_A": "0.85", "ALPHA_B": "0.85"}),
]
only = sys.argv[1:]
ok, bad = [], []
for path, env in SCRIPTS:
    name = os.path.basename(path)
    if only and not any(o in name for o in only): continue
    print(f"=== {name}"); sys.stdout.flush()
    os.environ.update(env)
    plt.rcParams.update(plt.rcParamsDefault); plt.rcParams.update(HOUSE_RC); matplotlib.use("Agg")
    try:
        runpy.run_path(path, run_name="__main__"); ok.append(name)
    except Exception:
        traceback.print_exc(); bad.append(name)
    plt.close("all")
print("\nOK:", ok); print("FAILED:", bad)
