"""Print-size style shared by every results figure in the paper (2026-09-23).

Until now each figure was drawn on a large canvas (5.3in to 20in wide) and shrunk by LaTeX to its
slot, so a 10pt label printed at about 4.5pt. Every figure is now drawn at its FINAL printed size
and included at 100% scale, so the sizes below are the sizes on the page: ticks and legends 7pt,
axis labels 8pt, against the paper's 9pt captions and 10pt body text.

Line weights are chosen for the page too. The old house style (spines 1.6pt, data lines 2.6 to
3.1pt) only looked right because it was shrunk to 30-45%; at 100% it would print twice as heavy.
The values here reproduce what those figures actually printed at: data lines about 1.2pt, spines
0.7pt, ticks 2.5pt long.

ICLR text width is 5.5in. Slots:
  HALF       (2.4, 1.6)  half-width wrapfigure (step size, Pythia family)
  FULL2      (5.5, 1.9)  full width, two panels (attribute-vocabulary ablation)
  FULL2_80   (4.4, 1.7)  two panels at 0.8 text width (fit overlay, initialization)
  FULL3      (5.5, 1.8)  full width, three panels (curriculum vs. none, body)
Titles that only restate the caption are dropped. Multi-panel figures keep a one-line panel label
(the model or the target L), because the caption refers to the panels by position.
"""
import matplotlib.pyplot as plt

HALF, FULL2, FULL2_80, FULL3 = (2.4, 1.6), (5.5, 1.9), (4.4, 1.7), (5.5, 1.8)

LW = 1.2        # data lines
LW_THIN = 0.9   # reference lines (thresholds, fitted curves drawn over data)
MS = 3.0        # marker size

RC = {
    "font.family": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans", "sans-serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.title_fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "axes.labelpad": 2.0, "axes.titlepad": 3.0, "axes.axisbelow": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "lines.linewidth": LW, "lines.markersize": MS,
    "grid.color": "0.6", "grid.alpha": 0.35, "grid.linewidth": 0.4,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.3, "legend.labelspacing": 0.25, "legend.borderpad": 0.3,
    "legend.columnspacing": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
}


def apply():
    plt.rcParams.update(RC)


def _plain(v, _):
    """1, 10, 100, 1k, 10k, 100k, 1M: log-axis tick labels without superscripts (a mathtext
    exponent prints at 70% of the tick size, i.e. 4.9pt), matching the k/M ticks elsewhere."""
    if v >= 1e6: return f"{v / 1e6:g}M"
    if v >= 1e3: return f"{v / 1e3:g}k"
    return f"{v:g}"


def plain_log_ticks(ax, axes="xy"):
    from matplotlib.ticker import FuncFormatter, NullFormatter
    for a in axes:
        axis = ax.xaxis if a == "x" else ax.yaxis
        axis.set_major_formatter(FuncFormatter(_plain))
        axis.set_minor_formatter(NullFormatter())


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color="0.6", alpha=0.35, linewidth=0.4, linestyle="-")


def save(fig, path_noext):
    """Write <path>.png (300 dpi) and <path>.pdf, cropped to content with a 0.01in pad."""
    for ext in ("png", "pdf"):
        p = f"{path_noext}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.01, **({"dpi": 300} if ext == "png" else {}))
        print("saved", p)
