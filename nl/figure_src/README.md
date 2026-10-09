# Generators for the paper's current figures

**Print size (2026-09-23).** Every results figure is drawn at its final printed size and meant to be
included at 100% scale, so the fonts in `print_style.py` are the fonts on the page (ticks and legends
7pt, axis labels 8pt). Include with `width=\linewidth` (Figs 2, 3, 9, 11), `width=0.8\linewidth`
(Figs 4, 7, 10) or `width=2.4in` (Figs 5, 6); each output is within 1% of that width, so the scale
stays at 100%. Do not enlarge a canvas and let LaTeX shrink it: that is what made the text print at
3.5-5pt. `render_house.py` now takes its style from `print_style.py` as well. The pre-change
renderings are kept in `paper/figures/_before_print_size_20260923/`.

These produce everything in `paper/figures/`. They lived in `/scratch/gautschi/huan2073/audit_tmp/`
while the September restyle was in progress; copied here on 2026-09-07 because scratch is not backed up.

- `render_house.py` is the entry point. It re-runs each generator under the hybrid style (no top or right
  spine, frameless legend, heavier lines, faint horizontal grid) and writes into `paper/figures/`.
  Run it with no arguments for the whole set, or pass name fragments to rebuild a subset.
- `plot_fig7_8_side_by_side.py` and `plot_fig7_8_house.py` build the merged step-size and Pythia-family
  figure. Colormaps and opacity come from the environment: CMA_NAME, CMA_LO, CMA_SPAN, ALPHA_A, ALPHA_B.
  The paper uses viridis on the left and inferno on the right at 85% opacity.
- `plot_pretrained_vs_reinit_combined.py` builds the merged pretrained versus random-init figure.
- `plot_pretrained_vs_reinit_single.py` draws the same four chains in one panel instead of two,
  with colour for initialization and linestyle for model. It writes its own output names, so both
  layouts can sit in `paper/figures` at once. It also emits two comparison variants, a shared
  1.5M cap and a log x-axis, as PNG only.
- `plot_fig5_overlay_completed.py` and `plot_pythia_lmax_scaling_completed.py` are the completed-stage
  versions of the fit-overlay and ceiling figures. METRIC=lmax gives the paper's legend.
- `chain_cache.py` caches parsed training chains keyed on log size and mtime, so a re-render takes
  seconds instead of re-reading every loss history.

The remaining generators stay in the project root: plot_curr_vs_nocurr.py, plot_vocab_ablation.py,
plot_stepsize_sweep.py (its chain list and loader are read by the merged figure script) and
plot_pythia_lmax_scaling.py.
