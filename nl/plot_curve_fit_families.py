#!/usr/bin/env python3
"""Generate per-family curve-fit plots (Qwen 0.6B vs Pythia 1.4B) in the attached
power-law-fit style: log-log axes, raw lines + dotted fit, simple legend.
Computes R^2 per family per run and prints best-fit summary.

Families:
- Power-law:                L = a*C^b + c
- Saturating exp (Weibull): L = Lmax*(1 - exp(-(C/C0)^k))
- BNSL n=1 (Caballero 2022): L = a + b*C^c0 * (1 + (C/d1)^(1/f1))^(-c1*f1)
- M4 (Alabdulmohsin 2022):  (L - L_inf) / (L0 - L)^alpha = beta*C^c   [implicit; numerically inverted]
- Saturating power:         L = Lmax*(1 - a*(C+b)^c)
- Logarithmic:              L = a*log(b*C) + c
"""
import json, os, numpy as np
from scipy.optimize import curve_fit, differential_evolution, brentq
import matplotlib.pyplot as plt

OUT_DIR = "/home/huan2073/nl-fine-tuning/nl/plots/curve_fits"
os.makedirs(OUT_DIR, exist_ok=True)

N_PARAMS = {"qwen06b": 596_049_408, "pythia14b": 1_414_647_808}

RUNS = {
    "Qwen 0.6B": {
        "model": "qwen06b",
        "color_raw": "#EF5350",
        "color_fit": "#1A237E",
        "dirs": [
            "/scratch/gautschi/huan2073/nl_output/search/job_10579765",
        ],
    },
    "Pythia 1.4B": {
        "model": "pythia14b",
        "color_raw": "#42A5F5",
        "color_fit": "#1A237E",
        "dirs": [
            "/scratch/gautschi/huan2073/nl_output/search/job_10579766",
        ],
    },
}


def load_stage_transitions(dirs, n_params):
    """Return (pflops, L) at every stage transition starting AT the first stage
    completion (i.e., skip the still-in-stage-1 initial point). The first kept
    point is when L first changes from L=1 to L=2."""
    factor = 6 * n_params / 1e15
    seen = -1; cum_tok = 0
    pflops_all = []; Ls_all = []
    prev_stage = None
    for d in dirs:
        path = os.path.join(d, "loss_history.jsonl")
        if not os.path.exists(path): continue
        for line in open(path):
            try: e = json.loads(line)
            except: continue
            s = e["step"]
            if s <= seen: continue
            seen = s
            cum_tok += e.get("tokens", 0)
            stage = e.get("stage"); L = e.get("effective_L")
            if stage is None or L is None: continue
            if prev_stage is not None and stage != prev_stage:
                pflops_all.append(cum_tok * factor); Ls_all.append(L)
            if prev_stage is None:
                prev_stage = stage
            elif stage != prev_stage:
                prev_stage = stage
    return np.array(pflops_all), np.array(Ls_all)


# ---- Function families ----
def f_powerlaw(x, a, b, c):
    return a * np.power(np.maximum(x, 1e-3), b) + c


def f_satexp(x, Lmax, x0, k):
    """Saturating exponential (Weibull): Lmax*(1 - exp(-(x/x0)^k))."""
    return Lmax * (1.0 - np.exp(-np.power(np.maximum(x, 0) / x0, k)))


def f_satpower(x, Lmax, a, b, c):
    """Saturating power: Lmax*(1 - a*(x+b)^c)  with c<0."""
    return Lmax * (1.0 - a * np.power(np.maximum(x + b, 1e-6), c))


def f_log(x, a, b, c):
    """Logarithmic: a*log(b*x) + c."""
    return a * np.log(np.maximum(b * x, 1e-9)) + c


def f_bnsl1(x, a, b, c0, d1, f1, c1):
    """BNSL with n=1 break (Caballero et al. 2022):
    y = a + b * x^c0 * (1 + (x/d1)^(1/f1))^(-c1*f1)
    """
    xx = np.maximum(x, 1e-9)
    base = b * np.power(xx, c0)
    bend = np.power(1.0 + np.power(xx / d1, 1.0 / f1), -c1 * f1)
    return a + base * bend


def f_m4_scalar(C, L_inf, L0, alpha, beta, c):
    """Solve (L - L_inf) / (L0 - L)^alpha = beta*C^c for L (one-shot brentq invert)."""
    if not (L_inf < L0):
        return np.nan
    rhs = beta * np.power(max(C, 1e-12), c)
    if not np.isfinite(rhs) or rhs <= 0:
        return np.nan
    eps = 1e-9 * (L0 - L_inf)
    lo = L_inf + eps
    hi = L0 - eps
    def g(L):
        return (L - L_inf) - rhs * np.power(max(L0 - L, 1e-30), alpha)
    try:
        return brentq(g, lo, hi, maxiter=80, xtol=1e-7)
    except (ValueError, RuntimeError):
        return np.nan


def f_m4(x, L_inf, L0, alpha, beta, c):
    """Vectorized inversion: only used after fit (for plotting + R^2)."""
    if np.isscalar(x):
        return f_m4_scalar(x, L_inf, L0, alpha, beta, c)
    out = np.empty(np.shape(x), dtype=float)
    for i, xi in enumerate(np.atleast_1d(x)):
        out.flat[i] = f_m4_scalar(float(xi), L_inf, L0, alpha, beta, c)
    return out


def m4_implicit_residual(L_inf, L0, alpha, beta, c, x, y):
    """Fast implicit residual in (L - L_inf) units; minimised during fitting."""
    if not (L_inf < L0): return np.full_like(y, 1e15, dtype=float)
    bad = (y <= L_inf) | (y >= L0)
    if bad.all(): return np.full_like(y, 1e15, dtype=float)
    gap = np.maximum(L0 - y, 1e-30)
    res = (y - L_inf) - beta * np.power(np.maximum(x, 1e-12), c) * np.power(gap, alpha)
    res[bad] = 1e15
    return res


def best_fit_m4(ranges, x, y, n_starts=100):
    """Two-phase M4 fit: implicit-residual DE+multistart for speed, then
    invert once at the data points to compute true L-space RSS and R^2."""
    bounds_lo = [r[0] for r in ranges]
    bounds_hi = [r[1] for r in ranges]
    bounds = list(zip(bounds_lo, bounds_hi))
    best_imp = None
    def implicit_obj(params):
        try:
            r = m4_implicit_residual(*params, x, y)
            if not np.all(np.isfinite(r)): return 1e30
            return float(np.sum(r ** 2))
        except Exception:
            return 1e30
    try:
        de = differential_evolution(implicit_obj, bounds, seed=42, maxiter=200, tol=1e-9,
                                    popsize=30, workers=1, polish=True, init="sobol")
        if np.isfinite(de.fun): best_imp = de.x
    except Exception:
        pass
    for _ in range(n_starts):
        p0 = sample_p0(ranges)
        v = implicit_obj(p0)
        if best_imp is None or (np.isfinite(v) and v < implicit_obj(best_imp)):
            # Local refine
            try:
                from scipy.optimize import minimize
                m = minimize(implicit_obj, p0, method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-7})
                if np.isfinite(m.fun) and (best_imp is None or m.fun < implicit_obj(best_imp)):
                    best_imp = m.x
            except Exception:
                pass
    if best_imp is None:
        return None
    # Phase 2: short L-space refinement seeded by implicit-best, to align with
    # the L-space RSS that other families optimise directly.
    def l_space_obj(params):
        try:
            yh = f_m4(x, *params)
            if not np.all(np.isfinite(yh)): return 1e30
            return float(np.sum((yh - y) ** 2))
        except Exception:
            return 1e30
    best_popt = best_imp
    try:
        from scipy.optimize import minimize
        m = minimize(l_space_obj, best_imp, method="Nelder-Mead",
                     options={"maxiter": 800, "xatol": 1e-6, "fatol": 1e-6})
        if np.isfinite(m.fun) and m.fun < l_space_obj(best_imp):
            best_popt = m.x
    except Exception:
        pass
    L_inf, L0, alpha, beta, c = best_popt
    yhat = f_m4(x, L_inf, L0, alpha, beta, c)
    if not np.all(np.isfinite(yhat)):
        yhat = np.where(np.isfinite(yhat), yhat, L0 - 1e-6)
    rss = float(np.sum((yhat - y) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else 0.0
    return {"popt": best_popt, "rss": rss, "r2": r2}


# (slug, display, func, k_params, ranges)
FAMILIES = [
    ("power_law",   "Power-law",
     f_powerlaw,   3, [(0.01, 100, "log"), (0.05, 1.5, "lin"), (0, 50, "lin")]),
    ("sat_exp",     "Saturating exponential (Weibull)",
     f_satexp,     3, [(50, 5000, "log"), (10, 1e7, "log"), (0.1, 3, "lin")]),
    ("bnsl1",       "BNSL (n=1 break)",
     f_bnsl1,      6, [(-100, 1000, "lin"), (-100, 100, "lin"), (-2, 2, "lin"),
                       (1e2, 1e10, "log"), (0.1, 50, "log"), (-2, 2, "lin")]),
    ("m4",          "M4 estimator",
     f_m4,         5, [(0.0, 5, "lin"), (50, 5000, "log"), (0.1, 3, "lin"),
                       (1e-9, 1e-1, "log"), (0.05, 2, "lin")]),
    ("sat_power",   "Saturating power",
     f_satpower,   4, [(50, 5000, "log"), (0.1, 2, "lin"),
                       (0.1, 1000, "log"), (-3, -0.05, "lin")]),
    ("logarithmic", "Logarithmic",
     f_log,        3, [(0.1, 1000, "log"), (1e-9, 1e3, "log"), (-1000, 1000, "lin")]),
]

N_RANDOM_P0 = 100
RNG = np.random.default_rng(42)


def sample_p0(ranges):
    out = []
    for lo, hi, kind in ranges:
        if kind == "log":
            out.append(float(np.exp(RNG.uniform(np.log(lo), np.log(hi)))))
        else:
            out.append(float(RNG.uniform(lo, hi)))
    return out


def best_fit(func, ranges, x, y, n_starts=N_RANDOM_P0):
    """Differential evolution + multistart Levenberg-Marquardt; lowest-RSS wins."""
    best = None
    bounds_lo = [r[0] for r in ranges]
    bounds_hi = [r[1] for r in ranges]
    bounds = list(zip(bounds_lo, bounds_hi))

    def rss_obj(params):
        try:
            yh = func(x, *params)
            if not np.all(np.isfinite(yh)): return 1e30
            return float(np.sum((yh - y) ** 2))
        except Exception:
            return 1e30
    cf_bounds = (bounds_lo, bounds_hi)  # pass to curve_fit to keep refinements in-range
    try:
        de = differential_evolution(rss_obj, bounds, seed=42, maxiter=200, tol=1e-9,
                                    popsize=30, workers=1, polish=False, init="sobol")
        if np.isfinite(de.fun):
            try:
                popt, _ = curve_fit(func, x, y, p0=list(de.x), bounds=cf_bounds, maxfev=30000)
                yh = func(x, *popt)
                if np.all(np.isfinite(yh)):
                    rss = float(np.sum((yh - y) ** 2))
                    tss = float(np.sum((y - y.mean()) ** 2))
                    best = {"popt": popt, "rss": rss, "r2": 1 - rss/tss}
            except Exception:
                rss = de.fun
                tss = float(np.sum((y - y.mean()) ** 2))
                best = {"popt": de.x, "rss": rss, "r2": 1 - rss/tss}
    except Exception:
        pass

    for _ in range(n_starts):
        p0 = sample_p0(ranges)
        try:
            popt, _ = curve_fit(func, x, y, p0=p0, bounds=cf_bounds, maxfev=30000)
            yhat = func(x, *popt)
            if not np.all(np.isfinite(yhat)): continue
            rss = float(np.sum((yhat - y) ** 2))
            tss = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - rss / tss if tss > 0 else 0.0
            if best is None or rss < best["rss"]:
                best = {"popt": popt, "rss": rss, "r2": r2}
        except Exception:
            continue
    return best


# Pre-load data
DATA = {}
for run_name, run in RUNS.items():
    pf, Ls = load_stage_transitions(run["dirs"], N_PARAMS[run["model"]])
    mask = (pf > 1.0) & (Ls >= 2)
    DATA[run_name] = (pf[mask], Ls[mask])
    print(f"[DATA] {run_name}: n={mask.sum()} points, L range [{Ls[mask].min():.0f}, {Ls[mask].max():.0f}], C range [{pf[mask].min():.1e}, {pf[mask].max():.1e}]")

# Fit all families
results = {}
for slug, display_name, func, k_params, ranges in FAMILIES:
    print(f"\n[FIT] {display_name}...")
    results[slug] = {}
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for run_name, run in RUNS.items():
        x, y = DATA[run_name]
        ax.plot(x, y, "-", color=run["color_raw"], linewidth=2.5, alpha=0.9, label=run_name)
        if slug == "m4":
            b = best_fit_m4(ranges, x, y)
        else:
            b = best_fit(func, ranges, x, y)
        results[slug][run_name] = b
        if b is not None:
            x_fit = np.geomspace(x.min(), x.max(), 400)
            y_fit = func(x_fit, *b["popt"])
            ax.plot(x_fit, y_fit, ":", color=run["color_fit"], linewidth=1.8, alpha=0.95)
            # Eval-only metric: compute R^2 / RMSE on L > 10 subset
            eval_mask = y > 10
            if eval_mask.sum() > 1:
                yh_eval = func(x[eval_mask], *b["popt"])
                if np.all(np.isfinite(yh_eval)):
                    y_eval = y[eval_mask]
                    rss_eval = float(np.sum((yh_eval - y_eval) ** 2))
                    tss_eval = float(np.sum((y_eval - y_eval.mean()) ** 2))
                    r2_eval = 1 - rss_eval/tss_eval if tss_eval > 0 else 0.0
                    rmse_eval = float(np.sqrt(rss_eval / eval_mask.sum()))
                    b["r2_eval_L_gt_10"] = r2_eval
                    b["rmse_eval_L_gt_10"] = rmse_eval
                    b["n_eval"] = int(eval_mask.sum())
                    print(f"  {run_name}: R²(all)={b['r2']:.5f}  R²(L>10)={r2_eval:.5f}  RMSE(L>10)={rmse_eval:.3f}  n(L>10)={int(eval_mask.sum())}")
                else:
                    print(f"  {run_name}: R²(all)={b['r2']:.5f}  (eval-subset has non-finite preds)")
            else:
                print(f"  {run_name}: R²={b['r2']:.5f}  (no L>10 data for eval)")
        else:
            print(f"  {run_name}: FIT FAILED")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Cumulative Compute (6N PFLOPs, log scale)", fontsize=13)
    ax.set_ylabel("Achieved Lookahead L (log scale)", fontsize=13)
    ax.set_title(f"{display_name} fit", fontsize=15)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"qwen06b_vs_pythia14b_{slug}_loglog_flops.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")

# Summary (R^2 over ALL data and over L>10 eval subset)
print("\n=== R^2 per family (fit on L>=2; eval-only column = R^2 over L>10 subset) ===")
print(f"{'family':<36s} {'Qwen all':>9s} {'Qwen L>10':>10s} {'Pyth all':>9s} {'Pyth L>10':>10s} {'mean L>10':>10s}")
print("-" * 95)
ranked = []
for slug, display_name, _, k_params, _ in FAMILIES:
    q = results[slug].get("Qwen 0.6B")
    p = results[slug].get("Pythia 1.4B")
    qr2 = q["r2"] if q else float("nan")
    pr2 = p["r2"] if p else float("nan")
    qr2e = q.get("r2_eval_L_gt_10", float("nan")) if q else float("nan")
    pr2e = p.get("r2_eval_L_gt_10", float("nan")) if p else float("nan")
    mean_r2 = (qr2 + pr2) / 2
    mean_r2e = (qr2e + pr2e) / 2
    nq = len(DATA["Qwen 0.6B"][1]); npy = len(DATA["Pythia 1.4B"][1])
    q_aic = (2*k_params + nq * np.log(q["rss"]/nq)) if q else float("inf")
    p_aic = (2*k_params + npy * np.log(p["rss"]/npy)) if p else float("inf")
    ranked.append((display_name, slug, qr2, pr2, mean_r2, q_aic, p_aic, q_aic+p_aic, qr2e, pr2e, mean_r2e))
    print(f"{display_name:<36s} {qr2:>9.5f} {qr2e:>10.5f} {pr2:>9.5f} {pr2e:>10.5f} {mean_r2e:>10.5f}")

ranked.sort(key=lambda r: -r[4])
print("\n=== Best family by mean R^2 (all data) ===")
for i, row in enumerate(ranked[:3]):
    name, _, qr2, pr2, mean_r2 = row[0], row[1], row[2], row[3], row[4]
    marker = "*" if i == 0 else " "
    print(f"  {marker} {name:<36s} mean R^2 = {mean_r2:.5f}  (Qwen={qr2:.4f}, Pythia={pr2:.4f})")

ranked.sort(key=lambda r: -r[10])
print("\n=== Best family by mean R^2 on L>10 eval subset ===")
for i, row in enumerate(ranked[:6]):
    name, qr2e, pr2e, mean_r2e = row[0], row[8], row[9], row[10]
    marker = "*" if i == 0 else " "
    print(f"  {marker} {name:<36s} mean R^2(L>10) = {mean_r2e:.5f}  (Qwen={qr2e:.4f}, Pythia={pr2e:.4f})")

ranked.sort(key=lambda r: r[7])
print("\n=== Best family by AIC sum (penalizes extra params; on full data) ===")
for i, row in enumerate(ranked[:3]):
    name, _, _, _, _, q_aic, p_aic, aic_sum = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
    marker = "*" if i == 0 else " "
    print(f"  {marker} {name:<36s} AIC sum = {aic_sum:>8.1f}  (Qwen={q_aic:.1f}, Pythia={p_aic:.1f})")

# Grid plot: 2x3 panel
plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 17,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
})
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
axes = axes.flatten()
for ax, (slug, display_name, func, k_params, ranges) in zip(axes, FAMILIES):
    for run_name, run in RUNS.items():
        x, y = DATA[run_name]
        ax.plot(x, y, "-", color=run["color_raw"], linewidth=2.8, alpha=0.9, label=run_name)
        b = results[slug].get(run_name)
        if b is not None:
            x_fit = np.geomspace(x.min(), x.max(), 400)
            y_fit = func(x_fit, *b["popt"])
            ax.plot(x_fit, y_fit, ":", color=run["color_fit"], linewidth=2.0, alpha=0.95)
    q_r2 = results[slug]["Qwen 0.6B"]["r2"] if results[slug].get("Qwen 0.6B") else float("nan")
    p_r2 = results[slug]["Pythia 1.4B"]["r2"] if results[slug].get("Pythia 1.4B") else float("nan")
    mean_r2 = (q_r2 + p_r2) / 2
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Cumulative Compute (6N PFLOPs, log)")
    ax.set_ylabel("Achieved Lookahead L (log)")
    ax.set_title(f"{display_name}  (mean R²={mean_r2:.5f})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
plt.tight_layout()
out = os.path.join(OUT_DIR, "qwen06b_vs_pythia14b_all_families_grid.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"\nSaved {out}")
