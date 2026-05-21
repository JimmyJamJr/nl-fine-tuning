#!/usr/bin/env python3
"""curr (step=8) vs nocurr at each target L.
Use stage_eval_history.json directly for nocurr (has both acc + loss).
Use eval_step8_infill + 5090 archived data for curr (acc only for now)."""
import json, os, glob, subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

N = 596_049_408
FACTOR_6N = 6 * N / 1e15
CONVERGE = 0.98

CURR_CHAIN = [
    "/scratch/gautschi/huan2073/nl_output/search/job_8533255",
    "/scratch/gautschi/huan2073/nl_output/search/job_8555128",
    "/scratch/gautschi/huan2073/nl_output/search/job_9495391",
    "/scratch/gautschi/huan2073/nl_output/search/job_jackie_qwen06b_L256_step8_20260427_100934",
    "/scratch/gautschi/huan2073/nl_output/search/job_qwen06b_step8_L256_resumed",
]

# Per-L nocurr chain (in compute order)
NOCURR_CHAINS = {
    96:  ["/scratch/gautschi/huan2073/nl_output/search/job_8555129",
          "/scratch/gautschi/huan2073/nl_output/search/job_8577238"],
    104: ["/scratch/gautschi/huan2073/nl_output/search/job_8555130",
          "/scratch/gautschi/huan2073/nl_output/search/job_8577239"],
    112: ["/scratch/gautschi/huan2073/nl_output/search/job_8555131",
          "/scratch/gautschi/huan2073/nl_output/search/job_8577240",
          "/scratch/gautschi/huan2073/nl_output/search/job_8595894",
          "/scratch/gautschi/huan2073/nl_output/search/job_9495339"],
    120: ["/scratch/gautschi/huan2073/nl_output/search/job_8555132",
          "/scratch/gautschi/huan2073/nl_output/search/job_8585533",
          "/scratch/gautschi/huan2073/nl_output/search/job_8604473",
          "/scratch/gautschi/huan2073/nl_output/search/job_9495340"],
    128: ["/scratch/gautschi/huan2073/nl_output/search/job_8555133",
          "/scratch/gautschi/huan2073/nl_output/search/job_8585534",
          "/scratch/gautschi/huan2073/nl_output/search/job_9495341"],
    256: ["/scratch/gautschi/huan2073/nl_output/search/job_jackie_qwen06b_nocurr256_fresh"],
}


def step_to_pflops(dirs):
    seen = -1; cum_tok = 0; mp = {}
    for d in dirs:
        p = os.path.join(d, "loss_history.jsonl")
        if not os.path.exists(p): continue
        for line in open(p):
            try: e = json.loads(line)
            except: continue
            s = e["step"]
            if s <= seen: continue
            seen = s
            cum_tok += e.get("tokens", 0)
            mp[s] = cum_tok * FACTOR_6N
    return mp


def truncate_at_convergence(xs, ys, threshold=CONVERGE):
    xs, ys = list(xs), list(ys)
    for i, y in enumerate(ys):
        if y >= threshold:
            return xs[:i+1], ys[:i+1]
    return xs, ys


def load_nocurr_stage_evals(L):
    """Pull all stage_eval entries from chain (has both greedy_full + tf_loss)."""
    out = []
    seen_steps = set()
    for d in NOCURR_CHAINS[L]:
        p = os.path.join(d, "stage_eval_history.json")
        if not os.path.exists(p): continue
        try:
            for e in json.load(open(p)):
                if e["step"] in seen_steps: continue
                seen_steps.add(e["step"])
                out.append(e)
        except: pass
    # Add final_metrics if present
    for d in NOCURR_CHAINS[L]:
        fm_path = os.path.join(d, "final_metrics.json")
        if os.path.exists(fm_path):
            try:
                fm = json.load(open(fm_path))
                # Find last loss_history step in this dir for compute mapping
                last_step = 0
                for line in open(os.path.join(d, "loss_history.jsonl")):
                    last_step = json.loads(line)["step"]
                if last_step not in seen_steps:
                    out.append({
                        "step": last_step,
                        "greedy_full": fm.get("greedy_hard", {}).get("full_word_acc"),
                        "tf_loss": None,  # final_metrics doesn't have tf_loss for greedy_hard alone
                    })
                    seen_steps.add(last_step)
            except: pass
    return sorted(out, key=lambda e: e["step"])


def load_curr_evals(target_L):
    """Curr step=8 from smallgpu (early infill) + LOCAL late-chain archives (was on vast-5090, now offline)."""
    by_step = {}
    for f in glob.glob(f"/home/huan2073/nl-fine-tuning/nl/eval_step8_infill/eval_step8_at_L{target_L}_*.json"):
        try:
            d = json.load(open(f))
            for r in d.get("results", []):
                by_step[r["step"]] = r
        except: pass
    # Local late-chain archives: eval_step8_at_L{L}.json{,.resumed.json,.extend.json}
    for suffix in ["", ".resumed.json", ".extend.json"]:
        f = f"/home/huan2073/nl-fine-tuning/nl/eval_step8_at_L{target_L}.json{suffix}"
        if os.path.exists(f):
            try:
                d = json.load(open(f))
                for r in d.get("results", []):
                    by_step[r["step"]] = r
            except: pass
    return list(by_step.values())


def load_nocurr_archive_evals(target_L):
    """Load nocurr post-hoc eval archives (have late-chain coverage missing from stage_eval_history)."""
    by_step = {}
    # Main archives: eval_nocurr_L{L}_*.json
    for f in glob.glob(f"/home/huan2073/nl-fine-tuning/nl/eval_nocurr_L{target_L}_*.json"):
        try:
            d = json.load(open(f))
            for r in d.get("results", []):
                by_step[r["step"]] = r
        except: pass
    # 5090 persistent archives: eval_results_5090/eval_nocurr_L{L}_*.json
    for f in glob.glob(f"/home/huan2073/nl-fine-tuning/nl/eval_results_5090/eval_nocurr_L{target_L}_*.json"):
        try:
            d = json.load(open(f))
            for r in d.get("results", []):
                by_step[r["step"]] = r
        except: pass
    return list(by_step.values())


# ---- Plot: per-L curriculum vs no-curriculum panels ----
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 19,
})

curr_pf = step_to_pflops(CURR_CHAIN)


def render_panels(target_Ls, nrows, ncols, figsize, out_basename):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=False)
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    for ax, L in zip(axes, target_Ls):
        # Curr line
        curr_evals = load_curr_evals(L)
        pts = sorted([(curr_pf.get(r["step"], np.nan), r["full_word_acc"])
                      for r in curr_evals if r["step"] in curr_pf])
        if pts:
            xs, ys = zip(*pts)
            xs_t, ys_t = truncate_at_convergence(xs, ys)
            ax.plot(xs_t, ys_t, "-", color=cm.viridis(0.9), linewidth=3.5,
                    label="Qwen 0.6B with Curriculum", alpha=0.85)
            if ys_t:
                ax.axvline(x=xs_t[-1], color=cm.viridis(0.9), linestyle=":", linewidth=2.5, alpha=1.0, zorder=10)

        # Nocurr line — merge stage_eval_history + post-hoc archive evals
        n_evals = load_nocurr_stage_evals(L)
        n_archive = load_nocurr_archive_evals(L)
        nocurr_by_step = {}
        for e in n_evals:
            if e.get("greedy_full") is not None:
                nocurr_by_step[e["step"]] = e["greedy_full"]
        for r in n_archive:
            if r.get("full_word_acc") is not None:
                nocurr_by_step[r["step"]] = r["full_word_acc"]
        pf_map = step_to_pflops(NOCURR_CHAINS[L])
        n_pts = sorted([(pf_map[s], a) for s, a in nocurr_by_step.items() if s in pf_map])
        if n_pts:
            nxs, nys = zip(*n_pts)
            nxs_t, nys_t = truncate_at_convergence(nxs, nys)
            ax.plot(nxs_t, nys_t, "-", color=cm.viridis(0.0), linewidth=3.5,
                    label="Qwen 0.6B without Curriculum", alpha=0.85)
            if nys_t and nys_t[-1] >= 0.5:
                ax.axvline(x=nxs_t[-1], color=cm.viridis(0.0), linestyle=":", linewidth=2.5, alpha=1.0, zorder=10)

        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Cumulative Compute (PFLOPs)")
        ax.set_ylabel("Eval Accuracy")
        ax.set_title(f"L={L}")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.98, color="red", linestyle="--", alpha=0.8, linewidth=2.5)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x else "0"))
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=2, frameon=True)
    plt.tight_layout(rect=[0, 0.10 if nrows == 1 else 0.06, 1, 1])
    out = f"/home/huan2073/nl-fine-tuning/nl/plots/qwen06b_combined/{out_basename}.png"
    plt.savefig(out, dpi=150)
    plt.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved {out} (and PDF)")


# Body figure: 3 representative L values (compact, 1×3)
render_panels([96, 128, 256], nrows=1, ncols=3, figsize=(20, 6),
              out_basename="curr_vs_nocurr_per_L_acc_3panel")

# Appendix figure: full sweep (2×3)
render_panels([96, 104, 112, 120, 128, 256], nrows=2, ncols=3, figsize=(20, 10),
              out_basename="curr_vs_nocurr_per_L_acc")
