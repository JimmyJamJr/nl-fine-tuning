#!/usr/bin/env python3
"""Compare the arms of bench/packing_bench.sbatch (2026-10-09 packing refactor).

For each depth L and arm reads $SCRATCH/nl_output/search/job_pbench_<arm>_L<L>/loss_history.jsonl and
bench/mem_<arm>_L<L>.csv and prints, per arm: steps completed, tokens per step, step time and tokens/s over
steps WARMUP+1..end, mean achieved TFLOP/s, peak GPU memory; then pairwise parity checks against the reference arm
(default custom_fa2): tokens/step must be IDENTICAL (same examples, same packing), and the per-step loss
difference is summarised (mean |d|, max |d|, and the first step where |d| exceeds TOL). bf16 kernel drift between
FA2 and FA3 or between forward paths is expected at the 1e-3 level and grows slowly over steps; a jump at step 1
or a difference in tokens/step means a semantic bug, not drift.
Usage: python bench/compare_packing_bench.py [--ref custom_fa2] [--warmup 20] [--tol 0.02]
"""
import argparse, csv, glob, json, os
import numpy as np

S = "/scratch/gautschi/huan2073/nl_output/search"
HERE = os.path.dirname(os.path.abspath(__file__))


def load(arm, L):
    p = f"{S}/job_pbench_{arm}_L{L}/loss_history.jsonl"
    if not os.path.exists(p):
        return None
    R = [json.loads(l) for l in open(p) if l.strip()]
    return {r["step"]: r for r in R}


def peak_mem(arm, L):
    p = f"{HERE}/mem_{arm}_L{L}.csv"
    if not os.path.exists(p):
        return None
    best = {}
    for row in csv.reader(open(p)):
        if len(row) < 3:
            continue
        try:
            idx, mem = int(row[1]), float(row[2].strip().split()[0])
        except ValueError:
            continue
        best[idx] = max(best.get(idx, 0.0), mem)
    return best


def param_sync_lines(arm, L):
    """[PARAM-SYNC] lines (deterministic arms log them via NL_DEBUG_PARAM_SYNC=1) from bench/log_<arm>_L<L>.txt:
    {step: [per-rank param sums as strings]}. Bitwise-equal strings across old/new arms = identical weights after
    every optimizer step, the strongest parity criterion we have."""
    p = f"{HERE}/log_{arm}_L{L}.txt"
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p, errors="replace"):
        if "[PARAM-SYNC] step" in line:
            try:
                step = int(line.split("step")[1].split(":")[0])
                sums = line.split("[")[2].split("]")[0].replace("'", "").replace(" ", "").split(",")
                out[step] = sums
            except (IndexError, ValueError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="custom_fa2")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args()
    arms = sorted({os.path.basename(d)[len("job_pbench_"):].rsplit("_L", 1)[0] for d in glob.glob(f"{S}/job_pbench_*_L*")})
    depths = sorted({int(os.path.basename(d).rsplit("_L", 1)[1]) for d in glob.glob(f"{S}/job_pbench_*_L*")})
    if not arms:
        print("no job_pbench_* runs found"); return
    for L in depths:
        print(f"\n=== L={L} ===")
        data = {arm: load(arm, L) for arm in arms}
        print(f"{'arm':<12}{'steps':>6}{'tok/step':>11}{'s/step':>9}{'tok/s':>11}{'TFLOP/s':>9}{'peak GiB':>10}")
        for arm in arms:
            R = data[arm]
            if not R:
                print(f"{arm:<12}{'-':>6}"); continue
            steps = sorted(R)
            win = [s for s in steps if s > a.warmup]
            if len(win) >= 2:
                dt = (R[win[-1]]["wall_time"] - R[win[0]]["wall_time"]) / (len(win) - 1)
                tok = np.mean([R[s]["tokens"] for s in win])
                tfl = np.mean([R[s].get("achieved_tflops", 0) for s in win])
            else:
                dt = tok = tfl = float("nan")
            pm = peak_mem(arm, L)
            pms = f"{max(pm.values())/1024:.1f}" if pm else "-"
            print(f"{arm:<12}{len(steps):>6}{tok:>11,.0f}{dt:>9.2f}{tok/dt if dt else float('nan'):>11,.0f}{tfl:>9.0f}{pms:>10}")
        ref = data.get(a.ref)
        if not ref:
            print(f"(reference arm {a.ref} missing; no parity check)"); continue
        print(f"\nparity vs {a.ref}:")
        for arm in arms:
            if arm == a.ref or not data[arm]:
                continue
            R = data[arm]
            common = sorted(set(R) & set(ref))
            tok_mismatch = [s for s in common if R[s]["tokens"] != ref[s]["tokens"]]
            d = np.array([abs(R[s]["loss"] - ref[s]["loss"]) for s in common])
            first_bad = next((s for s, x in zip(common, d) if x > a.tol), None)
            print(f"  {arm:<12} steps={len(common):>4}  tokens/step identical: {'YES' if not tok_mismatch else f'NO ({len(tok_mismatch)} steps, first {tok_mismatch[0]})'}"
                  f"  |dloss| mean={d.mean():.4f} max={d.max():.4f}  first>{a.tol}: {first_bad}")
            ps_a, ps_r = param_sync_lines(arm, L), param_sync_lines(a.ref, L)
            if ps_a and ps_r:
                steps = sorted(set(ps_a) & set(ps_r))
                exact = sum(1 for s in steps if ps_a[s] == ps_r[s])
                first_diff = next((s for s in steps if ps_a[s] != ps_r[s]), None)
                print(f"               [PARAM-SYNC] weights bitwise identical on {exact}/{len(steps)} steps"
                      f"{'' if first_diff is None else f'; first difference at step {first_diff}'}")


if __name__ == "__main__":
    main()
