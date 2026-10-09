#!/usr/bin/env python3
"""Compare the 1-GPU sparse-head validation arms (bench/sparse_1gpu.sbatch, 2026-10-09).

Parity (deterministic, FA2): per step, loss and the rank-0 parameter checksum of
  (a) hf_fa2_det_full_1g  vs rank 0 of the morning's 4-GPU hf_fa2_det   -> must be bitwise identical (rank independence)
  (b) hf_fa2_det_sparse_1g vs hf_fa2_det_full_1g                        -> rounding-level only (different head GEMM shape)
plus the rolling First=/Full= accuracies from the last [Stage] log line of each arm.
Speed (FA3, 120 steps): s/step, tok/s, peak memory of hf_fa3_full_1g vs hf_fa3_sparse_1g at L=16 and L=128.
"""
import csv, json, os, re
import numpy as np

S = "/scratch/gautschi/huan2073/nl_output/search"
B = os.path.dirname(os.path.abspath(__file__))


def losses(job):
    p = f"{S}/job_pbench_{job}/loss_history.jsonl"
    if not os.path.exists(p):
        return None
    return {r["step"]: r for r in (json.loads(l) for l in open(p) if l.strip())}


def sums(log, rank=0):
    p = f"{B}/log_{log}.txt"
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p, errors="replace"):
        if "[PARAM-SYNC] step" in line:
            try:
                step = int(line.split("step")[1].split(":")[0])
                vals = line.split("[")[2].split("]")[0].replace("'", "").replace(" ", "").split(",")
                out[step] = vals[rank]
            except (IndexError, ValueError):
                pass
    return out


def last_acc(log):
    p = f"{B}/log_{log}.txt"
    if not os.path.exists(p):
        return "n/a"
    acc = "n/a"
    for line in open(p, errors="replace"):
        m = re.search(r"\] step (\d+) \|.*First=([\d.]+)% \| Full=([\d.]+)%", line)
        if m:
            acc = f"step {m.group(1)}: First={m.group(2)}% Full={m.group(3)}%"
    return acc


def peak_mem(log):
    p = f"{B}/mem_{log}.csv"
    if not os.path.exists(p):
        return float("nan")
    best = 0.0
    for row in csv.reader(open(p)):
        try:
            best = max(best, float(row[2].strip().split()[0]))
        except (IndexError, ValueError):
            pass
    return best / 1024


def report_pair(name, a, b, steps_expected):
    La, Lb = losses(a), losses(b)
    if not La or not Lb:
        print(f"  {name}: missing ({'A' if not La else ''}{'B' if not Lb else ''})"); return
    common = sorted(set(La) & set(Lb))
    d = np.array([abs(La[s]["loss"] - Lb[s]["loss"]) for s in common])
    tok = all(La[s]["tokens"] == Lb[s]["tokens"] for s in common)
    print(f"  {name}: steps {len(common)}/{steps_expected} | tokens/step identical: {tok} | loss identical on "
          f"{int((d == 0).sum())}/{len(common)} steps, max |d| = {d.max():.5f}")


print("=== PARITY (FA2, deterministic) ===")
for L, n in ((16, 40), (128, 20)):
    print(f"-- L={L}")
    report_pair("full_1g vs 4-GPU rank0 (loss)", f"hf_fa2_det_full_1g_L{L}", f"hf_fa2_det_L{L}", n)
    sa, sb = sums(f"hf_fa2_det_full_1g_L{L}"), sums(f"hf_fa2_det_L{L}", rank=0)
    if sa and sb:
        common = sorted(set(sa) & set(sb)); same = sum(sa[s] == sb[s] for s in common)
        print(f"  full_1g vs 4-GPU rank0 (checksums): identical on {same}/{len(common)} steps")
    report_pair("sparse_1g vs full_1g (loss)", f"hf_fa2_det_sparse_1g_L{L}", f"hf_fa2_det_full_1g_L{L}", n)
    sa, sb = sums(f"hf_fa2_det_sparse_1g_L{L}"), sums(f"hf_fa2_det_full_1g_L{L}")
    if sa and sb:
        common = sorted(set(sa) & set(sb))
        diffs = [abs(float(sa[s]) - float(sb[s])) for s in common]
        print(f"  sparse_1g vs full_1g (checksums): identical on {sum(sa[s] == sb[s] for s in common)}/{len(common)} steps, "
              f"max |d(param sum)| = {max(diffs) if diffs else float('nan'):.4g}")
    for arm in (f"hf_fa2_det_full_1g_L{L}", f"hf_fa2_det_sparse_1g_L{L}"):
        print(f"  {arm}: {last_acc(arm)}")

print("\n=== SPEED (FA3, 120 steps, 1 GPU) ===")
print(f"{'arm':<28}{'steps':>6}{'tok/step':>10}{'s/step':>8}{'tok/s':>10}{'peak GiB':>10}")
for L in (16, 128):
    for head in ("full", "sparse"):
        job = f"hf_fa3_{head}_1g_L{L}"
        R = losses(job)
        if not R:
            print(f"{job:<28}{'-':>6}"); continue
        st = sorted(R); win = [s for s in st if s > 20]
        dt = (R[win[-1]]["wall_time"] - R[win[0]]["wall_time"]) / (len(win) - 1) if len(win) > 1 else float("nan")
        tok = np.mean([R[s]["tokens"] for s in win]) if win else float("nan")
        print(f"{job:<28}{len(st):>6}{tok:>10,.0f}{dt:>8.3f}{tok/dt if dt else float('nan'):>10,.0f}{peak_mem(job):>10.1f}")
