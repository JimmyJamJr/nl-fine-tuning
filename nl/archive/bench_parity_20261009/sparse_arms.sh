#!/bin/bash
# Validation + speed arms for the sparse head (NL_HEAD=sparse, 2026-10-09). Run inside a 4xH100 allocation:
#   setsid nohup srun --jobid=<alloc> --overlap env SLURM_JOB_GPUS=0,1,2,3 SLURM_CPUS_PER_TASK=56 NODE=_h007 \
#       bash bench/sparse_arms.sh > bench/sparse_arms.out 2>&1 < /dev/null & disown
# Part 1 (parity, deterministic kernels): sparse head vs the existing full-head deterministic arms (hf_fa3_det / hf_fa2_det
#   at L=16, 40 steps; hf_fa2_det at L=128, 20 steps). Not bitwise (different GEMM shape): expect |dloss| ~1e-3 and
#   identical tokens/step; the gate metrics are compared from the [Stage] log lines.
# Part 2 (speed, production settings): sparse head with FA3 and FA2 at both depths, 120 steps, TAG=_sparse$NODE so the
#   runs compare against the same-node full-head arms (hf_fa3$NODE etc.).
set -uo pipefail
cd /home/huan2073/nl-fine-tuning/nl
NODE=${NODE:-}
export NL_HEAD=sparse
echo "===== PART 1: sparse-head deterministic arms ====="
for arm in hf_fa3 hf_fa2; do DETERMINISTIC=1 TAG="_sparse" bash bench/run_arm.sh "$arm" 16 40; done
DETERMINISTIC=1 TAG="_sparse" bash bench/run_arm.sh hf_fa2 128 20
echo "===== PART 2: sparse-head speed arms (120 steps) ====="
for L in 16 128; do for arm in hf_fa3 hf_fa2; do TAG="_sparse${NODE}" bash bench/run_arm.sh "$arm" "$L" 120; done; done
echo "SPARSE-ARMS DONE $(date)"
