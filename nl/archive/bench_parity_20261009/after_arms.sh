#!/bin/bash
# Post-refactor arms of the packing benchmark (2026-10-09). Run inside a 4xH100 allocation, e.g.
#   setsid nohup srun --jobid=<alloc> --overlap env SLURM_JOB_GPUS=0,1,2,3 SLURM_CPUS_PER_TASK=56 TAG=_h007 \
#       bash bench/after_arms.sh > bench/after_arms.out 2>&1 < /dev/null & disown
# Part 1, PARITY (deterministic kernels, NL_DEBUG_PARAM_SYNC on): old vs new path on the SAME kernel family, so the
#   per-step loss, tokens/step and the per-rank parameter checksums must agree bitwise (any difference = semantic bug).
#   FA2 is the family whose deterministic backward HF definitely honours; FA3 det arms run too and are informative.
# Part 2, SPEED (production settings, non-deterministic): new path with FA3 and FA2 at both depths, compared with the
#   old-path anchors run earlier on the same node (TAG). Each arm skips itself if already complete.
set -uo pipefail
cd /home/huan2073/nl-fine-tuning/nl
P_STEPS=${P_STEPS:-40}
echo "===== PART 1: deterministic parity arms ($P_STEPS steps at L=16, $((P_STEPS/2)) at L=128) ====="
for arm in custom_fa2 hf_fa2 custom_fa3 hf_fa3; do DETERMINISTIC=1 TAG="" bash bench/run_arm.sh "$arm" 16 "$P_STEPS"; done
for arm in custom_fa2 hf_fa2; do DETERMINISTIC=1 TAG="" bash bench/run_arm.sh "$arm" 128 $((P_STEPS/2)); done
echo "===== PART 2: speed arms (120 steps) ====="
for L in 16 128; do for arm in hf_fa3 hf_fa2; do bash bench/run_arm.sh "$arm" "$L" 120; done; done
echo "AFTER-ARMS DONE $(date)"
