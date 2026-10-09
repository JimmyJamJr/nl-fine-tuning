#!/bin/bash
#SBATCH -J nl_qwen06b_step
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=7-00:00:00
#SBATCH --partition=ai
#SBATCH -A asaparov
#SBATCH -q normal
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#
# Step-size sweep, rerun. Parameterised by S_STEP (1, 8, 16 or 32) passed with --export.
#
# NO ARM SHOULD EVER STOP ON A LIMIT THIS SCRIPT CHOSE. Both stopping conditions were
# raised on 2026-09-22 so that the only thing which ends an arm is its curriculum gate
# failing to clear. Every earlier value here was scoped to a narrower question and had to
# be raised mid-flight, which cost queue turns; do not tighten any of them again.
#   compute:  a common 1,500,000 PFLOPs for all four arms (was 300K for s=16, 700K for
#             s=32, which would have censored each curve at a different point and made a
#             crossover unreadable).
#   lookahead: MAXL 512 with n_stages = 512/s, and max_input_size 3079 = 6*512+7, the
#             smallest input size whose ceiling L_eff = min(((n-4)/3-1)/2, (n-4)/6)
#             reaches 512 (2048 gave 340). The old MAXL=320 was about to bind: s=32 was
#             in stage 9 of 10 at L=288, so it would have ENDED at L=320 having cleared
#             every stage, which reads as "finished", not as the stall the experiment is
#             looking for. Under --linear_lookahead the graph at a given L is n-invariant,
#             so a larger n costs nothing per unit of L and may be raised mid-chain.
#             Micro-batch tokens already exceed NL_CKPT_RELAX_MAX_TOKENS at n=2048
#             (48 x 2048 = 98k > 88k), so the checkpointing regime does not change.
# 1.5M is matched to MAXL=512: at recent per-stage costs s=32 reaches 512 near 1.3M.
# The original s=16 budget existed for a narrower question: did its archived early-budget lead
# (144 at 100K, 208 at 200K) survive standardising accuracy_window from its archived 200 to 800?
# It did not, and that was settled inside 300K. The sweep is now answering a different question,
# which needs every arm carried far past its first stall: coarse steps advance more cheaply until
# they meet a stage jump they cannot clear, and finer steps then grind past them at equal or lower
# cumulative compute. A crossover of that kind is only observable if the arms stop because their
# curriculum gate stops clearing, not because their compute budget ran out. Unequal caps would
# censor each curve at a different point and make the crossover unreadable.
# 1.5M matches the s=1 canonical chain and the s=8 arm, so all four standardised arms now share
# one budget. Final achieved L is not the quantity of interest and is settable by the budget; the
# comparison is lookahead per unit of compute, and where each arm's curve goes flat.
#
# WHY THIS IS A RERUN AND NOT A RESUME. Every archived step-size arm lost its weights in the
# 2026-06-30 purge: the checkpoint-* directories survive but are empty, so nothing can be resumed.
#
# WHAT CHANGED FROM THE ARCHIVES, DELIBERATELY.
#   accuracy_window: archives used 800 (s=1), 200 (s=8), 1000 (s=32). That is a confound, not a
#     detail: the gate is a rolling exact-match rate over the last W search examples against a 0.98
#     threshold, so a smaller W is noisier and easier to cross by chance, advancing stages faster
#     independently of step size. s=8 had the smallest window and looks best in the archives.
#     Standardised here at 800 for all arms, matching the canonical s=1 chain.
#   warmup_steps: archives used 500 (s=1) and 100 (s=8, s=32). Standardised at 500.
#   batch geometry: archives used bs=96 ga=2; here bs=48 ga=4. Effective batch is 768 either way,
#     but halving the micro-batch lets the selective-checkpointing token gate actually engage at
#     low L. At bs=96 it never fires.
#   cap: max_lookahead 320 with max_input_size 2048, against 256/1536 in the archives. Under
#     --linear_lookahead the graph at a given L is n-invariant (see effective_search_L), so raising
#     the cap costs nothing per unit of L; it only removes the censoring that pinned every archived
#     arm near 256. 320 rather than 512 keeps sequences short enough to stay in tested memory
#     territory.
# Consequence: these runs are mutually comparable and comparable in compute to the archives, but
# they will NOT reproduce the archived s=8 and s=32 curves. That is the point.
#
# No --use_chunked_ce: the flag was removed from tuning_nl.py (2026-10-09); chunked CE is always on.

set -uo pipefail
cd /home/huan2073/nl-fine-tuning/nl
trap 'set +e; echo "[USR1] $(date): walltime near, stopping trainer and requeueing"; [ -n "${TPID:-}" ] && kill -TERM "$TPID" 2>/dev/null; sleep 60; scontrol requeue "$SLURM_JOB_ID" || { sleep 30; scontrol requeue "$SLURM_JOB_ID"; }; exit 0' USR1

module load conda
conda activate search

export SCRATCH="/scratch/gautschi/$USER"
export HF_HOME="$SCRATCH/model_cache"
export TORCH_HOME="$SCRATCH/model_cache"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NL_DDP_TIMEOUT_MIN=240
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Selective checkpointing: every 2nd layer while the micro-batch is small enough, falling back to
# every layer above 88k tokens. Verified on this batch geometry by probe 6 (L=104 and L=128 survive
# gated; the ungated control OOMs at the same depth). Gradients are identical either way.
export NL_CKPT_EVERY_N_LAYERS=2
export NL_CKPT_RELAX_MAX_TOKENS=88000

: "${S_STEP:?S_STEP must be set to 1, 8, 16 or 32 via --export}"
MAXL=512
case "$S_STEP" in
  1)  NSTAGES=512; PFLOP_CAP=1500000 ;;
  8)  NSTAGES=64;  PFLOP_CAP=1500000 ;;
  16) NSTAGES=32;  PFLOP_CAP=1500000 ;;
  32) NSTAGES=16;  PFLOP_CAP=1500000 ;;
  *)  echo "FATAL: S_STEP must be 1, 8, 16 or 32 (got $S_STEP)"; exit 1 ;;
esac
[ $(( MAXL / S_STEP )) -eq "$NSTAGES" ] || { echo "FATAL: n_stages $NSTAGES != $MAXL/$S_STEP"; exit 1; }

export TRITON_CACHE_DIR="$SCRATCH/triton_cache_step$S_STEP"
mkdir -p slurm "$TRITON_CACHE_DIR"

GPUS_PER_NODE=$(echo "$SLURM_JOB_GPUS" | tr "," "\n" | wc -l)
[ "$GPUS_PER_NODE" -eq 4 ] || { echo "FATAL: need 4 GPUs for eff_batch 768"; exit 1; }
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

JOB_ID="qwen06b_step${S_STEP}_rerun"
OUT_DIR="$SCRATCH/nl_output/search/job_$JOB_ID"
for d in "$OUT_DIR"/checkpoint-*; do [ -d "$d" ] || continue; python -c "import json,sys; json.load(open(sys.argv[1]))" "$d/curriculum_state.json" 2>/dev/null || { echo "[PRUNE] removing partial $d"; rm -rf "$d"; }; done

ARGS=(
    --task search
    --model_name Qwen/Qwen3-0.6B
    --cache_dir "$HF_HOME"
    --output_dir "$SCRATCH/nl_output"
    --scratch_dir "$SCRATCH"
    --job_id "$JOB_ID"
    --batch_size 48
    --gradient_accumulation_steps 4
    --learning_rate 5e-5
    --warmup_steps 500
    --seed 1234
    --num_shots 0
    --first_token_soft_weight 0.0
    --n_stages "$NSTAGES"
    --base_alpha 0.1
    --max_alpha 1.0
    --accuracy_threshold 0.98
    --min_steps_per_stage 200
    --check_every 25
    --accuracy_window 800
    --eval_every_steps 0
    --max_input_size 3079
    --max_lookahead "$MAXL"
    --linear_lookahead
    --base_lookahead "$S_STEP"
    --lookahead_step "$S_STEP"
    --max_frontier_size 12
    --max_branch_size 12
    --requested_backtrack 3
    --eval_samples 0
    --print_eval_examples 0
    --ce_chunk_size 4096
    --use_packing
    --use_liger
    --gradient_checkpointing
    --persist_every 2000
    --save_steps 500
    --save_total_limit 2
    --max_total_pflops "$PFLOP_CAP"
)

echo "JOB START $(date)  s=$S_STEP  n_stages=$NSTAGES  cap=${MAXL}  budget=${PFLOP_CAP} PFLOPs"
echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE tuning_nl.py ${ARGS[*]}"
torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}" &
TPID=$!
wait $TPID
echo "JOB END $(date) rc=$?"
