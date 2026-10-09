#!/bin/bash
#SBATCH -J nl_17b_curr48_thr99
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=56
#SBATCH --time=48:00:00
#SBATCH --partition=ai
#SBATCH -A asaparov
#SBATCH -q normal
#SBATCH --mail-user=huan2073@purdue.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

set -euo pipefail

# ========== Preemption handling ==========
# The first attempt of this run (16501338) hit its walltime and was NOT requeued: the
# trap never fired, because torchrun ran in the FOREGROUND and bash defers traps until
# the foreground command returns. The fix is to background torchrun and `wait` on it, so
# USR1 interrupts the wait and the handler actually runs. TPID is set by the training
# loop below; stop the trainer first so it can flush, then requeue.
trap 'set +e; echo "[SIG] USR1 @ $(date) — walltime near, stopping trainer and requeueing"; [ -n "${TPID:-}" ] && kill -TERM "$TPID" 2>/dev/null; sleep 120; scontrol requeue "$SLURM_JOB_ID" || { sleep 30; scontrol requeue "$SLURM_JOB_ID"; }; exit 0' USR1
trap 'echo "[SIG] TERM @ $(date)"; exit 0' TERM
TPID=""

mkdir -p ./slurm
echo "JOB START $(date)"

# ========== Environment ==========
module load conda
conda activate search

export SCRATCH="/scratch/gautschi/$USER"
mkdir -p "$SCRATCH/nl_output" "$SCRATCH/model_cache" "$SCRATCH/triton_cache"
export HF_HOME="$SCRATCH/model_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"   # liger/triton must never write to home quota
if [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
fi
export HF_HUB_OFFLINE=0  # Need online for streaming Dolci-Instruct-SFT mix
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ========== Hardware ==========
GPUS_PER_NODE=$(echo $SLURM_JOB_GPUS | tr "," "\n" | wc -l)
[ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
if [ "$GPUS_PER_NODE" -ne 4 ]; then
    echo "[FATAL] Need exactly 4 GPUs for eff_batch=768 (48 x 4 GA x 4 GPUs); got $GPUS_PER_NODE — refusing to train an incomparable run"
    exit 1
fi

export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
ulimit -n 131072 || true

# ========== CUDA / Torch ==========
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# ========== Distributed ==========
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"
export MASTER_PORT=$((10000 + (SLURM_JOB_ID % 50000)))

echo "GPUS=$GPUS_PER_NODE  OMP_THREADS=$OMP_NUM_THREADS"
nvidia-smi || true

# ==========================================================
#   1.7B curriculum, s=1, 6% Dolci, ADVANCEMENT THRESHOLD 0.99. Ceiling L=128.
#
#   This is job_retrain_17b_curr48_s1_dolci6 with exactly one training parameter
#   changed: accuracy_threshold 0.98 -> 0.99. Everything that sets training dynamics
#   is byte-identical to that run (eff_batch 768 = 48 x 4GA x 4 GPUs, lr 5e-5,
#   warmup 100, seed 1234, s=1, accuracy_window 200, min_steps_per_stage 200,
#   check_every 25), so the two are directly comparable at every L the 0.98 arm
#   reached and any difference is attributable to the threshold alone.
#
#   QUESTION. The curriculum arm may be advancing before it has really mastered
#   a stage. The gate is a rolling exact-match rate over the last 200 search
#   examples, and at 0.98 that is four permitted errors in the window. At 0.99 it
#   is two. If under-training per stage is what limits the achieved lookahead,
#   the stricter gate should spend more compute per stage and reach a given L
#   with better quality, at the cost of reaching it later.
#
#   CEILING RAISED 2026-09-21 (was L=48 with n=288). Under --linear_lookahead the
#   graph at lookahead L has 2L+1 edges regardless of max_input_size (see
#   alpha_for_lookahead), so raising n changes nothing at any L already trained.
#   n only sets the ceiling, L_eff = min(((n-4)//3 - 1)//2, (n-4)//6). The old
#   n=288 gives L_eff=46, so the "L=48" arms in fact train stages 47-48 on L=46
#   graphs; n=780 is the smallest n with L_eff=128. Stage checkpoints are written
#   at EVERY stage into stage_checkpoints/ (weights + curriculum_state, never
#   rotated), so L=16/32/48/64/.../128 are all retained. The alpha=1.0 "hard"
#   in-run eval set is regenerated at the new n and is not comparable to the old
#   one; downstream evals load stage_checkpoints/ and are unaffected.
#
#   Cap raised 200,000 -> 2,000,000 PFLOPs. 75,373 were spent completing L=32; the
#   0.99 gate costs ~1.8-2x the 0.98 arm per unit L and cumulative cost has grown
#   ~L^1.85 so far, so L=128 is of order 1M PFLOPs, i.e. 8+ days on 4xH100.
#
#   Walltime 48h with --requeue and a USR1 trap that actually fires. The output
#   directory is pinned to job_curr48_thr99 (EFFECTIVE_JOB_ID default below), so
#   every entry resumes from the last checkpoint in that dir. checkpoint-22500 was
#   truncated by the 16501338 walltime kill and is quarantined under
#   nl_output/search/_corrupt/; resume lands on checkpoint-22000 (stage 33, L=32
#   completed).
# ==========================================================

TASK="search"
MODEL_NAME="Qwen/Qwen3-1.7B"

# Training  (run_meta: --batch_size 48 --gradient_accumulation_steps 4
#            --learning_rate 5e-5 --seed 1234; world_size 4 -> eff_batch 768)
BATCH_SIZE=48
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=5e-5
SEED=1234
FIRST_TOKEN_SOFT_WEIGHT=0.0

# Curriculum (run_meta: --n_stages 96 --base_alpha 0.1 --max_alpha 1.0
#             --accuracy_threshold 0.98 --min_steps_per_stage 200 --check_every 25
#             --accuracy_window 200 --eval_every_steps 0)
N_STAGES=128
BASE_ALPHA=0.1
MAX_ALPHA=1.0
ACCURACY_THRESHOLD=0.99
MIN_STEPS_PER_STAGE=200
CHECK_EVERY=25
ACCURACY_WINDOW=200
EVAL_EVERY_STEPS=0

# Task parameters. n=780 is the smallest n with L_eff >= 128 (see header); the old
# 6*L convention (n=288 for L=48) undershoots the ceiling by two stages.
MAX_INPUT_SIZE=780
MAX_LOOKAHEAD=128

# Memory optimizations
CE_CHUNK_SIZE=4096

# Evaluation (run_meta: --eval_samples 500 --print_eval_examples 5 + all --do_* flags)
EVAL_SAMPLES=500
PRINT_EVAL_EXAMPLES=5

# Stop cap (see header)
MAX_TOTAL_PFLOPS=2000000

# ==========================================================

# Pinned: this script only ever continues the one run in job_curr48_thr99. A bare
# `sbatch` without JOB_ID_OVERRIDE used to default to $SLURM_JOB_ID and start a fresh
# run in a new directory, silently abandoning the chain.
EFFECTIVE_JOB_ID="${JOB_ID_OVERRIDE:-curr48_thr99}"
echo "Task: $TASK | Model: $MODEL_NAME | Max Input: $MAX_INPUT_SIZE"
echo "Output: $SCRATCH/nl_output/$TASK/job_${EFFECTIVE_JOB_ID}"
echo "eff_batch = $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS x $GPUS_PER_NODE = $(( BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * GPUS_PER_NODE ))  (must be 768)"

# ========== Build C++ generator if needed ==========
python -c "
try:
    import generator
    print('[OK] C++ generator present')
except:
    print('[INFO] Building C++ generator...')
    import subprocess, sys
    subprocess.check_call([sys.executable, 'nl_generator.py'])
"

# ========== Cache model ==========
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
m, c = '$MODEL_NAME', os.environ['HF_HOME']
AutoTokenizer.from_pretrained(m, cache_dir=c, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(m, cache_dir=c, trust_remote_code=True)
print('[OK] Model cached')
"

# ========== Build arguments ==========
ARGS=(
    --task "$TASK"
    --model_name "$MODEL_NAME"
    --cache_dir "$HF_HOME"
    --output_dir "$SCRATCH/nl_output"
    --scratch_dir "$SCRATCH"
    --job_id "$EFFECTIVE_JOB_ID"

    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --seed "$SEED"
    --first_token_soft_weight "$FIRST_TOKEN_SOFT_WEIGHT"

    --n_stages "$N_STAGES"
    --base_alpha "$BASE_ALPHA"
    --max_alpha "$MAX_ALPHA"
    --accuracy_threshold "$ACCURACY_THRESHOLD"
    --min_steps_per_stage "$MIN_STEPS_PER_STAGE"
    --check_every "$CHECK_EVERY"
    --accuracy_window "$ACCURACY_WINDOW"
    --eval_every_steps "$EVAL_EVERY_STEPS"

    --max_input_size "$MAX_INPUT_SIZE"
    --max_lookahead "$MAX_LOOKAHEAD"

    --eval_samples "$EVAL_SAMPLES"
    --print_eval_examples "$PRINT_EVAL_EXAMPLES"

    --linear_lookahead
    --base_lookahead 1
    --lookahead_step 1
    --mix_pretrain_data allenai/Dolci-Instruct-SFT
    --mix_pretrain_ratio 0.06
    --use_chat_template
    --stage_eval_every 8

    --gradient_checkpointing
    --use_liger
    --ce_chunk_size "$CE_CHUNK_SIZE"

    --do_baseline
    --do_final_eval
    --do_stage_eval

    --persist_every 500
    --max_total_pflops "$MAX_TOTAL_PFLOPS"
)
# NOTE: no --resume_from_job. Every entry, first submission or requeue, auto-resumes
# from the last complete checkpoint-* in the pinned job dir (get_last_checkpoint).

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"

# ========== Training loop with OOM restart ==========
MAX_RETRIES=10
RETRY_COUNT=0
JOB_OUTPUT_DIR="$SCRATCH/nl_output/$TASK/job_${EFFECTIVE_JOB_ID}"
RESTART_FLAG="$JOB_OUTPUT_DIR/RESTART_FLAG"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "========== Attempt $((RETRY_COUNT+1))/$MAX_RETRIES | Port $MASTER_PORT =========="

    set +e
    # Backgrounded so the USR1 trap can fire; see the trap comment at the top.
    torchrun --nproc_per_node=$GPUS_PER_NODE \
             --master_port=$MASTER_PORT \
             --max_restarts=0 \
             tuning_nl.py "${ARGS[@]}" &
    TPID=$!
    wait $TPID
    EXIT_CODE=$?
    TPID=""
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS $(date)"
        rm -f "$RESTART_FLAG"
        exit 0
    elif [ -f "$RESTART_FLAG" ]; then
        echo "[OOM] Restarting with reduced batch size..."
        MASTER_PORT=$((MASTER_PORT + 1))
        RETRY_COUNT=$((RETRY_COUNT + 1))
        rm -f "$RESTART_FLAG"
        sleep 5
    else
        echo "FAILED with exit code $EXIT_CODE $(date)"
        exit $EXIT_CODE
    fi
done

echo "Max retries ($MAX_RETRIES) reached"
exit 1
