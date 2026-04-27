#!/bin/bash
# ============================================================================
# Qwen 0.6B curriculum extension: L=128 → L=256 (step=16 continuation)
# LOCAL RECOVERY VARIANT
# ============================================================================
# Recovered from GCS backup after pod disk loss on 2026-04-26.
# Resumes from checkpoint-39000 (step 39000, Stage 16 / L=256, ~10.5 h into stage).
#
# Latest GCS-synced state (from sync_manifest.json):
#   - latest_checkpoint_step: 39000
#   - kept: checkpoint-39000, checkpoint-38500
#   - synced_at_utc: 2026-04-26T23:59:53Z (~50 min before crash)
#
# Hardware: 4× H100 80GB (local pod). Effective batch = 768 (matches original
# job_9273210 AdamW state for clean resume).
# ============================================================================

set -euo pipefail

trap 'echo "[SIG] TERM/INT @ $(date)"; exit 130' INT TERM

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
NL_DIR="$REPO_ROOT/nl"
LOG_DIR="$REPO_ROOT/slurm"
mkdir -p "$LOG_DIR"

# JOB_ID intentionally matches the existing run dir so tuning_nl.py auto-resumes
# from the latest local checkpoint (currently checkpoint-39000).
JOB_ID="${JOB_ID_OVERRIDE:-local_20260423_230739_L256_BS96_noGC}"
LAUNCH_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${LAUNCH_TS}_${JOB_ID}_recovery.out"

echo "JOB START $(date)"
echo "JOB_ID:   $JOB_ID"
echo "LOG:      $LOG_FILE"

# ========== Environment ==========
# `curriculum` env was wiped by the disk loss; the conda `base` env on this pod
# now holds the working ML stack (torch 2.8.0+cu128, transformers 4.57.6,
# flash-attn 2.8.3, accelerate, peft, datasets, faker, liger-kernel,
# google-cloud-storage). Re-activate via the conda hook.
CONDA_BASE="${CONDA_BASE:-/home/ray/anaconda3}"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate base

export SCRATCH="${SCRATCH:-/home/ray/scratch}"
mkdir -p "$SCRATCH/nl_output" "$SCRATCH/model_cache" "$SCRATCH/triton_cache"
export HF_HOME="$SCRATCH/model_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$REPO_ROOT/gcs-bucket-sa.json}"
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ========== Hardware ==========
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | wc -l)
else
    GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
fi
[ "$GPUS_PER_NODE" -lt 1 ] && GPUS_PER_NODE=1
TOTAL_CPUS=$(nproc)
export OMP_NUM_THREADS=$(( TOTAL_CPUS / GPUS_PER_NODE ))
[ "$OMP_NUM_THREADS" -lt 1 ] && OMP_NUM_THREADS=1
export MKL_NUM_THREADS=$OMP_NUM_THREADS
ulimit -n 131072 || true

# ========== CUDA / Torch ==========
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# ========== Distributed ==========
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((10000 + ($$ % 50000)))

echo "GPUS=$GPUS_PER_NODE  OMP_THREADS=$OMP_NUM_THREADS  PORT=$MASTER_PORT"
echo "SCRATCH=$SCRATCH"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

# ==========================================================
#                      CONFIGURATION
# ==========================================================

TASK="search"
MODEL_NAME="Qwen/Qwen3-0.6B"

# Training — keep the BS96 / GA2 config that was running pre-crash
# 4 GPU × 96 × 2 = 768 eff_batch (matches original job_9273210 AdamW state)
BATCH_SIZE="${BATCH_SIZE:-96}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
SEED="${SEED:-1234}"
NUM_SHOTS="${NUM_SHOTS:-0}"
FIRST_TOKEN_SOFT_WEIGHT="${FIRST_TOKEN_SOFT_WEIGHT:-0.0}"

# Curriculum — already at stage 16/16 in the resume checkpoint; n_stages=16 mirrors original.
N_STAGES="${N_STAGES:-16}"
BASE_ALPHA="${BASE_ALPHA:-0.1}"
MAX_ALPHA="${MAX_ALPHA:-1.0}"
ACCURACY_THRESHOLD="${ACCURACY_THRESHOLD:-0.98}"
MIN_STEPS_PER_STAGE="${MIN_STEPS_PER_STAGE:-200}"
CHECK_EVERY="${CHECK_EVERY:-25}"
ACCURACY_WINDOW="${ACCURACY_WINDOW:-200}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-0}"

# Task params — target L=256
MAX_INPUT_SIZE="${MAX_INPUT_SIZE:-1536}"
MAX_LOOKAHEAD="${MAX_LOOKAHEAD:-256}"
BASE_LOOKAHEAD="${BASE_LOOKAHEAD:-16}"
LOOKAHEAD_STEP="${LOOKAHEAD_STEP:-16}"
MAX_FRONTIER_SIZE="${MAX_FRONTIER_SIZE:-12}"
MAX_BRANCH_SIZE="${MAX_BRANCH_SIZE:-12}"
REQUESTED_BACKTRACK="${REQUESTED_BACKTRACK:-3}"

# Memory / speed knobs — GC=true is required at max_input=1536 even on H100 80GB
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
USE_LIGER="${USE_LIGER:-true}"
USE_CHUNKED_CE="${USE_CHUNKED_CE:-true}"
CE_CHUNK_SIZE="${CE_CHUNK_SIZE:-4096}"

# Disk hygiene — local pod has a 500GB ephemeral cap; the previous run died
# because save_total_limit=20 (default in tuning_nl.py) plus uncapped
# persistent_checkpoints/ filled the disk. We keep only 3 rolling local
# checkpoints (~10 GB) and rely on sync_to_gcs.sh + GCS backup for history.
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
PERSIST_EVERY="${PERSIST_EVERY:-2000}"

# Evaluation
EVAL_SAMPLES="${EVAL_SAMPLES:-500}"
PRINT_EVAL_EXAMPLES="${PRINT_EVAL_EXAMPLES:-5}"
DO_BASELINE="${DO_BASELINE:-true}"
DO_FINAL_EVAL="${DO_FINAL_EVAL:-true}"
DO_REDACTED_EVAL="${DO_REDACTED_EVAL:-true}"
DO_SEEN_EVAL="${DO_SEEN_EVAL:-true}"
DO_STAGE_EVAL="${DO_STAGE_EVAL:-true}"
STAGE_EVAL_EVERY="${STAGE_EVAL_EVERY:-8}"

JOB_OUTPUT_DIR="$SCRATCH/nl_output/$TASK/job_${JOB_ID}"
RESTART_FLAG="$JOB_OUTPUT_DIR/RESTART_FLAG"

# ==========================================================
echo "Task: $TASK | Model: $MODEL_NAME | Max Input: $MAX_INPUT_SIZE | Target L: $MAX_LOOKAHEAD"
echo "Output dir: $JOB_OUTPUT_DIR"
if [ -d "$JOB_OUTPUT_DIR" ]; then
    LATEST_CKPT=$(ls -1d "$JOB_OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    echo "Existing run dir found. Latest checkpoint: ${LATEST_CKPT:-<none>}"
fi

# Pre-flight: model cache (for tokenizer / config)
python -c "
from transformers import AutoTokenizer
m, c = '$MODEL_NAME', '$HF_HOME'
AutoTokenizer.from_pretrained(m, cache_dir=c, trust_remote_code=True)
print('[OK] Tokenizer cached')
"

# Pre-flight: build C++ generator if not importable
( cd "$NL_DIR" && python -c "
try:
    import generator
    print('[OK] C++ generator present')
except ImportError:
    print('[INFO] Building C++ generator...')
    import subprocess, sys
    subprocess.check_call([sys.executable, 'nl_generator.py'])
    print('[OK] C++ generator built')
" )

# ========== Build arguments ==========
ARGS=(
    --task "$TASK"
    --model_name "$MODEL_NAME"
    --cache_dir "$HF_HOME"
    --output_dir "$SCRATCH/nl_output"
    --scratch_dir "$SCRATCH"
    --job_id "$JOB_ID"

    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --warmup_steps "$WARMUP_STEPS"
    --seed "$SEED"
    --num_shots "$NUM_SHOTS"
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
    --max_frontier_size "$MAX_FRONTIER_SIZE"
    --max_branch_size "$MAX_BRANCH_SIZE"
    --requested_backtrack "$REQUESTED_BACKTRACK"

    --eval_samples "$EVAL_SAMPLES"
    --print_eval_examples "$PRINT_EVAL_EXAMPLES"
    --stage_eval_every "$STAGE_EVAL_EVERY"

    --use_packing
    --linear_lookahead
    --base_lookahead "$BASE_LOOKAHEAD"
    --lookahead_step "$LOOKAHEAD_STEP"

    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --persist_every "$PERSIST_EVERY"
)

# Conditional flags
$GRADIENT_CHECKPOINTING && ARGS+=(--gradient_checkpointing)
$USE_LIGER && ARGS+=(--use_liger)
$USE_CHUNKED_CE && ARGS+=(--use_chunked_ce --ce_chunk_size "$CE_CHUNK_SIZE")
$DO_BASELINE && ARGS+=(--do_baseline)
$DO_FINAL_EVAL && ARGS+=(--do_final_eval)
$DO_REDACTED_EVAL && ARGS+=(--do_redacted_eval)
$DO_SEEN_EVAL && ARGS+=(--do_seen_eval)
$DO_STAGE_EVAL && ARGS+=(--do_stage_eval)

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"

# ========== Training loop with OOM restart ==========
MAX_RETRIES="${MAX_RETRIES:-10}"
RETRY_COUNT=0

cd "$NL_DIR"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "========== Attempt $((RETRY_COUNT+1))/$MAX_RETRIES | Port $MASTER_PORT =========="

    set +e
    torchrun --nproc_per_node=$GPUS_PER_NODE \
             --master_port=$MASTER_PORT \
             --max_restarts=0 \
             tuning_nl.py "${ARGS[@]}"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS $(date)"
        rm -f "$RESTART_FLAG"
        exit 0
    elif [ -f "$RESTART_FLAG" ]; then
        echo "[OOM] Restarting after $(date)..."
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
