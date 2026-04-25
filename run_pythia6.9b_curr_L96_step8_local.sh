#!/bin/bash
# ============================================================================
# Local (non-SLURM) launcher for Pythia 6.9B curriculum training.
# Adapted from run_pythia1.4b_curr_L64_step8_local.sh for a 4x H200 141GB pod.
#
# Target: L=96 step=8 (matches Pythia 160M/410M/1B paper runs for direct
# apples-to-apples scaling comparison).
# Effective batch: 4 GPU x 24 x 4 = 384 (matches other Pythia scales).
# LR: 2e-5 (scaled down from 5e-5 at smaller scales).
#
# Usage:
#   bash run_pythia6.9b_curr_L96_step8_local.sh                 # fresh run
#   JOB_ID=<id> bash run_pythia6.9b_curr_L96_step8_local.sh     # resume into specific dir
#   PREV_JOB_ID=<id> bash run_pythia6.9b_curr_L96_step8_local.sh  # resume from prev job
# ============================================================================

set -euo pipefail

# ========== Resolve repo paths ==========
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR/nl"

mkdir -p "$REPO_DIR/slurm"

# ========== Conda env ==========
# If the pod uses conda, uncomment and adjust. Otherwise assume system Python
# has torch/transformers/datasets/pybind11 already available.
if [ -f /home/ray/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/ray/anaconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-curriculum}"
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-base}"
fi

# ========== Paths ==========
export SCRATCH="${SCRATCH:-/workspace}"
mkdir -p "$SCRATCH/nl_output" "$SCRATCH/model_cache" "$SCRATCH/triton_cache"
export HF_HOME="$SCRATCH/model_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
# Set HF_HUB_OFFLINE=1 once the 6.9B weights are cached locally.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ========== Hardware ==========
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TOTAL_CPUS="$(nproc)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
ulimit -n 131072 || true

# ========== CUDA / Torch ==========
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# ========== Distributed ==========
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR="127.0.0.1"

JOB_ID="${JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
export SLURM_JOB_ID="$JOB_ID"

MASTER_PORT="${MASTER_PORT:-$((10000 + ($(echo "$JOB_ID" | cksum | cut -d' ' -f1) % 50000)))}"
export MASTER_PORT

echo "========================================================="
echo "JOB_ID      = $JOB_ID"
echo "GPUS        = $GPUS_PER_NODE"
echo "OMP_THREADS = $OMP_NUM_THREADS  (of $TOTAL_CPUS cpus)"
echo "SCRATCH     = $SCRATCH"
echo "HF_HOME     = $HF_HOME"
echo "MASTER_PORT = $MASTER_PORT"
echo "========================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ==========================================================
#                      CONFIGURATION
# ==========================================================
TASK="search"
MODEL_NAME="EleutherAI/pythia-6.9b"

# Training — eff_batch = 4 GPU x 24 x 4 = 384 (matches Pythia 1B/1.4B optimal).
BATCH_SIZE="${BATCH_SIZE:-24}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE=2e-5
WARMUP_STEPS=400
SEED=1234
NUM_SHOTS=0
FIRST_TOKEN_SOFT_WEIGHT=0.0

# Curriculum — step=8, L=8..96 (12 stages). Matches 160M/410M/1B runs.
N_STAGES="${N_STAGES:-12}"
BASE_ALPHA=0.1
MAX_ALPHA=1.0
ACCURACY_THRESHOLD=0.98
MIN_STEPS_PER_STAGE=200
CHECK_EVERY=25
ACCURACY_WINDOW=200
EVAL_EVERY_STEPS=1000

# Task — target L=96, max_input = 6 * L = 576
MAX_INPUT_SIZE="${MAX_INPUT_SIZE:-576}"
MAX_LOOKAHEAD="${MAX_LOOKAHEAD:-96}"
MAX_FRONTIER_SIZE=12
MAX_BRANCH_SIZE=12
REQUESTED_BACKTRACK=3

# Memory
# H200 141GB has headroom: state ~83GB + activations ~25GB at batch 24 = ~108GB.
# GC off for ~25% speedup; flip to true if late-stage OOM (like Jackie hit on 1.4B).
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
USE_LIGER=false               # Pythia not supported
USE_CHUNKED_CE=true
CE_CHUNK_SIZE=4096

# Evaluation — all in-training evals OFF; eval checkpoints offline instead.
# Stage advancement is driven by accuracy_threshold + recent_losses, independent
# of these flags. stage_checkpoints/ and persistent_checkpoints/ still written.
EVAL_SAMPLES=500
PRINT_EVAL_EXAMPLES=0
DO_BASELINE=false
DO_FINAL_EVAL=false
DO_REDACTED_EVAL=false
DO_SEEN_EVAL=false
DO_STAGE_EVAL=false

# Rolling checkpoint window. stage_checkpoints/ + persistent_checkpoints/ are
# kept indefinitely; this only limits the regular checkpoint-XXXXX/ dirs.
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

PREV_JOB_ID="${PREV_JOB_ID:-}"

# ==========================================================

echo "Task: $TASK | Model: $MODEL_NAME | Max Input: $MAX_INPUT_SIZE | Target L: $MAX_LOOKAHEAD"
echo "Output: $SCRATCH/nl_output/$TASK/job_${JOB_ID}"
[ -n "$PREV_JOB_ID" ] && echo "Resuming from: job_$PREV_JOB_ID"

# ========== Build C++ generator if needed ==========
python -c "
try:
    import generator
    print('[OK] C++ generator present')
except Exception:
    print('[INFO] Building C++ generator...')
    import subprocess, sys
    subprocess.check_call([sys.executable, 'nl_generator.py'])
"

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
    --save_total_limit "$SAVE_TOTAL_LIMIT"

    --use_packing

    --linear_lookahead
    --base_lookahead 8
    --lookahead_step 8
)

$GRADIENT_CHECKPOINTING && ARGS+=(--gradient_checkpointing)
$USE_LIGER && ARGS+=(--use_liger)
$USE_CHUNKED_CE && ARGS+=(--use_chunked_ce --ce_chunk_size "$CE_CHUNK_SIZE")
$DO_BASELINE && ARGS+=(--do_baseline)
$DO_FINAL_EVAL && ARGS+=(--do_final_eval)
$DO_REDACTED_EVAL && ARGS+=(--do_redacted_eval)
$DO_SEEN_EVAL && ARGS+=(--do_seen_eval)
$DO_STAGE_EVAL && ARGS+=(--do_stage_eval)
[ -n "$PREV_JOB_ID" ] && ARGS+=(--resume_from_job "$PREV_JOB_ID")

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"

# ========== Training loop with OOM restart ==========
MAX_RETRIES=10
RETRY_COUNT=0
JOB_OUTPUT_DIR="$SCRATCH/nl_output/$TASK/job_${JOB_ID}"
RESTART_FLAG="$JOB_OUTPUT_DIR/RESTART_FLAG"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "========== Attempt $((RETRY_COUNT+1))/$MAX_RETRIES | Port $MASTER_PORT =========="

    set +e
    torchrun --nproc_per_node="$GPUS_PER_NODE" \
             --master_port="$MASTER_PORT" \
             --max_restarts=0 \
             tuning_nl.py "${ARGS[@]}"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS $(date)"
        rm -f "$RESTART_FLAG"

        # Auto-shutdown the RunPod pod on successful completion.
        # Set SHUTDOWN_ON_SUCCESS=false to disable. 60s grace period for any
        # tail logs / final rsync to flush.
        if [ "${SHUTDOWN_ON_SUCCESS:-true}" = "true" ]; then
            if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
                echo "[SHUTDOWN] Training complete — stopping pod $RUNPOD_POD_ID in 60s. Cancel with Ctrl-C."
                sleep 60
                runpodctl stop pod "$RUNPOD_POD_ID" || echo "[SHUTDOWN] runpodctl stop failed"
            else
                echo "[SHUTDOWN] Skipping auto-shutdown (RUNPOD_POD_ID unset or runpodctl missing)"
            fi
        fi
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
