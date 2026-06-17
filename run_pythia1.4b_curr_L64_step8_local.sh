#!/bin/bash
# ============================================================================
# Local (non-SLURM) launcher for Pythia 1.4B curriculum training.
# Adapted from tuning_job_pythia1.4b_curr_L64_step8.sh to run directly on a
# single pod with 4x H100 80GB GPUs, using the `curriculum` conda env.
#
# Usage:
#   bash run_pythia1.4b_curr_L64_step8_local.sh            # fresh run
#   JOB_ID=<id> bash run_pythia1.4b_curr_L64_step8_local.sh  # resume into specific dir
#   PREV_JOB_ID=<id> bash run_pythia1.4b_curr_L64_step8_local.sh  # resume from prev job
# ============================================================================

set -euo pipefail

# ========== Resolve repo paths ==========
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR/nl"

mkdir -p "$REPO_DIR/slurm"

# ========== Conda env ==========
source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate curriculum

# ========== Paths (no /scratch on this pod) ==========
export SCRATCH="${SCRATCH:-/home/ray/scratch}"
mkdir -p "$SCRATCH/nl_output" "$SCRATCH/model_cache" "$SCRATCH/triton_cache"
export HF_HOME="$SCRATCH/model_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
# Model is already cached (see run log); stay offline to avoid spurious hub lookups.
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ========== Hardware ==========
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TOTAL_CPUS="$(nproc)"
# Cap per-rank thread count to keep NUMA sane (the pod has 208 cores)
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

# Job id — must be stable for resume/OOM restart. Timestamp unless overridden.
JOB_ID="${JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
export SLURM_JOB_ID="$JOB_ID"   # referenced by the Python script as a fallback identifier

# Deterministic master port derived from JOB_ID hash (avoid collisions on reruns)
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
#          (mirrors tuning_job_pythia1.4b_curr_L64_step8.sh)
# ==========================================================
TASK="search"
MODEL_NAME="EleutherAI/pythia-1.4b"

# Training — eff_batch = 4 GPU × 96 × 1 = 384 (matches Pythia 1B optimal config)
BATCH_SIZE=96
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=5e-5
WARMUP_STEPS=200
SEED=1234
NUM_SHOTS=0
FIRST_TOKEN_SOFT_WEIGHT=0.0

# Curriculum — step=8, L=8..64 (8 stages by default; override N_STAGES=12 for L=96)
N_STAGES="${N_STAGES:-8}"
BASE_ALPHA=0.1
MAX_ALPHA=1.0
ACCURACY_THRESHOLD=0.98
MIN_STEPS_PER_STAGE=200
CHECK_EVERY=25
ACCURACY_WINDOW=200
EVAL_EVERY_STEPS=1000

# Task — target L=64, max_input = 6 * L = 384
# (override MAX_LOOKAHEAD=96 + MAX_INPUT_SIZE=576 to extend to L=96 from a completed L=64 run)
MAX_INPUT_SIZE="${MAX_INPUT_SIZE:-384}"
MAX_LOOKAHEAD="${MAX_LOOKAHEAD:-64}"
MAX_FRONTIER_SIZE=12
MAX_BRANCH_SIZE=12
REQUESTED_BACKTRACK=3

# Memory
# NOTE: flipped to true after an OOM at stage 3 L=24 (packed-seq activations push
# past 80GB on at least one rank). GC costs ~33% step time but gains ~10-15 GB
# headroom, which is needed as L grows to 64.
GRADIENT_CHECKPOINTING=true
USE_LIGER=false               # Pythia not supported
USE_CHUNKED_CE=true
CE_CHUNK_SIZE=4096

# Evaluation
EVAL_SAMPLES=500
PRINT_EVAL_EXAMPLES=5
DO_BASELINE=true
DO_FINAL_EVAL=true
DO_REDACTED_EVAL=true
DO_SEEN_EVAL=true
DO_STAGE_EVAL=true

# Resume: set PREV_JOB_ID=<id> from env to resume from another job
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
