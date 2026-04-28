#!/bin/bash
# =============================================================================
# Qwen 1.7B 6%-Dolci-Instruct curriculum — RESUME from GCS-cached job dir.
#
# This is the "pod crashed, env wiped, but GCS has the run" recovery script.
# Unlike the original launcher (which fabricates a checkpoint-26400 from a HF
# seed), this one pulls the full cached job dir back from GCS and lets the
# trainer auto-resume from the latest rolling checkpoint (which has a real
# optimizer.pt — full Adam-state continuity).
#
# Cached snapshot:
#   gs://jackierwzhang-purdue-research-curriculum/qwen17b-6pct-dolci-stage32-L48/job_local_20260426_172317_qwen17b_6pct_L48/
#
# At time of GCS sync the run was mid-stage 47 (step 32500); curriculum had
# advanced through L=32..47 already. Fresh run picks up at step 32500 and
# should clear stage 47 + train stage 48 to finish.
#
# Run on 2x H100 80GB pod after env setup:
#   bash tuning_job_qwen17b_6pct_L48_resume_from_gcs.sh
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR/nl"

if [ -f /home/ray/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/ray/anaconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-curriculum}"
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-base}"
fi

export SCRATCH="${SCRATCH:-/home/ray/scratch}"
mkdir -p "$SCRATCH/nl_output" "$SCRATCH/model_cache" "$SCRATCH/triton_cache"
export HF_HOME="$SCRATCH/model_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# ========== Job ID — must match the cached run on GCS ==========
JOB_ID="${JOB_ID:-local_20260426_172317_qwen17b_6pct_L48}"
JOB_DIR="$SCRATCH/nl_output/search/job_${JOB_ID}"

# ========== gcloud + GCS auth (used both for restore and the live sync loop) ==========
if [ -d /home/ray/google-cloud-sdk/bin ]; then
    export PATH="/home/ray/google-cloud-sdk/bin:$PATH"
fi
if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -f "$REPO_DIR/gcs-bucket-sa.json" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$REPO_DIR/gcs-bucket-sa.json"
fi
if command -v gcloud >/dev/null 2>&1; then
    gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS" --quiet 2>/dev/null || true
fi

# ========== Pull cached job dir from GCS if not present locally ==========
GCS_BUCKET="${GCS_BUCKET:-jackierwzhang-purdue-research-curriculum}"
GCS_PREFIX="${GCS_PREFIX:-qwen17b-6pct-dolci-stage32-L48/job_${JOB_ID}}"
GCS_SRC="gs://${GCS_BUCKET}/${GCS_PREFIX}"

if [ ! -d "$JOB_DIR/checkpoint-32500" ]; then
    echo "=== No local checkpoint-32500 — restoring from GCS ==="
    echo "    src:  $GCS_SRC"
    echo "    dest: $JOB_DIR"
    mkdir -p "$JOB_DIR"
    gcloud storage rsync "$GCS_SRC" "$JOB_DIR" --recursive --no-ignore-symlinks 2>&1 | tail -5
else
    echo "=== Local job dir already populated, skipping GCS restore ==="
fi

# Sanity: pick the latest rolling checkpoint and report what we'll resume from
LATEST_CKPT=$(ls -1d "$JOB_DIR"/checkpoint-* 2>/dev/null | sed 's:.*/checkpoint-::' | sort -n | tail -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "[FATAL] No checkpoint-* dirs found under $JOB_DIR after GCS restore" >&2
    exit 1
fi
echo "=== Will auto-resume from $JOB_DIR/checkpoint-$LATEST_CKPT ==="
ls -la "$JOB_DIR/checkpoint-$LATEST_CKPT/" | head -25

# ========== Hardware ==========
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR="127.0.0.1"
MASTER_PORT="${MASTER_PORT:-$((10000 + ($(echo "$JOB_ID" | cksum | cut -d' ' -f1) % 50000)))}"
ulimit -n 131072 || true

# ==========================================================
#                      CONFIGURATION
# Identical to the original L=48 resume launcher.
# ==========================================================
TASK="search"
MODEL_NAME="Qwen/Qwen3-1.7B"

BATCH_SIZE=48
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=5e-5
WARMUP_STEPS=500
SEED=1234

N_STAGES=48
BASE_ALPHA=0.1
MAX_ALPHA=1.0
ACCURACY_THRESHOLD=0.98
MIN_STEPS_PER_STAGE=200
CHECK_EVERY=25
ACCURACY_WINDOW=200

MAX_INPUT_SIZE=288
MAX_LOOKAHEAD=48
BASE_LOOKAHEAD=1
LOOKAHEAD_STEP=1
MAX_FRONTIER_SIZE=12
MAX_BRANCH_SIZE=12
REQUESTED_BACKTRACK=3

GRADIENT_CHECKPOINTING=true
USE_LIGER=true
USE_CHUNKED_CE=true
CE_CHUNK_SIZE=4096

ARGS=(
    --task "$TASK"
    --model_name "$MODEL_NAME"
    --cache_dir "$HF_HOME"
    --output_dir "$SCRATCH/nl_output"
    --scratch_dir "$SCRATCH"
    --job_id "$JOB_ID"
    --resume_from_job "$JOB_ID"

    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --warmup_steps "$WARMUP_STEPS"
    --seed "$SEED"
    --num_shots 0
    --first_token_soft_weight 0.0

    --n_stages "$N_STAGES"
    --base_alpha "$BASE_ALPHA"
    --max_alpha "$MAX_ALPHA"
    --accuracy_threshold "$ACCURACY_THRESHOLD"
    --min_steps_per_stage "$MIN_STEPS_PER_STAGE"
    --check_every "$CHECK_EVERY"
    --accuracy_window "$ACCURACY_WINDOW"
    --eval_every_steps 0

    --max_input_size "$MAX_INPUT_SIZE"
    --max_lookahead "$MAX_LOOKAHEAD"
    --max_frontier_size "$MAX_FRONTIER_SIZE"
    --max_branch_size "$MAX_BRANCH_SIZE"
    --requested_backtrack "$REQUESTED_BACKTRACK"

    --eval_samples 500
    --print_eval_examples 0
    --save_total_limit 2

    --use_packing
    --linear_lookahead
    --base_lookahead "$BASE_LOOKAHEAD"
    --lookahead_step "$LOOKAHEAD_STEP"

    --mix_pretrain_data allenai/Dolci-Instruct-SFT
    --mix_pretrain_ratio 0.06
    --use_chat_template
)

$GRADIENT_CHECKPOINTING && ARGS+=(--gradient_checkpointing)
$USE_LIGER             && ARGS+=(--use_liger)
$USE_CHUNKED_CE        && ARGS+=(--use_chunked_ce --ce_chunk_size "$CE_CHUNK_SIZE")

echo "================================================="
echo "Qwen 1.7B 6%-mix RESUME FROM GCS — finish L=48"
echo "JOB_ID:              $JOB_ID"
echo "Effective batch:     $((GPUS_PER_NODE * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))  ($GPUS_PER_NODE × $BATCH_SIZE × $GRADIENT_ACCUMULATION_STEPS)"
echo "Resume from:         $JOB_DIR/checkpoint-$LATEST_CKPT (full optimizer.pt)"
echo "Curriculum target:   L=$MAX_LOOKAHEAD ($N_STAGES stages)"
echo "================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"

# Build C++ generator if needed
python -c "
try:
    import generator
    print('[OK] C++ generator present')
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, 'nl_generator.py'])
"

exec torchrun --nproc_per_node="$GPUS_PER_NODE" \
              --master_port="$MASTER_PORT" \
              --max_restarts=0 \
              tuning_nl.py "${ARGS[@]}"
