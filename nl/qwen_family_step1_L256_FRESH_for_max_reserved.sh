#!/bin/bash
# Reserved-nodes variant of qwen_family_step1_L256_FRESH_for_max.sh.
#
# Use this when you have already allocated a SLURM reservation (e.g. via a
# long-running holder job like nl_long_training.sh) and want to fire one or
# more training runs into it without going through the queue each time. Good
# for hyperparameter sweeps: launch the reservation once, then run this
# script repeatedly with different BATCH_SIZE / GA / LR.
#
# Usage:
#   # 1. Allocate a reservation (any equivalent of nl_long_training.sh works
#   #    -- a long sbatch job that just sleeps and holds your GPUs). Note the
#   #    job ID it returns.
#   #
#   # 2. Set required env vars and run:
#   export RESID=12345678                          # the holder job's SLURM ID
#   export WORKDIR=$HOME/qwen_scaling              # outputs + caches
#   export CODE_DIR=$WORKDIR/nl-fine-tuning/nl     # where tuning_nl.py lives
#   export CONDA_ENV=search
#   export NUM_GPUS=4                              # GPUs the reservation holds
#   bash qwen_family_step1_L256_FRESH_for_max_reserved.sh
#
#   # 3. For a sweep, override per-run hypers and give each run a unique tag:
#   RUN_TAG=lr1e4    LEARNING_RATE=1e-4 bash qwen_family_step1_L256_FRESH_for_max_reserved.sh
#   RUN_TAG=bs48ga4  BATCH_SIZE=48 GRADIENT_ACCUMULATION_STEPS=4 bash ... (eff_batch preserved)
#   RUN_TAG=qwen17b  MODEL_NAME=Qwen/Qwen3-1.7B BATCH_SIZE=24 GRADIENT_ACCUMULATION_STEPS=8 bash ...
#
# Defaults below replicate the Qwen3-0.6B paper config exactly. Override via
# env vars; the script never edits them inline so the file stays clean across
# sweeps. Keep eff_batch = NUM_GPUS * BATCH_SIZE * GA = 768 to match the
# 0.6B trajectory you're scaling against.

set -euo pipefail

############################################
# Required / cluster-level (must be set in the caller's env)
: "${RESID:?Set RESID to the SLURM job ID of your reservation holder}"
: "${WORKDIR:=$HOME/qwen_scaling}"
: "${CODE_DIR:=$WORKDIR/nl-fine-tuning/nl}"
: "${CONDA_ENV:=search}"
: "${NUM_GPUS:=4}"
: "${MASTER_PORT:=29501}"
############################################

############################################
# Per-run knobs (override via env for sweeps; defaults = 0.6B paper config)
: "${MODEL_NAME:=Qwen/Qwen3-0.6B}"
: "${BATCH_SIZE:=96}"
: "${GRADIENT_ACCUMULATION_STEPS:=2}"
: "${GRADIENT_CHECKPOINTING:=true}"
: "${LEARNING_RATE:=5e-5}"
: "${WARMUP_STEPS:=500}"
: "${SEED:=1234}"
: "${RUN_TAG:=}"                              # optional suffix to disambiguate sweep runs
############################################

# Curriculum hypers locked to the 0.6B paper config — don't override these
# during a scaling experiment.
TASK="search"
TARGET_MAX_LOOKAHEAD=256
MAX_INPUT_SIZE=1536                           # 6 * L_max
ACCURACY_THRESHOLD=0.98
MIN_STEPS_PER_STAGE=200
ACCURACY_WINDOW=800
CHECK_EVERY=25
N_STAGES=$TARGET_MAX_LOOKAHEAD
BASE_LOOKAHEAD=1
LOOKAHEAD_STEP=1
BASE_ALPHA=0.1
MAX_ALPHA=1.0
MAX_FRONTIER_SIZE=12
MAX_BRANCH_SIZE=12
REQUESTED_BACKTRACK=3
SAVE_TOTAL_LIMIT=2
CE_CHUNK_SIZE=4096

# Unique job id per fire; include RUN_TAG so sweep runs don't collide.
MODEL_SHORT=$(echo "$MODEL_NAME" | sed 's|Qwen/Qwen3-||;s|\.||g')
SUFFIX="${RUN_TAG:+_${RUN_TAG}}"
JOB_ID="qwen_${MODEL_SHORT}_step1_L256${SUFFIX}_$(date +%Y%m%d_%H%M%S)"

# --- Setup ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="$WORKDIR/.hf_home"
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR="$WORKDIR/.triton_cache"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$WORKDIR/nl_output" "$CODE_DIR/slurm"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$(( $(nproc) / NUM_GPUS ))
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

cd "$CODE_DIR"

EFF_BATCH=$(( NUM_GPUS * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS ))
echo "=== Reserved-node Qwen scaling fire ==="
echo "Reservation: $RESID"
echo "MODEL:       $MODEL_NAME"
echo "JOB_ID:      $JOB_ID"
echo "GPUs:        $NUM_GPUS"
echo "bs=$BATCH_SIZE  GA=$GRADIENT_ACCUMULATION_STEPS  GC=$GRADIENT_CHECKPOINTING  -> eff_batch=$EFF_BATCH"
echo "LR=$LEARNING_RATE  warmup=$WARMUP_STEPS  seed=$SEED  W=$ACCURACY_WINDOW"
echo "Target L:    $TARGET_MAX_LOOKAHEAD  (max_input_size=$MAX_INPUT_SIZE)"
echo "Output:      $WORKDIR/nl_output/$TASK/job_${JOB_ID}"
echo ""

ARGS=(
    --task "$TASK"
    --model_name "$MODEL_NAME"
    --cache_dir "$HF_HOME"
    --output_dir "$WORKDIR/nl_output"
    --scratch_dir "$WORKDIR"
    --job_id "$JOB_ID"
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
    --max_lookahead "$TARGET_MAX_LOOKAHEAD"
    --base_lookahead "$BASE_LOOKAHEAD"
    --lookahead_step "$LOOKAHEAD_STEP"
    --max_frontier_size "$MAX_FRONTIER_SIZE"
    --max_branch_size "$MAX_BRANCH_SIZE"
    --requested_backtrack "$REQUESTED_BACKTRACK"
    --eval_samples 500
    --print_eval_examples 0
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --ce_chunk_size "$CE_CHUNK_SIZE"
    --use_packing
    --linear_lookahead
    --use_liger
    --use_chunked_ce
)
$GRADIENT_CHECKPOINTING && ARGS+=(--gradient_checkpointing)

LAUNCH=(srun --jobid="$RESID" --overlap --gres=gpu:"$NUM_GPUS" -N1 -n1 torchrun)

echo "Command: ${LAUNCH[*]} --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"
exec "${LAUNCH[@]}" --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" --max_restarts=0 \
    tuning_nl.py "${ARGS[@]}"
