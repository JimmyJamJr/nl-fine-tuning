#!/bin/bash
# Qwen family scaling experiment: train each model size from scratch under our
# curriculum with s=1 to L=256, no compute cap. Hyperparameters below are the
# Qwen3-0.6B optimal config from the paper (recent Gautschi resume run); copy
# this script and change the four MODEL-SPECIFIC lines for each size.
#
# NOTE: Qwen3-0.6B data already exists from prior Gautschi run
# (job_10579765 / jackie_qwen06b_L256_step1_extend_resumed, reached L=231 over
# ~5 days). Don't re-run 0.6B; use the existing trajectory for the 0.6B point
# of the scaling experiment.
#
# Usage:
#   # set WORKDIR/CODE_DIR/HF cache paths to match Max's cluster, then:
#   bash qwen_family_step1_L256_FRESH_for_max.sh
#
# To run a different size: edit MODEL_NAME, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
# and (if needed) GRADIENT_CHECKPOINTING. Keep eff_batch = NUM_GPUS * BATCH_SIZE *
# GRADIENT_ACCUMULATION_STEPS == 768 to match the 0.6B paper config.

set -euo pipefail

############################################
# >>> EDIT THESE FOR YOUR CLUSTER <<<
WORKDIR="$HOME/qwen_scaling"                  # outputs, HF cache, triton cache live here
CODE_DIR="$WORKDIR/nl-fine-tuning/nl"         # where tuning_nl.py lives (clone this repo)
CONDA_ENV="search"                            # env with FA2/3, transformers, liger, chunked_ce
NUM_GPUS=4
SLURM_JOBID=""                                # if running inside a reservation, set to fire srun --overlap
MASTER_PORT=29501
############################################

############################################
# >>> EDIT THESE 4 PER MODEL SIZE <<<
# Defaults below replicate the Qwen3-0.6B paper config exactly (4 GPUs, bs=96,
# GA=2, GC=on, eff_batch=768). For 1.7B / 4B / 8B, Max will need to tune
# BATCH_SIZE and GRADIENT_ACCUMULATION_STEPS for memory while preserving
# eff_batch = NUM_GPUS * BATCH_SIZE * GA = 768.
MODEL_NAME="Qwen/Qwen3-0.6B"
BATCH_SIZE=96
GRADIENT_ACCUMULATION_STEPS=2
GRADIENT_CHECKPOINTING=true
############################################

# Common Qwen 0.6B optimal hypers (DON'T change these for the scaling experiment)
TASK="search"
LEARNING_RATE=5e-5
WARMUP_STEPS=500
SEED=1234
TARGET_MAX_LOOKAHEAD=256
MAX_INPUT_SIZE=1536                           # 6 * L_max
ACCURACY_THRESHOLD=0.98
MIN_STEPS_PER_STAGE=200
ACCURACY_WINDOW=800
CHECK_EVERY=25
N_STAGES=$TARGET_MAX_LOOKAHEAD                # linear lookahead → n_stages == L_max
BASE_LOOKAHEAD=1
LOOKAHEAD_STEP=1
BASE_ALPHA=0.1
MAX_ALPHA=1.0
MAX_FRONTIER_SIZE=12
MAX_BRANCH_SIZE=12
REQUESTED_BACKTRACK=3
SAVE_TOTAL_LIMIT=2
CE_CHUNK_SIZE=4096

# Generate a fresh job id from timestamp (so each rerun gets its own output dir)
JOB_ID="qwen_$(echo "$MODEL_NAME" | sed 's|Qwen/Qwen3-||;s|\.||g')_step1_L256_$(date +%Y%m%d_%H%M%S)"

# --- Setup ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="$WORKDIR/.hf_home"
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR="$WORKDIR/.triton_cache"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$WORKDIR/nl_output"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$(( $(nproc) / NUM_GPUS ))
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

cd "$CODE_DIR"

EFF_BATCH=$(( NUM_GPUS * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS ))
echo "=== FRESH Qwen scaling run ==="
echo "MODEL:      $MODEL_NAME"
echo "JOB_ID:     $JOB_ID"
echo "GPUs:       $NUM_GPUS"
echo "bs=$BATCH_SIZE  GA=$GRADIENT_ACCUMULATION_STEPS  GC=$GRADIENT_CHECKPOINTING  -> eff_batch=$EFF_BATCH"
echo "Target L:   $TARGET_MAX_LOOKAHEAD  (max_input_size=$MAX_INPUT_SIZE)"
echo "LR=$LEARNING_RATE  warmup=$WARMUP_STEPS  acc_thr=$ACCURACY_THRESHOLD  acc_win=$ACCURACY_WINDOW"
echo "No compute cap."
echo "Output: $WORKDIR/nl_output/$TASK/job_${JOB_ID}"
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

if [ -n "$SLURM_JOBID" ]; then
    LAUNCH=(srun --jobid="$SLURM_JOBID" --overlap --gres=gpu:"$NUM_GPUS" -N1 -n1 torchrun)
else
    LAUNCH=(torchrun)
fi

echo "Command: ${LAUNCH[*]} --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"
exec "${LAUNCH[@]}" --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" --max_restarts=0 \
    tuning_nl.py "${ARGS[@]}"
