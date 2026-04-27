#!/bin/bash
# ============================================================================
# Qwen 0.6B curriculum extension: L=128 → L=256 (step=16 continuation)
# ============================================================================
# RESUME from job_9273210 (completed L=128 at step=16 on 2026-04-16).
# Adds 8 more curriculum stages: L=144, 160, 176, 192, 208, 224, 240, 256.
#
# Source chain summary:
#   job_9273210 (Qwen3-0.6B, base/step=16, n_stages=8, completed 21,925 steps)
#   final stage_checkpoint = stage_8_step_21925_L128
#   latest optimizer state = checkpoint-21500/  (3.4 GB; this is what we resume)
#
# Tested on 4× H100 80GB. Should also work on:
#   - 4× A100 80GB (comfortable; max_input=1536 is the heaviest constraint)
#   - 4× A100 40GB (TIGHT — set GRADIENT_CHECKPOINTING=true and possibly BS=32 GA=6)
#   - 8× A100 40GB (use BS=24 GA=4 to keep eff_batch=768)
#
# Adjust SBATCH directives for your cluster's conventions (partition, QoS,
# --mem rules, account). Then submit with:
#   sbatch tuning_job_qwen06b_curr256_step16_RESUME_jackie.sh
# ============================================================================

#SBATCH -J nl_qwen06b_curr256_step16_RESUME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4                     # 4 GPUs recommended
#SBATCH --cpus-per-task=56               # scale down if your cluster has fewer
#SBATCH --time=48:00:00                  # 48h initial; extend or rely on requeue
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
# Uncomment / modify for YOUR cluster:
# #SBATCH --partition=YOUR_PARTITION
# #SBATCH -A YOUR_ACCOUNT
# #SBATCH -q YOUR_QOS
# #SBATCH --mem=240G                     # Gilbreth requires; Gautschi rejects

set -euo pipefail

# ========== Preemption handling ==========
trap 'echo "[SIG] USR1 @ $(date) — grace period"; sleep 120; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
trap 'echo "[SIG] TERM @ $(date)"; exit 0' TERM

mkdir -p ./slurm
echo "JOB START $(date)"

# ========== Environment ==========
module load conda
conda activate search                    # or your env name

# Adjust SCRATCH for your cluster (e.g. /scratch/$USER, /scratch/gilbreth/$USER)
export SCRATCH="${SCRATCH:-/scratch/$USER}"
mkdir -p "$SCRATCH/nl_output" "$SCRATCH/model_cache" "$SCRATCH/triton_cache"
export HF_HOME="$SCRATCH/model_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
if [ -f "$(dirname "$0")/.env" ]; then source "$(dirname "$0")/.env"; fi
export HF_HUB_OFFLINE=0                  # need online for first-time model download
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ========== Hardware ==========
GPUS_PER_NODE=$(echo $SLURM_JOB_GPUS | tr "," "\n" | wc -l)
[ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
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
#                      CONFIGURATION
# ==========================================================

# Task / Model
TASK="search"
MODEL_NAME="Qwen/Qwen3-0.6B"

# Resume from HuggingFace-hosted checkpoint (see HANDOFF.md)
# We download the checkpoint into $SCRATCH/nl_output/search/job_9273210/ so that
# plot_training.py --combined can walk the resume chain back into the original
# training history (loss_history, run_meta).
HF_REPO_ID="JimmyJamJr/qwen06b-step16-L128-resume"
RESUME_JOB_ID=9273210
RESUME_DIR="$SCRATCH/nl_output/search/job_${RESUME_JOB_ID}"

# Training — eff_batch=768 matches original step=16 run (job_9273210)
BATCH_SIZE=48                            # per-GPU batch
GRADIENT_ACCUMULATION_STEPS=4            # 4 GPU × 48 × 4 = 768 eff_batch
LEARNING_RATE=5e-5                       # same as original
WARMUP_STEPS=100                         # short warmup since resuming converged state
SEED=1234
NUM_SHOTS=0
FIRST_TOKEN_SOFT_WEIGHT=0.0

# Curriculum — extending L=128 → L=256 with step=16
# Original was n_stages=8 (L=16,32,...,128). New target needs 16 stages
# (L=16,32,...,256). On resume, the curriculum bookkeeping inside the checkpoint
# knows we're at stage 8 / L=128 and will continue from stage 9 (L=144) under
# the new n_stages=16 schedule.
N_STAGES=16
BASE_ALPHA=0.1
MAX_ALPHA=1.0
ACCURACY_THRESHOLD=0.98
MIN_STEPS_PER_STAGE=200
CHECK_EVERY=25
ACCURACY_WINDOW=200
EVAL_EVERY_STEPS=0                       # disable midstage eval; rely on stage_eval

# Task parameters — target L=256
MAX_INPUT_SIZE=1536                      # CRITICAL: 6 * L = 6 * 256 = 1536
MAX_LOOKAHEAD=256
BASE_LOOKAHEAD=16                        # match original
LOOKAHEAD_STEP=16                        # match original
MAX_FRONTIER_SIZE=12
MAX_BRANCH_SIZE=12
REQUESTED_BACKTRACK=3

# Memory optimizations
# At max_input=1536, activations are 2× larger than the original (max_input=768).
# Recommend GRADIENT_CHECKPOINTING=true unless you have 80GB+ GPUs with room.
GRADIENT_CHECKPOINTING=true              # switch to false if VRAM allows
USE_LIGER=true                           # works for Qwen
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
STAGE_EVAL_EVERY=8

# ==========================================================

echo "Task: $TASK | Model: $MODEL_NAME | Max Input: $MAX_INPUT_SIZE | Target L: $MAX_LOOKAHEAD"
echo "Resume from: $RESUME_DIR/checkpoint-21500 (HF: $HF_REPO_ID)"
echo "Output: $SCRATCH/nl_output/$TASK/job_${SLURM_JOB_ID}"

# Download resume checkpoint from HuggingFace if not already present
if [ ! -d "$RESUME_DIR/checkpoint-21500" ]; then
    echo "Downloading checkpoint from HuggingFace: $HF_REPO_ID"
    mkdir -p "$RESUME_DIR"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$HF_REPO_ID',
    local_dir='$RESUME_DIR',
    local_dir_use_symlinks=False,
)
print('[OK] Downloaded to $RESUME_DIR')
"
fi
if [ ! -d "$RESUME_DIR/checkpoint-21500" ]; then
    echo "ERROR: download failed; expected $RESUME_DIR/checkpoint-21500"
    echo "If repo is private, run: huggingface-cli login (with read token)"
    exit 1
fi

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

# ========== Cache model (if not already) ==========
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
    --job_id "${JOB_ID_OVERRIDE:-$SLURM_JOB_ID}"

    # Training
    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --warmup_steps "$WARMUP_STEPS"
    --seed "$SEED"
    --num_shots "$NUM_SHOTS"
    --first_token_soft_weight "$FIRST_TOKEN_SOFT_WEIGHT"

    # Curriculum
    --n_stages "$N_STAGES"
    --base_alpha "$BASE_ALPHA"
    --max_alpha "$MAX_ALPHA"
    --accuracy_threshold "$ACCURACY_THRESHOLD"
    --min_steps_per_stage "$MIN_STEPS_PER_STAGE"
    --check_every "$CHECK_EVERY"
    --accuracy_window "$ACCURACY_WINDOW"
    --eval_every_steps "$EVAL_EVERY_STEPS"

    # Task parameters
    --max_input_size "$MAX_INPUT_SIZE"
    --max_lookahead "$MAX_LOOKAHEAD"
    --max_frontier_size "$MAX_FRONTIER_SIZE"
    --max_branch_size "$MAX_BRANCH_SIZE"
    --requested_backtrack "$REQUESTED_BACKTRACK"

    # Evaluation
    --eval_samples "$EVAL_SAMPLES"
    --print_eval_examples "$PRINT_EVAL_EXAMPLES"
    --stage_eval_every "$STAGE_EVAL_EVERY"

    --use_packing
    --linear_lookahead
    --base_lookahead "$BASE_LOOKAHEAD"
    --lookahead_step "$LOOKAHEAD_STEP"

    # Resume by job_id — points at $SCRATCH/nl_output/search/job_9273210/checkpoint-21500/
    # which we populated above from HuggingFace. Using --resume_from_job (not --resume_from_path)
    # ensures plot_training.py --combined can walk the resume chain back into the original
    # training history for cumulative loss/compute plots.
    --resume_from_job "$RESUME_JOB_ID"
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
MAX_RETRIES=10
RETRY_COUNT=0
JOB_OUTPUT_DIR="$SCRATCH/nl_output/$TASK/job_${SLURM_JOB_ID}"
RESTART_FLAG="$JOB_OUTPUT_DIR/RESTART_FLAG"

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
