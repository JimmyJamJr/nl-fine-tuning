#!/bin/bash
#SBATCH -J nl_search_1536_1.7B_la256_a1.0_timing_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --partition=ai
#SBATCH -A asaparov
#SBATCH -q normal
#SBATCH --mail-user=huan2073@purdue.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

set -euo pipefail

# ========== Preemption handling ==========
trap 'echo "[SIG] USR1 @ $(date) — grace period"; sleep 120; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
trap 'echo "[SIG] TERM @ $(date)"; exit 0' TERM

mkdir -p ./slurm
echo "JOB START $(date)"

# ========== Environment ==========
module load conda
conda activate search

export SCRATCH="/scratch/gautschi/$USER"
mkdir -p "$SCRATCH/nl_output" "$SCRATCH/model_cache"
export HF_HOME="$SCRATCH/model_cache"
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
#                    CONFIGURATION
# ==========================================================

# Task / Model
TASK="search"                    # si | dfs | search
MODEL_NAME="Qwen/Qwen3-1.7B"

# Training
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=3e-5
WARMUP_STEPS=500
SEED=1234
NUM_SHOTS=0                      # 0, 1, or 2

# LoRA
USE_LORA=true
LORA_RANK=16
LORA_DROPOUT=0.10

# Curriculum
N_STAGES=1
BASE_ALPHA=1.0
MAX_ALPHA=1.0                    # Cap training alpha (eval always uses 1.0)
ACCURACY_THRESHOLD=0.98
MIN_STEPS_PER_STAGE=500
CHECK_EVERY=50
FIRST_TOKEN_SOFT_WEIGHT=1.0

# Task parameters
MAX_INPUT_SIZE=1536
MAX_LOOKAHEAD=256                # search
MAX_FRONTIER_SIZE=12             # si
MAX_BRANCH_SIZE=12               # si
REQUESTED_BACKTRACK=3            # dfs

# Memory optimizations
GRADIENT_CHECKPOINTING=true
USE_LIGER=true                   # Fused RoPE/RMSNorm/SwiGLU kernels
USE_CHUNKED_CE=true              # Memory-efficient cross-entropy
CE_CHUNK_SIZE=1024

# OOM handling
OOM_AUTOSCALE=true

# Evaluation
EVAL_SAMPLES=500
PRINT_EVAL_EXAMPLES=0
DO_BASELINE=false                 # Pre-training baseline accuracy
DO_FINAL_EVAL=false               # Post-training TF + greedy eval
DO_REDACTED_EVAL=false            # Redacted sanity check (should be low)
DO_SEEN_EVAL=false                # Seen-samples sanity check (should be ~100%)
DO_STAGE_EVAL=false              # Eval at α=1.0 after each stage advancement

# Resume from previous job (leave empty for fresh start)
PREV_JOB_ID=""

# ==========================================================

echo "Task: $TASK | Model: $MODEL_NAME | Max Input: $MAX_INPUT_SIZE"
echo "Output: $SCRATCH/nl_output/$TASK/job_${SLURM_JOB_ID}"
[ -n "$PREV_JOB_ID" ] && echo "Resuming from: job_$PREV_JOB_ID"

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
    --job_id "$SLURM_JOB_ID"

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

    # Task parameters
    --max_input_size "$MAX_INPUT_SIZE"
    --max_lookahead "$MAX_LOOKAHEAD"
    --max_frontier_size "$MAX_FRONTIER_SIZE"
    --max_branch_size "$MAX_BRANCH_SIZE"
    --requested_backtrack "$REQUESTED_BACKTRACK"

    # Evaluation
    --eval_samples "$EVAL_SAMPLES"
    --print_eval_examples "$PRINT_EVAL_EXAMPLES"

    --use_packing
    --pack_length 16384
    --target_samples_per_batch 48
)

# Conditional flags
$USE_LORA && ARGS+=(--use_lora --lora_rank "$LORA_RANK" --lora_dropout "$LORA_DROPOUT")
$GRADIENT_CHECKPOINTING && ARGS+=(--gradient_checkpointing)
$USE_LIGER && ARGS+=(--use_liger)
$USE_CHUNKED_CE && ARGS+=(--use_chunked_ce --ce_chunk_size "$CE_CHUNK_SIZE")
$OOM_AUTOSCALE && ARGS+=(--oom_autoscale)
$DO_BASELINE && ARGS+=(--do_baseline)
$DO_FINAL_EVAL && ARGS+=(--do_final_eval)
$DO_REDACTED_EVAL && ARGS+=(--do_redacted_eval)
$DO_SEEN_EVAL && ARGS+=(--do_seen_eval)
$DO_STAGE_EVAL && ARGS+=(--do_stage_eval)
[ -n "$PREV_JOB_ID" ] && ARGS+=(--resume_from_job "$PREV_JOB_ID")

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT qwen_tuning_nl_multi.py ${ARGS[*]}"

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
             qwen_tuning_nl_multi.py "${ARGS[@]}"
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