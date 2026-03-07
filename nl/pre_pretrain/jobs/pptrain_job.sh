#!/bin/bash
#SBATCH -J pptrain--lr1e-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --partition=ai
#SBATCH -A asaparov
#SBATCH --mail-user=mnickel@purdue.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
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
echo "Job ID: $SLURM_JOB_ID"

# ========== Environment ==========
module load conda
source /home/mnickel/miniconda3/etc/profile.d/conda.sh
conda activate pptrain

export SCRATCH="/scratch/gautschi/$USER"
mkdir -p "$SCRATCH/model_cache"
export HF_HOME="$SCRATCH/model_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ========== Hardware ==========
GPUS_PER_NODE=$(echo $SLURM_JOB_GPUS | tr "," "\n" | wc -l)
[ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=8

export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
ulimit -n 131072 || true

# ========== CUDA / Torch ==========
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# ========== Distributed ==========
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "GPUS=$GPUS_PER_NODE  OMP_THREADS=$OMP_NUM_THREADS"
nvidia-smi || true

# ==========================================================
#                    CONFIGURATION
# ==========================================================
# All pptrain.py arguments are listed here for easy modification.
# ==========================================================

# Run configuration
RUN_NAME="pptrain--lr1e-4"
DATA_DIR="/scratch/gautschi/mnickel/data/nl_splits"

# Model
MODEL_NAME="EleutherAI/pythia-160m"
SEED=1337

# Training limits (empty = run until curriculum complete)
MAX_STEPS=""

# Batch configuration
SEQ_LEN=2048
MICRO_BATCH=16
GRAD_ACCUM=4

# Optimizer
LR="1e-4"
WEIGHT_DECAY="0.1"
ADAM_BETA1="0.9"
ADAM_BETA2="0.999"
ADAM_EPS="1e-6"
GRAD_CLIP="1.0"
WARMUP_STEPS=500

# Precision and checkpointing
MIXED_PRECISION="bf16"
SAVE_EVERY=200

# Curriculum settings (max_input_size = 6 * MAX_LOOKAHEAD)
MAX_LOOKAHEAD=32
START_STAGE=4
STAGE_STEP=4
CURR_WINDOW=1000
CURR_THRESHOLD="0.98"

# ==========================================================

echo "=========================================="
echo "PRE-PRETRAINING: $RUN_NAME"
echo "Run dir: $SCRATCH/$RUN_NAME"
echo "Data dir: $DATA_DIR"
echo "Model: $MODEL_NAME"
echo "Max lookahead: $MAX_LOOKAHEAD (max_input_size=$(( 6 * MAX_LOOKAHEAD )))"
echo "Batch: micro=$MICRO_BATCH grad_accum=$GRAD_ACCUM seq_len=$SEQ_LEN"
echo "Optimizer: lr=$LR warmup=$WARMUP_STEPS"
echo "Curriculum: start_stage=$START_STAGE step=$STAGE_STEP threshold=$CURR_THRESHOLD"
echo "=========================================="

# ========== Build C++ generator if needed ==========
cd /home/$USER/git/nl-fine-tuning/nl
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

# ========== Find latest checkpoint for auto-resume ==========
CKPT_DIR="$SCRATCH/$RUN_NAME/checkpoints"
LATEST_CKPT=""
if [ -d "$CKPT_DIR" ]; then
    LATEST_CKPT=$(ls -d "$CKPT_DIR"/step_* 2>/dev/null | sort | tail -n1 || true)
fi

if [ -n "$LATEST_CKPT" ]; then
    echo "Found checkpoint for resume: $LATEST_CKPT"
else
    echo "No checkpoint found, starting fresh"
fi

# ========== Build arguments ==========
ARGS=(
    --scratch_dir "$SCRATCH"
    --run_name "$RUN_NAME"
    --data_dir "$DATA_DIR"
    --model_name "$MODEL_NAME"
    --seed "$SEED"
    --seq_len "$SEQ_LEN"
    --micro_batch "$MICRO_BATCH"
    --grad_accum "$GRAD_ACCUM"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --adam_beta1 "$ADAM_BETA1"
    --adam_beta2 "$ADAM_BETA2"
    --adam_eps "$ADAM_EPS"
    --grad_clip "$GRAD_CLIP"
    --warmup_steps "$WARMUP_STEPS"
    --mixed_precision "$MIXED_PRECISION"
    --save_every "$SAVE_EVERY"
    --max_lookahead "$MAX_LOOKAHEAD"
    --start_stage "$START_STAGE"
    --stage_step "$STAGE_STEP"
    --curr_window "$CURR_WINDOW"
    --curr_threshold "$CURR_THRESHOLD"
)

# Add max_steps if specified
if [ -n "$MAX_STEPS" ]; then
    ARGS+=(--max_steps "$MAX_STEPS")
fi

# Add resume flag if checkpoint exists
if [ -n "$LATEST_CKPT" ]; then
    ARGS+=(--resume_from "$LATEST_CKPT")
fi

echo "Command: accelerate launch --num_processes=$GPUS_PER_NODE pptrain.py ${ARGS[*]}"

# ========== Run training ==========
cd /home/$USER/git/nl-fine-tuning/nl

accelerate launch \
    --num_processes=$GPUS_PER_NODE \
    pptrain.py "${ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS $(date)"
else
    echo "FAILED with exit code $EXIT_CODE $(date)"
fi

exit $EXIT_CODE
