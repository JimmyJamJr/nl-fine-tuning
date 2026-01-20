#!/bin/bash
#SBATCH -J pptrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=ai
#SBATCH -A asaparov
#SBATCH -q preemptible
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
conda activate pptrain

export SCRATCH="/scratch/gautschi/$USER"
mkdir -p "$SCRATCH/pptrain" "$SCRATCH/model_cache"
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

# Model
MODEL_NAME="EleutherAI/pythia-160m"

# Training
MICRO_BATCH=2
GRAD_ACCUM=2
LR=5e-4
WARMUP=125
MAX_STEPS=500000
SEQ_LEN=1024
MIXED_PRECISION="bf16"

# Curriculum
CURR_THRESHOLD=0.98
CURR_WINDOW=200
CURR_CHECK_EVERY=200
MAX_LOOKAHEAD=64

# Data
TASK="search"
MAX_INPUT_SIZE=256
DATA_DIR="$SCRATCH/pptrain/data"

# Checkpointing
SAVE_EVERY=200
LOG_EVERY=10

# Evaluation
NUM_EVAL=500

# ==========================================================

echo "=========================================="
echo "PRE-PRETRAINING (pptrain)"
echo "Model: $MODEL_NAME"
echo "Effective batch size: $((MICRO_BATCH * GRAD_ACCUM * GPUS_PER_NODE))"
echo "Scratch: $SCRATCH"
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
CKPT_DIR="$SCRATCH/pptrain/checkpoints"
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
    --model_name "$MODEL_NAME"
    --scratch_dir "$SCRATCH"
    --data_dir "$DATA_DIR"

    # Training
    --micro_batch "$MICRO_BATCH"
    --grad_accum "$GRAD_ACCUM"
    --lr "$LR"
    --warmup "$WARMUP"
    --max_steps "$MAX_STEPS"
    --seq_len "$SEQ_LEN"
    --mixed_precision "$MIXED_PRECISION"

    # Curriculum
    --curr_threshold "$CURR_THRESHOLD"
    --curr_window "$CURR_WINDOW"
    --curr_check_every "$CURR_CHECK_EVERY"
    --max_lookahead "$MAX_LOOKAHEAD"

    # Data generation
    --task "$TASK"
    --max_input_size "$MAX_INPUT_SIZE"

    # Checkpointing
    --save_every "$SAVE_EVERY"
    --log_every "$LOG_EVERY"

    # Evaluation
    --num_eval "$NUM_EVAL"
)

# Add resume flag if checkpoint exists
if [ -n "$LATEST_CKPT" ]; then
    ARGS+=(--resume_from "$LATEST_CKPT")
fi

echo "Command: accelerate launch --num_processes=$GPUS_PER_NODE pre_pretrain/pptrain.py ${ARGS[*]}"

# ========== Run training ==========
cd /home/$USER/git/nl-fine-tuning/nl

accelerate launch \
    --num_processes=$GPUS_PER_NODE \
    --mixed_precision="$MIXED_PRECISION" \
    pre_pretrain/pptrain.py "${ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS $(date)"
else
    echo "FAILED with exit code $EXIT_CODE $(date)"
fi

exit $EXIT_CODE
