#!/bin/bash
#SBATCH -J pretrain
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
mkdir -p "$SCRATCH/pretrain" "$SCRATCH/model_cache"
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

# Model (will load from pptrain checkpoint)
MODEL_NAME="EleutherAI/pythia-160m"

# Training
MICRO_BATCH=2
GRAD_ACCUM=2
LR=5e-4
MIN_LR_RATIO=0.1
WARMUP_STEPS=1000
TOTAL_STEPS=10000
SEQ_LEN=2048
MIXED_PRECISION="bf16"

# Optimizer
WEIGHT_DECAY=0.1
ADAM_BETA1=0.9
ADAM_BETA2=0.999
ADAM_EPS=1e-6
GRAD_CLIP=1.0

# Data
DATA_DIR="$SCRATCH/pptrain/data"
MIXIN_PERCENT=0.02

# Checkpointing
SAVE_EVERY=500
LOG_EVERY=10
EVAL_EVERY=500
EVAL_SUBSET_SIZE=500

# Pre-pretraining checkpoint ("latest" auto-finds most recent)
PPTRAIN_CHECKPOINT="latest"

# ==========================================================

# ========== Find latest checkpoint for auto-resume ==========
CKPT_DIR="$SCRATCH/pretrain/checkpoints"
LATEST_CKPT=""
if [ -d "$CKPT_DIR" ]; then
    LATEST_CKPT=$(ls -d "$CKPT_DIR"/step_* 2>/dev/null | sort | tail -n1 || true)
fi

if [ -n "$LATEST_CKPT" ]; then
    echo "Found checkpoint for resume: $LATEST_CKPT"
else
    echo "No checkpoint found, starting fresh from pptrain checkpoint"
fi

echo "=========================================="
echo "PRETRAINING (pretrain)"
echo "Model: $MODEL_NAME"
echo "PPTrain checkpoint: $PPTRAIN_CHECKPOINT"
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

# ========== Build arguments ==========
ARGS=(
    --scratch_dir "$SCRATCH"
    --model_name "$MODEL_NAME"
    --pptrain_checkpoint "$PPTRAIN_CHECKPOINT"
    --data_dir "$DATA_DIR"

    # Training
    --micro_batch "$MICRO_BATCH"
    --grad_accum "$GRAD_ACCUM"
    --lr "$LR"
    --min_lr_ratio "$MIN_LR_RATIO"
    --warmup_steps "$WARMUP_STEPS"
    --total_steps "$TOTAL_STEPS"
    --seq_len "$SEQ_LEN"
    --mixed_precision "$MIXED_PRECISION"

    # Optimizer
    --weight_decay "$WEIGHT_DECAY"
    --adam_beta1 "$ADAM_BETA1"
    --adam_beta2 "$ADAM_BETA2"
    --adam_eps "$ADAM_EPS"
    --grad_clip "$GRAD_CLIP"

    # Data
    --mixin_percent "$MIXIN_PERCENT"

    # Checkpointing & Eval
    --save_every "$SAVE_EVERY"
    --log_every "$LOG_EVERY"
    --eval_every "$EVAL_EVERY"
    --eval_subset_size "$EVAL_SUBSET_SIZE"
)

# Add resume flag if checkpoint exists
if [ -n "$LATEST_CKPT" ]; then
    ARGS+=(--resume_from "$LATEST_CKPT")
fi

echo "Command: accelerate launch --num_processes=$GPUS_PER_NODE pre_pretrain/pretrain.py ${ARGS[*]}"

# ========== Run training ==========
cd /home/$USER/git/nl-fine-tuning/nl

accelerate launch \
    --num_processes=$GPUS_PER_NODE \
    --mixed_precision="$MIXED_PRECISION" \
    pre_pretrain/pretrain.py "${ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS $(date)"
else
    echo "FAILED with exit code $EXIT_CODE $(date)"
fi

exit $EXIT_CODE
