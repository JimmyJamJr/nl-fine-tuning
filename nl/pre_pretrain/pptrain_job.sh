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
source /home/mnickel/miniconda3/etc/profile.d/conda.sh
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
# Only specify values that differ from pptrain.py defaults
# Defaults: model=pythia-160m, micro_batch=2, grad_accum=2,
#           lr=5e-4, warmup_steps=1000, seq_len=2048, etc.

MAX_STEPS=10000000000
DATA_DIR="/scratch/gautschi/mnickel/data/nl_splits/mis384_look64_seed12345"

# ==========================================================

echo "=========================================="
echo "PRE-PRETRAINING (pptrain)"
echo "Data dir: $DATA_DIR"
echo "Max steps: $MAX_STEPS"
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
m, c = 'EleutherAI/pythia-160m', os.environ['HF_HOME']
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

# ========== Build arguments (minimal - use defaults) ==========
ARGS=(
    --scratch_dir "$SCRATCH"
    --data_dir "$DATA_DIR"
    --max_steps "$MAX_STEPS"
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
    pre_pretrain/pptrain.py "${ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS $(date)"
else
    echo "FAILED with exit code $EXIT_CODE $(date)"
fi

exit $EXIT_CODE
