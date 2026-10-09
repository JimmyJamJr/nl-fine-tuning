#!/bin/bash
# Resume Qwen3-0.6B REINIT from job_14110834 latest checkpoint on 8 GPUs.
#
# Hypers preserved: bs=96 GA=2 on 8 GPUs -> eff_batch=1536, LR=3e-4, GC=on.
# This is the SAME eff_batch as the 4-GPU segment it continues (bs=96 GA=4 on
# 4 GPUs). Per-rank batch is unchanged at 96, so per-GPU memory and kernel
# shapes are identical to the previous segment; only the number of accumulation
# steps halves. Nothing else about the training dynamics changes.
# Latest ckpt: job_14110834 (stage 57, L=57) -- the arm is still in its
# near-linear regime, so the fitted Weibull ceiling is not yet identified
# (95% profile interval [263, 32047]); this segment is to extend the curve.
#SBATCH -J nl_qwen06b_REINIT_L256_resume
#SBATCH -o slurm/%j_%x.out
#SBATCH -e slurm/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=112
#SBATCH --time=2-00:00:00
#SBATCH -A asaparov
#SBATCH -p ai
#SBATCH -q preemptible
#SBATCH --signal=B:USR1@180
#SBATCH --exclude=h009

set -euo pipefail
module load conda
conda activate search

export SCRATCH="/scratch/gautschi/$USER"
export HF_HOME="$SCRATCH/model_cache"
export HF_HUB_OFFLINE=1
# Isolated, freshly-wiped Triton cache to avoid the recurring stale 'cubin' KeyError
export TRITON_CACHE_DIR="$SCRATCH/triton_cache_qwen_reinit"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
rm -rf "$TRITON_CACHE_DIR"
mkdir -p slurm "$TRITON_CACHE_DIR" "$SCRATCH/model_cache"
export PYTHONUNBUFFERED=1

cd /home/huan2073/nl-fine-tuning/nl

PREV_JOB_ID="14110834"
TASK="search"
MODEL_NAME="Qwen/Qwen3-0.6B"
BATCH_SIZE=96
GRADIENT_ACCUMULATION_STEPS=2       # 8 GPUs: 96*2*8 = 1536, same eff_batch as bs=96 GA=4 on 4 GPUs
LEARNING_RATE=3e-4
TARGET_MAX_LOOKAHEAD=256
MAX_INPUT_SIZE=1536

GPUS_PER_NODE=$(echo "$SLURM_JOB_GPUS" | tr "," "\n" | wc -l)
[ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "Resume Qwen 0.6B REINIT from $PREV_JOB_ID on 8 GPUs"
echo "GPUs=$GPUS_PER_NODE  bs=$BATCH_SIZE  GA=$GRADIENT_ACCUMULATION_STEPS  -> eff_batch=$(( BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * GPUS_PER_NODE )) (target 1536)"
echo "LR=$LEARNING_RATE  target L=$TARGET_MAX_LOOKAHEAD"
echo "Output: $SCRATCH/nl_output/$TASK/job_${SLURM_JOB_ID}"

ARGS=(
    --task "$TASK"
    --model_name "$MODEL_NAME"
    --cache_dir "$HF_HOME"
    --output_dir "$SCRATCH/nl_output"
    --scratch_dir "$SCRATCH"
    --job_id "$SLURM_JOB_ID"
    --resume_from_job "$PREV_JOB_ID"
    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --seed 1234
    --first_token_soft_weight 0.0
    --n_stages "$TARGET_MAX_LOOKAHEAD"
    --base_alpha 0.1
    --max_alpha 1.0
    --accuracy_threshold 0.98
    --min_steps_per_stage 200
    --check_every 25
    --accuracy_window 800
    --eval_every_steps 0
    --max_input_size "$MAX_INPUT_SIZE"
    --max_lookahead "$TARGET_MAX_LOOKAHEAD"
    --base_lookahead 1
    --lookahead_step 1
    --eval_samples 500
    --print_eval_examples 0
    --save_total_limit 2
    --ce_chunk_size 4096
    --linear_lookahead
    --use_liger
    --gradient_checkpointing
)

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"
torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}"
