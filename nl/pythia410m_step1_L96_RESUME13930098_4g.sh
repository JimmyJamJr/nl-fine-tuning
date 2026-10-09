#!/bin/bash
# FRESH Pythia-410M s=1 curriculum run (no resume, start from base checkpoint).
# Mirrors original hypers from job_9273225 (paper-canonical 410M run): bs=48 GA=2
# on 2 GPUs (eff=192), LR=5e-5, GC=off, no liger, max_input_size=576, max_L=96.
# 4x H100 80GB: bs=48 GA=1 GC=off (eff=192 preserved, fewer accum steps).
#SBATCH -J nl_pythia410m_step1_L96_fresh
#SBATCH -o slurm/%j_%x.out
#SBATCH -e slurm/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=3-00:00:00
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
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
mkdir -p slurm "$SCRATCH/triton_cache" "$SCRATCH/model_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd /home/huan2073/nl-fine-tuning/nl

PREV_JOB_ID="13930098"   # resume own dir after truncated-ckpt requeue failure
TASK="search"
MODEL_NAME="EleutherAI/pythia-410m"
BATCH_SIZE=48
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=5e-5
TARGET_MAX_LOOKAHEAD=96
MAX_INPUT_SIZE=576

GPUS_PER_NODE=$(echo "$SLURM_JOB_GPUS" | tr "," "\n" | wc -l)
[ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "FRESH Pythia 410M s=1  (target L=$TARGET_MAX_LOOKAHEAD)"
echo "GPUs=$GPUS_PER_NODE  eff_batch=$(( BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * GPUS_PER_NODE ))  GC=off"
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
    --accuracy_window 1000
    --eval_every_steps 0
    --max_input_size "$MAX_INPUT_SIZE"
    --max_lookahead "$TARGET_MAX_LOOKAHEAD"
    --base_lookahead 1
    --lookahead_step 1
    --eval_samples 0
    --print_eval_examples 0
    --save_total_limit 2
    --ce_chunk_size 4096
    --persist_every 0
    --use_packing
    --linear_lookahead
)

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"
torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}"
