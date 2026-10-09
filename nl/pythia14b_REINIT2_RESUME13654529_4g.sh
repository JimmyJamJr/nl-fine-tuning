#!/bin/bash
# Resume Pythia 1.4B REINIT 2 from job_11427690 latest checkpoint on 4 GPUs
# (down from 8 GPUs to free CPUs and stay under QOSMaxCpuPerUserLimit).
#
# Hypers preserved: bs=192 GA=1 on 4 GPUs -> eff_batch=768, LR=1e-4, GC=on.
# Matches original Reinit 2 config exactly (bs=192 GA=1 on 4 GPUs).
# Latest ckpt: job_13654529 (stage 35, L=35). 4 GPUs: the 8-GPU slot under the 24-GPU
# per-user cap goes to the Qwen reinit arm; bs=192 GA=1 on 4 GPUs -> eff_batch=768, unchanged.
#SBATCH -J nl_pythia14b_REINIT2_resume
#SBATCH -o slurm/%j_%x.out
#SBATCH -e slurm/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
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
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p slurm "$SCRATCH/triton_cache" "$SCRATCH/model_cache"
export PYTHONUNBUFFERED=1

cd /home/huan2073/nl-fine-tuning/nl

PREV_JOB_ID="13654529"
TASK="search"
MODEL_NAME="EleutherAI/pythia-1.4b"
BATCH_SIZE=192                      # matches original Reinit 2 (on 4 GPUs eff_batch = 4*192*1 = 768)
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-4
TARGET_MAX_LOOKAHEAD=96
MAX_INPUT_SIZE=576

GPUS_PER_NODE=$(echo "$SLURM_JOB_GPUS" | tr "," "\n" | wc -l)
[ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "Resume Pythia 1.4B REINIT 2 from $PREV_JOB_ID on 4 GPUs"
echo "GPUs=$GPUS_PER_NODE  bs=$BATCH_SIZE  GA=$GRADIENT_ACCUMULATION_STEPS  -> eff_batch=$(( BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * GPUS_PER_NODE )) (target 768)"
echo "LR=$LEARNING_RATE  target L=$TARGET_MAX_LOOKAHEAD  max_input_size=$MAX_INPUT_SIZE"
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
    --eval_samples 500
    --print_eval_examples 0
    --save_total_limit 2
    --ce_chunk_size 4096
    --persist_every 0
    --do_stage_eval
    --stage_eval_every 8
    --linear_lookahead
    --use_liger
    --gradient_checkpointing
)

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"
torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}"
