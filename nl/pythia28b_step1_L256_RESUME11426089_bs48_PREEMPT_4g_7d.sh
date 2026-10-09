#!/bin/bash
# Pythia-2.8B L=256 extension RESTART after OOM in job_11426089 at L=99 alpha=0.39.
# Resume from job_11426089/ckpt-35500. Drop bs from 96 -> 48 (GA 4 -> 8) to halve
# per-rank memory while preserving eff_batch = 4 * 48 * 8 = 1536 exactly.
#SBATCH -J nl_pythia28b_step1_L256_resume
#SBATCH -o slurm/%j_%x.out
#SBATCH -e slurm/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=7-00:00:00
#SBATCH -A asaparov
#SBATCH -p ai
#SBATCH -q preemptible
#SBATCH --signal=B:USR1@180

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

PREV_JOB_ID="11426089"        # resume from ckpt-35500 (L=99 alpha=0.39 just before OOM)
TASK="search"
MODEL_NAME="EleutherAI/pythia-2.8b"
BATCH_SIZE=48                 # halved from 96 to halve per-rank memory
GRADIENT_ACCUMULATION_STEPS=8 # doubled from 4 to preserve eff_batch = 48*8*4 = 1536
LEARNING_RATE=2.7e-5
TARGET_MAX_LOOKAHEAD=256
MAX_INPUT_SIZE=1536

# Allows this script to be run as a slot payload: JOB_ID_OVERRIDE pins the output
# dir (and therefore the local auto-resume) to the existing chain segment instead
# of the slot's own SLURM_JOB_ID.
EFFECTIVE_JOB_ID="${JOB_ID_OVERRIDE:-$SLURM_JOB_ID}"

GPUS_PER_NODE=$(echo "$SLURM_JOB_GPUS" | tr "," "\n" | wc -l)
[ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "Extend Pythia 2.8B s=1 from $PREV_JOB_ID  (target L=$TARGET_MAX_LOOKAHEAD, seqlen=$MAX_INPUT_SIZE)"
echo "GPUs=$GPUS_PER_NODE  eff_batch=$(( BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * GPUS_PER_NODE ))  GC=on"
echo "Output: $SCRATCH/nl_output/$TASK/job_${EFFECTIVE_JOB_ID}"

ARGS=(
    --task "$TASK"
    --model_name "$MODEL_NAME"
    --cache_dir "$HF_HOME"
    --output_dir "$SCRATCH/nl_output"
    --scratch_dir "$SCRATCH"
    --job_id "$EFFECTIVE_JOB_ID"
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
    --min_steps_per_stage 0
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
    --use_packing
    --linear_lookahead
    --gradient_checkpointing
)

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"
torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}"
