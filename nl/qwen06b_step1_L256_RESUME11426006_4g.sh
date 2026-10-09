#!/bin/bash
# Resume Qwen3-0.6B s=1 chain past its 1.5M PFLOPs end to answer whether the
# 1.7B (Max's run) crosses it — projected crossover ~1.8M PFLOPs, so cap the
# chain total at 2.0M via --max_total_pflops (cumulative across the chain,
# as verified on the 1.4B pretrained run which stopped at 5001K/5000K).
#
# Chain: 10696449 -> 10730891 -> 11426006 (CANCELLED at 7d wall, step 179830,
# stage 245, 1505K PFLOPs). Resumes from job_11426006/checkpoint-179500
# (verified complete: model.safetensors loads, optimizer/scheduler/rng/
# curriculum_state all present) — loses only 330 steps.
#
# Config identical to the previous segment (W=800, effective batch 4x96x2=768).
# --nice keeps this below the running reinit jobs; h009 excluded (sick node).
#SBATCH -J nl_qwen06b_step1_L256_W800_resume
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
#SBATCH --exclude=h009
#SBATCH --nice=100

set -euo pipefail
module load conda
conda activate search

export SCRATCH="/scratch/gautschi/$USER"
export HF_HOME="$SCRATCH/model_cache"
export HF_HUB_OFFLINE=1   # tokenizer/model already cached; skip revalidation to avoid HF 429 (killed job 10730891)
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
mkdir -p slurm "$SCRATCH/triton_cache" "$SCRATCH/model_cache"
export PYTHONUNBUFFERED=1

cd /home/huan2073/nl-fine-tuning/nl

PREV_JOB_ID="11426006"                            # resumes from latest ckpt in job_11426006 (ckpt-179500)
TASK="search"
MODEL_NAME="Qwen/Qwen3-0.6B"
BATCH_SIZE=96
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=5e-5
TARGET_MAX_LOOKAHEAD=256
MAX_INPUT_SIZE=1536

GPUS_PER_NODE=$(echo "$SLURM_JOB_GPUS" | tr "," "\n" | wc -l)
[ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "Resume Qwen 0.6B s=1 from $PREV_JOB_ID  W=800  (target L=$TARGET_MAX_LOOKAHEAD, cap 2.0M PFLOPs chain total)"
echo "GPUs=$GPUS_PER_NODE  effective_batch=$(( BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * GPUS_PER_NODE ))"
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
    --max_total_pflops 2000000
    --linear_lookahead
    --gradient_checkpointing
    --use_liger
)

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"
torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}"
