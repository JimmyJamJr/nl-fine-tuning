#!/bin/bash
# ============================================================================
# Qwen 0.6B step=8 curriculum extension: L=128 → L=256
# Resume from job_8555128 (which completed L=128 at step=8 on 2026-04-XX).
# Adds 16 more curriculum stages: L=136, 144, 152, ..., 256.
# ============================================================================

#SBATCH -J nl_qwen06b_curr256_step8_RESUME
#SBATCH -o slurm/%j_%x.out
#SBATCH -e slurm/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=48:00:00
#SBATCH -A asaparov
#SBATCH -p ai
#SBATCH -q preemptible
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

set -euo pipefail
trap 'echo "[SIG] USR1 @ $(date) — grace period"; sleep 120; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
trap 'echo "[SIG] TERM @ $(date)"; exit 0' TERM

module load conda
conda activate search

export SCRATCH="/scratch/gautschi/$USER"
export HF_HOME="$SCRATCH/model_cache"
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
mkdir -p slurm "$SCRATCH/triton_cache" "$SCRATCH/model_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

cd /home/huan2073/nl-fine-tuning/nl

# Resume from previous step=8 chain (8533255 → 8555128 completed L=128)
PREV_JOB_ID="8555128"
TARGET_MAX_LOOKAHEAD=256
MAX_INPUT_SIZE=1536              # 6 * 256

TASK="search"
MODEL_NAME="Qwen/Qwen3-0.6B"
BATCH_SIZE=48
GRADIENT_ACCUMULATION_STEPS=4    # 4 GPU * 48 * 4 = 768 effective (matches original)
LEARNING_RATE=5e-5
N_STAGES=32                      # 256 / 8 = 32 stages, base/step=8 (resumes at stage 17)
BASE_LOOKAHEAD=8
LOOKAHEAD_STEP=8

# GC=on — required at L=256 with max_input=1536 (per the GC=off OOM lesson)
GRADIENT_CHECKPOINTING=true

GPUS_PER_NODE=$(echo $SLURM_JOB_GPUS | tr "," "\n" | wc -l)
[ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
MASTER_PORT=$(( 50000 + RANDOM % 10000 ))

echo "Resume Qwen 0.6B step=8 from job_$PREV_JOB_ID, extending L=128 → L=256"
echo "GPUs=$GPUS_PER_NODE  effective_batch=$(( BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * GPUS_PER_NODE ))"
echo "n=$MAX_INPUT_SIZE  gc=$GRADIENT_CHECKPOINTING"
echo "Output: $SCRATCH/nl_output/$TASK/job_${SLURM_JOB_ID}"

ARGS=(
    --task "$TASK"
    --model_name "$MODEL_NAME"
    --cache_dir "$HF_HOME"
    --output_dir "$SCRATCH/nl_output"
    --scratch_dir "$SCRATCH"
    --job_id "${JOB_ID_OVERRIDE:-$SLURM_JOB_ID}"
    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --warmup_steps 100
    --seed 1234
    --num_shots 0
    --first_token_soft_weight 0.0
    --n_stages "$N_STAGES"
    --base_alpha 0.1
    --max_alpha 1.0
    --accuracy_threshold 0.98
    --min_steps_per_stage 200
    --check_every 25
    --accuracy_window 200
    --eval_every_steps 0
    --max_input_size "$MAX_INPUT_SIZE"
    --max_lookahead "$TARGET_MAX_LOOKAHEAD"
    --base_lookahead "$BASE_LOOKAHEAD"
    --lookahead_step "$LOOKAHEAD_STEP"
    --max_frontier_size 12
    --max_branch_size 12
    --requested_backtrack 3
    --eval_samples 500
    --print_eval_examples 5
    --stage_eval_every 8
    --ce_chunk_size 4096
    --resume_from_job "$PREV_JOB_ID"
    --do_baseline --do_final_eval --do_redacted_eval --do_seen_eval --do_stage_eval
    --use_packing
    --linear_lookahead
    --use_liger
)
$GRADIENT_CHECKPOINTING && ARGS+=(--gradient_checkpointing)

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"
torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}"
echo "DONE $(date)"
