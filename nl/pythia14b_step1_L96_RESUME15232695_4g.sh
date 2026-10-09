#!/bin/bash
# FORECAST TEST (resumed 2026-08-31). The parent run job_15232695 stopped at its
# 5,000,829 PFLOP budget cap at L=76, i.e. 90% of the Weibull-fitted ceiling
# Lmax = 84.45 (95% CI [83.1, 85.8]). This segment is an OUT-OF-SAMPLE test of
# that fit: the curve predicts, from data ending at L=76, that the next
# lookaheads arrive at these cumulative compute values --
#     L=77 ->  6,403,579 PFLOPs
#     L=78 ->  7,526,358 PFLOPs
#     L=79 ->  8,990,123 PFLOPs
# Budget raised to 9,000,000 PFLOPs so all three predictions can be checked.
# If the run tracks these, the asymptotic (finite-ceiling) model is validated
# out of sample; if it advances faster, the ceiling is underestimated.
# Nothing else changed: bs=24 x GA=2 x 4 GPUs = eff_batch 192, LR 2.7e-5, curriculum identical to job_15232695.
# Resume the Pythia-1.4B s=1 long curriculum run from latest checkpoint.
# Local mirror dir: job_local_20260429_215108_pythia14b_step1_L96
# Latest checkpoint: checkpoint-391000 (stage 55, eff_L 55, alpha 0.584)
# Effective batch preserved: 4 GPUs × 24 × 2 = 192 (was 2 GPUs × 24 × 4 = 192).
#SBATCH -J nl_pythia14b_step1_L96_resume
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
mkdir -p slurm "$SCRATCH/triton_cache" "$SCRATCH/model_cache"
export PYTHONUNBUFFERED=1

cd /home/huan2073/nl-fine-tuning/nl

PREV_JOB_ID="15232695"
TASK="search"
MODEL_NAME="EleutherAI/pythia-1.4b"
BATCH_SIZE=24
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=2.7e-5
TARGET_MAX_LOOKAHEAD=96
MAX_INPUT_SIZE=576

GPUS_PER_NODE=$(echo "$SLURM_JOB_GPUS" | tr "," "\n" | wc -l)
[ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "Resume Pythia 1.4B s=1 from $PREV_JOB_ID  (target L=$TARGET_MAX_LOOKAHEAD)"
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
    --warmup_steps 100
    --seed 1234
    --num_shots 0
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
    --max_frontier_size 12
    --max_branch_size 12
    --requested_backtrack 3
    --eval_samples 500
    --print_eval_examples 0
    --ce_chunk_size 4096
    --persist_every 0
    --max_total_pflops 9000000
    --do_stage_eval
    --stage_eval_every 8
    --use_packing
    --linear_lookahead
    --use_liger
)

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"
torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}"
