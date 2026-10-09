#!/bin/bash
# Resume Pythia-160M s=1 from job_13542861 (latest complete ckpt) in THROUGHPUT mode:
# the run is conclusively asymptoted on stage 9 (full-word acc flat at ~93% vs the 98%
# gate for 550K+ steps at alpha=0.1), so the goal is no longer advancement — it is to
# accumulate compute to the family figure's 1M PFLOPs budget as fast as possible.
#
# Changes vs the eff=192 segment (13542861):
#   - 8 GPUs x bs=256 x GA=1 -> eff_batch=2048 (~10x tokens/step): fewer, fatter
#     optimizer steps amortize per-step overhead and saturate the GEMMs. The run's
#     original batch was 1024; the drop to 192 was an advancement rescue that did
#     not work, and advancement is no longer the goal.
#   - whole node (112 CPUs) for data generation (likely the next bottleneck).
#   - --max_total_pflops 1000000: stops exactly at the 1M chain-total the figure needs.
# LR and all curriculum parameters unchanged. NOTE: batch kink at the resume point
# (as at the 1024->192 drop); the curve is flat at L=9 so achieved-L is unaffected.
#SBATCH -J nl_pythia160m_step1_L96_4g
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

# 4-GPU variant of the 8-GPU job: GA 1->2 keeps eff_batch at 256*2*4 = 2048,
# and JOB_ID_OVERRIDE continues the existing chain dir job_14249842.
export GA_OVERRIDE=2
export JOB_ID_OVERRIDE=14249842

PREV_JOB_ID="13930099"
TASK="search"
MODEL_NAME="EleutherAI/pythia-160m"
BATCH_SIZE=256             # x8 GPUs x GA=1 -> eff 2048 (throughput mode; advancement not the goal)
# GA_OVERRIDE lets this run on fewer GPUs while holding eff_batch at 2048:
# 8 GPUs -> GA 1, 4 GPUs -> GA 2. Per-rank batch (and memory) is unchanged.
GRADIENT_ACCUMULATION_STEPS="${GA_OVERRIDE:-1}"
LEARNING_RATE=1e-4
TARGET_MAX_LOOKAHEAD=96
MAX_INPUT_SIZE=576

# JOB_ID_OVERRIDE pins the output dir (and local auto-resume) to an existing run
# when this script is executed as a slot payload rather than its own sbatch job.
EFFECTIVE_JOB_ID="${JOB_ID_OVERRIDE:-$SLURM_JOB_ID}"

GPUS_PER_NODE=$(echo "$SLURM_JOB_GPUS" | tr "," "\n" | wc -l)
[ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "Resume Pythia 160M s=1 from $PREV_JOB_ID  (throughput mode -> 1M PFLOPs cap)"
echo "GPUs=$GPUS_PER_NODE  bs=$BATCH_SIZE GA=$GRADIENT_ACCUMULATION_STEPS  -> eff_batch=$(( BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * GPUS_PER_NODE ))"
echo "LR=$LEARNING_RATE  target L=$TARGET_MAX_LOOKAHEAD"
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
    --max_total_pflops 1000000
    --linear_lookahead
)

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"
torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}"
