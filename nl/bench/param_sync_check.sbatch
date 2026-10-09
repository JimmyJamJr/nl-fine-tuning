#!/bin/bash
#SBATCH -J nl_param_sync
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=28
#SBATCH --time=0:20:00
#SBATCH --partition=ai
#SBATCH -A asaparov
#SBATCH -q normal
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out
#
# Empirical DDP check (2026-10-09): does the CURRENT training path average gradients across ranks?
# The design review found that PackedSequenceTrainer.compute_loss unwraps the DDP model and runs the layers
# directly, so DistributedDataParallel.forward (which arms the reducer) is never called. Runs 4 optimizer steps on
# 2 GPUs with NL_DEBUG_PARAM_SYNC=1; the [PARAM-SYNC] lines report the spread of a parameter checksum across ranks.
# IN SYNC on every step = averaging works; DIVERGED from step 1 = each rank trains its own replica.
# warmup_steps 1 so the first step already applies the full learning rate.
set -uo pipefail
cd /home/huan2073/nl-fine-tuning/nl
module load conda
conda activate search
export SCRATCH="/scratch/gautschi/$USER"
export HF_HOME="$SCRATCH/model_cache" TORCH_HOME="$SCRATCH/model_cache" HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$SCRATCH/triton_cache_pbench"; mkdir -p slurm "$TRITON_CACHE_DIR"
export NL_DEBUG_PARAM_SYNC=1
GPUS_PER_NODE=$(echo "${SLURM_JOB_GPUS:-}" | tr "," "\n" | grep -c .)
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
JOB_ID="pbench_paramsync"; OUT="$SCRATCH/nl_output/search/job_$JOB_ID"
case "$OUT" in */job_pbench_*) rm -rf "$OUT" ;; esac
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "===== $(date) param-sync check on $GPUS_PER_NODE GPUs, host $(hostname) ====="
torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py \
    --task search --model_name Qwen/Qwen3-0.6B --cache_dir "$HF_HOME" \
    --output_dir "$SCRATCH/nl_output" --scratch_dir "$SCRATCH" --job_id "$JOB_ID" \
    --batch_size 16 --gradient_accumulation_steps 1 --learning_rate 5e-5 --warmup_steps 1 --seed 1234 \
    --num_shots 0 --first_token_soft_weight 0.0 \
    --n_stages 1 --base_alpha 0.1 --max_alpha 1.0 --accuracy_threshold 1.01 --min_steps_per_stage 200 \
    --check_every 25 --accuracy_window 200 --eval_every_steps 0 \
    --max_input_size 768 --max_lookahead 128 --linear_lookahead --base_lookahead 16 --lookahead_step 0 \
    --max_frontier_size 12 --max_branch_size 12 --requested_backtrack 3 \
    --eval_samples 500 --print_eval_examples 0 \
    --ce_chunk_size 4096 --use_packing --use_liger --gradient_checkpointing \
    --max_train_steps 4 --save_steps 100000 --save_total_limit 1 --persist_every 0
echo "===== $(date) rc=$? ====="
