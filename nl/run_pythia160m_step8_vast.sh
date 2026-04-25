#!/bin/bash
# Resume Pythia 160M step=8 on vast.ai 2x H100 instance.
# Continues the job_8378391 chain. Target: train until ~500K PFLOPS cumulative
# compute (matches what other model sizes were trained to).
#
# Usage on vast pod:
#   tmux new -s train
#   bash /workspace/nl-fine-tuning/nl/run_pythia160m_step8_vast.sh

set -euo pipefail

cd /workspace/nl-fine-tuning/nl

export SCRATCH=/workspace
export HF_HOME=$SCRATCH/model_cache
export TRITON_CACHE_DIR=$SCRATCH/triton_cache
mkdir -p $HF_HOME $TRITON_CACHE_DIR
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Job ID stays the same → output dir = $SCRATCH/nl_output/search/job_8378391/
# tuning_nl.py auto-detects checkpoint-1691000 and resumes from it.
JOB_ID=8378391

# Same hyperparameters as the original chain (batch=96, GA=2 → eff_batch=384 on 2 GPUs).
# Don't change these — would break apples-to-apples with other model sizes.
exec torchrun --nproc_per_node=2 --master_port=29501 tuning_nl.py \
    --task search \
    --model_name EleutherAI/pythia-160m \
    --cache_dir $HF_HOME \
    --output_dir $SCRATCH/nl_output \
    --scratch_dir $SCRATCH \
    --job_id $JOB_ID \
    --batch_size 96 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --warmup_steps 100 \
    --seed 1234 \
    --num_shots 0 \
    --first_token_soft_weight 0.0 \
    --n_stages 12 \
    --base_alpha 0.1 \
    --max_alpha 1.0 \
    --accuracy_threshold 0.98 \
    --min_steps_per_stage 200 \
    --check_every 25 \
    --accuracy_window 200 \
    --eval_every_steps 1000 \
    --max_input_size 576 \
    --max_lookahead 96 \
    --max_frontier_size 12 \
    --max_branch_size 12 \
    --requested_backtrack 3 \
    --eval_samples 500 \
    --print_eval_examples 5 \
    --use_packing \
    --linear_lookahead \
    --base_lookahead 8 \
    --lookahead_step 8 \
    --gradient_checkpointing \
    --use_chunked_ce \
    --ce_chunk_size 4096 \
    --save_total_limit 2 \
    --do_final_eval --do_redacted_eval --do_seen_eval --do_stage_eval
