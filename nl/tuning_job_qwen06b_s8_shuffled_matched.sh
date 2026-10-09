#!/bin/bash
# Exposure-matched SHUFFLED control for the s=8 curriculum (2026-09-25, user request).
#
# QUESTION. Does the curriculum help because of the ORDER of its examples, or only because of WHICH
# examples it saw? This run trains on the curriculum's realized multiset of examples in uniformly random
# order, with no stages and no advancement rule, and is compared on the same held-out sets.
#
# MULTISET. The paper's s=8 joint run (jobs 8533255 -> 8555128 -> 9495391) reached 98% on the L=128 held-out
# set (Table 1) at step 22,126 = 81,794 PFLOPs, while still training the L=120 stage. Its steps per stage up to
# that point, from its loss_history (768 examples per step): 8:250 16:850 24:1575 32:1675 40:925 48:2000 56:1550
# 64:1525 72:825 80:2550 88:2000 96:4025 104:200 112:1000 120:1175 = 22,125 steps. Its L=96 stage ran at
# max_input_size 576, where the generator caps lookahead at L_eff=94, so it enters the mix as L=94. The original
# examples cannot be regenerated exactly (generated on the fly by older code), so this draws the statistically
# same multiset: every example picks its L independently, weighted by those step counts (--shuffled_mixture),
# from the per-batch-reseeded worker RNG (seed 1234, deterministic). Under --linear_lookahead the graph at a
# given L does not depend on max_input_size, so n=768 throughout reproduces the per-L distribution.
#
# HYPERPARAMETERS. Copied from the joint run's command line (job 8555128): batch 48 x GA 4 x 4 GPUs = 768,
# lr 5e-5 constant after a single 100-step warmup (the curriculum never reset or re-warmed the LR: lr_scheduler
# "constant", no --stage_schedule), seed 1234, packing, liger, chunked CE, gradient checkpointing, no chat
# template, no Dolci, in-run eval every 1000 steps on 500 items. Stage machinery disabled: accuracy_threshold
# 1.01 can never be met, so the run stays in stage 1 (base_lookahead 120 is only the logged L). Stops at exactly
# 22,125 steps (--max_train_steps); compute should land within ~1% of 81.8K PFLOPs (same examples, same tokens).
#
# EVALUATION. Persistent (weights-only, never rotated) checkpoints every 250 steps from step 2,500 on (every 1000
# before; switched at checkpoint-2500 on 2026-09-25 so the 98% crossings resolve finely: the curriculum's L=128 curve
# has 17 points, median gap 1350 steps). Post hoc eval_checkpoints.py at target L = 96, 104, 112,
# 120, 128 (max_input_size 768, 500 items, eval_seed 99999), the same held-out sets as Table 1 / Figure 2.
#
# RUN IT INSIDE job 16622776 (h007), whose thr99 batch shell (pid 3346245) is frozen:
#   setsid nohup srun --jobid=16622776 --overlap env SLURM_JOB_GPUS=0,1,2,3 SLURM_CPUS_PER_TASK=56 \
#       bash tuning_job_qwen06b_s8_shuffled_matched.sh > slurm/qwen06b_s8_shuffled_matched.out 2>&1 < /dev/null & disown
# No USR1 requeue handler (inside a borrowed job). Re-running resumes from the latest checkpoint-N.

set -uo pipefail
cd /home/huan2073/nl-fine-tuning/nl
module load conda
conda activate search
export SCRATCH="/scratch/gautschi/$USER"
export HF_HOME="$SCRATCH/model_cache"
export TORCH_HOME="$SCRATCH/model_cache"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NL_DDP_TIMEOUT_MIN=240
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NL_CKPT_EVERY_N_LAYERS=2
export NL_CKPT_RELAX_MAX_TOKENS=88000
export TRITON_CACHE_DIR="$SCRATCH/triton_cache_s8_shuffled"
mkdir -p slurm "$TRITON_CACHE_DIR"

MIX="8:250,16:850,24:1575,32:1675,40:925,48:2000,56:1550,64:1525,72:825,80:2550,88:2000,94:4025,104:200,112:1000,120:1175"
STEPS=22125

GPUS_PER_NODE=$(echo "${SLURM_JOB_GPUS:-}" | tr "," "\n" | grep -c .)
[ "$GPUS_PER_NODE" -eq 4 ] || { echo "FATAL: need 4 GPUs for eff_batch 768 (SLURM_JOB_GPUS='${SLURM_JOB_GPUS:-}')"; exit 1; }
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-56} / GPUS_PER_NODE ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

JOB_ID="qwen06b_s8_shuffled_matched"
OUT_DIR="$SCRATCH/nl_output/search/job_$JOB_ID"
for d in "$OUT_DIR"/checkpoint-*; do [ -d "$d" ] || continue; python -c "import json,sys; json.load(open(sys.argv[1]))" "$d/curriculum_state.json" 2>/dev/null || { echo "[PRUNE] removing partial $d"; rm -rf "$d"; }; done

ARGS=(
    --task search
    --model_name Qwen/Qwen3-0.6B
    --cache_dir "$HF_HOME"
    --output_dir "$SCRATCH/nl_output"
    --scratch_dir "$SCRATCH"
    --job_id "$JOB_ID"
    --batch_size 48
    --gradient_accumulation_steps 4
    --learning_rate 5e-5
    --warmup_steps 100
    --seed 1234
    --num_shots 0
    --first_token_soft_weight 0.0
    --n_stages 16
    --base_alpha 0.1
    --max_alpha 1.0
    --accuracy_threshold 1.01
    --min_steps_per_stage 200
    --check_every 25
    --accuracy_window 200
    --eval_every_steps 1000
    --max_input_size 768
    --max_lookahead 128
    --max_frontier_size 12
    --max_branch_size 12
    --requested_backtrack 3
    --eval_samples 500
    --print_eval_examples 0
    --use_packing
    --linear_lookahead
    --base_lookahead 120
    --lookahead_step 8
    --gradient_checkpointing
    --use_liger
    --ce_chunk_size 4096
    --shuffled_mixture "$MIX"
    --max_train_steps "$STEPS"
    --persist_every 250
    --save_steps 500
    --save_total_limit 2
)
echo "JOB START $(date)  shuffled exposure-matched s=8 control, $STEPS steps  (inside SLURM job ${SLURM_JOB_ID:-?})"
echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE tuning_nl.py ${ARGS[*]}"
exec torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}"
