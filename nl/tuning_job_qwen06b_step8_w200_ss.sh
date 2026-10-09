#!/bin/bash
# STAGE-START variant (2026-09-23): the window-200 branch restarted at the START of the stuck stage
# L=216 (step 43,100, the stage checkpoint saved when L=208 completed), not at checkpoint-46000 near
# the end of the stall as in tuning_job_qwen06b_step8_w200.sh. Source dir job_qwen06b_step8_rerun_at43100
# holds checkpoint-43100 (symlinks to stage_checkpoints/stage_26_step_43100_L208 + a trainer_state.json
# with global_step 43100) and the parent's loss history cut at 43,100. Stage checkpoints carry no
# optimizer or RNG state, so this run starts with a FRESH AdamW and re-runs the 500-step LR warmup;
# data order is keyed to the step, so it replays the parent's data from step 43,101. Everything else
# identical to the window-200 branch below.
# Branch of the s=8 step-size rerun with a SMALLER advancement window: 800 -> 200 examples.
# Created 2026-09-22 at the user's request.
#
# WHY. job_qwen06b_step8_rerun (window 800) completed L=208 at 262,520 PFLOPs and then sat on
# L=216 for about 36 hours without clearing it, while the original April s=8 run climbed past the
# same stages much faster. The two runs process identical tokens per step. Companion to
# tuning_job_qwen06b_step16_w200.sh: both ask whether gate strictness holds the reruns back.
#
# WHAT A 200 WINDOW MEANS HERE. accuracy_window is a GLOBAL sample count, and one optimizer step
# is 768 examples, so a 200 window is about a quarter of a single step, checked every 25 steps.
# At the 0.98 threshold it passes on any 200-example stretch with at most 4 errors. It is a much
# noisier gate than 800 and can advance before the stage is truly mastered. That is the point
# of the test, but it also means curves at window 200 and window 800 are not comparable.
#
# HOW. A new run directory initialised from the rerun's latest checkpoint via --resume_from_job
# (checkpoint-46000, mid-way through L=216, with optimizer state). tuning_nl.py prepends the
# rerun's loss history up to the branch step, so this directory's loss_history.jsonl is one
# continuous curve: window 800 up to step 46,000, window 200 after. The rerun directory is left
# untouched as the window-800 record. Every other argument matches the LIVE s=8 trainer's command
# line exactly (n_stages 40, max_lookahead 320, max_input_size 2048); the s=8 job was submitted
# from an older version of tuning_job_qwen06b_stepsize.sh than the s=16 job.
#
# RUN IT INSIDE job 16455236 (h019), whose window-800 trainer is stopped and whose batch shell
# is frozen so the allocation stays alive:
#   setsid nohup srun --jobid=16455236 --overlap env SLURM_JOB_GPUS=0,1,2,3 SLURM_CPUS_PER_TASK=56 \
#       bash tuning_job_qwen06b_step8_w200.sh > slurm/qwen06b_step8_w200.out 2>&1 < /dev/null & disown
# There is deliberately NO USR1 requeue handler: inside job 16455236, `scontrol requeue` would
# requeue that job and restart the ORIGINAL window-800 script.

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

S_STEP=8; NSTAGES=40; MAXL=320; PFLOP_CAP=1500000
export TRITON_CACHE_DIR="$SCRATCH/triton_cache_step8_w200_ss"
mkdir -p slurm "$TRITON_CACHE_DIR"

GPUS_PER_NODE=$(echo "${SLURM_JOB_GPUS:-}" | tr "," "\n" | grep -c .)
[ "$GPUS_PER_NODE" -eq 4 ] || { echo "FATAL: need 4 GPUs for eff_batch 768 (SLURM_JOB_GPUS='${SLURM_JOB_GPUS:-}')"; exit 1; }
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-56} / GPUS_PER_NODE ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

JOB_ID="qwen06b_step8_w200_ss"
PARENT="qwen06b_step8_rerun_at43100"
OUT_DIR="$SCRATCH/nl_output/search/job_$JOB_ID"
for d in "$OUT_DIR"/checkpoint-*; do [ -d "$d" ] || continue; python -c "import json,sys; json.load(open(sys.argv[1]))" "$d/curriculum_state.json" 2>/dev/null || { echo "[PRUNE] removing partial $d"; rm -rf "$d"; }; done

ARGS=(
    --task search
    --model_name Qwen/Qwen3-0.6B
    --cache_dir "$HF_HOME"
    --output_dir "$SCRATCH/nl_output"
    --scratch_dir "$SCRATCH"
    --job_id "$JOB_ID"
    --resume_from_job "$PARENT"
    --batch_size 48
    --gradient_accumulation_steps 4
    --learning_rate 5e-5
    --seed 1234
    --first_token_soft_weight 0.0
    --n_stages "$NSTAGES"
    --base_alpha 0.1
    --max_alpha 1.0
    --accuracy_threshold 0.98
    --min_steps_per_stage 200
    --check_every 25
    --accuracy_window 200
    --eval_every_steps 0
    --max_input_size 2048
    --max_lookahead "$MAXL"
    --linear_lookahead
    --base_lookahead "$S_STEP"
    --lookahead_step "$S_STEP"
    --eval_samples 0
    --print_eval_examples 0
    --ce_chunk_size 4096
    --use_liger
    --gradient_checkpointing
    --persist_every 2000
    --save_steps 500
    --save_total_limit 2
    --max_total_pflops "$PFLOP_CAP"
)
echo "BRANCH START $(date)  s=8  window=200  parent=$PARENT"
echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE tuning_nl.py ${ARGS[*]}"
exec torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}"
