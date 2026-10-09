#!/bin/bash
# Branch of the fresh s=2 step-size rerun with a SMALLER advancement window: 800 -> 200 examples.
# Created 2026-09-24 at the user's request ("change to window 200 and resume").
#
# HOW. A new run directory initialised from job_qwen06b_step2_rerun's latest checkpoint via
# --resume_from_job (full checkpoint with optimizer, scheduler and RNG state). tuning_nl.py prepends the
# parent's loss history up to the branch step, so this directory's loss_history.jsonl is one continuous
# curve: window 800 up to the branch step, window 200 after. On any later relaunch the local checkpoint-N
# in this directory takes precedence over the parent. The saved 800-entry accuracy deque is restored into
# a deque of maxlen 200, so the gate starts from the last 200 results.
#
# WHAT A 200 WINDOW MEANS. accuracy_window is a GLOBAL sample count and one optimizer step is 768
# examples, so 200 is about a quarter of one step, checked every 25 steps. At the 0.98 threshold it passes
# on any 200-example stretch with at most 4 errors: a noisier gate than 800. Every other argument is
# identical to tuning_job_qwen06b_step2_rerun.sh.
#
# RUN IT INSIDE job 16622776 (h007), whose thr99 batch shell (pid 3346245) is frozen:
#   setsid nohup srun --jobid=16622776 --overlap env SLURM_JOB_GPUS=0,1,2,3 SLURM_CPUS_PER_TASK=56 \
#       bash tuning_job_qwen06b_step2_w200.sh > slurm/qwen06b_step2_w200.out 2>&1 < /dev/null & disown
# There is deliberately NO USR1 requeue handler: inside job 16622776, `scontrol requeue` would requeue
# that job and restart the thr99 script.

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

S_STEP=2; MAXL=512; NSTAGES=256; PFLOP_CAP=1500000
[ $(( MAXL / S_STEP )) -eq "$NSTAGES" ] || { echo "FATAL: n_stages $NSTAGES != $MAXL/$S_STEP"; exit 1; }
export TRITON_CACHE_DIR="$SCRATCH/triton_cache_step${S_STEP}_w200"
mkdir -p slurm "$TRITON_CACHE_DIR"

GPUS_PER_NODE=$(echo "${SLURM_JOB_GPUS:-}" | tr "," "\n" | grep -c .)
[ "$GPUS_PER_NODE" -eq 4 ] || { echo "FATAL: need 4 GPUs for eff_batch 768 (SLURM_JOB_GPUS='${SLURM_JOB_GPUS:-}')"; exit 1; }
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-56} / GPUS_PER_NODE ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

JOB_ID="qwen06b_step${S_STEP}_w200"
PARENT="qwen06b_step${S_STEP}_rerun"
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
    --max_input_size 3079
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
echo "BRANCH START $(date)  s=$S_STEP  window=200  parent=$PARENT  (inside SLURM job ${SLURM_JOB_ID:-?})"
echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE tuning_nl.py ${ARGS[*]}"
exec torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}"
