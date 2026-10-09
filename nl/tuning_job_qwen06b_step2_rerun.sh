#!/bin/bash
# Step-size sweep, s=2 arm, fresh rerun from base Qwen3-0.6B. Created 2026-09-24 at the user's request.
#
# WHY A RERUN. The archived s=2 chain (job_9281359, April/May, reached L=194 at step 51,400) lost all
# its weights in the 2026-06-30 purge (checkpoint-*, final/, stage and persistent checkpoints are empty),
# so it cannot be resumed. The archived s=4 chain (job_9281360) is purged too.
#
# CONFIG. Identical to tuning_job_qwen06b_stepsize.sh (the s=1/8/16/32 reruns) except the step size:
# effective batch 48 x 4 GA x 4 GPUs = 768, LR 5e-5, warmup 500, accuracy_window 800, threshold 0.98,
# MAXL 512 with n_stages 512/2 = 256, max_input_size 3079, budget 1,500,000 PFLOPs. See that script's
# header for why each value was standardised. The archived s=2 used window 1000, max_input_size 1536,
# warmup 100 and older code, so this curve will not reproduce it.
#
# RUN IT INSIDE job 16622776 (h007), whose 0.6B thr99 trainer is stopped and whose batch shell is
# frozen so the allocation stays alive:
#   setsid nohup srun --jobid=16622776 --overlap env SLURM_JOB_GPUS=0,1,2,3 SLURM_CPUS_PER_TASK=56 \
#       bash tuning_job_qwen06b_step2_rerun.sh > slurm/qwen06b_step2_rerun.out 2>&1 < /dev/null & disown
# There is deliberately NO USR1 requeue handler: inside job 16622776, `scontrol requeue` would requeue
# that job and restart the thr99 script. Re-running this script resumes from the latest checkpoint-N.

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
export TRITON_CACHE_DIR="$SCRATCH/triton_cache_step$S_STEP"
mkdir -p slurm "$TRITON_CACHE_DIR"

GPUS_PER_NODE=$(echo "${SLURM_JOB_GPUS:-}" | tr "," "\n" | grep -c .)
[ "$GPUS_PER_NODE" -eq 4 ] || { echo "FATAL: need 4 GPUs for eff_batch 768 (SLURM_JOB_GPUS='${SLURM_JOB_GPUS:-}')"; exit 1; }
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-56} / GPUS_PER_NODE ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

JOB_ID="qwen06b_step${S_STEP}_rerun"
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
    --seed 1234
    --first_token_soft_weight 0.0
    --n_stages "$NSTAGES"
    --base_alpha 0.1
    --max_alpha 1.0
    --accuracy_threshold 0.98
    --min_steps_per_stage 200
    --check_every 25
    --accuracy_window 800
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
echo "JOB START $(date)  s=$S_STEP  n_stages=$NSTAGES  cap=${MAXL}  budget=${PFLOP_CAP} PFLOPs  (inside SLURM job ${SLURM_JOB_ID:-?})"
echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE tuning_nl.py ${ARGS[*]}"
exec torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}"
