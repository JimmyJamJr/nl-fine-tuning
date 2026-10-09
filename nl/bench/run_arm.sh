#!/bin/bash
# One arm of the packing benchmark (2026-10-09): bash bench/run_arm.sh <arm> <L> [STEPS]
#   arm = <packing>_<kernel>, packing in {custom,hf} (NL_PACKING; 'hf' needs the refactored trainer), kernel in {fa2,fa3}.
# Runs STEPS optimizer steps of one fixed stage at lookahead L with the paper's s=8 Qwen3-0.6B hyperparameters
# (bs48 x GA4 x 4 GPUs = 768, lr 5e-5, seed 1234, n=768; nocurr-style single stage as job 8555133: n_stages 1,
# lookahead_step 0; threshold 1.01 never advances; evals off; no checkpoints). Idempotent: skips the arm if its
# loss_history already holds >= STEPS steps. Needs 4 GPUs in SLURM_JOB_GPUS (job script or `srun --overlap`).
# Outputs: $SCRATCH/nl_output/search/job_pbench_<arm>_L<L>/loss_history.jsonl and bench/mem_<arm>_L<L>.csv.
set -uo pipefail
cd /home/huan2073/nl-fine-tuning/nl
arm=$1; L=$2; STEPS=${3:-120}
command -v torchrun >/dev/null 2>&1 || { module load conda; conda activate search; }   # borrowed-allocation srun has no env
export SCRATCH="/scratch/gautschi/$USER"
export HF_HOME="$SCRATCH/model_cache" TORCH_HOME="$SCRATCH/model_cache" HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export NL_CKPT_EVERY_N_LAYERS=${NL_CKPT_EVERY_N_LAYERS:-2}       # same selective checkpointing in every arm
export NL_CKPT_RELAX_MAX_TOKENS=${NL_CKPT_RELAX_MAX_TOKENS:-88000}
export TRITON_CACHE_DIR="$SCRATCH/triton_cache_pbench"; mkdir -p bench "$TRITON_CACHE_DIR"
# DETERMINISTIC=1 (parity arms, per the design review): deterministic flash-attention backward + cuBLAS workspace, so
# old-vs-new loss curves must match to bf16 rounding instead of drifting apart chaotically. Slower; never use it for
# the speed arms. Runs get a _det suffix so speed and parity results do not overwrite each other.
SUFFIX=""
if [ "${DETERMINISTIC:-0}" = 1 ]; then
  export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=":4096:8"; SUFFIX="_det"
  export NL_DEBUG_PARAM_SYNC=1   # [PARAM-SYNC] per-rank parameter checksums every step: must match bitwise old vs new
fi
GPUS_PER_NODE=$(echo "${SLURM_JOB_GPUS:-}" | tr "," "\n" | grep -c .)
# NGPUS (default 4): the per-GPU work (48 x GA 4) is fixed, so a 1-GPU arm reproduces rank 0 of a 4-GPU arm exactly
# (per-rank seeded data, no cross-rank gradient averaging); tokens/step scale with the GPU count. Use TAG=_1g.
[ "$GPUS_PER_NODE" -eq "${NGPUS:-4}" ] || { echo "FATAL: need ${NGPUS:-4} GPUs (SLURM_JOB_GPUS='${SLURM_JOB_GPUS:-}')"; exit 1; }
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-56} / GPUS_PER_NODE ))

# TAG (optional, e.g. _h017): distinguishes arms re-run on another node; step times are only comparable within a node.
TAG=${TAG:-}
JOB_ID="pbench_${arm}${SUFFIX}${TAG}_L${L}"; OUT="$SCRATCH/nl_output/search/job_$JOB_ID"
LOG="bench/log_${arm}${SUFFIX}${TAG}_L${L}.txt"
if [ -f "$OUT/loss_history.jsonl" ] && [ "$(wc -l < "$OUT/loss_history.jsonl")" -ge "$STEPS" ]; then
  echo "[SKIP] $JOB_ID already has >= $STEPS steps"; exit 0
fi
case "$OUT" in */job_pbench_*) rm -rf "$OUT" ;; *) echo "refusing to delete $OUT"; exit 1 ;; esac   # fresh, never resume
export NL_PACKING="${arm%%_*}" NL_ATTN_KERNEL="${arm##*_}"
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
MEMLOG="bench/mem_${arm}${SUFFIX}${TAG}_L${L}.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader -l 2 > "$MEMLOG" & MEMPID=$!
echo "===== $(date)  arm=$arm  L=$L  NL_PACKING=$NL_PACKING  NL_ATTN_KERNEL=$NL_ATTN_KERNEL  steps=$STEPS  host=$(hostname) ====="
torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py \
    --task search --model_name Qwen/Qwen3-0.6B --cache_dir "$HF_HOME" \
    --output_dir "$SCRATCH/nl_output" --scratch_dir "$SCRATCH" --job_id "$JOB_ID" \
    --batch_size 48 --gradient_accumulation_steps 4 --learning_rate 5e-5 --seed 1234 \
    --first_token_soft_weight 0.0 \
    --n_stages 1 --base_alpha 0.1 --max_alpha 1.0 --accuracy_threshold 1.01 --min_steps_per_stage 200 \
    --check_every 25 --accuracy_window 200 --eval_every_steps 0 \
    --max_input_size 768 --max_lookahead 128 --linear_lookahead --base_lookahead "$L" --lookahead_step 0 \
    --eval_samples 500 --print_eval_examples 0 \
    --ce_chunk_size 4096 --use_liger --gradient_checkpointing \
    --max_train_steps "$STEPS" --save_steps 100000 --save_total_limit 1 --persist_every 0 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
kill "$MEMPID" 2>/dev/null; wait "$MEMPID" 2>/dev/null
echo "===== $(date)  arm=$arm  L=$L  rc=$rc  steps_logged=$(wc -l < "$OUT/loss_history.jsonl" 2>/dev/null) ====="
exit $rc
