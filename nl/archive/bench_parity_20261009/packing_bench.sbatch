#!/bin/bash
#SBATCH -J nl_pack_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=2:30:00
#SBATCH --partition=ai
#SBATCH -A asaparov
#SBATCH -q normal
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out
#SBATCH --open-mode=append
#
# Parity + speed benchmark for the packing refactor (2026-10-09). Same model, data, seed and hyperparameters
# in every arm; only the forward path and the attention kernel differ:
#   custom_fa2  old hand-rolled packing + custom layer loop, FlashAttention-2   (the paper's Qwen configuration)
#   custom_fa3  old path, FlashAttention-3                                      (isolates the kernel change)
#   hf_fa3      new Hugging Face packed path, FlashAttention-3                  (isolates the refactor)
#   hf_fa2      new path, FlashAttention-2                                      (closest bitwise comparison to custom_fa2)
# at two depths: L=16 (short examples, ~250 tokens) and L=128 (~770 tokens; paper micro-batches of ~117K tokens).
# Each arm runs STEPS optimizer steps of a single fixed stage (threshold 1.01 never advances), writing
# $SCRATCH/nl_output/search/job_pbench_<arm>_L<L>/loss_history.jsonl (loss, tokens, wall_time per step) and
# bench/mem_<arm>_L<L>.csv (nvidia-smi memory every 2 s). Compare with: python bench/compare_packing_bench.py
# Selection env vars honoured by tuning_nl.py: NL_PACKING=custom|hf (added by the refactor), NL_ATTN_KERNEL=fa2|fa3.
# Submit from nl/:  sbatch bench/packing_bench.sbatch        (ARMS="custom_fa2 hf_fa3" DEPTHS="128" to subset)
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
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NL_CKPT_EVERY_N_LAYERS=2          # same selective-checkpointing setting in every arm (paper s=8 runs)
export NL_CKPT_RELAX_MAX_TOKENS=88000
export TRITON_CACHE_DIR="$SCRATCH/triton_cache_pbench"
mkdir -p slurm bench "$TRITON_CACHE_DIR"

GPUS_PER_NODE=$(echo "${SLURM_JOB_GPUS:-}" | tr "," "\n" | grep -c .)
[ "$GPUS_PER_NODE" -eq 4 ] || { echo "FATAL: need 4 GPUs for eff_batch 768 (got '$GPUS_PER_NODE')"; exit 1; }
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))

# WAIT_FOR_GO=1: hold the allocation until bench/GO exists (the refactor landing), at most HOLD_MIN minutes, so the
# benchmark does not queue behind the partition backlog after the code is ready. The GO file is removed on pickup.
if [ "${WAIT_FOR_GO:-0}" = 1 ]; then
  HOLD_MIN=${HOLD_MIN:-120}; waited=0
  echo "$(date) holding for bench/GO (max ${HOLD_MIN} min)"
  until [ -f bench/GO ]; do
    sleep 30; waited=$((waited + 1))
    [ $((waited / 2)) -ge "$HOLD_MIN" ] && { echo "$(date) no GO after ${HOLD_MIN} min, releasing allocation"; exit 0; }
  done
  rm -f bench/GO; echo "$(date) GO received after $((waited / 2)) min"
fi
STEPS=${STEPS:-120}
ARMS=${ARMS:-"custom_fa2 custom_fa3 hf_fa3 hf_fa2"}
DEPTHS=${DEPTHS:-"16 128"}
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
# Per-arm logic lives in bench/run_arm.sh (idempotent: arms with a complete loss_history are skipped), so the same
# runner can be driven by this job or by `srun --jobid=<alloc> --overlap` on a borrowed allocation.
for L in $DEPTHS; do
  for arm in $ARMS; do
    bash bench/run_arm.sh "$arm" "$L" "$STEPS"
  done
done
echo "BENCH DONE $(date)"
