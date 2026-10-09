#!/bin/bash
# Finish the Pythia-160M s=1 chain to the 1M PFLOPs cutoff on Gautschi smallgpu.
#
# The chain ran on Gilbreth a100-40gb under `standby` QOS, which caps at 4 h at priority 1 and had not
# scheduled in 11 h. smallgpu is `normal` QOS with a 12 h limit and no preemption. Its Gilbreth job
# (11590957) is HELD so the two cannot diverge; release it only if this is abandoned.
#
# Resumes from checkpoint-7019000 (rsynced from Gilbreth 2026-09-08, curriculum stage 10, 995,723 PFLOPs).
# Effective batch is preserved exactly: Gilbreth ran 256 x GA2 x 4 GPUs = 2048; here 256 x GA4 x 2 = 2048.
# Trainer is tuning_nl.py (FlashAttention 2). The chain used tuning_nl_fa3.py, but FA3 needs Hopper and
# smallgpu is Ada (L40), so the FA3 kernel cannot run here. Both compute exact attention.
#SBATCH -J nl_p160m_finish1M
#SBATCH -A asaparov
#SBATCH -p smallgpu
#SBATCH -q normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=128 --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH -o /home/huan2073/nl-fine-tuning/nl/slurm/%j_%x.out --open-mode=append
#SBATCH --signal=B:USR1@300
set -uo pipefail
cd /home/huan2073/nl-fine-tuning/nl
module load conda; conda activate search
trap 'echo "[USR1] $(date): walltime near, stopping trainer"; [ -n "${TPID:-}" ] && kill -TERM "$TPID" 2>/dev/null; sleep 60; exit 0' USR1
export SCRATCH="/scratch/gautschi/$USER" HF_HOME="$SCRATCH/model_cache" TORCH_HOME="$SCRATCH/model_cache"
export HF_HUB_OFFLINE=1 TRITON_CACHE_DIR="$SCRATCH/triton_cache_p160m" PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NL_DDP_TIMEOUT_MIN=420   # rank 0 rewrites the 1.5 GB, 7M-line loss history on resume: ~2 h. 120 was a near miss (job 16046997).
mkdir -p slurm "$TRITON_CACHE_DIR"
GPUS=$(echo "$SLURM_JOB_GPUS" | tr "," "\n" | wc -l)
[ "$GPUS" -eq 2 ] || { echo "FATAL: need 2 GPUs for eff_batch 2048 at GA 4"; exit 1; }
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
OUT_DIR="$SCRATCH/nl_output/search/job_11590957"
for d in "$OUT_DIR"/checkpoint-*; do [ -d "$d" ] || continue; python -c "import json,sys; json.load(open(sys.argv[1]))" "$d/curriculum_state.json" 2>/dev/null || { echo "[PRUNE] removing partial $d"; rm -rf "$d"; }; done
echo "resuming from: $(ls -d $OUT_DIR/checkpoint-* | sort -t- -k2 -n | tail -1)"
echo "eff_batch = 256 x 4 x $GPUS = $((256*4*GPUS))  (must be 2048)"
ARGS=(
  --task search --model_name EleutherAI/pythia-160m
  --cache_dir "$HF_HOME" --output_dir "$SCRATCH/nl_output" --scratch_dir "$SCRATCH"
  --job_id 11590957
  --batch_size 256 --gradient_accumulation_steps 4
  --learning_rate 1e-4 --seed 1234 --first_token_soft_weight 0.0
  --n_stages 96 --base_alpha 0.1 --max_alpha 1.0 --accuracy_threshold 0.98
  --min_steps_per_stage 200 --check_every 25 --accuracy_window 1000 --eval_every_steps 0
  --max_input_size 576 --max_lookahead 96 --base_lookahead 1 --lookahead_step 1 --linear_lookahead
  --eval_samples 0 --print_eval_examples 0 --save_total_limit 2 --ce_chunk_size 4096
  --persist_every 0 --save_steps 500
  --max_total_pflops 1000000
)
echo "Command: torchrun --nproc_per_node=$GPUS tuning_nl.py ${ARGS[*]}"
torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}" &
TPID=$!
wait $TPID
