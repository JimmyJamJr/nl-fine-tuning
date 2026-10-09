#!/bin/bash
# Extra GPU smokes for the HF packed path (2026-10-09), each ~2-4 min on 4 GPUs, run inside an allocation via
#   srun --jobid=<alloc> --overlap env SLURM_JOB_GPUS=0,1,2,3 SLURM_CPUS_PER_TASK=56 bash bench/smoke_extra.sh
# 1. evals: in-run evals every 5 steps + final eval, exercising eval_attn_sdpa's flash -> sdpa -> flash round trip
#    (never exercised by the benchmark arms, which train with evals off).
# 2. lora: --use_lora --lora_rank 8 through the HF body (peft layers inside inner, enable_input_require_grads + GC).
# 3. pythia: EleutherAI/pythia-160m, the GPT-NeoX body path (use_cache=False, no Liger), FA3.
# Each must end with rc=0 and the expected log lines (checked by the caller).
set -uo pipefail
cd /home/huan2073/nl-fine-tuning/nl
command -v torchrun >/dev/null 2>&1 || { module load conda; conda activate search; }
export SCRATCH="/scratch/gautschi/$USER" HF_HOME="/scratch/gautschi/$USER/model_cache" TORCH_HOME="/scratch/gautschi/$USER/model_cache"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$SCRATCH/triton_cache_pbench"; mkdir -p bench
export NL_PACKING=hf NL_ATTN_KERNEL=fa3 NL_CKPT_EVERY_N_LAYERS=2 NL_CKPT_RELAX_MAX_TOKENS=88000
GPUS_PER_NODE=$(echo "${SLURM_JOB_GPUS:-}" | tr "," "\n" | grep -c .)
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-56} / GPUS_PER_NODE ))
COMMON=(--task search --cache_dir "$HF_HOME" --output_dir "$SCRATCH/nl_output" --scratch_dir "$SCRATCH"
        --batch_size 16 --gradient_accumulation_steps 1 --learning_rate 5e-5 --warmup_steps 10 --seed 1234
        --num_shots 0 --first_token_soft_weight 0.0 --n_stages 1 --base_alpha 0.1 --max_alpha 1.0
        --accuracy_threshold 1.01 --min_steps_per_stage 200 --check_every 25 --accuracy_window 200
        --max_input_size 768 --max_lookahead 128 --linear_lookahead --base_lookahead 16 --lookahead_step 0
        --max_frontier_size 12 --max_branch_size 12 --requested_backtrack 3 --print_eval_examples 0
        --ce_chunk_size 4096 --use_packing --gradient_checkpointing --save_steps 100000 --save_total_limit 1 --persist_every 0)
run() {  # name, then extra args
  local name=$1; shift
  local OUT="$SCRATCH/nl_output/search/job_pbench_smoke_$name"
  case "$OUT" in */job_pbench_*) rm -rf "$OUT" ;; esac
  local PORT; PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
  echo "===== $(date) smoke=$name ====="
  torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$PORT" tuning_nl.py "${COMMON[@]}" --job_id "pbench_smoke_$name" "$@" 2>&1 | tee "bench/log_smoke_$name.txt"
  echo "===== $(date) smoke=$name rc=${PIPESTATUS[0]} ====="
}
run evals  --model_name Qwen/Qwen3-0.6B --use_liger --eval_samples 50 --eval_every_steps 5 --do_final_eval --max_train_steps 12
run lora   --model_name Qwen/Qwen3-0.6B --use_liger --eval_samples 500 --eval_every_steps 0 --use_lora --lora_rank 8 --max_train_steps 6
run pythia --model_name EleutherAI/pythia-160m --eval_samples 50 --eval_every_steps 5 --do_final_eval --max_train_steps 12
echo "SMOKE-EXTRA DONE $(date)"
