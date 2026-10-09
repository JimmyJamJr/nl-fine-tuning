#!/bin/bash
# Real attribute-vocabulary ablation, "V at the max" arm, at the settings of the s=8 breadth runs (2026-09-25, user
# request: "start a run of the actual ablation where we put V at the max, same settings as the ongoing b=3,4 runs").
#
# WHAT IS FIXED. --vocab_pool fixed maps every vertex ID to ONE dictionary name (the name actually appears in the
# text, unlike the paper runs' vocab_pool=none, where names were fresh random strings per instance) and pins the ID
# range at the context maximum from stage 1. max_input_size 1544 makes that maximum (n-5)//3+1 = 514 = 2(256+1)
# names, exactly the vocabulary a growing curriculum reaches at the target L=256 (the paper's definition). At the
# breadth runs' n=3080 the pool would be 1026, twice what any stage uses. Under --linear_lookahead the graph at a
# given L does not depend on n, so n changes nothing else; 1544 still reaches L=256 at b=2 (L cap 256).
#
# SETTINGS copied from tuning_job_qwen06b_s8_breadth_w1000.sh: batch 48 x GA 4 x 4 GPUs = 768, lr 5e-5, warmup 100,
# seed 1234, no Dolci, no chat template, s=8 to L=256 (32 stages), gate 98% on W=1000, min 200 steps, check every 25,
# 1.5M PFLOPs, stage evals on. breadth = 2 (default). The matching "grow" arm (--vocab_pool grow, same settings) does
# not exist yet; without it the comparison vs the random-name runs mixes dictionary names with random names.
#
# RUNS INSIDE job 16455236 (h019), batch shell 437957 frozen. CAUTION: h019 GPU 2 throttles thermally under load
# (86-88 C, 345-735 MHz, ~35% of a healthy H100), so this 4-GPU DDP run goes at roughly half speed.
#   setsid nohup srun --jobid=16455236 --overlap env SLURM_JOB_GPUS=0,1,2,3 SLURM_CPUS_PER_TASK=56 \
#       bash tuning_job_qwen06b_s8_vfixed_w1000.sh > slurm/qwen06b_s8_vfixed_w1000.out 2>&1 < /dev/null & disown
# No USR1 requeue handler (borrowed job). Re-running resumes from the latest checkpoint-N.

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

S_STEP=8; MAXL=256; NSTAGES=32; NIN=1544; PFLOP_CAP=1500000; WINDOW=1000
[ $(( MAXL / S_STEP )) -eq "$NSTAGES" ] || { echo "FATAL: n_stages $NSTAGES != $MAXL/$S_STEP"; exit 1; }
[ $(( (NIN - 5) / 3 + 1 )) -eq $(( 2 * (MAXL + 1) )) ] || { echo "FATAL: fixed pool $(( (NIN-5)/3+1 )) != 2(L+1)"; exit 1; }
export TRITON_CACHE_DIR="$SCRATCH/triton_cache_s8_vfixed_w1000"
mkdir -p slurm "$TRITON_CACHE_DIR"

GPUS_PER_NODE=$(echo "${SLURM_JOB_GPUS:-}" | tr "," "\n" | grep -c .)
[ "$GPUS_PER_NODE" -eq 4 ] || { echo "FATAL: need 4 GPUs for eff_batch 768 (SLURM_JOB_GPUS='${SLURM_JOB_GPUS:-}')"; exit 1; }
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-56} / GPUS_PER_NODE ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

JOB_ID="qwen06b_s8_vfixed_w1000"
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
    --n_stages "$NSTAGES"
    --base_alpha 0.1
    --max_alpha 1.0
    --accuracy_threshold 0.98
    --min_steps_per_stage 200
    --check_every 25
    --accuracy_window "$WINDOW"
    --eval_every_steps 0
    --max_input_size "$NIN"
    --max_lookahead "$MAXL"
    --linear_lookahead
    --base_lookahead "$S_STEP"
    --lookahead_step "$S_STEP"
    --vocab_pool fixed
    --max_frontier_size 12
    --max_branch_size 12
    --requested_backtrack 3
    --eval_samples 500
    --print_eval_examples 0
    --ce_chunk_size 4096
    --use_packing
    --use_liger
    --gradient_checkpointing
    --do_stage_eval
    --do_final_eval
    --persist_every 2000
    --save_steps 500
    --save_total_limit 2
    --max_total_pflops "$PFLOP_CAP"
)
echo "JOB START $(date)  s=$S_STEP  vocab_pool=fixed (514 names)  window=$WINDOW  n=$NIN  cap=$MAXL  budget=$PFLOP_CAP  (inside SLURM job ${SLURM_JOB_ID:-?})"
echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE tuning_nl.py ${ARGS[*]}"
exec torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}"
