#!/bin/bash
# Exact-frontier eval: 300 items with lookahead EXACTLY L per checkpoint (the l==L bin of run_frontier_eval.sh had n=14-33), 2026-09-23.
# Runs inside eval_worker job 16604009 on g005 with PRIORITY: the depth-benchmark eval processes
# (eval_downstream.py) on this node are SIGSTOPped for the duration and resumed on exit, including
# on failure (trap). Nothing they hold is lost; they continue where they stopped.
#   srun --jobid=16604009 --overlap -N1 -n1 bash run_frontier_exact.sh
# Checkpoints = stage checkpoints saved at the exact step their gate passed (verified in the logs).
# The paper's own Qwen s=8 and s=1 chains were purged 2026-06-30; these are the surviving ones.
cd /home/huan2073/nl-fine-tuning/nl
export HF_HOME=/scratch/gautschi/huan2073/model_cache TORCH_HOME=/scratch/gautschi/huan2073/model_cache
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=$HOME/.conda/envs/search/bin/python
S=/scratch/gautschi/huan2073/nl_output/search
mkdir -p eval_frontier

PIDS=$(pgrep -u "$USER" -f "eval_downstream.py" | tr '\n' ' ')
echo "[exact] $(date +%H:%M:%S) pausing depth-benchmark eval pids: ${PIDS:-none}"
[ -n "$PIDS" ] && kill -STOP $PIDS
trap '[ -n "$PIDS" ] && kill -CONT $PIDS; echo "[exact] $(date +%H:%M:%S) resumed: ${PIDS:-none}"' EXIT

run() { tag=$1; shift; echo "######## $tag $(date +%H:%M:%S)"; $PY frontier_eval.py "$@" --exact_only 300 --out eval_frontier/exact_$tag.json || echo "FAILED $tag"; }

# headline depth first: Qwen3-0.6B s=32 rerun, gate passed at L=256 (W=800)
run qwen06b_s32_L256   --ckpt $S/job_qwen06b_step32_rerun/stage_checkpoints/stage_8_step_41025_L256  --model_name Qwen/Qwen3-0.6B       --L 256 --n 2048 --max_lookahead 320
run pythia28b_L133     --ckpt $S/job_11647896/stage_checkpoints/stage_133_step_60025_L133          --model_name EleutherAI/pythia-2.8b --L 133 --n 1536 --max_lookahead 256
run pythia14b_L75      --ckpt $S/job_15232695/stage_checkpoints/stage_75_step_2375625_L75          --model_name EleutherAI/pythia-1.4b --L 75  --n 576  --max_lookahead 96
run qwen06b_s8_L208    --ckpt $S/job_qwen06b_step8_rerun/stage_checkpoints/stage_26_step_43100_L208 --model_name Qwen/Qwen3-0.6B       --L 208 --n 2048 --max_lookahead 320
run qwen06b_s1_L88     --ckpt $S/job_qwen06b_step1_rerun/stage_checkpoints/stage_88_step_32850_L88  --model_name Qwen/Qwen3-0.6B       --L 88  --n 2048 --max_lookahead 320
# gate window 200 (looser) for contrast
run qwen06b_s16w200_L160 --ckpt $S/job_qwen06b_step16_w200/stage_checkpoints/stage_10_step_39700_L160 --model_name Qwen/Qwen3-0.6B   --L 160 --n 3079 --max_lookahead 512
echo "######## ALL DONE $(date +%H:%M:%S)"
