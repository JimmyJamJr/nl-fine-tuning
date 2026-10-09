#!/bin/bash
# Frontier accuracy vs the gate itself (2026-09-23): does a looser window (W=200) or a stricter
# threshold (0.99) change how well a checkpoint knows the frontier when its gate passes?
# Each checkpoint gets the frontier set (1000 uniform + 300 with la >= 0.9L) and 300 items at la == L.
# Pairs differ only in the gate (verified from run_meta; same model, n, cap, step, chat, mix):
#   window  s=8 : W=200 branch passing L=216 (the W=800 parent never passed it) vs W=800 at L=208 (done
#                 in run_frontier_eval.sh / run_frontier_exact.sh)
#           s=16: W=200 branch at L=160 (done there) vs W=800 run at L=144 (below)
#   thresh  Qwen3-0.6B 0.99 vs 0.98 at L=16 and L=22 (n=576, W=200, chat, 6% Dolci; identical otherwise)
#           Qwen3-1.7B 0.99 vs 0.98 at L=32 (n=780 vs 288; graph size at a given L is n-invariant under
#           linear lookahead, and L=32 is below the n=288 ceiling of 46)
# Runs on g005 with priority (depth-benchmark evals paused, resumed on exit).
cd /home/huan2073/nl-fine-tuning/nl
export HF_HOME=/scratch/gautschi/huan2073/model_cache TORCH_HOME=/scratch/gautschi/huan2073/model_cache
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=$HOME/.conda/envs/search/bin/python
S=/scratch/gautschi/huan2073/nl_output/search
PIDS=$(pgrep -u "$USER" -f "eval_downstream.py" | tr '\n' ' ')
echo "[gates] $(date +%H:%M:%S) pausing depth-benchmark eval pids: ${PIDS:-none}"
[ -n "$PIDS" ] && kill -STOP $PIDS
trap '[ -n "$PIDS" ] && kill -CONT $PIDS; echo "[gates] $(date +%H:%M:%S) resumed: ${PIDS:-none}"' EXIT

both() { tag=$1; shift
  echo "######## $tag $(date +%H:%M:%S)"
  $PY frontier_eval.py "$@" --out eval_frontier/$tag.json || echo "FAILED $tag"
  $PY frontier_eval.py "$@" --exact_only 300 --out eval_frontier/exact_$tag.json || echo "FAILED exact_$tag"; }

Q06=Qwen/Qwen3-0.6B; Q17=Qwen/Qwen3-1.7B
# threshold pairs first (requested first); the window pairs follow
both qwen06b_thr99_L16 --ckpt $S/job_qwen06b_curr48_thr99/stage_checkpoints/stage_16_step_14900_L16   --model_name $Q06 --L 16 --n 576 --max_lookahead 96 --chat
both qwen06b_thr98_L16 --ckpt $S/job_retrain_curr96_s1_dolci6/stage_checkpoints/stage_16_step_8525_L16  --model_name $Q06 --L 16 --n 576 --max_lookahead 96 --chat
both qwen06b_thr99_L22 --ckpt $S/job_qwen06b_curr48_thr99/stage_checkpoints/stage_22_step_21100_L22   --model_name $Q06 --L 22 --n 576 --max_lookahead 96 --chat
both qwen06b_thr98_L22 --ckpt $S/job_retrain_curr96_s1_dolci6/stage_checkpoints/stage_22_step_11500_L22 --model_name $Q06 --L 22 --n 576 --max_lookahead 96 --chat
both qwen17b_thr99_L32 --ckpt $S/job_curr48_thr99/stage_checkpoints/stage_32_step_21850_L32 --model_name $Q17 --L 32 --n 780 --max_lookahead 128 --chat
both qwen17b_thr98_L32 --ckpt $S/job_retrain_17b_curr48_s1_dolci6/stage_checkpoints/stage_32_step_12175_L32 --model_name $Q17 --L 32 --n 288 --max_lookahead 48 --chat
both qwen06b_s8w200_L216 --ckpt $S/job_qwen06b_step8_w200/stage_checkpoints/stage_27_step_46025_L216 --model_name $Q06 --L 216 --n 2048 --max_lookahead 320
both qwen06b_s16w800_L144 --ckpt $S/job_qwen06b_step16_rerun/stage_checkpoints/stage_9_step_28875_L144 --model_name $Q06 --L 144 --n 3079 --max_lookahead 512
echo "######## ALL DONE $(date +%H:%M:%S)"
