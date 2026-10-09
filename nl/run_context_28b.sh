#!/bin/bash
# Rerun of the Pythia-2.8B context-length test after its OOM at 03:13 (run_context_length_eval.sh);
# same arguments. No pause/resume here: run_context_length_eval.sh (running) and
# run_frontier_exact.sh (chained) already pause the depth-benchmark evals.
cd /home/huan2073/nl-fine-tuning/nl
export HF_HOME=/scratch/gautschi/huan2073/model_cache TORCH_HOME=/scratch/gautschi/huan2073/model_cache
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
S=/scratch/gautschi/huan2073/nl_output/search
echo "######## pythia28b_L133 $(date +%H:%M:%S)"
$HOME/.conda/envs/search/bin/python context_length_eval.py --ckpt $S/job_11647896/stage_checkpoints/stage_133_step_60025_L133 \
    --model_name EleutherAI/pythia-2.8b --n 1536 --max_lookahead 256 --stage_L 133 \
    --la_lo 30 --la_hi 70 --n_natural 600 --la 40,60 --n_base 150 --targets 1500,1900,2300,2700,3100,3500 \
    --out eval_context/pythia28b_L133.json || echo "FAILED pythia28b"
echo "######## DONE $(date +%H:%M:%S)"
