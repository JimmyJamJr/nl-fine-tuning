#!/bin/bash
#SBATCH -J cache_downstream_datasets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH -A asaparov
#SBATCH -q standby
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out

# One-shot: download all HF datasets used by eval_downstream.py into $SCRATCH,
# filter Lichess/chess-puzzles to mate-in-1/2/3 and cache locally as Parquet.

set -euo pipefail
module load conda
conda activate search

export SCRATCH="/scratch/gautschi/$USER"
export HF_HOME="$SCRATCH/model_cache"
export HF_HUB_OFFLINE=0
export DATA_DIR="$SCRATCH/nl_eval"
export PYTHONUNBUFFERED=1
mkdir -p slurm "$HF_HOME" "$DATA_DIR"

if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

cd /home/huan2073/nl-fine-tuning/nl/downstream
python setup_cache_datasets.py

echo "DONE $(date)"
