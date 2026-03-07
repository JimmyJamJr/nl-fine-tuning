#!/bin/bash
#SBATCH -J eval_ckpts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --partition=ai
#SBATCH -A asaparov
#SBATCH -q preemptible
#SBATCH --mail-user=mnickel@purdue.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out
#SBATCH --requeue

set -euo pipefail

mkdir -p ./slurm
echo "JOB START $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Environment
module load conda
source /home/mnickel/miniconda3/etc/profile.d/conda.sh
conda activate pptrain

export SCRATCH="/scratch/gautschi/$USER"
export HF_HOME="$SCRATCH/model_cache"
export PYTHONUNBUFFERED=1

cd /home/mnickel/git/nl-fine-tuning/nl/pre_pretrain/scripts

python eval_checkpoints.py \
    --run_dirs "$SCRATCH/pretrain_fresh_2000_steps" \
               "$SCRATCH/pretrain_mix_25%" \
               "$SCRATCH/pretrain_from_pptrain_25%" \
    --eval_data "$SCRATCH/data/nl_splits/eval.jsonl" \
    --output_dir "$SCRATCH/eval_results_25%" \
    --batch_size 64

echo "JOB END $(date)"
