#!/bin/bash

#SBATCH -J search_0.1alpha_mask_0.99        # job name
#SBATCH --mem=32G                       # total RAM
#SBATCH --gres=gpu:1                    # number of GPUs
#SBATCH -o slurm/%j_%x.out              # STDOUT (%x=jobname, %j=jobid)
#SBATCH -e slurm/%j_%x.out              # STDERR
#SBATCH --time=48:00:00
#SBATCH --partition=ai
#SBATCH --cpus-per-task=14
#SBATCH -A asaparov
#SBATCH -q preemptible
#SBATCH --mail-user=huan2073@purdue.edu
#SBATCH --mail-type=END,FAIL

module load conda
conda activate search

echo "JOB START $(date)"
echo "Search task with curriculum learning (alpha scaling), start alpha=0.1. Increase alpha at 0.99 training acc. With mask"
nvidia-smi

CMD="python train_search.py \
    --max-input-size=256 \
    --dataset-size=-1 \
    --max-lookahead=12 \
    --nlayers=6 \
    --nhead=1 \
    --hidden-dim=16 \
    --seed=42 \
    --bidirectional=n \
    --pos-emb=absolute \
    --learn-tok-emb=n \
    --toeplitz-attn=n \
    --toeplitz-reg=0.0 \
    --toeplitz-pos-only=n \
    --add-padding=n \
    --ablate=none \
    --preLN=y \
    --looped=n \
    --task=search \
    --distribution=crafted \
    --warm-up=0 \
    --batch-size=512 \
    --learning-rate=1e-5 \
    --update-rate=131072 \
    --grad-accumulation-steps=1 \
    --curriculum=y \
    --loss=bce"

echo "RUNNING COMMAND:"
echo "$CMD"
eval $CMD