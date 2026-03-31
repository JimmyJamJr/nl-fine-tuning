#!/bin/bash

#SBATCH -J sym_sophia_amp_compile
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o slurm/%j_%x.out
#SBATCH -e slurm/%j_%x.out
#SBATCH --time=96:00:00
#SBATCH --partition=ai
#SBATCH --cpus-per-task=14
#SBATCH -A asaparov
#SBATCH -q preemptible
#SBATCH --mail-user=huan2073@purdue.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180

set -euo pipefail

# ========== Preemption handling ==========
trap 'echo "[SIG] USR1 @ $(date) — grace period"; sleep 120; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
trap 'echo "[SIG] TERM @ $(date)"; exit 0' TERM

mkdir -p slurm
module load conda
conda activate search

echo "JOB START $(date)"
echo "Search task: SophiaG (hessian) + AMP + torch.compile"
nvidia-smi

CMD="python train_search_curriculum.py \
    --max-input-size=256 \
    --dataset-size=-1 \
    --max-lookahead=40 \
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
    --loss=ce \
    --base-lookahead=2 \
    --lookahead-step=2 \
    --accuracy-threshold=0.98 \
    --output-dir=/scratch/gautschi/huan2073/symbolic_results \
    --optimizer=sophiag_hessian --use-amp --use-compile"

echo "RUNNING COMMAND:"
echo "$CMD"
eval $CMD
