#!/bin/bash
#SBATCH -J nl_sparse_1g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=1:30:00
#SBATCH --partition=ai
#SBATCH -A asaparov
#SBATCH -q normal
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out
#
# 1-GPU validation of the sparse head (2026-10-09). Each rank's trajectory is independent (no cross-rank gradient
# averaging, per-rank seeded data), so a 1-GPU arm reproduces rank 0 of the morning's 4-GPU arms exactly. Arms (all
# NL_PACKING=hf, TAG=_1g, same per-GPU batch 48 x GA 4):
#   parity, deterministic: full head vs sparse head, FA2, L=16 (40 steps) and L=128 (20 steps); the full-head 1-GPU arm
#     must match rank 0 of hf_fa2_det bitwise ([PARAM-SYNC] first entry), the sparse arm within bf16 rounding.
#   speed: full vs sparse head, FA3, L=16 and L=128, 120 steps, same GPU back to back.
# Compare with: python bench/compare_sparse_1gpu.py
set -uo pipefail
cd /home/huan2073/nl-fine-tuning/nl
module load conda
conda activate search
export NGPUS=1
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
echo "===== PARITY (deterministic) ====="
for head in full sparse; do
  NL_HEAD=$head DETERMINISTIC=1 TAG="_${head}_1g" bash bench/run_arm.sh hf_fa2 16 40
done
for head in full sparse; do
  NL_HEAD=$head DETERMINISTIC=1 TAG="_${head}_1g" bash bench/run_arm.sh hf_fa2 128 20
done
echo "===== SPEED (120 steps) ====="
for L in 16 128; do for head in full sparse; do
  NL_HEAD=$head TAG="_${head}_1g" bash bench/run_arm.sh hf_fa3 "$L" 120
done; done
echo "SPARSE-1G DONE $(date)"
