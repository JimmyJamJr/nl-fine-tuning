#!/bin/bash

#SBATCH -J tokenize        # job name
#SBATCH --mem=32G                     # total RAM
#SBATCH --gres=gpu:1                  # number of GPUs
#SBATCH -o ./slurm/%j_%x.out    # STDOUT (%x=jobname, %j=jobid)
#SBATCH -e ./slurm/%j_%x.out    # STDERR
#SBATCH --time=12:00:00
#SBATCH --partition=ai
#SBATCH --cpus-per-task=14
#SBATCH -A asaparov
#SBATCH -q preemptible
#SBATCH --mail-user=huan2073@purdue.edu
#SBATCH --mail-type=END,FAIL

module load conda
conda activate search

echo "JOB START $(date)"
echo "Pre tokenizing for qwen"
nvidia-smi

export HF_TOKEN=""

CMD="python pretokenize_instruct_data.py"

echo "RUNNING COMMAND:"
echo "$CMD"
eval $CMD