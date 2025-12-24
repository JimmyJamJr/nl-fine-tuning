#!/bin/bash

#SBATCH -J instruct_dataset_download        # job name
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
echo "Tuning for qwen"
nvidia-smi

CMD="python instruct_data_analysis.py \
    --sample_size=200000 \
    --n_stages=10 \
    --tokenizer=Qwen/Qwen3-4B-Instruct-2507 \
    --output_dir=./curriculum_cutoffs \
    --cache_dir=/scratch/gautschi/huan2073/ \
    --length_weight=0.5 \
    --vocab_weight=0.5"

echo "RUNNING COMMAND:"
echo "$CMD"
eval $CMD