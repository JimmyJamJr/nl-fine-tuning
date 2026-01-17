salloc -A asaparov -p ai -N 1 --ntasks=1 --gres=gpu:8 -c 112 --time=01:00:00
source ~/miniconda3/etc/profile.d/conda.sh (might not need)
conda activate pptrain
RUN_DIR=/scratch/gautschi/mnickel/runs/pythia160_smoke
mkdir -p "$RUN_DIR"

accelerate launch \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision bf16 \
  train_stream.py \
  --stage prepretrain \
  --run_dir "$RUN_DIR" \
  --max_steps 50 \
  --micro_batch 2 \
  --seq_len 256


# IMPORTANT
2. Ensure Multi process working correctly with data iterator
4. Track Loss on Log scale, Track accuracy on dataset over time

# ISSUES
Currently taking the first valid anaswer for each example
Need to garuntee all examples can fit in our sequence length
Loss is only on the prediction right now (first prediction)