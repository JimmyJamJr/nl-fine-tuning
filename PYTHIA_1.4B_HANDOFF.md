# Pythia 1.4B curriculum training — handoff instructions

Goal: train `EleutherAI/pythia-1.4b` on the graph-search curriculum task,
targeting lookahead L=64 as a first experiment. See if it clears the L=48-56
capacity wall that Pythia 410M hit.

## Expected outcome

- **Best case**: clears L=64 comfortably (~100K-200K PFLOPs total), extend to L=96
- **Likely**: clears L=32 or 48, then hits wall similar to 410M. Documents capacity
  scaling from 410M (wall@L=56) → 1B (wall@L=96 based on nocurr) → 1.4B (unknown)
- **Training time**: ~2-3 days on 4× H100, possibly longer on smaller GPUs

## What you need

1. **A compute allocation with ≥4 GPUs**. Config written for 4× H100 80GB but
   works on 4× A100 80GB/40GB with minor tweaks (see below).
2. **SLURM cluster access** (same as typical HuggingFace training setup).
3. The repo cloned somewhere:
   ```bash
   git clone https://github.com/JimmyJamJr/nl-fine-tuning.git  # or wherever
   cd nl-fine-tuning/nl
   ```
4. **A conda env** named `search` (or edit the script). Required packages:
   - PyTorch (2.0+ with CUDA)
   - transformers, datasets
   - pybind11 (for C++ generator)
   - `pip install liger-kernel` (optional, not used for Pythia anyway)
5. **Gautschi/Purdue-style scratch dir** at `/scratch/<cluster>/$USER`, or
   edit `SCRATCH=` in the script.

## Steps to run

```bash
# 1. Get the repo onto your cluster
git clone <repo-url> nl-fine-tuning
cd nl-fine-tuning/nl

# 2. Create conda env (one-time)
conda create -n search python=3.11 -y
conda activate search
pip install torch transformers datasets accelerate pybind11 numpy pandas matplotlib
# plus whatever else the repo needs — check for requirements.txt if present

# 3. Build the C++ graph generator (script auto-tries this, but can pre-run)
python nl_generator.py

# 4. Edit tuning_job_pythia1.4b_curr_L64_step8.sh for your cluster
#    Key things to change:
#    - Top SBATCH directives (partition, QoS, account, --mem rules)
#    - SCRATCH path near the top
#    - Conda env name (if not "search")

# 5. Submit
mkdir -p slurm
sbatch tuning_job_pythia1.4b_curr_L64_step8.sh

# 6. Monitor
squeue -u $USER
tail -f slurm/<job_id>_nl_pythia1.4b_curr_L64_step8.out
```

## Config summary

All of these are defaults in the script; don't change unless needed:

| Hyperparameter | Value | Rationale |
| --- | --- | --- |
| LR | 5e-5 | Optimal at 410M and 1B scales |
| Per-GPU batch | 96 | On 80GB GPU, uses ~33GB with GC off |
| Grad accum | 1 | 4×96×1 = 384 effective batch (matches Pythia 1B optimal) |
| Target L | 64 | Pythia 410M walled at L=56; trying one step beyond |
| Curriculum step | 8 | Matches all our other baselines (Pythia 410M, 1B; Qwen 0.6B) for direct comparability. If 1.4B hits a wall around L=48-56 like 410M did, rerun with step=4 as a follow-up. |
| Accuracy threshold | 0.98 | Stage-advance threshold |
| Gradient checkpointing | off | Fits without; GC adds ~33% compute for no gain |
| Liger kernels | off | Not supported for Pythia (Qwen only) |
| Chunked CE | on, chunk=4096 | Pythia has 50K vocab; avoids VRAM spikes |

## If GPUs are smaller than 80GB

**On 40GB GPUs** (e.g. A100 40GB): memory is tighter. Either:
- Enable GC: set `GRADIENT_CHECKPOINTING=true` in the script (~33% slower per step, saves ~10GB activations)
- Or reduce batch: set `BATCH_SIZE=48` and `GRADIENT_ACCUMULATION_STEPS=2` to keep eff_batch=384

Memory estimate with `BS=96`, `GC=false`:
- Model state (bf16 weights + fp32 AdamW + grads): ~22 GB
- Activations (bs=96, seq=384, hidden=2048, 24 layers): ~7 GB
- Total: ~30 GB per GPU → comfortable on 80GB, tight but possible on 40GB with GC off

## Output

Results go to `$SCRATCH/nl_output/search/job_<jobid>/` with:
- `loss_history.jsonl` — per-step training metrics (for plotting)
- `stage_eval_history.json` — per-stage eval accuracy snapshots
- `checkpoint-<N>/` directories — HF Trainer checkpoints (model + optimizer state)
- `stage_checkpoints/stage_<L>_*/` — inference-ready checkpoints per curriculum stage
- Various `*.png` plots of loss / compute / lookahead

**What to send back**:
- The full `$SCRATCH/nl_output/search/job_<jobid>/` directory (rsync'd to Huan)
- Or at minimum: `loss_history.jsonl`, `stage_eval_history.json`, `run_meta.json`,
  and one final checkpoint (if model weights are wanted for eval later)

## How to tell it's working

- Stage 1 (L=8) should clear in <1000 steps with >98% full_word_acc
- Each subsequent stage takes longer; at L=32+ expect 2K-5K steps per stage
- If you see `full_word_acc` hovering below 0.9 for 1000+ steps at any stage,
  the model is walling — note the L, save the log, stop the run
- Compare trajectory to Pythia 410M run (job 8478416→8566793→8566793) at matching L
  for diagnosis

## Questions? Ping Huan.
