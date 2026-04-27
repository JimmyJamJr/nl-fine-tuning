# Qwen 0.6B step=16 curriculum extension — handoff instructions

Goal: extend `Qwen/Qwen3-0.6B` curriculum training from L=128 to **L=256**,
continuing the step=16 schedule. The original run (job_9273210, my cluster)
completed L=128 at step=16 in 21,925 steps on 2026-04-16. This handoff resumes
from that endpoint and adds 8 more curriculum stages: L=144, 160, 176, 192,
208, 224, 240, 256.

## Expected outcome

- **Best case**: clears L=144 → L=256 cleanly, total ~200-400 GPU-hours on 4×H100
- **Likely**: hits compute/capacity scaling at L=200+; may need to extend walltime
  or accept partial progress
- **What I've seen so far at 0.6B**: no walls observed at any L through L=128
  (smooth convergence per stage). Whether that persists at L=256 is the experiment.

## What you need

1. **A compute allocation with ≥4 GPUs**. Config written for 4× H100 80GB but
   works on 4× A100 80GB or 40GB with the gradient-checkpointing tweak (see below).
2. **SLURM cluster access** (same general setup as the Pythia 1.4B handoff).
3. **The repo cloned somewhere** — the core training code is committed at:
   ```bash
   git clone https://github.com/JimmyJamJr/nl-fine-tuning.git
   cd nl-fine-tuning/nl
   ```
   (`tuning_nl.py`, `nl_generator.py`, `generator.cpp`, `plot_training.py` are
   all up to date in main.)
4. **A conda env named `search`** (or edit the script). Required packages:
   - PyTorch (2.0+ with CUDA)
   - transformers, datasets, accelerate
   - pybind11 (for C++ generator)
   - **huggingface_hub** (for checkpoint download — usually included with `transformers`)
   - `pip install liger-kernel` (used here — Qwen supports it)
5. **HuggingFace access to the resume checkpoint** — the checkpoint is hosted at:
   - **`https://huggingface.co/JimmyJamJr/qwen06b-step16-L128-resume`** (private)
   - I'll add you as a collaborator OR send you a read-only token.
   - One-time setup: `huggingface-cli login` (paste the read token from `huggingface.co/settings/tokens`)
6. **Cluster scratch dir** at `/scratch/$USER` or similar. Edit `SCRATCH=` in
   the script if your cluster uses a different convention.

## What I'll send you

Just two things:

1. **Read-only HuggingFace token** for `JimmyJamJr/qwen06b-step16-L128-resume`
   (sent via Slack — small text string, just paste into `huggingface-cli login`)
2. **`tuning_job_qwen06b_curr256_step16_RESUME_jackie.sh`** — the SLURM script.
   You'll need to edit cluster-specific bits at the top.

The 3.4 GB checkpoint downloads automatically from HF the first time the script
runs. No tarball transfer needed.

## Steps to run

```bash
# 1. Get the repo onto your cluster
git clone https://github.com/JimmyJamJr/nl-fine-tuning.git
cd nl-fine-tuning/nl

# 2. Create conda env (one-time)
conda create -n search python=3.11 -y
conda activate search
pip install torch transformers datasets accelerate pybind11 numpy pandas matplotlib liger-kernel huggingface_hub

# 3. Authenticate to HuggingFace (one-time, with the read token I sent)
huggingface-cli login

# 4. Build the C++ graph generator (script auto-tries this, but can pre-run)
python nl_generator.py

# 5. Drop the SLURM script into the nl/ directory
cp ~/Downloads/tuning_job_qwen06b_curr256_step16_RESUME_jackie.sh .

# 6. Edit the SLURM script for your cluster
#    Key things to change:
#    - Top SBATCH directives (partition, QoS, account, --mem rules)
#    - SCRATCH path near the top (default: /scratch/$USER)
#    - Conda env name (if not "search")

# 7. Submit
mkdir -p slurm
sbatch tuning_job_qwen06b_curr256_step16_RESUME_jackie.sh

# 8. Monitor
squeue -u $USER
tail -f slurm/<job_id>_nl_qwen06b_curr256_step16_RESUME.out
```

The script auto-downloads the checkpoint (~3.4 GB from HuggingFace) into
`$SCRATCH/nl_output/search/job_9273210/` before training starts. The first run
takes ~5-10 minutes for the download depending on bandwidth; subsequent runs
skip the download since the data is cached.

## Why we use `--resume_from_job 9273210` not a raw path

The script downloads the checkpoint to `$SCRATCH/nl_output/search/job_9273210/`
specifically so that `plot_training.py --combined` can walk the resume chain
backward via `run_meta.json`. This means your final loss/compute/lookahead
plots will show **continuous progression from L=16 → L=128 → L=256**, not just
the L=144 → L=256 portion.

## Config summary

All of these are defaults in the script; don't change unless needed:

| Hyperparameter | Value | Rationale |
| --- | --- | --- |
| LR | 5e-5 | Same as the original run we're resuming |
| Per-GPU batch | 48 | Matches original (eff_batch=768) |
| Grad accum | 4 | 4×48×4 = 768 effective batch (matches original) |
| Target L | 256 | Extending from L=128 endpoint |
| Curriculum step | 16 | Matches original step=16 schedule |
| n_stages | 16 | 256/16 = 16 total stages (resumes at stage 9 of 16) |
| Accuracy threshold | 0.98 | Stage-advance threshold (matches original) |
| Max input size | **1536** | **CRITICAL: 6 × L = 6 × 256 = 1536**. Must be at least 6×L or alpha won't ramp to 1.0 at deep stages. |
| Gradient checkpointing | **on** | Recommended at L=256 / max_input=1536 (~2× activation memory vs the original L=128 run). Set false if you have memory headroom. |
| Liger kernels | on | Supported and used for Qwen |
| Chunked CE | on, chunk=4096 | Vocab is large; reduces memory spikes |

## If GPUs are smaller than 80GB

**On 40GB GPUs** (A100 40GB):
- Keep `GRADIENT_CHECKPOINTING=true` (already default)
- May need `BATCH_SIZE=32, GRADIENT_ACCUMULATION_STEPS=6` to keep eff_batch=768
- max_input=1536 is the heaviest demand — alternatively run with 8 GPUs:
  `BATCH_SIZE=24, GRADIENT_ACCUMULATION_STEPS=4` for eff_batch=768

**On 80GB GPUs (A100 / H100)**:
- Default config should fit
- If you have memory headroom, try `GRADIENT_CHECKPOINTING=false` for ~33% speed-up

Memory estimate with `BS=48, max_input=1536, GC=true`:
- Model state (bf16 weights + fp32 AdamW + grads): ~5 GB
- Activations with GC: ~15-25 GB depending on layer count
- Total: ~25-35 GB per GPU — comfortable on 80GB, tight on 40GB

## What carries over from the resume

The downloaded HuggingFace checkpoint includes everything needed for seamless resume:

| Item | File | Carried? |
|---|---|---|
| Model weights | `model.safetensors` | ✓ |
| AdamW optimizer state | `optimizer.pt` | ✓ |
| LR scheduler state | `scheduler.pt` | ✓ |
| Per-rank RNG state | `rng_state_X.pth` | ✓ |
| Training step count | `trainer_state.json` | ✓ |
| Curriculum stage + alpha | `curriculum_state.json` | ✓ |
| Training history (for cumulative plots) | `loss_history.jsonl`, `run_meta.json` | ✓ |

So your training picks up at L=128 weights, optimizer momentum continues, and
the curriculum knows it's at stage 8 → advances to stage 9 (L=144). All seamless.

## Output

Results go to `$SCRATCH/nl_output/search/job_<jobid>/` with:
- `loss_history.jsonl` — per-step training metrics (for plotting)
- `stage_eval_history.json` — per-stage eval accuracy snapshots
- `checkpoint-<N>/` directories — HF Trainer checkpoints (model + optimizer state)
- `stage_checkpoints/stage_<N>_step_<S>_L<L>/` — inference-ready per-stage checkpoints
- Various `*.png` plots of loss / compute / lookahead

**What to send back when done**:
- The full `$SCRATCH/nl_output/search/job_<jobid>/` directory (rsync to me)
- Or upload to a new HF repo (e.g. `JimmyJamJr/qwen06b-step16-L256`) for symmetry
- At minimum, send:
  - `loss_history.jsonl`, `stage_eval_history.json`, `run_meta.json`
  - `stage_checkpoints/stage_16_step_*_L256/` (if reached) — for downstream eval
  - `curriculum_state.json` (in case I want to extend further to L=512+)

## How to tell it's working

- First few hundred steps: loss should be **low (~0.01-0.05)** because we're
  resuming from a converged L=128 model. If you see high loss spikes, the
  checkpoint may not have loaded properly — check the resume path.
- Stage 9 (L=144) should clear in 1K-3K steps with >98% full_word_acc
- Each subsequent stage takes longer; expect 3K-10K steps per stage at L=200+
- Watch `effective_L` in the loss_history — it should ramp from 144 → 160 → 176 → … → 256
- If `full_word_acc` hovers below 0.9 for 5K+ steps at any stage, the model is
  walling. Note the L, save the log, and stop. That's the capacity ceiling
  finding either way.
- Compare trajectory to the original step=16 run (job_9273210 stages 1-8 at
  L=16 → 128) for relative pace.

## Differences vs the Pythia 1.4B handoff

This is structurally similar but easier in some respects:
- We're **resuming from a converged checkpoint**, not starting fresh
- Smaller model (0.6B vs 1.4B) → less memory but max_input is much larger (1536 vs 384)
- Liger kernels enabled (Qwen-supported), gradient checkpointing on (not off)
- **Checkpoint via HuggingFace, not tarball** — automatic download, resumable

## Questions? Ping Huan.
