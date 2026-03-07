# Pre-Pretraining Implementation

## Research Question

Does pre-pretraining on synthetic curriculum data improve sample efficiency compared to standard pretraining?

## Model

- **Architecture**: Pythia-160M (160 million parameters)
- **Initialization**: Random weights

## Experimental Conditions

| Condition | Phase 1 | Phase 2 | Job Script |
|-----------|---------|---------|------------|
| **A** | Curriculum pptrain | 95% C4 + 5% synthetic | `pretrain_from_pptrain_job.sh` |
| **B** | — | 100% C4 | `pretrain_fresh_job.sh` |
| **C** | — | 95% C4 + 5% synthetic | `pretrain_mix_job.sh` |

---

## Directory Structure

```
nl/
├── nl_generator.py        # C++ generator wrapper (compiles generator.cpp)
├── generator.cpp          # C++ source for fast generation
├── common.py              # Shared utilities (model loading, tokenization)
├── synthetic.py           # Synthetic data generation (SyntheticNL, build_heldout_set)
├── pptrain.py             # Phase 1: Pre-pretraining with curriculum
├── pretrain.py            # Phase 2: Standard pretraining on C4
└── pre_pretrain/
    ├── IMPLEMENTATION.md  # This file
    ├── jobs/              # SLURM job scripts
    │   ├── pptrain_job.sh
    │   ├── pretrain_fresh_job.sh
    │   ├── pretrain_mix_job.sh
    │   └── pretrain_from_pptrain_job.sh
    └── scripts/           # Utility scripts
        ├── generate_eval.py   # Generate eval.jsonl
        └── prepare_c4.py      # Tokenize C4 dataset
```

---

## Phase 1: Pre-Pretraining (pptrain)

Curriculum learning on synthetic NL data. Stages advance (4→8→12→...→32) when training accuracy reaches 98% over a rolling window.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Effective batch | 512 examples (16 micro × 4 accum × 8 GPUs) |
| Tokens per step | ~1M |
| Learning rate | 3×10⁻⁴ (linear warmup 500 steps) |
| Max lookahead | 32 (max_input_size = 192) |
| Curriculum threshold | 98% training accuracy |
| Curriculum window | 1000 examples (rolling) |

### Run

```bash
cd ~/git/nl-fine-tuning/nl/pre_pretrain
sbatch jobs/pptrain_job.sh
```

---

## Phase 2: Pretraining

Standard pretraining on C4 with optional synthetic data mixed in.

### Hyperparameters (Updated for Scaling Laws)

| Parameter | Value |
|-----------|-------|
| Total steps | 2000 |
| Tokens per step | ~2M (16 micro × 8 accum × 8 GPUs × 2048 seq) |
| **Total tokens** | **4B** |
| Learning rate | 6×10⁻⁴ with cosine decay |
| Min LR ratio | 0.1 |
| Warmup steps | 100 (5%) |
| Synthetic mix | 5% (Conditions A & C) |

### Run Jobs

```bash
cd ~/git/nl-fine-tuning/nl/pre_pretrain

# Condition A: Pre-pretrained + 5% synthetic (from latest pptrain checkpoint)
sbatch jobs/pretrain_from_pptrain_job.sh

# Condition B: Baseline (100% C4, fresh model)
sbatch jobs/pretrain_fresh_job.sh

# Condition C: Fresh model + 5% synthetic
sbatch jobs/pretrain_mix_job.sh
```

---

## Data

### Synthetic Evaluation Set

2000 examples at alpha=1.0 (full difficulty) for evaluating model checkpoints.

**Location**: `/scratch/gautschi/mnickel/data/nl_splits/eval.jsonl`

**Generate** (if needed):
```bash
cd ~/git/nl-fine-tuning/nl/pre_pretrain
python scripts/generate_eval.py --max_lookahead 32 --size 2000
```

### C4 Dataset

Tokenized C4 data packed into fixed-length blocks.

**Location**: `/scratch/gautschi/mnickel/data/c4_tokenized/`

**Prepare** (if needed):
```bash
cd ~/git/nl-fine-tuning/nl/pre_pretrain
python scripts/prepare_c4.py
```

---

## Key Constraints

1. **max_input_size = 6 × max_lookahead** (enforced everywhere)
2. **Synthetic seed differs from pptrain seed** (99999 vs 1337) to ensure novel examples
3. **eval.jsonl used for data leakage prevention** — prompts excluded from training generation

---

## Output Locations

```
/scratch/gautschi/mnickel/
├── data/
│   ├── nl_splits/eval.jsonl    # Eval examples
│   └── c4_tokenized/           # Packed C4 blocks
├── pptrain/                    # Phase 1 outputs
│   ├── checkpoints/
│   └── logs/
├── pretrain_fresh/             # Condition B
├── pretrain_mix/               # Condition C
└── pretrain_from_pptrain/      # Condition A
```

---

## Monitoring

```bash
squeue -u $USER                    # Job status
tail -f slurm/<jobid>_*.out        # Watch output
scancel <jobid>                    # Cancel job
```

---

## Scaling Laws Reference

Based on Kaplan (2020) and Chinchilla (2022):

| Parameter | Source | Value for 160M |
|-----------|--------|----------------|
| Optimal tokens | Chinchilla (D = 20N) | 3.2B |
| Minimum tokens | Kaplan (D = 5000×N^0.74) | 4B |
| **Training config** | User decision | **4B tokens** |
| Learning rate | Empirical | 6×10⁻⁴ |
| Batch size | Match LR | 2M tokens |
