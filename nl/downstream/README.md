# Downstream Eval Framework

Driver + benchmark suite for evaluating pretrained / curriculum-fine-tuned
causal LMs on 14 reasoning benchmarks. Designed for Qwen 0.6B / 1.7B and
Pythia 160M / 410M but works with any HuggingFace causal LM.

## Files

| File | Purpose |
| --- | --- |
| `eval_downstream.py` | Main eval driver. 14 primary + 5 few-shot variants (19 total). |
| `eval_full_n1000.sh` | Gautschi sbatch launcher for full `n=1000` eval (1 GPU, `smallgpu`, 8h walltime). Takes `$MODEL` env var. |
| `eval_smoketest.sh` | Quick `n=20` smoke test of the full benchmark suite (Gautschi `smallgpu`). |
| `eval_downstream.sh` | Legacy Gautschi launcher (`ai` partition, h100). Kept for reference; use `eval_full_n1000.sh` for current runs. |
| `eval_downstream_gilbreth.sh` | Gilbreth sbatch launcher (A30, `-p a30 -q standby`, 3h55m). |
| `submit_parallel.sh` | Gilbreth parallel-submit (one job per model). |
| `setup_cache_datasets.py` + `cache_datasets.sh` | One-shot: pre-download all HF datasets + filter Lichess puzzles into a local Parquet. Run once per cluster. |
| `prompts/` | Fallback hand-crafted few-shot prompts (ProofWriter/StepGame) + LegalBench author templates (auto-downloaded from HazyResearch repo on first use). |
| `results/` | Per-run output JSONs. |
| `RESULTS.md` | Hand-compiled results summary. |

## Supported benchmarks (14 primary + 5 few-shot variants)

Each primary adapter returns **three metrics in one forward pass**:
- `loglik_accuracy`: log-likelihood argmax over candidate labels (lm-eval-harness convention)
- `calibrated_accuracy`: contextual calibration (Zhao et al. ICML 2021) — subtract model's content-free prior
- `gen_accuracy`: constrained greedy — argmax over the full next-token distribution, mapped to a valid candidate

Gen-canonical benchmarks (`nlgraph_gen`, `legal`) use their authors' own generation-based scorers instead.

### Primary adapters (14)

| Key | Protocol | Canonical source | # classes | Scoring |
|---|---|---|---|---|
| `proofwriter` | 8-shot Direct (no CoT) | Saparov & He 2023 (CoT) — deviation | 3 (True/False/Unknown) | fused |
| `proofwriter_cwa` | 8-shot Direct | same | 2 | fused |
| `prontoqa_ood` | 8-shot Direct (no CoT demos) | Saparov 2023 (CoT) — deviation | 2 | fused, per-variant |
| `clutrr` | 0-shot | none (original: GNN-based) | 21 kinship | fused, per-k |
| `folio` | 8-shot from train | Han et al. EMNLP 2024 ✓ canonical | 3 | fused |
| `logiqa` | 0-shot | AGIEval log-lik ✓ canonical | 4 | fused |
| `ruletaker` | 0-shot | none (original: fine-tuning) | 2 | fused, per-depth |
| `logicbench_bqa` | 3-shot Direct | Parmar et al. ACL 2024 FS-Direct ✓ | 2 | fused |
| `logicbench_mcqa` | 3-shot Direct | same ✓ | 4 | fused |
| `multilogieval` | 0-shot Direct | Patel et al. EMNLP 2024 (0-shot CoT) — deviation | 2 | fused, per-(depth,logic) |
| `zebra_mc` | 0-shot | WildEval grid-JSON (we use MC mode) | 5–6 per puzzle | fused, per-size |
| `stepgame` | 8-shot from train | none (original: fine-tuning) | 8 directions | fused |
| `nlgraph_gen` | canonical 4-shot Direct | Wang et al. NeurIPS 2023 ✓ | varies per task | per-task author scorers (path validation, regex per task) |
| `legal` | canonical 6-shot author demos | Guha et al. NeurIPS 2023 ✓ | varies per subtask | `balanced_accuracy_score` + `normalize()` |

### Few-shot variants (5)

Alternative prompting for benchmarks that are 0-shot by default. Same metrics, demo block prepended:

| Key | Protocol | Rationale |
|---|---|---|
| `ruletaker_fs` | 5-shot Direct | Tests whether format-teaching helps escape class collapse |
| `clutrr_fs` | 5-shot Direct (stratified by k) | Same |
| `logiqa_fs` | 3-shot Direct | Deviates from AGIEval 0-shot canonical; alternative view |
| `multilogieval_fs` | 3-shot Direct | Same |
| `zebra_mc_fs` | 3-shot (demos from small 2×2/2×3 puzzles) | Same |

### Dropped (for reference)

- `chess_mate` — 0% (exact + first_move) across all 12 tested models; no signal at 0.6–1.7B scale.
- `stepgame_gen` — redundant with fused `stepgame` (pre-consolidation artifact).
- `grapharena_gen` — needs `rdkit` + `build_dataset.py` generation; no pre-built data.
- `blocksworld` / `mystery_blocksworld` / `logistics` (+`_first`, `_logprob`) — PlanBench, not in paper benchmark set.
- `bbh` / `bbh_cot` / `bbh_cot_chat` / `standard` — BBH & standard NLU, not in paper benchmark set.
- `game24` — too hard at 0.6B, consistently <5%.

### Test set sizes

- Small (≤1500): FOLIO val 203, LogiQA test 651, LogicBench MCQA 500, LogicBench BQA 1520, LegalBench subtasks 96–300 each, Multi-LogiEval ~2000 after yes/no filter
- Medium: NLGraph 1000 (8 tasks), ZebraLogic-MC 3259
- Large: ProofWriter OWA/CWA (20K × 6 depths each), RuleTaker 150K, CLUTRR 70K, StepGame 100K

`n=1000` cap is applied **per benchmark, stratified across subsets** via `_distribute_n` + seeded random `_subsample(seed=42)`.

## Usage

```bash
# Prereq: one-time dataset caching (run once per cluster)
sbatch cache_datasets.sh

# Smoke test (n=20, single model, all 19 benchmarks)
sbatch eval_smoketest.sh

# Full n=1000 eval for one model (Gautschi smallgpu)
MODEL=base sbatch eval_full_n1000.sh
MODEL=6pct_L32 sbatch eval_full_n1000.sh
MODEL=6pct_1.7b_L32 sbatch eval_full_n1000.sh

# Parallel sweep on Gilbreth (4 models × 4 A30 GPUs, fully parallel)
bash submit_parallel.sh
```

Models are resolved via `MODEL_REGISTRY` in `eval_downstream.py`:
- `base`, `base_1.7b` → HF repo (`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`)
- `instruct_only` → `_latest_ckpt("9152198")` with fallbacks
- `6pct_L{4,6,8,16,32,48,64,75}` → `_stage_ckpt("8894380", L)` or `_stage_ckpt("9001346", L)`
- `6pct_1.7b_L{8,16,32}` → `_stage_ckpt("runpod_qwen17b_L32_20260418_085434", L)`

Override with explicit `alias=path` via `--models` CLI flag.

## Environment

```bash
export SCRATCH=/scratch/gautschi/$USER        # or /scratch/gilbreth/$USER
export HF_HOME=$SCRATCH/model_cache
export HF_DATASETS_OFFLINE=1                  # use pre-cached datasets, no network
export DATA_DIR=$SCRATCH/nl_eval              # local paths for PrOntoQA zip, LogicBench/NLGraph/MultiLogiEval clones, chess parquet
```

HF token auto-propagates from `~/.cache/huggingface/token` if present. All
SLURM launchers (`eval_full_n1000.sh`, `eval_smoketest.sh`, `cache_datasets.sh`,
`eval_downstream_gilbreth.sh`) handle this automatically.

## Output format

One JSON per SLURM job at `results/eval_n1000_${MODEL}_${SLURM_JOB_ID}.json`:

```json
{
  "base": {
    "_path": "Qwen/Qwen3-0.6B",
    "_chat_template": true,
    "proofwriter": {
      "0": {
        "loglik_accuracy": 0.395,
        "gen_accuracy": 0.400,
        "calibrated_accuracy": 0.550,
        "total": 167,
        "per_class_loglik": {"True": {...}, "False": {...}, "Unknown": {...}}
      },
      "...": "...",
      "overall": {"..., baseline_logprobs": {"True": -18.1, "False": -13.9, "Unknown": -11.9}}
    },
    "legal": {"overall": {"accuracy": 0.438, "total": 1000}, "diversity_1": {...}, ...}
  }
}
```

See `RESULTS.md` for human-readable summary tables.

## Canonical sources vs. deviations

| Benchmark | Canonical method | Our approach | Deviation reason |
|---|---|---|---|
| NLGraph | 4-shot Direct + per-task regex (Wang 2023) | **canonical** | none |
| LegalBench | 6-shot author demos + balanced_accuracy (Guha 2023) | **canonical** | none |
| LogiQA | 0-shot log-lik (AGIEval) | **canonical** | none |
| FOLIO | 8-shot from train (Han 2024) | matches few-shot protocol; log-lik scoring instead of gen | author uses gen, we can't without CoT banned |
| LogicBench BQA/MCQA | 0-shot CoT OR 3-shot Direct (Parmar 2024) | **3-shot Direct** ✓ (matches one of paper's 4 conditions) | none |
| Multi-LogiEval | 0-shot CoT (Patel 2024) | 0-shot Direct | CoT banned per our paper framing |
| ProofWriter OWA/CWA | 8-shot CoT (Saparov & He 2023) | 8-shot Direct | CoT banned |
| PrOntoQA-OOD | 8-shot CoT (Saparov 2023) | 8-shot Direct | CoT banned |
| CLUTRR | original: GNN; no LLM canonical | 0-shot + calibration | none exists |
| RuleTaker | original: fine-tuning; no LLM canonical | 0-shot + calibration | none exists |
| StepGame | original: fine-tuning; no LLM canonical | 8-shot from train | none exists |
| ZebraLogic | grid-JSON output (Lin 2024) | MC mode log-lik | we use mc_mode variant, not grid_mode |
