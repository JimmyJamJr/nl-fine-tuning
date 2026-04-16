# Downstream Eval Framework

Benchmarks suite + driver for evaluating pretrained/fine-tuned models on
downstream tasks. Works with any HuggingFace causal LM (Qwen, Pythia, custom
checkpoints, etc.).

## Files

| File | Purpose |
| --- | --- |
| `eval_downstream.py` | Main eval driver. 23 benchmarks; one CLI, one output JSON. |
| `eval_downstream.sh` | Gautschi sbatch launcher (H100, `-p ai -q normal`, 16h walltime). |
| `eval_downstream_gilbreth.sh` | Gilbreth sbatch launcher (A30, `-p a30 -q standby`, 3h55m). |
| `prompts/` | Hand-crafted few-shot prompts + verbatim LegalBench author prompts. |
| `results/` | Per-run output JSONs (one per sbatch job). |
| `RESULTS.md` | Manually maintained human-readable results doc. Update by hand. |

## Supported benchmarks

23 benchmark keys total — many underlying datasets have two scoring variants
(our zero-shot log-lik version + a generation/paper-matching version).

### Reasoning & search-transfer (our additions)

Columns below are the `--benchmarks` keys for each scoring variant on the same
underlying dataset. A dash (`—`) means that variant doesn't exist for that dataset.

| Dataset | Sub-tasks | Log-lik (zero-shot, ours) | Generation / paper-matching | Log-prob of gold |
| --- | --- | --- | --- | --- |
| LegalBench (3 binary tasks) | hearsay, international_citizenship_questions, proa | `legal` — Yes/No log-lik w/ author prompts | `legal_gen` — few-shot generation (Guha 2023) | — |
| ProofWriter OWA | depths 0–5 | `proofwriter` — 3-class (T/F/Unknown) log-lik | `proofwriter_gen` — 3-shot generation | — |
| ProofWriter CWA | depths 0–5 | `proofwriter_cwa` — 2-class (T/F) log-lik | `proofwriter_cwa_gen` — 3-shot generation | — |
| ZebraLogic MC | overall | `zebra_mc` — 4-way log-lik | `zebra_mc_gen` — few-shot generation | — |
| StepGame | hops 1…10 | `stepgame` — 9-way log-lik | `stepgame_gen` — 5-shot generation (Li 2024) | — |
| Game of 24 (ToT hard subset) | puzzles 900–999 | — | `game24` — generation + expression eval | — |
| PlanBench Blocksworld | plan generation | — | `blocksworld` — gen + exact-match & prefix-match | `blocksworld_logprob` — mean log-prob of gold plan |
| PlanBench Mystery-BW | plan generation | — | `mystery_blocksworld` — gen + exact-match | `mystery_blocksworld_logprob` |
| PlanBench Logistics | plan generation | — | `logistics` — gen + exact-match | `logistics_logprob` |
| Chess mate-in-N (Lichess) | mate-in-1/2/3 | — | `chess_mate` — gen + exact-match | `chess_mate_logprob` — log-prob of gold moves |

**Scoring notes:**

- *Log-lik (ours)*: zero-shot comparison of log P(label | prompt) across the
  valid labels. Fast, no prompt engineering, works on every model (no chat
  template or CoT required). This is our main reasoning-transfer measurement.
- *Generation / paper-matching*: replicates the author's setup as closely as
  possible without fine-tuning or CoT thinking — typically few-shot prompting
  plus answer-extraction. Slower and higher variance but comparable to
  published numbers.
- *Log-prob of gold*: mean per-token log-probability of the gold solution.
  Useful for tasks where the solution is a long structured sequence (plans,
  move lists) and exact-match is too brittle; rewards models that are close
  without penalizing them for minor format deviations.

**Removed benchmarks:**

- `zebra` (ZebraLogic-grid): removed 2026-04-16. Bundled reasoning capability
  with JSON-format-following, and fine-tuned models pay a "format tax"
  (parse rate dropped 99% → 71%) that wiped out their reasoning advantage.
  Same puzzles are still measured via `zebra_mc` (log-likelihood scoring),
  which isolates reasoning from format adherence. Old grid results are
  preserved in [RESULTS.md](RESULTS.md) for reference but no new grid runs.

### Standard NLU (via lm-evaluation-harness)

All three use the lm-eval-harness default scoring for each task (log-lik for
multiple-choice, exact-match for gsm8k).

| Key | Tasks | Shot style |
| --- | --- | --- |
| `standard` | 12 tasks: hellaswag, winogrande, piqa, arc_easy, arc_challenge, boolq, openbookqa, sciq, copa, commonsense_qa, truthfulqa_mc1, gsm8k | zero-shot |
| `bbh` | 11 BBH reasoning subtasks | zero-shot |
| `bbh_cot` | Same 11 BBH subtasks | 3-shot Chain-of-Thought |

## Model specification

`--models` accepts any combination of:

1. **Registry alias** (Qwen-specific shortcuts — see `MODEL_REGISTRY` in `eval_downstream.py`)
   - `base` → Qwen/Qwen3-0.6B
   - `instruct_only` → latest checkpoint from the instruct-only SFT chain
   - `6pct_L{4,8,16,32,48,64,75}` → 6%-Dolci curriculum stage checkpoints
2. **Alias=path** explicit mapping: `my_ckpt=/scratch/.../checkpoint-87000`
3. **Bare path or HF repo**: `/path/to/ckpt` or `EleutherAI/pythia-410m`

Registry aliases require `--checkpoints-root` to point at the dir containing
`job_<id>/` subdirs.

## Running

### Basic usage (direct CLI)

```bash
conda activate search

python eval_downstream.py \
    --models base instruct_only EleutherAI/pythia-410m \
    --benchmarks zebra_mc legal proofwriter bbh \
    --output results/my_run.json \
    --hf-cache /scratch/gautschi/$USER/model_cache \
    --prompts-dir ./prompts \
    --checkpoints-root /scratch/gautschi/$USER/nl_output/search \
    --data-dir /scratch/gautschi/$USER/nl_eval
```

### Via sbatch (recommended)

Both sbatch scripts are env-driven. Typical invocation:

```bash
# Gautschi
MODELS="base instruct_only 6pct_L75" \
BENCHMARKS="zebra_mc legal proofwriter bbh standard" \
sbatch eval_downstream.sh

# Gilbreth (note: use explicit name=path for non-base models since Gilbreth
# doesn't mount Gautschi's nl_output tree)
MODELS="base instruct_only=$SCRATCH/models/instruct_only 6pct_L75=$SCRATCH/models/6pct_L75" \
BENCHMARKS="zebra_mc legal" \
sbatch eval_downstream_gilbreth.sh
```

### Controlling sample size

`--n` sets the **default** examples per sub-task. `None` (default) = full test set.

`--n-per-benchmark bench=N ...` overrides on a per-benchmark basis. The sbatch
scripts default to caps for generation-heavy tasks and full sets for log-lik tasks:

```
game24=100 blocksworld=50 mystery_blocksworld=50 logistics=50
chess_mate=50 stepgame_gen=100 proofwriter_gen=200 proofwriter_cwa_gen=200
```

Override via env var:

```bash
N_OVERRIDES="bbh=250 game24=200" sbatch eval_downstream.sh
```

Or CLI:

```bash
python eval_downstream.py \
    --benchmarks zebra_mc bbh \
    --n 100 \
    --n-per-benchmark bbh=50 \
    ...
```

### Debug mode

Set `DEBUG_SAMPLES=N` (or `--debug-samples N` on the CLI) to print the first
N examples per (benchmark, sub-task) with the full prompt excerpt, gold answer,
predicted answer, and per-class log-likelihoods (or generation output for
generation benchmarks). Useful for spotting label-bias collapse, format issues,
or tokenization confounds.

```bash
DEBUG_SAMPLES=3 BENCHMARKS="legal proofwriter zebra_mc" sbatch eval_downstream.sh
```

Counters reset between models, so each model gets its own N samples per benchmark.
Set `DEBUG_SAMPLES=0` (default) to disable.

### Chat template behavior

Auto-detected from the tokenizer:
- Tokenizer has `chat_template` (Qwen, Llama chat, etc.) → chat-wrapped prompts
- No template (Pythia, base GPT-NeoX, etc.) → plain completion

Override with `--chat-template always|never` for ablations.

### Few-shot via train-split ICEs

Set `USE_TRAIN_ICES=1` (optional `TRAIN_ICE_K=5`) in the environment to load
few-shot examples from the dataset's own train split instead of the static
prompts in `prompts/`. Supported for ProofWriter and StepGame generation
variants.

## Path configuration (4 CLI args, all required or plumbed from sbatch env)

| Arg | Purpose | Required when |
| --- | --- | --- |
| `--hf-cache` | HuggingFace model/data cache | always recommended (else `$HF_HOME` is used) |
| `--prompts-dir` | Dir containing prompt template files | has a default (`prompts/` next to the script) |
| `--checkpoints-root` | Root containing `job_<id>/` dirs | required iff registry aliases (`instruct_only`, `6pct_L*`) are used |
| `--data-dir` | Contains `game24_data/`, `proofwriter_raw/` | required iff `game24` or `proofwriter*` benchmarks are requested |

## Output format

A single JSON keyed by model → benchmark → sub-task:

```json
{
  "base": {
    "_path": "Qwen/Qwen3-0.6B",
    "_chat_template": true,
    "zebra_mc": { "overall": { "accuracy": 0.297, "n": 3260 } },
    "legal":    { "hearsay": { "accuracy": 0.638, ... }, ... }
  },
  "6pct_L75": { ... }
}
```

The driver writes atomically after each benchmark so preemption mid-run doesn't
corrupt partial results.

## After a run

1. Inspect the JSON output in `results/`.
2. **Manually update [RESULTS.md](RESULTS.md)** — find the matching table,
   fill in the new cells, bump the changelog. This is the canonical
   human-readable results doc; there is no auto-compiler.

## Model compatibility

Works with any HuggingFace `AutoModelForCausalLM`. Tested on:

- Qwen3 (0.6B) — chat template auto-used
- Pythia (160M/410M/1B) — plain completion auto-used
- Any other HF repo or local checkpoint via `--models user/repo` or `--models alias=/path`

Caveats:
- `torch_dtype=torch.bfloat16` hardcoded. Works on Ampere+ (A30/A100/H100). Needs `--dtype` patch for pre-Ampere (e.g. V100).
- `device_map="cuda"` means single-GPU. No tensor-parallel sharding for very large models.
