# Prompt templates used by eval_downstream.py

Each file contains a text prompt template with few-shot examples. Loaded at
import time by eval_downstream.py via `_load_prompt(path)`.

## Sources

### `legal/*.txt` — Authors' official templates (canonical)
Auto-downloaded from the [LegalBench repository](https://github.com/HazyResearch/legalbench)
(one per task's `base_prompt.txt`) by `_ensure_legal_template()` in eval_downstream.py,
cached here on first use. Used by the `legal` benchmark. The `{input}` placeholder
is substituted with each test example at runtime.

Currently uses subtasks: `diversity_1..6`, `sara_entailment`, `sara_numeric`.
Each author template contains 6 in-context examples.

### `proofwriter_owa_fewshot.txt`, `proofwriter_cwa_fewshot.txt` — Fallback only
3 hand-crafted few-shot examples each, covering True/False/Unknown (OWA) or
True/False (CWA). **Used only as fallback** when the primary train-split
sampling path fails (`_USE_TRAIN_ICES=0` env var or raw ProofWriter dataset
not found).

Primary path: `_proofwriter_few_shot_block()` samples 8 balanced demos from
`meta-test.jsonl` of depth-3 (we eval on depth-5, so no overlap). The original
ProofWriter paper (Tafjord et al. 2021) fine-tunes T5 rather than few-shot
prompting LLMs, so these hand-crafted prompts are our own approximation.

### `stepgame_fewshot.txt` — Fallback only
5 hand-crafted few-shot examples covering the 8-label spatial direction space.
**Used only as fallback**; primary path is train-split sampling via
`_stepgame_few_shot_block()`. The original StepGame paper (Shi et al. 2022)
fine-tunes a classifier; no canonical LLM few-shot protocol exists.

## Alternative: dynamic train-split ICEs (default)

The primary path for ProofWriter and StepGame demos is to sample from the
dataset's train split (or depth-3 test for ProofWriter since its train split
isn't shipped with the data we have locally). Set `USE_TRAIN_ICES=0` to
force the static fallback .txt files above.

Default is `USE_TRAIN_ICES=1` (on) with `TRAIN_ICE_K=8` demos per benchmark.
See `_proofwriter_few_shot_block()` and `_stepgame_few_shot_block()` in
eval_downstream.py for the per-task logic.

## Other benchmarks (no local prompt files)

Most benchmarks build their few-shot demos on the fly at runtime from the test
or train data, not from static .txt files:

- **FOLIO**: 8-shot from train split (`_folio_few_shot_block`)
- **LogicBench BQA/MCQA**: 3-shot Direct, sampled balanced from the data itself
- **PrOntoQA-OOD**: 8-shot Direct (CoT stripped from canonical author demos)
- **NLGraph**: 4-shot from `Arthur-Heng/NLGraph/<task>/prompt/k-shot-prompt.txt`
  (cloned repo in `$DATA_DIR/nlgraph/`, not here)
- **LogiQA, RuleTaker, CLUTRR, Multi-LogiEval, ZebraLogic**: 0-shot primary;
  optional `_fs` variants sample demos at runtime

See `README.md` in the parent directory for the full benchmark protocol table.
