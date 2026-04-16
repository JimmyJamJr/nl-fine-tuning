# Prompt templates used by eval_downstream.py

Each file contains a text prompt template with few-shot examples. Loaded at
import time by eval_downstream.py via `_load_prompt(path)`.

## Sources

### `legal/*.txt` — Authors' official templates
Copied verbatim from the [LegalBench repository](https://github.com/HazyResearch/legalbench)
(one per task's `base_prompt.txt`). Used by both `legal` (log-lik) and `legal_gen`
(generation) benchmarks. The `{input}` placeholder is substituted with each
test example at runtime.

### `proofwriter_owa_fewshot.txt`, `proofwriter_cwa_fewshot.txt` — Hand-crafted
3 few-shot examples each, covering True/False/Unknown (OWA) or True/False (CWA).
Used by `proofwriter_gen` and `proofwriter_cwa_gen`. The original ProofWriter
paper (Tafjord 2021) fine-tunes T5 rather than few-shot prompting, so these
are our own approximation.

### `stepgame_fewshot.txt` — Hand-crafted
5 few-shot examples covering the 9-label spatial direction space. Used by
`stepgame_gen`. The original StepGame paper (Shi 2022) fine-tunes a classifier;
follow-up LLM papers (e.g., Li et al. 2024 SpatialLM-StepGame) use few-shot
prompting of various formats — ours approximates that.

## Alternative: dynamic train-split ICEs

For benchmarks where the underlying dataset has a train split, you can load
few-shot examples from the dataset itself rather than using these static
hand-crafted prompts. Set `USE_TRAIN_ICES=1` environment variable to enable.
This uses the first K examples from the train split as in-context examples,
giving more in-distribution prompts. See eval_downstream.py's
`_load_train_ices_*` helpers for per-task implementations.
