# AGI Eval English

This benchmark uses `lm-evaluation-harness` with the `vllm` backend and the `agieval_en` task group.

## Environment

```bash
conda env create -f benchmarks/agieval_en/environment.yml
conda activate qwen-agieval-en
```

## Smoke Test

```bash
python benchmarks/agieval_en/run_compare.py --gpus 0,1 --limit 5
```

## Full Run

```bash
python benchmarks/agieval_en/run_compare.py --gpus 0,1
```
