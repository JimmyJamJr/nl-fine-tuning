# IFBench

This benchmark uses `vllm` for generation and the official `allenai/IFBench` scorer.

The first run will clone the scorer repo into `benchmarks/ifbench/vendor/IFBench` if it is not already present.

## Environment

```bash
conda env create -f benchmarks/ifbench/environment.yml
conda activate qwen-ifbench
python -m spacy download en_core_web_sm
```

## Smoke Test

```bash
python benchmarks/ifbench/run_compare.py --gpus 0,1 --limit 5
```

## Full Run

```bash
python benchmarks/ifbench/run_compare.py --gpus 0,1
```
