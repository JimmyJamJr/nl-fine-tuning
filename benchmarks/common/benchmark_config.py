from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"


@dataclass(frozen=True)
class ModelConfig:
    alias: str
    model_id: str
    apply_chat_template: bool = True
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 64


@dataclass(frozen=True)
class BenchmarkConfig:
    slug: str
    display_name: str
    kind: str
    task_name: str
    max_new_tokens: int
    default_limit: int | None
    metric_preferences: tuple[str, ...]
    notes: str


MODELS = {
    "base": ModelConfig(alias="base", model_id="Qwen/Qwen3-0.6B"),
    "finetuned": ModelConfig(
        alias="finetuned",
        model_id="JimmyJamJr/qwen06b-search-curr-L128",
    ),
}


BENCHMARKS = {
    "ifeval": BenchmarkConfig(
        slug="ifeval",
        display_name="IFEval",
        kind="lm_eval",
        task_name="ifeval",
        max_new_tokens=1536,
        default_limit=None,
        metric_preferences=(
            "prompt_level_loose_acc,none",
            "prompt_level_loose_acc",
            "inst_level_loose_acc,none",
            "inst_level_loose_acc",
        ),
        notes="Instruction-following benchmark with verifiable constraints.",
    ),
    "bbh": BenchmarkConfig(
        slug="bbh",
        display_name="BigBenchHard",
        kind="lm_eval",
        task_name="bbh",
        max_new_tokens=1024,
        default_limit=None,
        metric_preferences=(
            "acc_norm,none",
            "acc,none",
            "exact_match,none",
            "exact_match",
        ),
        notes="Reasoning-heavy benchmark suite; defaults to lm-eval's BBH group.",
    ),
    "agieval_en": BenchmarkConfig(
        slug="agieval_en",
        display_name="AGI Eval English",
        kind="lm_eval",
        task_name="agieval_en",
        max_new_tokens=1024,
        default_limit=None,
        metric_preferences=(
            "acc_norm,none",
            "acc,none",
            "acc",
            "exact_match,none",
        ),
        notes="English AGIEval group from lm-evaluation-harness.",
    ),
    "ifbench": BenchmarkConfig(
        slug="ifbench",
        display_name="IFBench",
        kind="ifbench",
        task_name="allenai/IFBench_test",
        max_new_tokens=1536,
        default_limit=None,
        metric_preferences=(
            "prompt_level_loose_accuracy",
            "instruction_level_loose_accuracy",
            "prompt_level_strict_accuracy",
            "instruction_level_strict_accuracy",
        ),
        notes="Official IFBench generate-then-score pipeline.",
    ),
}


def get_model(alias: str) -> ModelConfig:
    try:
        return MODELS[alias]
    except KeyError as exc:
        valid = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown model alias '{alias}'. Expected one of: {valid}") from exc


def get_benchmark(slug: str) -> BenchmarkConfig:
    try:
        return BENCHMARKS[slug]
    except KeyError as exc:
        valid = ", ".join(sorted(BENCHMARKS))
        raise ValueError(f"Unknown benchmark '{slug}'. Expected one of: {valid}") from exc


def benchmark_results_dir(slug: str) -> Path:
    return BENCHMARKS_ROOT / slug / "results"
