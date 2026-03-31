from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.benchmark_config import benchmark_results_dir, get_benchmark
from benchmarks.common.io_utils import ensure_dir, pick_primary_metric, read_json, write_json, write_text


def _prepend_env_paths(env: dict[str, str]) -> dict[str, str]:
    env = dict(env)
    path_entries = [str(REPO_ROOT), "/usr/local/nvidia/bin"]
    current_path = env.get("PATH", "")
    if current_path:
        path_entries.append(current_path)
    env["PATH"] = os.pathsep.join(path_entries)

    lib_entries = [str(Path(sys.prefix) / "lib"), "/usr/local/nvidia/lib64"]
    current_ld = env.get("LD_LIBRARY_PATH", "")
    if current_ld:
        lib_entries.append(current_ld)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_entries)
    return env


def build_runner_command(
    benchmark_slug: str,
    model_alias: str,
    output_dir: Path,
    gpu_id: str,
    limit: int | None,
) -> list[str]:
    common_dir = Path(__file__).resolve().parent
    if get_benchmark(benchmark_slug).kind == "lm_eval":
        runner = common_dir / "lm_eval_runner.py"
        command = [
            sys.executable,
            str(runner),
            "--benchmark",
            benchmark_slug,
            "--model-alias",
            model_alias,
            "--output-dir",
            str(output_dir),
            "--gpu-id",
            gpu_id,
        ]
    else:
        runner = common_dir / "ifbench_runner.py"
        command = [
            sys.executable,
            str(runner),
            "--model-alias",
            model_alias,
            "--output-dir",
            str(output_dir),
            "--gpu-id",
            gpu_id,
        ]

    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def aggregate_results(benchmark_slug: str, results_dir: Path) -> dict:
    benchmark = get_benchmark(benchmark_slug)
    base_metrics = read_json(results_dir / "base" / "metrics.json")
    finetuned_metrics = read_json(results_dir / "finetuned" / "metrics.json")

    shared_keys = sorted(
        set(base_metrics["metrics"]).intersection(finetuned_metrics["metrics"])
    )
    deltas = {
        key: finetuned_metrics["metrics"][key] - base_metrics["metrics"][key] for key in shared_keys
    }
    primary_metric_name, _ = pick_primary_metric(deltas, benchmark.metric_preferences)

    comparison = {
        "benchmark": benchmark.slug,
        "display_name": benchmark.display_name,
        "base_model": base_metrics,
        "finetuned_model": finetuned_metrics,
        "metric_deltas": deltas,
        "primary_metric_name": primary_metric_name,
        "primary_metric_delta": deltas.get(primary_metric_name) if primary_metric_name else None,
    }
    write_json(results_dir / "comparison.json", comparison)

    summary_lines = [
        f"# {benchmark.display_name} comparison",
        "",
        f"Primary delta: `{primary_metric_name}` = `{comparison['primary_metric_delta']}`",
        "",
        "| Metric | Base | Finetuned | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in shared_keys:
        summary_lines.append(
            "| "
            f"{key} | {base_metrics['metrics'][key]:.6f} | {finetuned_metrics['metrics'][key]:.6f} "
            f"| {deltas[key]:+.6f} |"
        )
    write_text(results_dir / "summary.md", "\n".join(summary_lines) + "\n")
    return comparison


def run_comparison(benchmark_slug: str, gpu_ids: list[str], limit: int | None) -> dict:
    if not gpu_ids:
        raise ValueError("At least one GPU id must be provided.")

    results_dir = ensure_dir(benchmark_results_dir(benchmark_slug))
    base_dir = ensure_dir(results_dir / "base")
    finetuned_dir = ensure_dir(results_dir / "finetuned")

    jobs = [
        ("base", base_dir, gpu_ids[0]),
        ("finetuned", finetuned_dir, gpu_ids[1]),
    ]
    env = _prepend_env_paths(os.environ.copy())
    pythonpath = env.get("PYTHONPATH", "")
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = repo_root if not pythonpath else os.pathsep.join([repo_root, pythonpath])

    failures = []
    if len(gpu_ids) == 1:
        for model_alias, output_dir, _ in jobs:
            command = build_runner_command(
                benchmark_slug=benchmark_slug,
                model_alias=model_alias,
                output_dir=output_dir,
                gpu_id=gpu_ids[0],
                limit=limit,
            )
            exit_code = subprocess.run(command, env=env, check=False).returncode
            if exit_code != 0:
                failures.append((model_alias, exit_code))
    else:
        processes = []
        for model_alias, output_dir, gpu_id in jobs:
            command = build_runner_command(
                benchmark_slug=benchmark_slug,
                model_alias=model_alias,
                output_dir=output_dir,
                gpu_id=gpu_id,
                limit=limit,
            )
            process = subprocess.Popen(command, env=env)
            processes.append((model_alias, process))

        for model_alias, process in processes:
            exit_code = process.wait()
            if exit_code != 0:
                failures.append((model_alias, exit_code))

    if failures:
        details = ", ".join(f"{alias} exited {code}" for alias, code in failures)
        raise RuntimeError(f"{benchmark_slug} comparison failed: {details}")

    return aggregate_results(benchmark_slug, results_dir)


def main(default_benchmark: str | None = None) -> None:
    benchmark_choices = [default_benchmark] if default_benchmark else sorted(
        ["ifeval", "ifbench", "bbh", "agieval_en"]
    )

    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned Qwen on one benchmark.")
    parser.add_argument("--benchmark", choices=benchmark_choices, default=default_benchmark)
    parser.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated GPU ids. Use one id for sequential same-GPU runs or two ids for parallel.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap for smoke tests.")
    args = parser.parse_args()

    benchmark_slug = args.benchmark or default_benchmark
    if benchmark_slug is None:
        raise ValueError("A benchmark slug is required.")

    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    run_comparison(benchmark_slug=benchmark_slug, gpu_ids=gpu_ids, limit=args.limit)


if __name__ == "__main__":
    main()
