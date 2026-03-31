from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.benchmark_config import benchmark_results_dir
from benchmarks.common.io_utils import ensure_dir, read_json, write_json, write_text


BENCHMARK_ORDER = ["ifeval", "ifbench", "bbh", "agieval_en"]


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


def parse_gpu_pairs(gpu_pairs_arg: str | None, gpus_arg: str | None) -> list[list[str]]:
    if gpu_pairs_arg:
        parsed = []
        for pair in gpu_pairs_arg.split(";"):
            gpu_ids = [item.strip() for item in pair.split(",") if item.strip()]
            if gpu_ids:
                parsed.append(gpu_ids)
        return parsed

    if not gpus_arg:
        raise ValueError("Provide either --gpu-pairs or --gpus.")

    gpu_ids = [item.strip() for item in gpus_arg.split(",") if item.strip()]
    if len(gpu_ids) == 1:
        return [[gpu_ids[0]]]

    pairs = []
    idx = 0
    while idx < len(gpu_ids):
        chunk = gpu_ids[idx : idx + 2]
        pairs.append(chunk)
        idx += 2
    return pairs


def build_compare_command(benchmark_slug: str, gpu_ids: list[str], limit: int | None) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "compare_benchmark.py"),
        "--benchmark",
        benchmark_slug,
        "--gpus",
        ",".join(gpu_ids),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def write_cross_benchmark_report(output_path: Path) -> None:
    summaries = []
    markdown = [
        "# Qwen 0.6B benchmark comparison",
        "",
        "| Benchmark | Primary metric | Delta | Base | Finetuned |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for slug in BENCHMARK_ORDER:
        comparison_path = benchmark_results_dir(slug) / "comparison.json"
        if not comparison_path.exists():
            continue
        comparison = read_json(comparison_path)
        metric = comparison.get("primary_metric_name")
        delta = comparison.get("primary_metric_delta")
        base = comparison["base_model"]["metrics"].get(metric)
        finetuned = comparison["finetuned_model"]["metrics"].get(metric)
        summaries.append(comparison)
        markdown.append(
            f"| {comparison['display_name']} | {metric} | {delta:+.6f} | {base:.6f} | {finetuned:.6f} |"
        )

    payload = {"benchmarks": summaries}
    write_json(output_path.with_suffix(".json"), payload)
    write_text(output_path, "\n".join(markdown) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all four benchmark comparisons.")
    parser.add_argument("--gpu-pairs", default=None, help="Semicolon-separated GPU pairs, e.g. 0,1;2,3")
    parser.add_argument("--gpus", default=None, help="Flat comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap for smoke tests.")
    args = parser.parse_args()

    gpu_pairs = parse_gpu_pairs(args.gpu_pairs, args.gpus)
    if not gpu_pairs:
        raise ValueError("No usable GPU pairs were parsed.")

    env = _prepend_env_paths(os.environ.copy())
    pythonpath = env.get("PYTHONPATH", "")
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = repo_root if not pythonpath else os.pathsep.join([repo_root, pythonpath])

    pair_queue = deque(gpu_pairs)
    pending = deque(BENCHMARK_ORDER)
    running: list[tuple[str, list[str], subprocess.Popen[bytes]]] = []

    while pending or running:
        while pending and pair_queue:
            benchmark_slug = pending.popleft()
            gpu_ids = pair_queue.popleft()
            process = subprocess.Popen(
                build_compare_command(benchmark_slug, gpu_ids, args.limit),
                env=env,
            )
            running.append((benchmark_slug, gpu_ids, process))

        if not running:
            break

        benchmark_slug, gpu_ids, process = running.pop(0)
        exit_code = process.wait()
        pair_queue.append(gpu_ids)
        if exit_code != 0:
            raise RuntimeError(f"Benchmark {benchmark_slug} failed with exit code {exit_code}")

    summary_path = ensure_dir(Path(__file__).resolve().parents[1] / "results") / "summary.md"
    write_cross_benchmark_report(summary_path)


if __name__ == "__main__":
    main()
