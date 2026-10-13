#!/usr/bin/env python3
"""Collect a reproducible DFlash2 GDN metadata baseline.

The default invocation runs four independent processes covering async
scheduling on/off and DFlash2 drafter FULL graph on/off.  Each worker warms up
the engine, resets the counters, then records deterministic generation output,
end-to-end latency, ``GPUModelRunner._build_attention_metadata`` host time and
the number/time of full ``GDNAttentionMetadataBuilder.build`` calls.

Example:

    python ci/scripts/tests/profile_dflash2_gdn_metadata.py \
        --target-model /models/Qwen3.5-27B \
        --draft-model /models/Qwen3.5-0.8B-DFlash2 \
        --output-dir /tmp/dflash2-gdn-baseline

Run this script from the repository root in the same Python environment used
for vLLM-Kunlun inference.  The Python counters require the default TP=1 and
force the V1 engine core in-process.  Use ``--tensor-parallel-size`` only for
an XProfiler-only deployment-shape run; worker-local Python counters may then
be incomplete.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


PROMPTS = (
    "A short explanation of gravity:",
    "The first three prime numbers are",
    "In one sentence, explain photosynthesis:",
    "Write a Python function that adds two integers:",
    "The capital of France is",
    "A healthy morning routine can begin with",
    "Translate 'good morning' to Chinese:",
    "The next number in 1, 1, 2, 3, 5 is",
)


@dataclass
class Timings:
    values_ns: list[int] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, elapsed_ns: int) -> None:
        with self.lock:
            self.values_ns.append(elapsed_ns)

    def reset(self) -> None:
        with self.lock:
            self.values_ns.clear()

    def summary(self) -> dict[str, float | int]:
        with self.lock:
            values_ms = [value / 1_000_000 for value in self.values_ns]
        if not values_ms:
            return {
                "calls": 0,
                "total_ms": 0.0,
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "max_ms": 0.0,
            }
        ordered = sorted(values_ms)
        p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
        return {
            "calls": len(values_ms),
            "total_ms": sum(values_ms),
            "mean_ms": statistics.fmean(values_ms),
            "p50_ms": statistics.median(values_ms),
            "p95_ms": ordered[p95_index],
            "max_ms": max(values_ms),
        }


def _timed_method(method: Callable[..., Any], timings: Timings) -> Callable[..., Any]:
    @functools.wraps(method)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        start_ns = time.perf_counter_ns()
        try:
            return method(*args, **kwargs)
        finally:
            timings.add(time.perf_counter_ns() - start_ns)

    return wrapped


def _install_timing_hooks() -> tuple[Timings, Timings]:
    # Import only in the worker, after its environment has been fixed and
    # before constructing LLM/GPUModelRunner instances. Import Kunlun's GDN
    # implementation first: it replaces the upstream module symbol, which the
    # model-runner module then binds when imported.
    from vllm_kunlun.v1.attention.backends.gdn_attn import (
        GDNAttentionMetadataBuilder,
    )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    attention_metadata = Timings()
    gdn_build = Timings()
    setattr(
        GPUModelRunner,
        "_build_attention_metadata",
        _timed_method(GPUModelRunner._build_attention_metadata, attention_metadata),
    )
    setattr(
        GDNAttentionMetadataBuilder,
        "build",
        _timed_method(GDNAttentionMetadataBuilder.build, gdn_build),
    )
    return attention_metadata, gdn_build


def _maybe_start_xprofiler(enabled: bool) -> Any | None:
    if not enabled:
        return None
    try:
        import xtorch_ops
    except ImportError as exc:
        raise RuntimeError(
            "--xprofiler requested but xtorch_ops is unavailable"
        ) from exc
    xtorch_ops.kunlun_profiler_start()
    return xtorch_ops


def _stop_xprofiler(module: Any | None) -> None:
    if module is not None:
        module.kunlun_profiler_end()


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _run_worker(args: argparse.Namespace) -> None:
    # In-process EngineCore makes the monkey-patched runner visible.  TP=1 is
    # the reference configuration for the host counters.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_KUNLUN_DRAFTER_FULL_CUDAGRAPH"] = (
        "1" if args.full_graph == "on" else "0"
    )
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

    attention_metadata, gdn_build = _install_timing_hooks()

    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("Run the worker on a Kunlun CUDA-compatible device")

    prompts = list(PROMPTS[: args.batch_size])
    if len(prompts) != args.batch_size:
        raise ValueError(f"--batch-size must be between 1 and {len(PROMPTS)}")

    llm = LLM(
        model=args.target_model,
        speculative_config={
            "model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
        },
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=max(args.batch_size, args.max_num_seqs),
        enforce_eager=False,
        dtype=args.dtype,
        seed=args.seed,
        async_scheduling=args.async_scheduling == "on",
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    for _ in range(args.warmup_iterations):
        llm.generate(prompts, sampling_params, use_tqdm=False)

    # Exclude model loading, graph capture and kernel warm-up from steady-state
    # counters.  XProfiler is also narrowed to this measured region.
    attention_metadata.reset()
    gdn_build.reset()
    profiler = _maybe_start_xprofiler(args.xprofiler)
    iteration_ms: list[float] = []
    token_ids_by_iteration: list[list[list[int]]] = []
    total_output_tokens = 0
    try:
        for _ in range(args.iterations):
            start_ns = time.perf_counter_ns()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            iteration_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000)
            token_ids = [list(output.outputs[0].token_ids) for output in outputs]
            token_ids_by_iteration.append(token_ids)
            total_output_tokens += sum(len(ids) for ids in token_ids)
    finally:
        _stop_xprofiler(profiler)

    elapsed_s = sum(iteration_ms) / 1000
    metadata_summary = attention_metadata.summary()
    gdn_summary = gdn_build.summary()
    result = {
        "schema_version": 1,
        "git_revision": _git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "config": {
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "async_scheduling": args.async_scheduling == "on",
            "dflash2_full_graph": args.full_graph == "on",
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_num_seqs": max(args.batch_size, args.max_num_seqs),
            "max_model_len": args.max_model_len,
            "num_speculative_tokens": args.num_speculative_tokens,
            "max_tokens": args.max_tokens,
            "warmup_iterations": args.warmup_iterations,
            "iterations": args.iterations,
        },
        "prompts": prompts,
        "metrics": {
            "build_attention_metadata": metadata_summary,
            "gdn_full_build": gdn_summary,
            "gdn_full_builds_per_metadata_call": (
                gdn_summary["calls"] / metadata_summary["calls"]
                if metadata_summary["calls"]
                else None
            ),
            "iteration_ms": iteration_ms,
            "mean_iteration_ms": statistics.fmean(iteration_ms),
            "p50_iteration_ms": statistics.median(iteration_ms),
            "throughput_output_tokens_per_s": (
                total_output_tokens / elapsed_s if elapsed_s else 0.0
            ),
            "mean_wall_ms_per_output_token": (
                sum(iteration_ms) / total_output_tokens if total_output_tokens else None
            ),
            "total_output_tokens": total_output_tokens,
        },
        "token_ids_by_iteration": token_ids_by_iteration,
        "notes": {
            "python_counter_scope": (
                "complete for the reference TP=1 in-process configuration"
                if args.tensor_parallel_size == 1
                else "may omit worker-local calls when tensor_parallel_size > 1"
            ),
            "device_metrics": (
                "Inspect the matching XProfiler trace for D2H/H2D and device gaps."
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"Baseline written to {output}")


def _case_name(async_scheduling: str, full_graph: str) -> str:
    return f"async-{async_scheduling}_full-graph-{full_graph}"


def _run_matrix(args: argparse.Namespace) -> None:
    async_modes = (
        ("off", "on")
        if args.async_scheduling == "both"
        else (args.async_scheduling,)
    )
    graph_modes = ("off", "on") if args.full_graph == "both" else (args.full_graph,)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_paths: list[Path] = []

    for async_mode in async_modes:
        for graph_mode in graph_modes:
            case_name = _case_name(async_mode, graph_mode)
            result_path = output_dir / f"{case_name}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--target-model",
                args.target_model,
                "--draft-model",
                args.draft_model,
                "--async-scheduling",
                async_mode,
                "--full-graph",
                graph_mode,
                "--output",
                str(result_path),
            ]
            for name in (
                "tensor_parallel_size",
                "batch_size",
                "max_num_seqs",
                "max_model_len",
                "num_speculative_tokens",
                "max_tokens",
                "warmup_iterations",
                "iterations",
                "seed",
            ):
                command.extend(
                    (f"--{name.replace('_', '-')}", str(getattr(args, name)))
                )
            command.extend(("--dtype", args.dtype))
            if args.xprofiler:
                command.append("--xprofiler")

            print(f"\n=== {case_name} ===", flush=True)
            subprocess.run(command, check=True)
            result_paths.append(result_path)

    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    reference_tokens = results[0]["token_ids_by_iteration"]
    token_match = {
        path.stem: result["token_ids_by_iteration"] == reference_tokens
        for path, result in zip(result_paths, results)
    }
    summary = {
        "schema_version": 1,
        "reference_case": result_paths[0].stem,
        "deterministic_tokens_match_reference": token_match,
        "all_tokens_match": all(token_match.values()),
        "cases": {
            path.stem: result["metrics"]
            for path, result in zip(result_paths, results)
        },
    }
    summary_path = output_dir / "matrix-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nMatrix summary written to {summary_path}")
    if not summary["all_tokens_match"]:
        raise AssertionError(
            "Deterministic token output differs between baseline configurations"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--output-dir", default="dflash2-gdn-baseline")
    parser.add_argument("--output", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--async-scheduling", choices=("off", "on", "both"), default="both"
    )
    parser.add_argument("--full-graph", choices=("off", "on", "both"), default="both")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--num-speculative-tokens", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument(
        "--xprofiler",
        action="store_true",
        help="Call xtorch_ops profiler start/end around measured iterations",
    )
    args = parser.parse_args()
    if args.worker:
        if args.async_scheduling == "both" or args.full_graph == "both":
            parser.error("worker mode requires one async and one graph mode")
        if not args.output:
            parser.error("worker mode requires --output")
    if args.batch_size < 1 or args.batch_size > len(PROMPTS):
        parser.error(f"--batch-size must be between 1 and {len(PROMPTS)}")
    return args


def main() -> None:
    args = _parse_args()
    if args.worker:
        _run_worker(args)
    else:
        _run_matrix(args)


if __name__ == "__main__":
    main()
