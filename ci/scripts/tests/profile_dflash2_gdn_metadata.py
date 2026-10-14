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

Stage 2 can additionally validate every later GDN group's full-build state
indices against ``update_block_table()`` with ``--verify-gdn-update``. This
keeps runtime reuse disabled while checking value and graph-address safety.
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
from collections import Counter
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

STAGGERED_PROMPTS = (
    "Explain gravity briefly:",
    (
        "Give a detailed numbered scientific explanation of stellar evolution, "
        "including molecular clouds, protostars, main-sequence fusion, red giants, "
        "supernovae, white dwarfs, neutron stars, and black holes. "
    )
    * 12,
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


@dataclass
class MetadataPathCoverage:
    """Record which GDN metadata branches a measured run actually executes."""

    build: Counter[str] = field(default_factory=Counter)
    update: Counter[str] = field(default_factory=Counter)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def _paths(metadata: Any) -> set[str]:
        paths: set[str] = set()
        num_prefills = metadata.num_prefills
        num_decodes = metadata.num_decodes
        num_spec_decodes = metadata.num_spec_decodes

        if num_prefills > 0:
            paths.add("prefill")
            if num_spec_decodes > 0:
                paths.add("mixed_prefill_spec")
        if num_decodes > 0:
            paths.add("decode")
        if num_spec_decodes > 0 and num_prefills == 0 and num_decodes == 0:
            paths.add("pure_spec")
        if metadata.spec_pad_gather_idx is not None:
            paths.add("variable_spec")

        # FULL graph metadata is padded to the selected capture size.  Prove
        # that at least one measured step contains inactive graph rows instead
        # of assuming max_num_seqs alone exercised the padding path.
        spec_masks = metadata.spec_sequence_masks
        if (
            spec_masks is not None
            and num_prefills == 0
            and num_decodes == 0
            and spec_masks.shape[0] > num_spec_decodes
        ):
            paths.add("graph_padding")
        non_spec_states = metadata.non_spec_state_indices_tensor
        if (
            non_spec_states is not None
            and num_prefills == 0
            and num_spec_decodes == 0
            and non_spec_states.shape[0] > num_decodes
        ):
            paths.add("graph_padding")
        return paths

    def observe(self, operation: str, metadata: Any) -> None:
        with self.lock:
            counter = self.build if operation == "build" else self.update
            counter.update(self._paths(metadata))

    def reset(self) -> None:
        with self.lock:
            self.build.clear()
            self.update.clear()

    def summary(self) -> dict[str, dict[str, int]]:
        with self.lock:
            build = dict(sorted(self.build.items()))
            update = dict(sorted(self.update.items()))
            total = dict(sorted((self.build + self.update).items()))
        return {"build": build, "update": update, "total": total}


class GDNUpdateEquivalenceVerifier:
    """Compare Stage-2 block-table updates with the full build they replace."""

    _PURE_SPEC_UNUSED_CPU_FIELDS = (
        "spec_state_indices_tensor_cpu",
        "spec_conv_state_indices_tensor_cpu",
    )
    _STATE_FIELDS = (
        "spec_state_indices_tensor",
        "spec_state_indices_tensor_cpu",
        "spec_conv_state_indices_tensor",
        "spec_conv_state_indices_tensor_cpu",
        "non_spec_state_indices_tensor",
        "non_spec_state_indices_tensor_cpu",
    )
    _PRIVATE_BUFFER_FIELDS = (
        "spec_state_indices_tensor",
        "spec_conv_state_indices_tensor",
        "non_spec_state_indices_tensor",
    )

    def __init__(self) -> None:
        self.sources: dict[tuple[Any, type], tuple[Any, Any, dict[str, Any]]] = {}
        self.steps = 0
        self.checks = 0

    def begin_step(self) -> None:
        self.sources.clear()
        self.steps += 1

    @staticmethod
    def _assert_same_tensor(name: str, expected: Any, actual: Any) -> None:
        import torch

        if expected is None or actual is None:
            if expected is not actual:
                raise AssertionError(
                    f"GDN update field {name} differs: expected={expected}, "
                    f"actual={actual}"
                )
            return
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise AssertionError(
                f"GDN update field {name} metadata differs: "
                f"expected shape/dtype={expected.shape}/{expected.dtype}, "
                f"actual={actual.shape}/{actual.dtype}"
            )
        if not torch.equal(expected, actual):
            raise AssertionError(f"GDN update field {name} has different values")

    def observe(self, builder: Any, metadata: Any, common_metadata: Any) -> None:
        if builder.vllm_config.cache_config.mamba_cache_mode not in ("none", "all"):
            return
        key = (builder.kv_cache_spec, type(builder))
        source = self.sources.get(key)
        if source is None:
            source_state = {
                name: (
                    value.clone()
                    if (value := getattr(metadata, name)) is not None
                    else None
                )
                for name in self._STATE_FIELDS
            }
            self.sources[key] = (builder, metadata, source_state)
            return

        source_builder, source_metadata, source_state = source
        if builder is source_builder:
            raise AssertionError(
                "GDN groups unexpectedly share one metadata builder; per-group "
                "state buffers cannot be kept independent"
            )

        for name in self._PRIVATE_BUFFER_FIELDS:
            if (
                getattr(builder, name).data_ptr()
                == getattr(source_builder, name).data_ptr()
            ):
                raise AssertionError(f"GDN group-private buffer {name} is shared")

        if metadata.spec_pad_gather_idx is None and builder._stage_spec_decode(
            metadata.num_prefills,
            metadata.num_decodes,
            metadata.num_spec_decodes,
            metadata.num_spec_decode_tokens,
        ):
            common_fields = (
                "spec_sequence_masks",
                "spec_sequence_masks_cpu",
                "spec_token_masks",
                "spec_token_indx",
                "non_spec_token_indx",
                "spec_query_start_loc",
                "num_accepted_tokens",
            )
        elif builder._stage_decode(
            metadata.num_prefills,
            metadata.num_decodes,
            metadata.num_spec_decodes,
        ):
            common_fields = ("non_spec_query_start_loc",)
        else:
            common_fields = ()
        for name in common_fields:
            source_tensor = getattr(source_metadata, name)
            actual_tensor = getattr(metadata, name)
            if source_tensor is None or actual_tensor is None:
                raise AssertionError(
                    f"GDN common graph buffer {name} is unexpectedly None"
                )
            if source_tensor.data_ptr() != actual_tensor.data_ptr():
                raise AssertionError(f"GDN common graph buffer {name} is not shared")

        # Clone before update: the destination builder's full-build metadata
        # points at its state buffers, which update_block_table writes in place.
        expected = {
            name: (
                value.clone()
                if (value := getattr(metadata, name)) is not None
                else None
            )
            for name in self._STATE_FIELDS
        }
        updated = builder.update_block_table(
            source_metadata,
            common_metadata.block_table_tensor,
            common_metadata.slot_mapping,
        )
        if updated is source_metadata:
            raise AssertionError("GDN update mutated and returned the source metadata")
        for name, expected_value in expected.items():
            if (
                metadata.num_prefills == 0
                and metadata.num_decodes == 0
                and name in self._PURE_SPEC_UNUSED_CPU_FIELDS
            ):
                if getattr(updated, name) is not None:
                    raise AssertionError(
                        f"unused pure-spec CPU field {name} was materialized"
                    )
                continue
            self._assert_same_tensor(name, expected_value, getattr(updated, name))
        for name, source_value in source_state.items():
            self._assert_same_tensor(
                f"source.{name}", source_value, getattr(source_metadata, name)
            )
        self.checks += 1

    def reset(self) -> None:
        self.sources.clear()
        self.steps = 0
        self.checks = 0

    def summary(self) -> dict[str, int]:
        return {"steps": self.steps, "group_updates_checked": self.checks}


def _timed_method(method: Callable[..., Any], timings: Timings) -> Callable[..., Any]:
    @functools.wraps(method)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        start_ns = time.perf_counter_ns()
        try:
            return method(*args, **kwargs)
        finally:
            timings.add(time.perf_counter_ns() - start_ns)

    return wrapped


def _install_timing_hooks(
    verify_gdn_update: bool,
) -> tuple[
    Timings,
    Timings,
    Timings,
    GDNUpdateEquivalenceVerifier | None,
    MetadataPathCoverage,
]:
    # Import only in the worker, after its environment has been fixed and
    # before constructing LLM/GPUModelRunner instances. Import Kunlun's GDN
    # implementation first: it replaces the upstream module symbol, which the
    # model-runner module then binds when imported.
    # isort: off
    from vllm_kunlun.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    # isort: on

    attention_metadata = Timings()
    gdn_build = Timings()
    gdn_update = Timings()
    verifier = GDNUpdateEquivalenceVerifier() if verify_gdn_update else None
    path_coverage = MetadataPathCoverage()
    runner_method = _timed_method(
        GPUModelRunner._build_attention_metadata, attention_metadata
    )
    build_method = _timed_method(GDNAttentionMetadataBuilder.build, gdn_build)
    update_method = _timed_method(
        GDNAttentionMetadataBuilder.update_block_table, gdn_update
    )

    @functools.wraps(runner_method)
    def wrapped_runner(*args: Any, **kwargs: Any) -> Any:
        if verifier is not None:
            verifier.begin_step()
        return runner_method(*args, **kwargs)

    @functools.wraps(build_method)
    def wrapped_build(*args: Any, **kwargs: Any) -> Any:
        metadata = build_method(*args, **kwargs)
        path_coverage.observe("build", metadata)
        if verifier is not None:
            builder = args[0]
            common_metadata = kwargs.get("common_attn_metadata")
            if common_metadata is None:
                common_metadata = args[2]
            verifier.observe(builder, metadata, common_metadata)
        return metadata

    @functools.wraps(update_method)
    def wrapped_update(*args: Any, **kwargs: Any) -> Any:
        metadata = update_method(*args, **kwargs)
        path_coverage.observe("update", metadata)
        return metadata

    setattr(
        GPUModelRunner,
        "_build_attention_metadata",
        wrapped_runner,
    )
    setattr(
        GDNAttentionMetadataBuilder,
        "build",
        wrapped_build,
    )
    setattr(
        GDNAttentionMetadataBuilder,
        "update_block_table",
        wrapped_update,
    )
    return attention_metadata, gdn_build, gdn_update, verifier, path_coverage


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
    # Equivalence mode needs every group to execute a full build so it can
    # compare that reference with the update path. Normal Stage-3 runs leave
    # reuse enabled unless the caller explicitly sets this emergency opt-out.
    if args.verify_gdn_update:
        os.environ["VLLM_KUNLUN_DISABLE_GDN_METADATA_REUSE"] = "1"
    elif args.expect_gdn_reuse and os.getenv(
        "VLLM_KUNLUN_DISABLE_GDN_METADATA_REUSE", "0"
    ).lower() in ("1", "true", "on"):
        raise RuntimeError(
            "--expect-gdn-reuse conflicts with "
            "VLLM_KUNLUN_DISABLE_GDN_METADATA_REUSE"
        )
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

    (
        attention_metadata,
        gdn_build,
        gdn_update,
        update_verifier,
        path_coverage,
    ) = _install_timing_hooks(args.verify_gdn_update)

    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("Run the worker on a Kunlun CUDA-compatible device")

    prompt_pool = STAGGERED_PROMPTS if args.prompt_set == "staggered" else PROMPTS
    prompts = list(prompt_pool[: args.batch_size])
    if len(prompts) != args.batch_size:
        raise ValueError(
            f"--batch-size must be between 1 and {len(prompt_pool)} for "
            f"--prompt-set={args.prompt_set}"
        )

    llm_kwargs: dict[str, Any] = {
        "model": args.target_model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": max(args.batch_size, args.max_num_seqs),
        "enforce_eager": args.target_eager,
        "dtype": args.dtype,
        "seed": args.seed,
        "async_scheduling": args.async_scheduling == "on",
    }
    if not args.disable_speculative:
        llm_kwargs["speculative_config"] = {
            "model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.enable_chunked_prefill:
        llm_kwargs["enable_chunked_prefill"] = True
    llm = LLM(**llm_kwargs)
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
    gdn_update.reset()
    path_coverage.reset()
    if update_verifier is not None:
        update_verifier.reset()
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

    if update_verifier is not None and update_verifier.checks == 0:
        raise AssertionError(
            "--verify-gdn-update did not observe a later equivalent GDN group"
        )

    elapsed_s = sum(iteration_ms) / 1000
    metadata_summary = attention_metadata.summary()
    gdn_summary = gdn_build.summary()
    gdn_update_summary = gdn_update.summary()
    path_summary = path_coverage.summary()
    missing_paths = [
        path
        for path in args.require_gdn_path
        if path_summary["total"].get(path, 0) == 0
    ]
    if missing_paths:
        raise AssertionError(
            "Required GDN metadata paths were not observed: "
            f"{missing_paths}; observed={path_summary['total']}"
        )
    if args.expect_gdn_reuse:
        if gdn_summary["calls"] != metadata_summary["calls"]:
            raise AssertionError(
                "GDN metadata reuse expected exactly one full build per "
                "_build_attention_metadata call, got "
                f"{gdn_summary['calls']} builds for "
                f"{metadata_summary['calls']} metadata calls"
            )
        if gdn_update_summary["calls"] == 0:
            raise AssertionError(
                "GDN metadata reuse expected block-table updates, but none ran"
            )
    result = {
        "schema_version": 1,
        "git_revision": _git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "config": {
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "speculative_enabled": not args.disable_speculative,
            "async_scheduling": args.async_scheduling == "on",
            "dflash2_full_graph": args.full_graph == "on",
            "target_eager": args.target_eager,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_num_seqs": max(args.batch_size, args.max_num_seqs),
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "prompt_set": args.prompt_set,
            "num_speculative_tokens": args.num_speculative_tokens,
            "max_tokens": args.max_tokens,
            "warmup_iterations": args.warmup_iterations,
            "iterations": args.iterations,
            "gdn_metadata_reuse_disabled": os.getenv(
                "VLLM_KUNLUN_DISABLE_GDN_METADATA_REUSE", "0"
            ).lower()
            in ("1", "true", "on"),
        },
        "prompts": prompts,
        "metrics": {
            "build_attention_metadata": metadata_summary,
            "gdn_full_build": gdn_summary,
            "gdn_block_table_update": gdn_update_summary,
            "gdn_update_equivalence": (
                update_verifier.summary() if update_verifier is not None else None
            ),
            "gdn_metadata_paths": path_summary,
            "gdn_full_builds_per_metadata_call": (
                gdn_summary["calls"] / metadata_summary["calls"]
                if metadata_summary["calls"]
                else None
            ),
            "gdn_updates_per_metadata_call": (
                gdn_update_summary["calls"] / metadata_summary["calls"]
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


def _worker_command(
    args: argparse.Namespace,
    result_path: Path,
    overrides: dict[str, Any] | None = None,
) -> list[str]:
    values = vars(args) | (overrides or {})
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--target-model",
        args.target_model,
        "--draft-model",
        args.draft_model,
        "--async-scheduling",
        values["async_scheduling"],
        "--full-graph",
        values["full_graph"],
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
        command.extend((f"--{name.replace('_', '-')}", str(values[name])))
    if values.get("max_num_batched_tokens") is not None:
        command.extend(
            ("--max-num-batched-tokens", str(values["max_num_batched_tokens"]))
        )
    command.extend(("--dtype", values["dtype"]))
    command.extend(("--prompt-set", values.get("prompt_set", "default")))
    for path in values.get("require_gdn_path", ()):
        command.extend(("--require-gdn-path", path))
    for enabled, flag in (
        (values.get("xprofiler"), "--xprofiler"),
        (values.get("verify_gdn_update"), "--verify-gdn-update"),
        (values.get("expect_gdn_reuse"), "--expect-gdn-reuse"),
        (values.get("target_eager"), "--target-eager"),
        (values.get("disable_speculative"), "--disable-speculative"),
        (values.get("enable_chunked_prefill"), "--enable-chunked-prefill"),
    ):
        if enabled:
            command.append(flag)
    return command


def _tokens_are_deterministic(result: dict[str, Any]) -> bool:
    iterations = result["token_ids_by_iteration"]
    return bool(iterations) and all(
        tokens == iterations[0] for tokens in iterations[1:]
    )


def _assert_external_reference(
    case_name: str, result: dict[str, Any], reference_dir: Path
) -> None:
    reference_path = reference_dir / f"{case_name}.json"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing external reference: {reference_path}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if result["prompts"] != reference.get("prompts"):
        raise AssertionError(f"{case_name}: prompts differ from external reference")
    comparable_config = (
        "speculative_enabled",
        "async_scheduling",
        "dflash2_full_graph",
        "target_eager",
        "tensor_parallel_size",
        "dtype",
        "seed",
        "batch_size",
        "max_num_seqs",
        "max_model_len",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "prompt_set",
        "num_speculative_tokens",
        "max_tokens",
        "warmup_iterations",
        "iterations",
    )
    for name in comparable_config:
        if result["config"].get(name) != reference.get("config", {}).get(name):
            raise AssertionError(
                f"{case_name}: config.{name} differs from external reference: "
                f"candidate={result['config'].get(name)!r}, "
                f"reference={reference.get('config', {}).get(name)!r}"
            )
    actual_tokens = result["token_ids_by_iteration"]
    expected_tokens = reference.get("token_ids_by_iteration")
    if actual_tokens != expected_tokens:
        for iteration, (actual_batch, expected_batch) in enumerate(
            zip(actual_tokens, expected_tokens or [])
        ):
            if actual_batch == expected_batch:
                continue
            for request, (actual, expected) in enumerate(
                zip(actual_batch, expected_batch)
            ):
                if actual != expected:
                    raise AssertionError(
                        f"{case_name}: token mismatch at iteration={iteration}, "
                        f"request={request}: candidate={actual}, reference={expected}"
                    )
        raise AssertionError(
            f"{case_name}: token output shape differs from external reference"
        )


def _run_matrix(args: argparse.Namespace) -> None:
    async_modes = (
        ("off", "on") if args.async_scheduling == "both" else (args.async_scheduling,)
    )
    graph_modes = ("off", "on") if args.full_graph == "both" else (args.full_graph,)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_paths: list[Path] = []

    for async_mode in async_modes:
        for graph_mode in graph_modes:
            case_name = _case_name(async_mode, graph_mode)
            result_path = output_dir / f"{case_name}.json"
            command = _worker_command(
                args,
                result_path,
                {"async_scheduling": async_mode, "full_graph": graph_mode},
            )

            print(f"\n=== {case_name} ===", flush=True)
            subprocess.run(command, check=True)
            result_paths.append(result_path)

    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    reference_tokens = results[0]["token_ids_by_iteration"]
    token_match = {
        path.stem: result["token_ids_by_iteration"] == reference_tokens
        for path, result in zip(result_paths, results)
    }
    deterministic = {
        path.stem: _tokens_are_deterministic(result)
        for path, result in zip(result_paths, results)
    }
    external_match: dict[str, bool] | None = None
    if args.reference_dir:
        external_match = {}
        reference_dir = Path(args.reference_dir)
        for path, result in zip(result_paths, results):
            _assert_external_reference(path.stem, result, reference_dir)
            external_match[path.stem] = True
    summary = {
        "schema_version": 1,
        "reference_case": result_paths[0].stem,
        "deterministic_tokens_match_reference": token_match,
        "deterministic_within_case": deterministic,
        "external_reference_tokens_match": external_match,
        "all_tokens_match": all(token_match.values()) and all(deterministic.values()),
        "all_tokens_match_external_reference": (
            all(external_match.values()) if external_match is not None else None
        ),
        "cases": {
            path.stem: result["metrics"] for path, result in zip(result_paths, results)
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


def _precision_scenarios(
    args: argparse.Namespace,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    # Historical reference revisions can represent graph padding differently
    # from the candidate and only need to provide known-good token output.
    # Enforce current metadata branch coverage on the reuse-enabled candidate.
    def required_paths(*paths: str) -> tuple[str, ...]:
        return paths if args.expect_gdn_reuse else ()

    common = {
        "warmup_iterations": args.warmup_iterations,
        "iterations": args.iterations,
        "num_speculative_tokens": args.num_speculative_tokens,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": None,
        "enable_chunked_prefill": False,
        "prompt_set": "default",
        "disable_speculative": False,
        "target_eager": False,
        "expect_gdn_reuse": args.expect_gdn_reuse,
        "verify_gdn_update": False,
        "xprofiler": False,
    }
    return (
        (
            "regular-b1",
            common
            | {
                "disable_speculative": True,
                "batch_size": 1,
                "max_num_seqs": 4,
                "async_scheduling": "off",
                "full_graph": "on",
                "max_tokens": 17,
                "require_gdn_path": required_paths("prefill", "decode"),
            },
        ),
        (
            "regular-b3-graph-padding",
            common
            | {
                "disable_speculative": True,
                "batch_size": 3,
                "max_num_seqs": 4,
                "async_scheduling": "off",
                "full_graph": "on",
                "max_tokens": 17,
                "require_gdn_path": required_paths(
                    "prefill", "decode", "graph_padding"
                ),
            },
        ),
        (
            "spec-b1",
            common
            | {
                "batch_size": 1,
                "max_num_seqs": 4,
                "async_scheduling": "off",
                "full_graph": "on",
                "max_tokens": 17,
                "require_gdn_path": required_paths("prefill", "pure_spec"),
            },
        ),
        (
            "spec-b3-graph-padding",
            common
            | {
                "batch_size": 3,
                "max_num_seqs": 4,
                "async_scheduling": "off",
                "full_graph": "on",
                "max_tokens": 17,
                "require_gdn_path": required_paths(
                    "prefill", "pure_spec", "graph_padding"
                ),
            },
        ),
        (
            "spec-b4-async",
            common
            | {
                "batch_size": 4,
                "max_num_seqs": 8,
                "async_scheduling": "on",
                "full_graph": "on",
                "max_tokens": 17,
                "require_gdn_path": required_paths("prefill", "pure_spec"),
            },
        ),
        (
            "spec-b2-target-eager",
            common
            | {
                "batch_size": 2,
                "max_num_seqs": 4,
                "async_scheduling": "off",
                "full_graph": "off",
                "target_eager": True,
                "max_tokens": 17,
                "require_gdn_path": required_paths("prefill", "pure_spec"),
            },
        ),
        (
            "spec-b2-mixed-prefill",
            common
            | {
                "batch_size": 2,
                "max_num_seqs": 4,
                "max_model_len": 512,
                "max_num_batched_tokens": 64,
                "enable_chunked_prefill": True,
                "prompt_set": "staggered",
                "async_scheduling": "off",
                "full_graph": "on",
                "max_tokens": 24,
                "require_gdn_path": required_paths("prefill", "mixed_prefill_spec"),
            },
        ),
    )


def _run_precision_suite(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = Path(args.reference_dir) if args.reference_dir else None
    summary_cases: dict[str, Any] = {}

    for case_name, overrides in _precision_scenarios(args):
        result_path = output_dir / f"{case_name}.json"
        print(f"\n=== precision: {case_name} ===", flush=True)
        subprocess.run(_worker_command(args, result_path, overrides), check=True)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        deterministic = _tokens_are_deterministic(result)
        if not deterministic:
            raise AssertionError(
                f"{case_name}: token output differs between iterations"
            )
        external_match: bool | None = None
        if reference_dir is not None:
            _assert_external_reference(case_name, result, reference_dir)
            external_match = True
        summary_cases[case_name] = {
            "deterministic_within_case": deterministic,
            "external_reference_tokens_match": external_match,
            "config": result["config"],
            "gdn_metadata_paths": result["metrics"]["gdn_metadata_paths"],
            "gdn_full_builds_per_metadata_call": result["metrics"][
                "gdn_full_builds_per_metadata_call"
            ],
            "gdn_updates_per_metadata_call": result["metrics"][
                "gdn_updates_per_metadata_call"
            ],
        }

    summary = {
        "schema_version": 1,
        "external_reference_dir": str(reference_dir) if reference_dir else None,
        "mamba_cache_mode_varied": False,
        "all_cases_deterministic": all(
            case["deterministic_within_case"] for case in summary_cases.values()
        ),
        "all_cases_match_external_reference": (
            all(
                case["external_reference_tokens_match"]
                for case in summary_cases.values()
            )
            if reference_dir is not None
            else None
        ),
        "cases": summary_cases,
    }
    summary_path = output_dir / "precision-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nPrecision summary written to {summary_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--output-dir", default="dflash2-gdn-baseline")
    parser.add_argument(
        "--reference-dir",
        help="Directory containing known-good JSON files for exact token comparison",
    )
    parser.add_argument("--output", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--precision-suite",
        action="store_true",
        help="Run the expanded cross-scenario correctness suite",
    )
    parser.add_argument(
        "--async-scheduling", choices=("off", "on", "both"), default="both"
    )
    parser.add_argument("--full-graph", choices=("off", "on", "both"), default="both")
    parser.add_argument(
        "--target-eager",
        action="store_true",
        help="Disable target-model CUDAGraphs for correctness bisection",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--num-speculative-tokens", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument(
        "--prompt-set", choices=("default", "staggered"), default="default"
    )
    parser.add_argument(
        "--disable-speculative",
        action="store_true",
        help="Run target-only prefill/decode without a speculative model",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        help="Enable chunked prefill (used to create mixed prefill/spec batches)",
    )
    parser.add_argument(
        "--require-gdn-path",
        action="append",
        default=[],
        choices=(
            "prefill",
            "decode",
            "pure_spec",
            "mixed_prefill_spec",
            "variable_spec",
            "graph_padding",
        ),
        help="Fail unless the measured region executes this GDN metadata path",
    )
    parser.add_argument(
        "--xprofiler",
        action="store_true",
        help="Call xtorch_ops profiler start/end around measured iterations",
    )
    parser.add_argument(
        "--verify-gdn-update",
        action="store_true",
        help=(
            "Compare Stage-2 update_block_table results with each later "
            "group's full build; reuse remains disabled"
        ),
    )
    parser.add_argument(
        "--expect-gdn-reuse",
        action="store_true",
        help=(
            "Require exactly one GDN full build per metadata call and at "
            "least one block-table update"
        ),
    )
    args = parser.parse_args()
    if args.verify_gdn_update and args.expect_gdn_reuse:
        parser.error("--verify-gdn-update and --expect-gdn-reuse are exclusive")
    if args.precision_suite and args.verify_gdn_update:
        parser.error("--precision-suite does not support --verify-gdn-update")
    if args.precision_suite and args.xprofiler:
        parser.error("--precision-suite does not support --xprofiler")
    if args.worker:
        if args.async_scheduling == "both" or args.full_graph == "both":
            parser.error("worker mode requires one async and one graph mode")
        if not args.output:
            parser.error("worker mode requires --output")
    prompt_count = (
        len(STAGGERED_PROMPTS) if args.prompt_set == "staggered" else len(PROMPTS)
    )
    if args.batch_size < 1 or args.batch_size > prompt_count:
        parser.error(
            f"--batch-size must be between 1 and {prompt_count} for "
            f"--prompt-set={args.prompt_set}"
        )
    return args


def main() -> None:
    args = _parse_args()
    if args.worker:
        _run_worker(args)
    elif args.precision_suite:
        _run_precision_suite(args)
    else:
        _run_matrix(args)


if __name__ == "__main__":
    main()
