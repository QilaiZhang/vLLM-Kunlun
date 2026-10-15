"""Host-only slot mapping instrumentation for the DFlash2 baseline runner.

The hooks never read tensor contents, synchronize devices or alter outputs.
Host timings are nested: batch includes group, and group includes native entry.
Use XProfiler to distinguish device execution from host waits inside an entry.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable


def _git_identity(source: Path) -> dict[str, Any] | None:
    for directory in (source, *source.parents):
        if not (directory / ".git").exists():
            continue
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        status = subprocess.run(
            ["git", "-C", str(directory), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return {
            "root": str(directory),
            "revision": result.stdout.strip() if result.returncode == 0 else None,
            "error": result.stderr.strip() if result.returncode != 0 else None,
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        }
    return None


def environment_snapshot(torch: Any) -> dict[str, Any]:
    """Record loaded modules and devices outside the measured interval."""
    packages = {}
    for name, distribution in (
        ("vllm", "vllm"),
        ("vllm_kunlun", "vllm-kunlun"),
        ("kunlun_ops", "kunlun-ops"),
        ("torch", "torch"),
    ):
        module = importlib.import_module(name)
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        source = getattr(module, "__file__", None)
        packages[name] = {
            "distribution_version": version,
            "module_version": str(getattr(module, "__version__", "unknown")),
            "source": source,
            "git": _git_identity(Path(source).resolve().parent) if source else None,
        }
    # Native binaries may come from a different installation than Python files.
    native_modules = {
        name: module.__file__
        for name, module in list(sys.modules.items())
        if ("xpu_flash" in name or "xflash" in name)
        and getattr(module, "__file__", "")
        and Path(module.__file__).suffix in (".so", ".pyd")
    }
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
            }
        )
    return {
        "hostname": platform.node(),
        "python_executable": sys.executable,
        "packages": packages,
        "native_modules": native_modules,
        "profiling_script_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "profile_dflash2_gdn_metadata.py",
                "profile_slot_mappings.py",
            )
        },
        "devices": devices,
        "execution_environment": {
            name: os.environ.get(name)
            for name in (
                "XPU_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "XPU_DUMMY_EVENT",
                "VLLM_USE_V2_MODEL_RUNNER",
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "VLLM_KUNLUN_DRAFTER_FULL_CUDAGRAPH",
                "VLLM_KUNLUN_DISABLE_GDN_METADATA_REUSE",
                "XMLIR_FORCE_USE_XPU_GRAPH",
                "XMLIR_ENABLE_MOCK_TORCH_COMPILE",
            )
        },
    }


class SlotMappingProfiler:
    """Count batch/group/native entries only while the measured window is open."""

    def __init__(self, timings_factory: Callable[[], Any]):
        self.batch = timings_factory()
        self.group = timings_factory()
        self.native = timings_factory()
        self.unattributed_native = timings_factory()
        self._timings_factory = timings_factory
        self._groups: dict[int, dict[str, Any]] = {}
        self._runners: dict[int, int] = {}
        self.runtime: list[dict[str, Any]] = []
        self.batch_shapes: Counter[tuple[int, int]] = Counter()
        self._context = threading.local()
        self.active = False

    def register_runner(self, runner: Any, mamba_spec_cls: type) -> None:
        if self.active:
            raise RuntimeError("KV layout changed during the measured window")
        runner_index = self._runners.setdefault(id(runner), len(self._runners))
        groups = runner.kv_cache_config.kv_cache_groups
        tables = runner.input_batch.block_table.block_tables
        if len(groups) != len(tables):
            raise RuntimeError("KV cache groups and V1 block tables differ in length")
        self._groups = {
            key: group
            for key, group in self._groups.items()
            if group["descriptor"]["runner_index"] != runner_index
        }
        descriptors = []
        for group_id, (group, table) in enumerate(zip(groups, tables)):
            spec = group.kv_cache_spec
            specs = list(getattr(spec, "kv_cache_specs", {}).values()) or [spec]
            kind = (
                "gdn_mamba"
                if all(isinstance(item, mamba_spec_cls) for item in specs)
                else "attention_or_other"
            )
            mode = getattr(table, "slot_mapping_mode", None)
            descriptor = {
                "runner_index": runner_index,
                "group_id": group_id,
                "kind": kind,
                "spec_types": sorted({type(item).__name__ for item in specs}),
                "layer_count": len(group.layer_names),
                "manager_block_size": getattr(spec, "block_size", None),
                "kernel_block_size": table.block_size,
                "blocks_per_kv_block": getattr(table, "blocks_per_kv_block", None),
                "block_table_shape": list(table.block_table.gpu.shape),
                "block_table_stride": table.block_table.gpu.stride(0),
                "slot_mapping_capacity": table.max_num_batched_tokens,
                "slot_mapping_mode": getattr(mode, "value", None),
                "pcp_world_size": table.pcp_world_size,
                "dcp_world_size": table.dcp_world_size,
                "cp_interleave": table.cp_kv_cache_interleave_size,
            }
            self._groups[id(table)] = {
                "table": table,
                "descriptor": descriptor,
                "method": self._timings_factory(),
                "native": self._timings_factory(),
            }
            descriptors.append(descriptor)
        description = {
            "runner_index": runner_index,
            "runner_class": f"{type(runner).__module__}.{type(runner).__name__}",
            "mamba_cache_mode": runner.vllm_config.cache_config.mamba_cache_mode,
            "groups": descriptors,
        }
        model_config = getattr(runner.vllm_config, "model_config", None)
        text_config = getattr(model_config, "hf_text_config", None)
        if text_config is not None:
            description["model_type"] = getattr(text_config, "model_type", None)
            description["layer_type_counts"] = dict(
                Counter(getattr(text_config, "layer_types", ()) or ())
            )
        if runner_index == len(self.runtime):
            self.runtime.append(description)
        else:
            self.runtime[runner_index] = description

    def start(self) -> None:
        self.batch.reset()
        self.group.reset()
        self.native.reset()
        self.unattributed_native.reset()
        self.batch_shapes.clear()
        for group in self._groups.values():
            group["method"].reset()
            group["native"].reset()
        self.active = True

    def stop(self) -> None:
        self.active = False

    def install(
        self,
        runner_cls: type,
        block_cls: type,
        multi_cls: type,
        ops: Any,
        mamba_spec_cls: type,
    ) -> None:
        initialize = runner_cls.initialize_kv_cache
        batch_method = multi_cls.compute_slot_mapping
        group_method = block_cls.compute_slot_mapping
        native_method = ops.compute_slot_mappings

        @functools.wraps(initialize)
        def wrapped_initialize(runner, *args, **kwargs):
            result = initialize(runner, *args, **kwargs)
            self.register_runner(runner, mamba_spec_cls)
            return result

        @functools.wraps(batch_method)
        def wrapped_batch(table, num_reqs, query_start_loc, positions):
            if not self.active:
                return batch_method(table, num_reqs, query_start_loc, positions)
            start = time.perf_counter_ns()
            try:
                return batch_method(table, num_reqs, query_start_loc, positions)
            finally:
                self.batch.add(time.perf_counter_ns() - start)
                self.batch_shapes[(num_reqs, positions.shape[0])] += 1

        @functools.wraps(group_method)
        def wrapped_group(table, *args, **kwargs):
            if not self.active:
                return group_method(table, *args, **kwargs)
            previous = getattr(self._context, "group", None)
            group = self._groups.get(id(table))
            self._context.group = group
            start = time.perf_counter_ns()
            try:
                return group_method(table, *args, **kwargs)
            finally:
                elapsed = time.perf_counter_ns() - start
                self.group.add(elapsed)
                if group is not None:
                    group["method"].add(elapsed)
                self._context.group = previous

        @functools.wraps(native_method)
        def wrapped_native(*args, **kwargs):
            if not self.active:
                return native_method(*args, **kwargs)
            start = time.perf_counter_ns()
            try:
                return native_method(*args, **kwargs)
            finally:
                elapsed = time.perf_counter_ns() - start
                self.native.add(elapsed)
                group = getattr(self._context, "group", None)
                if group is None:
                    self.unattributed_native.add(elapsed)
                else:
                    group["native"].add(elapsed)

        runner_cls.initialize_kv_cache = wrapped_initialize
        multi_cls.compute_slot_mapping = wrapped_batch
        block_cls.compute_slot_mapping = wrapped_group
        ops.compute_slot_mappings = wrapped_native

    def summary(self) -> dict[str, Any]:
        return {
            "batch": self.batch.summary(),
            "group_method": self.group.summary(),
            "native_op_entry": self.native.summary(),
            "unattributed_native_op_entry": self.unattributed_native.summary(),
            "per_group": [
                group["descriptor"]
                | {
                    "method": group["method"].summary(),
                    "native_op_entry": group["native"].summary(),
                }
                for group in self._groups.values()
            ],
            "batch_shapes": [
                {"num_reqs": reqs, "num_tokens": tokens, "calls": calls}
                for (reqs, tokens), calls in sorted(self.batch_shapes.items())
            ],
        }
