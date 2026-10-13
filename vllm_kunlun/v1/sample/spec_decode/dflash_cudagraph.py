# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Full CUDA graph capture of the DFlash2 draft model forward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.compilation import monitor
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import graph_capture
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class _DFlashBatchDescriptor(BatchDescriptor):
    # The ordinary descriptor holds padded tokens but not the actual request
    # count or the attention launch plan captured by the graph.
    attention_signature: tuple[tuple[int, int, bool], ...] = ()


def _graph_signature(
    attn_metadata: Any, query_len: int, causal: bool
) -> tuple[int, tuple[tuple[int, int, bool], ...]] | None:
    if not attn_metadata or query_len <= 1:
        return None
    values = (
        attn_metadata.values() if isinstance(attn_metadata, dict) else (attn_metadata,)
    )
    groups = {id(md): md for md in values}.values()
    num_reqs = None
    signature = []
    for md in groups:
        table = getattr(md, "block_tables", None)
        if table is None or table.ndim != 2 or table.shape[0] <= 0:
            return None
        current_reqs = int(table.shape[0])
        if num_reqs is not None and current_reqs != num_reqs:
            return None
        if (
            not getattr(md, "dflash_static_lod", False)
            or getattr(md, "causal", None) is not causal
            or md.num_decodes != 0
            or md.num_prefills != current_reqs
            or md.max_query_len != query_len
            or md.num_prefill_tokens != current_reqs * query_len
            or md.kv_lod_xpu is None
            or md.kv_lod_cpu is None
            or md.query_start_loc is None
            or md.max_kv_len is None
        ):
            return None
        signature.append((int(md.max_kv_len), int(table.shape[1]), causal))
        num_reqs = current_reqs
    if num_reqs is None:
        return None
    return num_reqs, tuple(signature)


class DFlashDrafterCUDAGraphWrapper(CUDAGraphWrapper, BreakableCUDAGraphWrapper):
    """Capture DFlash's fixed-width prefill draft on its first real request.

    The upstream proposer dispatches its drafter as PIECEWISE even when the
    configured graph mode is FULL. Dummy runs have no attention metadata, so
    this wrapper switches to FULL and captures lazily with live metadata.
    """

    def __init__(
        self,
        runnable: torch.nn.Module,
        vllm_config: VllmConfig,
        device: torch.device,
        query_len: int,
        causal: bool,
    ) -> None:
        super().__init__(runnable, vllm_config, CUDAGraphMode.FULL)
        self._device = device
        self._query_len = query_len
        self._causal = causal
        self._attention_layers = [
            module
            for module in runnable.modules()
            if hasattr(module, "impl")
            and hasattr(module, "use_direct_call")
            and hasattr(module, "num_heads")
        ]
        self._attention_buffers: dict[
            _DFlashBatchDescriptor,
            dict[torch.nn.Module, tuple[torch.Tensor, torch.Tensor]],
        ] = {}
        self._input_addresses: dict[_DFlashBatchDescriptor, tuple[int, ...]] = {}
        self._unstable_descriptors: set[_DFlashBatchDescriptor] = set()

    def clear_graphs(self) -> None:
        self._detach_buffers()
        super().clear_graphs()
        self._attention_buffers.clear()
        self._input_addresses.clear()
        self._unstable_descriptors.clear()

    def _attach_buffers(self, desc: _DFlashBatchDescriptor) -> None:
        per_layer = self._attention_buffers.get(desc)
        if per_layer is None:
            assert desc.num_tokens is not None
            dtype = self.vllm_config.model_config.dtype
            per_layer = {}
            for layer in self._attention_layers:
                shape = (desc.num_tokens, layer.num_heads, layer.head_size_v)
                per_layer[layer] = (
                    torch.empty(shape, dtype=dtype, device=self._device),
                    torch.empty(shape, dtype=dtype, device=self._device),
                )
            self._attention_buffers[desc] = per_layer
        for layer, buffers in per_layer.items():
            layer._kunlun_dflash_graph_buffers = buffers

    def _detach_buffers(self) -> None:
        for layer in self._attention_layers:
            if hasattr(layer, "_kunlun_dflash_graph_buffers"):
                del layer._kunlun_dflash_graph_buffers

    @staticmethod
    def _addresses(args: tuple[Any, ...], kwargs: dict[str, Any], metadata: Any):
        addresses = [
            value.data_ptr()
            for value in (*args, *kwargs.values())
            if isinstance(value, torch.Tensor)
        ]
        groups = metadata.values() if isinstance(metadata, dict) else (metadata,)
        for md in {id(value): value for value in groups}.values():
            for field in (
                "query_start_loc",
                "kv_lod_xpu",
                "block_tables",
                "slot_mapping",
                "seq_lens_tensor",
            ):
                value = getattr(md, field, None)
                if isinstance(value, torch.Tensor):
                    addresses.append(value.data_ptr())
        return tuple(addresses)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not is_forward_context_available():
            return self.runnable(*args, **kwargs)
        ctx = get_forward_context()
        if (
            ctx.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
            or ctx.batch_descriptor is None
        ):
            return self.runnable(*args, **kwargs)

        graph_shape = _graph_signature(ctx.attn_metadata, self._query_len, self._causal)
        if graph_shape is None:
            return self.runnable(*args, **kwargs)
        num_reqs, attention_signature = graph_shape
        prev_desc = ctx.batch_descriptor
        desc = _DFlashBatchDescriptor(
            num_tokens=prev_desc.num_tokens,
            num_reqs=num_reqs,
            uniform=False,
            has_lora=prev_desc.has_lora,
            num_active_loras=prev_desc.num_active_loras,
            attention_signature=attention_signature,
        )
        if desc in self._unstable_descriptors:
            return self.runnable(*args, **kwargs)

        addresses = self._addresses(args, kwargs, ctx.attn_metadata)
        captured_addresses = self._input_addresses.setdefault(desc, addresses)
        if captured_addresses != addresses:
            self._unstable_descriptors.add(desc)
            logger.warning(
                "DFlash FULL graph input address changed for %s; using PIECEWISE",
                desc,
            )
            return self.runnable(*args, **kwargs)

        prev_mode = ctx.cudagraph_runtime_mode
        prev_enabled = monitor.cudagraph_capturing_enabled
        prev_skip_compiled = ctx.skip_compiled
        ctx.batch_descriptor = desc
        ctx.cudagraph_runtime_mode = CUDAGraphMode.FULL
        ctx.skip_compiled = True
        self._attach_buffers(desc)
        try:
            entry = self.concrete_cudagraph_entries.get(desc)
            if entry is not None and entry.cudagraph is not None:
                return super().__call__(*args, **kwargs)

            monitor.set_cudagraph_capturing_enabled(True)
            with graph_capture(self._device):
                super().__call__(*args, **kwargs)
            entry = self.concrete_cudagraph_entries[desc]
            assert entry.cudagraph is not None
            # Capture does not execute the graph for this live proposal.
            entry.cudagraph.replay()
            logger.info("Captured DFlash2 drafter FULL graph: %s", desc)
            return entry.output
        finally:
            self._detach_buffers()
            monitor.set_cudagraph_capturing_enabled(prev_enabled)
            ctx.skip_compiled = prev_skip_compiled
            ctx.cudagraph_runtime_mode = prev_mode
            ctx.batch_descriptor = prev_desc
