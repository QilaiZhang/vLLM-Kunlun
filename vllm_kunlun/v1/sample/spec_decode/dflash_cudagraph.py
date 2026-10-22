# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DFlash2 proposal graphs and the existing query-only graph path."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any

import torch
from vllm.compilation import monitor
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import graph_capture
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

logger = init_logger(__name__)
_MISSING_GRAPH_BINDING = object()


def _max_dflash_graph_entries(config: VllmConfig, query_len: int) -> int:
    # MRV1 can finalize capture sizes after the draft model has loaded.
    max_batch = config.scheduler_config.max_num_seqs
    max_capture = config.compilation_config.max_cudagraph_capture_size
    if max_capture is not None:
        max_batch = min(max_batch, max_capture // query_len)
    return max(1, max_batch)


def _capture_dflash_graph(
    wrapper: CUDAGraphWrapper, desc: BatchDescriptor, args: tuple[torch.Tensor, ...]
):
    prev_enabled = monitor.cudagraph_capturing_enabled
    try:
        monitor.set_cudagraph_capturing_enabled(True)
        with graph_capture(wrapper._device):
            output = CUDAGraphWrapper.__call__(wrapper, *args)
        entry = wrapper.concrete_cudagraph_entries[desc]
        assert entry.cudagraph is not None
        entry.output = output  # Strong reference for the proposal's graph-owned output.
        return entry
    finally:
        monitor.set_cudagraph_capturing_enabled(prev_enabled)


@dataclass
class _DFlashContextInputs:
    hidden_states: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor

    def stage(self, hidden_states, positions, slots) -> None:
        capacity = self.hidden_states.shape[0]
        if hidden_states.shape[0] == capacity:
            # The common full-capacity case needs no source/destination views.
            self.hidden_states.copy_(hidden_states)
            self.positions.copy_(positions)
            self.slots.copy_(slots)
            return
        count = min(capacity, hidden_states.shape[0])
        self.hidden_states[:count].copy_(hidden_states[:count])
        self.positions[:count].copy_(positions[:count])
        self.slots[:count].copy_(slots[:count])
        if count < capacity:
            self.hidden_states[count:].zero_()
            self.positions[count:].zero_()
            self.slots[count:].fill_(PAD_SLOT_ID)


@dataclass(frozen=True)
class _DFlashBatchDescriptor(BatchDescriptor):
    # The ordinary descriptor holds padded tokens but not the actual request
    # count or the attention launch plan captured by the graph.
    attention_signature: tuple[tuple[int, int, bool], ...] = ()


def _graph_signature(
    attn_metadata: Any, query_len: int
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
            or getattr(md, "causal", None) is not False
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
        signature.append((int(md.max_kv_len), int(table.shape[1]), False))
        num_reqs = current_reqs
    if num_reqs is None:
        return None
    return num_reqs, tuple(signature)


def _attention_layers(model):
    return [
        module
        for module in model.modules()
        if hasattr(module, "impl")
        and hasattr(module, "use_direct_call")
        and hasattr(module, "num_heads")
    ]


class _DFlashCUDAGraphWrapper(CUDAGraphWrapper):
    """Shared attention handoff buffers; query and proposal own their dispatch."""

    def __init__(self, model, vllm_config, device, query_len):
        super().__init__(model, vllm_config, CUDAGraphMode.FULL)
        self._device, self._query_len = device, query_len
        self._model = model
        self._attention_layers = _attention_layers(model)
        self._attention_buffers = {}

    def clear_graphs(self):
        self._detach_buffers()
        super().clear_graphs()
        self._attention_buffers.clear()
        self._attention_layers = _attention_layers(self._model)

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


class DFlashDrafterCUDAGraphWrapper(_DFlashCUDAGraphWrapper, BreakableCUDAGraphWrapper):
    """Query-only FULL graphs for non-causal DFlash2's fixed-width prefill."""

    def __init__(self, model, vllm_config, device, query_len):
        super().__init__(model, vllm_config, device, query_len)
        self._input_addresses = {}
        self._unstable_descriptors = set()

    def clear_graphs(self):
        super().clear_graphs()
        self._input_addresses.clear()
        self._unstable_descriptors.clear()

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

        graph_shape = _graph_signature(ctx.attn_metadata, self._query_len)
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


@dataclass(frozen=True)
class _DFlashProposalDescriptor(_DFlashBatchDescriptor):
    context_capacity: int = 0
    metadata_layout: tuple[Any, ...] = ()
    input_layout: tuple[Any, ...] = ()


@dataclass
class _DFlashProposalInputs:
    context: _DFlashContextInputs
    input_ids: torch.Tensor
    positions: torch.Tensor
    sample_indices: torch.Tensor
    metadata: dict[str, Any]
    host_checks: tuple[tuple[str, str, torch.Tensor], ...] = ()
    device_copies: tuple[tuple[str, str, torch.Tensor], ...] = ()
    capture_slot_mapping: dict[str, torch.Tensor] | None = None


class DFlashProposalCUDAGraphWrapper(_DFlashCUDAGraphWrapper):
    """One context/query/selection graph, with KV-inert cold capture.

    Returning None always precedes live KV writes: the proposer can then run
    its original first pass exactly once. Capture errors propagate, including
    OOM; they are never swallowed and retried on potentially modified state.
    Entries share the upstream global graph pool; capture failures propagate.
    """

    def __init__(self, model, tail, returns_tuple, vllm_config, device, query_len):
        super().__init__(model, vllm_config, device, query_len)
        self._tail = tail
        self._returns_tuple = returns_tuple
        self.runnable = self._run_proposal
        self.cudagraph_options = CUDAGraphOptions(weak_ref_output=False)
        # Actual B and attention planning buckets both contribute entries.
        default_entries = _max_dflash_graph_entries(vllm_config, query_len) * max(
            1, vllm_config.model_config.max_model_len.bit_length()
        )
        self.max_entries = int(
            os.getenv(
                "VLLM_KUNLUN_DFLASH2_PROPOSAL_CUDAGRAPH_MAX_ENTRIES", default_entries
            )
        )
        if self.max_entries < 1:
            raise ValueError("DFlash2 proposal graph entry limit must be positive")
        self.generation = 0
        self.clear_graphs()

    def planning_upper_bound(self, page_capacity, required_upper=None):
        # Match main's fixed maximum-length attention planning for FULL graphs.
        # MRV1 appends Q query positions to the target's CPU upper bound. Keep
        # that allowance, and never plan beyond the allocated page-table width.
        # _resolve_prefill_swa omits a non-causal window when the planning KV
        # length fits it. Such short-SWA inputs are full-attention-equivalent.
        # Keep geometric buckets whenever the fixed bound would cross a window.
        bound = min(
            self.vllm_config.model_config.max_model_len + self._query_len,
            page_capacity,
        )
        windows = tuple(
            getattr(layer.impl, "sliding_window", _MISSING_GRAPH_BINDING)
            for layer in self._attention_layers
        )
        reason = None
        if not windows:
            reason = "no_attention_layers"
        elif any(w is not None and not isinstance(w, int) for w in windows):
            reason = "unknown_window"
        elif any(w is not None and w < bound for w in windows):
            reason = "window_below_fixed_bound"
        elif required_upper is not None and required_upper > bound:
            reason = "host_upper_exceeds_fixed_bound"
        selected = "geometric" if reason else "fixed_full_equivalent"
        reason = reason or (
            "full_attention"
            if all(w is None for w in windows)
            else "window_covers_bound"
        )
        self.planning_counts[reason] = self.planning_counts.get(reason, 0) + 1
        self.planning_state = {
            "mode": selected,
            "reason": reason,
            "fixed_upper_bound": bound,
            "required_upper_bound": required_upper,
            "page_capacity": page_capacity,
            "sliding_windows": [
                w if w is None or isinstance(w, int) else "unknown" for w in windows
            ],
        }
        return bound if selected == "fixed_full_equivalent" else None

    def reset_result(self):
        self.last_path = "proposal_graph_not_called"
        self.last_fallback_reason = None
        self.last_num_context = self.last_capacity = self.last_overflow = 0
        self.last_capture_attempted = False

    def record_fallback(self, reason, num_reqs=0):
        self.last_path = "proposal_graph_fallback"
        self.last_fallback_reason = reason
        if (reason, num_reqs) not in self._reported_fallbacks:
            self._reported_fallbacks.add((reason, num_reqs))
            logger.info(
                "DFlash2 proposal graph fallback: reason=%s B=%d entries=%d/%d",
                reason,
                num_reqs,
                len(self._inputs),
                self.max_entries,
            )

    def clear_graphs(self):
        """Release a generation at a quiescent boundary, as in upstream managers."""
        super().clear_graphs()
        self._inputs = {}
        self._reported_fallbacks = set()
        self._stream_id = self._capture_tensors = None
        self.capture_preparations = 0
        self.capture_attempts = 0
        self.planning_state, self.planning_counts = None, {}
        self.generation += 1
        self.last_clear_reason = "clear_graphs"
        self.reset_result()

    @staticmethod
    def _metadata_layout(metadata):
        groups, mapping, layouts = {}, [], []
        for name, md in metadata.items():
            key = id(md)
            if key not in groups:
                groups[key] = len(groups)
                layouts.append(
                    tuple(
                        (field, tuple(value.shape), value.dtype, value.device.type)
                        for field, value in vars(md).items()
                        if isinstance(value, torch.Tensor)
                    )
                )
            mapping.append((name, groups[key]))
        return tuple(mapping), tuple(layouts)

    @staticmethod
    def _clone_metadata(metadata):
        # Build the persistent metadata and its copy/check plan in one pass.
        cloned, groups, host, device = {}, {}, [], []
        for name, md in metadata.items():
            if id(md) not in groups:
                staged = copy.copy(md)
                for field, value in vars(md).items():
                    if field in ("_cached_prefill_metadata", "_cached_decode_metadata"):
                        continue
                    if isinstance(value, torch.Tensor):
                        value = value.clone()
                        setattr(staged, field, value)
                        fields = host if value.device.type == "cpu" else device
                        fields.append((name, field, value))
                staged._cached_prefill_metadata = staged._cached_decode_metadata = None
                groups[id(md)] = staged
            cloned[name] = groups[id(md)]
        return cloned, tuple(host), tuple(device)

    @staticmethod
    def _host_metadata_matches(inputs, current):
        # Native XFA consumes host planning LoDs during capture. They stay
        # immutable for this entry; exact rejected-token lengths live on device.
        return all(
            torch.equal(value, getattr(current[name], field))
            for name, field, value in inputs.host_checks
        )

    def _run_proposal(
        self, context, context_positions, context_slots, ids, positions, samples
    ):
        self._model.precompute_and_store_context_kv(
            context, context_positions, context_slots
        )
        result = self._model(input_ids=ids, positions=positions, inputs_embeds=None)
        hidden = result[0] if self._returns_tuple else result
        actual = samples.numel() // (self._query_len - 1) * self._query_len
        return self._tail(hidden, samples, ids[:actual])

    def stage_inputs(
        self,
        inputs,
        context,
        context_positions,
        context_slots,
        ids,
        positions,
        samples,
        metadata,
    ):
        inputs.context.stage(context, context_positions, context_slots)
        inputs.input_ids.copy_(ids)
        inputs.positions.copy_(positions)
        inputs.sample_indices.copy_(samples)
        for name, field, value in inputs.device_copies:
            value.copy_(getattr(metadata[name], field))

    def run_overflow(self, context, positions, slots, capacity):
        self._model.precompute_and_store_context_kv(
            context[capacity:], positions[capacity:], slots[capacity:]
        )

    def prepare_capture(self):
        """Resolve static dependencies once per generation, never on replay.

        The MRV1 lifecycle hooks clear graphs before resources can change.
        Direct users must likewise clear at a quiescent mutation boundary.
        Retain captured storages until clear, including non-Parameter KV/RoPE.
        """
        if self._capture_tensors is None:
            tensors = self._model.prepare_for_graph()
            if tensors is None:
                return False
            self._capture_tensors = (*tensors, *self._model.parameters())
            self.capture_preparations += 1
        return True

    def make_descriptor(self, batch, attention_signature, capacity, metadata, tensors):
        return _DFlashProposalDescriptor(
            num_tokens=tensors[3].numel(),
            num_reqs=batch,
            uniform=False,
            attention_signature=attention_signature,
            context_capacity=capacity,
            metadata_layout=self._metadata_layout(metadata),
            input_layout=tuple(
                (t.dtype, t.device, tuple(t.shape[1:])) for t in tensors
            ),
        )

    def sync_offloader(self):
        get_offloader().sync_prev_onload()

    def replay_graph(self, entry):
        entry.cudagraph.replay()

    @staticmethod
    def copy_output(output):
        # Each caller owns its result across subsequent replay and clear.
        return output.clone()

    def _create_inputs(
        self,
        desc,
        context,
        context_positions,
        context_slots,
        ids,
        positions,
        samples,
        metadata,
    ):
        staged, host, device = self._clone_metadata(metadata)
        inputs = _DFlashProposalInputs(
            _DFlashContextInputs(
                context.new_zeros((desc.context_capacity, context.shape[1])),
                context_positions.new_zeros(desc.context_capacity),
                context_slots.new_full((desc.context_capacity,), PAD_SLOT_ID),
            ),
            ids.clone(),
            positions.clone(),
            samples.clone(),
            staged,
            host,
            device,
            capture_slot_mapping={name: md.slot_mapping for name, md in staged.items()},
        )
        for md in {id(md): md for md in staged.values()}.values():
            md.slot_mapping.fill_(PAD_SLOT_ID)
        return inputs

    def _capture_entry(self, desc, inputs):
        """Cold, KV-inert warmup/capture in the inherited global pool."""
        ctx = get_forward_context()
        original = (
            ctx.attn_metadata,
            ctx.slot_mapping,
            ctx.batch_descriptor,
            ctx.cudagraph_runtime_mode,
            ctx.skip_compiled,
        )
        args = (
            inputs.context.hidden_states,
            inputs.context.positions,
            inputs.context.slots,
            inputs.input_ids,
            inputs.positions,
            inputs.sample_indices,
        )
        self.last_capture_attempted = True
        self.capture_attempts += 1
        try:
            ctx.attn_metadata, ctx.slot_mapping = (
                inputs.metadata,
                inputs.capture_slot_mapping,
            )
            ctx.batch_descriptor, ctx.cudagraph_runtime_mode, ctx.skip_compiled = (
                desc,
                CUDAGraphMode.FULL,
                True,
            )
            self._attach_buffers(desc)
            self.runnable(*args)  # All context/query slots are PAD.
            entry = _capture_dflash_graph(self, desc, args)
            self._inputs[desc] = inputs
            logger.info(
                "Captured DFlash2 proposal graph: %s pool=global backend=%s.%s",
                desc,
                type(entry.cudagraph).__module__,
                type(entry.cudagraph).__qualname__,
            )
            return entry
        except BaseException:
            self._discard(desc)
            raise
        finally:
            self._detach_buffers()
            (
                ctx.attn_metadata,
                ctx.slot_mapping,
                ctx.batch_descriptor,
                ctx.cudagraph_runtime_mode,
                ctx.skip_compiled,
            ) = original

    def __call__(
        self,
        context,
        context_positions,
        context_slots,
        ids,
        positions,
        samples,
        capacity,
    ):
        self.reset_result()
        self.last_num_context = context.shape[0]
        if not is_forward_context_available():
            self.record_fallback("no_forward_context")
            return None
        ctx = get_forward_context()
        shape = _graph_signature(ctx.attn_metadata, self._query_len)
        reason = None
        if (
            self.vllm_config.model_config.dtype != torch.float16
            or context.dtype != torch.float16
        ):
            reason = "unsupported_dtype"
        elif self._query_len != 4:
            reason = "unsupported_static_k"
        elif (
            ctx.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
            or ctx.batch_descriptor is None
        ):
            reason = "query_runtime_not_piecewise"
        elif torch.cuda.is_current_stream_capturing():
            reason = "already_capturing"
        elif shape is None or not isinstance(ctx.attn_metadata, dict):
            reason = "metadata_ineligible"
        stream = torch.cuda.current_stream(self._device).cuda_stream
        if reason is None and self._stream_id is not None and stream != self._stream_id:
            reason = "different_stream"
        if reason:
            self.record_fallback(reason)
            return None
        batch, attention_signature = shape
        actual = batch * self._query_len
        if any(
            not isinstance(getattr(md, name, None), torch.Tensor)
            for md in {id(m): m for m in ctx.attn_metadata.values()}.values()
            for name in ("slot_mapping", "query_start_loc_host", "seq_lens_tensor")
        ):
            self.record_fallback("metadata_ineligible", batch)
            return None
        if (
            capacity <= 0
            or ids.ndim != 1
            or positions.shape != ids.shape
            or ids.numel() != ctx.batch_descriptor.num_tokens
            or actual > ids.numel()
            or samples.numel() != batch * (self._query_len - 1)
            or context.ndim != 2
            or context_positions.shape != (context.shape[0],)
            or context_slots.shape != context_positions.shape
        ):
            raise ValueError("Invalid DFlash2 proposal graph inputs")
        desc = self.make_descriptor(
            batch,
            attention_signature,
            capacity,
            ctx.attn_metadata,
            (context, context_positions, context_slots, ids, positions, samples),
        )
        inputs = self._inputs.get(desc)
        capturing = inputs is None
        staging_metadata = ctx.attn_metadata
        if capturing:
            if len(self._inputs) >= self.max_entries:
                self.record_fallback("entry_limit", batch)
                return None
            if not self.prepare_capture():
                self.record_fallback("model_state_unavailable", batch)
                return None
            # A cold capture may reuse the target graph's pool allocations.
            # Preserve live context (including overflow) before capturing.
            context, context_positions, context_slots = (
                t.clone() for t in (context, context_positions, context_slots)
            )
            inputs = self._create_inputs(
                desc,
                context,
                context_positions,
                context_slots,
                ids,
                positions,
                samples,
                ctx.attn_metadata,
            )
            slots = [
                (value, getattr(ctx.attn_metadata[name], field).clone())
                for name, field, value in (*inputs.host_checks, *inputs.device_copies)
                if field == "slot_mapping"
            ]
            entry = self._capture_entry(desc, inputs)
            for target, source in slots:
                target.copy_(source)
            # Query/metadata were cloned before capture; do not reread pooled
            # source tensors after warmup/capture. Only slot mappings used PAD.
            ids, positions, samples = (
                inputs.input_ids,
                inputs.positions,
                inputs.sample_indices,
            )
            staging_metadata = inputs.metadata
            self._stream_id = stream
        else:
            if not self._host_metadata_matches(inputs, ctx.attn_metadata):
                self.record_fallback("host_planning_changed", batch)
                return None
            entry = self.concrete_cudagraph_entries[desc]

        # Replay uses captured addresses; only capture executes Python forward
        # and needs the private metadata/attention handoff in ForwardContext.
        self.stage_inputs(
            inputs,
            context,
            context_positions,
            context_slots,
            ids,
            positions,
            samples,
            staging_metadata,
        )
        if context.shape[0] > capacity:
            self.run_overflow(context, context_positions, context_slots, capacity)
        self.sync_offloader()
        self.replay_graph(entry)
        self.last_path = (
            "proposal_graph_capture" if capturing else "proposal_graph_replay"
        )
        self.last_capacity = capacity
        self.last_overflow = max(0, context.shape[0] - capacity)
        return self.copy_output(entry.output)

    def _discard(self, desc):
        self.concrete_cudagraph_entries.pop(desc, None)
        self._inputs.pop(desc, None)
        self._attention_buffers.pop(desc, None)
