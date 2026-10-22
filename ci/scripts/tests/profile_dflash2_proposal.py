"""Opt-in DFlash2 MRV1 host instrumentation for the existing baseline runner.

Performance hooks belong around Python call boundaries, do not read tensor
values, and never wrap compiled ``forward`` bodies. Optional CUDA-compatible
events measure stream intervals and are resolved after the window closes.
The explicit correctness modes read tensor values synchronously and must be
excluded from performance conclusions.
"""

from __future__ import annotations

import functools
import heapq
import inspect
import math
import os
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable


def device_memory_snapshot(cuda: Any, reset_peak: bool = False) -> dict[str, Any]:
    """Report allocator counters explicitly as unavailable on unsupported stacks."""
    try:
        if reset_peak:
            cuda.reset_peak_memory_stats()
        return {
            "available": True,
            "allocated_bytes": cuda.memory_allocated(),
            "reserved_bytes": cuda.memory_reserved(),
            "peak_allocated_bytes": cuda.max_memory_allocated(),
            "peak_reserved_bytes": cuda.max_memory_reserved(),
        }
    except (AttributeError, NotImplementedError, RuntimeError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def graph_replay_implementations(entries) -> list[dict[str, Any]]:
    """Collect the loaded replay method outside the measured window."""
    implementations = {}
    for entry in entries:
        graph = entry.cudagraph
        if graph is None or type(graph) in implementations:
            continue
        cls = type(graph)
        info = {
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "source_file": None,
            "source": None,
            "source_truncated": False,
        }
        try:
            info["source_file"] = inspect.getsourcefile(graph.replay)
            source = inspect.getsource(graph.replay)
            info["source"] = source[:8192]
            info["source_truncated"] = len(source) > 8192
        except (OSError, TypeError):
            pass  # Native-only implementations need runtime source/trace.
        implementations[cls] = info
    return list(implementations.values())


@dataclass
class _Span:
    name: str
    start_ns: int
    elapsed_ns: int = 0
    children_ns: int = 0
    events: Any = None


@dataclass
class _Proposal:
    spans: list[_Span] = field(default_factory=list)
    stack: list[_Span] = field(default_factory=list)
    graph_path: str = "without_full_graph_wrapper"
    fallback_reason: str | None = None
    proposal_graph_path: str = "proposal_disabled"
    proposal_fallback_reason: str | None = None
    query_end_ns: int | None = None
    capture: bool = False


class ProposalProfiler:
    """Collect nested host spans, excluding cold capture proposals as a unit."""

    def __init__(self, timings_factory: Callable, cuda: Any = None):
        self._timings_factory = timings_factory
        self.cuda = cuda
        self.active = False
        self._local = threading.local()
        self.runtime: list[dict[str, Any]] = []
        self._proposal_wrappers = []
        self._patches: list[tuple[Any, str, bool, Any]] = []
        self._lock = threading.Lock()
        self._in_flight = 0
        self._reset()

    def _reset(self) -> None:
        self.stages: dict[str, dict[str, Any]] = {}
        self.outside_propose: dict[str, dict[str, Any]] = {}
        self.graph_paths: Counter[str] = Counter()
        self.fallback_reasons: Counter[str] = Counter()
        self.proposal_graph_paths: Counter[str] = Counter()
        self.proposal_fallback_reasons: Counter[str] = Counter()
        self.proposal_rows: Counter[str] = Counter()
        self.duplicate_segment_dispatches = 0
        self.capture_proposals = self._timings_factory()
        self.errors = 0
        self.num_drafts = self.num_draft_tokens = self.num_accepted_tokens = 0
        self.accepted_per_position: Counter[int] = Counter()
        self.drafted_per_position: Counter[int] = Counter()
        self._event_pairs: list[tuple[str, Any]] = []
        self.in_flight_at_stop = 0
        self._proposal_ordinal = 0
        self._slow_proposals = []

    def start(self) -> None:
        with self._lock:
            if self._in_flight:
                raise RuntimeError("Cannot reset profiling during an active proposal")
            self._reset()
            self.active = True

    def stop(self) -> None:
        with self._lock:
            self.active = False
            self.in_flight_at_stop = self._in_flight

    def _events(self):
        if self.cuda is None or self.cuda.is_current_stream_capturing():
            return None
        start = self.cuda.Event(enable_timing=True)
        end = self.cuda.Event(enable_timing=True)
        start.record()
        return start, end

    def _add(self, destination, span: _Span) -> None:
        timing = destination.get(span.name)
        if timing is None:
            timing = {
                "host_total": self._timings_factory(),
                "host_self": self._timings_factory(),
            }
            destination[span.name] = timing
        timing["host_total"].add(span.elapsed_ns)
        timing["host_self"].add(max(0, span.elapsed_ns - span.children_ns))

    @contextmanager
    def stage(self, name: str):
        if not self.active:
            yield
            return
        proposal = getattr(self._local, "proposal", None)
        if proposal is not None and any(s.name == name for s in proposal.stack):
            # A FULL wrapper may invoke the raw model while capturing/falling back.
            yield
            return
        span = _Span(
            name,
            time.perf_counter_ns(),
            events=self._events() if proposal is not None else None,
        )
        if proposal is not None:
            proposal.stack.append(span)
        try:
            yield
        finally:
            if span.events is not None:
                span.events[1].record()
            end_ns = time.perf_counter_ns()
            span.elapsed_ns = end_ns - span.start_ns
            if proposal is not None:
                proposal.stack.pop()
                if proposal.stack:
                    proposal.stack[-1].children_ns += span.elapsed_ns
                proposal.spans.append(span)
                if name == "query_forward":
                    proposal.query_end_ns = end_ns
            else:
                with self._lock:
                    if self.active:
                        self._add(self.outside_propose, span)

    def _gather_gap(self, proposal: _Proposal, stage_name: str) -> None:
        if proposal.query_end_ns is None:
            return
        span = _Span(stage_name, proposal.query_end_ns)
        span.elapsed_ns = time.perf_counter_ns() - span.start_ns
        proposal.spans.append(span)
        if proposal.stack:
            proposal.stack[-1].children_ns += span.elapsed_ns
        proposal.query_end_ns = None

    def wrap_stage(self, name: str, method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapped(*args, **kwargs):
            if not self.active:
                return method(*args, **kwargs)
            proposal = getattr(self._local, "proposal", None)
            if proposal is not None:
                if name == "draft_token_selection":
                    self._gather_gap(proposal, "sample_gather_and_bookkeeping")
            with self.stage(name):
                return method(*args, **kwargs)

        return wrapped

    def wrap_propose(self, method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapped(drafter, *args, **kwargs):
            if not self.active or not drafter.is_dflash2:
                return method(drafter, *args, **kwargs)
            if getattr(self._local, "proposal", None) is not None:
                raise RuntimeError("Nested DFlash2 proposals cannot be profiled")
            with self._lock:
                profiling = self.active
                if profiling:
                    self._in_flight += 1
            if not profiling:
                return method(drafter, *args, **kwargs)
            proposal = _Proposal()
            self._local.proposal = proposal
            try:
                with self.stage("proposal"):
                    result = method(drafter, *args, **kwargs)
            except BaseException:
                with self._lock:
                    if self.active:
                        self.errors += 1
                raise
            else:
                combined = getattr(drafter, "_dflash_proposal_graph", None)
                combined_path = (
                    combined.last_path if combined is not None else "proposal_disabled"
                )
                proposal.proposal_graph_path = combined_path
                proposal.proposal_fallback_reason = (
                    combined.last_fallback_reason if combined is not None else None
                )
                combined_hit = combined_path in (
                    "proposal_graph_capture",
                    "proposal_graph_replay",
                )
                if combined_hit:
                    if proposal.graph_path != "without_full_graph_wrapper":
                        self.duplicate_segment_dispatches += 1
                    proposal.graph_path = combined_path
                    proposal.fallback_reason = None
                if combined_path == "proposal_graph_capture" or (
                    combined is not None and combined.last_capture_attempted
                ):
                    proposal.capture = True
                with self._lock:
                    if not self.active:
                        return result
                    self._proposal_ordinal += 1
                    self.graph_paths[proposal.graph_path] += 1
                    if proposal.fallback_reason:
                        self.fallback_reasons[proposal.fallback_reason] += 1
                    self.proposal_graph_paths[combined_path] += 1
                    if combined is not None:
                        if combined.last_fallback_reason:
                            self.proposal_fallback_reasons[
                                combined.last_fallback_reason
                            ] += 1
                        if combined_hit:
                            count, capacity = (
                                combined.last_num_context,
                                combined.last_capacity,
                            )
                            self.proposal_rows["input"] += count
                            self.proposal_rows["graph_input"] += min(count, capacity)
                            self.proposal_rows["padding"] += max(0, capacity - count)
                            self.proposal_rows["overflow"] += combined.last_overflow
                    if proposal.capture:
                        self.capture_proposals.add(proposal.spans[-1].elapsed_ns)
                    else:
                        for span in proposal.spans:
                            self._add(self.stages, span)
                            if span.events is not None:
                                self._event_pairs.append((span.name, span.events))
                        item = (
                            proposal.spans[-1].elapsed_ns,
                            self._proposal_ordinal,
                            proposal,
                        )
                        if len(self._slow_proposals) < 3:
                            heapq.heappush(self._slow_proposals, item)
                        elif item[0] > self._slow_proposals[0][0]:
                            heapq.heapreplace(self._slow_proposals, item)
                return result
            finally:
                self._local.proposal = None
                with self._lock:
                    self._in_flight -= 1

        return wrapped

    def _slow_proposal_summary(self):
        result = []
        for elapsed, ordinal, proposal in sorted(self._slow_proposals, reverse=True):
            start = proposal.spans[-1].start_ns
            result.append(
                {
                    "ordinal": ordinal,
                    "graph_path": proposal.graph_path,
                    "fallback_reason": proposal.fallback_reason,
                    "proposal_graph_path": proposal.proposal_graph_path,
                    "proposal_fallback_reason": proposal.proposal_fallback_reason,
                    "host_total_ms": elapsed / 1e6,
                    "spans": [
                        {
                            "name": span.name,
                            "start_offset_ms": (span.start_ns - start) / 1e6,
                            "host_total_ms": span.elapsed_ns / 1e6,
                            "host_self_ms": max(0, span.elapsed_ns - span.children_ns)
                            / 1e6,
                        }
                        for span in proposal.spans
                    ],
                }
            )
        return result

    def observe_acceptance(self, drafted: int, accepted: int) -> None:
        if self.active:
            with self._lock:
                if not self.active:
                    return
                self.num_drafts += 1
                self.num_draft_tokens += drafted
                self.num_accepted_tokens += accepted
                self.drafted_per_position.update(range(drafted))
                self.accepted_per_position.update(range(accepted))

    def _patch(self, owner, name, replacement) -> None:
        # Remember inherited descriptors too, so uninstall does not leave overrides.
        own = name in vars(owner)
        self._patches.append((owner, name, own, vars(owner).get(name)))
        setattr(owner, name, replacement)

    def uninstall(self) -> None:
        self.stop()
        for owner, name, own, original in reversed(self._patches):
            if own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self._patches.clear()

    def install(
        self,
        proposer_cls,
        model_cls,
        selector_cls,
        graph_module,
        base_graph_cls,
        ops,
        stats_cls,
        get_context,
        graph_mode,
    ) -> None:
        self._patch(proposer_cls, "propose", self.wrap_propose(proposer_cls.propose))
        for name, stage in (
            ("prepare_next_token_ids_padded", "prepare_next_token_ids"),
            ("prepare_inputs_padded", "prepare_inputs"),
            ("set_inputs_first_pass", "set_inputs_first_pass"),
            ("build_per_group_and_layer_attn_metadata", "draft_attention_metadata"),
            ("_determine_batch_execution_and_padding", "batch_dispatch"),
            ("_get_slot_mapping", "query_slot_mapping"),
            ("compute_draft_token_ids", "draft_token_selection"),
        ):
            self._patch(
                proposer_cls, name, self.wrap_stage(stage, getattr(proposer_cls, name))
            )
        for name, stage in (
            ("combine_hidden_states", "aux_hidden_combine"),
            ("precompute_and_store_context_kv", "context_precompute"),
            ("forward", "query_forward"),
            ("compute_candidates", "candidate_topk"),
        ):
            self._patch(
                model_cls, name, self.wrap_stage(stage, getattr(model_cls, name))
            )
        # Decorated selector.__call__ dispatches to its compiled forward. Wrapping
        # forward/_score_edges would insert Python timers into Dynamo's graph.
        self._patch(
            selector_cls,
            "__call__",
            self.wrap_stage("candidate_selector", selector_cls.__call__),
        )
        for name, stage in (
            ("copy_and_expand_dflash_inputs", "input_expansion"),
            ("dflash_greedy_select_path", "greedy_walk"),
        ):
            self._patch(ops, name, self.wrap_stage(stage, getattr(ops, name)))

        original_load = proposer_cls.load_model

        @functools.wraps(original_load)
        def load(drafter, *args, **kwargs):
            result = original_load(drafter, *args, **kwargs)
            if drafter.is_dflash2:
                combined = drafter._dflash_proposal_graph
                if combined is not None:
                    self._proposal_wrappers.append(combined)
                self.runtime.append(
                    {
                        "proposer_class": f"{type(drafter).__module__}.{type(drafter).__name__}",
                        "draft_class": f"{type(drafter.model).__module__}.{type(drafter.model).__name__}",
                        "num_speculative_tokens": drafter.num_speculative_tokens,
                        "query_start_loc_policy": (
                            "static_dflash2"
                            if getattr(drafter, "_dflash_query_start_loc_cpu", None)
                            is not None
                            else "per_step"
                        ),
                        "causal": drafter.dflash_causal,
                        "full_graph_wrapper": isinstance(
                            drafter.model, graph_module.DFlashDrafterCUDAGraphWrapper
                        ),
                        "graph_architecture": "proposal_or_eager_context_tail",
                        "graph_state_policy": "fixed_dependencies_explicit_invalidation",
                        "graph_lifecycle_ready": drafter.graph_lifecycle_ready,
                        "proposal_graph_wrapper": combined is not None,
                        "proposal_graph_max_entries": (
                            combined.max_entries if combined is not None else None
                        ),
                        "proposal_graph_pool_policy": (
                            "upstream_global" if combined is not None else None
                        ),
                    }
                )
            return result

        self._patch(proposer_cls, "load_model", load)
        original_observe = stats_cls.observe_draft

        @functools.wraps(original_observe)
        def observe(stats, num_draft_tokens, num_accepted_tokens):
            result = original_observe(stats, num_draft_tokens, num_accepted_tokens)
            self.observe_acceptance(num_draft_tokens, num_accepted_tokens)
            return result

        self._patch(stats_cls, "observe_draft", observe)
        wrapper_cls = graph_module.DFlashDrafterCUDAGraphWrapper
        original_call = wrapper_cls.__call__
        original_base_call = base_graph_cls.__call__
        original_signature = graph_module._graph_signature

        @functools.wraps(original_signature)
        def signature(*args, **kwargs):
            with self.stage("attention_graph_signature"):
                result = original_signature(*args, **kwargs)
            proposal = getattr(self._local, "proposal", None)
            if proposal is not None and result is None:
                proposal.fallback_reason = "metadata_ineligible"
            return result

        @functools.wraps(original_base_call)
        def base_call(wrapper, *args, **kwargs):
            proposal = getattr(self._local, "proposal", None)
            if proposal is not None and wrapper is getattr(self._local, "graph", None):
                ctx = get_context()
                if ctx.cudagraph_runtime_mode == graph_mode.FULL:
                    entry = wrapper.concrete_cudagraph_entries.get(ctx.batch_descriptor)
                    query_capture = entry is None or entry.cudagraph is None
                    proposal.capture = proposal.capture or query_capture
                    proposal.graph_path = (
                        "full_graph_capture" if query_capture else "full_graph_replay"
                    )
                    proposal.fallback_reason = None
            return original_base_call(wrapper, *args, **kwargs)

        @functools.wraps(original_call)
        def graph_call(wrapper, *args, **kwargs):
            proposal = getattr(self._local, "proposal", None)
            if proposal is None:
                return original_call(wrapper, *args, **kwargs)
            proposal.graph_path = "full_graph_fallback"
            ctx = get_context()
            if ctx is None:
                proposal.fallback_reason = "no_forward_context"
            elif ctx.cudagraph_runtime_mode != graph_mode.PIECEWISE:
                proposal.fallback_reason = "runtime_mode_not_piecewise"
            elif ctx.batch_descriptor is None:
                proposal.fallback_reason = "no_batch_descriptor"
            else:
                proposal.fallback_reason = "unstable_input_addresses"
            self._local.graph = wrapper
            try:
                with self.stage("query_forward"):
                    return original_call(wrapper, *args, **kwargs)
            finally:
                self._local.graph = None

        self._patch(graph_module, "_graph_signature", signature)
        self._patch(base_graph_cls, "__call__", base_call)
        self._patch(wrapper_cls, "__call__", graph_call)
        proposal_cls = graph_module.DFlashProposalCUDAGraphWrapper
        for name, stage in (
            ("__call__", "proposal_graph_dispatch"),
            ("stage_inputs", "proposal_graph_staging"),
            ("run_overflow", "proposal_graph_overflow"),
            ("prepare_capture", "proposal_capture_prepare"),
            ("_create_inputs", "proposal_capture_inputs"),
            ("_capture_entry", "proposal_capture_entry"),
            ("_attach_buffers", "proposal_buffer_attach"),
            ("_detach_buffers", "proposal_buffer_detach"),
            ("make_descriptor", "proposal_descriptor"),
            ("sync_offloader", "proposal_offloader_sync"),
            ("replay_graph", "proposal_graph_submit"),
        ):
            self._patch(
                proposal_cls, name, self.wrap_stage(stage, getattr(proposal_cls, name))
            )
        for name, stage in (
            ("_host_metadata_matches", "proposal_host_plan_check"),
            ("copy_output", "proposal_output_copy"),
        ):
            self._patch(
                proposal_cls,
                name,
                staticmethod(self.wrap_stage(stage, getattr(proposal_cls, name))),
            )

    def summary(self) -> dict[str, Any]:
        # Caller synchronizes once after stop(), never per stage or inside a graph.
        device = {}
        for name, (start, end) in self._event_pairs:
            if name not in device:
                device[name] = self._timings_factory()
            device[name].add(round(start.elapsed_time(end) * 1_000_000))
        stages = {
            name: {key: value.summary() for key, value in timings.items()}
            | {"device_stream": device[name].summary() if name in device else None}
            for name, timings in sorted(self.stages.items())
        }
        positions = range(max(self.drafted_per_position, default=-1) + 1)
        return {
            "stages": stages,
            "outside_propose": {
                name: {key: value.summary() for key, value in timings.items()}
                for name, timings in sorted(self.outside_propose.items())
            },
            "graph_paths": dict(self.graph_paths),
            "fallback_reasons": dict(self.fallback_reasons),
            "proposal_graph_paths": dict(self.proposal_graph_paths),
            "proposal_fallback_reasons": dict(self.proposal_fallback_reasons),
            "proposal_rows": dict(self.proposal_rows),
            "duplicate_segment_dispatches": self.duplicate_segment_dispatches,
            "slow_proposals": self._slow_proposal_summary(),
            "proposal_graph_cache": [
                {
                    "entries": len(wrapper._inputs),
                    "max_entries": wrapper.max_entries,
                    "generation": wrapper.generation,
                    "last_clear_reason": wrapper.last_clear_reason,
                    "capture_preparations": wrapper.capture_preparations,
                    "captured_tensor_count": len(wrapper._capture_tensors or ()),
                    "capture_attempts": wrapper.capture_attempts,
                    "state_policy": "fixed_dependencies_explicit_invalidation",
                    "replay_implementations": graph_replay_implementations(
                        wrapper.concrete_cudagraph_entries.values()
                    ),
                    "planning_policy": "fixed_full_equivalent; geometric_across_window",
                    "planning_state": wrapper.planning_state,
                    "planning_counts": dict(wrapper.planning_counts),
                    "entry_shapes": [
                        {
                            "B": desc.num_reqs,
                            "num_tokens": desc.num_tokens,
                            "context_capacity": desc.context_capacity,
                            "attention_signature": desc.attention_signature,
                        }
                        for desc in wrapper._inputs
                    ],
                    "pool_policy": "upstream_global",
                }
                for wrapper in self._proposal_wrappers
            ],
            "capture_proposals_excluded": self.capture_proposals.summary(),
            "steady_state_clean": (
                self.capture_proposals.summary()["calls"] == 0
                and self.in_flight_at_stop == 0
            ),
            "in_flight_at_stop": self.in_flight_at_stop,
            "errors": self.errors,
            "acceptance": {
                "available": self.num_drafts > 0,
                "num_drafts": self.num_drafts,
                "num_draft_tokens": self.num_draft_tokens,
                "num_accepted_tokens": self.num_accepted_tokens,
                "acceptance_rate": (
                    self.num_accepted_tokens / self.num_draft_tokens
                    if self.num_draft_tokens
                    else None
                ),
                "mean_acceptance_length_including_bonus": (
                    1 + self.num_accepted_tokens / self.num_drafts
                    if self.num_drafts
                    else None
                ),
                "accepted_per_position": [
                    self.accepted_per_position[p] for p in positions
                ],
                "drafted_per_position": [
                    self.drafted_per_position[p] for p in positions
                ],
            },
            "timing_semantics": {
                "host": "wall time; self subtracts observed nested spans only; stages are not additive",
                "device": "optional current-stream event elapsed time, not kernel-only time; nested and not additive",
                "gather": "eager selection: query-return to selection includes gather/bookkeeping; combined replay bypasses Python gather/selection",
                "capture": "entire proposals that capture a query or combined proposal graph (including warmup) are excluded from stage timings; generation wall time/acceptance still include them",
                "context": "context_precompute measures eager model precompute; combined replay bypasses it, and proposal_graph_overflow measures extra rows outside the fixed-capacity graph",
                "graph_evidence": "graph_paths/proposal_graph_paths record software wrapper/API dispatch; they do not prove native device graph launches or backend capture coverage; validate with runtime trace",
                "scope": "MRV1 DFlash2 combined context/query/selection graph or eager context/selection with configured query execution; no standalone context/tail graphs",
                "proposal_graph": "proposal_graph_dispatch includes dynamic eligibility/metadata checks, cold capture, staging, overflow, replay and independent output clone; all capture attempts exclude the whole proposal from stage timings; duplicate_segment_dispatches counts combined hits that also dispatched the query-only graph wrapper; cache/capture counters include warmup and reset on clear; generation increases on clear",
                "proposal_dispatch_breakdown": "descriptor and host_plan_check keep full validation; offloader_sync precedes graph_submit; graph_submit is the replay API call, not device kernel duration; output_copy retains independent storage; attention_graph_signature covers both combined and legacy query paths; added nested stages change dispatch self accounting",
                "graph_lifecycle": "model/KV/RoPE dependencies stay fixed between explicit clear boundaries; capture preparation holds tensors once per generation, never scans model parameters on replay; runner/worker hooks invalidate before and after resource changes; direct external changes must clear at a quiescent boundary",
                "slow_proposals": "up to 3 slowest successful non-capture proposals, with all existing spans from the same call; ordinal includes successful cold proposals since start(); inclusive nested spans are not additive; serialization runs in summary without new device synchronization",
            },
        }


def install_proposal_profiler(timings_factory: Callable, device_events: bool):
    if device_events and os.getenv("XPU_DUMMY_EVENT") == "1":
        raise RuntimeError(
            "Device event timing requires real events; XPU_DUMMY_EVENT=1"
        )
    import kunlun_ops
    import torch
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.v1.spec_decode.metrics import SpecDecodingStats

    from vllm_kunlun.models.qwen3_dflash2 import (
        CandidateSelector,
        DFlash2Qwen3ForCausalLM,
    )
    from vllm_kunlun.v1.sample.spec_decode import dflash_cudagraph
    from vllm_kunlun.v1.sample.spec_decode.dflash import DFlashProposer

    profiler = ProposalProfiler(timings_factory, torch.cuda if device_events else None)
    profiler.install(
        DFlashProposer,
        DFlash2Qwen3ForCausalLM,
        CandidateSelector,
        dflash_cudagraph,
        CUDAGraphWrapper,
        kunlun_ops,
        SpecDecodingStats,
        lambda: get_forward_context() if is_forward_context_available() else None,
        CUDAGraphMode,
    )
    return profiler


class TargetDiffDiagnostics:
    """Synchronous B1/eager observations for the target/spec precision investigation.

    Installed only by --diagnose-target-diff. Reads small logits/metadata to CPU
    and observes one target GDN layer. The optional shadow/repair experiment
    replays native recurrence privately; only repair changes the live base slot.
    These records and the corresponding wall times are not performance data.
    """

    def __init__(self, torch, state_mode: str = "observe"):
        self.torch = torch
        self.state_mode = state_mode
        self.active = False
        self.iteration = 0
        self.events: list[dict[str, Any]] = []
        self._prefix: list[int] = []
        self._rejection_raw = None
        self._gdn = None
        self._layer = None
        self._cache_mode = None
        self._current_layer = None
        self.state_writeback_checks: list[dict[str, Any]] = []
        self._patches = []
        self.watch_ids = (411, 22525)

    def start_iteration(self, iteration: int) -> None:
        self.iteration = iteration
        self._prefix = []
        self._gdn = None
        self.active = True

    def stop(self) -> None:
        self.active = False

    @staticmethod
    def _cpu(value):
        return value.detach().cpu().tolist() if value is not None else None

    def _logits(self, logits):
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Target diff diagnostics cannot run inside a graph")
        values, ids = logits.float().topk(min(5, logits.shape[-1]), dim=-1)
        watches = [i for i in self.watch_ids if i < logits.shape[-1]]
        return {
            "argmax_ids": self._cpu(logits.argmax(dim=-1)),
            "top_ids": self._cpu(ids),
            "top_values": [
                [v if math.isfinite(v) else None for v in row]
                for row in self._cpu(values)
            ],
            "watch_ids": watches,
            "watch_values": [
                [v if math.isfinite(v) else None for v in row]
                for row in self._cpu(logits[:, watches].float())
            ],
            "nonfinite_count": int((~self.torch.isfinite(logits)).sum().item()),
        }

    def _record(self, event, output):
        if output.ndim != 2 or output.shape[0] != 1:
            raise RuntimeError("Target diff diagnostics require B1 sampler output")
        ids = self._cpu(output)[0]
        emitted = [int(i) for i in ids if i >= 0]
        event.update(
            iteration=self.iteration,
            generated_start=len(self._prefix),
            generated_prefix=list(self._prefix),
            sampled_ids=ids,
            emitted_ids=emitted,
            gdn=self._gdn,
        )
        self.events.append(event)
        self._prefix.extend(emitted)

    def install(self) -> None:
        import vllm.v1.sample.rejection_sampler as rejection_module
        from vllm.forward_context import get_forward_context
        from vllm.v1.sample.sampler import Sampler

        import vllm_kunlun.models.qwen3_next as target_gdn_module
        from vllm_kunlun.models.qwen3_next import Qwen3NextGatedDeltaNet
        from vllm_kunlun.v1.attention.backends.gdn_attn import (
            GDNAttentionMetadataBuilder,
        )

        # Reuse descriptor-aware patch/uninstall bookkeeping from the host profiler.
        def patch(owner, name, fn):
            ProposalProfiler._patch(self, owner, name, fn)

        if self.state_mode != "observe":
            original_recurrent = target_gdn_module.fused_recurrent_gated_delta_rule
            recurrent_signature = inspect.signature(original_recurrent)

            @functools.wraps(original_recurrent)
            def recurrent(*args, **kwargs):
                bound = recurrent_signature.bind(*args, **kwargs)
                bound.apply_defaults()
                values = bound.arguments
                accepted = values["num_accepted_tokens"]
                if accepted is None:
                    return original_recurrent(*args, **kwargs)
                if self.torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("GDN state experiment requires eager execution")
                counts = self._cpu(accepted)
                if len(counts) != 1:
                    raise RuntimeError("GDN state experiment requires one sequence")
                if counts[0] <= 1:
                    return original_recurrent(*args, **kwargs)
                state = values["initial_state"]
                indices = self._cpu(values["ssm_state_indices"])
                q = values["q"]
                width = len(indices[0])
                if (
                    not values["inplace_final_state"]
                    or len(indices) != 1
                    or not 1 < counts[0] <= width
                    or min(indices[0]) <= 0
                    or max(indices[0]) >= state.shape[0]
                    or len(set(indices[0])) != width
                    or q.shape[:2] != (1, width)
                    or self._cpu(values["cu_seqlens"]) != [0, width]
                    or self._current_layer is None
                ):
                    raise RuntimeError(
                        "GDN state experiment requires B1 padded unique valid candidate slots"
                    )
                # Same native kernel, inputs and query width, but private slots:
                # move the real accepted initial state into private column 0 and
                # set accepted=1, preventing its extra commit from overwriting S0.
                # The live call still returns its original output and candidates.
                private = state.new_zeros((width + 1, *state.shape[1:]))
                private[1].copy_(state[indices[0][counts[0] - 1]])
                shadow_values = dict(values)
                shadow_values.update(
                    initial_state=private,
                    ssm_state_indices=self.torch.arange(
                        1,
                        width + 1,
                        dtype=values["ssm_state_indices"].dtype,
                        device=state.device,
                    ).view(1, width),
                    num_accepted_tokens=self.torch.ones_like(accepted),
                )
                output = original_recurrent(*args, **kwargs)
                shadow_output = original_recurrent(**shadow_values)
                output_delta = (output[0].float() - shadow_output[0].float()).abs()
                base_delta = (state[indices[0][0]].float() - private[1].float()).abs()
                output_max = float(output_delta.max().item())
                base_max = float(base_delta.max().item())
                if not math.isfinite(output_max) or not math.isfinite(base_max):
                    raise RuntimeError("Nonfinite GDN state experiment result")
                if self.active:
                    self.state_writeback_checks.append(
                        {
                            "iteration": self.iteration,
                            "generated_start": len(self._prefix),
                            "layer": self._current_layer,
                            "accepted": counts[0],
                            "state_indices": indices[0],
                            "native_shadow_output_max_abs": output_max,
                            "base_vs_shadow_first_candidate_max_abs": base_max,
                            "base_repaired": self.state_mode == "repair",
                        }
                    )
                if self.state_mode == "repair":
                    # Isolate writeback: require bitwise identical current outputs
                    # before replacing only column 0 with native-computed S0.
                    if not self.torch.equal(output[0], shadow_output[0]):
                        raise RuntimeError(
                            "GDN native/private outputs differ; refusing state repair: "
                            f"layer={self._current_layer}, max_abs={output_max}"
                        )
                    state[indices[0][0]].copy_(private[1])
                return output

            patch(target_gdn_module, "fused_recurrent_gated_delta_rule", recurrent)

        original_build = GDNAttentionMetadataBuilder.build

        @functools.wraps(original_build)
        def build(builder, *args, **kwargs):
            if self.active:
                self._cache_mode = builder.vllm_config.cache_config.mamba_cache_mode
            return original_build(builder, *args, **kwargs)

        patch(GDNAttentionMetadataBuilder, "build", build)
        original_sampler = Sampler.forward
        sampler_signature = inspect.signature(original_sampler)

        @functools.wraps(original_sampler)
        def sample(sampler, *args, **kwargs):
            if not self.active or self._rejection_raw is not None:
                return original_sampler(sampler, *args, **kwargs)
            bound = sampler_signature.bind(sampler, *args, **kwargs).arguments
            snapshot = self._logits(bound["logits"])
            output = original_sampler(sampler, *args, **kwargs)
            self._record(
                {"kind": "target_only_or_regular", "raw_logits": snapshot},
                output.sampled_token_ids,
            )
            return output

        patch(Sampler, "forward", sample)
        rejection_cls = rejection_module.RejectionSampler
        original_forward = rejection_cls.forward
        forward_signature = inspect.signature(original_forward)

        @functools.wraps(original_forward)
        def forward(sampler, *args, **kwargs):
            if not self.active:
                return original_forward(sampler, *args, **kwargs)
            bound = forward_signature.bind(sampler, *args, **kwargs).arguments
            metadata = bound["metadata"]
            previous = self._rejection_raw
            self._rejection_raw = {
                "raw_logits": self._logits(bound["logits"]),
                "target_logits_indices": self._cpu(metadata.target_logits_indices),
                "bonus_logits_indices": self._cpu(metadata.bonus_logits_indices),
            }
            try:
                return original_forward(sampler, *args, **kwargs)
            finally:
                self._rejection_raw = previous

        patch(rejection_cls, "forward", forward)
        original_reject = rejection_module.rejection_sample
        reject_signature = inspect.signature(original_reject)

        @functools.wraps(original_reject)
        def reject(*args, **kwargs):
            if not self.active:
                return original_reject(*args, **kwargs)
            bound = reject_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            if (
                len(values["num_draft_tokens"]) != 1
                or not values["sampling_metadata"].all_greedy
                or values["synthetic_mode"]
            ):
                raise RuntimeError(
                    "Target diff diagnostics require B1, greedy, non-synthetic sampling"
                )
            drafts = self._cpu(values["draft_token_ids"])
            processed = self._logits(values["target_probs"])
            bonus = int(values["bonus_token_ids"].reshape(-1)[0].item())
            targets = processed["argmax_ids"]
            expected = []
            for draft, target in zip(drafts, targets):
                expected.append(int(target))
                if draft != target:
                    break
            else:
                expected.append(bonus)
            expected += [-1] * (values["max_spec_len"] + 1 - len(expected))
            output = original_reject(*args, **kwargs)
            self._record(
                {
                    "kind": "rejection",
                    **(self._rejection_raw or {}),
                    "draft_ids": drafts,
                    "num_draft_tokens": list(values["num_draft_tokens"]),
                    "cu_num_draft_tokens": self._cpu(values["cu_num_draft_tokens"]),
                    "processed_target_logits": processed,
                    "bonus_id": bonus,
                    "expected_greedy_ids": expected,
                    "greedy_kernel_matches": self._cpu(output)[0] == expected,
                },
                output,
            )
            return output

        patch(rejection_module, "rejection_sample", reject)
        original_core = Qwen3NextGatedDeltaNet._forward_core

        @functools.wraps(original_core)
        def core(layer, *args, **kwargs):
            if not self.active or (
                self._layer is not None and self._layer != layer.prefix
            ):
                return original_core(layer, *args, **kwargs)
            context = get_forward_context()
            metadata = context.attn_metadata
            if not metadata or layer.prefix not in metadata:
                return original_core(layer, *args, **kwargs)
            self._layer = layer.prefix
            md = metadata[layer.prefix]
            snapshot = {
                "layer": layer.prefix,
                "num_prefills": md.num_prefills,
                "num_decodes": md.num_decodes,
                "num_spec_decodes": md.num_spec_decodes,
                "num_actual_tokens": md.num_actual_tokens,
                "mamba_cache_mode": self._cache_mode,
                "conv_state_dtype": str(layer.kv_cache[0].dtype),
                "ssm_state_dtype": str(layer.kv_cache[1].dtype),
                "accepted": self._cpu(md.num_accepted_tokens),
                "state_indices": self._cpu(md.spec_state_indices_tensor),
                "non_spec_state_indices": self._cpu(md.non_spec_state_indices_tensor),
                "conv_indices": self._cpu(md.spec_conv_state_indices_tensor),
                "spec_query_start_loc": self._cpu(md.spec_query_start_loc),
                "non_spec_query_start_loc": self._cpu(md.non_spec_query_start_loc),
            }
            result = original_core(layer, *args, **kwargs)
            if md.num_spec_decodes and snapshot["state_indices"]:
                indices = snapshot["state_indices"][0]
                accepted = snapshot["accepted"][0]
                if (
                    1 < accepted <= len(indices)
                    and indices[0] > 0
                    and indices[accepted - 1] > 0
                    and indices[0] != indices[accepted - 1]
                ):
                    state = layer.kv_cache[1]
                    delta = (
                        state[indices[0]].float() - state[indices[accepted - 1]].float()
                    ).abs()
                    value = float(delta.max().item())
                    snapshot["base_vs_previous_acceptance_candidate_max_abs"] = (
                        value if math.isfinite(value) else None
                    )
            self._gdn = snapshot
            return result

        @functools.wraps(original_core)
        def tracked_core(layer, *args, **kwargs):
            previous = self._current_layer
            self._current_layer = layer.prefix
            try:
                return core(layer, *args, **kwargs)
            finally:
                self._current_layer = previous

        patch(Qwen3NextGatedDeltaNet, "_forward_core", tracked_core)

    def uninstall(self) -> None:
        ProposalProfiler.uninstall(self)

    def summary(self, token_ids_by_iteration) -> dict[str, Any]:
        emitted_matches = []
        for iteration, outputs in enumerate(token_ids_by_iteration):
            emitted = [
                token
                for event in self.events
                if event["iteration"] == iteration
                for token in event["emitted_ids"]
            ]
            emitted_matches.append(emitted[: len(outputs[0])] == outputs[0])
        return {
            "schema_version": 2,
            "not_for_performance": True,
            "gdn_state_mode": self.state_mode,
            "modifies_live_gdn_state": self.state_mode == "repair",
            "state_writeback_checks": self.state_writeback_checks,
            "gdn_layer": self._layer,
            "events": self.events,
            "greedy_kernel_mismatches": sum(
                e.get("greedy_kernel_matches") is False for e in self.events
            ),
            "emitted_prefix_matches_output": emitted_matches,
            "notes": "B1/eager only; CPU reads perturb timing; GDN slot equality is evidence, not a root-cause assertion; extra sampled IDs past max_tokens may be discarded by the engine",
        }
