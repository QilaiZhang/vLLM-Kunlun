# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kunlun DFlash/DFlash2 proposer compatibility for model runner V1."""

from __future__ import annotations

import os
import sys
from typing import Any

import kunlun_ops
import numpy as np
import torch
from torch import nn
from typing_extensions import override
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer as UpstreamDFlashProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_kunlun.v1.sample.spec_decode.dflash_cudagraph import (
    DFlashDrafterCUDAGraphWrapper,
    DFlashProposalCUDAGraphWrapper,
)

logger = init_logger(__name__)


def _graph_lifecycle_ready() -> bool:
    """Fail closed if the MRV1 resource invalidation hooks were not installed."""
    return all(
        getattr(
            getattr(sys.modules.get(module), name, None),
            "_kunlun_dflash_lifecycle_patched",
            False,
        )
        for module, name in (
            ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
            ("vllm.v1.worker.gpu_worker", "Worker"),
        )
    )


def is_dflash2_draft(speculative_config: Any) -> bool:
    """Return whether a speculative config selects a DFlash2 draft model."""
    if speculative_config is None or speculative_config.method != "dflash":
        return False
    draft_config = speculative_config.draft_model_config
    return "DFlash2DraftModel" in (draft_config.architectures or [])


def copy_and_expand_dflash_inputs_native(
    next_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    num_speculative_tokens: int,
    parallel_drafting_token_id: int,
    num_rejected_tokens: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Native Torch fallback for the upstream CUDA Triton expansion kernel."""
    if target_positions.ndim > 1:
        # Text-only Qwen M-RoPE repeats the same position in every row. This
        # mirrors the 1-D pointer consumed by the upstream DFlash kernel.
        target_positions = target_positions[0]

    batch_size = next_token_ids.shape[0]
    num_query_per_req = num_speculative_tokens + 1
    device = next_token_ids.device
    offsets = torch.arange(
        num_query_per_req, device=device, dtype=target_positions.dtype
    )

    last_indices = query_start_loc[1:].to(torch.long) - 1
    if num_rejected_tokens is not None:
        last_indices -= num_rejected_tokens.to(torch.long)
    last_positions = target_positions[last_indices]
    query_positions_2d = last_positions[:, None] + 1 + offsets[None, :]
    query_positions = query_positions_2d.reshape(-1)

    context_lengths = query_start_loc[1:] - query_start_loc[:-1]
    request_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=device),
        context_lengths.to(torch.long),
        output_size=target_positions.numel(),
    )
    max_block_index = block_table.shape[1] - 1
    context_block_numbers = (target_positions // block_size).clamp_max(max_block_index)
    context_block_ids = block_table[
        request_indices, context_block_numbers.to(torch.long)
    ]
    context_slots = context_block_ids * block_size + target_positions % block_size

    input_ids = torch.full(
        (batch_size, num_query_per_req),
        int(parallel_drafting_token_id),
        dtype=next_token_ids.dtype,
        device=device,
    )
    input_ids[:, 0] = next_token_ids

    block_numbers = (query_positions_2d // block_size).clamp_max(max_block_index)
    block_ids = block_table.gather(1, block_numbers.to(torch.long))
    query_slots = (block_ids * block_size + query_positions_2d % block_size).reshape(-1)

    query_indices = torch.arange(
        batch_size * num_query_per_req,
        device=device,
        dtype=torch.int32,
    ).view(batch_size, num_query_per_req)
    token_indices_to_sample = query_indices[:, 1:].reshape(-1)
    return (
        input_ids.reshape(-1),
        target_positions,
        query_positions,
        context_slots,
        query_slots,
        token_indices_to_sample,
    )


class DFlashProposer(UpstreamDFlashProposer):
    """DFlash V1 proposer with native expansion and DFlash2 selection."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        self.is_dflash2 = is_dflash2_draft(vllm_config.speculative_config)
        self._dflash_proposal_graph: DFlashProposalCUDAGraphWrapper | None = None
        super().__init__(vllm_config, device, runner=runner)
        self._dflash_sample_indices_buffer = torch.empty(
            self.input_ids.numel(),
            dtype=torch.int32,
            device=device,
        )
        self._dflash_seq_lens_buffer = torch.empty(
            self.max_batch_size, dtype=torch.int32, device=device
        )
        if self.is_dflash2:
            assert vllm_config.speculative_config is not None
            if vllm_config.speculative_config.draft_sample_method == "probabilistic":
                raise ValueError(
                    "DFlash2 probabilistic draft sampling is not supported on "
                    "the V1 model runner; use greedy (the default) instead."
                )
            self.selector_top_k = int(self.dflash_config["selector_top_k"])
            # The grouped convolution uses a fixed request block size. Varying
            # K needs a separately compiled model/kernel and is intentionally
            # outside this first V1 implementation.
            self.dflash2_num_speculative_tokens = self.num_speculative_tokens
            query_len = 1 + self.dflash2_num_speculative_tokens
            # Fixed K makes these prefixes immutable across batches. Keep
            # separate storage from the runner's mutable arange buffers.
            self._dflash_query_start_loc_buffer = (
                self.arange[: self.max_batch_size + 1] * query_len
            )
            self._dflash_query_start_loc_cpu = (
                torch.arange(self.max_batch_size + 1, dtype=torch.int32, device="cpu")
                * query_len
            )
            compilation_config = self.vllm_config.compilation_config
            max_capture_size = compilation_config.max_cudagraph_capture_size
            if (
                compilation_config.cudagraph_mode != CUDAGraphMode.NONE
                and compilation_config.cudagraph_capture_sizes
                and max_capture_size is not None
                and max_capture_size >= query_len
            ):
                # MRV1 only aligns the shared graph sizes when the target's
                # decode graph is FULL. The DFlash2 drafter also uses those
                # sizes in PIECEWISE mode, and every convolution requires a
                # whole number of fixed-width request blocks.
                compilation_config.adjust_cudagraph_sizes_for_spec_decode(
                    query_len,
                    self.vllm_config.parallel_config.tensor_parallel_size,
                )

    @property
    def dflash_config(self) -> dict[str, Any]:
        config = dict(
            getattr(self.draft_model_config.hf_config, "dflash_config", None) or {}
        )
        config.setdefault(
            "causal",
            bool(getattr(self.draft_model_config.hf_config, "is_causal", False)),
        )
        return config

    @override
    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
        if self.is_dflash2:
            # dummy_run may request fewer than K+1 query tokens, including
            # when graph use is disabled. Real drafting already has N*(K+1).
            query_len = 1 + self.dflash2_num_speculative_tokens
            num_tokens = ((num_tokens + query_len - 1) // query_len) * query_len
        (
            mode,
            num_input_tokens,
            num_tokens_across_dp,
        ) = super()._determine_batch_execution_and_padding(
            num_tokens, use_cudagraphs=use_cudagraphs
        )
        if self.is_dflash2 and num_input_tokens % query_len:
            raise ValueError(
                "DFlash2 graph padding must preserve complete request blocks: "
                f"num_input_tokens={num_input_tokens}, query_len={query_len}"
            )
        return mode, num_input_tokens, num_tokens_across_dp

    @override
    def load_model(self, target_model: nn.Module) -> None:
        self.clear_graphs("load_model")
        self._dflash_proposal_graph = None
        super().load_model(target_model)
        self.graph_lifecycle_ready = _graph_lifecycle_ready()
        if (
            self.is_dflash2
            and os.getenv("VLLM_KUNLUN_DFLASH2_PROPOSAL_CUDAGRAPH", "0") == "1"
        ):
            if not self.graph_lifecycle_ready:
                raise RuntimeError(
                    "DFlash2 proposal graphs require the MRV1 runner/worker "
                    "lifecycle hooks; deploy the matching compat_patches.py"
                )
            self._dflash_proposal_graph = DFlashProposalCUDAGraphWrapper(
                self.model,
                self._run_dflash2_tail,
                self.model_returns_tuple(),
                self.vllm_config,
                self.device,
                query_len=1 + self.dflash2_num_speculative_tokens,
            )
            logger.info(
                "DFlash2 proposal graph enabled (FP16/TP1/K3): max_entries=%d, "
                "pool=global; fallback uses eager context/selection and configured query execution",
                self._dflash_proposal_graph.max_entries,
            )
        if (
            self.is_dflash2
            and not self.dflash_causal
            and os.getenv("VLLM_KUNLUN_DRAFTER_FULL_CUDAGRAPH", "0") == "1"
        ):
            self.model = DFlashDrafterCUDAGraphWrapper(
                self.model,
                self.vllm_config,
                self.device,
                query_len=1 + self.dflash2_num_speculative_tokens,
            )

    def clear_graphs(self, reason: str = "external_change") -> None:
        """Invalidate graphs before/after a quiescent resource mutation.

        Engine lifecycle hooks call this for KV, weights, config and sleep.
        Direct changes to captured model/KV/RoPE dependencies must do the same.
        """
        if not self.is_dflash2:
            return
        model = getattr(self, "model", None)
        query = model if isinstance(model, DFlashDrafterCUDAGraphWrapper) else None
        proposal = self._dflash_proposal_graph
        graphs = [graph for graph in (proposal, query) if graph is not None]
        if any(graph.concrete_cudagraph_entries for graph in graphs):
            torch.cuda.synchronize(self.device)
        for graph in graphs:
            graph.clear_graphs()
        if proposal is not None:
            proposal.last_clear_reason = reason
        raw_model = query.unwrap() if query is not None else model
        if raw_model is not None:
            raw_model.reset_graph_state()

    def _dflash2_graph_ineligibility(self) -> str | None:
        config = self.vllm_config
        parallel = config.parallel_config
        if self.dflash_causal or not self.parallel_drafting:
            return "requires_noncausal_parallel_dflash2"
        if (
            parallel.tensor_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
        ):
            return "unsupported_parallel_config"
        if config.model_config.enforce_eager:
            return "target_enforce_eager"
        if self.speculative_config.enforce_eager:
            return "draft_enforce_eager"
        if config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            return "compilation_graph_mode_none"
        if self._enable_probabilistic_draft_probs:
            return "probabilistic_draft"
        if self.eplb_state is not None:
            return "eplb"
        if config.lora_config is not None:
            return "lora_config"
        return None

    @override
    def propose(
        self,
        num_speculative_tokens,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: (
            dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None
        ) = None,
    ) -> torch.Tensor:
        proposal_graph = self._dflash_proposal_graph
        reason = None
        if proposal_graph is not None:
            proposal_graph.reset_result()
            reason = self._dflash2_graph_ineligibility()
            # MRV1 also supplies an empty MM tuple for text-only Qwen batches.
            # Inspect the host list without reading the device mask.
            if mm_embed_inputs is not None and mm_embed_inputs[0]:
                reason = "multimodal_inputs"
            if reason is not None:
                proposal_graph.record_fallback(
                    reason, common_attn_metadata.batch_size()
                )
        if proposal_graph is None or reason is not None:
            return super().propose(
                num_speculative_tokens,
                target_token_ids,
                target_positions,
                target_hidden_states,
                next_token_ids,
                token_indices_to_sample,
                common_attn_metadata,
                sampling_metadata,
                mm_embed_inputs,
                num_rejected_tokens_gpu,
                slot_mappings,
            )
        # Prepare once before attempting the combined graph. Rejection has made
        # no live KV writes, so the prepared inputs can continue eagerly below.
        self.num_speculative_tokens = num_speculative_tokens
        self._last_draft_probs = None
        target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
        assert target_hidden_states.shape[-1] == self.hidden_size
        num_tokens, sample_indices, draft_cad = self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
        _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
            draft_cad
        )
        mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )
        # Decide before build_model_inputs_first_pass writes context KV. A
        # declined proposal graph has made no live writes, so this first pass
        # can safely continue on the original segmented/eager path.
        query_slots = self._get_slot_mapping(num_input_tokens, draft_cad.slot_mapping)
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=mode,
            slot_mapping=query_slots,
        ):
            count = self._dflash_num_context
            output = proposal_graph(
                self._dflash_hidden_states,
                self._context_positions_buffer[:count],
                self._context_slot_mapping_buffer[:count],
                self.input_ids[:num_input_tokens],
                self._get_positions(num_input_tokens),
                sample_indices,
                capacity=min(num_tokens, self.max_num_tokens),
            )
        if output is not None:
            return output.view(-1, self.num_speculative_tokens)
        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=mode,
            slot_mapping=(
                query_slots
                if slot_mapping_size == num_input_tokens
                else self._get_slot_mapping(slot_mapping_size, draft_cad.slot_mapping)
            ),
        ):
            result = self.model(**model_kwargs)
            last_hidden_states = result[0] if self.model_returns_tuple() else result
        return self._run_dflash2_tail(
            last_hidden_states, sample_indices, self.input_ids[:num_tokens]
        ).view(-1, self.num_speculative_tokens)

    def _run_dflash2_tail(
        self,
        hidden_states: torch.Tensor,
        sample_indices: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        anchor_ids = input_ids.view(-1, 1 + self.num_speculative_tokens)[:, 0]
        return self.compute_draft_token_ids(hidden_states[sample_indices], anchor_ids)

    @override
    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare padded tokens with the Kunlun EAGLE kernel."""
        num_reqs = gpu_input_batch.num_reqs
        self.backup_next_token_ids.np[:num_reqs] = np.asarray(
            [
                requests[gpu_input_batch.req_ids[req_idx]].get_token_id(
                    gpu_input_batch.num_tokens_no_spec[req_idx] - 1
                )
                for req_idx in range(num_reqs)
            ],
            dtype=np.int32,
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)

        batch_size, num_tokens = sampled_token_ids.shape
        assert batch_size == num_reqs
        next_token_ids = torch.empty(
            batch_size, dtype=torch.int32, device=sampled_token_ids.device
        )
        valid_sampled_tokens_count = next_token_ids.new_empty(batch_size)

        kunlun_ops.eagle_prepare_next_token_padded(
            sampled_token_ids,
            discard_request_mask[:batch_size],
            self.backup_next_token_ids.gpu[:batch_size],
            next_token_ids,
            valid_sampled_tokens_count,
            gpu_input_batch.vocab_size,
            num_tokens,
            batch_size,
        )
        return next_token_ids, valid_sampled_tokens_count

    @override
    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """Prepare padded inputs with the Kunlun EAGLE kernel."""
        num_reqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device
        token_indices_to_sample = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )
        num_rejected_tokens_gpu = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )
        kunlun_ops.eagle_prepare_inputs_padded_v2(
            spec_decode_metadata.cu_num_draft_tokens[:num_reqs],
            valid_sampled_tokens_count[:num_reqs],
            common_attn_metadata.query_start_loc[: num_reqs + 1],
            token_indices_to_sample,
            num_rejected_tokens_gpu,
            num_reqs,
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        total_num_tokens = int(query_start_loc_cpu[-1])
        padded_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=common_attn_metadata.seq_lens,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=(common_attn_metadata._num_computed_tokens_cpu),
            seq_lens_cpu_upper_bound=(common_attn_metadata.seq_lens_cpu_upper_bound),
            num_reqs=num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=int(query_lens_cpu.max()),
            max_seq_len=common_attn_metadata.max_seq_len,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )
        return (
            padded_metadata,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
        )

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        del token_indices_to_sample
        if (
            self.is_dflash2
            and self.num_speculative_tokens != self.dflash2_num_speculative_tokens
        ):
            raise ValueError(
                "DFlash2 dynamic num_speculative_tokens is not supported by "
                "the V1 Kunlun implementation. Use the configured static value "
                f"{self.dflash2_num_speculative_tokens}."
            )
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        self._dflash_num_context = num_context
        self._dflash_hidden_states = target_hidden_states
        token_indices_to_sample = self._dflash_sample_indices_buffer[
            : batch_size * self.num_speculative_tokens
        ]
        kunlun_ops.copy_and_expand_dflash_inputs(
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            query_start_loc=cad.query_start_loc,
            block_table=cad.block_table_tensor,
            block_size=self.block_size,
            num_speculative_tokens=self.num_speculative_tokens,
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            num_rejected_tokens=num_rejected_tokens_gpu,
            input_ids_out=self.input_ids[:num_query_total],
            context_positions_out=self._context_positions_buffer[:num_context],
            query_positions_out=self.positions[:num_query_total],
            context_slots_out=self._context_slot_mapping_buffer[:num_context],
            query_slots_out=self._slot_mapping_buffer[:num_query_total],
            sample_indices_out=token_indices_to_sample,
        )

        if self.is_dflash2:
            new_query_start_loc = self._dflash_query_start_loc_buffer[: batch_size + 1]
            new_query_start_loc_cpu = self._dflash_query_start_loc_cpu[: batch_size + 1]
        else:
            new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req
            new_query_start_loc_cpu = (
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * num_query_per_req
            )
        effective_seq_lens = self._dflash_seq_lens_buffer[:batch_size]
        effective_seq_lens.copy_(cad.seq_lens[:batch_size])
        if num_rejected_tokens_gpu is not None:
            effective_seq_lens.sub_(num_rejected_tokens_gpu[:batch_size])
        effective_seq_lens.add_(num_query_per_req)
        seq_lens_cpu_upper_bound = cad.seq_lens_cpu_upper_bound
        if seq_lens_cpu_upper_bound is None:
            # Older vLLM runners do not populate the per-request upper bound.
            # An existing host mirror is safe; otherwise max_seq_len is the
            # batch-wide CPU bound computed by the runner. Neither path reads
            # the device seq_lens back to the host.
            seq_lens_cpu_upper_bound = getattr(cad, "_seq_lens_cpu", None)
        if seq_lens_cpu_upper_bound is None:
            seq_lens_cpu_upper_bound = torch.full(
                (batch_size,),
                int(cad.max_seq_len),
                dtype=torch.int32,
                device="cpu",
            )
        new_seq_lens_cpu_upper_bound = (
            seq_lens_cpu_upper_bound[:batch_size] + num_query_per_req
        )
        max_seq_len = int(new_seq_lens_cpu_upper_bound.max().item())
        use_proposal_graph = (
            self._dflash_proposal_graph is not None
            and self._dflash2_graph_ineligibility() is None
            and self.vllm_config.model_config.dtype == torch.float16
            and self.dflash2_num_speculative_tokens == 3
        )
        if isinstance(self.model, DFlashDrafterCUDAGraphWrapper) or use_proposal_graph:
            # One conservative host planning LoD per graph bucket. The device
            # LoD below retains the real, possibly rejected-token length.
            capacity = cad.block_table_tensor.shape[1] * self.block_size
            if capacity <= 0:
                raise ValueError("DFlash requires a nonempty KV page table")
            upper = max_seq_len
            bucket = (
                self._dflash_proposal_graph.planning_upper_bound(capacity, upper)
                if use_proposal_graph
                else None
            )
            # Never under-plan an unexpected CPU bound beyond the configured
            # maximum; fall back to the original page-capacity-bounded plan.
            if bucket is None or upper > bucket:
                bucket = 1 << (upper - 1).bit_length()
            # Allocated pages also bound every valid device length, even when
            # the scheduler's optimistic upper bound includes rejected tokens.
            bucket = min(bucket, capacity)
            new_seq_lens_cpu_upper_bound = torch.full_like(
                new_seq_lens_cpu_upper_bound, bucket
            )
            max_seq_len = bucket
        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens,
            query_start_loc_cpu=new_query_start_loc_cpu,
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu_upper_bound,
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=max_seq_len,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=self._slot_mapping_buffer[:num_query_total],
            causal=self.dflash_causal,
        )
        # The exact lengths stay on the device during asynchronous drafting.
        # The host mirror is an upper bound for attention planning only.
        new_cad._kunlun_dflash_static_lod = self.is_dflash2 and not self.dflash_causal
        return num_query_total, token_indices_to_sample, new_cad

    @override
    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        if self.is_dflash2:
            if getattr(self.model, "draft_id_to_target_id", None) is not None:
                raise ValueError(
                    "DFlash2 does not support a reduced draft vocabulary; "
                    "the selector TopK requires the target vocabulary."
                )
            self.model.has_own_lm_head = False
        super()._maybe_share_lm_head(target_language_model)

    @override
    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.is_dflash2:
            return super()._greedy_sample(hidden_states)
        return self.compute_draft_token_ids(hidden_states)

    def compute_draft_token_ids(
        self,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_steps = self.num_speculative_tokens
        num_samples = hidden_states.shape[0]
        if num_steps <= 0 or num_samples % num_steps != 0:
            raise ValueError(
                "DFlash2 expected a positive speculative length dividing the "
                f"hidden-state rows, got rows={num_samples}, steps={num_steps}."
            )
        num_reqs = num_samples // num_steps
        hidden = hidden_states.view(num_reqs, num_steps, -1)
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(num_reqs, num_steps, self.selector_top_k)
        unary_logits = unary_logits.view(num_reqs, num_steps, self.selector_top_k)
        if anchor_token_ids is None:
            anchor_indices = self.arange[:num_reqs].to(torch.long) * (1 + num_steps)
            anchor_token_ids = self.input_ids[anchor_indices]
        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
        )
        return kunlun_ops.dflash_greedy_select_path(candidate_ids, scores).reshape(-1)
