# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kunlun DFlash/DFlash2 proposer compatibility for model runner V1."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from typing_extensions import override
from vllm.config import VllmConfig
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer as UpstreamDFlashProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


def is_dflash2_draft(speculative_config: Any) -> bool:
    """Return whether a speculative config selects a DFlash2 draft model."""
    if speculative_config is None or speculative_config.method != "dflash":
        return False
    draft_config = speculative_config.draft_model_config
    return "DFlash2DraftModel" in (draft_config.architectures or [])


def greedy_select_path(
    candidate_ids: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    """Walk the best DFlash2 candidate path from candidate index zero."""
    if candidate_ids.ndim != 3 or scores.ndim != 4:
        raise ValueError(
            "DFlash2 path selection expects candidate_ids [B,L,K] and "
            f"scores [B,L,K,K], got {candidate_ids.shape} and {scores.shape}."
        )
    num_reqs, num_steps, top_k = candidate_ids.shape
    if scores.shape != (num_reqs, num_steps, top_k, top_k):
        raise ValueError(
            f"DFlash2 score shape {scores.shape} does not match "
            f"candidate shape {candidate_ids.shape}."
        )

    request_indices = torch.arange(num_reqs, device=candidate_ids.device)
    predecessor_indices = torch.zeros(
        num_reqs, dtype=torch.long, device=candidate_ids.device
    )
    selected = []
    for step in range(num_steps):
        edge_scores = scores[request_indices, step, predecessor_indices]
        predecessor_indices = edge_scores.argmax(dim=-1)
        selected.append(candidate_ids[request_indices, step, predecessor_indices])
    if not selected:
        return candidate_ids.new_empty((num_reqs, 0))
    return torch.stack(selected, dim=1)


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
        super().__init__(vllm_config, device, runner=runner)
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
    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Torch replacement for vLLM's Triton padded-token kernel."""
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

        # A discarded row must use its target-side backup token regardless of
        # the sampled contents. Mark the whole row invalid before counting.
        valid_sampled_token_ids = sampled_token_ids.clone()
        discarded = torch.nonzero(
            discard_request_mask[:num_reqs], as_tuple=False
        ).flatten()
        if discarded.numel() > 0:
            discarded = discarded.to(
                device=valid_sampled_token_ids.device,
                dtype=torch.long,
                non_blocking=True,
            )
            valid_sampled_token_ids.index_fill_(0, discarded, -1)

        valid_mask = (valid_sampled_token_ids != -1) & (
            valid_sampled_token_ids < gpu_input_batch.vocab_size
        )
        valid_count_long = valid_mask.sum(dim=1)
        valid_sampled_tokens_count = valid_count_long.to(torch.int32)

        # Rejections are padded with -1, but use the mask rather than assuming
        # they form a contiguous suffix so this matches the Triton reference.
        token_columns = torch.arange(
            valid_sampled_token_ids.shape[1],
            device=valid_sampled_token_ids.device,
        )
        last_valid_indices = torch.where(
            valid_mask,
            token_columns[None, :],
            token_columns.new_full((), -1),
        ).amax(dim=1)
        safe_indices = last_valid_indices.clamp_min(0)
        last_valid_tokens = valid_sampled_token_ids.gather(
            1, safe_indices[:, None]
        ).squeeze(1)
        backup_tokens = self.backup_next_token_ids.gpu[:num_reqs]
        next_token_ids = torch.where(
            last_valid_indices >= 0, last_valid_tokens, backup_tokens
        )
        return next_token_ids, valid_sampled_tokens_count

    @override
    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """Torch replacement for vLLM's Triton padded-input kernel."""
        num_reqs = common_attn_metadata.num_reqs
        cumulative_draft_tokens = spec_decode_metadata.cu_num_draft_tokens
        num_draft_tokens = cumulative_draft_tokens.clone()
        if num_reqs > 1:
            num_draft_tokens[1:] = (
                cumulative_draft_tokens[1:] - cumulative_draft_tokens[:-1]
            )

        valid_count = valid_sampled_tokens_count.to(num_draft_tokens.dtype)
        num_rejected_tokens = num_draft_tokens + 1 - valid_count
        num_rejected_tokens = torch.where(
            num_draft_tokens > 0,
            num_rejected_tokens,
            torch.zeros_like(num_rejected_tokens),
        ).to(torch.int32)
        last_query_token = common_attn_metadata.query_start_loc[1:] - 1
        token_indices_to_sample = (last_query_token - num_rejected_tokens).to(
            torch.int32
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
            num_rejected_tokens,
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
        (
            input_ids,
            context_positions,
            query_positions,
            context_slot_mapping,
            query_slot_mapping,
            token_indices_to_sample,
        ) = copy_and_expand_dflash_inputs_native(
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            query_start_loc=cad.query_start_loc,
            block_table=cad.block_table_tensor,
            block_size=self.block_size,
            num_speculative_tokens=self.num_speculative_tokens,
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            num_rejected_tokens=num_rejected_tokens_gpu,
        )
        self.input_ids[:num_query_total].copy_(input_ids)
        self.positions[:num_query_total].copy_(query_positions)
        self._context_positions_buffer[:num_context].copy_(context_positions)
        self._slot_mapping_buffer[:num_query_total].copy_(query_slot_mapping)
        self._context_slot_mapping_buffer[:num_context].copy_(context_slot_mapping)

        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req
        effective_seq_lens = cad.seq_lens
        if num_rejected_tokens_gpu is not None:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu
        new_seq_lens_cpu_upper_bound = (
            cad.seq_lens_cpu_upper_bound + num_query_per_req
            if cad.seq_lens_cpu_upper_bound is not None
            else None
        )
        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens + num_query_per_req,
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * num_query_per_req
            ),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu_upper_bound,
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=cad.max_seq_len + num_query_per_req,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=self._slot_mapping_buffer[:num_query_total],
            causal=self.dflash_causal,
        )
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

    def compute_draft_token_ids(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
        anchor_indices = self.arange[:num_reqs].to(torch.long) * (1 + num_steps)
        anchor_token_ids = self.input_ids[anchor_indices]
        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
        )
        return greedy_select_path(candidate_ids, scores).reshape(-1)
