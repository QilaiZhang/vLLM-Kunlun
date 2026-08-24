# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import itertools
import logging
from collections.abc import Callable
from math import prod
from typing import Any

import torch
from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    get_conv_copy_spec,
    get_temporal_copy_spec,
    is_conv_state_dim_first,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch


@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    src_ptr = tl.load(src_ptrs + pid)
    dst_ptr = tl.load(dst_ptrs + pid)
    size = tl.load(sizes + pid)

    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size

        curr_src_ptr = (src_ptr + i + offsets).to(tl.pointer_type(tl.uint8))
        curr_dst_ptr = (dst_ptr + i + offsets).to(tl.pointer_type(tl.uint8))

        data = tl.load(curr_src_ptr, mask=mask)
        tl.store(curr_dst_ptr, data, mask=mask)


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch
    torch.ops.xspeedgate_ops.batch_memcpy(src_ptrs, dst_ptrs, sizes)


#     grid = (batch,)
#     BLOCK_SIZE = 1024
#     batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)

# def _make_uint8_view_from_ptr(ptr: int, size: int, device: torch.device) -> torch.Tensor:
#    storage = torch._C._construct_storage_from_data_pointer(ptr, device, size)
#    tensor = torch.empty(0, dtype=torch.uint8, device=device)
#    return tensor.set_(storage, 0, (size,), (1,))


# def batch_memcpy(src_ptrs, dst_ptrs, sizes):
#    batch = src_ptrs.shape[0]
#    assert dst_ptrs.shape[0] == batch
#    assert sizes.shape[0] == batch
#    if batch == 0:
#        return
#
#    device = src_ptrs.device
#    src_ptrs_cpu = src_ptrs.detach().cpu().tolist()
#    dst_ptrs_cpu = dst_ptrs.detach().cpu().tolist()
#    sizes_cpu = sizes.detach().cpu().tolist()
#
#    for src_ptr, dst_ptr, size in zip(src_ptrs_cpu, dst_ptrs_cpu, sizes_cpu):
#        if size <= 0 or src_ptr == dst_ptr:
#            continue
#        src = _make_uint8_view_from_ptr(src_ptr, size, device)
#        dst = _make_uint8_view_from_ptr(dst_ptr, size, device)
#        dst.copy_(src, non_blocking=True)


def get_mamba_groups(kv_cache_config: KVCacheConfig) -> tuple[list[int], MambaSpec]:
    mamba_group_ids: list[int] = []
    mamba_specs: list[MambaSpec] = []
    for i in range(len(kv_cache_config.kv_cache_groups)):
        kv_cache_spec = kv_cache_config.kv_cache_groups[i].kv_cache_spec
        if isinstance(kv_cache_spec, MambaSpec):
            mamba_group_ids.append(i)
            mamba_specs.append(kv_cache_spec)
    assert len(mamba_group_ids) > 0, "no mamba layers in the model"
    assert all(mamba_specs[0] == spec for spec in mamba_specs)
    return mamba_group_ids, mamba_specs[0]


@dataclasses.dataclass
class MambaCopyBuffers:
    src_ptrs: CpuGpuBuffer
    dst_ptrs: CpuGpuBuffer
    sizes: CpuGpuBuffer
    mamba_group_ids: list[int]
    mamba_spec: MambaSpec
    offset: int = 0

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: tuple[MambaStateCopyFunc, ...],
        make_buffer: Callable[..., CpuGpuBuffer],
    ) -> "MambaCopyBuffers":
        mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
        entries_per_req = sum(
            len(kv_cache_config.kv_cache_groups[gid].layer_names)
            for gid in mamba_group_ids
        ) * len(copy_funcs)
        n = max_num_reqs * entries_per_req
        return cls(
            src_ptrs=make_buffer(n, dtype=torch.int64),
            dst_ptrs=make_buffer(n, dtype=torch.int64),
            sizes=make_buffer(n, dtype=torch.int64),
            mamba_group_ids=mamba_group_ids,
            mamba_spec=mamba_spec,
        )


def collect_mamba_copy_meta(
    copy_bufs: MambaCopyBuffers,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    src_ptrs_np = copy_bufs.src_ptrs.np
    dst_ptrs_np = copy_bufs.dst_ptrs.np
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache
            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                copy_spec = state_copy_func(
                    state, block_ids, src_block_idx, accept_token_bias + 1
                )

                src_ptrs_np[offset] = copy_spec.start_addr
                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1

    copy_bufs.offset = offset


def do_mamba_copy_block(copy_bufs: MambaCopyBuffers):
    n = copy_bufs.offset
    if n == 0:
        return
    batch_memcpy(
        copy_bufs.src_ptrs.copy_to_gpu(n),
        copy_bufs.dst_ptrs.copy_to_gpu(n),
        copy_bufs.sizes.copy_to_gpu(n),
    )


def reanchor_spec_to_non_spec_states(
    runner: Any,
    scheduler_output: SchedulerOutput,
) -> int:
    """Move accepted speculative candidates to canonical non-spec state slots.

    In Mamba cache mode ``none``, each request owns one canonical block followed
    by speculative candidate blocks. A speculative GDN step writes conv history
    candidate ``a`` at row offset ``a - 1`` of the canonical block and writes the
    temporal candidate to block ``a - 1``. If the next suffix step has no draft,
    the request switches to non-spec kernels, which read only canonical offset 0.
    Re-anchor only those transitions before attention metadata is built.
    """
    if (
        runner.cache_config.mamba_cache_mode != "none"
        or not runner.speculative_config
        or not runner.model_config.is_hybrid
    ):
        return 0

    scheduled_spec = scheduler_output.scheduled_spec_decode_tokens
    transitions: list[tuple[int, str, int]] = []
    for row, req_id in enumerate(runner.input_batch.req_ids):
        accepted = int(runner.num_accepted_tokens.np[row])
        draft_len = len(scheduled_spec.get(req_id, ()))
        is_decode = (
            runner.input_batch.num_computed_tokens_cpu[row]
            >= runner.input_batch.num_prompt_tokens[row]
        )
        current_is_spec = draft_len > 0 and is_decode
        if accepted > 1 and not current_is_spec:
            transitions.append((row, req_id, accepted))

    if not transitions:
        return 0

    copy_funcs = runner.model.get_mamba_state_copy_func()
    mamba_group_ids, mamba_spec = get_mamba_groups(runner.kv_cache_config)
    num_spec = mamba_spec.num_speculative_blocks
    assert len(copy_funcs) == 2, "GDN re-anchor expects conv and temporal states"

    for group_id in mamba_group_ids:
        layer_names = runner.kv_cache_config.kv_cache_groups[group_id].layer_names
        dest_block_ids: list[int] = []
        source_block_ids: list[int] = []
        offsets: list[int] = []
        for _, req_id, accepted in transitions:
            block_ids = runner.requests[req_id].block_ids[group_id]
            assert (
                1 <= accepted <= len(block_ids)
            ), f"accepted count {accepted} exceeds block table for {req_id}"
            dest_block_ids.append(block_ids[0])
            source_block_ids.append(block_ids[accepted - 1])
            offsets.append(accepted - 1)

        first_attention = runner.compilation_config.static_forward_context[
            layer_names[0]
        ]
        first_conv_state = first_attention.kv_cache[0]
        history_len = first_conv_state.shape[1] - num_spec
        assert history_len > 0

        device = first_conv_state.device
        dest = torch.tensor(dest_block_ids, dtype=torch.long, device=device)
        source = torch.tensor(source_block_ids, dtype=torch.long, device=device)
        offset = torch.tensor(offsets, dtype=torch.long, device=device)
        history_rows = torch.arange(history_len, device=device)
        source_rows = offset[:, None] + history_rows[None, :]

        for layer_name in layer_names:
            attention = runner.compilation_config.static_forward_context[layer_name]
            conv_state, temporal_state = attention.kv_cache
            assert conv_state.shape[1] - num_spec == history_len

            conv_history = conv_state[dest[:, None], source_rows].clone()
            conv_state[dest[:, None], history_rows[None, :]] = conv_history

            temporal_candidates = temporal_state.index_select(0, source)
            temporal_state.index_copy_(0, dest, temporal_candidates)

    rows = [row for row, _, _ in transitions]
    runner.input_batch.num_accepted_tokens_cpu[rows] = 1
    runner.num_accepted_tokens.np[rows] = 1
    row_tensor = torch.tensor(rows, dtype=torch.long, device=runner.device)
    runner.num_accepted_tokens.gpu.index_fill_(0, row_tensor, 1)
    return len(transitions)


def patch_gpu_model_runner(module: Any) -> None:
    """Patch the upstream runner at the post-input-preparation boundary."""
    cls = module.GPUModelRunner
    original = cls._prepare_inputs
    if getattr(original, "_kunlun_spec_reanchor_patched", False):
        return

    def _prepare_inputs_with_reanchor(self, scheduler_output, num_scheduled_tokens):
        result = original(self, scheduler_output, num_scheduled_tokens)
        reanchor_spec_to_non_spec_states(self, scheduler_output)
        return result

    _prepare_inputs_with_reanchor._kunlun_spec_reanchor_patched = True
    cls._prepare_inputs = _prepare_inputs_with_reanchor
    logging.getLogger(__name__).info(
        "[KunlunPlugin] GPUModelRunner speculative state re-anchor patched"
    )


def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.
    """
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    # TODO(Chen): we need to optimize this function a lot
    assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    # We need to clear mamba_state_idx for resumed requests. When requests are
    # force-preempted (e.g., during reset_prefix_cache / KV cache flush),
    # they appear in resumed_req_ids without a corresponding entry in
    # preempted_req_ids, leaving stale mamba_state_idx entries that can
    # point to block indices beyond the new (smaller) block allocation.
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, resumed_req_ids):
        mamba_state_idx.pop(req_id, None)

    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            # new / resumed request, no previous state
            # if num_computed_tokens is 0, prev_state_idx will be -1
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks: int = (
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size)
            + num_speculative_blocks
        )

        # We always save the current running state at the last
        # (1 + num_speculative_blocks) block.
        # A corner case worth mention here: assume we have block_size = 4 and
        # num_speculative_tokens = 2. The request is [A, B, C] and contains 2 draft
        # tokens [draft 1, draft 2]. Then we will have:
        # Block 0: [A, B, C, draft 1]
        # Block 1: [draft 2, TOFILL, TOFILL, TOFILL]
        # Block 2: speculative block
        # Block 3: speculative block
        # And use block 1 to save the running state.
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                prev_state_idx,
                curr_state_idx,
                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state,
                forward_context,
            )
            input_batch.num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)


@dataclasses.dataclass
class MambaSpecDecodeGPUContext:
    """Kunlun device-side state-copy context for hybrid spec decoding.

    vLLM's CUDA implementation uses a Triton kernel to compute copy decisions
    and move Mamba states. Kunlun already provides ``batch_memcpy`` in
    xspeedgate, so this context computes the same addresses with device tensor
    operations and submits all copies through that operator. No accepted-token
    value is synchronized back to Python on the critical path.
    """

    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    state_elem_sizes: torch.Tensor
    state_inner_sizes: torch.Tensor
    state_conv_widths: torch.Tensor
    state_group_indices: torch.Tensor
    block_size: int
    max_num_reqs: int
    num_layers: int
    num_state_types: int
    mamba_group_ids: list[int]
    num_groups: int
    num_accepted_tokens_out: torch.Tensor
    block_table_ptrs: torch.Tensor
    block_table_stride_req: int = 0
    mamba_state_idx_buf: CpuGpuBuffer | None = None
    num_scheduled_tokens_buf: CpuGpuBuffer | None = None
    num_computed_tokens_buf: CpuGpuBuffer | None = None
    num_draft_tokens_buf: CpuGpuBuffer | None = None
    block_tables: list[torch.Tensor] = dataclasses.field(default_factory=list)
    state_group_indices_host: list[int] = dataclasses.field(default_factory=list)
    state_is_conv_host: list[bool] = dataclasses.field(default_factory=list)
    is_initialized: bool = False

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        num_state_types: int,
        device: torch.device,
        make_buffer: Callable[..., CpuGpuBuffer],
    ) -> "MambaSpecDecodeGPUContext":
        mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
        num_layers = sum(
            len(kv_cache_config.kv_cache_groups[gid].layer_names)
            for gid in mamba_group_ids
        )
        total_states = num_layers * num_state_types
        return cls(
            state_base_addrs=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            state_block_strides=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            state_elem_sizes=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            state_inner_sizes=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            state_conv_widths=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            state_group_indices=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            block_size=mamba_spec.block_size,
            max_num_reqs=max_num_reqs,
            num_layers=num_layers,
            num_state_types=num_state_types,
            mamba_group_ids=mamba_group_ids,
            num_groups=len(mamba_group_ids),
            num_accepted_tokens_out=torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            block_table_ptrs=torch.zeros(
                len(mamba_group_ids), dtype=torch.int64, device=device
            ),
            mamba_state_idx_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            num_scheduled_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            num_computed_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            num_draft_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
        )

    def initialize_from_forward_context(
        self,
        kv_cache_config: KVCacheConfig,
        forward_context: dict[str, Any],
        mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
        block_tables: list[torch.Tensor],
    ) -> None:
        """Bind persistent state tensors and block tables on first use."""
        if self.is_initialized:
            return
        if is_conv_state_dim_first():
            raise NotImplementedError(
                "Kunlun hybrid speculative decoding currently requires the "
                "default SD Mamba conv-state layout"
            )
        if len(block_tables) != self.num_groups:
            raise ValueError(
                f"expected {self.num_groups} Mamba block tables, "
                f"got {len(block_tables)}"
            )

        state_group_indices_host: list[int] = []
        state_is_conv_host: list[bool] = []
        idx = 0
        for group_local_idx, mamba_group_id in enumerate(self.mamba_group_ids):
            layer_names = kv_cache_config.kv_cache_groups[
                mamba_group_id
            ].layer_names
            for layer_name in layer_names:
                kv_caches: list[torch.Tensor] = forward_context[layer_name].kv_cache
                if len(kv_caches) != self.num_state_types:
                    raise ValueError(
                        f"Mamba layer {layer_name!r} exposes {len(kv_caches)} "
                        f"states, expected {self.num_state_types}"
                    )
                for state_type_idx, state in enumerate(kv_caches):
                    copy_func = mamba_state_copy_funcs[state_type_idx]
                    if copy_func not in (get_conv_copy_spec, get_temporal_copy_spec):
                        raise ValueError(f"unexpected Mamba copy func: {copy_func}")
                    is_conv = copy_func is get_conv_copy_spec
                    if is_conv and state.dim() != 3:
                        raise ValueError(
                            "Expected 3D conv state cache, got "
                            f"shape {tuple(state.shape)}"
                        )

                    elem_size = state.element_size()
                    block_stride = (
                        state.stride(0) if state.dim() > 1 else state.numel()
                    )
                    self.state_base_addrs[idx] = state.data_ptr()
                    self.state_block_strides[idx] = block_stride * elem_size
                    self.state_elem_sizes[idx] = elem_size
                    if is_conv:
                        self.state_conv_widths[idx] = state.size(1)
                        self.state_inner_sizes[idx] = state.stride(1)
                    else:
                        self.state_conv_widths[idx] = 0
                        self.state_inner_sizes[idx] = (
                            state[0].numel() if state.dim() > 1 else 1
                        )
                    self.state_group_indices[idx] = group_local_idx
                    state_group_indices_host.append(group_local_idx)
                    state_is_conv_host.append(is_conv)
                    idx += 1

        strides = {table.stride(0) for table in block_tables}
        if len(strides) != 1:
            raise ValueError(
                "all Mamba block tables must share stride(0), "
                f"got {strides}"
            )
        self.block_table_stride_req = int(next(iter(strides)))
        full_block_tables: list[torch.Tensor] = []
        for i, table in enumerate(block_tables):
            self.block_table_ptrs[i] = table.data_ptr()
            # get_device_tensor(num_reqs) returns a logical row slice of a
            # persistent max_num_reqs allocation. Keep a full-row view so a
            # later, larger batch does not retain the first batch's row bound.
            full_block_tables.append(
                table.as_strided(
                    (self.max_num_reqs, table.shape[1]),
                    table.stride(),
                    table.storage_offset(),
                )
            )
        self.block_tables = full_block_tables
        self.state_group_indices_host = state_group_indices_host
        self.state_is_conv_host = state_is_conv_host
        self.is_initialized = True

    def run_fused_postprocess(
        self,
        num_reqs: int,
        num_accepted_tokens_gpu: torch.Tensor,
        mamba_state_idx_gpu: torch.Tensor,
        num_scheduled_tokens_gpu: torch.Tensor,
        num_computed_tokens_gpu: torch.Tensor,
        num_draft_tokens_gpu: torch.Tensor,
    ) -> None:
        """Compute state-copy addresses on XPU and submit one batched copy."""
        if num_reqs == 0 or not self.is_initialized:
            return
        if num_reqs > self.max_num_reqs:
            raise ValueError(
                f"num_reqs {num_reqs} exceeds maximum {self.max_num_reqs}"
            )

        accepted = num_accepted_tokens_gpu[:num_reqs].to(torch.int64)
        src_col = mamba_state_idx_gpu[:num_reqs].to(torch.int64)
        running_tokens = (
            num_computed_tokens_gpu[:num_reqs].to(torch.int64)
            + num_scheduled_tokens_gpu[:num_reqs].to(torch.int64)
            - num_draft_tokens_gpu[:num_reqs].to(torch.int64)
        )
        new_computed = running_tokens + accepted - 1
        aligned_computed = new_computed // self.block_size * self.block_size
        needs_copy = aligned_computed >= running_tokens
        token_bias = aligned_computed - running_tokens
        dest_col = aligned_computed // self.block_size - 1
        same_col = src_col == dest_col

        accepted_out = torch.where(
            needs_copy & same_col, torch.ones_like(accepted), accepted
        )
        self.num_accepted_tokens_out[:num_reqs].copy_(
            accepted_out.to(torch.int32)
        )
        copy_mask = needs_copy & ~(same_col & (token_bias == 0))

        rows = torch.arange(
            num_reqs, dtype=torch.long, device=num_accepted_tokens_gpu.device
        )
        group_block_ids: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for table in self.block_tables:
            last_col = table.shape[1] - 1
            dest_ids = table[rows, dest_col.clamp(0, last_col)].to(torch.int64)
            conv_src_ids = table[rows, src_col.clamp(0, last_col)].to(torch.int64)
            temporal_src_ids = table[
                rows, (src_col + token_bias).clamp(0, last_col)
            ].to(torch.int64)
            group_block_ids.append((dest_ids, conv_src_ids, temporal_src_ids))

        src_ptrs: list[torch.Tensor] = []
        dst_ptrs: list[torch.Tensor] = []
        copy_sizes: list[torch.Tensor] = []
        total_states = self.num_layers * self.num_state_types
        for state_idx in range(total_states):
            group_idx = self.state_group_indices_host[state_idx]
            dest_ids, conv_src_ids, temporal_src_ids = group_block_ids[group_idx]
            base_addr = self.state_base_addrs[state_idx]
            block_stride = self.state_block_strides[state_idx]
            elem_size = self.state_elem_sizes[state_idx]
            inner_size = self.state_inner_sizes[state_idx]
            dst_ptr = base_addr + dest_ids * block_stride

            if self.state_is_conv_host[state_idx]:
                src_ptr = (
                    base_addr
                    + conv_src_ids * block_stride
                    + token_bias * inner_size * elem_size
                )
                size = (
                    (self.state_conv_widths[state_idx] - token_bias)
                    .clamp_min(0)
                    * inner_size
                    * elem_size
                )
            else:
                src_ptr = base_addr + temporal_src_ids * block_stride
                size = torch.ones_like(token_bias) * inner_size * elem_size

            src_ptrs.append(src_ptr)
            dst_ptrs.append(dst_ptr)
            copy_sizes.append(torch.where(copy_mask, size, torch.zeros_like(size)))

        # [request, state] order is convenient for inspecting failures, but the
        # batch memcpy operator only requires the three flattened arrays to
        # have matching order.
        batch_memcpy(
            torch.stack(src_ptrs, dim=1).reshape(-1),
            torch.stack(dst_ptrs, dim=1).reshape(-1),
            torch.stack(copy_sizes, dim=1).reshape(-1),
        )


@dataclasses.dataclass
class MambaBuffers:
    """Single owner for all mamba-specific runner buffers."""

    preprocess: MambaCopyBuffers
    postprocess_align: MambaSpecDecodeGPUContext | None

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: tuple[MambaStateCopyFunc, ...],
        make_buffer: Callable[..., CpuGpuBuffer],
        device: torch.device,
        with_postprocess_align: bool = False,
    ) -> "MambaBuffers":
        return cls(
            preprocess=MambaCopyBuffers.create(
                max_num_reqs, kv_cache_config, copy_funcs, make_buffer
            ),
            postprocess_align=(
                MambaSpecDecodeGPUContext.create(
                    max_num_reqs=max_num_reqs,
                    kv_cache_config=kv_cache_config,
                    num_state_types=len(copy_funcs),
                    device=device,
                    make_buffer=make_buffer,
                )
                if with_postprocess_align
                else None
            ),
        )


def get_hybrid_attention_mamba_layout(
    kv_cache_shape: tuple[int, ...],
    kv_cache_stride: tuple[int, ...],
    kv_cache_spec: AttentionSpec,
    block_dim: int,
    layer_idx: int,
    kernel_block_size: int,
) -> tuple[tuple[int, ...], int]:
    """
    Compute the stride and storage offset for the hybrid attention+mamba layout.

    Args:
        kv_cache_shape: The shape of the KV cache tensor.
        kv_cache_stride: The stride of the KV cache tensor.
        kv_cache_spec: The specification of the KV cache.
        layer_idx: The index of the layer.
        kernel_num_blocks: The number of kernel blocks.
        kernel_block_size: The size of the kernel block.
    Returns:
        A tuple containing the target stride and storage offset.
    """
    target_stride_list = list(kv_cache_stride)
    storage_offset = 0

    attn_pack_size = kv_cache_spec.pack_size
    # block_dim: 0 means (num_blocks, 2, ...); 1 means (2, num_blocks, ...).
    if block_dim != 0:
        # Hybrid attention+mamba uses (2, num_blocks, ...) logical shape but
        # (num_blocks, 2, ...) physical layout.
        assert kv_cache_shape[0] == 2, (
            "Fail to determine whether the layout is "
            "(2, num_blocks, ...) or (num_blocks, 2, ...) for "
            f"a tensor of shape {kv_cache_shape}"
        )
        assert block_dim == 1
        hidden_size = prod(kv_cache_shape[2:])
        target_stride_list[0] = hidden_size
        target_stride_list[1] = 2 * hidden_size
    # When multiple attention layers share one physical KV cache block
    # (attn_pack_size > 1), scale the block-dim stride by attn_pack_size
    # and compute this layer's element offset within the shared block.
    if attn_pack_size > 1:
        target_stride_list[block_dim] *= attn_pack_size
        dtype_size = get_dtype_size(kv_cache_spec.dtype)
        num_element_per_page = kv_cache_spec.page_size_bytes // dtype_size
        num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
        num_element_per_attn_pack = (
            num_element_per_page // num_blocks_per_kv_block // attn_pack_size
        )
        attn_pack_idx = layer_idx % attn_pack_size
        storage_offset = attn_pack_idx * num_element_per_attn_pack
    return tuple(target_stride_list), storage_offset


def postprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
):
    """
    If a blocks is converted from partial block to full block in this step, copy the
    state from the block for running state to the new full block.
    """
    num_scheduled_tokens_dict = scheduler_output.num_scheduled_tokens
    scheduled_spec_decode_tokens_dict = scheduler_output.scheduled_spec_decode_tokens
    num_accepted_tokens_cpu = input_batch.num_accepted_tokens_cpu
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        num_computed_tokens = req_state.num_computed_tokens
        num_draft_tokens = len(scheduled_spec_decode_tokens_dict.get(req_id, []))
        num_scheduled_tokens = num_scheduled_tokens_dict[req_id]
        num_accepted_tokens = num_accepted_tokens_cpu[i]
        num_tokens_running_state = (
            num_computed_tokens + num_scheduled_tokens - num_draft_tokens
        )
        new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens - 1
        aligned_new_computed_tokens = (
            new_num_computed_tokens // mamba_spec.block_size * mamba_spec.block_size
        )
        # TODO: how to ensure all blocks that cache_blocks called are cached here?
        if aligned_new_computed_tokens >= num_tokens_running_state:
            accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
            src_block_idx = mamba_state_idx[req_id]
            dest_block_idx = aligned_new_computed_tokens // mamba_spec.block_size - 1
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                src_block_idx,
                dest_block_idx,
                accept_token_bias,
                req_state,
                forward_context,
            )
            if src_block_idx == dest_block_idx:
                num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)


def cleanup_mamba_state_idx(
    scheduler_output: SchedulerOutput,
    mamba_state_idx: dict[str, int],
) -> None:
    """Drop state-column entries invalidated by scheduler lifecycle events."""
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(
        scheduler_output.finished_req_ids,
        scheduler_output.preempted_req_ids or set(),
        resumed_req_ids,
    ):
        mamba_state_idx.pop(req_id, None)


def postprocess_mamba_all(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
    num_spec_tokens: int,
    num_reqs: int,
) -> None:
    """Track the last speculative state column for ``all`` cache mode."""
    if num_spec_tokens <= 0:
        return
    _, mamba_spec = get_mamba_groups(kv_cache_config)
    full_decode_len = 1 + num_spec_tokens
    for req_id in input_batch.req_ids[:num_reqs]:
        num_query = scheduler_output.num_scheduled_tokens.get(req_id, 0)
        if num_query == full_decode_len:
            seq_len = requests[req_id].num_computed_tokens + num_query
            mamba_state_idx[req_id] = max(
                0, (seq_len - 1) // mamba_spec.block_size
            )
        else:
            mamba_state_idx.pop(req_id, None)


def preprocess_mamba_all_specdec(
    scheduler_output: SchedulerOutput,
    input_batch: GPUInputBatch,
    mamba_state_idx: dict[str, int],
    num_reqs: int,
    prev_last_scheduled_idx_buf: CpuGpuBuffer,
) -> None:
    """Stage the previous speculative state columns for ``all`` mode."""
    cleanup_mamba_state_idx(scheduler_output, mamba_state_idx)
    np_view = prev_last_scheduled_idx_buf.np
    for i, req_id in enumerate(input_batch.req_ids[:num_reqs]):
        np_view[i] = mamba_state_idx.get(req_id, -1)
    np_view[num_reqs:].fill(-1)
    prev_last_scheduled_idx_buf.copy_to_gpu()


def postprocess_mamba_align_gpu(
    *,
    bufs: MambaBuffers,
    num_reqs: int,
    num_accepted_tokens_gpu: torch.Tensor,
    num_accepted_tokens_cpu_tensor: torch.Tensor,
    input_batch: GPUInputBatch,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
) -> None:
    """Run Kunlun's device-side align-mode speculative postprocess."""
    ctx = bufs.postprocess_align
    assert ctx is not None
    assert ctx.mamba_state_idx_buf is not None
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None

    if not ctx.is_initialized:
        ctx.initialize_from_forward_context(
            kv_cache_config,
            forward_context,
            mamba_state_copy_funcs,
            [
                input_batch.block_table[gid].get_device_tensor(num_reqs)
                for gid in ctx.mamba_group_ids
            ],
        )
    ctx.run_fused_postprocess(
        num_reqs=num_reqs,
        num_accepted_tokens_gpu=num_accepted_tokens_gpu,
        mamba_state_idx_gpu=ctx.mamba_state_idx_buf.gpu,
        num_scheduled_tokens_gpu=ctx.num_scheduled_tokens_buf.gpu,
        num_computed_tokens_gpu=ctx.num_computed_tokens_buf.gpu,
        num_draft_tokens_gpu=ctx.num_draft_tokens_buf.gpu,
    )
    num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
        ctx.num_accepted_tokens_out[:num_reqs], non_blocking=True
    )


def stage_postprocess_metadata_to_gpu(
    scheduler_output: SchedulerOutput,
    req_ids: list[str],
    num_reqs: int,
    requests: dict[str, CachedRequestState],
    num_scheduled_tokens_buf: CpuGpuBuffer,
    num_computed_tokens_buf: CpuGpuBuffer,
    num_draft_tokens_buf: CpuGpuBuffer,
) -> None:
    """Stage per-request align postprocess inputs without a device sync."""
    scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    num_scheduled = scheduler_output.num_scheduled_tokens
    scheduled_np = num_scheduled_tokens_buf.np
    computed_np = num_computed_tokens_buf.np
    draft_np = num_draft_tokens_buf.np
    for i in range(num_reqs):
        req_id = req_ids[i]
        scheduled_np[i] = num_scheduled[req_id]
        computed_np[i] = requests[req_id].num_computed_tokens
        draft_np[i] = len(scheduled_spec_tokens.get(req_id, ()))
    num_scheduled_tokens_buf.copy_to_gpu(num_reqs)
    num_computed_tokens_buf.copy_to_gpu(num_reqs)
    num_draft_tokens_buf.copy_to_gpu(num_reqs)


def stage_mamba_state_idx_to_gpu(
    mamba_state_idx: dict[str, int],
    req_ids: list[str],
    num_reqs: int,
    gpu_buf: CpuGpuBuffer,
) -> None:
    """Materialize the preprocess result in batch order on the device."""
    np_view = gpu_buf.np
    for i in range(num_reqs):
        req_id = req_ids[i]
        state_idx = mamba_state_idx.get(req_id)
        assert state_idx is not None, (
            f"mamba_state_idx missing entry for {req_id!r}; "
            "preprocess_mamba must run before postprocess staging"
        )
        np_view[i] = state_idx
    gpu_buf.copy_to_gpu(num_reqs)


def stage_postprocess_inputs_to_gpu(
    ctx: MambaSpecDecodeGPUContext,
    scheduler_output: SchedulerOutput,
    req_ids: list[str],
    num_reqs: int,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
) -> None:
    """Stage every input consumed by ``run_fused_postprocess``."""
    assert ctx.mamba_state_idx_buf is not None
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None
    stage_mamba_state_idx_to_gpu(
        mamba_state_idx, req_ids, num_reqs, ctx.mamba_state_idx_buf
    )
    stage_postprocess_metadata_to_gpu(
        scheduler_output,
        req_ids,
        num_reqs,
        requests,
        ctx.num_scheduled_tokens_buf,
        ctx.num_computed_tokens_buf,
        ctx.num_draft_tokens_buf,
    )
