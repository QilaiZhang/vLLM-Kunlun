# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun Mamba copies and speculative state management.

Patch upstream in place to retain its export surface and lifecycle helpers.
Copies use xspeedgate with int64 buffers; speculative postprocessing computes
copy addresses with device tensor operations instead of Triton.
"""

import dataclasses
import logging
from collections.abc import Callable
from typing import Any

import torch
import vllm.v1.worker.mamba_utils as _up
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    get_conv_copy_spec,
    get_temporal_copy_spec,
    is_conv_state_dim_first,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.utils import CpuGpuBuffer

logger = logging.getLogger("vllm_kunlun")


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    """xspeedgate stand-in for upstream's Triton ``batch_memcpy_kernel``.

    ``xspeedgate_ops.batch_memcpy`` is specified for int64 pointer and size
    tensors. Both the ``MambaCopyBuffers.create`` override below and the
    speculative postprocess context construct buffers with those dtypes.

    The dtypes are therefore asserted rather than coerced. An earlier version
    reinterpreted mismatches with ``Tensor.view``, which is only lossless
    between same-itemsize dtypes -- an int32 buffer would have been silently
    re-read as half as many int64 values. Nothing produces such a buffer today,
    so a mismatch means the call path changed and should fail loudly.
    """
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch
    if batch == 0:
        return
    for name, tensor in (
        ("src_ptrs", src_ptrs),
        ("dst_ptrs", dst_ptrs),
        ("sizes", sizes),
    ):
        assert tensor.dtype is torch.int64, (
            f"xspeedgate_ops.batch_memcpy expects int64 {name}, got "
            f"{tensor.dtype}; buffers should come from the Kunlun "
            f"MambaCopyBuffers.create override in {__name__}"
        )
    torch.ops.xspeedgate_ops.batch_memcpy(src_ptrs, dst_ptrs, sizes)


def _mamba_copy_buffers_create(
    cls,
    max_num_reqs,
    kv_cache_config,
    copy_funcs,
    make_buffer,
):
    """Same as upstream ``MambaCopyBuffers.create`` but with int64 buffers.

    Upstream allocates ``uint64`` pointers and ``int32`` sizes
    (mamba_utils.py:449-451); the xspeedgate op is specified for int64.

    Note the ``v0.15.0-dev`` branch keeps ``int32`` sizes here (#351), so the two
    release branches disagree on what ``xspeedgate_ops.batch_memcpy`` wants. The
    int64 choice predates this change -- it is what the v0.25.1 branch has
    shipped since #392 -- and is kept as-is; the assertion in ``batch_memcpy``
    above turns any future mismatch into a loud failure rather than a silent
    reinterpret.
    """
    mamba_group_ids, mamba_spec = _up.get_mamba_groups(kv_cache_config)
    entries_per_req = sum(
        len(kv_cache_config.kv_cache_groups[gid].layer_names) for gid in mamba_group_ids
    ) * len(copy_funcs)
    n = max_num_reqs * entries_per_req
    return cls(
        src_ptrs=make_buffer(n, dtype=torch.int64),
        dst_ptrs=make_buffer(n, dtype=torch.int64),
        sizes=make_buffer(n, dtype=torch.int64),
        mamba_group_ids=mamba_group_ids,
        mamba_spec=mamba_spec,
    )


# ``do_mamba_copy_block`` and friends resolve ``batch_memcpy`` from the upstream
# module globals, and gpu_model_runner.py:202 imports the module object rather
# than the name, so both call sites pick these up at call time.
_up.batch_memcpy = batch_memcpy
_up.MambaCopyBuffers.create = classmethod(_mamba_copy_buffers_create)
logger.info(
    "[KunlunPlugin] mamba_utils patched (xspeedgate batch_memcpy, int64 buffers)"
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
    mamba_group_ids, mamba_spec = _up.get_mamba_groups(runner.kv_cache_config)
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
        mamba_group_ids, mamba_spec = _up.get_mamba_groups(kv_cache_config)
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
            layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
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
                    block_stride = state.stride(0) if state.dim() > 1 else state.numel()
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
                "all Mamba block tables must share stride(0), " f"got {strides}"
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
            raise ValueError(f"num_reqs {num_reqs} exceeds maximum {self.max_num_reqs}")

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
        self.num_accepted_tokens_out[:num_reqs].copy_(accepted_out.to(torch.int32))
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
                    (self.state_conv_widths[state_idx] - token_bias).clamp_min(0)
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


# Upstream MambaBuffers.create resolves this class from its module globals.
_up.MambaSpecDecodeGPUContext = MambaSpecDecodeGPUContext
