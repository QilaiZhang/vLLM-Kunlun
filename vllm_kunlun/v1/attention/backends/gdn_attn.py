# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

import os
from dataclasses import dataclass, replace
from weakref import WeakValueDictionary

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends import gdn_attn as _upstream_gdn_attn
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
from vllm.v1.utils import CpuGpuBuffer

logger = init_logger(__name__)


def _to_cpu(t: torch.Tensor | None) -> torch.Tensor | None:
    """Force a metadata mirror onto the host, no-op if it is already there.

    Several ``*_cpu`` locals in ``build()`` are produced by indexing the *device*
    block table, so they are not actually on the host despite the name. Their
    consumers pass them to kunlun ops as host pointers.
    """
    if t is None or t.device.type == "cpu":
        return t
    return t.to("cpu")


def _cpu_state_block_table(
    block_table: torch.Tensor,
    cache_mode: str,
) -> torch.Tensor | None:
    """Get vLLM's persistent host block-table mirror when it is exact.

    ``none`` and ``all`` preserve the device block-table layout on the host.
    ``align`` gathers columns using exact device sequence lengths, so it keeps
    the existing device-to-host fallback until an exact CPU index is available.
    """
    if cache_mode not in ("none", "all"):
        return None
    from vllm_kunlun.v1.worker.block_table import get_cpu_block_table

    return get_cpu_block_table(block_table)


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None
    # [Kunlun] CPU mirror — consumed by causal_conv1d_fn(has_initial_state_cpu=...)
    has_initial_state_cpu: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )
    # [Kunlun] CPU mirror — consumed by causal_conv1d_fn(query_start_loc_cpu=...)
    non_spec_query_start_loc_cpu: torch.Tensor | None = None

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    spec_state_indices_tensor_cpu: torch.Tensor | None = (
        None  # shape: [batch, num_spec]
    )
    spec_conv_state_indices_tensor: torch.Tensor | None = None  # shape: [batch,]
    spec_conv_state_indices_tensor_cpu: torch.Tensor | None = None  # shape: [batch,]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    # [Kunlun] CPU mirror — consumed as cache_indices_cpu / conv_state_indices_cpu
    non_spec_state_indices_tensor_cpu: torch.Tensor | None = None

    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_sequence_masks_cpu: torch.Tensor | None = None  # shape: [batch,]
    # [Kunlun] bool mask used by qwen3_next.py to split spec / non-spec tokens
    # via mixed_qkv[spec_token_masks] / g[:, spec_token_masks] etc.
    spec_token_masks: torch.Tensor | None = None
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]
    num_accepted_tokens_cpu: torch.Tensor | None = None  # shape: [batch,]

    # Variable-length speculative decode uses a padded layout only for the
    # fixed-width Kunlun conv kernel, then restores the real token layout before
    # the recurrent kernel. Both are None for uniform MTP batches.
    spec_pad_gather_idx: torch.Tensor | None = None
    spec_unpad_idx: torch.Tensor | None = None

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


def build_spec_pad_indices(
    spec_query_lens_cpu: torch.Tensor,
    spec_width: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Map variable-length spec tokens to fixed-width rows and back."""
    num_spec_decodes = spec_query_lens_cpu.size(0)
    lens = spec_query_lens_cpu.to(torch.int64)
    if int(lens.sum().item()) == num_spec_decodes * spec_width:
        return None

    cu = torch.zeros(num_spec_decodes + 1, dtype=torch.int64)
    torch.cumsum(lens, dim=0, out=cu[1:])
    starts = cu[:-1].unsqueeze(1)
    pos = torch.arange(spec_width, dtype=torch.int64).unsqueeze(0)
    last_real = (lens - 1).clamp_(min=0).unsqueeze(1)

    # Replicate the final real token so the conv input stays finite. These tail
    # outputs are removed before the recurrent kernel and cannot be accepted.
    pad_gather_idx = (starts + torch.minimum(pos, last_real)).reshape(-1)
    real_mask = pos < lens.unsqueeze(1)
    unpad_idx = (
        torch.arange(num_spec_decodes, dtype=torch.int64).unsqueeze(1) * spec_width
        + pos
    )[real_mask]
    return pad_gather_idx.to(torch.int32), unpad_idx.to(torch.int32)


class _GDNCommonGraphBuffers:
    """Batch-level FULL-graph buffers shared by equivalent GDN builders.

    During capture vLLM 0.25.1 builds every attention group independently,
    while metadata reuse later keeps the first group's batch-level metadata.
    Sharing these buffers makes both paths use the same addresses. State-index
    buffers intentionally stay on each builder because they vary by KV group.
    """

    def __init__(
        self,
        metadata_buffer_size: int,
        decode_cudagraph_max_bs: int,
        num_spec: int,
        device: torch.device,
    ) -> None:
        self.spec_sequence_masks = CpuGpuBuffer(
            metadata_buffer_size,
            dtype=torch.bool,
            device=device,
            with_numpy=False,
        )
        self.non_spec_sequence_masks = CpuGpuBuffer(
            metadata_buffer_size,
            dtype=torch.bool,
            device=device,
            with_numpy=False,
        )
        token_capacity = decode_cudagraph_max_bs * (num_spec + 1)
        self.spec_token_masks = torch.empty(
            (token_capacity,), dtype=torch.bool, device=device
        )
        self.spec_token_indx = torch.empty(
            (token_capacity,), dtype=torch.int32, device=device
        )
        self.non_spec_token_indx = torch.empty(
            (token_capacity,), dtype=torch.int32, device=device
        )
        self.spec_query_start_loc = torch.empty(
            (decode_cudagraph_max_bs + 1,), dtype=torch.int32, device=device
        )
        self.non_spec_query_start_loc = torch.empty(
            (decode_cudagraph_max_bs + 1,), dtype=torch.int32, device=device
        )
        self.num_accepted_tokens = torch.empty(
            (decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )


_COMMON_GRAPH_BUFFERS: WeakValueDictionary[tuple, _GDNCommonGraphBuffers] = (
    WeakValueDictionary()
)


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    # Instance initialization enables this only for the graph-safe subset.
    supports_update_block_table = False

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec
        self.device = device
        # [Kunlun] Track which capture sizes have had their spec-decode
        # fused_recurrent kernel eagerly warmed up (see
        # _maybe_warmup_spec_kernel).
        self._spec_kernel_warmed: set[int] = set()

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        reuse_disabled = os.getenv(
            "VLLM_KUNLUN_DISABLE_GDN_METADATA_REUSE", "0"
        ).lower() in ("1", "true", "on")
        self.supports_update_block_table = (
            type(self) is GDNAttentionMetadataBuilder
            and self.vllm_config.cache_config.mamba_cache_mode in ("none", "all")
            and not self.vllm_config.parallel_config.use_ubatching
            and not reuse_disabled
        )

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )
        metadata_buffer_size = max(
            self.decode_cudagraph_max_bs,
            self.vllm_config.scheduler_config.max_num_seqs,
        )

        common_key = (
            id(vllm_config),
            # v0.25.1 keeps one persistent-buffer builder per microbatch.
            # GDN speculative decode rejects u-batching, and non-spec GDN must
            # not make different microbatches alias the same graph buffers.
            # The reuse opt-out must also restore per-builder graph buffers;
            # otherwise it does not provide a true Stage-1 correctness
            # baseline and can hide bugs in the Stage-2 address-sharing path.
            (
                id(self)
                if vllm_config.parallel_config.use_ubatching or reuse_disabled
                else None
            ),
            type(self),
            kv_cache_spec,
            device,
            self.num_spec,
            self.decode_cudagraph_max_bs,
            metadata_buffer_size,
        )
        common_buffers = _COMMON_GRAPH_BUFFERS.get(common_key)
        if common_buffers is None:
            common_buffers = _GDNCommonGraphBuffers(
                metadata_buffer_size,
                self.decode_cudagraph_max_bs,
                self.num_spec,
                device,
            )
            _COMMON_GRAPH_BUFFERS[common_key] = common_buffers
        # A strong reference keeps the weak-registry entry alive for exactly
        # the engine/builders that use it and prevents stale engine-id reuse.
        self._common_graph_buffers = common_buffers

        self._spec_state_indices = CpuGpuBuffer(
            metadata_buffer_size,
            self.num_spec + 1,
            dtype=torch.int32,
            device=device,
            with_numpy=False,
        )
        self.spec_state_indices_tensor = self._spec_state_indices.gpu
        self.spec_state_indices_tensor_cpu = self._spec_state_indices.cpu
        self._spec_conv_state_indices = CpuGpuBuffer(
            metadata_buffer_size,
            dtype=torch.int32,
            device=device,
            with_numpy=False,
        )
        self.spec_conv_state_indices_tensor = self._spec_conv_state_indices.gpu
        self.spec_conv_state_indices_tensor_cpu = self._spec_conv_state_indices.cpu
        self._non_spec_state_indices = CpuGpuBuffer(
            metadata_buffer_size,
            dtype=torch.int32,
            device=device,
            with_numpy=False,
        )
        self.non_spec_state_indices_tensor = self._non_spec_state_indices.gpu
        self.non_spec_state_indices_tensor_cpu = self._non_spec_state_indices.cpu
        self._spec_sequence_masks = common_buffers.spec_sequence_masks
        self.spec_sequence_masks = self._spec_sequence_masks.gpu
        self.spec_sequence_masks_cpu = self._spec_sequence_masks.cpu
        self._non_spec_sequence_masks = common_buffers.non_spec_sequence_masks
        self.non_spec_sequence_masks_cpu = self._non_spec_sequence_masks.cpu
        self._has_initial_state = CpuGpuBuffer(
            metadata_buffer_size,
            dtype=torch.int32,
            device=device,
            with_numpy=False,
        )
        self.has_initial_state = self._has_initial_state.gpu
        self.has_initial_state_cpu = self._has_initial_state.cpu
        # Lazily grown transfer buffers cover prefill chunk metadata and the
        # uncommon variable-width speculative fallback. Each host allocation
        # is persistent and pinned by CpuGpuBuffer, so steady-state build does
        # not call Tensor.pin_memory() or allocate a new device tensor.
        self._transfer_buffers: dict[str, CpuGpuBuffer] = {}
        self.spec_token_indx = common_buffers.spec_token_indx
        self.non_spec_token_indx = common_buffers.non_spec_token_indx
        self.spec_token_masks = common_buffers.spec_token_masks
        self.spec_query_start_loc = common_buffers.spec_query_start_loc
        self.non_spec_query_start_loc = common_buffers.non_spec_query_start_loc
        self.num_accepted_tokens = common_buffers.num_accepted_tokens

    def _copy_cpu_to_device(self, name: str, source: torch.Tensor) -> torch.Tensor:
        """Copy a CPU tensor through a persistent pinned host/device pair."""
        assert source.device.type == "cpu"
        required = source.numel()
        buffer = self._transfer_buffers.get(name)
        if (
            buffer is None
            or buffer.cpu.numel() < required
            or buffer.cpu.dtype != source.dtype
        ):
            capacity = max(1, 1 << max(0, required - 1).bit_length())
            buffer = CpuGpuBuffer(
                capacity,
                dtype=source.dtype,
                device=self.device,
                with_numpy=False,
            )
            self._transfer_buffers[name] = buffer
        buffer.cpu[:required].copy_(source.reshape(-1))
        buffer.copy_to_gpu(required)
        return buffer.gpu[:required].view(source.shape)

    def _stage_spec_decode(
        self,
        num_prefills: int,
        num_decodes: int,
        num_spec_decodes: int,
        num_spec_decode_tokens: int,
    ) -> bool:
        """Whether spec metadata must use stable FULL-graph buffers."""
        return (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        )

    def _stage_decode(
        self, num_prefills: int, num_decodes: int, num_spec_decodes: int
    ) -> bool:
        """Whether decode metadata must use stable FULL-graph buffers."""
        return (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        context_lens_tensor = m.compute_num_computed_tokens()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        cache_mode = self.vllm_config.cache_config.mamba_cache_mode
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            cache_mode,
        )
        cpu_block_table = _cpu_state_block_table(m.block_table_tensor, cache_mode)
        spec_conv_state_indices_tensor: torch.Tensor | None = None
        spec_conv_state_indices_tensor_cpu: torch.Tensor | None = None
        spec_sequence_masks_cpu: torch.Tensor | None = None
        spec_pad_gather_idx: torch.Tensor | None = None
        spec_unpad_idx: torch.Tensor | None = None
        # [Kunlun] These CPU mirrors are only produced on some branches below but
        # are always read when the metadata is built, so default them here.
        spec_state_indices_tensor_cpu: torch.Tensor | None = None
        non_spec_state_indices_tensor_cpu: torch.Tensor | None = None
        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            num_mask_rows = num_decode_draft_tokens_cpu.numel()
            spec_sequence_masks_cpu = self.spec_sequence_masks_cpu[:num_mask_rows]
            torch.ge(
                num_decode_draft_tokens_cpu,
                0,
                out=spec_sequence_masks_cpu,
            )
            num_spec_decodes = int(spec_sequence_masks_cpu.sum().item())
            if (
                num_spec_decodes == 0
                or int(
                    num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum().item()
                )
                == 0
            ):
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                self._spec_sequence_masks.copy_to_gpu(num_mask_rows)
                spec_sequence_masks = self.spec_sequence_masks[:num_mask_rows]

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            spec_token_masks = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            spec_conv_state_indices_tensor = None
            # ``[:, 0]`` is a strided column view (typically stride 4 in MTP
            # align mode). Kunlun causal-conv consumes this tensor as a raw
            # contiguous int32 array, so materialize it once here. Without
            # this, logical indices [45, 61] are read as physical neighbors
            # [45, 46] by the batched state-scatter path.
            non_spec_state_indices_tensor = block_table_tensor[:, 0].contiguous()
            if cpu_block_table is not None:
                num_state_rows = non_spec_state_indices_tensor.size(0)
                non_spec_state_indices_tensor_cpu = cpu_block_table[
                    :num_state_rows, 0
                ].contiguous()
                padded_rows = query_start_loc_cpu[1:] == query_start_loc_cpu[:-1]
                non_spec_state_indices_tensor_cpu.masked_fill_(
                    padded_rows,
                    NULL_BLOCK_ID,
                )
            else:
                non_spec_state_indices_tensor_cpu = non_spec_state_indices_tensor
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            non_spec_sequence_masks_cpu = self.non_spec_sequence_masks_cpu[
                : spec_sequence_masks_cpu.numel()
            ]
            torch.logical_not(
                spec_sequence_masks_cpu,
                out=non_spec_sequence_masks_cpu,
            )
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            # Use CPU tensors to avoid CPU-GPU sync
            non_spec_query_lens_cpu = query_lens_cpu[non_spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            # Exclude zero-length padded sequences from prefill count.
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            # num_decodes and num_spec_decodes are mutually exclusive.
            # Reclassify non-spec decodes as prefills when spec decodes
            # exist — the prefill kernel handles 1-token sequences with
            # initial state correctly, producing identical results.
            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                # [Kunlun] all tokens are spec tokens in this branch
                spec_token_masks = torch.ones(
                    spec_token_size,
                    dtype=torch.bool,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Padded rows trail the real speculative decodes, so slicing
                # avoids the host-mask gather (implicit H2D + device kernel).
                spec_state_indices_tensor = block_table_tensor[
                    :num_spec_decodes, : self.num_spec + 1
                ]
                if cpu_block_table is not None:
                    spec_state_indices_tensor_cpu = cpu_block_table[
                        :num_spec_decodes, : self.num_spec + 1
                    ].contiguous()
                else:
                    spec_state_indices_tensor_cpu = spec_state_indices_tensor
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                if cpu_block_table is not None:
                    spec_state_indices_tensor_cpu = cpu_block_table[
                        spec_sequence_masks_cpu, : self.num_spec + 1
                    ].contiguous()
                else:
                    spec_state_indices_tensor_cpu = spec_state_indices_tensor
                non_spec_state_indices_tensor = block_table_tensor[
                    non_spec_sequence_masks_cpu, 0
                ].contiguous()
                if cpu_block_table is not None:
                    non_spec_state_indices_tensor_cpu = cpu_block_table[
                        non_spec_sequence_masks_cpu, 0
                    ].contiguous()
                    non_spec_state_indices_tensor_cpu.masked_fill_(
                        non_spec_query_lens_cpu == 0,
                        NULL_BLOCK_ID,
                    )
                else:
                    non_spec_state_indices_tensor_cpu = non_spec_state_indices_tensor

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks_cpu],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            if num_prefills == 0 and num_decodes == 0:
                num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]
            else:
                num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

            pad_indices = build_spec_pad_indices(
                query_lens_cpu[spec_sequence_masks_cpu], self.num_spec + 1
            )
            if pad_indices is not None:
                logger.warning_once(
                    "[KunlunPlugin] GDN spec decode hit non-uniform query "
                    "lengths; using the padded-conv fallback"
                )
                spec_pad_gather_idx = self._copy_cpu_to_device(
                    "spec_pad_gather_idx", pad_indices[0]
                )
                spec_unpad_idx = self._copy_cpu_to_device(
                    "spec_unpad_idx", pad_indices[1]
                )

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        if num_prefills > 0:
            # Only prefill batches use FLA chunk ops.
            # Pre-compute on CPU and async-copy to GPU to avoid
            # GPU→CPU sync (.tolist()) in prepare_chunk_indices.
            from vllm.model_executor.layers.fla.ops.index import (
                prepare_chunk_indices,
                prepare_chunk_offsets,
            )
            from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

            chunk_indices = self._copy_cpu_to_device(
                "chunk_indices",
                prepare_chunk_indices(non_spec_query_start_loc_cpu, FLA_CHUNK_SIZE),
            )
            chunk_offsets = self._copy_cpu_to_device(
                "chunk_offsets",
                prepare_chunk_offsets(non_spec_query_start_loc_cpu, FLA_CHUNK_SIZE),
            )

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[non_spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            has_initial_state_cpu = has_initial_state.cpu()
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
        else:
            has_initial_state = None
            has_initial_state_cpu = None

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (
            num_decodes > 0 and num_spec_decodes > 0
        ), f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        batch_size = m.num_actual_tokens

        if spec_pad_gather_idx is None and self._stage_spec_decode(
            num_prefills,
            num_decodes,
            num_spec_decodes,
            num_spec_decode_tokens,
        ):
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            self.spec_conv_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor[:, 0], non_blocking=True
            )
            self.spec_conv_state_indices_tensor_cpu[:num_spec_decodes].copy_(
                spec_state_indices_tensor_cpu[:num_spec_decodes, 0]
                if spec_state_indices_tensor_cpu is not None
                else spec_state_indices_tensor[:, 0].to(device="cpu", dtype=torch.int32)
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)
            spec_conv_state_indices_tensor = self.spec_conv_state_indices_tensor[
                :batch_size
            ]
            spec_conv_state_indices_tensor[num_spec_decodes:].fill_(0)
            spec_conv_state_indices_tensor_cpu = (
                self.spec_conv_state_indices_tensor_cpu[:batch_size]
            )
            spec_conv_state_indices_tensor_cpu[num_spec_decodes:].fill_(0)
            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[:num_spec_decodes].fill_(True)
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert spec_token_masks is not None
            self.spec_token_masks[: spec_token_masks.size(0)].copy_(
                spec_token_masks, non_blocking=True
            )
            spec_token_masks = self.spec_token_masks[: spec_token_masks.size(0)]

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)
        elif spec_state_indices_tensor is not None:
            spec_conv_state_indices_tensor = (
                spec_state_indices_tensor[:, 0].to(torch.int32).contiguous()
            )
            spec_conv_state_indices_tensor_cpu = (
                spec_state_indices_tensor_cpu[:, 0].contiguous()
                if spec_state_indices_tensor_cpu is not None
                else _to_cpu(spec_conv_state_indices_tensor)
            )
            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens.to(dtype=torch.int32)

        if self._stage_decode(num_prefills, num_decodes, num_spec_decodes):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)
        elif (
            non_spec_state_indices_tensor is not None
            and non_spec_state_indices_tensor_cpu is None
        ):
            non_spec_state_indices_tensor_cpu = non_spec_state_indices_tensor.to(
                device="cpu", dtype=torch.int32
            )

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            has_initial_state_cpu=has_initial_state_cpu,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            non_spec_query_start_loc_cpu=non_spec_query_start_loc_cpu,
            spec_state_indices_tensor=spec_state_indices_tensor,
            # [Kunlun] These locals are named ``*_cpu`` but are produced by
            # indexing the *device* ``block_table_tensor`` (see the branches
            # above), so they can still live on the device. The consumers hand
            # them to kunlun ops as host pointers, so they must be forced to CPU
            # -- passing a device tensor there faults with
            # "an illegal memory access was encountered".
            spec_state_indices_tensor_cpu=_to_cpu(spec_state_indices_tensor_cpu),
            spec_conv_state_indices_tensor=spec_conv_state_indices_tensor,
            spec_conv_state_indices_tensor_cpu=_to_cpu(
                spec_conv_state_indices_tensor_cpu
            ),
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            non_spec_state_indices_tensor_cpu=_to_cpu(
                non_spec_state_indices_tensor_cpu
            ),
            spec_sequence_masks=spec_sequence_masks,
            spec_sequence_masks_cpu=spec_sequence_masks_cpu,
            spec_token_masks=spec_token_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            # [Kunlun] No consumer reads this: ``causal_conv1d_update`` accepts a
            # ``num_accepted_tokens_cpu`` argument but never forwards it to the
            # kernel. Dropping the blocking ``.cpu()`` removes one device sync
            # per spec-decode step per attention group.
            num_accepted_tokens_cpu=None,
            spec_pad_gather_idx=spec_pad_gather_idx,
            spec_unpad_idx=spec_unpad_idx,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata

    def update_block_table(
        self,
        metadata: GDNAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> GDNAttentionMetadata:
        """Rebuild only group-specific state indices for another KV group.

        Batch-level fields stay shared with ``metadata``. Device indices are
        always derived from the device block table. When a kernel consumes
        host indices, they are derived independently from vLLM's persistent
        CPU mirror; this path must never hide a per-group D2H synchronization.

        Pure speculative decode consumes only device state indices. Avoid
        materializing its unused CPU mirrors: apart from wasting host work,
        keeping them in reusable pinned buffers would violate the host-pointer
        lifetime required by Kunlun kernels that actually consume CPU inputs.
        """
        del slot_mapping  # GDN addresses state pages through the block table.
        cache_mode = self.vllm_config.cache_config.mamba_cache_mode
        if cache_mode not in ("none", "all"):
            raise NotImplementedError(
                "GDN metadata block-table updates currently support only "
                f"mamba_cache_mode='none'/'all', got {cache_mode!r}"
            )

        m = metadata
        masks = m.spec_sequence_masks_cpu
        pure_spec_decode = (
            masks is not None and m.num_prefills == 0 and m.num_decodes == 0
        )
        cpu_block_table = None
        if not pure_spec_decode:
            cpu_block_table = _cpu_state_block_table(blk_table, cache_mode)
            if cpu_block_table is None:
                raise RuntimeError(
                    "GDN metadata reuse requires the registered CPU block-table "
                    "mirror for non-spec metadata; refusing to fall back to a "
                    "device-to-host copy"
                )

        spec_indices: torch.Tensor | None = None
        spec_indices_cpu: torch.Tensor | None = None
        spec_conv_indices: torch.Tensor | None = None
        spec_conv_indices_cpu: torch.Tensor | None = None
        non_spec_indices: torch.Tensor | None = None
        non_spec_indices_cpu: torch.Tensor | None = None

        if masks is None:
            assert cpu_block_table is not None
            assert m.non_spec_state_indices_tensor is not None
            num_rows = m.non_spec_state_indices_tensor.shape[0]
            non_spec_indices = blk_table[:num_rows, 0].contiguous()
            non_spec_indices_cpu = cpu_block_table[:num_rows, 0].contiguous()
            if m.non_spec_query_start_loc_cpu is not None:
                padded_rows = (
                    m.non_spec_query_start_loc_cpu[1:]
                    == m.non_spec_query_start_loc_cpu[:-1]
                )
                non_spec_indices_cpu.masked_fill_(padded_rows, NULL_BLOCK_ID)
        else:
            if pure_spec_decode:
                # Padded rows trail real speculative decodes.
                spec_indices = blk_table[: m.num_spec_decodes, : self.num_spec + 1]
            else:
                assert cpu_block_table is not None
                non_spec_masks = self.non_spec_sequence_masks_cpu[: masks.numel()]
                torch.logical_not(masks, out=non_spec_masks)
                spec_indices = blk_table[masks, : self.num_spec + 1]
                spec_source_cpu = cpu_block_table[masks, : self.num_spec + 1]
                non_spec_indices = blk_table[non_spec_masks, 0].contiguous()
                non_spec_indices_cpu = cpu_block_table[non_spec_masks, 0].contiguous()
                if m.non_spec_query_start_loc_cpu is not None:
                    padded_rows = (
                        m.non_spec_query_start_loc_cpu[1:]
                        == m.non_spec_query_start_loc_cpu[:-1]
                    )
                    non_spec_indices_cpu.masked_fill_(padded_rows, NULL_BLOCK_ID)

                spec_indices_cpu = spec_source_cpu.contiguous()
                spec_conv_indices_cpu = spec_indices_cpu[:, 0].contiguous()

        stage_spec_decode = m.spec_pad_gather_idx is None and self._stage_spec_decode(
            m.num_prefills,
            m.num_decodes,
            m.num_spec_decodes,
            m.num_spec_decode_tokens,
        )
        if stage_spec_decode:
            assert m.spec_state_indices_tensor is not None
            assert spec_indices is not None
            assert m.spec_conv_state_indices_tensor is not None

            state_rows = m.spec_state_indices_tensor.shape[0]
            self.spec_state_indices_tensor[: m.num_spec_decodes].copy_(
                spec_indices, non_blocking=True
            )
            spec_indices = self.spec_state_indices_tensor[:state_rows]
            spec_indices[m.num_spec_decodes :].fill_(NULL_BLOCK_ID)

            conv_rows = m.spec_conv_state_indices_tensor.shape[0]
            self.spec_conv_state_indices_tensor[: m.num_spec_decodes].copy_(
                spec_indices[:, 0], non_blocking=True
            )
            spec_conv_indices = self.spec_conv_state_indices_tensor[:conv_rows]
            spec_conv_indices[m.num_spec_decodes :].fill_(0)
        elif spec_indices is not None:
            # Non-graph execution still requires a contiguous device vector,
            # but it does not need an intermediate when staging graph buffers.
            spec_conv_indices = spec_indices[:, 0].to(torch.int32).contiguous()

        if self._stage_decode(m.num_prefills, m.num_decodes, m.num_spec_decodes):
            assert m.non_spec_state_indices_tensor is not None
            assert non_spec_indices is not None
            state_rows = m.non_spec_state_indices_tensor.shape[0]
            self.non_spec_state_indices_tensor[: m.num_decodes].copy_(
                non_spec_indices[: m.num_decodes], non_blocking=True
            )
            non_spec_indices = self.non_spec_state_indices_tensor[:state_rows]
            non_spec_indices[m.num_decodes :].fill_(NULL_BLOCK_ID)

        return replace(
            m,
            spec_state_indices_tensor=spec_indices,
            spec_state_indices_tensor_cpu=spec_indices_cpu,
            spec_conv_state_indices_tensor=spec_conv_indices,
            spec_conv_state_indices_tensor_cpu=spec_conv_indices_cpu,
            non_spec_state_indices_tensor=non_spec_indices,
            non_spec_state_indices_tensor_cpu=non_spec_indices_cpu,
        )

    def _maybe_warmup_spec_kernel(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> None:
        """[Kunlun] Eagerly warm up the spec-decode fused_recurrent kernel
        before it is recorded into a FULL CUDA graph.

        Why: with MTP, the pre-capture warmup pass (_warmup_and_capture ->
        _dummy_run(cudagraph_runtime_mode=NONE)) builds metadata via the normal
        ``build()`` path. Because the dummy run never populates
        ``num_decode_draft_tokens``, ``build()`` takes the *non-spec* branch and
        the uniform (query_len = 1 + num_spec) requests are treated as prefill
        (chunk path). The actual capture, however, goes through
        ``build_for_cudagraph_capture`` which synthesizes *spec* metadata from
        ``diff(query_start_loc)`` and runs the *spec* ``fused_recurrent``. That
        spec kernel is therefore launched for the very first time inside the
        capture region.

        The Kunlun native op does capture-illegal work on its first launch for a
        new shape (load-based kernel auto-selection, L3 scratch allocation,
        module setup), which fails during capture with
        "CUDA error: unrecognized error code" (dump shows l3_size=0). Running it
        once in eager beforehand moves that first-launch cost outside the graph.

        This is invoked from ``build_for_cudagraph_capture``, which runs *before*
        the ``torch.cuda.graph`` capture region begins; an
        ``is_current_stream_capturing()`` guard makes sure we never launch the
        kernel while a capture is in progress.

        How to apply: only relevant for FULL cudagraph + spec decode; a no-op
        otherwise. Warms once per capture batch size.
        """
        if not (self.use_full_cuda_graph and self.use_spec_decode):
            return
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            return

        num_reqs = int(common_attn_metadata.num_reqs)
        if num_reqs <= 0 or num_reqs in self._spec_kernel_warmed:
            return

        try:
            from vllm_kunlun.ops.fla.fused_recurrent import (
                fused_recurrent_gated_delta_rule,
            )

            hf_config = self.vllm_config.model_config.hf_config
            tc = getattr(hf_config, "text_config", hf_config)
            num_k_heads = tc.linear_num_key_heads
            num_v_heads = tc.linear_num_value_heads
            head_k_dim = tc.linear_key_head_dim
            head_v_dim = tc.linear_value_head_dim

            spec_width = self.num_spec + 1
            total = num_reqs * spec_width
            dev = self.device
            # Match the real spec call: fp16 io / g / beta, fp16 state, and a
            # non-contiguous (transposed) h0 like the page-padded ssm_state view.
            io_dtype = torch.float16
            q = torch.zeros(
                1, total, num_k_heads, head_k_dim, dtype=io_dtype, device=dev
            )
            k = torch.zeros_like(q)
            v = torch.zeros(
                1, total, num_v_heads, head_v_dim, dtype=io_dtype, device=dev
            )
            g = torch.zeros(1, total, num_v_heads, dtype=io_dtype, device=dev)
            beta = torch.zeros(1, total, num_v_heads, dtype=io_dtype, device=dev)
            h0 = torch.zeros(
                num_reqs,
                num_v_heads,
                head_v_dim,
                head_k_dim,
                dtype=io_dtype,
                device=dev,
            ).transpose(-1, -2)
            cu_seqlens = torch.arange(
                0, total + 1, spec_width, dtype=torch.int32, device=dev
            )
            ssm_state_indices = (
                torch.arange(total, dtype=torch.int32, device=dev)
                .remainder(num_reqs)
                .reshape(num_reqs, spec_width)
            )
            num_accepted_tokens = torch.full(
                (num_reqs,), spec_width, dtype=torch.int32, device=dev
            )

            fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=h0,
                inplace_final_state=True,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=ssm_state_indices,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
            )
            torch.cuda.synchronize()
            self._spec_kernel_warmed.add(num_reqs)
            logger.info(
                "[KunlunPlugin] warmed spec GDN fused_recurrent kernel for "
                "cudagraph capture (num_reqs=%d, tokens=%d)",
                num_reqs,
                total,
            )
        except Exception as e:  # warmup must never break startup
            logger.warning(
                "[KunlunPlugin] spec GDN kernel warmup failed (num_reqs=%s): %r",
                num_reqs,
                e,
            )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        # Warm the spec fused_recurrent kernel before graph capture.
        self._maybe_warmup_spec_kernel(common_attn_metadata)

        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )
        num_accepted_tokens = torch.diff(m.query_start_loc)
        # During v0.25.1 FULL-graph capture the device query boundaries are
        # authoritative. The CPU mirror can still describe padded rows from
        # the dummy input; using it may capture only the first real request and
        # leave later batch rows permanently zero on replay. This synchronization
        # happens only during startup capture, not on the steady-state metadata
        # path that Stage 1 optimizes.
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()
        assert num_decode_draft_tokens_cpu.shape == num_accepted_tokens.shape

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)


# Monkey-patch upstream vllm so the GDN_ATTN backend registry resolves to
# Kunlun classes. Required because qwen3_next.py does
# `isinstance(attn_metadata, GDNAttentionMetadata)` against the Kunlun class,
# while the backend lookup via MambaAttentionBackendEnum.GDN_ATTN otherwise
# instantiates the upstream class.
_upstream_gdn_attn.GDNAttentionMetadata = GDNAttentionMetadata
_upstream_gdn_attn.GDNAttentionMetadataBuilder = GDNAttentionMetadataBuilder
_upstream_gdn_attn.GDNAttentionBackend = GDNAttentionBackend
