# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific monkey-patch for ``vllm.v1.worker.block_table``.

Replaces ``BlockTable.compute_slot_mapping`` (which dispatches a Triton
kernel ``_compute_slot_mapping_kernel`` upstream) with the native Kunlun
XPU op ``kunlun_ops.compute_slot_mappings``. Kunlun XPU cannot JIT-compile
Triton kernels via the CUDA driver path; the previous torch-native fallback
relied on ``torch.searchsorted``, which has no Kunlun XPU implementation and
silently falls back to CPU (host sync + D2H/H2D round trip every step). The
native op computes the whole mapping on-device in a single launch.

Mamba/GDN state-cache groups skip token slot mapping. On older vLLM versions
without SlotMappingMode, the V1 runner marks groups from their KV cache specs
after input buffers are created or rebuilt.

Each active table also owns an immutable device block-size tensor, prepared
with the input batch and reused across steps. A block-size or device change
refreshes it; rebuilding the input batch gives the new tables fresh caches.

The existing CPU block-table mirror is also registered by device pointer for
GDN metadata construction. This lets the GDN builder obtain host state indices
without reading the device block table back once per attention group.

Triggering: imported from ``vllm_kunlun.__init__`` post-import hook
once ``vllm.v1.worker.block_table`` is loaded. Idempotent under fork()
and re-import via the ``_kunlun_slot_patched`` flag on the class.
"""

import functools
import logging
from weakref import WeakValueDictionary

import kunlun_ops
import torch
from vllm.v1.worker.block_table import PAD_SLOT_ID
from vllm.v1.worker.block_table import BlockTable as _upstream_cls

logger = logging.getLogger("vllm_kunlun")


def _get_device_tensor_with_cpu_mirror(self, num_reqs):
    original = _upstream_cls._kunlun_original_get_device_tensor
    tensor = original(self, num_reqs)
    self._kunlun_cpu_mirror_tables[tensor.data_ptr()] = self
    return tensor


def get_cpu_block_table(device_tensor: torch.Tensor) -> torch.Tensor | None:
    """Return the persistent host mirror backing ``device_tensor``.

    ``GPUModelRunner._get_block_table`` obtains the device view through
    ``BlockTable.get_device_tensor``. Registering that view's base pointer lets
    attention builders recover the already-maintained ``CpuGpuBuffer.cpu``
    tensor without changing vLLM's ``CommonAttentionMetadata`` API.
    """
    registry = getattr(_upstream_cls, "_kunlun_cpu_mirror_tables", None)
    table = registry.get(device_tensor.data_ptr()) if registry is not None else None
    if table is None:
        return None
    rows, columns = device_tensor.shape
    return table.get_cpu_tensor()[:rows, :columns]


def _get_slot_mapping_block_sizes(table):
    # Cache only the immutable scalar metadata, never block-table/slot pointers.
    # Replacing an input/output buffer on the same device does not invalidate it.
    key = (table.block_size, table.block_table.gpu.device)
    block_sizes = getattr(table, "_kunlun_slot_mapping_block_sizes", None)
    if (
        block_sizes is None
        or getattr(table, "_kunlun_slot_mapping_block_sizes_key", None) != key
    ):
        block_sizes = torch.tensor([key[0]], dtype=torch.int32, device=key[1])
        table._kunlun_slot_mapping_block_sizes = block_sizes
        table._kunlun_slot_mapping_block_sizes_key = key
    return block_sizes


def _set_slot_mapping_enabled(runner, kv_cache_config):
    from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec, MambaSpec

    specs = [
        group.kv_cache_spec
        for group in kv_cache_config.kv_cache_groups
        if not isinstance(group.kv_cache_spec, EncoderOnlyAttentionSpec)
    ]
    tables = runner.input_batch.block_table.block_tables
    if len(specs) != len(tables):
        raise RuntimeError("KV cache specs and V1 block tables differ in length")
    for spec, table in zip(specs, tables):
        layer_specs = list(getattr(spec, "kv_cache_specs", {}).values()) or [spec]
        table._kunlun_slot_mapping_enabled = not all(
            isinstance(layer_spec, MambaSpec) for layer_spec in layer_specs
        )
        mode = getattr(table, "slot_mapping_mode", None)
        if (
            table._kunlun_slot_mapping_enabled
            and getattr(mode, "value", None) != "none"
        ):
            # Prepare before graph capture and the measured steady-state window.
            _get_slot_mapping_block_sizes(table)


def patch_slot_mapping_runner(module):
    """Configure V1 groups and scalar metadata after buffers are created/rebuilt."""
    cls = module.GPUModelRunner
    original = getattr(cls, "may_reinitialize_input_batch", None)
    if original is None or getattr(
        original, "_kunlun_slot_mapping_modes_patched", False
    ):
        return

    @functools.wraps(original)
    def may_reinitialize_input_batch(self, kv_cache_config, *args, **kwargs):
        result = original(self, kv_cache_config, *args, **kwargs)
        _set_slot_mapping_enabled(self, kv_cache_config)
        return result

    may_reinitialize_input_batch._kunlun_slot_mapping_modes_patched = True
    cls.may_reinitialize_input_batch = may_reinitialize_input_batch


def _compute_slot_mapping(self, num_reqs, query_start_loc, positions):
    mode = getattr(self, "slot_mapping_mode", None)
    if getattr(mode, "value", None) == "none" or not getattr(
        self, "_kunlun_slot_mapping_enabled", True
    ):
        return
    num_tokens = positions.shape[0]
    total_cp_world_size = self.pcp_world_size * self.dcp_world_size
    total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
    # Lazy initialization also supports callers outside GPUModelRunner.
    block_sizes = _get_slot_mapping_block_sizes(self)
    kunlun_ops.compute_slot_mappings(
        [self.slot_mapping.gpu],  # list
        [self.block_table.gpu],  # list
        positions,
        query_start_loc,
        block_sizes,  # int32 tensor
        num_reqs,
        num_tokens,
        PAD_SLOT_ID,
        total_cp_world_size,
        total_cp_rank,
        self.cp_kv_cache_interleave_size,
    )


# def _compute_slot_mapping(self, num_reqs, query_start_loc, positions):
#     num_tokens = positions.shape[0]
#     max_num_tokens = self.max_num_batched_tokens
#     block_size = self.block_size
#     # CPU mirrors maintained by CpuGpuBuffer. block_table.np is kept in
#     # sync via commit_block_table (CPU->GPU H2D each step), so reading
#     # from it here observes the same data the upstream Triton kernel
#     # would have read from block_table.gpu.
#     block_table_np = self.block_table.np
#     slot_mapping_np = self.slot_mapping.np
#     total_cp = self.pcp_world_size * self.dcp_world_size

#     # Common case: no context parallelism. Pure NumPy path.
#     if total_cp == 1:
#         if num_tokens > 0:
#             # One D2H sync for two small tensors:
#             #   positions: int64[num_tokens] (<= 64 KB at max_num_batched_tokens=8192)
#             #   qsl:       int32[num_reqs+1]
#             pos_np = positions[:num_tokens].cpu().numpy()
#             qsl_np = query_start_loc[: num_reqs + 1].cpu().numpy()
#             # Per-token req index via searchsorted (matches the Triton
#             # kernel's behavior).
#             token_arange = np.arange(num_tokens, dtype=qsl_np.dtype)
#             req_idx = np.searchsorted(qsl_np, token_arange, side="right") - 1
#             np.clip(req_idx, 0, num_reqs - 1, out=req_idx)
#             block_idx = pos_np // block_size
#             offset = pos_np - block_idx * block_size
#             block_num = block_table_np[req_idx, block_idx].astype(np.int64)
#             np.add(
#                 block_num * block_size,
#                 offset,
#                 out=slot_mapping_np[:num_tokens],
#             )
#         if max_num_tokens > num_tokens:
#             slot_mapping_np[num_tokens:max_num_tokens] = PAD_SLOT_ID
#         # H2D the prepared slot_mapping in one shot.
#         self.slot_mapping.copy_to_gpu()
#         return

#     # CP path: per-req loop on host, also in NumPy.
#     total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
#     cp_int = self.cp_kv_cache_interleave_size
#     virtual_block_size = block_size * total_cp
#     qsl_np = query_start_loc[: num_reqs + 1].cpu().numpy()
#     pos_np = positions[:num_tokens].cpu().numpy() if num_tokens > 0 else None
#     for r in range(num_reqs):
#         s, e = int(qsl_np[r]), int(qsl_np[r + 1])
#         if e <= s:
#             continue
#         pos = pos_np[s:e]
#         block_indices = pos // virtual_block_size
#         block_numbers = block_table_np[r, block_indices].astype(np.int64)
#         virtual_off = pos - block_indices * virtual_block_size
#         is_local = (virtual_off // cp_int) % total_cp == total_cp_rank
#         local_off = (virtual_off // (total_cp * cp_int)) * cp_int + (
#             virtual_off % cp_int
#         )
#         slot = block_numbers * block_size + local_off
#         slot_mapping_np[s:e] = np.where(is_local, slot, PAD_SLOT_ID)
#     if max_num_tokens > num_tokens:
#         slot_mapping_np[num_tokens:max_num_tokens] = PAD_SLOT_ID
#     self.slot_mapping.copy_to_gpu()

# Idempotent monkey-patch: safe under fork() and re-import.
if not getattr(_upstream_cls, "_kunlun_slot_patched", False):
    _upstream_cls._kunlun_cpu_mirror_tables = WeakValueDictionary()
    _upstream_cls._kunlun_original_get_device_tensor = _upstream_cls.get_device_tensor
    _upstream_cls.get_device_tensor = _get_device_tensor_with_cpu_mirror
    _upstream_cls.compute_slot_mapping = _compute_slot_mapping
    _upstream_cls._kunlun_slot_patched = True
    logger.info(
        "[KunlunPlugin] BlockTable.compute_slot_mapping patched "
        "in vllm_kunlun/v1/worker/block_table.py"
    )
