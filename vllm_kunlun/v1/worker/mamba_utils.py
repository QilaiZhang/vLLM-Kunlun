# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun overrides for ``vllm.v1.worker.mamba_utils``.

Exactly two things differ from upstream on Kunlun XPU:

* ``batch_memcpy`` must go through ``torch.ops.xspeedgate_ops.batch_memcpy``
  instead of launching the Triton ``batch_memcpy_kernel``.
* ``MambaCopyBuffers.create`` allocates ``int64`` pointer/size buffers, which is
  what the xspeedgate op expects (upstream uses ``uint64``/``int32``).

Everything else -- the 5 Triton kernels, ``MambaSpecDecodeGPUContext``,
``MambaBuffers``, and the V1 pre/postprocess helpers -- is left untouched.

That is safe because ``@triton.jit`` is lazy: decorating a kernel compiles
nothing, only a ``kernel[grid](...)`` launch does. The kernels we do not
replace are reachable only from the mamba "align" cache mode, which requires
prefix caching to be enabled.

This replaces a 396-line fork of an *older* upstream revision that was missing
12 symbols the current upstream imports. Two of them broke Qwen3.5 outright::

    vllm/v1/worker/gpu/model_states/mamba_hybrid.py:27
    ImportError: cannot import name 'MambaSpecDecodeGPUContext'

and four more were latent ``AttributeError``s on the V1 path
(``gpu_model_runner.py`` lines 1547, 1570, 2098, 4258). Patching the two real
deltas in place, instead of hand-maintaining a whole export surface, removes
that class of failure entirely.

Also dropped here: ``get_hybrid_attention_mamba_layout`` and
``postprocess_mamba``, two symbols the old fork carried that exist neither
upstream nor in any caller.
"""

import logging
from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput

import torch
import vllm.v1.worker.mamba_utils as _up

logger = logging.getLogger("vllm_kunlun")


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    """xspeedgate stand-in for upstream's Triton ``batch_memcpy_kernel``.

    ``xspeedgate_ops.batch_memcpy`` is specified for int64 pointer and size
    tensors, and every buffer that reaches it comes from the
    ``MambaCopyBuffers.create`` override below, which allocates exactly that:
    the op has a single call path (upstream ``preprocess_mamba`` ->
    ``do_mamba_copy_block``), and ``MambaCopyBuffers`` has a single construction
    site (upstream ``MambaBuffers.create``), which goes through the override.

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


