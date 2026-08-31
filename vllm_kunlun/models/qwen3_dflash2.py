# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DFlash2 draft model backport for the vLLM 0.25.1 model runner V1."""

from __future__ import annotations

from threading import RLock

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Attention,
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.model_executor.models.utils import maybe_prefix

# vLLM 0.25.1 predates the decoder_layer_cls/model_cls extension points added
# with upstream DFlash2. Model construction is serialized by vLLM; the lock also
# makes the temporary compatibility substitution safe between DFlash2 instances.
_MODEL_CONSTRUCTION_LOCK = RLock()


def _dflash_add_rms_norm(
    norm: nn.Module,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DFlash residual-add/RMSNorm without FP16 residual overflow.

    Dynamic convolution can legitimately produce values large enough that the
    fused FP16 ``residual += hidden`` overflows before RMSNorm rescales them.
    Keep only the residual stream and RMSNorm reduction in FP32; the normalized
    hidden stream returns to the model dtype for projections and attention.
    BF16 does not have this range limitation and keeps the optimized fused path.
    """
    # A finish-side grouped convolution intentionally keeps its output in
    # FP32, so select this path from the model parameter dtype rather than only
    # from ``hidden_states.dtype``.
    if norm.weight.dtype != torch.float16:
        if residual is None:
            return norm(hidden_states), hidden_states
        return norm(hidden_states, residual)

    residual_fp32 = hidden_states.float()
    if residual is not None:
        residual_fp32 = residual.float() + residual_fp32
    variance = residual_fp32.square().mean(dim=-1, keepdim=True)
    eps = float(norm.variance_epsilon)
    normalized = residual_fp32 * torch.rsqrt(variance + eps)
    normalized = normalized * norm.weight.float()
    return normalized.to(norm.weight.dtype), residual_fp32


def _prepare_rope_indices(rotary_emb: nn.Module) -> None:
    """Register static RoPE permutation tables outside the compiled forward."""
    if hasattr(rotary_emb, "_kunlun_rope_rotate_indices"):
        return

    rotary_dim = rotary_emb.rotary_dim
    half_rotary_dim = rotary_dim // 2
    head_size = rotary_emb.head_size
    pass_dim = head_size - rotary_dim
    device = rotary_emb.cos_sin_cache.device
    if rotary_emb.is_neox_style:
        rotate_values = list(range(half_rotary_dim, rotary_dim)) + list(
            range(half_rotary_dim)
        )
        frequency_values = list(range(half_rotary_dim)) * 2
        sign_values = [-1] * half_rotary_dim + [1] * half_rotary_dim
    else:
        rotate_values = [dim ^ 1 for dim in range(rotary_dim)]
        frequency_values = [dim // 2 for dim in range(rotary_dim)]
        sign_values = [-1 if dim % 2 == 0 else 1 for dim in range(rotary_dim)]

    rotate_values.extend(range(rotary_dim, head_size))
    frequency_values.extend([0] * pass_dim)
    sign_values.extend([1] * pass_dim)

    rotate_indices = torch.tensor(rotate_values, dtype=torch.int64, device=device)
    frequency_indices = torch.tensor(frequency_values, dtype=torch.int64, device=device)
    signs = torch.tensor(sign_values, dtype=torch.int8, device=device)
    rotary_emb.register_buffer(
        "_kunlun_rope_rotate_indices", rotate_indices, persistent=False
    )
    rotary_emb.register_buffer(
        "_kunlun_rope_cos_indices", frequency_indices, persistent=False
    )
    rotary_emb.register_buffer(
        "_kunlun_rope_sin_indices",
        frequency_indices + half_rotary_dim,
        persistent=False,
    )
    rotary_emb.register_buffer("_kunlun_rope_signs", signs, persistent=False)
    rotary_emb.register_buffer(
        "_kunlun_rope_mask",
        torch.tensor(
            [True] * rotary_dim + [False] * pass_dim,
            dtype=torch.bool,
            device=device,
        ),
        persistent=False,
    )


def _apply_rope_without_cat(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply RoPE without concatenation, stacking, or sliced-view writes."""
    _prepare_rope_indices(rotary_emb)
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin_cache = rotary_emb._match_cos_sin_cache_dtype(query)
    cos_sin = cos_sin_cache.index_select(0, positions)
    rotary_dim = rotary_emb.rotary_dim
    head_size = rotary_emb.head_size
    cos = cos_sin.index_select(-1, rotary_emb._kunlun_rope_cos_indices).unsqueeze(-2)
    sin = cos_sin.index_select(-1, rotary_emb._kunlun_rope_sin_indices).unsqueeze(-2)
    signed_sin = sin * rotary_emb._kunlun_rope_signs.view(1, 1, head_size)

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        original_shape = tensor.shape
        heads = tensor.reshape(num_tokens, -1, head_size)
        rotated = heads.index_select(-1, rotary_emb._kunlun_rope_rotate_indices)
        output = heads * cos + rotated * signed_sin
        if rotary_dim < head_size:
            output = torch.where(
                rotary_emb._kunlun_rope_mask.view(1, 1, head_size), output, heads
            )
        return output.reshape(original_shape)

    return rotate(query), None if key is None else rotate(key)


def _make_conv_shift_matrices(
    block_size: int,
    taps: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    matrices = [
        [
            [
                1 if position >= tap and source == position - tap else 0
                for source in range(block_size)
            ]
            for position in range(block_size)
        ]
        for tap in range(taps)
    ]
    return torch.tensor(matrices, dtype=dtype, device=device)


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
    shift_matrices: torch.Tensor | None = None,
    keep_fp32_output: bool = False,
) -> torch.Tensor:
    """Apply DFlash2's block-local dynamic grouped depthwise convolution."""
    # Kunlun elementwise kernels require contiguous inputs in several eager /
    # compiled paths. Attention and MLP outputs are not guaranteed to retain a
    # contiguous layout, so normalize it at this boundary.
    blocks = hidden_states.contiguous().unflatten(-1, (num_groups, group_size))

    # Parallel drafting lays tokens out as fixed request blocks. Preserve that
    # dimension explicitly so a shifted tap can never read the previous
    # request. This also avoids device-side arange/bitwise/remainder operations,
    # whose XMLIR implementations are not available on all Kunlun runtimes.
    torch._check(hidden_states.shape[0] % block_size == 0)
    block_count = hidden_states.shape[0] // block_size
    blocked_states = blocks.reshape(block_count, block_size, num_groups, group_size)
    blocked_delta = delta.unflatten(0, (block_count, block_size))
    base_coefficients = base.reshape(taps, num_groups, group_size)
    fp32_accumulation = hidden_states.dtype == torch.float16
    if fp32_accumulation:
        blocked_delta = blocked_delta.float()
        base_coefficients = base_coefficients.float()
    if shift_matrices is None:
        shift_matrices = _make_conv_shift_matrices(
            block_size, taps, hidden_states.dtype, hidden_states.device
        )
    blocked_states_flat = blocked_states.flatten(2)

    def contribution(tap: int, shifted: torch.Tensor) -> torch.Tensor:
        # Distribute (base + delta) * shifted. Each broadcast multiply produces
        # a full [B, L, G, S] result, so the following add is shape-matched.
        # This avoids both the unsupported two-way broadcast add and the
        # expand(...).contiguous() copy kernel used by the previous fallback.
        if fp32_accumulation:
            shifted = shifted.float()
        base_part = shifted * base_coefficients[tap]
        delta_part = shifted * blocked_delta[:, :, tap].unsqueeze(-1)
        return base_part + delta_part

    # Keep every accumulation out-of-place and exactly shape-matched. Kunlun's
    # stack implementation lowers to the same unsupported copy kernel as cat,
    # while the old in-place broadcast accumulation produced runtime error 719.
    output = contribution(0, blocked_states)
    for tap in range(1, taps):
        # A tiny Toeplitz 0/1 matmul implements the block-local shift without
        # invoking pad, gather, stack, or the backend's failing copy kernel.
        shifted = torch.matmul(shift_matrices[tap], blocked_states_flat).unflatten(
            -1, (num_groups, group_size)
        )
        output = output + contribution(tap, shifted)
    output = output.flatten(0, 1).flatten(-2)
    if fp32_accumulation and not keep_fp32_output:
        output = output.to(hidden_states.dtype)
    return output


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        if taps < 1:
            raise ValueError(f"conv_kernel_size must be positive, got {taps}.")
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        shift_matrices = _make_conv_shift_matrices(
            block_size,
            taps,
            params_dtype,
            self.base_kernel.device,
        )
        self.register_buffer("shift_matrices", shift_matrices, persistent=False)
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        side: int,
        keep_fp32_output: bool = False,
    ) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
            self.shift_matrices,
            keep_fp32_output,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        # The finish-side result is consumed directly by residual-add/RMSNorm.
        # Keep it in FP32 so values outside the FP16 range can be normalized
        # before converting the hidden stream back to the model dtype.
        return self._convolve(
            hidden_states, coefficients, 1, keep_fp32_output=True
        )


class DFlash2Qwen3Attention(DFlashQwen3Attention):
    """DFlash attention whose regular forward avoids xflashinfer RoPE."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _prepare_rope_indices(self.rotary_emb)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)

        # xflashinfer rejects the DFlash layout, while vLLM's native fallback
        # reconstructs Q/K with torch.cat, which Kunlun XMLIR also rejects.
        q, k = _apply_rope_without_cat(self.rotary_emb, positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        import vllm.model_executor.models.qwen3_dflash as dflash_module

        # 0.25.1 hard-codes DFlashQwen3Attention in the decoder constructor.
        with _MODEL_CONSTRUCTION_LOCK:
            original_attention = dflash_module.DFlashQwen3Attention
            dflash_module.DFlashQwen3Attention = DFlash2Qwen3Attention
            try:
                super().__init__(
                    vllm_config,
                    config=config,
                    layer_idx=layer_idx,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=prefix,
                )
            finally:
                dflash_module.DFlashQwen3Attention = original_attention
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            block_size=1 + speculative_config.num_speculative_tokens,
            params_dtype=vllm_config.model_config.dtype,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = _dflash_add_rms_norm(
            self.input_layernorm, hidden_states, residual
        )

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = _dflash_add_rms_norm(
            self.post_attention_layernorm, hidden_states, residual
        )
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
    previous_step_indices: torch.Tensor | None = None,
    first_step_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute selector edge scores for every predecessor/candidate pair."""
    successors = successor_table[candidate_ids]
    num_steps = candidate_ids.shape[1]
    if previous_step_indices is None or first_step_mask is None:
        device = candidate_ids.device
        previous_step_indices = torch.tensor(
            [max(step - 1, 0) for step in range(num_steps)],
            dtype=torch.int64,
            device=device,
        )
        first_step_mask = torch.tensor(
            [True] + [False] * (num_steps - 1),
            dtype=torch.bool,
            device=device,
        )
    predecessor_ids = candidate_ids.index_select(1, previous_step_indices)
    anchors = anchor_token_ids[:, None, None].expand_as(predecessor_ids)
    use_anchor = first_step_mask.view(1, num_steps, 1).expand_as(predecessor_ids)
    predecessor_ids = torch.where(use_anchor, anchors, predecessor_ids)
    predecessors = predecessor_table[predecessor_ids]
    pairwise = torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )
    return unary_logits[:, :, None] + pairwise


@support_torch_compile(
    dynamic_arg_dims={
        "candidate_ids": {0: "batch"},
        "unary_logits": {0: "batch"},
        "hidden_states": {0: "batch"},
        "anchor_token_ids": {0: "batch"},
    }
)
class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        num_steps: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        device = torch.empty(0, dtype=params_dtype).device
        self.register_buffer(
            "previous_step_indices",
            torch.tensor(
                [max(step - 1, 0) for step in range(num_steps)],
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "first_step_mask",
            torch.tensor(
                [True] + [False] * (num_steps - 1),
                dtype=torch.bool,
                device=device,
            ),
            persistent=False,
        )
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        torch._check(hidden_states.shape[0] == candidate_ids.shape[0])
        torch._check(hidden_states.shape[1] == candidate_ids.shape[1])
        hidden = self.hidden_projection(hidden_states.flatten(0, 1))
        hidden = hidden.view(*hidden_states.shape[:-1], -1)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
            self.previous_step_indices,
            self.first_step_mask,
        )


class DFlash2Qwen3Model(DFlashQwen3Model):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        import vllm.model_executor.models.qwen3_dflash as dflash_module

        # 0.25.1 hard-codes DFlashQwen3DecoderLayer inside the parent ctor.
        with _MODEL_CONSTRUCTION_LOCK:
            original_layer = dflash_module.DFlashQwen3DecoderLayer
            dflash_module.DFlashQwen3DecoderLayer = DFlash2Qwen3DecoderLayer
            try:
                super().__init__(
                    vllm_config=vllm_config,
                    start_layer_id=start_layer_id,
                    prefix=prefix,
                )
            finally:
                dflash_module.DFlashQwen3DecoderLayer = original_layer

        draft_config = self.config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                num_steps=speculative_config.num_speculative_tokens,
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the draft stack with an FP32 residual stream for FP16 models."""
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = _dflash_add_rms_norm(
            self.norm, hidden_states, residual
        )
        return hidden_states

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Precompute context K/V without vLLM's CUDA-only custom-op ABI.

        vLLM 0.25.1 calls ``ops.rms_norm`` and ``ops.rotary_embedding``
        directly here. Their four/six-argument CUDA schemas do not match the
        legacy Kunlun ``_C`` registrations. RMSNorm therefore goes through
        Kunlun's OOT layer modules, while RoPE uses equivalent no-cat math.
        """
        if not hasattr(self, "_num_attn_layers"):
            self._build_fused_kv_buffers()

        num_ctx = context_states.shape[0]
        num_layers = self._num_attn_layers
        kv_size = self._kv_size
        head_dim = self._head_dim
        num_kv_heads = self._num_kv_heads

        # Keep the upstream fused projection: one GEMM produces K/V for all
        # decoder layers after the shared hidden-state normalization.
        normed_context_states = self.hidden_norm(context_states)
        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )
        all_kv = (
            all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        all_k = all_kv[0]
        all_v = all_kv[1]

        # The Kunlun RMSNorm kernel takes one weight vector. Normalize each
        # layer separately instead of passing upstream's [L, H] grouped weight.
        all_k_normed = torch.empty_like(all_k)
        for layer_idx, layer in enumerate(self.layers):
            all_k_normed[layer_idx] = layer.self_attn.k_norm(all_k[layer_idx])

        # Use the same no-cat fallback as the regular attention path. It
        # supports the query-only form required by context K and preserves the
        # configured RoPE cache/scaling.
        all_k_flat = all_k_normed.view(num_layers * num_ctx, kv_size)
        positions_repeated = context_positions.repeat(num_layers)
        all_k_flat, _ = _apply_rope_without_cat(
            self.layers[0].self_attn.rotary_emb, positions_repeated, all_k_flat, None
        )

        if context_slot_mapping is None:
            return

        all_k_final = all_k_flat.view(num_layers, num_ctx, num_kv_heads, head_dim)
        per_layer_mapping = isinstance(context_slot_mapping, (list, tuple))
        for layer_idx in range(num_layers):
            slot_mapping = (
                context_slot_mapping[layer_idx]
                if per_layer_mapping
                else context_slot_mapping
            )
            if slot_mapping is None:
                continue
            attn = self._attn_layers[layer_idx]
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[layer_idx],
                all_v[layer_idx],
                attn.kv_cache,
                slot_mapping,
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    """DFlash2 draft head; its candidate TopK shares the target LM head."""

    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        import vllm.model_executor.models.qwen3_dflash as dflash_module

        # 0.25.1 similarly hard-codes DFlashQwen3Model in the parent ctor.
        with _MODEL_CONSTRUCTION_LOCK:
            original_model = dflash_module.DFlashQwen3Model
            dflash_module.DFlashQwen3Model = DFlash2Qwen3Model
            try:
                super().__init__(vllm_config=vllm_config, prefix=prefix)
            finally:
                dflash_module.DFlashQwen3Model = original_model

        draft_config = self.config.dflash_config
        self.output_multiplier = float(draft_config.get("output_multiplier", 1.0))
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.final_logit_softcapping = softcap if softcap > 0 else None

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return global-vocabulary candidate IDs and their unary logits."""
        if not isinstance(self.lm_head.quant_method, UnquantizedEmbeddingMethod):
            raise ValueError(
                "DFlash2 requires an unquantized target LM head for candidate TopK."
            )

        selector = self.model.candidate_selector
        logits = self.lm_head.quant_method.apply(self.lm_head, hidden_states, bias=None)
        num_padding = self.lm_head.shard_indices.num_org_vocab_padding
        if num_padding > 0:
            logits = logits[..., :-num_padding]
        values, ids = torch.topk(logits, selector.top_k, dim=-1)
        ids = ids.to(torch.int64)
        ids += self.lm_head.shard_indices.org_vocab_start_index

        if get_tensor_model_parallel_world_size() > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            values, selected = torch.topk(values, selector.top_k, dim=-1)
            ids = ids.gather(-1, selected)

        values = values.float() * self.output_multiplier
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            values = torch.tanh(values / cap) * cap
        return ids, values
