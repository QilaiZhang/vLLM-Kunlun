# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DFlash2 draft model backport for the vLLM 0.25.1 model runner V1."""

from __future__ import annotations

from threading import RLock

import kunlun_ops
import torch
import torch.nn.functional as F
from torch import nn
from torch.library import custom_op
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

    hidden_fp32 = hidden_states.float()
    weight_fp32 = getattr(norm, "_dflash_weight_fp32", None)
    if weight_fp32 is None:
        # This fallback only applies when model weight finalization was skipped,
        # for example in an isolated unit test.
        weight_fp32 = norm.weight.float()

    if residual is None:
        residual_fp32 = hidden_fp32
        normalized_fp32 = torch.empty_like(residual_fp32)
        _dflash_rms_norm_fp32(
            residual_fp32,
            weight_fp32,
            normalized_fp32,
            float(norm.variance_epsilon),
        )
    else:
        residual_fp32 = residual.float()
        _dflash_add_rms_norm_fp32(
            hidden_fp32,
            residual_fp32,
            weight_fp32,
            float(norm.variance_epsilon),
        )
        normalized_fp32 = hidden_fp32

    return normalized_fp32.to(norm.weight.dtype), residual_fp32


@custom_op(
    "vllm::dflash2_rms_norm_fp32",
    mutates_args={"output"},
    device_types="cuda",
)
def _dflash_rms_norm_fp32(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    epsilon: float,
) -> None:
    """Run the existing Kunlun RMSNorm kernel with FP32 buffers."""
    kunlun_ops.rmsnorm(x, weight, output, epsilon)


@_dflash_rms_norm_fp32.register_fake
def _fake_dflash_rms_norm_fp32(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    epsilon: float,
) -> None:
    return None


@custom_op(
    "vllm::dflash2_add_rms_norm_fp32",
    mutates_args={"x", "residual"},
    device_types="cuda",
)
def _dflash_add_rms_norm_fp32(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """Fuse FP32 residual addition and RMSNorm using the Kunlun kernel."""
    kunlun_ops.add_rmsnorm(
        x,
        residual,
        weight,
        x,
        epsilon,
        residual_output=residual,
    )


@_dflash_add_rms_norm_fp32.register_fake
def _fake_dflash_add_rms_norm_fp32(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    return None


@custom_op(
    "vllm::dflash2_rotary_embedding",
    mutates_args={"query", "key"},
    device_types="cuda",
)
def _dflash2_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> None:
    """Expose Kunlun's in-place RoPE kernel to the compiled model graph."""
    kunlun_ops.rotary_embedding(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox_style,
    )


@_dflash2_rotary_embedding.register_fake
def _fake_dflash2_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> None:
    return None


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


@custom_op("vllm::dflash2_grouped_conv", mutates_args=())
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
    """Dispatch DFlash2's grouped convolution to the Kunlun operator."""
    if shift_matrices is None:
        shift_matrices = _make_conv_shift_matrices(
            block_size, taps, hidden_states.dtype, hidden_states.device
        )
    return kunlun_ops.dflash_grouped_conv(
        hidden_states,
        delta,
        base,
        shift_matrices,
        block_size,
        num_groups,
        group_size,
        taps,
        keep_fp32_output,
    )


def _grouped_conv_fake(
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
    del delta, base, block_size, num_groups, group_size, taps, shift_matrices
    output_dtype = (
        torch.float32
        if keep_fp32_output and hidden_states.dtype == torch.float16
        else hidden_states.dtype
    )
    return torch.empty(
        hidden_states.shape,
        dtype=output_dtype,
        device=hidden_states.device,
    )


_grouped_conv.register_fake(_grouped_conv_fake)


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
        return self._convolve(hidden_states, coefficients, 1, keep_fp32_output=True)


class DFlash2Qwen3Attention(DFlashQwen3Attention):
    """DFlash attention backed by Kunlun's fused RoPE kernel."""

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

        # DFlash flattens its token blocks before attention. Flatten positions
        # to match that layout, then rotate Q/K in place with one Kunlun kernel.
        cos_sin_cache = self.rotary_emb._match_cos_sin_cache_dtype(q)
        _dflash2_rotary_embedding(
            positions.flatten(),
            q,
            k,
            self.rotary_emb.head_size,
            cos_sin_cache,
            self.rotary_emb.is_neox_style,
        )

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

    def _build_fused_kv_buffers(self) -> None:
        """Build upstream KV buffers and cache FP32 weights for FP16 norms."""
        super()._build_fused_kv_buffers()
        if self.norm.weight.dtype != torch.float16:
            return

        norms = [self.norm]
        for layer in self.layers:
            norms.extend((layer.input_layernorm, layer.post_attention_layernorm))

        for norm in norms:
            weight_fp32 = norm.weight.detach().float()
            if hasattr(norm, "_dflash_weight_fp32"):
                norm._dflash_weight_fp32 = weight_fp32
            else:
                norm.register_buffer(
                    "_dflash_weight_fp32",
                    weight_fp32,
                    persistent=False,
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
        hidden_states, _ = _dflash_add_rms_norm(self.norm, hidden_states, residual)
        return hidden_states

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Precompute context K/V with the native Kunlun kernels."""
        if not hasattr(self, "_num_attn_layers"):
            self._build_fused_kv_buffers()

        num_ctx = context_states.shape[0]
        num_layers = self._num_attn_layers
        kv_size = self._kv_size
        head_dim = self._head_dim
        num_kv_heads = self._num_kv_heads

        # Keep the upstream fused projection: one GEMM produces K/V for all
        # decoder layers after the shared hidden-state normalization.
        normed_context_states = torch.empty_like(context_states)
        kunlun_ops.rmsnorm(
            context_states,
            self._hidden_norm_weight,
            normed_context_states,
            self._rms_norm_eps,
        )
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

        # The Kunlun kernel accepts one weight vector per invocation. Write
        # directly into the destination slice to avoid a temporary result and
        # the following slice-copy kernel.
        all_k_normed = torch.empty_like(all_k)
        for layer_idx in range(num_layers):
            kunlun_ops.rmsnorm(
                all_k[layer_idx],
                self._k_norm_weights[layer_idx],
                all_k_normed[layer_idx],
                self._rms_norm_eps,
            )

        # Run one in-place RoPE kernel across all layers. The Kunlun operator
        # supports the query-only form used by context K.
        all_k_flat = all_k_normed.view(num_layers * num_ctx, kv_size)
        positions_repeated = context_positions.repeat(num_layers)
        rotary_emb = self.layers[0].self_attn.rotary_emb
        cos_sin_cache = rotary_emb._match_cos_sin_cache_dtype(all_k_flat)
        kunlun_ops.rotary_embedding(
            positions_repeated,
            all_k_flat,
            None,
            rotary_emb.head_size,
            cos_sin_cache,
            rotary_emb.is_neox_style,
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
