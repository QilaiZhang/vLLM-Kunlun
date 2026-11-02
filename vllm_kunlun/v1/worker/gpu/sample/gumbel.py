# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun native Gumbel sampling overrides for the V2 model runner.

Random requests use the native Philox stream, not the former SplitMix64
stream. Seeded output is not bit-identical to the former implementation.
"""

import kunlun_ops
import torch


def apply_temperature(logits, expanded_idx_mapping, temperature):
    idx = expanded_idx_mapping.to(torch.long)
    temp = temperature[idx].to(torch.float32)
    safe = torch.where(temp == 0.0, torch.ones_like(temp), temp)
    logits.div_(safe.unsqueeze(1).to(logits.dtype))


def gumbel_sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    assert output_processed_logits is None, "Processed logits are not supported"
    assert output_processed_logits_col is None, "Processed logits are not supported"
    assert not use_fp64, "FP64 Gumbel sampling is not supported"
    n = logits.shape[0]
    output = torch.empty(n, dtype=torch.int64, device=logits.device)
    if n == 0:
        return output
    # The native binding requires nonnegative request indices. Expand request
    # state per sampled row and turn padded rows into greedy argmax requests.
    idx = expanded_idx_mapping.to(torch.long).clamp_min(0)
    row_temperature = temperature[idx].to(torch.float32)
    row_temperature = torch.where(
        expanded_idx_mapping >= 0, row_temperature, torch.zeros_like(row_temperature))
    row_seeds = seed[idx].to(torch.int64).contiguous()
    mapping = torch.arange(n, device=logits.device, dtype=torch.int64)
    kunlun_ops.gumbel_sample(
        logits.contiguous(), mapping, row_temperature.contiguous(), row_seeds,
        pos.to(torch.int64).contiguous(), output, apply_temperature,
    )
    return output
