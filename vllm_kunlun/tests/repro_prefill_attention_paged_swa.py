#!/usr/bin/env python3
"""Minimal operator-only reproducer for paged non-causal SWA failures.

This script intentionally imports neither vLLM nor vllm-kunlun.  It calls
``kunlun_ops.prefill_attention`` with contiguous BHLD cache tensors and a
single valid page.  ``--case all`` runs each case in a fresh process because a
device 719 error invalidates the current device context.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    cache_mode: str
    causal: bool
    swa_left: int
    swa_right: int
    explicit_mask: bool = False


CASES = {
    # Known-good control: same paged/prefix/non-causal contract, no SWA.
    "paged_no_swa": Case("paged", False, -1, -1),
    # Narrows the failure to the right-window handling if this succeeds.
    "paged_left_only": Case("paged", False, 2048, 0),
    "paged_right_one": Case("paged", False, 2048, 1),
    # DFlash2 contract that reproduces the device 719 failure.
    "paged_bidirectional_swa": Case("paged", False, 2048, 2048),
    # Conventional decoder SWA control.
    "paged_causal_swa": Case("paged", True, 2048, 0),
    # Distinguishes a general XFA SWA problem from its paged-cache branch.
    "dense_bidirectional_swa": Case("dense", False, 2048, 2048),
    # Validates the only cheap fallback candidate: keep paged K/V, disable the
    # broken right-window argument, and express the window as an additive mask.
    "paged_explicit_mask": Case("paged", False, -1, -1, True),
    "dense_explicit_mask": Case("dense", False, -1, -1, True),
}


def run_child(case_name: str) -> int:
    import kunlun_ops
    import torch

    case = CASES[case_name]
    device = torch.device("cuda")
    dtype = torch.float16
    q_len = 8
    kv_len = 61
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    block_size = 64

    generator = torch.Generator(device=device)
    generator.manual_seed(20250901)
    q = torch.randn(
        (q_len, num_heads, head_size),
        generator=generator,
        dtype=dtype,
        device=device,
    )
    out = torch.empty_like(q)
    q_lod_cpu = torch.tensor([0, q_len], dtype=torch.int32)
    kv_lod_cpu = torch.tensor([0, kv_len], dtype=torch.int32)
    q_lod_xpu = q_lod_cpu.to(device)
    kv_lod_xpu = kv_lod_cpu.to(device)

    if case.cache_mode == "paged":
        # Compact BHLD pages: no hybrid packing and no unusual tensor stride.
        k = torch.randn(
            (9, num_kv_heads, block_size, head_size),
            generator=generator,
            dtype=dtype,
            device=device,
        )
        v = torch.randn(
            k.shape,
            generator=generator,
            dtype=dtype,
            device=device,
        )
        block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
        is_prefix_cache = True
    else:
        k = torch.randn(
            (kv_len, num_kv_heads, head_size),
            generator=generator,
            dtype=dtype,
            device=device,
        )
        v = torch.randn(
            k.shape,
            generator=generator,
            dtype=dtype,
            device=device,
        )
        block_table = None
        is_prefix_cache = False

    # The public wrapper currently requires FP32 mask input.  Mask several
    # valid keys rather than using an all-zero mask: this verifies that XFA
    # consumes the bias with the expected dtype and [B, H, Q, KV] layout.
    mask = None
    if case.explicit_mask:
        mask = torch.zeros(
            (1, 1, q_len, kv_len), dtype=torch.float32, device=device
        )
        mask[..., :7] = -10000.0

    print(
        f"CASE={case_name} cache_mode={case.cache_mode} "
        f"is_prefix_cache={is_prefix_cache} causal={case.causal} "
        f"swa_left={case.swa_left} swa_right={case.swa_right} "
        f"explicit_mask={case.explicit_mask} "
        f"q_shape={tuple(q.shape)} q_stride={tuple(q.stride())} "
        f"k_shape={tuple(k.shape)} k_stride={tuple(k.stride())} "
        f"q_lod={q_lod_cpu.tolist()} kv_lod={kv_lod_cpu.tolist()} "
        f"block_table={None if block_table is None else [[0]]}",
        flush=True,
    )

    control_out = None
    if case.explicit_mask:
        control_out = torch.empty_like(q)
        kunlun_ops.prefill_attention(
            q=q,
            k=k,
            v=v,
            out=control_out,
            is_causal=False,
            is_prefix_cache=is_prefix_cache,
            context_qlen_lod_cpu=q_lod_cpu,
            context_qlen_lod_xpu=q_lod_xpu,
            context_kvlen_lod_cpu=kv_lod_cpu,
            context_kvlen_lod_xpu=kv_lod_xpu,
            block_table=block_table,
            swa_left=-1,
            swa_right=-1,
        )

    kunlun_ops.prefill_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        is_causal=case.causal,
        is_prefix_cache=is_prefix_cache,
        context_qlen_lod_cpu=q_lod_cpu,
        context_qlen_lod_xpu=q_lod_xpu,
        context_kvlen_lod_cpu=kv_lod_cpu,
        context_kvlen_lod_xpu=kv_lod_xpu,
        block_table=block_table,
        mask=mask,
        swa_left=case.swa_left,
        swa_right=case.swa_right,
    )

    # item() provides a device completion point without torch.cuda.synchronize,
    # which is not supported by the Kunlun PyTorch compatibility layer.
    max_abs = out.float().abs().max().item()
    mask_control_delta = (
        None
        if control_out is None
        else (out.float() - control_out.float()).abs().max().item()
    )
    mask_reference_error = None
    if case.explicit_mask:
        if case.cache_mode == "paged":
            dense_k = k[0, :, :kv_len, :].permute(1, 0, 2)
            dense_v = v[0, :, :kv_len, :].permute(1, 0, 2)
        else:
            dense_k = k
            dense_v = v
        groups = num_heads // num_kv_heads
        ref_k = dense_k.float().repeat_interleave(groups, dim=1)
        ref_v = dense_v.float().repeat_interleave(groups, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q.float(), ref_k)
        scores *= 1.0 / math.sqrt(head_size)
        scores += mask[0, 0].unsqueeze(0)
        probs = torch.softmax(scores, dim=-1)
        reference = torch.einsum("hqk,khd->qhd", probs, ref_v)
        mask_reference_error = (out.float() - reference).abs().max().item()
    print(
        f"PASS case={case_name} output_max_abs={max_abs:.6g} "
        f"mask_control_delta={mask_control_delta} "
        f"mask_reference_error={mask_reference_error}",
        flush=True,
    )
    return 0


def run_all() -> int:
    script = str(Path(__file__).resolve())
    env = os.environ.copy()
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    failed: list[str] = []
    for case_name in CASES:
        print(f"\n===== {case_name} =====", flush=True)
        completed = subprocess.run(
            [sys.executable, script, "--case", case_name, "--child"],
            env=env,
            check=False,
        )
        print(f"RESULT case={case_name} returncode={completed.returncode}", flush=True)
        if completed.returncode != 0:
            failed.append(case_name)

    print("\n===== summary =====")
    print(f"failed={failed}")
    print(
        "Key comparison: paged_no_swa vs paged_bidirectional_swa. "
        "If only positive-right paged cases fail, the defect is below the "
        "vLLM integration layer and is specific to paged/prefix SWA handling."
    )
    # Failures are expected diagnostic output, not a harness failure.
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        if args.case == "all":
            parser.error("--child requires one concrete case")
        return run_child(args.case)
    if args.case == "all":
        return run_all()

    env = os.environ.copy()
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--case", args.case, "--child"],
        env=env,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
