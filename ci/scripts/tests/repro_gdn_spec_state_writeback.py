"""Reproduce the native GDN speculative base-slot overwrite (no vLLM needed).

Run: python ci/scripts/tests/repro_gdn_spec_state_writeback.py

Only coordinate [0, 0] is active: q=k=1, scale=1, g=0, beta=1/2.
The recurrence reduces exactly to h[t] = (h[t-1] + v[t]) / 2.
Incoming accepted=4 selects initial h=2. Each candidate must retain its own
state: [1, 1.5, 2.75, 4.375]. The buggy kernel overwrites candidate 0 with
4.375, corrupting the next round when incoming accepted becomes 1.
"""

import kunlun_ops
import torch

DIM = 128


def run_step(state, values, accepted):
    """One request, four input tokens, one head; state is physical [V, K]."""
    q = torch.zeros((1, 4, 1, DIM), dtype=torch.float16, device="cuda")
    q[..., 0] = 1
    v = torch.zeros_like(q)
    v[0, :, 0, 0] = torch.tensor(values, dtype=q.dtype, device=q.device)
    output, _ = kunlun_ops.fused_recurrent_gated_delta_rule_fwdv2(
        q,
        q,
        v,
        torch.zeros((1, 4, 1), dtype=torch.float32, device=q.device),
        torch.full((1, 4, 1), 0.5, dtype=q.dtype, device=q.device),
        1.0,
        state,
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32, device=q.device),
        h0_indices=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device=q.device),
        num_accepted_tokens=torch.tensor(
            [accepted], dtype=torch.int32, device=q.device
        ),
        use_qk_l2norm_in_kernel=False,
        is_h0_transposed=True,
    )
    return output.cpu().float()


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Run in the Kunlun CUDA-compatible Python environment")

    # Slots 1..4 are candidates; 0 and 5 are untouched guards.
    state = torch.zeros((6, 1, DIM, DIM), dtype=torch.float32, device="cuda")
    state[0].fill_(-99)
    state[5].fill_(99)
    state[1:5, 0, 0, 0] = torch.tensor([0.25, 0.5, 1, 2], device=state.device)
    expected_state = state.cpu().clone()
    expected_state[1:5, 0, 0, 0] = torch.tensor([1, 1.5, 2.75, 4.375])

    first = run_step(state, [0, 2, 4, 6], accepted=4)
    first_state = state.cpu().clone()
    # Assume round 1 accepts one output; round 2 must then read S0=1.
    second = run_step(state, [8, 10, 12, 14], accepted=1)

    expected_first = torch.zeros_like(first)
    expected_first[0, :, 0, 0] = torch.tensor([1, 1.5, 2.75, 4.375])
    expected_second = torch.zeros_like(second)
    expected_second[0, :, 0, 0] = torch.tensor([4.5, 7.25, 9.625, 11.8125])

    print("Round 1 output:", first[0, :, 0, 0].tolist(), flush=True)
    print("Round 1 candidates:", first_state[1:5, 0, 0, 0].tolist(), flush=True)
    print("Round 2 output:", second[0, :, 0, 0].tolist(), flush=True)
    torch.testing.assert_close(first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(
        first_state,
        expected_state,
        rtol=0,
        atol=0,
        msg="GDN writeback bug: candidate 0 must retain S0=1, not S3=4.375",
    )
    torch.testing.assert_close(second, expected_second, rtol=0, atol=0)
    print("PASS: candidate writeback and the accepted=4 -> 1 transition are correct")


if __name__ == "__main__":
    main()
