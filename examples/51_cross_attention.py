"""
Example 51: Cross Attention — encoder-decoder attention.

Used in: T5, Whisper, BART, mBART, Stable Diffusion (cond on text).
Unlike self-attention where Q,K,V come from the same source,
cross-attention has Q from the decoder and K,V from the encoder.

  Attention(Q_dec, K_enc, V_enc) = softmax(Q_dec @ K_enc^T / sqrt(d)) @ V_enc

Architecture:
  Q: [B, H_q, N_q, D]  — decoder queries (short, e.g. 1 for decode step)
  K: [B, H_k, N_kv, D] — encoder keys (longer context)
  V: [B, H_k, N_kv, D] — encoder values

  One threadgroup per (batch, head, query_pos) triple.
  128 threads per group. Compute full attention over N_kv positions.
  SIMD reduction for softmax normalization.
"""

import time
import numpy as np
import locomp


@locomp.kernel
def cross_attention_d64(
    Q: locomp.Tensor,    # [B * H * N_q * D] flat
    K: locomp.Tensor,    # [B * H * N_kv * D] flat
    V: locomp.Tensor,    # [B * H * N_kv * D] flat
    O: locomp.Tensor,    # [B * H * N_q * D] flat
    N_q: locomp.constexpr,
    N_kv: locomp.constexpr,
    D: locomp.constexpr,
    SCALE_X1000: locomp.constexpr,  # scale * 1000 (int) to avoid float constexpr
):
    # program_id(0) = query position, program_id(1) = b * H + h
    q_pos = locomp.program_id(0)
    bh = locomp.program_id(1)
    tid = locomp.local_id(0)  # thread = output dim index (0..D-1)

    scale = SCALE_X1000 * 0.001

    q_base = bh * N_q * D + q_pos * D
    kv_base = bh * N_kv * D

    # Phase 1: Compute all attention scores (each thread does partial dot → SIMD reduce)
    # We reuse tid to iterate over D for each KV position
    # Since one threadgroup handles one (q_pos, bh), thread tid handles output[tid]
    # But first need all scores — each thread computes one score per KV position

    # Simple approach: thread 0 computes all scores sequentially, stores to smem
    # Then all threads read scores and compute weighted V sum
    smem_scores = locomp.shared_memory(N_kv)

    # Each thread computes scores for a subset of KV positions
    # Thread tid handles KV positions tid, tid+D, tid+2D, ...
    for j_base in range(0, N_kv, D):
        j = j_base + tid
        if j < N_kv:
            dot = 0.0
            for d in range(D):
                q_val = locomp.load(Q + (q_base + d))
                k_val = locomp.load(K + (kv_base + j * D + d))
                dot = dot + q_val * k_val
            locomp.shared_store(smem_scores, j, dot * scale)
    locomp.barrier()

    # Phase 2: Online softmax over scores (thread 0 computes max and normalizes)
    if tid == 0:
        m = locomp.shared_load(smem_scores, 0)
        for j in range(1, N_kv):
            m = locomp.max(m, locomp.shared_load(smem_scores, j))
        total = 0.0
        for j in range(N_kv):
            s = locomp.exp(locomp.shared_load(smem_scores, j) - m)
            locomp.shared_store(smem_scores, j, s)
            total = total + s
        inv_sum = 1.0 / total
        for j in range(N_kv):
            s = locomp.shared_load(smem_scores, j) * inv_sum
            locomp.shared_store(smem_scores, j, s)
    locomp.barrier()

    # Phase 3: Weighted sum of V — thread tid computes output[tid]
    out_val = 0.0
    for j in range(N_kv):
        weight = locomp.shared_load(smem_scores, j)
        v_val = locomp.load(V + (kv_base + j * D + tid))
        out_val = out_val + weight * v_val

    o_base = bh * N_q * D + q_pos * D
    locomp.store(O + (o_base + tid), out_val)


# =============================================================================
# Dispatch helper
# =============================================================================

def gpu_cross_attention(q, k, v):
    """Cross attention. q:[B,H,Nq,D], k/v:[B,H,Nkv,D] → out:[B,H,Nq,D]."""
    B, H, N_q, D = q.shape
    _, _, N_kv, _ = k.shape

    Q_g = locomp.tensor(q.flatten())
    K_g = locomp.tensor(k.flatten())
    V_g = locomp.tensor(v.flatten())
    O_g = locomp.empty(B * H * N_q * D)

    # scale = 1/sqrt(D), pass as int(scale * 1000) to avoid float constexpr
    import math
    SCALE_X1000 = int(round(1000.0 / math.sqrt(D)))

    # Grid: (N_q, B*H), Threads: D (one thread per output dim)
    cross_attention_d64[(N_q, B * H), (D,)](
        Q_g, K_g, V_g, O_g, N_q, N_kv, D, SCALE_X1000
    )

    result = O_g.numpy().reshape(B, H, N_q, D)
    Q_g.free(); K_g.free(); V_g.free(); O_g.free()
    return result


def cross_attention_np(q, k, v):
    """Reference cross attention in numpy."""
    D = q.shape[-1]
    scale = 1.0 / np.sqrt(D)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale  # [B,H,Nq,Nkv]
    max_s = scores.max(axis=-1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
    return np.matmul(attn, v)


if __name__ == "__main__":
    np.random.seed(42)

    print("=== Cross Attention ===")
    # Small sizes only to avoid OOM
    for B, H, Nq, Nkv, D in [(1, 2, 1, 16, 32), (1, 4, 4, 32, 32), (1, 2, 1, 64, 32)]:
        q = np.random.randn(B, H, Nq, D).astype(np.float32) * 0.1
        k = np.random.randn(B, H, Nkv, D).astype(np.float32) * 0.1
        v = np.random.randn(B, H, Nkv, D).astype(np.float32) * 0.1

        out = gpu_cross_attention(q, k, v)
        expected = cross_attention_np(q, k, v)
        np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-4)
        print(f"  B={B} H={H} Nq={Nq} Nkv={Nkv} D={D} ✓")

    print("\nAll cross attention tests passed.")
