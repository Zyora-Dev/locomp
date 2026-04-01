"""
Example 46: KV-Cache Append — append new K,V tokens to the cache.

Critical for LLM autoregressive serving: at each decode step,
we compute K,V for the new token and append them to the cache.

Architecture:
  cache_k / cache_v: [MAX_SEQ, H, D] pre-allocated
  new_k / new_v: [B, H, D] the new token's K,V for B requests
  seq_lens: [B] current sequence length per request (position to write)

  One threadgroup per (batch, head). Threads cooperatively copy D elements.
"""

import time
import numpy as np
import locomp


@locomp.kernel
def kv_cache_append(
    cache_k: locomp.Tensor,    # [MAX_SEQ * H * D] flat KV cache for K
    cache_v: locomp.Tensor,    # [MAX_SEQ * H * D] flat KV cache for V
    new_k: locomp.Tensor,      # [B * H * D] new K vectors
    new_v: locomp.Tensor,      # [B * H * D] new V vectors
    dst_offsets: locomp.Tensor, # [B] pre-computed offsets: seq_len[b] * H * D (as float)
    H: locomp.constexpr,
    D: locomp.constexpr,
):
    b = locomp.program_id(0)     # batch index
    h = locomp.program_id(1)     # head index
    tid = locomp.local_id(0)     # thread (0..D-1)

    # Source: new_k[b, h, d] = new_k[b*H*D + h*D + d]
    src_base = b * H * D + h * D
    # Dest: precomputed base + h*D + d
    dst_base_f = locomp.load(dst_offsets + b)
    # dst_base_f = seq_len[b] * H * D (pre-multiplied on host)
    # We add h*D + tid as integer offset; the addition promotes to float
    # but the store index comes from the sum with a float base

    k_val = locomp.load(new_k + (src_base + tid))
    v_val = locomp.load(new_v + (src_base + tid))

    locomp.store(cache_k + (dst_base_f + h * D + tid), k_val)
    locomp.store(cache_v + (dst_base_f + h * D + tid), v_val)


# =============================================================================
# Dispatch
# =============================================================================

def gpu_kv_cache_append(cache_k, cache_v, new_k, new_v, seq_lens):
    """
    Append new K,V to cache. All shapes pre-validated.
    cache_k, cache_v: [MAX_SEQ, H, D]
    new_k, new_v: [B, H, D]
    seq_lens: [B] int positions
    """
    B, H, D = new_k.shape

    # Pre-compute destination offsets: seq_lens[b] * H * D
    dst_offsets = (seq_lens * H * D).astype(np.float32)

    ck = locomp.tensor(cache_k.flatten())
    cv = locomp.tensor(cache_v.flatten())
    nk = locomp.tensor(new_k.flatten())
    nv = locomp.tensor(new_v.flatten())
    do = locomp.tensor(dst_offsets)

    kv_cache_append[(B, H), (D,)](ck, cv, nk, nv, do, H, D)

    out_k = ck.numpy().reshape(cache_k.shape)
    out_v = cv.numpy().reshape(cache_v.shape)
    ck.free(); cv.free(); nk.free(); nv.free(); do.free()
    return out_k, out_v


if __name__ == "__main__":
    configs = [
        (1, 8, 64, 256),    # B=1, H=8, D=64, MAX_SEQ=256
        (4, 8, 64, 512),    # B=4
        (1, 32, 128, 1024), # LLaMA style
    ]

    print("=== KV-Cache Append ===")
    for B, H, D, MAX_SEQ in configs:
        cache_k = np.zeros((MAX_SEQ, H, D), dtype=np.float32)
        cache_v = np.zeros((MAX_SEQ, H, D), dtype=np.float32)
        new_k = np.random.randn(B, H, D).astype(np.float32)
        new_v = np.random.randn(B, H, D).astype(np.float32)

        # Simulate: first request at pos=5, second at pos=10, etc.
        seq_lens = np.array([5 + i * 3 for i in range(B)], dtype=np.int32)

        out_k, out_v = gpu_kv_cache_append(cache_k, cache_v, new_k, new_v, seq_lens)

        # Verify
        for bi in range(B):
            pos = seq_lens[bi]
            np.testing.assert_allclose(out_k[pos], new_k[bi], atol=1e-6)
            np.testing.assert_allclose(out_v[pos], new_v[bi], atol=1e-6)

        print(f"  B={B}, H={H}, D={D}, MAX_SEQ={MAX_SEQ} ✓")

    print("\nAll KV-cache append tests passed.")
