"""
Example 40: Embedding Lookup — token indices → vectors.

Maps integer token IDs to embedding vectors from a weight table.
Core primitive for any transformer's first layer (input embeddings)
and last layer (output projection via tied weights).

Architecture:
  One threadgroup per token. 128 threads cooperatively load D floats.
  Supports arbitrary vocab_size and embed_dim.
"""

import time
import numpy as np
import locomp


@locomp.kernel
def embedding_lookup(tokens: locomp.Tensor,       # [N] float32 (token IDs as floats)
                     weight: locomp.Tensor,         # [V, D] float32 embedding table
                     output: locomp.Tensor,          # [N, D] float32 output
                     D: locomp.constexpr):
    n = locomp.program_id(0)            # token index
    tid = locomp.local_id(0)            # thread in group [0..127]

    # Read token ID (stored as float, cast to int via floor)
    token_id_f = locomp.load(tokens + n)
    # Use as integer index: token_id * D + offset
    # Since constexpr is int, we use multiply to compute base offset
    base = token_id_f * D

    # Each thread loads D/128 elements (stride of 128)
    ITERS = D // 128
    for i in range(ITERS):
        idx = tid + i * 128
        val = locomp.load(weight + (base + idx))
        locomp.store(output + (n * D + idx), val)


@locomp.kernel
def embedding_lookup_small(tokens: locomp.Tensor,
                           weight: locomp.Tensor,
                           output: locomp.Tensor,
                           D: locomp.constexpr):
    """For D < 128 — single thread per element."""
    n = locomp.program_id(0)
    d = locomp.program_id(1)

    token_id_f = locomp.load(tokens + n)
    base = token_id_f * D
    val = locomp.load(weight + (base + d))
    locomp.store(output + (n * D + d), val)


# =============================================================================
# Dispatch
# =============================================================================

def gpu_embedding(tokens, weight):
    """Embedding lookup. tokens:[N] int, weight:[V,D] → out:[N,D]."""
    N = tokens.shape[0]
    V, D = weight.shape

    tokens_f = tokens.astype(np.float32)
    T_g = locomp.tensor(tokens_f)
    W_g = locomp.tensor(weight.reshape(-1))
    O_g = locomp.empty(N * D)

    if D >= 128 and D % 128 == 0:
        embedding_lookup[(N,), (128,)](T_g, W_g, O_g, D)
    else:
        embedding_lookup_small[(N, D), (1,)](T_g, W_g, O_g, D)

    return O_g.numpy().reshape(N, D)


if __name__ == "__main__":
    WARMUP = 5
    RUNS = 15

    print(f"\n{'='*70}")
    print("Embedding Lookup: token IDs → vectors")
    print(f"{'='*70}")
    print(f"{'Config':>35} | {'GPU':>8} | {'NumPy':>8} | GPU/NP | {'Error':>8}")
    print("-" * 72)

    configs = [
        (32, 50257, 768, "GPT-2 style"),
        (64, 50257, 768, "GPT-2 longer"),
        (128, 32000, 4096, "Llama-7B style"),
        (256, 32000, 4096, "Llama-7B long"),
        (512, 128256, 4096, "Llama-3 vocab"),
    ]

    for N, V, D, label in configs:
        np.random.seed(42)
        tokens = np.random.randint(0, V, size=N).astype(np.int32)
        weight = np.random.randn(V, D).astype(np.float32) * 0.02

        ref = weight[tokens]
        gpu_out = gpu_embedding(tokens, weight)
        err = np.max(np.abs(gpu_out - ref))

        for _ in range(WARMUP):
            gpu_embedding(tokens, weight)
        times_gpu = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            gpu_embedding(tokens, weight)
            times_gpu.append((time.perf_counter() - t0) * 1000)
        t_gpu = sorted(times_gpu)[RUNS // 2]

        for _ in range(WARMUP):
            weight[tokens]
        times_np = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            weight[tokens]
            times_np.append((time.perf_counter() - t0) * 1000)
        t_np = sorted(times_np)[RUNS // 2]

        r = t_gpu / t_np
        desc = f"{label} N={N} V={V} D={D}"
        print(f"{desc:>35} | {t_gpu:>6.3f}ms | {t_np:>6.3f}ms | {r:>5.2f}x | {err:.2e}")
