"""
Example 33: RoPE — Rotary Positional Embedding.

Used in LLaMA, Mistral, Gemma, GPT-NeoX — all modern LLMs.
Applies rotation to pairs of elements:
  x[2i]'   = x[2i] * cos(theta_i) - x[2i+1] * sin(theta_i)
  x[2i+1]' = x[2i] * sin(theta_i) + x[2i+1] * cos(theta_i)
where theta_i = pos / 10000^(2i/D)

Input shape: [B, H, N, D] — batch, heads, seq_len, head_dim
"""

import time
import numpy as np
import locomp


@locomp.kernel
def rope_kernel(X: locomp.Tensor, COS: locomp.Tensor, SIN: locomp.Tensor,
                OUT: locomp.Tensor,
                N: locomp.constexpr, D: locomp.constexpr,
                HALF_D: locomp.constexpr, STRIDE: locomp.constexpr):
    # program_id(0) = seq position, program_id(1) = batch*heads
    pos = locomp.program_id(0)
    bh = locomp.program_id(1)
    tid = locomp.local_id(0)  # thread handles one pair

    base = bh * STRIDE + pos * D
    cos_base = pos * HALF_D

    # Each thread handles one pair (tid, tid + HALF_D)
    i = tid
    x0 = locomp.load(X + (base + i))
    x1 = locomp.load(X + (base + i + HALF_D))
    c = locomp.load(COS + (cos_base + i))
    s = locomp.load(SIN + (cos_base + i))

    locomp.store(OUT + (base + i), x0 * c - x1 * s)
    locomp.store(OUT + (base + i + HALF_D), x0 * s + x1 * c)


def rope_np(x, cos, sin):
    """Reference RoPE in numpy. x: [B, H, N, D], cos/sin: [N, D//2]."""
    D = x.shape[-1]
    half = D // 2
    x0 = x[..., :half]
    x1 = x[..., half:]
    # Broadcast cos/sin from [N, D//2] to [1, 1, N, D//2]
    c = cos[np.newaxis, np.newaxis, :, :]
    s = sin[np.newaxis, np.newaxis, :, :]
    out = np.empty_like(x)
    out[..., :half] = x0 * c - x1 * s
    out[..., half:] = x0 * s + x1 * c
    return out


def make_rope_freqs(N, D, base=10000.0):
    """Precompute cos/sin tables."""
    half = D // 2
    freq = 1.0 / (base ** (np.arange(0, half).astype(np.float32) / half))
    pos = np.arange(N, dtype=np.float32)
    angles = np.outer(pos, freq)  # [N, D//2]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


if __name__ == "__main__":
    import mlx.core as mx

    configs = [
        # (B, H, N, D)
        (1, 8, 64, 64),      # Small
        (1, 8, 128, 64),     # Medium seq
        (1, 32, 128, 128),   # LLaMA-7B style
        (1, 32, 256, 128),   # Longer seq
        (4, 32, 128, 128),   # Batched
    ]

    WARMUP = 5; RUNS = 15

    print("RoPE: rotary positional embedding, 1 threadgroup per (pos, batch×head)")
    print(f"{'Config':>25} | {'Locomp':>8} | {'MLX':>8} | {'Ratio':>7} | {'Error':>8}")
    print("-" * 70)

    for B, H, N, D in configs:
        HALF_D = D // 2
        STRIDE = N * D
        label = f"{B}×{H}×{N}×{D}"

        np.random.seed(42)
        x_np = np.random.randn(B, H, N, D).astype(np.float32)
        cos_table, sin_table = make_rope_freqs(N, D)
        expected = rope_np(x_np, cos_table, sin_table)

        x_t = locomp.tensor(x_np.flatten())
        cos_t = locomp.tensor(cos_table.flatten())
        sin_t = locomp.tensor(sin_table.flatten())
        o_t = locomp.empty(B * H * N * D)

        grid = (N, B * H)
        tg = (HALF_D,)

        # Correctness
        rope_kernel[grid, tg](x_t, cos_t, sin_t, o_t, N, D, HALF_D, STRIDE)
        result = o_t.numpy().reshape(B, H, N, D)
        err = np.max(np.abs(result - expected))

        # Benchmark locomp
        for _ in range(WARMUP):
            rope_kernel[grid, tg](x_t, cos_t, sin_t, o_t, N, D, HALF_D, STRIDE)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            rope_kernel[grid, tg](x_t, cos_t, sin_t, o_t, N, D, HALF_D, STRIDE)
            times.append((time.perf_counter() - t0) * 1000)
        t_loc = sorted(times)[RUNS // 2]

        # MLX RoPE: manual implementation (MLX has nn.RoPE but it's higher-level)
        x_mlx = mx.array(x_np)
        cos_mlx = mx.array(cos_table)  # [N, D//2]
        sin_mlx = mx.array(sin_table)
        mx.eval(x_mlx, cos_mlx, sin_mlx)

        def mlx_rope(x, c, s):
            x0 = x[..., :HALF_D]
            x1 = x[..., HALF_D:]
            c_ = c[None, None, :, :]
            s_ = s[None, None, :, :]
            return mx.concatenate([x0 * c_ - x1 * s_, x0 * s_ + x1 * c_], axis=-1)

        for _ in range(WARMUP):
            out = mlx_rope(x_mlx, cos_mlx, sin_mlx)
            mx.eval(out)
        mt = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = mlx_rope(x_mlx, cos_mlx, sin_mlx)
            mx.eval(out)
            mt.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mt)[RUNS // 2]

        ratio = t_loc / t_mlx
        print(f"{label:>25} | {t_loc:>6.3f}ms | {t_mlx:>6.3f}ms | {ratio:>6.2f}x | {err:.2e}")
