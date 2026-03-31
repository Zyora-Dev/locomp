"""
Example 30: LayerNorm — fused layer normalization.

Used in GPT-2, BERT, and many other transformers.
  out[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]

One threadgroup per row. Two-pass: compute mean, then variance, then normalize.
SIMD reduce + shared memory for cross-group reductions.
"""

import time
import numpy as np
import locomp


@locomp.kernel
def layer_norm(X: locomp.Tensor, W: locomp.Tensor, B: locomp.Tensor,
               OUT: locomp.Tensor,
               ROWS: locomp.constexpr, D: locomp.constexpr,
               THREADS: locomp.constexpr, NUM_SIMD: locomp.constexpr,
               ELEMS: locomp.constexpr):
    row = locomp.program_id(0)
    lid = locomp.local_id(0)
    lane = locomp.simd_lane_id()
    simd_gid = locomp.simd_group_id()

    smem = locomp.shared_memory(NUM_SIMD)
    base = row * D

    # --- Pass 1: compute mean ---
    local_sum = 0.0
    for j in range(ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (base + idx))
        local_sum = local_sum + val

    group_sum = locomp.simd_sum(local_sum)
    if lane == 0:
        locomp.shared_store(smem, simd_gid, group_sum)
    locomp.barrier()

    total = locomp.shared_load(smem, 0)
    for g in range(1, NUM_SIMD):
        total = total + locomp.shared_load(smem, g)
    mean = total / D
    locomp.barrier()

    # --- Pass 2: compute variance ---
    local_var = 0.0
    for j in range(ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (base + idx))
        diff = val - mean
        local_var = local_var + diff * diff

    group_var = locomp.simd_sum(local_var)
    if lane == 0:
        locomp.shared_store(smem, simd_gid, group_var)
    locomp.barrier()

    var_total = locomp.shared_load(smem, 0)
    for g in range(1, NUM_SIMD):
        var_total = var_total + locomp.shared_load(smem, g)
    inv_std = locomp.rsqrt(var_total / D + 1e-5)
    locomp.barrier()

    # --- Pass 3: normalize ---
    for j in range(ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (base + idx))
        w = locomp.load(W + idx)
        b = locomp.load(B + idx)
        normed = (val - mean) * inv_std * w + b
        locomp.store(OUT + (base + idx), normed)


def layer_norm_np(x, w, b, eps=1e-5):
    """Reference LayerNorm in numpy."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


if __name__ == "__main__":
    import mlx.core as mx
    import mlx.nn as nn

    configs = [
        (32, 256),
        (32, 512),
        (32, 768),     # GPT-2 small
        (32, 1024),    # GPT-2 medium
        (32, 2048),
        (32, 4096),
        (128, 4096),
    ]

    WARMUP = 5; RUNS = 15

    print("LayerNorm: fused, 1 threadgroup/row, SIMD+smem reduce")
    print(f"{'Config':>20} | {'Locomp':>8} | {'MLX':>8} | {'Ratio':>7} | {'Error':>8}")
    print("-" * 65)

    for rows, D in configs:
        THREADS = min(D, 256)
        NUM_SIMD = THREADS // 32
        ELEMS = D // THREADS
        label = f"{rows}×{D}"

        np.random.seed(42)
        x_np = np.random.randn(rows, D).astype(np.float32)
        w_np = np.random.randn(D).astype(np.float32)
        b_np = np.random.randn(D).astype(np.float32)
        expected = layer_norm_np(x_np, w_np, b_np)

        x_t = locomp.tensor(x_np.flatten())
        w_t = locomp.tensor(w_np)
        b_t = locomp.tensor(b_np)
        o_t = locomp.empty(rows * D)

        grid = (rows,)
        tg = (THREADS,)

        # Correctness
        layer_norm[grid, tg](x_t, w_t, b_t, o_t, rows, D, THREADS, NUM_SIMD, ELEMS)
        result = o_t.numpy().reshape(rows, D)
        err = np.max(np.abs(result - expected))

        # Benchmark locomp
        for _ in range(WARMUP):
            layer_norm[grid, tg](x_t, w_t, b_t, o_t, rows, D, THREADS, NUM_SIMD, ELEMS)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            layer_norm[grid, tg](x_t, w_t, b_t, o_t, rows, D, THREADS, NUM_SIMD, ELEMS)
            times.append((time.perf_counter() - t0) * 1000)
        t_loc = sorted(times)[RUNS // 2]

        # MLX LayerNorm
        x_mlx = mx.array(x_np)
        w_mlx = mx.array(w_np)
        b_mlx = mx.array(b_np)
        ln_layer = nn.LayerNorm(D, eps=1e-5)
        ln_layer.weight = w_mlx
        ln_layer.bias = b_mlx
        mx.eval(ln_layer.weight, ln_layer.bias)
        for _ in range(WARMUP):
            out = ln_layer(x_mlx)
            mx.eval(out)
        mt = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = ln_layer(x_mlx)
            mx.eval(out)
            mt.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mt)[RUNS // 2]

        ratio = t_loc / t_mlx
        print(f"{label:>20} | {t_loc:>6.3f}ms | {t_mlx:>6.3f}ms | {ratio:>6.2f}x | {err:.2e}")
