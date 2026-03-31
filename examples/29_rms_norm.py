"""
Example 29: RMSNorm — fused root-mean-square normalization.

Used in LLaMA, Mistral, Gemma, and most modern transformers.
  out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]

One threadgroup per row. SIMD reduce + shared memory for cross-group sum.
"""

import time
import numpy as np
import locomp


@locomp.kernel
def rms_norm(X: locomp.Tensor, W: locomp.Tensor, OUT: locomp.Tensor,
             ROWS: locomp.constexpr, D: locomp.constexpr,
             THREADS: locomp.constexpr, NUM_SIMD: locomp.constexpr,
             ELEMS: locomp.constexpr):
    row = locomp.program_id(0)
    lid = locomp.local_id(0)
    lane = locomp.simd_lane_id()
    simd_gid = locomp.simd_group_id()

    smem = locomp.shared_memory(NUM_SIMD)
    base = row * D

    # --- Phase 1: compute sum of squares ---
    local_sum_sq = 0.0
    for j in range(ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (base + idx))
        local_sum_sq = local_sum_sq + val * val

    # SIMD reduce within group
    group_sum = locomp.simd_sum(local_sum_sq)

    # Write lane-0 of each SIMD group to shared memory
    if lane == 0:
        locomp.shared_store(smem, simd_gid, group_sum)
    locomp.barrier()

    # Cross-group reduce
    total = locomp.shared_load(smem, 0)
    for g in range(1, NUM_SIMD):
        total = total + locomp.shared_load(smem, g)
    locomp.barrier()

    # rms = rsqrt(mean_sq + eps)
    rms = locomp.rsqrt(total / D + 1e-6)

    # --- Phase 2: normalize and write output ---
    for j in range(ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (base + idx))
        w = locomp.load(W + idx)
        locomp.store(OUT + (base + idx), val * rms * w)


def rms_norm_np(x, w, eps=1e-6):
    """Reference RMSNorm in numpy."""
    mean_sq = np.mean(x ** 2, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(mean_sq + eps)) * w


if __name__ == "__main__":
    import mlx.core as mx
    import mlx.nn as nn

    configs = [
        (32, 256),     # Small: 32 rows, D=256
        (32, 512),     # Medium: 32 rows, D=512
        (32, 1024),    # Standard: 32 rows, D=1024
        (32, 2048),    # Large: 32 rows, D=2048
        (32, 4096),    # LLaMA-7B hidden dim
        (128, 4096),   # Bigger batch
    ]

    WARMUP = 5; RUNS = 15

    print("RMSNorm: fused, 1 threadgroup/row, SIMD+smem reduce")
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
        expected = rms_norm_np(x_np, w_np)

        x_t = locomp.tensor(x_np.flatten())
        w_t = locomp.tensor(w_np)
        o_t = locomp.empty(rows * D)

        grid = (rows,)
        tg = (THREADS,)

        # Correctness
        rms_norm[grid, tg](x_t, w_t, o_t, rows, D, THREADS, NUM_SIMD, ELEMS)
        result = o_t.numpy().reshape(rows, D)
        err = np.max(np.abs(result - expected))

        # Benchmark locomp
        for _ in range(WARMUP):
            rms_norm[grid, tg](x_t, w_t, o_t, rows, D, THREADS, NUM_SIMD, ELEMS)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            rms_norm[grid, tg](x_t, w_t, o_t, rows, D, THREADS, NUM_SIMD, ELEMS)
            times.append((time.perf_counter() - t0) * 1000)
        t_loc = sorted(times)[RUNS // 2]

        # MLX RMSNorm
        x_mlx = mx.array(x_np)
        w_mlx = mx.array(w_np)
        rms_layer = nn.RMSNorm(D)
        rms_layer.weight = w_mlx
        mx.eval(rms_layer.weight)
        for _ in range(WARMUP):
            out = rms_layer(x_mlx)
            mx.eval(out)
        mt = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = rms_layer(x_mlx)
            mx.eval(out)
            mt.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mt)[RUNS // 2]

        ratio = t_loc / t_mlx
        print(f"{label:>20} | {t_loc:>6.3f}ms | {t_mlx:>6.3f}ms | {ratio:>6.2f}x | {err:.2e}")
