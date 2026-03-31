"""
Example 34: Fused Residual Add + RMSNorm.

In every transformer layer:  x = RMSNorm(residual + hidden)
Fuses the residual addition into the normalization kernel — saves one full
memory pass (read residual, read hidden, write sum, then read sum again for norm).

  residual_out = residual + hidden
  normed_out = RMSNorm(residual_out) * weight
"""

import time
import numpy as np
import locomp


@locomp.kernel
def fused_residual_rms_norm(Residual: locomp.Tensor, Hidden: locomp.Tensor,
                            W: locomp.Tensor,
                            ResOut: locomp.Tensor, NormOut: locomp.Tensor,
                            ROWS: locomp.constexpr, D: locomp.constexpr,
                            THREADS: locomp.constexpr, NUM_SIMD: locomp.constexpr,
                            ELEMS: locomp.constexpr):
    row = locomp.program_id(0)
    lid = locomp.local_id(0)
    lane = locomp.simd_lane_id()
    simd_gid = locomp.simd_group_id()

    smem = locomp.shared_memory(NUM_SIMD)
    base = row * D

    # --- Phase 1: residual add + accumulate sum_sq ---
    local_sum_sq = 0.0
    for j in range(ELEMS):
        idx = lid + j * THREADS
        r = locomp.load(Residual + (base + idx))
        h = locomp.load(Hidden + (base + idx))
        x = r + h
        # Write residual output (for next layer's residual connection)
        locomp.store(ResOut + (base + idx), x)
        local_sum_sq = local_sum_sq + x * x

    # SIMD reduce + cross-group reduce
    group_sum = locomp.simd_sum(local_sum_sq)
    if lane == 0:
        locomp.shared_store(smem, simd_gid, group_sum)
    locomp.barrier()

    total = locomp.shared_load(smem, 0)
    for g in range(1, NUM_SIMD):
        total = total + locomp.shared_load(smem, g)
    rms = locomp.rsqrt(total / D + 1e-6)
    locomp.barrier()

    # --- Phase 2: normalize and write ---
    for j in range(ELEMS):
        idx = lid + j * THREADS
        x = locomp.load(ResOut + (base + idx))
        w = locomp.load(W + idx)
        locomp.store(NormOut + (base + idx), x * rms * w)


def fused_residual_rms_norm_np(residual, hidden, weight, eps=1e-6):
    """Reference in numpy."""
    x = residual + hidden
    mean_sq = np.mean(x ** 2, axis=-1, keepdims=True)
    normed = x * (1.0 / np.sqrt(mean_sq + eps)) * weight
    return x, normed


if __name__ == "__main__":
    import mlx.core as mx
    import mlx.nn as nn

    configs = [
        (32, 512),
        (32, 1024),
        (32, 2048),
        (32, 4096),
        (128, 4096),
    ]

    WARMUP = 5; RUNS = 15

    print("Fused Residual+RMSNorm: single kernel, 1 memory pass")
    print(f"{'Config':>20} | {'Locomp':>8} | {'MLX':>8} | {'Ratio':>7} | {'Error':>8}")
    print("-" * 65)

    for rows, D in configs:
        THREADS = min(D, 256)
        NUM_SIMD = THREADS // 32
        ELEMS = D // THREADS
        label = f"{rows}×{D}"

        np.random.seed(42)
        res_np = np.random.randn(rows, D).astype(np.float32)
        hid_np = np.random.randn(rows, D).astype(np.float32)
        w_np = np.random.randn(D).astype(np.float32)
        exp_resout, exp_normed = fused_residual_rms_norm_np(res_np, hid_np, w_np)

        res_t = locomp.tensor(res_np.flatten())
        hid_t = locomp.tensor(hid_np.flatten())
        w_t = locomp.tensor(w_np)
        resout_t = locomp.empty(rows * D)
        normout_t = locomp.empty(rows * D)

        grid = (rows,)
        tg = (THREADS,)

        # Correctness
        fused_residual_rms_norm[grid, tg](res_t, hid_t, w_t, resout_t, normout_t,
                                          rows, D, THREADS, NUM_SIMD, ELEMS)
        res_result = resout_t.numpy().reshape(rows, D)
        norm_result = normout_t.numpy().reshape(rows, D)
        err_res = np.max(np.abs(res_result - exp_resout))
        err_norm = np.max(np.abs(norm_result - exp_normed))
        err = max(err_res, err_norm)

        # Benchmark locomp
        for _ in range(WARMUP):
            fused_residual_rms_norm[grid, tg](res_t, hid_t, w_t, resout_t, normout_t,
                                              rows, D, THREADS, NUM_SIMD, ELEMS)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            fused_residual_rms_norm[grid, tg](res_t, hid_t, w_t, resout_t, normout_t,
                                              rows, D, THREADS, NUM_SIMD, ELEMS)
            times.append((time.perf_counter() - t0) * 1000)
        t_loc = sorted(times)[RUNS // 2]

        # MLX: separate residual add + RMSNorm
        res_mlx = mx.array(res_np)
        hid_mlx = mx.array(hid_np)
        w_mlx = mx.array(w_np)
        rms_layer = nn.RMSNorm(D)
        rms_layer.weight = w_mlx
        mx.eval(rms_layer.weight)
        for _ in range(WARMUP):
            x = res_mlx + hid_mlx
            out = rms_layer(x)
            mx.eval(x, out)
        mt = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            x = res_mlx + hid_mlx
            out = rms_layer(x)
            mx.eval(x, out)
            mt.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mt)[RUNS // 2]

        ratio = t_loc / t_mlx
        print(f"{label:>20} | {t_loc:>6.3f}ms | {t_mlx:>6.3f}ms | {ratio:>6.2f}x | {err:.2e}")
