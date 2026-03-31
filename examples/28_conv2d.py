"""
Example 28: Conv2D — 2D convolution on Apple GPU.

Implements NCHW direct convolution:
  out[n, co, oh, ow] = sum_ci sum_kh sum_kw in[n, ci, oh+kh, ow+kw] * w[co, ci, kh, kw]

Each threadgroup computes one output pixel (all output channels).
Thread-level parallelism over output channels.
Shared memory caches the input patch (reused by all output channel threads).
"""

import time
import numpy as np
import locomp


@locomp.kernel
def conv2d(Input: locomp.Tensor, Weight: locomp.Tensor, Output: locomp.Tensor,
           N: locomp.constexpr, CI: locomp.constexpr, H: locomp.constexpr, W: locomp.constexpr,
           CO: locomp.constexpr, KH: locomp.constexpr, KW: locomp.constexpr,
           OH: locomp.constexpr, OW: locomp.constexpr,
           PATCH: locomp.constexpr):
    # program_id(0) = output spatial index (oh * OW + ow)
    # program_id(1) = batch index
    tid = locomp.local_id(0)   # thread within group → output channel
    spatial = locomp.program_id(0)
    batch = locomp.program_id(1)

    oh = spatial // OW
    ow = spatial % OW

    # Cache input patch in shared memory (CI * KH * KW values)
    patch = locomp.shared_memory(PATCH)

    # Cooperative load: threads stride across patch
    for i in range(tid, PATCH, CO):
        ci = i // (KH * KW)
        rem = i % (KH * KW)
        kh = rem // KW
        kw = rem % KW
        ih = oh + kh
        iw = ow + kw
        val = locomp.load(Input + (batch * CI * H * W + ci * H * W + ih * W + iw))
        locomp.shared_store(patch, i, val)
    locomp.barrier()

    # Each thread computes one output channel from shared patch
    co = tid
    acc = 0.0
    for i in range(PATCH):
        acc = acc + locomp.shared_load(patch, i) * locomp.load(Weight + (co * PATCH + i))

    locomp.store(Output + (batch * CO * OH * OW + co * OH * OW + oh * OW + ow), acc)


def conv2d_np(x, w):
    """Reference numpy conv2d (NCHW)."""
    N, CI, H, W = x.shape
    CO, CI2, KH, KW = w.shape
    OH = H - KH + 1
    OW = W - KW + 1
    out = np.zeros((N, CO, OH, OW), dtype=np.float32)
    for n in range(N):
        for co in range(CO):
            for ci in range(CI):
                for kh in range(KH):
                    for kw in range(KW):
                        out[n, co] += x[n, ci, kh:kh+OH, kw:kw+OW] * w[co, ci, kh, kw]
    return out


if __name__ == "__main__":
    import mlx.core as mx

    configs = [
        # (N, CI, H, W, CO, KH, KW)
        (1, 3, 32, 32, 16, 3, 3),     # Small: 3→16, 3×3
        (1, 16, 32, 32, 32, 3, 3),    # Medium: 16→32, 3×3
        (1, 32, 16, 16, 64, 3, 3),    # Larger filters
        (4, 3, 32, 32, 16, 3, 3),     # Batched
        (1, 64, 8, 8, 128, 3, 3),     # Many channels
    ]

    WARMUP = 5; RUNS = 15

    print("Conv2D: direct convolution, NCHW, 1 thread per output channel")
    print(f"{'Config':>30} | {'Locomp':>8} | {'MLX':>8} | {'Ratio':>7} | {'Error':>8}")
    print("-" * 75)

    for cfg in configs:
        N_batch, CI, H, W, CO, KH, KW = cfg
        OH = H - KH + 1
        OW = W - KW + 1
        label = f"{N_batch}×{CI}×{H}×{W} → {CO}×{KH}×{KW}"

        np.random.seed(42)
        x_np = np.random.randn(N_batch, CI, H, W).astype(np.float32)
        w_np = np.random.randn(CO, CI, KH, KW).astype(np.float32)
        expected = conv2d_np(x_np, w_np)

        x_t = locomp.tensor(x_np.flatten())
        w_t = locomp.tensor(w_np.flatten())
        o_t = locomp.empty(N_batch * CO * OH * OW)

        PATCH = CI * KH * KW
        grid = (OH * OW, N_batch)
        tg = (CO,)

        # Correctness
        conv2d[grid, tg](x_t, w_t, o_t, N_batch, CI, H, W, CO, KH, KW, OH, OW, PATCH)
        result = o_t.numpy().reshape(N_batch, CO, OH, OW)
        err = np.max(np.abs(result - expected))

        # Benchmark locomp
        for _ in range(WARMUP):
            conv2d[grid, tg](x_t, w_t, o_t, N_batch, CI, H, W, CO, KH, KW, OH, OW, PATCH)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            conv2d[grid, tg](x_t, w_t, o_t, N_batch, CI, H, W, CO, KH, KW, OH, OW, PATCH)
            times.append((time.perf_counter() - t0) * 1000)
        t_loc = sorted(times)[RUNS // 2]

        # MLX conv2d (NHWC format — MLX default)
        x_mlx = mx.array(x_np.transpose(0, 2, 3, 1))  # NCHW → NHWC
        w_mlx = mx.array(w_np.transpose(0, 2, 3, 1))  # (CO,CI,KH,KW) → (CO,KH,KW,CI)
        mx.eval(x_mlx, w_mlx)
        for _ in range(WARMUP):
            out = mx.conv2d(x_mlx, w_mlx)
            mx.eval(out)
        mt = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = mx.conv2d(x_mlx, w_mlx)
            mx.eval(out)
            mt.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mt)[RUNS // 2]

        ratio = t_loc / t_mlx
        print(f"{label:>30} | {t_loc:>6.3f}ms | {t_mlx:>6.3f}ms | {ratio:>6.2f}x | {err:.2e}")
