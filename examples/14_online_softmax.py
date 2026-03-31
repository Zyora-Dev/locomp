"""
2-Pass Online Softmax — fuses max-finding + exp-sum into one pass.

Standard: 3 passes over data (find max, exp sum, normalize)
Online:   2 passes (fused max+exp_sum, normalize)
Saves 33% memory bandwidth.

Uses SIMD intrinsics for warp-level reduction (no barriers).
"""

import time
import numpy as np
import locomp


@locomp.kernel
def online_softmax_32(X: locomp.Tensor, OUT: locomp.Tensor,
                      ROWS: locomp.constexpr, D: locomp.constexpr,
                      ELEMS: locomp.constexpr):
    row = locomp.program_id(0)
    lid = locomp.local_id(0)
    guard = row < ROWS

    # Pass 1: Online max + exp sum (fused, single pass over data)
    local_max = locomp.load(X + (row * D + lid))
    local_sum = 1.0

    for j in range(1, ELEMS):
        idx = lid + j * 32
        val = locomp.load(X + (row * D + idx))
        new_max = locomp.where(val > local_max, val, local_max)
        local_sum = local_sum * locomp.exp(local_max - new_max) + locomp.exp(val - new_max)
        local_max = new_max

    # SIMD reduction: merge (max, sum) across 32 lanes
    row_max = locomp.simd_max(local_max)
    local_sum = local_sum * locomp.exp(local_max - row_max)
    total_sum = locomp.simd_sum(local_sum)

    # Pass 2: Normalize (one pass)
    for j in range(ELEMS):
        idx = lid + j * 32
        val = locomp.load(X + (row * D + idx))
        result = locomp.exp(val - row_max) / total_sum
        locomp.store(OUT + (row * D + idx), result, mask=guard)


@locomp.kernel
def online_softmax_256(X: locomp.Tensor, OUT: locomp.Tensor,
                       ROWS: locomp.constexpr, D: locomp.constexpr,
                       ELEMS: locomp.constexpr, NUM_SIMD: locomp.constexpr):
    row = locomp.program_id(0)
    lid = locomp.local_id(0)
    guard = row < ROWS

    smem = locomp.shared_memory(256)

    # Pass 1: Online max + exp sum (fused)
    local_max = locomp.load(X + (row * D + lid))
    local_sum = 1.0

    for j in range(1, ELEMS):
        idx = lid + j * 256
        val = locomp.load(X + (row * D + idx))
        new_max = locomp.where(val > local_max, val, local_max)
        local_sum = local_sum * locomp.exp(local_max - new_max) + locomp.exp(val - new_max)
        local_max = new_max

    # Phase 1a: SIMD max within each 32-thread group
    group_max = locomp.simd_max(local_max)
    # Correct partial sum for group max
    local_sum = local_sum * locomp.exp(local_max - group_max)
    group_sum = locomp.simd_sum(local_sum)

    # Phase 1b: Cross-group reduction via shared memory
    simd_gid = locomp.simd_group_id()
    simd_lid = locomp.simd_lane_id()

    # Write each group's max and sum
    if simd_lid == 0:
        locomp.shared_store(smem, simd_gid, group_max)
        locomp.shared_store(smem, simd_gid + NUM_SIMD, group_sum)
    locomp.barrier()

    # Thread 0 merges all groups
    if lid == 0:
        global_max = locomp.shared_load(smem, 0)
        global_sum = locomp.shared_load(smem, NUM_SIMD)
        for g in range(1, NUM_SIMD):
            gm = locomp.shared_load(smem, g)
            gs = locomp.shared_load(smem, g + NUM_SIMD)
            new_gmax = locomp.where(gm > global_max, gm, global_max)
            global_sum = global_sum * locomp.exp(global_max - new_gmax) + gs * locomp.exp(gm - new_gmax)
            global_max = new_gmax
        locomp.shared_store(smem, 0, global_max)
        locomp.shared_store(smem, 1, global_sum)
    locomp.barrier()

    row_max = locomp.shared_load(smem, 0)
    total_sum = locomp.shared_load(smem, 1)

    # Pass 2: Normalize
    for j in range(ELEMS):
        idx = lid + j * 256
        val = locomp.load(X + (row * D + idx))
        result = locomp.exp(val - row_max) / total_sum
        locomp.store(OUT + (row * D + idx), result, mask=guard)


def run_bench(label, kern, x, out, args, grid, tg, expected):
    """Test + benchmark a kernel."""
    kern[grid, tg](*args)
    result = out.numpy()
    err = np.max(np.abs(result - expected.flatten()))

    WARMUP = 3; RUNS = 10
    for _ in range(WARMUP):
        kern[grid, tg](*args)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        kern[grid, tg](*args)
        times.append((time.perf_counter() - t0) * 1000)
    ms = sorted(times)[5]
    return ms, err


if __name__ == "__main__":
    import mlx.core as mx

    print("Online Softmax (2-pass) vs 3-pass vs MLX")
    print(f"{'Shape':>10} | {'Online32':>9} | {'Online256':>10} | {'MLX':>7} | {'Best/MLX':>9}")
    print("-" * 60)

    shapes = [(32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]

    for ROWS, D in shapes:
        x_np = np.random.randn(ROWS, D).astype(np.float32)
        from scipy.special import softmax as scipy_softmax
        expected = scipy_softmax(x_np, axis=1)

        x = locomp.tensor(x_np.flatten())

        # Online 32-thread (D must be multiple of 32)
        if D >= 32 and D % 32 == 0:
            ELEMS = D // 32
            out = locomp.empty(ROWS * D)
            ms_32, err_32 = run_bench("online32", online_softmax_32, x, out,
                [x, out, ROWS, D, ELEMS], (ROWS,), (32,), expected)
        else:
            ms_32 = None; err_32 = 0

        # Online 256-thread (D must be multiple of 256)
        if D >= 256 and D % 256 == 0:
            ELEMS = D // 256
            NUM_SIMD = 256 // 32  # 8 SIMD groups
            out = locomp.empty(ROWS * D)
            ms_256, err_256 = run_bench("online256", online_softmax_256, x, out,
                [x, out, ROWS, D, ELEMS, NUM_SIMD], (ROWS,), (256,), expected)
        else:
            ms_256 = None; err_256 = 0

        # MLX
        mx_x = mx.random.normal((ROWS, D)); mx.eval(mx_x)
        for _ in range(3):
            mx_o = mx.softmax(mx_x, axis=1); mx.eval(mx_o)
        mlx_t = []
        for _ in range(10):
            t0 = time.perf_counter()
            mx_o = mx.softmax(mx_x, axis=1); mx.eval(mx_o)
            mlx_t.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mlx_t)[5]

        s32 = f"{ms_32:.3f}ms" if ms_32 else "  N/A"
        s256 = f"{ms_256:.3f}ms" if ms_256 else "   N/A"
        best = min(ms_32 or 999, ms_256 or 999)
        ratio = best / t_mlx if best < 999 else 0
        print(f"  {ROWS:>3}×{D:<4} | {s32:>9} | {s256:>10} | {t_mlx:>5.3f}ms | {ratio:>8.2f}×")
        if ms_32:
            print(f"           err32={err_32:.2e}", end="")
        if ms_256:
            print(f"  err256={err_256:.2e}", end="")
        print()
