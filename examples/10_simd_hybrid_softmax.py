"""
Example: Hybrid SIMD+Shared Memory Softmax.

Uses simd_sum/simd_max WITHIN each 32-thread SIMD group to skip tree
reduction within each group, then uses shared memory only for the 8
cross-group sums/maxes. This is the optimal pattern.
"""

import locomp
import numpy as np


@locomp.kernel
def simd_hybrid_softmax(X: locomp.Tensor, OUT: locomp.Tensor,
                        ROWS: locomp.constexpr, D: locomp.constexpr,
                        THREADS: locomp.constexpr, ELEMS: locomp.constexpr,
                        NUM_SIMD: locomp.constexpr):
    row = locomp.program_id(0)
    lid = locomp.local_id(0)
    guard = row < ROWS
    lane = locomp.simd_lane_id()
    simd_gid = locomp.simd_group_id()

    smem = locomp.shared_memory(256)

    # Phase 1: local max → simd_max → shared mem → global max
    local_max = locomp.load(X + (row * D + lid))
    for j in range(1, ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (row * D + idx))
        local_max = locomp.where(val > local_max, val, local_max)

    group_max = locomp.simd_max(local_max)
    if lane == 0:
        locomp.shared_store(smem, simd_gid, group_max)
    locomp.barrier()

    # Cross-group max via shared memory (NUM_SIMD entries)
    global_max = locomp.shared_load(smem, 0)
    for g in range(1, NUM_SIMD):
        gval = locomp.shared_load(smem, g)
        global_max = locomp.where(gval > global_max, gval, global_max)
    locomp.barrier()

    # Phase 2: local exp_sum → simd_sum → shared mem → global sum
    local_sum = 0.0
    for j in range(ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (row * D + idx))
        e = locomp.exp(val - global_max)
        local_sum = local_sum + e

    group_sum = locomp.simd_sum(local_sum)
    if lane == 0:
        locomp.shared_store(smem, simd_gid, group_sum)
    locomp.barrier()

    total_sum = locomp.shared_load(smem, 0)
    for g in range(1, NUM_SIMD):
        total_sum = total_sum + locomp.shared_load(smem, g)
    locomp.barrier()

    # Phase 3: normalize
    for j in range(ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (row * D + idx))
        e = locomp.exp(val - global_max)
        result = e / total_sum
        locomp.store(OUT + (row * D + idx), result, mask=guard)


def main():
    import time

    print("=" * 60)
    print("SIMD HYBRID SOFTMAX")
    print("=" * 60)

    for ROWS, D in [(32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]:
        THREADS = min(256, D)
        ELEMS = D // THREADS
        NUM_SIMD = THREADS // 32
        if ELEMS < 1 or D % THREADS != 0:
            continue

        x_np = np.random.randn(ROWS * D).astype(np.float32)
        x = locomp.tensor(x_np)
        out = locomp.empty(ROWS * D)

        simd_hybrid_softmax[(ROWS,), (THREADS,)](x, out, ROWS, D, THREADS, ELEMS, NUM_SIMD)

        x_2d = x_np.reshape(ROWS, D)
        x_shifted = x_2d - x_2d.max(axis=1, keepdims=True)
        expected = np.exp(x_shifted) / np.exp(x_shifted).sum(axis=1, keepdims=True)
        result = out.numpy()
        err = np.max(np.abs(result - expected.flatten()))

        # Timing
        for _ in range(3):
            o2 = locomp.empty(ROWS * D)
            simd_hybrid_softmax[(ROWS,), (THREADS,)](x, o2, ROWS, D, THREADS, ELEMS, NUM_SIMD)
        times = []
        for _ in range(10):
            o2 = locomp.empty(ROWS * D)
            t0 = time.perf_counter()
            simd_hybrid_softmax[(ROWS,), (THREADS,)](x, o2, ROWS, D, THREADS, ELEMS, NUM_SIMD)
            times.append((time.perf_counter() - t0) * 1000)
        med = sorted(times)[5]

        print(f"{ROWS:>3}×{D:<4} | T={THREADS:>3} | SIMD groups={NUM_SIMD} | {med:.3f}ms | err={err:.2e}")


if __name__ == "__main__":
    main()
