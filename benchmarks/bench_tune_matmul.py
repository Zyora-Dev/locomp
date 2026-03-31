"""
Tuning the 2×2 micro-tiled matmul: sweep BM/BN/BK combinations.
Also try the 4×4 micro-tile for comparison.
"""

import time
import numpy as np
import locomp


def make_micro_2x2(BM, BN, BK):
    """Generate a 2×2 micro-tiled matmul kernel with given block sizes."""
    TILE_R = BM // 2  # threadgroup rows
    TILE_C = BN // 2  # threadgroup cols

    @locomp.kernel
    def kern(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
             M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr,
             NUM_K_TILES: locomp.constexpr, BLOCK_K: locomp.constexpr,
             BLOCK_M: locomp.constexpr, BLOCK_N: locomp.constexpr):
        trow = locomp.local_id(1)
        tcol = locomp.local_id(0)
        brow = locomp.program_id(1)
        bcol = locomp.program_id(0)
        As = locomp.shared_memory(BM * BK)
        Bs = locomp.shared_memory(BK * BN)
        acc00 = 0.0
        acc01 = 0.0
        acc10 = 0.0
        acc11 = 0.0
        for t in range(NUM_K_TILES):
            a_base_row = brow * BLOCK_M + trow * 2
            a_base_col = t * BLOCK_K + tcol
            a_val0 = locomp.load(A + ((a_base_row + 0) * K + a_base_col))
            a_val1 = locomp.load(A + ((a_base_row + 1) * K + a_base_col))
            locomp.shared_store(As, (trow * 2 + 0) * BLOCK_K + tcol, a_val0)
            locomp.shared_store(As, (trow * 2 + 1) * BLOCK_K + tcol, a_val1)
            b_base_row = t * BLOCK_K + trow
            b_base_col = bcol * BLOCK_N + tcol * 2
            b_val0 = locomp.load(B + (b_base_row * N + b_base_col + 0))
            b_val1 = locomp.load(B + (b_base_row * N + b_base_col + 1))
            locomp.shared_store(Bs, trow * BLOCK_N + tcol * 2 + 0, b_val0)
            locomp.shared_store(Bs, trow * BLOCK_N + tcol * 2 + 1, b_val1)
            locomp.barrier()
            for k in range(BLOCK_K):
                a0 = locomp.shared_load(As, (trow * 2 + 0) * BLOCK_K + k)
                a1 = locomp.shared_load(As, (trow * 2 + 1) * BLOCK_K + k)
                b0 = locomp.shared_load(Bs, k * BLOCK_N + tcol * 2 + 0)
                b1 = locomp.shared_load(Bs, k * BLOCK_N + tcol * 2 + 1)
                acc00 = acc00 + a0 * b0
                acc01 = acc01 + a0 * b1
                acc10 = acc10 + a1 * b0
                acc11 = acc11 + a1 * b1
            locomp.barrier()
        out_row = brow * BLOCK_M + trow * 2
        out_col = bcol * BLOCK_N + tcol * 2
        locomp.store(C + ((out_row + 0) * N + out_col + 0), acc00)
        locomp.store(C + ((out_row + 0) * N + out_col + 1), acc01)
        locomp.store(C + ((out_row + 1) * N + out_col + 0), acc10)
        locomp.store(C + ((out_row + 1) * N + out_col + 1), acc11)

    return kern, TILE_R, TILE_C


def bench_config(label, kern, tile_r, tile_c, bm, bn, bk, sz):
    M = N = K = sz
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)
    a = locomp.tensor(a_np.flatten())
    b = locomp.tensor(b_np.flatten())
    c = locomp.empty(M * N)

    nkt = K // bk
    grid = (N // bn, M // bm)
    tg = (tile_c, tile_r)

    # Verify correctness first
    kern[grid, tg](a, b, c, M, N, K, nkt, bk, bm, bn)
    result = c.numpy().reshape(M, N)
    expected = a_np @ b_np
    err = np.max(np.abs(result - expected))

    WARMUP = 3; RUNS = 10
    for _ in range(WARMUP):
        kern[grid, tg](a, b, c, M, N, K, nkt, bk, bm, bn)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        kern[grid, tg](a, b, c, M, N, K, nkt, bk, bm, bn)
        times.append((time.perf_counter() - t0) * 1000)
    ms = sorted(times)[5]
    return ms, err


if __name__ == "__main__":
    # Configs: (label, BM, BN, BK)
    configs = [
        ("2x2 BM32 BK16", 32, 32, 16),
        ("2x2 BM32 BK32", 32, 32, 32),
        ("2x2 BM64 BK16", 64, 64, 16),
        ("2x2 BM64 BK32", 64, 64, 32),
    ]

    sizes = [128, 256, 512]

    print("2×2 Micro-Tile Parameter Sweep")
    print(f"{'Config':>18} | " + " | ".join(f"{sz:>9}" for sz in sizes))
    print("-" * (22 + 12 * len(sizes)))

    for label, bm, bn, bk in configs:
        kern, tile_r, tile_c = make_micro_2x2(bm, bn, bk)
        results = []
        for sz in sizes:
            if sz < bm or sz < bn:
                results.append("  N/A")
                continue
            # Thread count check: tile_r * tile_c must be <= 1024
            if tile_r * tile_c > 1024:
                results.append("  >1024")
                continue
            ms, err = bench_config(label, kern, tile_r, tile_c, bm, bn, bk, sz)
            results.append(f"{ms:>7.3f}ms")
            if err > 0.01:
                results[-1] += f" E{err:.0e}"
        print(f"{label:>18} | " + " | ".join(f"{r:>9}" for r in results))
