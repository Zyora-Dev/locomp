"""
Optimized 2×2 micro-tiled matmul with proper cooperative loading.

BM=BN=32, BK=32, Threadgroup=16×16, 2×2 micro-tile.
Loading: linearized thread ID for arbitrary tile shapes.
Each tile step does 2× more k-work, halving barrier count.
"""

import time
import numpy as np
import locomp

TILE = 16
TM = 2; TN = 2
BM = TILE * TM   # 32
BN = TILE * TN   # 32
BK = 32           # Doubled k-tile


@locomp.kernel
def matmul_opt(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
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

    # Linearized thread id for cooperative loading
    lid = trow * 16 + tcol   # 0..255

    for t in range(NUM_K_TILES):
        # Load As[32][32] = 1024 elements with 256 threads → 4 per thread
        # Linearized: as_idx = lid*4+i, row = as_idx / BK, col = as_idx % BK
        a_idx0 = lid * 4
        a_idx1 = lid * 4 + 1
        a_idx2 = lid * 4 + 2
        a_idx3 = lid * 4 + 3
        a_row0 = a_idx0 / BLOCK_K
        a_col0 = a_idx0 % BLOCK_K
        a_row1 = a_idx1 / BLOCK_K
        a_col1 = a_idx1 % BLOCK_K
        a_row2 = a_idx2 / BLOCK_K
        a_col2 = a_idx2 % BLOCK_K
        a_row3 = a_idx3 / BLOCK_K
        a_col3 = a_idx3 % BLOCK_K
        gA_row = brow * BLOCK_M + a_row0
        gA_col = t * BLOCK_K + a_col0
        locomp.shared_store(As, a_idx0, locomp.load(A + (gA_row * K + gA_col)))
        gA_row = brow * BLOCK_M + a_row1
        gA_col = t * BLOCK_K + a_col1
        locomp.shared_store(As, a_idx1, locomp.load(A + (gA_row * K + gA_col)))
        gA_row = brow * BLOCK_M + a_row2
        gA_col = t * BLOCK_K + a_col2
        locomp.shared_store(As, a_idx2, locomp.load(A + (gA_row * K + gA_col)))
        gA_row = brow * BLOCK_M + a_row3
        gA_col = t * BLOCK_K + a_col3
        locomp.shared_store(As, a_idx3, locomp.load(A + (gA_row * K + gA_col)))

        # Load Bs[32][32] = 1024 elements with 256 threads → 4 per thread
        b_idx0 = lid * 4
        b_idx1 = lid * 4 + 1
        b_idx2 = lid * 4 + 2
        b_idx3 = lid * 4 + 3
        b_row0 = b_idx0 / BLOCK_N
        b_col0 = b_idx0 % BLOCK_N
        b_row1 = b_idx1 / BLOCK_N
        b_col1 = b_idx1 % BLOCK_N
        b_row2 = b_idx2 / BLOCK_N
        b_col2 = b_idx2 % BLOCK_N
        b_row3 = b_idx3 / BLOCK_N
        b_col3 = b_idx3 % BLOCK_N
        gB_row = t * BLOCK_K + b_row0
        gB_col = bcol * BLOCK_N + b_col0
        locomp.shared_store(Bs, b_idx0, locomp.load(B + (gB_row * N + gB_col)))
        gB_row = t * BLOCK_K + b_row1
        gB_col = bcol * BLOCK_N + b_col1
        locomp.shared_store(Bs, b_idx1, locomp.load(B + (gB_row * N + gB_col)))
        gB_row = t * BLOCK_K + b_row2
        gB_col = bcol * BLOCK_N + b_col2
        locomp.shared_store(Bs, b_idx2, locomp.load(B + (gB_row * N + gB_col)))
        gB_row = t * BLOCK_K + b_row3
        gB_col = bcol * BLOCK_N + b_col3
        locomp.shared_store(Bs, b_idx3, locomp.load(B + (gB_row * N + gB_col)))

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


def test(sz):
    M = N = K = sz
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)
    expected = a_np @ b_np

    a = locomp.tensor(a_np.flatten())
    b = locomp.tensor(b_np.flatten())
    c = locomp.empty(M * N)

    nkt = K // BK
    matmul_opt[(N//BN, M//BM), (TILE, TILE)](a, b, c, M, N, K, nkt, BK, BM, BN)
    result = c.numpy().reshape(M, N)
    return np.max(np.abs(result - expected))


def bench(sz):
    M = N = K = sz
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)
    a = locomp.tensor(a_np.flatten())
    b = locomp.tensor(b_np.flatten())
    c = locomp.empty(M * N)

    nkt = K // BK
    grid = (N//BN, M//BM)
    tg = (TILE, TILE)

    for _ in range(3):
        matmul_opt[grid, tg](a, b, c, M, N, K, nkt, BK, BM, BN)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        matmul_opt[grid, tg](a, b, c, M, N, K, nkt, BK, BM, BN)
        times.append((time.perf_counter() - t0) * 1000)
    return sorted(times)[5]


if __name__ == "__main__":
    import mlx.core as mx

    print("Optimized 2×2 Matmul (BK=32, cooperative loading)")
    print(f"BM={BM}, BN={BN}, BK={BK}, TG=({TILE},{TILE})")
    print(f"{'Size':>8} | {'Locust':>8} | {'MLX':>8} | {'Ratio':>6} | {'Error':>8}")
    print("-" * 50)

    for sz in [64, 128, 256, 512]:
        err = test(sz)
        ms = bench(sz)

        # MLX
        ma = mx.random.normal((sz, sz)); mb = mx.random.normal((sz, sz)); mx.eval(ma, mb)
        for _ in range(3):
            mc = ma @ mb; mx.eval(mc)
        mlx_t = []
        for _ in range(10):
            t0 = time.perf_counter()
            mc = ma @ mb; mx.eval(mc)
            mlx_t.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mlx_t)[5]

        ratio = ms / t_mlx
        print(f"  {sz:>3}×{sz} | {ms:>6.3f}ms | {t_mlx:>6.3f}ms | {ratio:>5.2f}× | {err:.2e}")
