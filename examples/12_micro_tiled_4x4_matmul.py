"""
4×4 Register Micro-Tiled Matrix Multiply.

Each thread computes a 4×4 block of the output matrix.
Block: BM=64, BN=64, BK=16
Threadgroup: 16×16 = 256 threads
Shared memory: As[64×16], Bs[16×64]
Arithmetic intensity: 16 FMAs per k-step (vs 4 in 2×2)
"""

import time
import numpy as np
import locomp

TILE = 16   # threadgroup dim (16×16 = 256 threads)
TM = 4      # micro-tile rows per thread
TN = 4      # micro-tile cols per thread
BM = TILE * TM   # 64
BN = TILE * TN   # 64
BK = 16


@locomp.kernel
def matmul_4x4(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
               M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr,
               NUM_K_TILES: locomp.constexpr, BLOCK_K: locomp.constexpr,
               BLOCK_M: locomp.constexpr, BLOCK_N: locomp.constexpr):
    trow = locomp.local_id(1)
    tcol = locomp.local_id(0)
    brow = locomp.program_id(1)
    bcol = locomp.program_id(0)

    As = locomp.shared_memory(BM * BK)
    Bs = locomp.shared_memory(BK * BN)

    # 16 accumulators for 4×4 output tile
    acc00 = 0.0; acc01 = 0.0; acc02 = 0.0; acc03 = 0.0
    acc10 = 0.0; acc11 = 0.0; acc12 = 0.0; acc13 = 0.0
    acc20 = 0.0; acc21 = 0.0; acc22 = 0.0; acc23 = 0.0
    acc30 = 0.0; acc31 = 0.0; acc32 = 0.0; acc33 = 0.0

    for t in range(NUM_K_TILES):
        # Load A tile: thread (trow, tcol) loads 4 rows × 1 col
        # A[brow*BM + trow*4+i, t*BK + tcol]
        a_base_row = brow * BLOCK_M + trow * 4
        a_col = t * BLOCK_K + tcol
        a_val0 = locomp.load(A + ((a_base_row + 0) * K + a_col))
        a_val1 = locomp.load(A + ((a_base_row + 1) * K + a_col))
        a_val2 = locomp.load(A + ((a_base_row + 2) * K + a_col))
        a_val3 = locomp.load(A + ((a_base_row + 3) * K + a_col))
        locomp.shared_store(As, (trow * 4 + 0) * BLOCK_K + tcol, a_val0)
        locomp.shared_store(As, (trow * 4 + 1) * BLOCK_K + tcol, a_val1)
        locomp.shared_store(As, (trow * 4 + 2) * BLOCK_K + tcol, a_val2)
        locomp.shared_store(As, (trow * 4 + 3) * BLOCK_K + tcol, a_val3)

        # Load B tile: thread (trow, tcol) loads 1 row × 4 cols
        # B[t*BK + trow, bcol*BN + tcol*4+j]
        b_row = t * BLOCK_K + trow
        b_base_col = bcol * BLOCK_N + tcol * 4
        b_val0 = locomp.load(B + (b_row * N + b_base_col + 0))
        b_val1 = locomp.load(B + (b_row * N + b_base_col + 1))
        b_val2 = locomp.load(B + (b_row * N + b_base_col + 2))
        b_val3 = locomp.load(B + (b_row * N + b_base_col + 3))
        locomp.shared_store(Bs, trow * BLOCK_N + tcol * 4 + 0, b_val0)
        locomp.shared_store(Bs, trow * BLOCK_N + tcol * 4 + 1, b_val1)
        locomp.shared_store(Bs, trow * BLOCK_N + tcol * 4 + 2, b_val2)
        locomp.shared_store(Bs, trow * BLOCK_N + tcol * 4 + 3, b_val3)

        locomp.barrier()

        # Compute 4×4 outer product per k-step
        for k in range(BLOCK_K):
            # Load 4 A values from shared memory
            a0 = locomp.shared_load(As, (trow * 4 + 0) * BLOCK_K + k)
            a1 = locomp.shared_load(As, (trow * 4 + 1) * BLOCK_K + k)
            a2 = locomp.shared_load(As, (trow * 4 + 2) * BLOCK_K + k)
            a3 = locomp.shared_load(As, (trow * 4 + 3) * BLOCK_K + k)

            # Load 4 B values from shared memory
            b0 = locomp.shared_load(Bs, k * BLOCK_N + tcol * 4 + 0)
            b1 = locomp.shared_load(Bs, k * BLOCK_N + tcol * 4 + 1)
            b2 = locomp.shared_load(Bs, k * BLOCK_N + tcol * 4 + 2)
            b3 = locomp.shared_load(Bs, k * BLOCK_N + tcol * 4 + 3)

            # 16 FMAs
            acc00 = acc00 + a0 * b0; acc01 = acc01 + a0 * b1
            acc02 = acc02 + a0 * b2; acc03 = acc03 + a0 * b3
            acc10 = acc10 + a1 * b0; acc11 = acc11 + a1 * b1
            acc12 = acc12 + a1 * b2; acc13 = acc13 + a1 * b3
            acc20 = acc20 + a2 * b0; acc21 = acc21 + a2 * b1
            acc22 = acc22 + a2 * b2; acc23 = acc23 + a2 * b3
            acc30 = acc30 + a3 * b0; acc31 = acc31 + a3 * b1
            acc32 = acc32 + a3 * b2; acc33 = acc33 + a3 * b3

        locomp.barrier()

    # Store 4×4 output tile
    out_row = brow * BLOCK_M + trow * 4
    out_col = bcol * BLOCK_N + tcol * 4
    locomp.store(C + ((out_row + 0) * N + out_col + 0), acc00)
    locomp.store(C + ((out_row + 0) * N + out_col + 1), acc01)
    locomp.store(C + ((out_row + 0) * N + out_col + 2), acc02)
    locomp.store(C + ((out_row + 0) * N + out_col + 3), acc03)
    locomp.store(C + ((out_row + 1) * N + out_col + 0), acc10)
    locomp.store(C + ((out_row + 1) * N + out_col + 1), acc11)
    locomp.store(C + ((out_row + 1) * N + out_col + 2), acc12)
    locomp.store(C + ((out_row + 1) * N + out_col + 3), acc13)
    locomp.store(C + ((out_row + 2) * N + out_col + 0), acc20)
    locomp.store(C + ((out_row + 2) * N + out_col + 1), acc21)
    locomp.store(C + ((out_row + 2) * N + out_col + 2), acc22)
    locomp.store(C + ((out_row + 2) * N + out_col + 3), acc23)
    locomp.store(C + ((out_row + 3) * N + out_col + 0), acc30)
    locomp.store(C + ((out_row + 3) * N + out_col + 1), acc31)
    locomp.store(C + ((out_row + 3) * N + out_col + 2), acc32)
    locomp.store(C + ((out_row + 3) * N + out_col + 3), acc33)


def test(M, N, K):
    """Test correctness."""
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)
    expected = a_np @ b_np

    a = locomp.tensor(a_np.flatten())
    b = locomp.tensor(b_np.flatten())
    c = locomp.empty(M * N)

    nkt = K // BK
    grid = (N // BN, M // BM)
    tg = (TILE, TILE)

    matmul_4x4[grid, tg](a, b, c, M, N, K, nkt, BK, BM, BN)
    result = c.numpy().reshape(M, N)
    err = np.max(np.abs(result - expected))
    return err


def bench(M, N, K):
    """Benchmark."""
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)
    a = locomp.tensor(a_np.flatten())
    b = locomp.tensor(b_np.flatten())
    c = locomp.empty(M * N)

    nkt = K // BK
    grid = (N // BN, M // BM)
    tg = (TILE, TILE)

    WARMUP = 3
    RUNS = 10

    for _ in range(WARMUP):
        matmul_4x4[grid, tg](a, b, c, M, N, K, nkt, BK, BM, BN)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        matmul_4x4[grid, tg](a, b, c, M, N, K, nkt, BK, BM, BN)
        times.append((time.perf_counter() - t0) * 1000)

    return sorted(times)[5]


if __name__ == "__main__":
    print("4×4 Micro-Tiled Matrix Multiply")
    print(f"TM=TN={TM}, BM=BN={BM}, BK={BK}, Threadgroup=({TILE},{TILE})")
    print("-" * 55)

    # Need size >= 64 for BM=BN=64
    for sz in [64, 128, 256, 512]:
        err = test(sz, sz, sz)
        ms = bench(sz, sz, sz)
        print(f"  {sz:>4}×{sz}: {ms:>6.3f} ms  err={err:.2e}")
