"""
Example: Register Micro-Tiled Matmul.

Each thread computes a TM×TN = 2×2 block of the output matrix.
This means each threadgroup computes (TILE*TM) × (TILE*TN) output elements.

With TILE=16, TM=TN=2: each threadgroup handles 32×32 output block.
Each thread loads from shared memory, accumulates in 4 registers.

Arithmetic intensity: TM*TN = 4 FLOPs per shared memory read
vs 1 FLOP/read in the basic tiled version = 4× better register reuse.
"""

import locomp
import numpy as np

TILE = 16  # threads per dimension in threadgroup
TM = 2     # micro-tile rows per thread
TN = 2     # micro-tile cols per thread
BM = TILE * TM  # = 32, block rows
BN = TILE * TN  # = 32, block cols
BK = 16    # K-dimension tile size


@locomp.kernel
def micro_tiled_matmul(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                       M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr,
                       NUM_K_TILES: locomp.constexpr, BLOCK_K: locomp.constexpr,
                       BLOCK_M: locomp.constexpr, BLOCK_N: locomp.constexpr):
    # Thread position within 16×16 threadgroup
    trow = locomp.local_id(1)   # 0..15
    tcol = locomp.local_id(0)   # 0..15

    # Block position in grid
    brow = locomp.program_id(1)  # tile row
    bcol = locomp.program_id(0)  # tile col

    # Shared memory for A tile (BM × BK = 32×16) and B tile (BK × BN = 16×32)
    As = locomp.shared_memory(BM * BK)  # 512 floats
    Bs = locomp.shared_memory(BK * BN)  # 512 floats

    # 4 accumulators for 2×2 micro-tile
    acc00 = 0.0
    acc01 = 0.0
    acc10 = 0.0
    acc11 = 0.0

    # Slide along K dimension
    for t in range(NUM_K_TILES):
        # === Cooperatively load A tile (BM × BK = 32×16) ===
        # Each of 256 threads loads 2 elements (32*16/256 = 2)
        # Method: each thread loads A[brow*BM + trow*2 + r, t*BK + tcol] for r in 0,1
        a_base_row = brow * BLOCK_M + trow * 2
        a_base_col = t * BLOCK_K + tcol
        a_val0 = locomp.load(A + ((a_base_row + 0) * K + a_base_col))
        a_val1 = locomp.load(A + ((a_base_row + 1) * K + a_base_col))
        locomp.shared_store(As, (trow * 2 + 0) * BLOCK_K + tcol, a_val0)
        locomp.shared_store(As, (trow * 2 + 1) * BLOCK_K + tcol, a_val1)

        # === Cooperatively load B tile (BK × BN = 16×32) ===
        # Each of 256 threads loads 2 elements (16*32/256 = 2)
        b_base_row = t * BLOCK_K + trow
        b_base_col = bcol * BLOCK_N + tcol * 2
        b_val0 = locomp.load(B + (b_base_row * N + b_base_col + 0))
        b_val1 = locomp.load(B + (b_base_row * N + b_base_col + 1))
        locomp.shared_store(Bs, trow * BLOCK_N + tcol * 2 + 0, b_val0)
        locomp.shared_store(Bs, trow * BLOCK_N + tcol * 2 + 1, b_val1)

        locomp.barrier()

        # === Compute 2×2 micro-tile from shared memory ===
        for k in range(BLOCK_K):
            # Load A values for this thread's 2 rows
            a0 = locomp.shared_load(As, (trow * 2 + 0) * BLOCK_K + k)
            a1 = locomp.shared_load(As, (trow * 2 + 1) * BLOCK_K + k)

            # Load B values for this thread's 2 cols
            b0 = locomp.shared_load(Bs, k * BLOCK_N + tcol * 2 + 0)
            b1 = locomp.shared_load(Bs, k * BLOCK_N + tcol * 2 + 1)

            # 4 FMA operations
            acc00 = acc00 + a0 * b0
            acc01 = acc01 + a0 * b1
            acc10 = acc10 + a1 * b0
            acc11 = acc11 + a1 * b1

        locomp.barrier()

    # === Write 2×2 result block to global memory ===
    out_row = brow * BLOCK_M + trow * 2
    out_col = bcol * BLOCK_N + tcol * 2
    locomp.store(C + ((out_row + 0) * N + out_col + 0), acc00)
    locomp.store(C + ((out_row + 0) * N + out_col + 1), acc01)
    locomp.store(C + ((out_row + 1) * N + out_col + 0), acc10)
    locomp.store(C + ((out_row + 1) * N + out_col + 1), acc11)


def main():
    import time

    print("=" * 60)
    print("REGISTER MICRO-TILED MATMUL (2×2 per thread)")
    print("=" * 60)

    for size in [32, 64, 128, 256, 512]:
        M = N = K = size
        if size < BM:
            continue  # need at least 32

        num_k_tiles = K // BK

        a_np = np.random.randn(M, K).astype(np.float32)
        b_np = np.random.randn(K, N).astype(np.float32)
        a = locomp.tensor(a_np.flatten())
        b = locomp.tensor(b_np.flatten())
        c = locomp.empty(M * N)

        # Grid: (N/BN, M/BM) threadgroups, each (TILE, TILE) threads
        grid = (N // BN, M // BM)
        tg = (TILE, TILE)

        micro_tiled_matmul[grid, tg](a, b, c, M, N, K, num_k_tiles, BK, BM, BN)
        result = c.numpy().reshape(M, N)
        expected = a_np @ b_np
        err = np.max(np.abs(result - expected))

        # Timing
        for _ in range(3):
            c2 = locomp.empty(M * N)
            micro_tiled_matmul[grid, tg](a, b, c2, M, N, K, num_k_tiles, BK, BM, BN)
        times = []
        for _ in range(10):
            c2 = locomp.empty(M * N)
            t0 = time.perf_counter()
            micro_tiled_matmul[grid, tg](a, b, c2, M, N, K, num_k_tiles, BK, BM, BN)
            times.append((time.perf_counter() - t0) * 1000)
        med = sorted(times)[5]

        print(f"{size:>4}×{size} | {med:>7.3f}ms | err={err:.8f}")


if __name__ == "__main__":
    main()
