"""
Example: Tiled Matrix Multiplication with Shared Memory.

This is the KEY optimization that closes the gap with MLX/PyTorch.
Instead of each thread reading K values from global memory,
we load TILE×TILE blocks into shared memory cooperatively.

Global memory reads drop from O(M*N*K) to O(M*N*K/TILE).
For K=512, TILE=16: 32× fewer global memory accesses.

Demonstrates: shared_memory, shared_load, shared_store, barrier,
              2D threadgroups, 2D grid, cooperative loading.
"""

import locomp
import numpy as np


TILE = 16  # Tile size — each threadgroup handles TILE×TILE output block


@locomp.kernel
def tiled_matmul(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                 M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr,
                 NUM_TILES: locomp.constexpr, BLOCK: locomp.constexpr):
    # Thread position within the BLOCK×BLOCK threadgroup
    row = locomp.local_id(1)   # 0..BLOCK-1
    col = locomp.local_id(0)   # 0..BLOCK-1

    # Which tile of the output matrix this threadgroup computes
    brow = locomp.program_id(1)  # tile row
    bcol = locomp.program_id(0)  # tile col

    # Shared memory for A and B tiles (BLOCK*BLOCK each)
    As = locomp.shared_memory(TILE * TILE)
    Bs = locomp.shared_memory(TILE * TILE)

    # Accumulator for this thread's output element
    acc = 0.0

    # Slide along K dimension in steps of BLOCK
    for t in range(NUM_TILES):
        # Cooperatively load A tile: A[brow*BLOCK+row, t*BLOCK+col]
        a_row = brow * BLOCK + row
        a_col = t * BLOCK + col
        a_val = locomp.load(A + (a_row * K + a_col))
        locomp.shared_store(As, row * BLOCK + col, a_val)

        # Cooperatively load B tile: B[t*BLOCK+row, bcol*BLOCK+col]
        b_row = t * BLOCK + row
        b_col = bcol * BLOCK + col
        b_val = locomp.load(B + (b_row * N + b_col))
        locomp.shared_store(Bs, row * BLOCK + col, b_val)

        # Wait for all threads to finish loading
        locomp.barrier()

        # Each thread computes partial dot product from shared memory
        for k in range(BLOCK):
            a_shared = locomp.shared_load(As, row * BLOCK + k)
            b_shared = locomp.shared_load(Bs, k * BLOCK + col)
            acc = acc + a_shared * b_shared

        # Wait before next tile load overwrites shared memory
        locomp.barrier()

    # Write result to global memory
    out_idx = (brow * BLOCK + row) * N + (bcol * BLOCK + col)
    locomp.store(C + out_idx, acc)


def main():
    print("=" * 60)
    print("TILED MATMUL (shared memory optimization)")
    print("=" * 60)

    M, N, K = 128, 128, 128

    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)

    a = locomp.tensor(a_np.flatten())
    b = locomp.tensor(b_np.flatten())
    c = locomp.empty(M * N)

    # 2D grid: (N/TILE, M/TILE) threadgroups, each (TILE, TILE) threads
    num_tiles = K // TILE
    grid = (N // TILE, M // TILE)
    tg = (TILE, TILE)
    tiled_matmul[grid, tg](a, b, c, M, N, K, num_tiles, TILE)

    result = c.numpy().reshape(M, N)
    expected = a_np @ b_np

    max_error = np.max(np.abs(result - expected))
    print(f"Size: {M}×{K} @ {K}×{N}, TILE={TILE}")
    print(f"Max error vs NumPy: {max_error}")
    print(f"Result[0,:4]: {result[0, :4]}")
    print(f"Expected[0,:4]: {expected[0, :4]}")

    # Quick timing
    import time
    times = []
    for _ in range(10):
        c2 = locomp.empty(M * N)
        t0 = time.perf_counter()
        tiled_matmul[grid, tg](a, b, c2, M, N, K, num_tiles, TILE)
        times.append((time.perf_counter() - t0) * 1000)
    median = sorted(times)[5]
    print(f"Median time (10 runs): {median:.3f} ms")


if __name__ == "__main__":
    main()
