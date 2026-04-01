"""
Example 48: Concat / Split — tensor assembly and disassembly.

Concat: join tensors along a given axis.
Split: slice a tensor into equal-sized chunks.

Used in: multi-head attention (concat heads), tensor parallelism
(split/gather across GPUs), skip connections, decoder outputs.

Architecture:
  Concat: one thread per output element, reads from correct source.
  Split: one thread per output element per chunk.
"""

import time
import numpy as np
import locomp


# =============================================================================
# Concat two tensors along last axis: [ROWS, D1] + [ROWS, D2] → [ROWS, D1+D2]
# =============================================================================

@locomp.kernel
def concat_last_axis(A: locomp.Tensor, B: locomp.Tensor, OUT: locomp.Tensor,
                     D1: locomp.constexpr, D2: locomp.constexpr,
                     D_OUT: locomp.constexpr):
    row = locomp.program_id(0)   # row index
    col = locomp.program_id(1)   # column in output

    out_idx = row * D_OUT + col

    if col < D1:
        val = locomp.load(A + (row * D1 + col))
        locomp.store(OUT + out_idx, val)
    else:
        val = locomp.load(B + (row * D2 + (col - D1)))
        locomp.store(OUT + out_idx, val)


# =============================================================================
# Split along last axis: [ROWS, D] → [ROWS, CHUNK] × NUM_CHUNKS
# Each output chunk is stored contiguously.
# =============================================================================

@locomp.kernel
def split_last_axis(X: locomp.Tensor, OUT: locomp.Tensor,
                    D: locomp.constexpr, CHUNK: locomp.constexpr,
                    CHUNK_IDX: locomp.constexpr):
    row = locomp.program_id(0)
    col = locomp.program_id(1)   # column within chunk

    src_idx = row * D + CHUNK_IDX * CHUNK + col
    dst_idx = row * CHUNK + col

    val = locomp.load(X + src_idx)
    locomp.store(OUT + dst_idx, val)


# =============================================================================
# Concat along first axis: [N1, D] + [N2, D] → [N1+N2, D]
# =============================================================================

@locomp.kernel
def concat_first_axis(A: locomp.Tensor, B: locomp.Tensor, OUT: locomp.Tensor,
                      N1: locomp.constexpr, D: locomp.constexpr):
    row = locomp.program_id(0)   # output row
    col = locomp.program_id(1)   # column

    out_idx = row * D + col

    if row < N1:
        val = locomp.load(A + (row * D + col))
        locomp.store(OUT + out_idx, val)
    else:
        val = locomp.load(B + ((row - N1) * D + col))
        locomp.store(OUT + out_idx, val)


# =============================================================================
# Dispatch helpers
# =============================================================================

def gpu_concat_last(a, b):
    """Concat [ROWS, D1] + [ROWS, D2] → [ROWS, D1+D2]."""
    ROWS = a.shape[0]
    D1, D2 = a.shape[1], b.shape[1]
    D_OUT = D1 + D2
    A_g = locomp.tensor(a.flatten())
    B_g = locomp.tensor(b.flatten())
    O_g = locomp.empty(ROWS * D_OUT)
    concat_last_axis[(ROWS, D_OUT)](A_g, B_g, O_g, D1, D2, D_OUT)
    result = O_g.numpy().reshape(ROWS, D_OUT)
    A_g.free(); B_g.free(); O_g.free()
    return result


def gpu_split_last(x, num_chunks):
    """Split [ROWS, D] → list of [ROWS, D//num_chunks]."""
    ROWS, D = x.shape
    CHUNK = D // num_chunks
    X_g = locomp.tensor(x.flatten())
    results = []
    for c in range(num_chunks):
        O_g = locomp.empty(ROWS * CHUNK)
        split_last_axis[(ROWS, CHUNK)](X_g, O_g, D, CHUNK, c)
        results.append(O_g.numpy().reshape(ROWS, CHUNK))
        O_g.free()
    X_g.free()
    return results


def gpu_concat_first(a, b):
    """Concat [N1, D] + [N2, D] → [N1+N2, D]."""
    N1, D = a.shape
    N2 = b.shape[0]
    A_g = locomp.tensor(a.flatten())
    B_g = locomp.tensor(b.flatten())
    O_g = locomp.empty((N1 + N2) * D)
    concat_first_axis[(N1 + N2, D)](A_g, B_g, O_g, N1, D)
    result = O_g.numpy().reshape(N1 + N2, D)
    A_g.free(); B_g.free(); O_g.free()
    return result


if __name__ == "__main__":
    np.random.seed(42)

    print("=== Concat Last Axis ===")
    for ROWS, D1, D2 in [(32, 64, 64), (64, 128, 256), (128, 512, 512)]:
        a = np.random.randn(ROWS, D1).astype(np.float32)
        b = np.random.randn(ROWS, D2).astype(np.float32)
        out = gpu_concat_last(a, b)
        expected = np.concatenate([a, b], axis=1)
        np.testing.assert_allclose(out, expected, atol=1e-6)
        print(f"  [{ROWS}×{D1}] + [{ROWS}×{D2}] → [{ROWS}×{D1+D2}] ✓")

    print("\n=== Split Last Axis ===")
    for ROWS, D, NC in [(32, 128, 2), (64, 256, 4), (16, 512, 8)]:
        x = np.random.randn(ROWS, D).astype(np.float32)
        chunks = gpu_split_last(x, NC)
        expected = np.split(x, NC, axis=1)
        for i in range(NC):
            np.testing.assert_allclose(chunks[i], expected[i], atol=1e-6)
        print(f"  [{ROWS}×{D}] → {NC}×[{ROWS}×{D//NC}] ✓")

    print("\n=== Concat First Axis ===")
    for N1, N2, D in [(32, 64, 128), (128, 128, 256)]:
        a = np.random.randn(N1, D).astype(np.float32)
        b = np.random.randn(N2, D).astype(np.float32)
        out = gpu_concat_first(a, b)
        expected = np.concatenate([a, b], axis=0)
        np.testing.assert_allclose(out, expected, atol=1e-6)
        print(f"  [{N1}×{D}] + [{N2}×{D}] → [{N1+N2}×{D}] ✓")

    print("\nAll concat/split tests passed.")
