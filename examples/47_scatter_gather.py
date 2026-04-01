"""
Example 47: Scatter / Gather — index-based tensor operations.

Gather: select rows by index → embedding-style lookup.
  out[i] = src[indices[i]]  (along first axis, copy full rows)

Scatter Add: accumulate values into destination by index.
  dst[indices[i]] += src[i]  (gradient of gather, used in backprop)

Core primitives for: embedding gradients, sparse ops, index_select,
scatter-reduce in GNNs, dynamic batching.
"""

import time
import numpy as np
import locomp


# =============================================================================
# Gather: out[i, :] = src[indices[i], :]
# =============================================================================

@locomp.kernel
def gather_rows(SRC: locomp.Tensor, IDX: locomp.Tensor, OUT: locomp.Tensor,
                D: locomp.constexpr):
    i = locomp.program_id(0)   # output row
    tid = locomp.local_id(0)   # element within row

    idx_f = locomp.load(IDX + i)  # index as float
    src_base = idx_f * D
    out_base = i * D

    val = locomp.load(SRC + (src_base + tid))
    locomp.store(OUT + (out_base + tid), val)


# =============================================================================
# Scatter Add: dst[indices[i], :] += src[i, :]  (atomic for safety)
# =============================================================================

@locomp.kernel
def scatter_add_rows(SRC: locomp.Tensor, IDX: locomp.Tensor, DST: locomp.Tensor,
                     D: locomp.constexpr):
    i = locomp.program_id(0)   # source row
    tid = locomp.local_id(0)   # element within row

    idx_f = locomp.load(IDX + i)
    dst_base = idx_f * D
    src_base = i * D

    val = locomp.load(SRC + (src_base + tid))
    locomp.atomic_add(DST + (dst_base + tid), val)


# =============================================================================
# 1D Gather: out[i] = src[indices[i]]  (element-wise)
# =============================================================================

@locomp.kernel
def gather_1d(SRC: locomp.Tensor, IDX: locomp.Tensor, OUT: locomp.Tensor):
    i = locomp.program_id(0)
    idx_f = locomp.load(IDX + i)
    val = locomp.load(SRC + idx_f)
    locomp.store(OUT + i, val)


# =============================================================================
# Dispatch helpers
# =============================================================================

def gpu_gather_rows(src, indices):
    """Gather rows: src[V, D], indices[N] → out[N, D]."""
    V, D = src.shape
    N = indices.shape[0]
    S = locomp.tensor(src.flatten())
    I = locomp.tensor(indices.astype(np.float32))
    O = locomp.empty(N * D)
    if D <= 1024:
        gather_rows[(N,), (D,)](S, I, O, D)
    else:
        # For large D, use multiple threads per element
        gather_rows[(N,), (D,)](S, I, O, D)
    result = O.numpy().reshape(N, D)
    S.free(); I.free(); O.free()
    return result


def gpu_scatter_add(src, indices, dst_size, D):
    """Scatter add: src[N, D], indices[N] → dst[dst_size, D] (accumulated)."""
    N = src.shape[0]
    S = locomp.tensor(src.flatten())
    I = locomp.tensor(indices.astype(np.float32))
    O = locomp.zeros(dst_size * D)
    scatter_add_rows[(N,), (D,)](S, I, O, D)
    result = O.numpy().reshape(dst_size, D)
    S.free(); I.free(); O.free()
    return result


def gpu_gather_1d(src, indices):
    """1D gather: src[V], indices[N] → out[N]."""
    N = indices.shape[0]
    S = locomp.tensor(src.flatten())
    I = locomp.tensor(indices.astype(np.float32))
    O = locomp.empty(N)
    gather_1d[(N,)](S, I, O)
    result = O.numpy()
    S.free(); I.free(); O.free()
    return result


if __name__ == "__main__":
    np.random.seed(42)

    print("=== Gather Rows ===")
    for V, D, N in [(1000, 64, 128), (5000, 128, 256), (10000, 256, 512)]:
        src = np.random.randn(V, D).astype(np.float32)
        indices = np.random.randint(0, V, size=N).astype(np.int32)
        out = gpu_gather_rows(src, indices)
        expected = src[indices]
        np.testing.assert_allclose(out, expected, atol=1e-5)
        print(f"  [{V}×{D}] gather {N} rows ✓")

    print("\n=== 1D Gather ===")
    for V, N in [(1000, 256), (10000, 1024)]:
        src = np.random.randn(V).astype(np.float32)
        indices = np.random.randint(0, V, size=N).astype(np.int32)
        out = gpu_gather_1d(src, indices)
        expected = src[indices]
        np.testing.assert_allclose(out, expected, atol=1e-5)
        print(f"  [{V}] gather {N} elements ✓")

    print("\n=== Scatter Add ===")
    for V, D, N in [(100, 32, 256), (500, 64, 512)]:
        src = np.random.randn(N, D).astype(np.float32)
        indices = np.random.randint(0, V, size=N).astype(np.int32)
        out = gpu_scatter_add(src, indices, V, D)
        expected = np.zeros((V, D), dtype=np.float32)
        for i in range(N):
            expected[indices[i]] += src[i]
        np.testing.assert_allclose(out, expected, atol=1e-4)
        print(f"  [{N}×{D}] scatter into [{V}×{D}] ✓")

    print("\nAll scatter/gather tests passed.")
