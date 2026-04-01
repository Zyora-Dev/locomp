"""
Example 44: Transpose / Permute — layout changes for tensor operations.

Supports 2D transpose and batched transpose (permute dim 1↔2).
Core primitive: attention needs [B,H,N,D] ↔ [B,N,H,D] reshuffles,
and matmul needs [M,K] → [K,M] transposes.

Architecture:
  2D: one thread per element. Grid = (M, N).
  Batched: one thread per element. Grid = (N*D, B*H).
"""

import time
import numpy as np
import locomp


# =============================================================================
# 2D Transpose: [M, N] → [N, M]
# =============================================================================

@locomp.kernel
def transpose_2d(X: locomp.Tensor, OUT: locomp.Tensor,
                 M: locomp.constexpr, N: locomp.constexpr):
    row = locomp.program_id(0)   # output row (0..N-1)
    col = locomp.program_id(1)   # output col (0..M-1)
    # OUT[row, col] = X[col, row]
    val = locomp.load(X + (col * N + row))
    locomp.store(OUT + (row * M + col), val)


# =============================================================================
# Batched transpose: [B, H, N, D] → [B, N, H, D] (swap dims 1 and 2)
# =============================================================================

@locomp.kernel
def permute_0213(X: locomp.Tensor, OUT: locomp.Tensor,
                 H: locomp.constexpr, N: locomp.constexpr,
                 D: locomp.constexpr):
    # Grid: program_id(0) = linear index within (N, D), program_id(1) = b*H + h
    nd = locomp.program_id(0)
    bh = locomp.program_id(1)

    B_stride = H * N * D
    b = bh // H
    h = bh % H
    n = nd // D
    d = nd % D

    # src: X[b, h, n, d] = X[b*H*N*D + h*N*D + n*D + d]
    src_idx = b * B_stride + h * N * D + n * D + d
    # dst: OUT[b, n, h, d] = OUT[b*N*H*D + n*H*D + h*D + d]
    dst_idx = b * N * H * D + n * H * D + h * D + d

    val = locomp.load(X + src_idx)
    locomp.store(OUT + dst_idx, val)


# =============================================================================
# Dispatch helpers
# =============================================================================

def gpu_transpose_2d(x):
    """Transpose [M, N] → [N, M]."""
    M, N = x.shape
    X_g = locomp.tensor(x.flatten())
    O_g = locomp.empty(M * N)
    transpose_2d[(N, M)](X_g, O_g, M, N)
    result = O_g.numpy().reshape(N, M)
    X_g.free()
    O_g.free()
    return result


def gpu_permute_0213(x):
    """Permute [B, H, N, D] → [B, N, H, D]."""
    B, H, N, D = x.shape
    X_g = locomp.tensor(x.flatten())
    O_g = locomp.empty(B * H * N * D)
    permute_0213[(N * D, B * H)](X_g, O_g, H, N, D)
    result = O_g.numpy().reshape(B, N, H, D)
    X_g.free()
    O_g.free()
    return result


if __name__ == "__main__":
    print("=== 2D Transpose ===")
    for M, N in [(64, 128), (256, 512), (1024, 1024)]:
        x = np.random.randn(M, N).astype(np.float32)
        out = gpu_transpose_2d(x)
        np.testing.assert_allclose(out, x.T, atol=1e-6)
        print(f"  [{M}×{N}] → [{N}×{M}] ✓")

    print("\n=== Batched Permute [B,H,N,D] → [B,N,H,D] ===")
    for B, H, N, D in [(1, 8, 64, 64), (2, 32, 128, 128), (4, 8, 256, 64)]:
        x = np.random.randn(B, H, N, D).astype(np.float32)
        out = gpu_permute_0213(x)
        expected = x.transpose(0, 2, 1, 3)
        np.testing.assert_allclose(out, expected, atol=1e-6)
        print(f"  [{B},{H},{N},{D}] → [{B},{N},{H},{D}] ✓")

    print("\nAll transpose/permute tests passed.")
