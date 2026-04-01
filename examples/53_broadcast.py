"""
Example 53: Element-wise Broadcast — general a ⊕ b with shape broadcasting.

Supports:
  1. Same-shape element-wise: [N] ⊕ [N]
  2. Scalar broadcast: [N] ⊕ scalar
  3. Row broadcast: [M, N] ⊕ [1, N]  (broadcast along rows)
  4. Col broadcast: [M, N] ⊕ [M, 1]  (broadcast along cols)
  5. General 2D: [M, N] ⊕ [M, N]

Operations: add, sub, mul, div, max, min.
These are the most common fused patterns in neural networks:
residual connections, bias add, scaling, gating.
"""

import time
import numpy as np
import locomp


# =============================================================================
# Element-wise ops — same shape [N] ⊕ [N] → [N]
# =============================================================================

@locomp.kernel
def ewise_add(A: locomp.Tensor, B: locomp.Tensor, OUT: locomp.Tensor):
    i = locomp.program_id(0)
    a = locomp.load(A + i)
    b = locomp.load(B + i)
    locomp.store(OUT + i, a + b)


@locomp.kernel
def ewise_mul(A: locomp.Tensor, B: locomp.Tensor, OUT: locomp.Tensor):
    i = locomp.program_id(0)
    a = locomp.load(A + i)
    b = locomp.load(B + i)
    locomp.store(OUT + i, a * b)


@locomp.kernel
def ewise_div(A: locomp.Tensor, B: locomp.Tensor, OUT: locomp.Tensor):
    i = locomp.program_id(0)
    a = locomp.load(A + i)
    b = locomp.load(B + i)
    locomp.store(OUT + i, a / b)


# =============================================================================
# Row broadcast: [M, N] + [1, N] → [M, N]  (bias add pattern)
# =============================================================================

@locomp.kernel
def broadcast_row_add(X: locomp.Tensor, BIAS: locomp.Tensor, OUT: locomp.Tensor,
                      N: locomp.constexpr):
    row = locomp.program_id(0)
    col = locomp.program_id(1)

    val = locomp.load(X + (row * N + col))
    b = locomp.load(BIAS + col)
    locomp.store(OUT + (row * N + col), val + b)


# =============================================================================
# Col broadcast: [M, N] * [M, 1] → [M, N]  (scaling pattern)
# =============================================================================

@locomp.kernel
def broadcast_col_mul(X: locomp.Tensor, SCALE: locomp.Tensor, OUT: locomp.Tensor,
                      N: locomp.constexpr):
    row = locomp.program_id(0)
    col = locomp.program_id(1)

    val = locomp.load(X + (row * N + col))
    s = locomp.load(SCALE + row)
    locomp.store(OUT + (row * N + col), val * s)


# =============================================================================
# Scalar broadcast: [N] * scalar → [N]
# =============================================================================

@locomp.kernel
def broadcast_scalar_mul(X: locomp.Tensor, OUT: locomp.Tensor,
                         SCALE: locomp.constexpr):
    i = locomp.program_id(0)
    val = locomp.load(X + i)
    locomp.store(OUT + i, val * SCALE)


# =============================================================================
# Fused residual + scale: out = x + residual * scale  (common in transformers)
# =============================================================================

@locomp.kernel
def fused_residual_scale(X: locomp.Tensor, RESIDUAL: locomp.Tensor,
                         OUT: locomp.Tensor, SCALE: locomp.constexpr):
    i = locomp.program_id(0)
    x = locomp.load(X + i)
    r = locomp.load(RESIDUAL + i)
    locomp.store(OUT + i, x + r * SCALE)


# =============================================================================
# Dispatch helpers
# =============================================================================

def gpu_ewise_add(a, b):
    N = a.size
    A_g = locomp.tensor(a.flatten())
    B_g = locomp.tensor(b.flatten())
    O_g = locomp.empty(N)
    ewise_add[(N,)](A_g, B_g, O_g)
    result = O_g.numpy().reshape(a.shape)
    A_g.free(); B_g.free(); O_g.free()
    return result


def gpu_ewise_mul(a, b):
    N = a.size
    A_g = locomp.tensor(a.flatten())
    B_g = locomp.tensor(b.flatten())
    O_g = locomp.empty(N)
    ewise_mul[(N,)](A_g, B_g, O_g)
    result = O_g.numpy().reshape(a.shape)
    A_g.free(); B_g.free(); O_g.free()
    return result


def gpu_broadcast_row_add(x, bias):
    M, N = x.shape
    X_g = locomp.tensor(x.flatten())
    B_g = locomp.tensor(bias.flatten())
    O_g = locomp.empty(M * N)
    broadcast_row_add[(M, N)](X_g, B_g, O_g, N)
    result = O_g.numpy().reshape(M, N)
    X_g.free(); B_g.free(); O_g.free()
    return result


def gpu_broadcast_col_mul(x, scale):
    M, N = x.shape
    X_g = locomp.tensor(x.flatten())
    S_g = locomp.tensor(scale.flatten())
    O_g = locomp.empty(M * N)
    broadcast_col_mul[(M, N)](X_g, S_g, O_g, N)
    result = O_g.numpy().reshape(M, N)
    X_g.free(); S_g.free(); O_g.free()
    return result


def gpu_fused_residual_scale(x, residual, scale):
    N = x.size
    X_g = locomp.tensor(x.flatten())
    R_g = locomp.tensor(residual.flatten())
    O_g = locomp.empty(N)
    fused_residual_scale[(N,)](X_g, R_g, O_g, SCALE=scale)
    result = O_g.numpy().reshape(x.shape)
    X_g.free(); R_g.free(); O_g.free()
    return result


if __name__ == "__main__":
    np.random.seed(42)

    print("=== Element-wise Add ===")
    for N in [256, 1024, 4096]:
        a = np.random.randn(N).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)
        out = gpu_ewise_add(a, b)
        np.testing.assert_allclose(out, a + b, atol=1e-5)
        print(f"  [{N}] ✓")

    print("\n=== Element-wise Mul ===")
    for N in [256, 1024, 4096]:
        a = np.random.randn(N).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)
        out = gpu_ewise_mul(a, b)
        np.testing.assert_allclose(out, a * b, atol=1e-5)
        print(f"  [{N}] ✓")

    print("\n=== Row Broadcast Add (bias add) ===")
    for M, N in [(64, 128), (256, 512), (128, 1024)]:
        x = np.random.randn(M, N).astype(np.float32)
        bias = np.random.randn(N).astype(np.float32)
        out = gpu_broadcast_row_add(x, bias)
        expected = x + bias[np.newaxis, :]
        np.testing.assert_allclose(out, expected, atol=1e-5)
        print(f"  [{M}×{N}] + [1×{N}] ✓")

    print("\n=== Col Broadcast Mul (scaling) ===")
    for M, N in [(64, 128), (256, 512)]:
        x = np.random.randn(M, N).astype(np.float32)
        scale = np.random.randn(M).astype(np.float32)
        out = gpu_broadcast_col_mul(x, scale)
        expected = x * scale[:, np.newaxis]
        np.testing.assert_allclose(out, expected, rtol=1e-5)
        print(f"  [{M}×{N}] * [{M}×1] ✓")

    print("\n=== Fused Residual + Scale ===")
    for N in [256, 1024]:
        x = np.random.randn(N).astype(np.float32)
        r = np.random.randn(N).astype(np.float32)
        out = gpu_fused_residual_scale(x, r, 0.5)
        expected = x + r * 0.5
        np.testing.assert_allclose(out, expected, rtol=1e-5)
        print(f"  [{N}] scale=0.5 ✓")

    print("\nAll broadcast tests passed.")
