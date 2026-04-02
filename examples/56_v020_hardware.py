"""Smoke test for Float16, 2D grid, and reductions on hardware."""
import numpy as np
import locomp

# ── Float16 ───────────────────────────────────────────────────────────────────

@locomp.kernel
def f16_add(X: locomp.Float16, Y: locomp.Float16, O: locomp.Float16, N: locomp.constexpr):
    i = locomp.program_id(0)
    x = locomp.load(X + i)
    y = locomp.load(Y + i)
    locomp.store(O + i, x + y)

@locomp.kernel
def f16_scale(X: locomp.Float16, O: locomp.Float16, scale: locomp.constexpr, N: locomp.constexpr):
    i = locomp.program_id(0)
    x = locomp.load(X + i)
    locomp.store(O + i, x * scale)

# ── 2D grid ───────────────────────────────────────────────────────────────────

@locomp.kernel
def mat_add(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
            M: locomp.constexpr, N: locomp.constexpr):
    row = locomp.program_id(0)
    col = locomp.program_id(1)
    idx = row * N + col
    a = locomp.load(A + idx)
    b = locomp.load(B + idx)
    locomp.store(C + idx, a + b)

# ── Reductions ────────────────────────────────────────────────────────────────

@locomp.kernel
def global_sum(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    val = locomp.load(X + i)
    locomp.reduce_sum(val, Out)

@locomp.kernel
def global_max(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    val = locomp.load(X + i)
    locomp.reduce_max(val, Out)

@locomp.kernel
def global_min(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    val = locomp.load(X + i)
    locomp.reduce_min(val, Out)

# ── Tests ─────────────────────────────────────────────────────────────────────

def test_float16_add():
    N = 128
    a = np.random.randn(N).astype(np.float16)
    b = np.random.randn(N).astype(np.float16)
    expected = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16).astype(np.float32)
    x = locomp.tensor(a); y = locomp.tensor(b); o = locomp.empty(N, dtype=np.float16)
    f16_add[(N,)](x, y, o, N=N)
    result = o.numpy().astype(np.float32)
    assert np.allclose(result, expected, rtol=1e-2, atol=1e-3), \
        f"Float16 add FAILED max_err={np.max(np.abs(result - expected)):.4f}"
    print(f"  [PASS]  f16_add          N={N}  max_err={np.max(np.abs(result - expected)):.4f}")


def test_float16_scale():
    N = 64
    scale = 3.0
    a = np.random.randn(N).astype(np.float16)
    expected = (a.astype(np.float32) * scale).astype(np.float16).astype(np.float32)
    x = locomp.tensor(a); o = locomp.empty(N, dtype=np.float16)
    f16_scale[(N,)](x, o, scale=scale, N=N)
    result = o.numpy().astype(np.float32)
    assert np.allclose(result, expected, rtol=1e-2, atol=1e-3), \
        f"Float16 scale FAILED max_err={np.max(np.abs(result - expected)):.4f}"
    print(f"  [PASS]  f16_scale         N={N}  scale={scale}  max_err={np.max(np.abs(result - expected)):.4f}")


def test_2d_mat_add():
    M, N = 16, 32
    a_np = np.random.randn(M, N).astype(np.float32)
    b_np = np.random.randn(M, N).astype(np.float32)
    expected = a_np + b_np
    A = locomp.tensor(a_np.flatten()); B = locomp.tensor(b_np.flatten())
    C = locomp.empty(M * N)
    mat_add[(M, N)](A, B, C, M=M, N=N)
    result = C.numpy().reshape(M, N)
    assert np.allclose(result, expected, rtol=1e-5, atol=1e-5), \
        f"2D mat_add FAILED max_err={np.max(np.abs(result - expected)):.6f}"
    print(f"  [PASS]  2d_mat_add        M={M} N={N}  max_err={np.max(np.abs(result - expected)):.6f}")


def test_reduce_sum():
    N = 1024
    a = np.random.randn(N).astype(np.float32)
    expected = float(np.sum(a))
    X = locomp.tensor(a)
    out = locomp.zeros(1)  # initialized to 0
    global_sum[(N,)](X, out, N=N)
    result = float(out.numpy()[0])
    # Parallel reduction: allow ~0.1% relative error
    rel_err = abs(result - expected) / (abs(expected) + 1e-6)
    assert rel_err < 0.001, f"reduce_sum FAILED expected={expected:.4f} got={result:.4f} rel_err={rel_err:.4e}"
    print(f"  [PASS]  reduce_sum        N={N}  expected={expected:.4f}  got={result:.4f}  rel_err={rel_err:.4e}")


def test_reduce_max():
    N = 512
    # Use positive values so the CAS float trick works correctly
    a = np.abs(np.random.randn(N).astype(np.float32))
    expected = float(np.max(a))
    # Initialize accumulator to 0 (all values positive)
    init = np.zeros(1, dtype=np.float32)
    X = locomp.tensor(a)
    out = locomp.tensor(init)
    global_max[(N,)](X, out, N=N)
    result = float(out.numpy()[0])
    assert abs(result - expected) < 1e-4, \
        f"reduce_max FAILED expected={expected:.4f} got={result:.4f}"
    print(f"  [PASS]  reduce_max        N={N}  expected={expected:.4f}  got={result:.4f}")


def test_reduce_min():
    N = 512
    # Use positive values so the CAS float trick works correctly
    a = np.abs(np.random.randn(N).astype(np.float32)) + 0.1
    expected = float(np.min(a))
    # Initialize accumulator to a large value (larger than all elements)
    init = np.array([1e30], dtype=np.float32)
    X = locomp.tensor(a)
    out = locomp.tensor(init)
    global_min[(N,)](X, out, N=N)
    result = float(out.numpy()[0])
    assert abs(result - expected) < 1e-4, \
        f"reduce_min FAILED expected={expected:.4f} got={result:.4f}"
    print(f"  [PASS]  reduce_min        N={N}  expected={expected:.4f}  got={result:.4f}")


if __name__ == "__main__":
    print("v0.2.0 hardware validation: Float16 + 2D Grid + Reductions")
    print("-" * 55)
    tests = [
        test_float16_add,
        test_float16_scale,
        test_2d_mat_add,
        test_reduce_sum,
        test_reduce_max,
        test_reduce_min,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL]  {t.__name__}: {e}")
    print("-" * 55)
    print(f"  {passed}/{len(tests)} passed")
    if passed < len(tests):
        raise SystemExit(1)
    print("  All v0.2.0 features validated on Apple Silicon hardware")
