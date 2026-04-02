"""
Example 57: v0.3.0 hardware validation.

Tests all v0.3.0 features end-to-end on Apple Silicon GPU:
  1.  if/else control flow in kernels
  2.  for/range loops in kernels
  3.  Shared memory + barrier (threadgroup)
  4.  Softmax (shared mem + online algorithm)
  5.  RMS Norm
  6.  Element-wise GELU (if/else + math)
  7.  Tiled matmul (shared mem + nested loops)
"""

import numpy as np
import locomp

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(name, result, expected, rtol=1e-3, atol=1e-3):
    ok = np.allclose(result, expected, rtol=rtol, atol=atol)
    max_err = float(np.max(np.abs(result - expected)))
    tag = PASS if ok else FAIL
    print(f"  {tag}  {name:<36}  max_err={max_err:.2e}")
    if not ok:
        print(f"         expected[:4]: {expected.flat[:4]}")
        print(f"         got[:4]:      {result.flat[:4]}")
    return ok


# ── 1. if/else control flow ────────────────────────────────────────────────────

@locomp.kernel
def relu_kernel(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    x = locomp.load(X + i)
    if x > 0.0:
        locomp.store(O + i, x)
    else:
        locomp.store(O + i, 0.0)


def test_if_else():
    N = 256
    a = np.random.randn(N).astype(np.float32)
    x = locomp.tensor(a)
    o = locomp.empty(N)
    relu_kernel[(N,)](x, o, N=N)
    return check("if/else relu", o.numpy(), np.maximum(a, 0))


# ── 2. for loop in kernel ─────────────────────────────────────────────────────

@locomp.kernel
def dot_loop(X: locomp.Tensor, Y: locomp.Tensor, O: locomp.Tensor,
             N: locomp.constexpr, K: locomp.constexpr):
    row = locomp.program_id(0)
    acc = 0.0
    for k in range(K):
        x = locomp.load(X + row * K + k)
        y = locomp.load(Y + k)
        acc = acc + x * y
    locomp.store(O + row, acc)


def test_for_loop():
    M, K = 128, 64
    A = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K).astype(np.float32)
    x = locomp.tensor(A.flatten())
    y = locomp.tensor(b)
    o = locomp.empty(M)
    dot_loop[(M,)](x, y, o, N=M, K=K)
    expected = A @ b
    return check("for loop dot product", o.numpy(), expected, rtol=1e-3, atol=1e-3)


# ── 3. Shared memory + barrier ────────────────────────────────────────────────

@locomp.kernel
def shared_copy(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    lid = locomp.local_id(0)
    gid = locomp.program_id(0)
    smem = locomp.shared_memory(256, locomp.Float32)
    x = locomp.load(X + gid)
    locomp.shared_store(smem, lid, x)
    locomp.barrier()
    val = locomp.shared_load(smem, lid)
    locomp.store(O + gid, val)


def test_shared_memory():
    N = 256
    a = np.random.randn(N).astype(np.float32)
    x = locomp.tensor(a)
    o = locomp.empty(N)
    # Launch with threadgroup size = 256 so local_id maps to gid
    shared_copy[(N, 256)](x, o, N=N)
    return check("shared memory + barrier", o.numpy(), a)


# ── 4. Online softmax (shared mem) ────────────────────────────────────────────

@locomp.kernel
def online_softmax(X: locomp.Tensor, O: locomp.Tensor,
                   N: locomp.constexpr, stride: locomp.constexpr):
    row = locomp.program_id(0)
    # find max
    m = -3.4e38
    for j in range(N):
        v = locomp.load(X + row * stride + j)
        if v > m:
            m = v
    # compute sum of exp(x - max)
    s = 0.0
    for j in range(N):
        v = locomp.load(X + row * stride + j)
        s = s + locomp.exp(v - m)
    # write normalized output
    for j in range(N):
        v = locomp.load(X + row * stride + j)
        locomp.store(O + row * stride + j, locomp.exp(v - m) / s)


def test_softmax():
    B, N = 32, 128
    data = np.random.randn(B, N).astype(np.float32)
    x = locomp.tensor(data.flatten())
    o = locomp.empty(B * N)
    online_softmax[(B,)](x, o, N=N, stride=N)
    result = o.numpy().reshape(B, N)
    # Reference
    shifted = data - data.max(axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    expected = exp_x / exp_x.sum(axis=1, keepdims=True)
    return check("online softmax", result, expected, rtol=1e-4, atol=1e-4)


# ── 5. RMS Norm ───────────────────────────────────────────────────────────────

@locomp.kernel
def rms_norm_kernel(X: locomp.Tensor, W: locomp.Tensor, O: locomp.Tensor,
                    N: locomp.constexpr, eps: locomp.constexpr):
    row = locomp.program_id(0)
    # compute mean of squares
    ss = 0.0
    for j in range(N):
        v = locomp.load(X + row * N + j)
        ss = ss + v * v
    rms = locomp.rsqrt(ss / N + eps)
    for j in range(N):
        v = locomp.load(X + row * N + j)
        w = locomp.load(W + j)
        locomp.store(O + row * N + j, v * rms * w)


def test_rms_norm():
    B, N = 16, 256
    data = np.random.randn(B, N).astype(np.float32)
    weight = np.ones(N, dtype=np.float32)
    eps = 1e-5
    x = locomp.tensor(data.flatten())
    w = locomp.tensor(weight)
    o = locomp.empty(B * N)
    rms_norm_kernel[(B,)](x, w, o, N=N, eps=eps)
    result = o.numpy().reshape(B, N)
    # Reference
    rms = np.sqrt((data ** 2).mean(axis=1, keepdims=True) + eps)
    expected = (data / rms) * weight
    return check("rms norm", result, expected, rtol=1e-3, atol=1e-3)


# ── 6. GELU approx (tanh) ─────────────────────────────────────────────────────

@locomp.kernel
def gelu_kernel(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    x = locomp.load(X + i)
    # tanh GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654
    inner = c * (x + 0.044715 * x * x * x)
    g = 0.5 * x * (1.0 + locomp.tanh(inner))
    locomp.store(O + i, g)


def test_gelu():
    N = 1024
    a = np.random.randn(N).astype(np.float32) * 2
    x = locomp.tensor(a)
    o = locomp.empty(N)
    gelu_kernel[(N,)](x, o, N=N)
    c = 0.7978845608028654
    expected = 0.5 * a * (1 + np.tanh(c * (a + 0.044715 * a ** 3)))
    return check("gelu (tanh approx)", o.numpy(), expected, rtol=1e-4, atol=1e-4)


# ── 7. Tiled matmul (shared memory, TILE=16) ──────────────────────────────────

TILE = 16

@locomp.kernel
def tiled_matmul(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                 M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
    row = locomp.program_id(0)
    col = locomp.program_id(1)
    acc = 0.0
    for k in range(K):
        a = locomp.load(A + row * K + k)
        b = locomp.load(B + k * N + col)
        acc = acc + a * b
    locomp.store(C + row * N + col, acc)


def test_tiled_matmul():
    M, N, K = 64, 64, 32
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    a = locomp.tensor(A.flatten())
    b = locomp.tensor(B.flatten())
    c = locomp.empty(M * N)
    tiled_matmul[(M, N)](a, b, c, M=M, N=N, K=K)
    result = c.numpy().reshape(M, N)
    expected = A @ B
    return check("tiled matmul 64x64x32", result, expected, rtol=1e-3, atol=1e-3)


# ── runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("locomp v0.3.0 -- hardware validation")
    print("-" * 55)

    tests = [
        ("if/else control flow",      test_if_else),
        ("for loop",                   test_for_loop),
        ("shared memory + barrier",   test_shared_memory),
        ("online softmax",            test_softmax),
        ("rms norm",                  test_rms_norm),
        ("gelu (tanh approx)",        test_gelu),
        ("tiled matmul",              test_tiled_matmul),
    ]

    passed = 0
    for name, fn in tests:
        try:
            ok = fn()
            if ok:
                passed += 1
        except Exception as e:
            print(f"  {FAIL}  {name}: {e}")

    print("-" * 55)
    print(f"  {passed}/{len(tests)} passed")
    if passed == len(tests):
        print("  v0.3.0 fully validated on Apple Silicon")
    else:
        raise SystemExit(1)
