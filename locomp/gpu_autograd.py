"""
locomp GPU Autograd — tape-based reverse-mode autodiff with GPU-backed tensors.

Forward and backward passes execute as real locomp kernels on Metal/CUDA/RISC-V.
Gradients stay on-device until .numpy() is called.

Supported ops:
    add, sub, mul, div, exp, log, relu, sum, mean, matmul (matvec)

Usage:
    ga = locomp.gpu_ag
    x = ga.tensor(np.random.randn(N), requires_grad=True)
    y = ga.tensor(np.random.randn(N), requires_grad=True)
    z = ga.add(x, y)
    loss = ga.sum(z)
    ga.backward(loss)
    print(x.grad.numpy())   # GPU gradient, read back to CPU
"""

import functools
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np
import locomp
from locomp.frontend import Tensor as _T, constexpr as _cx


# ─── Kernel registry — compiled lazily per backend ──────────────────────────

_kc: dict = {}   # (name, backend) → KernelLauncher


def _k(name: str, fn: Callable, backend: str) -> "locomp.KernelLauncher":
    key = (name, backend)
    if key not in _kc:
        _kc[key] = locomp.kernel(backend=backend)(fn)
    return _kc[key]


# ─── Helper: allocate zeros GPU tensor same shape as t ──────────────────────

def _zeros_like(t: "GPUTensor") -> "GPUTensor":
    n = t._locomp.size
    z = locomp.zeros(n, dtype=t._locomp.dtype, backend=t._backend)
    return GPUTensor(z, requires_grad=False, backend=t._backend)


# ─── Forward + backward kernel definitions ──────────────────────────────────

# ── element-wise add ────────────────────────────────────────────────────────

def _fwd_add(A: _T, B: _T, Out: _T,
             N: _cx):
    i = locomp.program_id(0)
    locomp.store(Out + i, locomp.load(A + i) + locomp.load(B + i))

def _bwd_add_accumulate(GradOut: _T, GradIn: _T,
                        N: _cx):
    """GradIn += GradOut (broadcast-sum handled by caller if shapes differ)"""
    i = locomp.program_id(0)
    locomp.store(GradIn + i, locomp.load(GradIn + i) + locomp.load(GradOut + i))

# ── element-wise sub ────────────────────────────────────────────────────────

def _fwd_sub(A: _T, B: _T, Out: _T,
             N: _cx):
    i = locomp.program_id(0)
    locomp.store(Out + i, locomp.load(A + i) - locomp.load(B + i))

def _bwd_sub_b(GradOut: _T, GradIn: _T,
               N: _cx):
    """GradIn += -GradOut"""
    i = locomp.program_id(0)
    locomp.store(GradIn + i, locomp.load(GradIn + i) - locomp.load(GradOut + i))

# ── element-wise mul ────────────────────────────────────────────────────────

def _fwd_mul(A: _T, B: _T, Out: _T,
             N: _cx):
    i = locomp.program_id(0)
    locomp.store(Out + i, locomp.load(A + i) * locomp.load(B + i))

def _bwd_mul(GradOut: _T, Other: _T, GradIn: _T,
             N: _cx):
    """GradIn += GradOut * Other"""
    i = locomp.program_id(0)
    g = locomp.load(GradOut + i) * locomp.load(Other + i)
    locomp.store(GradIn + i, locomp.load(GradIn + i) + g)

# ── element-wise div ────────────────────────────────────────────────────────

def _fwd_div(A: _T, B: _T, Out: _T,
             N: _cx):
    i = locomp.program_id(0)
    locomp.store(Out + i, locomp.load(A + i) / locomp.load(B + i))

def _bwd_div_a(GradOut: _T, B: _T, GradIn: _T,
               N: _cx):
    """GradIn += GradOut / B"""
    i = locomp.program_id(0)
    g = locomp.load(GradOut + i) / locomp.load(B + i)
    locomp.store(GradIn + i, locomp.load(GradIn + i) + g)

def _bwd_div_b(GradOut: _T, A: _T, B: _T,
               GradIn: _T, N: _cx):
    """GradIn += -GradOut * A / B^2"""
    i = locomp.program_id(0)
    a = locomp.load(A + i)
    b = locomp.load(B + i)
    g = -locomp.load(GradOut + i) * a / (b * b)
    locomp.store(GradIn + i, locomp.load(GradIn + i) + g)

# ── exp ─────────────────────────────────────────────────────────────────────

def _fwd_exp(A: _T, Out: _T, N: _cx):
    i = locomp.program_id(0)
    locomp.store(Out + i, locomp.exp(locomp.load(A + i)))

def _bwd_exp(GradOut: _T, FwdOut: _T, GradIn: _T,
             N: _cx):
    """GradIn += GradOut * exp(a)  (= GradOut * fwd_out)"""
    i = locomp.program_id(0)
    g = locomp.load(GradOut + i) * locomp.load(FwdOut + i)
    locomp.store(GradIn + i, locomp.load(GradIn + i) + g)

# ── log ─────────────────────────────────────────────────────────────────────

def _fwd_log(A: _T, Out: _T, N: _cx):
    i = locomp.program_id(0)
    locomp.store(Out + i, locomp.log(locomp.load(A + i)))

def _bwd_log(GradOut: _T, FwdIn: _T, GradIn: _T,
             N: _cx):
    """GradIn += GradOut / a"""
    i = locomp.program_id(0)
    g = locomp.load(GradOut + i) / locomp.load(FwdIn + i)
    locomp.store(GradIn + i, locomp.load(GradIn + i) + g)

# ── relu ─────────────────────────────────────────────────────────────────────

def _fwd_relu(A: _T, Out: _T, N: _cx):
    i = locomp.program_id(0)
    x = locomp.load(A + i)
    locomp.store(Out + i, locomp.where(x > 0.0, x, 0.0))

def _bwd_relu(GradOut: _T, FwdIn: _T, GradIn: _T,
              N: _cx):
    """GradIn += GradOut * (fwd_in > 0)"""
    i = locomp.program_id(0)
    x = locomp.load(FwdIn + i)
    mask = locomp.where(x > 0.0, 1.0, 0.0)
    g = locomp.load(GradOut + i) * mask
    locomp.store(GradIn + i, locomp.load(GradIn + i) + g)

# ── sum (reduce) → scalar GPUTensor of size 1 ────────────────────────────────

def _fwd_sum_flat(A: _T, Out: _T, N: _cx):
    """Single-threaded naive sum — for small N or as reference."""
    i = locomp.program_id(0)   # only 1 thread
    acc = 0.0
    for k in range(N):
        acc = acc + locomp.load(A + k)
    locomp.store(Out + i, acc)

def _bwd_broadcast(GradIn: _T, N: _cx, G: _cx):
    """GradIn[i] += G  (scalar gradient broadcast — G is a float constexpr)"""
    i = locomp.program_id(0)
    locomp.store(GradIn + i, locomp.load(GradIn + i) + G)

# ── mean ─────────────────────────────────────────────────────────────────────

def _fwd_mean_flat(A: _T, Out: _T, N: _cx):
    i = locomp.program_id(0)
    acc = 0.0
    for k in range(N):
        acc = acc + locomp.load(A + k)
    locomp.store(Out + i, acc / N)

# _bwd_mean reuses _bwd_broadcast with G = grad_scalar / N computed on CPU

# ── pow ──────────────────────────────────────────────────────────────────────

def _fwd_pow(A: _T, Out: _T, N: _cx, Exp: _cx):
    i = locomp.program_id(0)
    locomp.store(Out + i, locomp.pow(locomp.load(A + i), Exp))

def _bwd_pow(GradOut: _T, FwdIn: _T, GradIn: _T, N: _cx, Exp: _cx):
    """GradIn += GradOut * Exp * a^(Exp-1)"""
    i = locomp.program_id(0)
    g = locomp.load(GradOut + i) * Exp * locomp.pow(locomp.load(FwdIn + i), Exp - 1.0)
    locomp.store(GradIn + i, locomp.load(GradIn + i) + g)

# ── sigmoid ──────────────────────────────────────────────────────────────────

def _fwd_sigmoid_k(A: _T, Out: _T, N: _cx):
    i = locomp.program_id(0)
    locomp.store(Out + i, locomp.sigmoid(locomp.load(A + i)))

def _bwd_sigmoid_k(GradOut: _T, FwdOut: _T, GradIn: _T, N: _cx):
    """GradIn += GradOut * s*(1-s)"""
    i = locomp.program_id(0)
    s = locomp.load(FwdOut + i)
    g = locomp.load(GradOut + i) * s * (1.0 - s)
    locomp.store(GradIn + i, locomp.load(GradIn + i) + g)

# ── tanh ─────────────────────────────────────────────────────────────────────

def _fwd_tanh_k(A: _T, Out: _T, N: _cx):
    i = locomp.program_id(0)
    locomp.store(Out + i, locomp.tanh(locomp.load(A + i)))

def _bwd_tanh_k(GradOut: _T, FwdOut: _T, GradIn: _T, N: _cx):
    """GradIn += GradOut * (1 - tanh^2)"""
    i = locomp.program_id(0)
    t = locomp.load(FwdOut + i)
    g = locomp.load(GradOut + i) * (1.0 - t * t)
    locomp.store(GradIn + i, locomp.load(GradIn + i) + g)

# ── matmul: C = A @ B  (A: M×K, B: K×N, C: M×N, all flattened) ─────────────

def _fwd_matmul(A: _T, B: _T, C: _T, M: _cx, K: _cx, N: _cx):
    """1 thread per output element. idx = row*N + col."""
    idx = locomp.program_id(0)
    row = idx // N
    col = idx % N
    acc = 0.0
    for k in range(K):
        acc = acc + locomp.load(A + row * K + k) * locomp.load(B + k * N + col)
    locomp.store(C + idx, acc)

def _bwd_matmul_A(GradC: _T, B: _T, GradA: _T, M: _cx, K: _cx, N: _cx):
    """dA[row,k] += sum_col(dC[row,col] * B[k,col])  — 1 thread per (row,k)"""
    idx = locomp.program_id(0)
    row = idx // K
    k = idx % K
    acc = 0.0
    for col in range(N):
        acc = acc + locomp.load(GradC + row * N + col) * locomp.load(B + k * N + col)
    locomp.store(GradA + idx, locomp.load(GradA + idx) + acc)

def _bwd_matmul_B(GradC: _T, A: _T, GradB: _T, M: _cx, K: _cx, N: _cx):
    """dB[k,col] += sum_row(A[row,k] * dC[row,col])  — 1 thread per (k,col)"""
    idx = locomp.program_id(0)
    k = idx // N
    col = idx % N
    acc = 0.0
    for row in range(M):
        acc = acc + locomp.load(A + row * K + k) * locomp.load(GradC + row * N + col)
    locomp.store(GradB + idx, locomp.load(GradB + idx) + acc)

# ── softmax (row-wise, flattened M×N) ────────────────────────────────────────

def _fwd_softmax_row(A: _T, Out: _T, M: _cx, N: _cx):
    """1 thread per row. 3-pass: max → exp_sum → normalize."""
    row = locomp.program_id(0)
    mx = -3.4e38
    for k in range(N):
        v = locomp.load(A + row * N + k)
        mx = locomp.where(v > mx, v, mx)
    s = 0.0
    for k in range(N):
        s = s + locomp.exp(locomp.load(A + row * N + k) - mx)
    for k in range(N):
        v = locomp.exp(locomp.load(A + row * N + k) - mx)
        locomp.store(Out + row * N + k, v / s)

def _bwd_softmax_row(GradOut: _T, FwdOut: _T, GradIn: _T, M: _cx, N: _cx):
    """dX[i] += S[i] * (dOut[i] - dot(dOut, S))  per row."""
    row = locomp.program_id(0)
    dot = 0.0
    for k in range(N):
        dot = dot + locomp.load(GradOut + row * N + k) * locomp.load(FwdOut + row * N + k)
    for k in range(N):
        s_i = locomp.load(FwdOut + row * N + k)
        g_i = locomp.load(GradOut + row * N + k)
        grad = s_i * (g_i - dot)
        locomp.store(GradIn + row * N + k, locomp.load(GradIn + row * N + k) + grad)

# ── cross_entropy: CE = -sum(P * log(Q + eps))  (P=target, Q=softmax output) ─

def _fwd_ce_row(P: _T, Q: _T, Loss: _T, B: _cx, N: _cx):
    """1 thread per example. Loss[b] = -sum_k P[b,k]*log(Q[b,k]+eps)."""
    b = locomp.program_id(0)
    acc = 0.0
    for k in range(N):
        p = locomp.load(P + b * N + k)
        q = locomp.load(Q + b * N + k)
        acc = acc - p * locomp.log(q + 1e-8)
    locomp.store(Loss + b, acc)

def _bwd_ce_dq(GradLoss: _T, P: _T, Q: _T, GradQ: _T, B: _cx, N: _cx):
    """dQ[b,k] += GradLoss[b] * (-P[b,k] / (Q[b,k]+eps))."""
    b = locomp.program_id(0)
    g = locomp.load(GradLoss + b)
    for k in range(N):
        p = locomp.load(P + b * N + k)
        q = locomp.load(Q + b * N + k)
        idx = b * N + k
        grad = -g * p / (q + 1e-8)
        locomp.store(GradQ + idx, locomp.load(GradQ + idx) + grad)

# ── matvec: out = A @ x  (A: M×K, x: K, out: M) ────────────────────────────

def _fwd_matvec(A: _T, X: _T, Out: _T,
                M: _cx, K: _cx):
    row = locomp.program_id(0)
    acc = 0.0
    for k in range(K):
        acc = acc + locomp.load(A + row * K + k) * locomp.load(X + k)
    locomp.store(Out + row, acc)

def _bwd_matvec_x(GradOut: _T, A: _T, GradX: _T,
                  M: _cx, K: _cx):
    """GradX[k] += sum_m(GradOut[m] * A[m,k])  — i.e., A^T @ grad_out"""
    k = locomp.program_id(0)
    acc = 0.0
    for m in range(M):
        acc = acc + locomp.load(GradOut + m) * locomp.load(A + m * K + k)
    locomp.store(GradX + k, locomp.load(GradX + k) + acc)

def _bwd_matvec_A(GradOut: _T, X: _T, GradA: _T,
                  M: _cx, K: _cx):
    """GradA[m,k] += GradOut[m] * X[k]  — outer product, one row per thread"""
    m = locomp.program_id(0)
    g_m = locomp.load(GradOut + m)
    for k in range(K):
        old = locomp.load(GradA + m * K + k)
        locomp.store(GradA + m * K + k, old + g_m * locomp.load(X + k))


# ─── Tape ────────────────────────────────────────────────────────────────────

_tape: List[Tuple["GPUTensor", List["GPUTensor"]]] = []
_tape_enabled = True


def no_grad():
    class _NoGrad:
        def __enter__(self):
            global _tape_enabled
            self._prev = _tape_enabled
            _tape_enabled = False
        def __exit__(self, *_):
            global _tape_enabled
            _tape_enabled = self._prev
    return _NoGrad()


def zero_grad(*tensors: "GPUTensor"):
    for t in tensors:
        t.grad = None


# ─── GPUTensor ───────────────────────────────────────────────────────────────

class GPUTensor:
    """
    A GPU-backed tensor that participates in autograd.

    Wraps a locomp LocompTensor. Gradient is also a GPUTensor (on-device).
    """
    __slots__ = ("_locomp", "grad", "requires_grad", "_backward_fn", "shape",
                 "dtype", "_backend", "size")

    def __init__(self, locomp_tensor, requires_grad: bool = False,
                 backend: str = "auto"):
        self._locomp = locomp_tensor
        self.grad: Optional[GPUTensor] = None
        self.requires_grad = requires_grad
        self._backward_fn: Optional[Callable] = None
        self.shape = locomp_tensor.shape
        self.size = locomp_tensor.size
        self.dtype = locomp_tensor.dtype
        self._backend = backend

    def numpy(self) -> np.ndarray:
        return self._locomp.numpy()

    def item(self) -> float:
        return float(self.numpy().ravel()[0])

    def zero_grad(self):
        self.grad = None

    def _accum_grad(self, g: "GPUTensor"):
        """Accumulate gradient: self.grad += g (on GPU)."""
        n = g.size
        if self.grad is None:
            self.grad = _zeros_like(g)
        k = _k("add_accum", _bwd_add_accumulate, self._backend)
        k[(n,), (1,)](g._locomp, self.grad._locomp, n)

    def __repr__(self):
        return f"GPUTensor(shape={self.shape}, backend={self._backend}, requires_grad={self.requires_grad})"


def tensor(data: np.ndarray, requires_grad: bool = False,
           backend: str = "auto") -> GPUTensor:
    """Create a GPUTensor from a NumPy array."""
    arr = np.asarray(data, dtype=np.float32)
    lt = locomp.tensor(arr, backend=backend)
    return GPUTensor(lt, requires_grad=requires_grad, backend=backend)


def empty(shape, backend: str = "auto") -> GPUTensor:
    """Allocate an uninitialized GPUTensor."""
    if isinstance(shape, int):
        n = shape
    else:
        n = 1
        for s in shape:
            n *= s
    lt = locomp.empty(n, backend=backend)
    return GPUTensor(lt, requires_grad=False, backend=backend)


# ─── Record helper ───────────────────────────────────────────────────────────

def _record(out: GPUTensor, bwd_fn: Callable, inputs: List[GPUTensor]):
    if not _tape_enabled:
        return
    if not any(t.requires_grad for t in inputs):
        return
    out.requires_grad = True
    out._backward_fn = bwd_fn
    _tape.append((out, inputs))


# ─── backward ────────────────────────────────────────────────────────────────

def backward(loss: GPUTensor):
    """Run reverse-mode autodiff from scalar `loss`."""
    if loss.size != 1:
        raise ValueError(
            f"backward() requires a scalar loss (size=1), got size={loss.size}. "
            "Use locomp.gpu_ag.sum(x) to reduce to scalar first."
        )
    # Seed gradient: ones (on GPU)
    if loss.grad is None:
        loss.grad = tensor(np.ones(1, dtype=np.float32),
                           backend=loss._backend)

    # Topological sort via DFS
    visited = set()
    order = []

    def _topo(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        for out, inputs in _tape:
            if out is node:
                for inp in inputs:
                    _topo(inp)
                order.append(node)
                return
        order.append(node)

    _topo(loss)

    for node in reversed(order):
        if node._backward_fn is not None:
            node._backward_fn()

    _tape.clear()


# ─── Op implementations ──────────────────────────────────────────────────────

def add(a: GPUTensor, b: GPUTensor) -> GPUTensor:
    n = a.size
    out = empty(n, backend=a._backend)
    fwd = _k("fwd_add", _fwd_add, a._backend)
    fwd[(n,), (1,)](a._locomp, b._locomp, out._locomp, n)

    def _bwd():
        g = out.grad
        if g is None:
            return
        if a.requires_grad:
            a._accum_grad(g)
        if b.requires_grad:
            b._accum_grad(g)

    _record(out, _bwd, [a, b])
    return out


def sub(a: GPUTensor, b: GPUTensor) -> GPUTensor:
    n = a.size
    out = empty(n, backend=a._backend)
    fwd = _k("fwd_sub", _fwd_sub, a._backend)
    fwd[(n,), (1,)](a._locomp, b._locomp, out._locomp, n)

    def _bwd():
        g = out.grad
        if g is None:
            return
        if a.requires_grad:
            a._accum_grad(g)
        if b.requires_grad:
            # grad for b is -g
            neg_g = empty(n, backend=b._backend)
            k_neg = _k("fwd_sub_neg", _bwd_sub_b, b._backend)
            neg_g_lt = locomp.zeros(n, backend=b._backend)
            k_neg[(n,), (1,)](g._locomp, neg_g_lt, n)
            b._accum_grad(GPUTensor(neg_g_lt, backend=b._backend))

    _record(out, _bwd, [a, b])
    return out


def mul(a: GPUTensor, b: GPUTensor) -> GPUTensor:
    a_saved, b_saved = a._locomp, b._locomp
    n = a.size
    out = empty(n, backend=a._backend)
    fwd = _k("fwd_mul", _fwd_mul, a._backend)
    fwd[(n,), (1,)](a._locomp, b._locomp, out._locomp, n)

    def _bwd():
        g = out.grad
        if g is None:
            return
        k_mul = _k("bwd_mul", _bwd_mul, a._backend)
        if a.requires_grad:
            da = GPUTensor(locomp.zeros(n, backend=a._backend), backend=a._backend)
            k_mul[(n,), (1,)](g._locomp, b_saved, da._locomp, n)
            a._accum_grad(da)
        if b.requires_grad:
            db = GPUTensor(locomp.zeros(n, backend=b._backend), backend=b._backend)
            k_mul[(n,), (1,)](g._locomp, a_saved, db._locomp, n)
            b._accum_grad(db)

    _record(out, _bwd, [a, b])
    return out


def div(a: GPUTensor, b: GPUTensor) -> GPUTensor:
    a_saved, b_saved = a._locomp, b._locomp
    n = a.size
    out = empty(n, backend=a._backend)
    fwd = _k("fwd_div", _fwd_div, a._backend)
    fwd[(n,), (1,)](a._locomp, b._locomp, out._locomp, n)

    def _bwd():
        g = out.grad
        if g is None:
            return
        if a.requires_grad:
            da = GPUTensor(locomp.zeros(n, backend=a._backend), backend=a._backend)
            k = _k("bwd_div_a", _bwd_div_a, a._backend)
            k[(n,), (1,)](g._locomp, b_saved, da._locomp, n)
            a._accum_grad(da)
        if b.requires_grad:
            db = GPUTensor(locomp.zeros(n, backend=b._backend), backend=b._backend)
            k = _k("bwd_div_b", _bwd_div_b, b._backend)
            k[(n,), (1,)](g._locomp, a_saved, b_saved, db._locomp, n)
            b._accum_grad(db)

    _record(out, _bwd, [a, b])
    return out


def exp(a: GPUTensor) -> GPUTensor:
    n = a.size
    out = empty(n, backend=a._backend)
    fwd = _k("fwd_exp", _fwd_exp, a._backend)
    fwd[(n,), (1,)](a._locomp, out._locomp, n)
    out_saved = out._locomp

    def _bwd():
        g = out.grad
        if g is None or not a.requires_grad:
            return
        da = GPUTensor(locomp.zeros(n, backend=a._backend), backend=a._backend)
        k = _k("bwd_exp", _bwd_exp, a._backend)
        k[(n,), (1,)](g._locomp, out_saved, da._locomp, n)
        a._accum_grad(da)

    _record(out, _bwd, [a])
    return out


def log(a: GPUTensor) -> GPUTensor:
    a_saved = a._locomp
    n = a.size
    out = empty(n, backend=a._backend)
    fwd = _k("fwd_log", _fwd_log, a._backend)
    fwd[(n,), (1,)](a._locomp, out._locomp, n)

    def _bwd():
        g = out.grad
        if g is None or not a.requires_grad:
            return
        da = GPUTensor(locomp.zeros(n, backend=a._backend), backend=a._backend)
        k = _k("bwd_log", _bwd_log, a._backend)
        k[(n,), (1,)](g._locomp, a_saved, da._locomp, n)
        a._accum_grad(da)

    _record(out, _bwd, [a])
    return out


def relu(a: GPUTensor) -> GPUTensor:
    a_saved = a._locomp
    n = a.size
    out = empty(n, backend=a._backend)
    fwd = _k("fwd_relu", _fwd_relu, a._backend)
    fwd[(n,), (1,)](a._locomp, out._locomp, n)

    def _bwd():
        g = out.grad
        if g is None or not a.requires_grad:
            return
        da = GPUTensor(locomp.zeros(n, backend=a._backend), backend=a._backend)
        k = _k("bwd_relu", _bwd_relu, a._backend)
        k[(n,), (1,)](g._locomp, a_saved, da._locomp, n)
        a._accum_grad(da)

    _record(out, _bwd, [a])
    return out


def sum(a: GPUTensor) -> GPUTensor:
    """Reduce entire tensor to scalar (size=1 GPUTensor)."""
    n = a.size
    out_lt = locomp.empty(1)
    fwd = _k("fwd_sum", _fwd_sum_flat, a._backend)
    fwd[(1,), (1,)](a._locomp, out_lt, n)
    out = GPUTensor(out_lt, backend=a._backend)

    def _bwd():
        g = out.grad
        if g is None or not a.requires_grad:
            return
        g_cpu = float(g._locomp.numpy()[0])  # read scalar grad to CPU
        da = GPUTensor(locomp.zeros(n, backend=a._backend), backend=a._backend)
        k = _k("bwd_broadcast", _bwd_broadcast, a._backend)
        k[(n,), (1,)](da._locomp, n, g_cpu)
        a._accum_grad(da)

    _record(out, _bwd, [a])
    return out


def mean(a: GPUTensor) -> GPUTensor:
    """Reduce entire tensor to scalar mean."""
    n = a.size
    out_lt = locomp.empty(1)
    fwd = _k("fwd_mean", _fwd_mean_flat, a._backend)
    fwd[(1,), (1,)](a._locomp, out_lt, n)
    out = GPUTensor(out_lt, backend=a._backend)

    def _bwd():
        g = out.grad
        if g is None or not a.requires_grad:
            return
        g_cpu = float(g._locomp.numpy()[0]) / n  # scalar grad / N
        da = GPUTensor(locomp.zeros(n, backend=a._backend), backend=a._backend)
        k = _k("bwd_broadcast", _bwd_broadcast, a._backend)
        k[(n,), (1,)](da._locomp, n, g_cpu)
        a._accum_grad(da)

    _record(out, _bwd, [a])
    return out


def matvec(A: GPUTensor, x: GPUTensor, M: int, K: int) -> GPUTensor:
    """Matrix-vector multiply: out = A @ x  (A: M×K flattened, x: K, out: M)"""
    A_saved, x_saved = A._locomp, x._locomp
    out_lt = locomp.empty(M)
    fwd = _k("fwd_matvec", _fwd_matvec, A._backend)
    fwd[(M,), (1,)](A._locomp, x._locomp, out_lt, M, K)
    out = GPUTensor(out_lt, backend=A._backend)

    def _bwd():
        g = out.grad
        if g is None:
            return
        if x.requires_grad:
            dx = GPUTensor(locomp.zeros(K), backend=x._backend)
            k = _k("bwd_matvec_x", _bwd_matvec_x, x._backend)
            k[(K,), (1,)](g._locomp, A_saved, dx._locomp, M, K)
            x._accum_grad(dx)
        if A.requires_grad:
            dA = GPUTensor(locomp.zeros(M * K), backend=A._backend)
            k = _k("bwd_matvec_A", _bwd_matvec_A, A._backend)
            k[(M,), (1,)](g._locomp, x_saved, dA._locomp, M, K)
            A._accum_grad(dA)

    _record(out, _bwd, [A, x])
    return out


def pow(a: GPUTensor, exp: float) -> GPUTensor:
    """Element-wise a^exp."""
    n = a.size
    a_saved = a._locomp
    out = empty(n, backend=a._backend)
    fwd = _k("fwd_pow", _fwd_pow, a._backend)
    fwd[(n,), (1,)](a._locomp, out._locomp, n, float(exp))

    def _bwd():
        g = out.grad
        if g is None or not a.requires_grad:
            return
        da = GPUTensor(locomp.zeros(n, backend=a._backend), backend=a._backend)
        k = _k("bwd_pow", _bwd_pow, a._backend)
        k[(n,), (1,)](g._locomp, a_saved, da._locomp, n, float(exp))
        a._accum_grad(da)

    _record(out, _bwd, [a])
    return out


def sigmoid(a: GPUTensor) -> GPUTensor:
    """Element-wise sigmoid."""
    n = a.size
    out = empty(n, backend=a._backend)
    fwd = _k("fwd_sigmoid", _fwd_sigmoid_k, a._backend)
    fwd[(n,), (1,)](a._locomp, out._locomp, n)
    out_saved = out._locomp

    def _bwd():
        g = out.grad
        if g is None or not a.requires_grad:
            return
        da = GPUTensor(locomp.zeros(n, backend=a._backend), backend=a._backend)
        k = _k("bwd_sigmoid", _bwd_sigmoid_k, a._backend)
        k[(n,), (1,)](g._locomp, out_saved, da._locomp, n)
        a._accum_grad(da)

    _record(out, _bwd, [a])
    return out


def tanh(a: GPUTensor) -> GPUTensor:
    """Element-wise tanh."""
    n = a.size
    out = empty(n, backend=a._backend)
    fwd = _k("fwd_tanh", _fwd_tanh_k, a._backend)
    fwd[(n,), (1,)](a._locomp, out._locomp, n)
    out_saved = out._locomp

    def _bwd():
        g = out.grad
        if g is None or not a.requires_grad:
            return
        da = GPUTensor(locomp.zeros(n, backend=a._backend), backend=a._backend)
        k = _k("bwd_tanh", _bwd_tanh_k, a._backend)
        k[(n,), (1,)](g._locomp, out_saved, da._locomp, n)
        a._accum_grad(da)

    _record(out, _bwd, [a])
    return out


def matmul(A: GPUTensor, B: GPUTensor, M: int, K: int, N: int) -> GPUTensor:
    """Matrix multiply: C = A @ B  (A: M×K, B: K×N, C: M×N, all row-major flat)."""
    A_saved, B_saved = A._locomp, B._locomp
    out_lt = locomp.empty(M * N)
    fwd = _k("fwd_matmul", _fwd_matmul, A._backend)
    fwd[(M * N,), (1,)](A._locomp, B._locomp, out_lt, M, K, N)
    out = GPUTensor(out_lt, backend=A._backend)

    def _bwd():
        g = out.grad
        if g is None:
            return
        if A.requires_grad:
            dA = GPUTensor(locomp.zeros(M * K), backend=A._backend)
            k = _k("bwd_matmul_A", _bwd_matmul_A, A._backend)
            k[(M * K,), (1,)](g._locomp, B_saved, dA._locomp, M, K, N)
            A._accum_grad(dA)
        if B.requires_grad:
            dB = GPUTensor(locomp.zeros(K * N), backend=B._backend)
            k = _k("bwd_matmul_B", _bwd_matmul_B, B._backend)
            k[(K * N,), (1,)](g._locomp, A_saved, dB._locomp, M, K, N)
            B._accum_grad(dB)

    _record(out, _bwd, [A, B])
    return out


def softmax(a: GPUTensor, M: int, N: int) -> GPUTensor:
    """Row-wise softmax. Input: M×N flattened. Output: M×N flattened."""
    out_lt = locomp.empty(M * N)
    fwd = _k("fwd_softmax", _fwd_softmax_row, a._backend)
    fwd[(M,), (1,)](a._locomp, out_lt, M, N)
    out = GPUTensor(out_lt, backend=a._backend)
    out_saved = out_lt

    def _bwd():
        g = out.grad
        if g is None or not a.requires_grad:
            return
        da = GPUTensor(locomp.zeros(M * N), backend=a._backend)
        k = _k("bwd_softmax", _bwd_softmax_row, a._backend)
        k[(M,), (1,)](g._locomp, out_saved, da._locomp, M, N)
        a._accum_grad(da)

    _record(out, _bwd, [a])
    return out


def cross_entropy(q: GPUTensor, p: GPUTensor, B: int, N: int) -> GPUTensor:
    """Cross entropy loss: Loss[b] = -sum_k p[b,k]*log(q[b,k]+eps).

    Args:
        q: softmax probabilities (B×N flattened), requires_grad
        p: one-hot targets (B×N flattened), no grad
        B: batch size, N: num classes
    Returns:
        GPUTensor of size B (per-example loss).
    """
    out_lt = locomp.empty(B)
    fwd = _k("fwd_ce", _fwd_ce_row, q._backend)
    fwd[(B,), (1,)](p._locomp, q._locomp, out_lt, B, N)
    out = GPUTensor(out_lt, backend=q._backend)
    p_saved, q_saved = p._locomp, q._locomp

    def _bwd():
        g = out.grad
        if g is None or not q.requires_grad:
            return
        dq = GPUTensor(locomp.zeros(B * N), backend=q._backend)
        k = _k("bwd_ce_dq", _bwd_ce_dq, q._backend)
        k[(B,), (1,)](g._locomp, p_saved, q_saved, dq._locomp, B, N)
        q._accum_grad(dq)

    _record(out, _bwd, [q])
    return out
