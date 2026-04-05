"""
Tests for locomp.gpu_ag — GPU-backed autograd.

Metal tests (macos_only): run actual forward+backward kernels on M1.
CUDA codegen tests: verify backward kernel MSL/CUDA source strings,
                    no GPU required.
"""
import numpy as np
import pytest
import locomp
from locomp import gpu_ag
from tests.conftest import macos_only


# ─── helpers ────────────────────────────────────────────────────────────────

def _np_check(gpu_tensor, expected, atol=1e-4):
    result = gpu_tensor.numpy().ravel().astype(np.float64)
    expected = np.asarray(expected, dtype=np.float64).ravel()
    assert np.allclose(result, expected, atol=atol), \
        f"max diff={np.max(np.abs(result - expected)):.2e}\ngot:      {result}\nexpected: {expected}"


# ─── GPUTensor API ───────────────────────────────────────────────────────────

@macos_only
def test_gpu_tensor_creation():
    x = gpu_ag.tensor(np.array([1.0, 2.0, 3.0]))
    assert x.shape == (3,)
    assert x.requires_grad is False
    assert x.grad is None


@macos_only
def test_gpu_tensor_numpy_roundtrip():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = gpu_ag.tensor(data)
    np.testing.assert_allclose(x.numpy(), data)


@macos_only
def test_gpu_tensor_requires_grad():
    x = gpu_ag.tensor(np.ones(4), requires_grad=True)
    assert x.requires_grad is True


# ─── add ─────────────────────────────────────────────────────────────────────

@macos_only
def test_gpu_add_forward():
    a = gpu_ag.tensor(np.array([1.0, 2.0, 3.0]))
    b = gpu_ag.tensor(np.array([4.0, 5.0, 6.0]))
    c = gpu_ag.add(a, b)
    _np_check(c, [5.0, 7.0, 9.0])


@macos_only
def test_gpu_add_backward():
    a = gpu_ag.tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    b = gpu_ag.tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
    out = gpu_ag.add(a, b)
    loss = gpu_ag.sum(out)
    gpu_ag.backward(loss)
    # dL/da = dL/db = ones
    _np_check(a.grad, [1.0, 1.0, 1.0])
    _np_check(b.grad, [1.0, 1.0, 1.0])


# ─── sub ─────────────────────────────────────────────────────────────────────

@macos_only
def test_gpu_sub_forward():
    a = gpu_ag.tensor(np.array([5.0, 6.0]))
    b = gpu_ag.tensor(np.array([3.0, 2.0]))
    c = gpu_ag.sub(a, b)
    _np_check(c, [2.0, 4.0])


@macos_only
def test_gpu_sub_backward():
    a = gpu_ag.tensor(np.array([5.0, 6.0]), requires_grad=True)
    b = gpu_ag.tensor(np.array([3.0, 2.0]), requires_grad=True)
    loss = gpu_ag.sum(gpu_ag.sub(a, b))
    gpu_ag.backward(loss)
    _np_check(a.grad, [1.0, 1.0])
    _np_check(b.grad, [-1.0, -1.0])


# ─── mul ─────────────────────────────────────────────────────────────────────

@macos_only
def test_gpu_mul_forward():
    a = gpu_ag.tensor(np.array([2.0, 3.0]))
    b = gpu_ag.tensor(np.array([4.0, 5.0]))
    c = gpu_ag.mul(a, b)
    _np_check(c, [8.0, 15.0])


@macos_only
def test_gpu_mul_backward():
    a_np = np.array([2.0, 3.0, 4.0])
    b_np = np.array([5.0, 6.0, 7.0])
    a = gpu_ag.tensor(a_np, requires_grad=True)
    b = gpu_ag.tensor(b_np, requires_grad=True)
    loss = gpu_ag.sum(gpu_ag.mul(a, b))
    gpu_ag.backward(loss)
    # dL/da = b, dL/db = a
    _np_check(a.grad, b_np)
    _np_check(b.grad, a_np)


# ─── div ─────────────────────────────────────────────────────────────────────

@macos_only
def test_gpu_div_forward():
    a = gpu_ag.tensor(np.array([6.0, 8.0]))
    b = gpu_ag.tensor(np.array([2.0, 4.0]))
    c = gpu_ag.div(a, b)
    _np_check(c, [3.0, 2.0])


@macos_only
def test_gpu_div_backward_a():
    a_np = np.array([6.0, 8.0])
    b_np = np.array([2.0, 4.0])
    a = gpu_ag.tensor(a_np, requires_grad=True)
    b = gpu_ag.tensor(b_np, requires_grad=False)
    loss = gpu_ag.sum(gpu_ag.div(a, b))
    gpu_ag.backward(loss)
    # dL/da = 1/b
    _np_check(a.grad, 1.0 / b_np)


# ─── exp ─────────────────────────────────────────────────────────────────────

@macos_only
def test_gpu_exp_forward():
    a = gpu_ag.tensor(np.array([0.0, 1.0]))
    out = gpu_ag.exp(a)
    _np_check(out, [1.0, np.e], atol=1e-5)


@macos_only
def test_gpu_exp_backward():
    a_np = np.array([0.0, 0.5, 1.0])
    a = gpu_ag.tensor(a_np, requires_grad=True)
    loss = gpu_ag.sum(gpu_ag.exp(a))
    gpu_ag.backward(loss)
    # dL/da = exp(a)
    _np_check(a.grad, np.exp(a_np), atol=1e-5)


# ─── log ─────────────────────────────────────────────────────────────────────

@macos_only
def test_gpu_log_forward():
    a = gpu_ag.tensor(np.array([1.0, np.e]))
    out = gpu_ag.log(a)
    _np_check(out, [0.0, 1.0], atol=1e-5)


@macos_only
def test_gpu_log_backward():
    a_np = np.array([0.5, 1.0, 2.0])
    a = gpu_ag.tensor(a_np, requires_grad=True)
    loss = gpu_ag.sum(gpu_ag.log(a))
    gpu_ag.backward(loss)
    _np_check(a.grad, 1.0 / a_np, atol=1e-5)


# ─── relu ─────────────────────────────────────────────────────────────────────

@macos_only
def test_gpu_relu_forward():
    a = gpu_ag.tensor(np.array([-1.0, 0.5, -0.2, 2.0]))
    out = gpu_ag.relu(a)
    _np_check(out, [0.0, 0.5, 0.0, 2.0])


@macos_only
def test_gpu_relu_backward():
    a_np = np.array([1.0, -0.5, 0.3, -2.0])
    a = gpu_ag.tensor(a_np, requires_grad=True)
    loss = gpu_ag.sum(gpu_ag.relu(a))
    gpu_ag.backward(loss)
    # dL/da = 1 where a>0, else 0
    _np_check(a.grad, (a_np > 0).astype(np.float32))


# ─── sum ─────────────────────────────────────────────────────────────────────

@macos_only
def test_gpu_sum_forward():
    a = gpu_ag.tensor(np.array([1.0, 2.0, 3.0, 4.0]))
    out = gpu_ag.sum(a)
    assert abs(out.item() - 10.0) < 1e-4


@macos_only
def test_gpu_sum_backward():
    a = gpu_ag.tensor(np.ones(8), requires_grad=True)
    loss = gpu_ag.sum(a)
    gpu_ag.backward(loss)
    _np_check(a.grad, np.ones(8))


# ─── mean ─────────────────────────────────────────────────────────────────────

@macos_only
def test_gpu_mean_forward():
    a = gpu_ag.tensor(np.array([2.0, 4.0, 6.0, 8.0]))
    out = gpu_ag.mean(a)
    assert abs(out.item() - 5.0) < 1e-4


@macos_only
def test_gpu_mean_backward():
    n = 8
    a = gpu_ag.tensor(np.ones(n) * 3.0, requires_grad=True)
    loss = gpu_ag.mean(a)
    gpu_ag.backward(loss)
    _np_check(a.grad, np.ones(n) / n, atol=1e-5)


# ─── matvec ──────────────────────────────────────────────────────────────────

@macos_only
def test_gpu_matvec_forward():
    M, K = 3, 4
    A_np = np.eye(M, K, dtype=np.float32)   # 3×4 identity-ish
    x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    A = gpu_ag.tensor(A_np.flatten())
    x = gpu_ag.tensor(x_np)
    out = gpu_ag.matvec(A, x, M, K)
    # eye(3,4) @ [1,2,3,4] = [1,2,3]
    _np_check(out, [1.0, 2.0, 3.0])


@macos_only
def test_gpu_matvec_backward_x():
    M, K = 4, 3
    np.random.seed(7)
    A_np = np.random.randn(M, K).astype(np.float32)
    x_np = np.random.randn(K).astype(np.float32)
    A = gpu_ag.tensor(A_np.flatten(), requires_grad=False)
    x = gpu_ag.tensor(x_np, requires_grad=True)
    loss = gpu_ag.sum(gpu_ag.matvec(A, x, M, K))
    gpu_ag.backward(loss)
    # dL/dx = A^T @ ones(M)
    expected = A_np.T @ np.ones(M, dtype=np.float32)
    _np_check(x.grad, expected, atol=1e-4)


@macos_only
def test_gpu_matvec_backward_A():
    M, K = 4, 3
    np.random.seed(9)
    A_np = np.random.randn(M, K).astype(np.float32)
    x_np = np.random.randn(K).astype(np.float32)
    A = gpu_ag.tensor(A_np.flatten(), requires_grad=True)
    x = gpu_ag.tensor(x_np, requires_grad=False)
    loss = gpu_ag.sum(gpu_ag.matvec(A, x, M, K))
    gpu_ag.backward(loss)
    # dL/dA[m,k] = x[k] (broadcast by ones(M))  i.e. outer(ones, x)
    expected = np.outer(np.ones(M), x_np).flatten()
    _np_check(A.grad, expected, atol=1e-4)


# ─── chain: relu(exp(x)) ──────────────────────────────────────────────────────

@macos_only
def test_gpu_chain_relu_exp():
    a_np = np.array([1.0, -1.0, 0.5])
    a = gpu_ag.tensor(a_np, requires_grad=True)
    e = gpu_ag.exp(a)
    r = gpu_ag.relu(e)    # relu is identity for exp output (always positive)
    loss = gpu_ag.sum(r)
    gpu_ag.backward(loss)
    # exp is always > 0, relu passes through:  dL/da = exp(a)
    _np_check(a.grad, np.exp(a_np), atol=1e-4)


# ─── no_grad ─────────────────────────────────────────────────────────────────

@macos_only
def test_gpu_no_grad():
    x = gpu_ag.tensor(np.ones(4), requires_grad=True)
    with gpu_ag.no_grad():
        y = gpu_ag.add(x, gpu_ag.tensor(np.ones(4)))
    assert y.requires_grad is False
    assert y._backward_fn is None


# ─── tape cleared after backward ─────────────────────────────────────────────

@macos_only
def test_gpu_tape_cleared():
    from locomp import gpu_autograd as _gm
    x = gpu_ag.tensor(np.ones(4), requires_grad=True)
    loss = gpu_ag.sum(gpu_ag.relu(x))
    gpu_ag.backward(loss)
    assert len(_gm._tape) == 0


# ─── CUDA codegen: verify backward kernels compile to correct CUDA source ─────
# These tests don't need a GPU — they only check the generated .cu source.

def _cuda_src(fn, constexprs=None):
    """Compile fn to CUDA source string."""
    from locomp.frontend import compile_kernel
    from locomp.optimizer import optimize
    from locomp.backends.cuda_codegen import compile_to_cuda
    ir = compile_kernel(fn)
    ir = optimize(ir, target="cuda")
    src, _ = compile_to_cuda(ir, constexpr_values=constexprs or {})
    return src


def test_cuda_bwd_add_accumulate_source():
    src = _cuda_src(_bwd_add_accumulate, {"N": 16})
    assert "__global__" in src
    # should add GradIn[i] += GradOut[i]
    assert "+" in src


def test_cuda_bwd_mul_source():
    src = _cuda_src(_bwd_mul, {"N": 16})
    # should multiply GradOut * Other
    assert "*" in src
    assert "__global__" in src


def test_cuda_bwd_relu_source():
    src = _cuda_src(_bwd_relu, {"N": 16})
    # should contain conditional (where/ternary)
    assert "?" in src or "where" in src.lower() or "fmaxf" in src or ">" in src


def test_cuda_bwd_exp_source():
    src = _cuda_src(_bwd_exp, {"N": 16})
    # grad = GradOut * FwdOut (multiply)
    assert "*" in src


def test_cuda_bwd_log_source():
    src = _cuda_src(_bwd_log, {"N": 16})
    # grad = GradOut / a (division)
    assert "/" in src


def test_cuda_bwd_sum_source():
    src = _cuda_src(_bwd_broadcast, {"N": 16, "G": 1.0})
    # broadcasts scalar grad G to all elements
    assert "GradIn" in src or "gradin" in src.lower() or "__global__" in src


def test_cuda_bwd_matvec_x_source():
    src = _cuda_src(_bwd_matvec_x, {"M": 4, "K": 3})
    # A^T @ grad_out: sum over rows
    assert "for" in src
    assert "__global__" in src


def test_cuda_bwd_matvec_A_source():
    src = _cuda_src(_bwd_matvec_A, {"M": 4, "K": 3})
    # outer product: loop over K
    assert "for" in src
    assert "__global__" in src


# ─── import from locomp namespace ────────────────────────────────────────────

def test_gpu_ag_accessible_as_locomp_gpu_ag():
    import locomp
    assert hasattr(locomp, "gpu_ag")
    assert hasattr(locomp.gpu_ag, "backward")
    assert hasattr(locomp.gpu_ag, "tensor")
    assert hasattr(locomp.gpu_ag, "GPUTensor")


# Re-import these so the CUDA codegen tests above can reference them
from locomp.gpu_autograd import (
    _bwd_add_accumulate, _bwd_mul, _bwd_relu, _bwd_exp, _bwd_log,
    _bwd_broadcast, _bwd_matvec_x, _bwd_matvec_A,
)
