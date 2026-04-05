"""
Tests for locomp.ag (autograd) — tape-based reverse-mode autodiff.

All tests run on CPU (NumPy), no GPU required.
Gradients are verified by finite differences at tolerance 1e-4.
"""
import numpy as np
import pytest
import locomp
from locomp import ag


# ─── helpers ────────────────────────────────────────────────────────────────

def numerical_grad(fn, x_np: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Compute numerical gradient of fn(x) w.r.t. x via central differences."""
    x_np = x_np.astype(np.float64)  # use float64 for numerical stability
    grad = np.zeros_like(x_np)
    it = np.nditer(x_np, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x_np[idx]
        x_np[idx] = orig + eps
        fp = float(fn(x_np.astype(np.float32)))
        x_np[idx] = orig - eps
        fm = float(fn(x_np.astype(np.float32)))
        x_np[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad.astype(np.float32)


def check_grad(fn, x_np, atol=1e-2):
    """Run autograd and numerical grad, assert they match."""
    x = ag.tensor(x_np.copy(), requires_grad=True)
    loss = fn(x)
    ag.backward(loss)
    auto = x.grad.copy()

    num = numerical_grad(lambda v: fn(ag.tensor(v, requires_grad=False)).item(), x_np.copy())
    assert np.allclose(auto, num, atol=atol), f"max diff={np.max(np.abs(auto - num)):.2e}"


# ─── AGTensor basics ─────────────────────────────────────────────────────────

def test_ag_tensor_creation():
    x = ag.tensor(np.array([1.0, 2.0, 3.0]))
    assert x.shape == (3,)
    assert x.requires_grad is False
    assert x.grad is None


def test_ag_tensor_requires_grad():
    x = ag.tensor(np.ones(4), requires_grad=True)
    assert x.requires_grad is True


def test_ag_tensor_numpy():
    data = np.array([1.0, 2.0])
    x = ag.tensor(data)
    np.testing.assert_array_equal(x.numpy(), data.astype(np.float32))


def test_ag_tensor_item():
    x = ag.tensor(np.array(3.14))
    assert abs(x.item() - 3.14) < 1e-4


# ─── no_grad ─────────────────────────────────────────────────────────────────

def test_no_grad_disables_tape():
    x = ag.tensor(np.ones(3), requires_grad=True)
    with ag.no_grad():
        y = ag.add(x, x)
    assert y.requires_grad is False
    assert y._backward_fn is None


def test_no_grad_restores_tape():
    x = ag.tensor(np.ones(3), requires_grad=True)
    with ag.no_grad():
        pass
    y = ag.add(x, x)
    assert y.requires_grad is True


# ─── add ─────────────────────────────────────────────────────────────────────

def test_add_forward():
    a = ag.tensor(np.array([1.0, 2.0, 3.0]))
    b = ag.tensor(np.array([4.0, 5.0, 6.0]))
    c = ag.add(a, b)
    np.testing.assert_allclose(c.data, [5.0, 7.0, 9.0])


def test_add_backward():
    check_grad(lambda x: ag.sum(ag.add(x, ag.tensor(np.ones(4)))),
               np.random.randn(4).astype(np.float32))


def test_add_grad_both_inputs():
    a = ag.tensor(np.array([1.0, 2.0]), requires_grad=True)
    b = ag.tensor(np.array([3.0, 4.0]), requires_grad=True)
    loss = ag.sum(ag.add(a, b))
    ag.backward(loss)
    np.testing.assert_allclose(a.grad, [1.0, 1.0])
    np.testing.assert_allclose(b.grad, [1.0, 1.0])


# ─── sub ─────────────────────────────────────────────────────────────────────

def test_sub_backward():
    check_grad(lambda x: ag.sum(ag.sub(x, ag.tensor(np.ones(4)))),
               np.random.randn(4).astype(np.float32))


# ─── mul ─────────────────────────────────────────────────────────────────────

def test_mul_forward():
    a = ag.tensor(np.array([2.0, 3.0]))
    b = ag.tensor(np.array([4.0, 5.0]))
    c = ag.mul(a, b)
    np.testing.assert_allclose(c.data, [8.0, 15.0])


def test_mul_backward():
    check_grad(lambda x: ag.sum(ag.mul(x, ag.tensor(np.array([2.0, 3.0, 4.0])))),
               np.array([1.0, 1.0, 1.0], dtype=np.float32))


# ─── div ─────────────────────────────────────────────────────────────────────

def test_div_backward():
    check_grad(lambda x: ag.sum(ag.div(ag.tensor(np.ones(4) * 6.0), x)),
               np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


# ─── matmul ──────────────────────────────────────────────────────────────────

def test_matmul_forward():
    a = ag.tensor(np.eye(3, dtype=np.float32))
    b = ag.tensor(np.ones((3, 3), dtype=np.float32))
    c = ag.matmul(a, b)
    np.testing.assert_allclose(c.data, np.ones((3, 3)))


def test_matmul_backward_a():
    M, N, K = 4, 3, 5
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)

    a = ag.tensor(a_np, requires_grad=True)
    b = ag.tensor(b_np, requires_grad=False)
    loss = ag.sum(ag.matmul(a, b))
    ag.backward(loss)

    # dL/dA = G @ B^T where G=ones(M,N)
    expected = np.ones((M, N), dtype=np.float32) @ b_np.T
    np.testing.assert_allclose(a.grad, expected, atol=1e-4)


def test_matmul_backward_b():
    M, N, K = 4, 3, 5
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)

    a = ag.tensor(a_np, requires_grad=False)
    b = ag.tensor(b_np, requires_grad=True)
    loss = ag.sum(ag.matmul(a, b))
    ag.backward(loss)

    expected = a_np.T @ np.ones((M, N), dtype=np.float32)
    np.testing.assert_allclose(b.grad, expected, atol=1e-4)


# ─── exp ─────────────────────────────────────────────────────────────────────

def test_exp_backward():
    check_grad(lambda x: ag.sum(ag.exp(x)),
               np.array([0.1, 0.5, -0.3], dtype=np.float32))


# ─── log ─────────────────────────────────────────────────────────────────────

def test_log_backward():
    check_grad(lambda x: ag.sum(ag.log(x)),
               np.array([0.5, 1.0, 2.0], dtype=np.float32))


# ─── relu ─────────────────────────────────────────────────────────────────────

def test_relu_forward():
    x = ag.tensor(np.array([-1.0, 0.5, -0.2, 2.0]))
    y = ag.relu(x)
    np.testing.assert_allclose(y.data, [0.0, 0.5, 0.0, 2.0])


def test_relu_backward():
    check_grad(lambda x: ag.sum(ag.relu(x)),
               np.array([1.0, -0.5, 0.3, -2.0], dtype=np.float32))


# ─── pow ─────────────────────────────────────────────────────────────────────

def test_pow_backward():
    check_grad(lambda x: ag.sum(ag.pow(x, 3.0)),
               np.array([0.5, 1.0, 1.5], dtype=np.float32))


# ─── sum ─────────────────────────────────────────────────────────────────────

def test_sum_backward_all():
    x = ag.tensor(np.ones((3, 4), dtype=np.float32), requires_grad=True)
    loss = ag.sum(x)
    ag.backward(loss)
    np.testing.assert_allclose(x.grad, np.ones((3, 4)))


def test_sum_backward_axis():
    x = ag.tensor(np.ones((3, 4), dtype=np.float32), requires_grad=True)
    y = ag.sum(x, axis=1)     # shape (3,)
    loss = ag.sum(y)
    ag.backward(loss)
    np.testing.assert_allclose(x.grad, np.ones((3, 4)))


# ─── mean ────────────────────────────────────────────────────────────────────

def test_mean_backward():
    x = ag.tensor(np.ones((2, 4), dtype=np.float32) * 2.0, requires_grad=True)
    loss = ag.mean(x)
    ag.backward(loss)
    np.testing.assert_allclose(x.grad, np.ones((2, 4)) / 8.0, atol=1e-6)


# ─── sigmoid ─────────────────────────────────────────────────────────────────

def test_sigmoid_backward():
    check_grad(lambda x: ag.sum(ag.sigmoid(x)),
               np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32))


# ─── tanh ────────────────────────────────────────────────────────────────────

def test_tanh_backward():
    check_grad(lambda x: ag.sum(ag.tanh(x)),
               np.array([0.0, 0.5, -0.5], dtype=np.float32))


# ─── softmax ─────────────────────────────────────────────────────────────────

def test_softmax_forward_sums_to_one():
    x = ag.tensor(np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]))
    s = ag.softmax(x, axis=-1)
    np.testing.assert_allclose(s.data.sum(axis=-1), [1.0, 1.0], atol=1e-6)


def test_softmax_backward():
    x_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    x = ag.tensor(x_np, requires_grad=True)
    s = ag.softmax(x, axis=-1)
    loss = ag.sum(ag.mul(s, ag.tensor(np.array([[1.0, 0.0, 0.0]]))))
    ag.backward(loss)
    # Gradient should exist and be finite
    assert x.grad is not None
    assert np.all(np.isfinite(x.grad))


# ─── cross_entropy ───────────────────────────────────────────────────────────

def test_cross_entropy_forward():
    logits_np = np.array([[2.0, 0.5, 0.1], [0.1, 3.0, 0.5]], dtype=np.float32)
    targets = np.array([0, 1])
    logits = ag.tensor(logits_np)
    loss = ag.cross_entropy(logits, targets)
    # Loss should be positive and finite
    assert loss.item() > 0
    assert np.isfinite(loss.item())


def test_cross_entropy_backward():
    logits_np = np.array([[2.0, 0.5, 0.1]], dtype=np.float32)
    targets = np.array([0])
    logits = ag.tensor(logits_np, requires_grad=True)
    loss = ag.cross_entropy(logits, targets)
    ag.backward(loss)
    assert logits.grad is not None
    assert np.all(np.isfinite(logits.grad))


# ─── chained ops ─────────────────────────────────────────────────────────────

def test_chain_add_mul_sum():
    """(a + b) * c → sum — gradient flows through both add and mul."""
    a = ag.tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    b = ag.tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
    c = ag.tensor(np.array([2.0, 2.0, 2.0]), requires_grad=True)

    loss = ag.sum(ag.mul(ag.add(a, b), c))
    ag.backward(loss)

    # dL/da = c, dL/db = c, dL/dc = a+b
    np.testing.assert_allclose(a.grad, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(b.grad, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(c.grad, [5.0, 7.0, 9.0])


def test_mlp_forward_backward():
    """2-layer MLP: loss = mean(relu(X @ W1) @ W2)"""
    np.random.seed(42)
    X_np = np.random.randn(8, 4).astype(np.float32)
    W1_np = np.random.randn(4, 8).astype(np.float32) * 0.1
    W2_np = np.random.randn(8, 1).astype(np.float32) * 0.1

    X = ag.tensor(X_np)
    W1 = ag.tensor(W1_np, requires_grad=True)
    W2 = ag.tensor(W2_np, requires_grad=True)

    h = ag.relu(ag.matmul(X, W1))   # (8, 8)
    out = ag.matmul(h, W2)           # (8, 1)
    loss = ag.mean(out)

    ag.backward(loss)

    assert W1.grad is not None
    assert W2.grad is not None
    assert W1.grad.shape == W1_np.shape
    assert W2.grad.shape == W2_np.shape
    assert np.all(np.isfinite(W1.grad))
    assert np.all(np.isfinite(W2.grad))


# ─── backward called once clears tape ────────────────────────────────────────

def test_tape_cleared_after_backward():
    from locomp import autograd as _ag_mod
    x = ag.tensor(np.ones(3), requires_grad=True)
    loss = ag.sum(ag.relu(x))
    ag.backward(loss)
    assert len(_ag_mod._tape) == 0


def test_zero_grad():
    x = ag.tensor(np.ones(3), requires_grad=True)
    x.grad = np.ones(3)
    ag.zero_grad(x)
    assert x.grad is None
