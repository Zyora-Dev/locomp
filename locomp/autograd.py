"""
locomp Autograd — tape-based reverse-mode automatic differentiation.

Works above the kernel layer: each operation registers a backward function
on a global tape. Call `backward()` on a scalar loss to propagate gradients.

Supported ops:
    add, sub, mul, div, matmul, exp, log, relu, sum, mean, pow (scalar exp)

Usage:
    a = locomp.tensor(np.random.randn(N), requires_grad=True)
    b = locomp.tensor(np.random.randn(N), requires_grad=True)
    c = locomp.ag.add(a, b)          # element-wise add
    loss = locomp.ag.sum(c)          # scalar
    locomp.ag.backward(loss)
    print(a.grad)  # dL/da
    print(b.grad)  # dL/db
"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional, Tuple

# ─── Tape ────────────────────────────────────────────────────────────────────

# Each entry: (backward_fn, output_tensor, input_tensors_that_need_grad)
_tape: List[Tuple[Callable, "AGTensor", List["AGTensor"]]] = []
_tape_enabled = True


def no_grad():
    """Context manager: disable tape recording."""
    class _NoGrad:
        def __enter__(self):
            global _tape_enabled
            self._prev = _tape_enabled
            _tape_enabled = False
        def __exit__(self, *_):
            global _tape_enabled
            _tape_enabled = self._prev
    return _NoGrad()


def zero_grad(*tensors: "AGTensor"):
    """Zero out .grad on all given tensors."""
    for t in tensors:
        t.grad = None


# ─── AGTensor ────────────────────────────────────────────────────────────────

class AGTensor:
    """
    A thin wrapper around a NumPy array that participates in autograd.

    Attributes:
        data (np.ndarray): The forward-pass value.
        grad (np.ndarray | None): Accumulated gradient (same shape as data).
        requires_grad (bool): Whether to track gradients for this tensor.
    """

    __slots__ = ("data", "grad", "requires_grad", "_backward_fn", "shape", "dtype")

    def __init__(self, data: np.ndarray, requires_grad: bool = False):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad
        self._backward_fn: Optional[Callable[[], None]] = None
        self.shape = self.data.shape
        self.dtype = self.data.dtype

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return float(self.data)

    def zero_grad(self):
        self.grad = None

    def _accumulate_grad(self, g: np.ndarray):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += g

    def __repr__(self):
        return f"AGTensor(shape={self.shape}, requires_grad={self.requires_grad})"


def tensor(data, requires_grad: bool = False) -> AGTensor:
    """Create an AGTensor from a NumPy array or scalar."""
    return AGTensor(np.asarray(data, dtype=np.float32), requires_grad=requires_grad)


# ─── Internal: record op on tape ─────────────────────────────────────────────

def _record(out: AGTensor, backward_fn: Callable, inputs: List[AGTensor]):
    """Attach a backward function to `out` if tape is active and any input needs grad."""
    if not _tape_enabled:
        return
    if not any(t.requires_grad for t in inputs):
        return
    out.requires_grad = True

    # Capture in closure
    def _bwd():
        backward_fn(out.grad if out.grad is not None else np.ones_like(out.data))

    out._backward_fn = _bwd
    _tape.append((out, inputs))


# ─── backward pass ───────────────────────────────────────────────────────────

def backward(loss: AGTensor):
    """
    Run reverse-mode autodiff from `loss` (must be scalar).

    Walks the tape in reverse and calls each op's backward function.
    """
    if loss.data.size != 1:
        raise ValueError(
            f"backward() requires a scalar loss, got shape {loss.shape}. "
            "Use locomp.ag.sum(x) to reduce to scalar first."
        )
    # Seed gradient
    if loss.grad is None:
        loss.grad = np.ones((), dtype=np.float32)

    # Topological order: tape was built in forward order, reverse to get backward order
    # Find all reachable nodes via DFS from loss
    visited = set()
    order = []

    def _topo(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        # find the tape entry for this node
        for out, inputs in _tape:
            if out is node:
                for inp in inputs:
                    _topo(inp)
                order.append(node)
                return
        # leaf node (no tape entry)
        order.append(node)

    _topo(loss)

    for node in reversed(order):
        if node._backward_fn is not None:
            node._backward_fn()

    # Clear tape
    _tape.clear()


# ─── Op implementations ───────────────────────────────────────────────────────

def add(a: AGTensor, b: AGTensor) -> AGTensor:
    """Element-wise addition: out = a + b"""
    out = tensor(a.data + b.data)

    def _bwd(g):
        if a.requires_grad:
            a._accumulate_grad(_unbroadcast(g, a.shape))
        if b.requires_grad:
            b._accumulate_grad(_unbroadcast(g, b.shape))

    _record(out, _bwd, [a, b])
    return out


def sub(a: AGTensor, b: AGTensor) -> AGTensor:
    """Element-wise subtraction: out = a - b"""
    out = tensor(a.data - b.data)

    def _bwd(g):
        if a.requires_grad:
            a._accumulate_grad(_unbroadcast(g, a.shape))
        if b.requires_grad:
            b._accumulate_grad(_unbroadcast(-g, b.shape))

    _record(out, _bwd, [a, b])
    return out


def mul(a: AGTensor, b: AGTensor) -> AGTensor:
    """Element-wise multiplication: out = a * b"""
    a_data, b_data = a.data.copy(), b.data.copy()
    out = tensor(a_data * b_data)

    def _bwd(g):
        if a.requires_grad:
            a._accumulate_grad(_unbroadcast(g * b_data, a.shape))
        if b.requires_grad:
            b._accumulate_grad(_unbroadcast(g * a_data, b.shape))

    _record(out, _bwd, [a, b])
    return out


def div(a: AGTensor, b: AGTensor) -> AGTensor:
    """Element-wise division: out = a / b"""
    a_data, b_data = a.data.copy(), b.data.copy()
    out = tensor(a_data / b_data)

    def _bwd(g):
        if a.requires_grad:
            a._accumulate_grad(_unbroadcast(g / b_data, a.shape))
        if b.requires_grad:
            b._accumulate_grad(_unbroadcast(-g * a_data / (b_data ** 2), b.shape))

    _record(out, _bwd, [a, b])
    return out


def matmul(a: AGTensor, b: AGTensor) -> AGTensor:
    """Matrix multiply: out = a @ b"""
    a_data, b_data = a.data.copy(), b.data.copy()
    out = tensor(a_data @ b_data)

    def _bwd(g):
        if a.requires_grad:
            a._accumulate_grad(g @ b_data.T)
        if b.requires_grad:
            b._accumulate_grad(a_data.T @ g)

    _record(out, _bwd, [a, b])
    return out


def exp(a: AGTensor) -> AGTensor:
    """Element-wise exp: out = e^a"""
    out_data = np.exp(a.data)
    out = tensor(out_data)

    def _bwd(g):
        if a.requires_grad:
            a._accumulate_grad(g * out_data)

    _record(out, _bwd, [a])
    return out


def log(a: AGTensor) -> AGTensor:
    """Element-wise natural log: out = log(a)"""
    a_data = a.data.copy()
    out = tensor(np.log(a_data))

    def _bwd(g):
        if a.requires_grad:
            a._accumulate_grad(g / a_data)

    _record(out, _bwd, [a])
    return out


def relu(a: AGTensor) -> AGTensor:
    """Element-wise ReLU: out = max(a, 0)"""
    mask = (a.data > 0).astype(np.float32)
    out = tensor(a.data * mask)

    def _bwd(g):
        if a.requires_grad:
            a._accumulate_grad(g * mask)

    _record(out, _bwd, [a])
    return out


def pow(a: AGTensor, exponent: float) -> AGTensor:
    """Element-wise power: out = a ^ exponent (scalar exponent only)"""
    a_data = a.data.copy()
    out = tensor(a_data ** exponent)

    def _bwd(g):
        if a.requires_grad:
            a._accumulate_grad(g * exponent * a_data ** (exponent - 1))

    _record(out, _bwd, [a])
    return out


def sum(a: AGTensor, axis=None, keepdims: bool = False) -> AGTensor:
    """Sum reduction."""
    out = tensor(np.sum(a.data, axis=axis, keepdims=keepdims))

    def _bwd(g):
        if a.requires_grad:
            a._accumulate_grad(np.broadcast_to(np.expand_dims(g, axis=axis) if (axis is not None and not keepdims) else g, a.shape).copy())

    _record(out, _bwd, [a])
    return out


def mean(a: AGTensor, axis=None, keepdims: bool = False) -> AGTensor:
    """Mean reduction."""
    n = a.data.size if axis is None else a.data.shape[axis]
    out = tensor(np.mean(a.data, axis=axis, keepdims=keepdims))

    def _bwd(g):
        if a.requires_grad:
            g_expanded = np.expand_dims(g, axis=axis) if (axis is not None and not keepdims) else g
            a._accumulate_grad(np.broadcast_to(g_expanded, a.shape).copy() / n)

    _record(out, _bwd, [a])
    return out


def sigmoid(a: AGTensor) -> AGTensor:
    """Sigmoid: out = 1 / (1 + e^-a)"""
    s = 1.0 / (1.0 + np.exp(-a.data))
    out = tensor(s)

    def _bwd(g):
        if a.requires_grad:
            a._accumulate_grad(g * s * (1.0 - s))

    _record(out, _bwd, [a])
    return out


def tanh(a: AGTensor) -> AGTensor:
    """Tanh: out = tanh(a)"""
    t = np.tanh(a.data)
    out = tensor(t)

    def _bwd(g):
        if a.requires_grad:
            a._accumulate_grad(g * (1.0 - t ** 2))

    _record(out, _bwd, [a])
    return out


def softmax(a: AGTensor, axis: int = -1) -> AGTensor:
    """Softmax along given axis."""
    e = np.exp(a.data - np.max(a.data, axis=axis, keepdims=True))
    s = e / np.sum(e, axis=axis, keepdims=True)
    out = tensor(s)

    def _bwd(g):
        if a.requires_grad:
            # d(softmax)/dx = s * (g - (g * s).sum(axis, keepdims=True))
            dot = np.sum(g * s, axis=axis, keepdims=True)
            a._accumulate_grad(s * (g - dot))

    _record(out, _bwd, [a])
    return out


def cross_entropy(logits: AGTensor, targets: np.ndarray) -> AGTensor:
    """
    Softmax cross-entropy loss.

    Args:
        logits: shape (N, C)
        targets: integer class indices, shape (N,)
    Returns:
        scalar loss
    """
    N = logits.data.shape[0]
    e = np.exp(logits.data - np.max(logits.data, axis=1, keepdims=True))
    probs = e / np.sum(e, axis=1, keepdims=True)
    log_probs = np.log(probs[np.arange(N), targets] + 1e-9)
    loss_val = -np.mean(log_probs)
    out = tensor(loss_val)

    def _bwd(g):
        if logits.requires_grad:
            grad = probs.copy()
            grad[np.arange(N), targets] -= 1.0
            grad /= N
            logits._accumulate_grad(g * grad)

    _record(out, _bwd, [logits])
    return out


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _unbroadcast(g: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum gradient axes that were broadcast during the forward pass."""
    if g.shape == shape:
        return g
    # Sum over leading axes if shapes differ in rank
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    # Sum over axes where shape is 1
    for i, (gs, ss) in enumerate(zip(g.shape, shape)):
        if ss == 1 and gs > 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)
