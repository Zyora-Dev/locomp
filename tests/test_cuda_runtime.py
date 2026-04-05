"""
Tests for locomp CUDA runtime (cuda_runtime.py).

Most tests run on macOS / CI without CUDA and verify:
  - Import succeeds (no crash on import)
  - `is_available()` returns False gracefully without a GPU
  - `cuda_available()` exposed via locomp namespace
  - `CUDATensor` class is importable and type-checks correctly
  - `tensor(backend="auto")` returns LocompTensor on non-CUDA hosts
  - `tensor(backend="cuda")` raises CUDARuntimeError on non-CUDA hosts

Tests marked `nvidia_only` are skipped unless CUDA is present.
"""

import platform
import numpy as np
import pytest

import locomp
from locomp.backends.cuda_runtime import (
    CUDATensor,
    CUDARuntime,
    CUDARuntimeError,
    is_available,
    get_runtime,
)
from locomp.api import LocompTensor

HAS_CUDA = is_available()
nvidia_only = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


# ─────────────────────────────────────────────────────────────────────────────
# Import / namespace tests (always run)
# ─────────────────────────────────────────────────────────────────────────────

def test_cuda_runtime_importable():
    """cuda_runtime module can be imported without side effects."""
    from locomp.backends import cuda_runtime  # noqa: F401


def test_cudatensor_class_accessible_from_locomp():
    assert locomp.CUDATensor is CUDATensor


def test_cuda_available_in_locomp_namespace():
    result = locomp.cuda_available()
    assert isinstance(result, bool)


def test_is_available_returns_bool():
    result = is_available()
    assert isinstance(result, bool)


def test_is_available_false_on_macos():
    if platform.system() == "Darwin":
        assert is_available() is False


# ─────────────────────────────────────────────────────────────────────────────
# Factory functions — non-CUDA host (backend="auto" / default)
# ─────────────────────────────────────────────────────────────────────────────

def test_tensor_default_returns_locomp_tensor_on_non_cuda():
    if HAS_CUDA:
        pytest.skip("Only meaningful on non-CUDA host")
    t = locomp.tensor([1.0, 2.0, 3.0])
    assert isinstance(t, LocompTensor)


def test_zeros_default_returns_locomp_tensor_on_non_cuda():
    if HAS_CUDA:
        pytest.skip("Only meaningful on non-CUDA host")
    t = locomp.zeros((4,))
    assert isinstance(t, LocompTensor)


def test_empty_default_returns_locomp_tensor_on_non_cuda():
    if HAS_CUDA:
        pytest.skip("Only meaningful on non-CUDA host")
    t = locomp.empty((4,))
    assert isinstance(t, LocompTensor)


def test_ones_default_returns_locomp_tensor_on_non_cuda():
    if HAS_CUDA:
        pytest.skip("Only meaningful on non-CUDA host")
    t = locomp.ones((4,))
    assert isinstance(t, LocompTensor)


def test_tensor_backend_cuda_raises_on_no_gpu():
    if HAS_CUDA:
        pytest.skip("Only runs without CUDA")
    with pytest.raises((CUDARuntimeError, OSError, Exception)):
        locomp.tensor([1.0, 2.0], backend="cuda")


def test_tensor_backend_metal_bypasses_cuda():
    # Even on a CUDA machine, backend="metal" should give LocompTensor
    t = locomp.tensor([1.0, 2.0], backend="metal")
    assert isinstance(t, LocompTensor)


def test_tensor_backend_cpu_bypasses_cuda():
    t = locomp.tensor([1.0, 2.0], backend="cpu")
    assert isinstance(t, LocompTensor)


# ─────────────────────────────────────────────────────────────────────────────
# CUDATensor class / type tests (no GPU required)
# ─────────────────────────────────────────────────────────────────────────────

def test_cudatensor_has_required_attributes():
    """CUDATensor exposes the same interface expected by _launch_cuda."""
    # Build a mock CUDATensor without allocating GPU memory
    class _FakeRT:
        def download(self, t):
            return np.zeros(t.size, dtype=t.dtype).reshape(t.shape)
        def free(self, t): pass

    t = CUDATensor(
        d_ptr=0x1000,
        size=4,
        dtype=np.float32,
        shape=(4,),
        runtime=_FakeRT(),
        owns=False,  # don't attempt real free
    )
    assert hasattr(t, "_cuda_ptr")
    assert hasattr(t, "size")
    assert hasattr(t, "dtype")
    assert hasattr(t, "shape")
    assert t._cuda_ptr == 0x1000
    assert t.size == 4
    assert t.dtype == np.float32
    assert t.shape == (4,)


def test_cudatensor_numpy_calls_runtime_download():
    """CUDATensor.numpy() delegates to runtime.download()."""
    called = []

    class _FakeRT:
        def download(self, t):
            called.append(True)
            return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        def free(self, t): pass

    t = CUDATensor(0, 4, np.float32, (4,), _FakeRT(), owns=False)
    result = t.numpy()
    assert called, "download() should have been called"
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])


def test_cudatensor_repr():
    class _FakeRT:
        def free(self, t): pass

    t = CUDATensor(0, 3, np.float16, (3,), _FakeRT(), owns=False)
    r = repr(t)
    assert "CUDATensor" in r
    assert "float16" in r


# ─────────────────────────────────────────────────────────────────────────────
# CUDARuntime instantiation error (no CUDA, graceful)
# ─────────────────────────────────────────────────────────────────────────────

def test_get_runtime_raises_cuda_error_without_gpu():
    if HAS_CUDA:
        pytest.skip("Only runs without CUDA")
    # get_runtime should raise CUDARuntimeError if libcudart not found
    import locomp.backends.cuda_runtime as _m
    # Reset singleton so we get a fresh attempt
    old = _m._runtime
    _m._runtime = None
    try:
        with pytest.raises((CUDARuntimeError, OSError)):
            get_runtime()
    finally:
        _m._runtime = old


# ─────────────────────────────────────────────────────────────────────────────
# CUDA runtime integration tests (nvidia_only)
# ─────────────────────────────────────────────────────────────────────────────

@nvidia_only
def test_cuda_runtime_device_count():
    rt = get_runtime()
    assert rt.device_count() >= 1


@nvidia_only
def test_cuda_upload_download_roundtrip():
    rt = get_runtime()
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    t = rt.upload(arr)
    try:
        result = t.numpy()
        np.testing.assert_allclose(result, arr)
    finally:
        t.free()


@nvidia_only
def test_cuda_zeros_roundtrip():
    rt = get_runtime()
    t = rt.zeros(8, np.float32)
    try:
        result = t.numpy()
        np.testing.assert_array_equal(result, np.zeros(8, dtype=np.float32))
    finally:
        t.free()


@nvidia_only
def test_cuda_tensor_factory_returns_cudatensor():
    t = locomp.tensor([1.0, 2.0, 3.0], backend="cuda")
    assert isinstance(t, CUDATensor)
    t.free()


@nvidia_only
def test_cuda_zeros_factory_returns_cudatensor():
    t = locomp.zeros((4,), backend="cuda")
    assert isinstance(t, CUDATensor)
    t.free()


@nvidia_only
def test_cuda_empty_factory_shape():
    t = locomp.empty((3, 4), backend="cuda")
    assert isinstance(t, CUDATensor)
    assert t.shape == (3, 4)
    assert t.size == 12
    t.free()


@nvidia_only
def test_cuda_sync_no_error():
    rt = get_runtime()
    rt.sync()  # should not raise


@nvidia_only
def test_cuda_2d_upload_download():
    rt = get_runtime()
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = rt.upload(arr)
    try:
        result = t.numpy()
        np.testing.assert_allclose(result, arr)
    finally:
        t.free()


@nvidia_only
def test_cuda_kernel_vector_add():
    """End-to-end: compile + run a vector-add kernel on NVIDIA GPU."""
    import locomp

    @locomp.kernel(backend="cuda")
    def vec_add(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        b = locomp.load(B + i)
        locomp.store(Out + i, a + b)

    N = 1024
    rt = get_runtime()
    a = rt.upload(np.ones(N, dtype=np.float32))
    b = rt.upload(np.ones(N, dtype=np.float32) * 2.0)
    out = rt.zeros(N, np.float32)

    vec_add[(N,)](a, b, out, N=N)

    result = out.numpy()
    np.testing.assert_allclose(result, np.full(N, 3.0, dtype=np.float32))

    a.free(); b.free(); out.free()
