"""
locomp ROCm Runtime — Buffer management and kernel launch via libamdhip64.

Uses the HIP Runtime API (libamdhip64.so) via ctypes.
No AMD ROCm Python SDK required — works anywhere hipcc is installed.

Usage:
    from locomp.backends.rocm_runtime import get_runtime, ROCmTensor
    rt = get_runtime()
    t = rt.upload(np.array([1., 2., 3.], dtype=np.float32))
    result = rt.download(t)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import numpy as np
from typing import Optional


# ── dtype helpers ──────────────────────────────────────────────────────────────

_NP_DTYPE_MAP = {
    np.float16: np.float16,
    np.float32: np.float32,
    np.float64: np.float64,
    np.int8:    np.int8,
    np.int16:   np.int16,
    np.int32:   np.int32,
    np.int64:   np.int64,
    np.uint8:   np.uint8,
    np.uint16:  np.uint16,
    np.uint32:  np.uint32,
    np.bool_:   np.int32,
}


class ROCmRuntimeError(Exception):
    pass


def _check(rc: int, msg: str = ""):
    if rc != 0:
        raise ROCmRuntimeError(f"HIP error {rc}" + (f": {msg}" if msg else ""))


# ── ROCmTensor ─────────────────────────────────────────────────────────────────

class ROCmTensor:
    """A GPU-backed tensor wrapping a HIP device pointer.

    Data lives on the AMD GPU. Call .numpy() to read back to host.
    """

    def __init__(self, d_ptr: int, size: int, dtype: np.dtype,
                 shape: tuple, runtime: "ROCmRuntime", owns: bool = True):
        self._hip_ptr: int = d_ptr        # raw device pointer (integer)
        self._size: int = size            # number of elements
        self._dtype: np.dtype = np.dtype(dtype)
        self._shape: tuple = shape
        self._runtime: ROCmRuntime = runtime
        self._owns: bool = owns           # True → free on del
        self._freed: bool = False

    @property
    def size(self) -> int:
        return self._size

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> tuple:
        return self._shape

    def numpy(self) -> np.ndarray:
        """Copy device memory to host and return as numpy array."""
        return self._runtime.download(self)

    def __del__(self):
        if not self._freed and self._owns:
            try:
                self._runtime.free(self)
            except Exception:
                pass

    def free(self):
        if not self._freed and self._owns:
            self._runtime.free(self)
            self._freed = True

    def __repr__(self) -> str:
        return f"ROCmTensor(shape={self._shape}, dtype={self._dtype})"


# ── ROCmRuntime ────────────────────────────────────────────────────────────────

class ROCmRuntime:
    """Singleton ROCm Runtime — hipMalloc, hipFree, hipMemcpy via ctypes."""

    def __init__(self):
        self._lib = self._load_hiprt()
        self._setup_functions()

    def _load_hiprt(self) -> ctypes.CDLL:
        """Load libamdhip64 dynamically."""
        sys = platform.system()
        candidates = []
        if sys == "Linux":
            candidates = [
                "libamdhip64.so",
                "libamdhip64.so.6",
                "libamdhip64.so.5",
                # Common ROCm install paths
                "/opt/rocm/lib/libamdhip64.so",
                "/opt/rocm/lib/libamdhip64.so.6",
                "/opt/rocm/lib/libamdhip64.so.5",
                "/opt/rocm-6.0/lib/libamdhip64.so",
                "/opt/rocm-5.7/lib/libamdhip64.so",
            ]
        elif sys == "Windows":
            candidates = ["amdhip64.dll"]
        elif sys == "Darwin":
            candidates = []  # ROCm is Linux-only

        for lib_name in candidates:
            try:
                return ctypes.CDLL(lib_name)
            except OSError:
                continue

        # Last resort: ctypes.util.find_library
        found = ctypes.util.find_library("amdhip64")
        if found:
            try:
                return ctypes.CDLL(found)
            except OSError:
                pass

        raise ROCmRuntimeError(
            "locomp ROCm runtime: cannot find libamdhip64.\n"
            "Install ROCm: https://docs.amd.com/en/latest/deploy/linux/quick_start.html\n"
            "Or set LD_LIBRARY_PATH to include your ROCm lib directory."
        )

    def _setup_functions(self):
        """Set argtypes/restype for all HIP runtime functions we use."""
        lib = self._lib

        # hipMalloc(void** devPtr, size_t size) → hipError_t
        lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.hipMalloc.restype = ctypes.c_int

        # hipFree(void* devPtr) → hipError_t
        lib.hipFree.argtypes = [ctypes.c_void_p]
        lib.hipFree.restype = ctypes.c_int

        # hipMemcpy(dst, src, count, kind) → hipError_t
        # kind: 1=hipMemcpyHostToDevice, 2=hipMemcpyDeviceToHost
        lib.hipMemcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        ]
        lib.hipMemcpy.restype = ctypes.c_int

        # hipDeviceSynchronize() → hipError_t
        lib.hipDeviceSynchronize.argtypes = []
        lib.hipDeviceSynchronize.restype = ctypes.c_int

        # hipGetDeviceCount(int* count) → hipError_t
        lib.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.hipGetDeviceCount.restype = ctypes.c_int

        # hipSetDevice(int device) → hipError_t
        lib.hipSetDevice.argtypes = [ctypes.c_int]
        lib.hipSetDevice.restype = ctypes.c_int

        # hipGetLastError() → hipError_t
        lib.hipGetLastError.argtypes = []
        lib.hipGetLastError.restype = ctypes.c_int

        # hipMemset(devPtr, value, count) → hipError_t
        lib.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        lib.hipMemset.restype = ctypes.c_int

    # ── memory management ──────────────────────────────────────────────────────

    def alloc_raw(self, n_bytes: int) -> int:
        """Allocate n_bytes on device. Returns device pointer as int."""
        d_ptr = ctypes.c_void_p()
        rc = self._lib.hipMalloc(ctypes.byref(d_ptr), ctypes.c_size_t(n_bytes))
        _check(rc, f"hipMalloc({n_bytes} bytes)")
        return d_ptr.value

    def free_raw(self, d_ptr: int):
        """Free a device pointer."""
        rc = self._lib.hipFree(ctypes.c_void_p(d_ptr))
        _check(rc, "hipFree")

    def upload(self, arr: np.ndarray) -> ROCmTensor:
        """Copy numpy array to device. Returns ROCmTensor."""
        arr = np.ascontiguousarray(arr)
        n_bytes = arr.nbytes
        d_ptr = self.alloc_raw(n_bytes)
        rc = self._lib.hipMemcpy(
            ctypes.c_void_p(d_ptr),
            arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(n_bytes),
            ctypes.c_int(1)   # hipMemcpyHostToDevice
        )
        _check(rc, "hipMemcpy H2D")
        return ROCmTensor(d_ptr, arr.size, arr.dtype, arr.shape, self)

    def download(self, t: ROCmTensor) -> np.ndarray:
        """Copy device memory to host numpy array."""
        out = np.empty(t.size, dtype=t.dtype)
        n_bytes = out.nbytes
        rc = self._lib.hipMemcpy(
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(t._hip_ptr),
            ctypes.c_size_t(n_bytes),
            ctypes.c_int(2)   # hipMemcpyDeviceToHost
        )
        _check(rc, "hipMemcpy D2H")
        return out.reshape(t.shape)

    def zeros(self, size: int, dtype: np.dtype = np.float32) -> ROCmTensor:
        """Allocate zero-filled device tensor."""
        dtype = np.dtype(dtype)
        n_bytes = size * dtype.itemsize
        d_ptr = self.alloc_raw(n_bytes)
        rc = self._lib.hipMemset(ctypes.c_void_p(d_ptr), 0, ctypes.c_size_t(n_bytes))
        _check(rc, "hipMemset")
        return ROCmTensor(d_ptr, size, dtype, (size,), self)

    def empty(self, size: int, dtype: np.dtype = np.float32) -> ROCmTensor:
        """Allocate uninitialized device tensor."""
        dtype = np.dtype(dtype)
        d_ptr = self.alloc_raw(size * dtype.itemsize)
        return ROCmTensor(d_ptr, size, dtype, (size,), self)

    def free(self, t: ROCmTensor):
        """Free device memory for a ROCmTensor."""
        if not t._freed and t._hip_ptr:
            self.free_raw(t._hip_ptr)
            t._freed = True

    def sync(self):
        """Wait for all HIP operations to complete."""
        rc = self._lib.hipDeviceSynchronize()
        _check(rc, "hipDeviceSynchronize")

    def device_count(self) -> int:
        """Return number of ROCm devices available."""
        n = ctypes.c_int(0)
        self._lib.hipGetDeviceCount(ctypes.byref(n))
        return n.value

    def set_device(self, idx: int):
        """Select ROCm device by index."""
        rc = self._lib.hipSetDevice(ctypes.c_int(idx))
        _check(rc, f"hipSetDevice({idx})")

    def is_available(self) -> bool:
        """Return True if at least one ROCm device is present."""
        try:
            return self.device_count() > 0
        except Exception:
            return False


# ── singleton ─────────────────────────────────────────────────────────────────

_runtime: Optional[ROCmRuntime] = None


def get_runtime() -> ROCmRuntime:
    """Return the global ROCmRuntime singleton (lazy init)."""
    global _runtime
    if _runtime is None:
        _runtime = ROCmRuntime()
    return _runtime


def is_available() -> bool:
    """Return True if ROCm runtime is available on this machine."""
    try:
        rt = get_runtime()
        return rt.is_available()
    except ROCmRuntimeError:
        return False
