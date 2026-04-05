"""
locomp CUDA Runtime — Buffer management and kernel launch via libcudart.

Uses the CUDA Runtime API (libcudart.so / cudart64.dll) via ctypes.
No CUDA Python SDK required — works anywhere nvcc is installed.

Usage:
    from locomp.backends.cuda_runtime import get_runtime, CUDATensor
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


class CUDARuntimeError(Exception):
    pass


def _check(rc: int, msg: str = ""):
    if rc != 0:
        raise CUDARuntimeError(f"CUDA error {rc}" + (f": {msg}" if msg else ""))


# ── CUDATensor ─────────────────────────────────────────────────────────────────

class CUDATensor:
    """A GPU-backed tensor wrapping a CUDA device pointer.

    Data lives on the NVIDIA GPU. Call .numpy() to read back to host.
    """

    def __init__(self, d_ptr: int, size: int, dtype: np.dtype,
                 shape: tuple, runtime: "CUDARuntime", owns: bool = True):
        self._cuda_ptr: int = d_ptr       # raw device pointer (integer)
        self._size: int = size            # number of elements
        self._dtype: np.dtype = np.dtype(dtype)
        self._shape: tuple = shape
        self._runtime: CUDARuntime = runtime
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
        return f"CUDATensor(shape={self._shape}, dtype={self._dtype})"


# ── CUDARuntime ────────────────────────────────────────────────────────────────

class CUDARuntime:
    """Singleton CUDA Runtime — cudaMalloc, cudaFree, cudaMemcpy via ctypes."""

    def __init__(self):
        self._lib = self._load_cudart()
        self._setup_functions()

    def _load_cudart(self) -> ctypes.CDLL:
        """Load libcudart dynamically."""
        sys = platform.system()
        candidates = []
        if sys == "Linux":
            candidates = [
                "libcudart.so",
                "libcudart.so.12",
                "libcudart.so.11",
                # Common install paths
                "/usr/local/cuda/lib64/libcudart.so",
                "/usr/local/cuda/lib64/libcudart.so.12",
                "/usr/local/cuda/lib64/libcudart.so.11",
            ]
        elif sys == "Windows":
            candidates = [f"cudart64_{v}.dll" for v in ("120", "110", "100")]
        elif sys == "Darwin":
            candidates = []  # No CUDA on macOS (except old pre-Apple-Silicon)

        for lib_name in candidates:
            try:
                return ctypes.CDLL(lib_name)
            except OSError:
                continue

        # Last resort: use ctypes.util.find_library
        found = ctypes.util.find_library("cudart")
        if found:
            try:
                return ctypes.CDLL(found)
            except OSError:
                pass

        raise CUDARuntimeError(
            "locomp CUDA runtime: cannot find libcudart.\n"
            "Install CUDA: https://developer.nvidia.com/cuda-downloads\n"
            "Or set LD_LIBRARY_PATH to include your CUDA lib64 directory."
        )

    def _setup_functions(self):
        """Set argtypes/restype for all CUDA runtime functions we use."""
        lib = self._lib

        # cudaMalloc(void** devPtr, size_t size) → cudaError_t
        lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.cudaMalloc.restype = ctypes.c_int

        # cudaFree(void* devPtr) → cudaError_t
        lib.cudaFree.argtypes = [ctypes.c_void_p]
        lib.cudaFree.restype = ctypes.c_int

        # cudaMemcpy(dst, src, count, kind) → cudaError_t
        # kind: 1=HostToDevice, 2=DeviceToHost
        lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        ]
        lib.cudaMemcpy.restype = ctypes.c_int

        # cudaDeviceSynchronize() → cudaError_t
        lib.cudaDeviceSynchronize.argtypes = []
        lib.cudaDeviceSynchronize.restype = ctypes.c_int

        # cudaGetDeviceCount(int* count) → cudaError_t
        lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cudaGetDeviceCount.restype = ctypes.c_int

        # cudaSetDevice(int device) → cudaError_t
        lib.cudaSetDevice.argtypes = [ctypes.c_int]
        lib.cudaSetDevice.restype = ctypes.c_int

        # cudaGetLastError() → cudaError_t
        lib.cudaGetLastError.argtypes = []
        lib.cudaGetLastError.restype = ctypes.c_int

        # cudaMemset(devPtr, value, count) → cudaError_t
        lib.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        lib.cudaMemset.restype = ctypes.c_int

    # ── memory management ──────────────────────────────────────────────────────

    def alloc_raw(self, n_bytes: int) -> int:
        """Allocate n_bytes on device. Returns device pointer as int."""
        d_ptr = ctypes.c_void_p()
        rc = self._lib.cudaMalloc(ctypes.byref(d_ptr), ctypes.c_size_t(n_bytes))
        _check(rc, f"cudaMalloc({n_bytes} bytes)")
        return d_ptr.value

    def free_raw(self, d_ptr: int):
        """Free a device pointer."""
        rc = self._lib.cudaFree(ctypes.c_void_p(d_ptr))
        _check(rc, "cudaFree")

    def upload(self, arr: np.ndarray) -> CUDATensor:
        """Copy numpy array to device. Returns CUDATensor."""
        arr = np.ascontiguousarray(arr)
        n_bytes = arr.nbytes
        d_ptr = self.alloc_raw(n_bytes)
        rc = self._lib.cudaMemcpy(
            ctypes.c_void_p(d_ptr),
            arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(n_bytes),
            ctypes.c_int(1)   # cudaMemcpyHostToDevice
        )
        _check(rc, "cudaMemcpy H2D")
        return CUDATensor(d_ptr, arr.size, arr.dtype, arr.shape, self)

    def download(self, t: CUDATensor) -> np.ndarray:
        """Copy device memory to host numpy array."""
        out = np.empty(t.size, dtype=t.dtype)
        n_bytes = out.nbytes
        rc = self._lib.cudaMemcpy(
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(t._cuda_ptr),
            ctypes.c_size_t(n_bytes),
            ctypes.c_int(2)   # cudaMemcpyDeviceToHost
        )
        _check(rc, "cudaMemcpy D2H")
        return out.reshape(t.shape)

    def zeros(self, size: int, dtype: np.dtype = np.float32) -> CUDATensor:
        """Allocate zero-filled device tensor."""
        dtype = np.dtype(dtype)
        n_bytes = size * dtype.itemsize
        d_ptr = self.alloc_raw(n_bytes)
        rc = self._lib.cudaMemset(ctypes.c_void_p(d_ptr), 0, ctypes.c_size_t(n_bytes))
        _check(rc, "cudaMemset")
        return CUDATensor(d_ptr, size, dtype, (size,), self)

    def empty(self, size: int, dtype: np.dtype = np.float32) -> CUDATensor:
        """Allocate uninitialized device tensor."""
        dtype = np.dtype(dtype)
        d_ptr = self.alloc_raw(size * dtype.itemsize)
        return CUDATensor(d_ptr, size, dtype, (size,), self)

    def free(self, t: CUDATensor):
        """Free device memory for a CUDATensor."""
        if not t._freed and t._cuda_ptr:
            self.free_raw(t._cuda_ptr)
            t._freed = True

    def sync(self):
        """Wait for all CUDA operations to complete."""
        rc = self._lib.cudaDeviceSynchronize()
        _check(rc, "cudaDeviceSynchronize")

    def device_count(self) -> int:
        """Return number of CUDA devices available."""
        n = ctypes.c_int(0)
        self._lib.cudaGetDeviceCount(ctypes.byref(n))
        return n.value

    def set_device(self, idx: int):
        """Select CUDA device by index."""
        rc = self._lib.cudaSetDevice(ctypes.c_int(idx))
        _check(rc, f"cudaSetDevice({idx})")

    def is_available(self) -> bool:
        """Return True if at least one CUDA device is present."""
        try:
            return self.device_count() > 0
        except Exception:
            return False


# ── singleton ─────────────────────────────────────────────────────────────────

_runtime: Optional[CUDARuntime] = None


def get_runtime() -> CUDARuntime:
    """Return the global CUDARuntime singleton (lazy init)."""
    global _runtime
    if _runtime is None:
        _runtime = CUDARuntime()
    return _runtime


def is_available() -> bool:
    """Return True if CUDA runtime is available on this machine."""
    try:
        rt = get_runtime()
        return rt.is_available()
    except CUDARuntimeError:
        return False
