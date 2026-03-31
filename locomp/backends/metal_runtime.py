"""
Locust Metal Runtime — Dispatches compiled MSL kernels on Apple Silicon GPU.

Uses PyObjC to interface with Apple's Metal framework:
1. Creates Metal device (GPU handle)
2. Compiles MSL source → Metal library → compute pipeline
3. Manages GPU buffers (allocate, upload, download)
4. Dispatches kernel execution via command queue

This is the bridge between Locust IR/codegen and the actual Apple GPU.
"""

from __future__ import annotations

import ctypes
import os
import platform
import struct
from typing import Optional

import numpy as np


def _check_metal_available():
    if platform.system() != "Darwin":
        raise RuntimeError("Metal backend requires macOS (Apple Silicon)")


def _load_fast_dispatch():
    """Load the native C bridge for fast Metal dispatch."""
    dylib = os.path.join(os.path.dirname(__file__), "..", "_native", "fast_dispatch.dylib")
    dylib = os.path.abspath(dylib)
    if not os.path.exists(dylib):
        return None
    lib = ctypes.cdll.LoadLibrary(dylib)
    lib.locust_dispatch.restype = ctypes.c_double
    lib.locust_dispatch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.locust_dispatch_async.restype = None
    lib.locust_dispatch_async.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.locust_sync.restype = ctypes.c_double
    lib.locust_sync.argtypes = []
    lib.locust_dispatch_repeat.restype = ctypes.c_double
    lib.locust_dispatch_repeat.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int,
    ]
    return lib


class MetalRuntime:
    """Runtime for executing compiled MSL kernels on Apple GPU."""

    def __init__(self):
        _check_metal_available()

        import Metal
        import MetalPerformanceShaders as MPS

        self._Metal = Metal
        self._MPS = MPS

        # Get default GPU device
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal-compatible GPU found")

        self.device_name = self.device.name()
        self.command_queue = self.device.newCommandQueue()

        # Cache compiled pipelines
        self._pipeline_cache: dict[str, any] = {}
        # Cache int buffers to avoid repeated allocation
        self._int_buffer_cache: dict[int, any] = {}

        # Native C bridge for fast dispatch
        self._fast_lib = _load_fast_dispatch()
        if self._fast_lib is not None:
            import objc
            self._objc = objc
            self._queue_ptr = objc.pyobjc_id(self.command_queue)

    def compile_msl(self, source: str, kernel_name: str):
        """Compile MSL source code into a compute pipeline state."""
        cache_key = f"{kernel_name}:{hash(source)}"
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]

        Metal = self._Metal

        # Compile MSL source → Metal library
        options = Metal.MTLCompileOptions.alloc().init()
        library, error = self.device.newLibraryWithSource_options_error_(
            source, options, None
        )
        if error is not None:
            raise RuntimeError(f"Metal MSL compilation failed:\n{error}")

        # Get kernel function from library
        func = library.newFunctionWithName_(kernel_name)
        if func is None:
            raise RuntimeError(f"Kernel function '{kernel_name}' not found in compiled MSL")

        # Create compute pipeline state
        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
            func, None
        )
        if error is not None:
            raise RuntimeError(f"Metal pipeline creation failed:\n{error}")

        self._pipeline_cache[cache_key] = pipeline
        return pipeline

    def allocate_buffer(self, data: np.ndarray) -> any:
        """Allocate a Metal buffer and upload numpy data."""
        Metal = self._Metal
        byte_data = data.tobytes()
        buffer = self.device.newBufferWithBytes_length_options_(
            byte_data, len(byte_data),
            Metal.MTLResourceStorageModeShared
        )
        if buffer is None:
            raise RuntimeError(f"Failed to allocate Metal buffer of {len(byte_data)} bytes")
        return buffer

    def allocate_empty_buffer(self, size_bytes: int) -> any:
        """Allocate an empty Metal buffer."""
        Metal = self._Metal
        buffer = self.device.newBufferWithLength_options_(
            size_bytes, Metal.MTLResourceStorageModeShared
        )
        if buffer is None:
            raise RuntimeError(f"Failed to allocate Metal buffer of {size_bytes} bytes")
        return buffer

    def allocate_int_buffer(self, value: int) -> any:
        """Allocate a Metal buffer containing a single int32 (cached)."""
        if value in self._int_buffer_cache:
            return self._int_buffer_cache[value]
        data = struct.pack("i", value)
        Metal = self._Metal
        buffer = self.device.newBufferWithBytes_length_options_(
            data, len(data), Metal.MTLResourceStorageModeShared
        )
        self._int_buffer_cache[value] = buffer
        return buffer

    def read_buffer(self, buffer, dtype, count: int) -> np.ndarray:
        """Read data from a Metal buffer back to numpy."""
        dt = np.dtype(dtype)
        byte_length = count * dt.itemsize
        contents = buffer.contents()
        mv = contents.as_buffer(byte_length)
        return np.frombuffer(mv, dtype=dt, count=count).copy()

    def dispatch(self, pipeline, buffers: list, grid: tuple[int, ...],
                 threadgroup_size: Optional[tuple[int, ...]] = None):
        """Dispatch a compute kernel on the GPU (async when C bridge available)."""
        # Default threadgroup size: 1 thread per group (each group = one program)
        if threadgroup_size is None:
            threadgroup_size = (1, 1, 1)

        # Pad grid to 3D
        while len(grid) < 3:
            grid = grid + (1,)
        while len(threadgroup_size) < 3:
            threadgroup_size = threadgroup_size + (1,)

        # Fast path: native C bridge (async — no wait)
        if self._fast_lib is not None:
            objc = self._objc
            pipeline_ptr = objc.pyobjc_id(pipeline)
            buf_ptrs = (ctypes.c_void_p * len(buffers))(
                *[objc.pyobjc_id(b) for b in buffers]
            )
            self._fast_lib.locust_dispatch_async(
                self._queue_ptr, pipeline_ptr,
                buf_ptrs, len(buffers),
                grid[0], grid[1], grid[2],
                threadgroup_size[0], threadgroup_size[1], threadgroup_size[2],
            )
            self._last_gpu_start = 0.0
            self._last_gpu_end = 0.0
            return

        # Slow path: PyObjC (synchronous)
        Metal = self._Metal

        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)

        # Bind buffers
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

        # Dispatch threadgroups: grid = number of threadgroups
        grid_size = Metal.MTLSizeMake(*grid)
        tg_size = Metal.MTLSizeMake(*threadgroup_size)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Store GPU timing for profiling
        self._last_gpu_start = command_buffer.GPUStartTime()
        self._last_gpu_end = command_buffer.GPUEndTime()

        # Check for errors
        error = command_buffer.error()
        if error is not None:
            raise RuntimeError(f"Metal kernel execution failed: {error}")

    def sync(self):
        """Wait for any pending async GPU work to complete."""
        if self._fast_lib is not None:
            self._fast_lib.locust_sync()

    def dispatch_repeat(self, pipeline, buffers: list, grid: tuple[int, ...],
                        threadgroup_size: tuple[int, ...], repeat: int) -> float:
        """Dispatch same kernel N times in one command buffer. Returns GPU time in ms."""
        Metal = self._Metal
        while len(grid) < 3:
            grid = grid + (1,)
        while len(threadgroup_size) < 3:
            threadgroup_size = threadgroup_size + (1,)

        command_buffer = self.command_queue.commandBuffer()
        for _ in range(repeat):
            encoder = command_buffer.computeCommandEncoder()
            encoder.setComputePipelineState_(pipeline)
            for i, buf in enumerate(buffers):
                encoder.setBuffer_offset_atIndex_(buf, 0, i)
            grid_size = Metal.MTLSizeMake(*grid)
            tg_size = Metal.MTLSizeMake(*threadgroup_size)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
            encoder.endEncoding()

        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        gpu_ms = (command_buffer.GPUEndTime() - command_buffer.GPUStartTime()) * 1000
        return gpu_ms / repeat


# Global runtime instance (lazy init)
_runtime: Optional[MetalRuntime] = None


def get_runtime() -> MetalRuntime:
    """Get or create the global Metal runtime."""
    global _runtime
    if _runtime is None:
        _runtime = MetalRuntime()
    return _runtime
