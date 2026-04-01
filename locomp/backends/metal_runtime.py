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

    def __init__(self, device_index: int = 0):
        _check_metal_available()

        import Metal
        import MetalPerformanceShaders as MPS

        self._Metal = Metal
        self._MPS = MPS

        # Device selection: pick by index from available devices
        all_devices = Metal.MTLCopyAllDevices()
        if not all_devices or len(all_devices) == 0:
            # Fallback to default device
            self.device = Metal.MTLCreateSystemDefaultDevice()
        elif device_index < len(all_devices):
            self.device = all_devices[device_index]
        else:
            raise RuntimeError(
                f"GPU device index {device_index} out of range. "
                f"Available devices: {len(all_devices)}"
            )
        if self.device is None:
            raise RuntimeError("No Metal-compatible GPU found")

        self.device_name = self.device.name()
        self.command_queue = self.device.newCommandQueue()

        # Hardware capabilities
        self.gpu_memory = self.device.recommendedMaxWorkingSetSize()  # bytes
        self.max_buffer_size = self.device.maxBufferLength()          # bytes
        self.max_threads_per_tg = self.device.maxThreadsPerThreadgroup()
        # Safety: reserve 25% for OS/system, only use 75% for allocations
        self._memory_budget = int(self.gpu_memory * 0.75)
        self._allocated = 0

        # Cache compiled pipelines
        self._pipeline_cache: dict[str, any] = {}
        # Cache int buffers to avoid repeated allocation
        self._int_buffer_cache: dict[int, any] = {}

        # Command buffer batching state
        self._batch_mode = False
        self._batch_buffer = None
        self._batch_encoder = None

        # Native C bridge for fast dispatch
        self._fast_lib = _load_fast_dispatch()
        if self._fast_lib is not None:
            import objc
            self._objc = objc
            self._queue_ptr = objc.pyobjc_id(self.command_queue)

    def _check_allocation(self, size_bytes: int):
        """Guard against OOM — refuse allocation if it would exceed memory budget."""
        if size_bytes > self.max_buffer_size:
            raise MemoryError(
                f"Buffer size {size_bytes / 1e6:.1f}MB exceeds Metal max buffer "
                f"({self.max_buffer_size / 1e6:.1f}MB)"
            )
        if self._allocated + size_bytes > self._memory_budget:
            raise MemoryError(
                f"Allocation of {size_bytes / 1e6:.1f}MB would exceed GPU memory budget "
                f"({self._memory_budget / 1e6:.1f}MB used: {self._allocated / 1e6:.1f}MB). "
                f"Total GPU memory: {self.gpu_memory / 1e6:.1f}MB. "
                f"Use smaller sizes or free tensors."
            )

    def hardware_info(self) -> dict:
        """Return GPU hardware capabilities."""
        return {
            "device": self.device_name,
            "gpu_memory_mb": round(self.gpu_memory / 1e6),
            "max_buffer_mb": round(self.max_buffer_size / 1e6),
            "memory_budget_mb": round(self._memory_budget / 1e6),
            "allocated_mb": round(self._allocated / 1e6, 1),
            "max_threads_per_threadgroup": (
                self.max_threads_per_tg.width,
                self.max_threads_per_tg.height,
                self.max_threads_per_tg.depth,
            ),
        }

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
        self._check_allocation(len(byte_data))
        buffer = self.device.newBufferWithBytes_length_options_(
            byte_data, len(byte_data),
            Metal.MTLResourceStorageModeShared
        )
        if buffer is None:
            raise RuntimeError(f"Failed to allocate Metal buffer of {len(byte_data)} bytes")
        self._allocated += len(byte_data)
        return buffer

    def allocate_empty_buffer(self, size_bytes: int) -> any:
        """Allocate an empty Metal buffer."""
        Metal = self._Metal
        self._check_allocation(size_bytes)
        buffer = self.device.newBufferWithLength_options_(
            size_bytes, Metal.MTLResourceStorageModeShared
        )
        if buffer is None:
            raise RuntimeError(f"Failed to allocate Metal buffer of {size_bytes} bytes")
        self._allocated += size_bytes
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

        # Slow path: PyObjC (synchronous or batch mode)
        Metal = self._Metal

        if self._batch_mode and self._batch_buffer is not None:
            # Batch mode: reuse shared command buffer, don't commit yet
            encoder = self._batch_buffer.computeCommandEncoder()
            encoder.setComputePipelineState_(pipeline)
            for i, buf in enumerate(buffers):
                encoder.setBuffer_offset_atIndex_(buf, 0, i)
            grid_size = Metal.MTLSizeMake(*grid)
            tg_size = Metal.MTLSizeMake(*threadgroup_size)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
            encoder.endEncoding()
            return

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

    def begin_batch(self):
        """Start batching mode — multiple dispatches share one command buffer."""
        if self._batch_mode:
            return  # already in batch mode
        self._batch_mode = True
        self._batch_buffer = self.command_queue.commandBuffer()

    def end_batch(self):
        """End batching mode — commit the shared command buffer and wait."""
        if not self._batch_mode:
            return
        self._batch_mode = False
        if self._batch_buffer is not None:
            self._batch_buffer.commit()
            self._batch_buffer.waitUntilCompleted()
            error = self._batch_buffer.error()
            if error is not None:
                raise RuntimeError(f"Metal batch execution failed: {error}")
            self._batch_buffer = None


# Global runtime instance (lazy init)
_runtime: Optional[MetalRuntime] = None


def get_runtime() -> MetalRuntime:
    """Get or create the global Metal runtime."""
    global _runtime
    if _runtime is None:
        _runtime = MetalRuntime()
    return _runtime


def set_device(index: int = 0):
    """Switch to a different GPU device. Resets the runtime."""
    global _runtime
    _runtime = MetalRuntime(device_index=index)


def list_devices() -> list[str]:
    """List all available Metal GPU devices."""
    _check_metal_available()
    import Metal
    devices = Metal.MTLCopyAllDevices()
    if not devices:
        return []
    return [d.name() for d in devices]
