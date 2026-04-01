"""
Locust Public API — The user-facing interface.

Provides:
- @locomp.kernel decorator — compiles Python functions to GPU kernels
- locomp.tensor() — create GPU tensors
- locomp.program_id(), locomp.arange(), etc. — kernel primitives
- KernelLauncher — handles grid dispatch
"""

from __future__ import annotations

import functools
import platform
from typing import Any, Callable

import numpy as np

from locomp.frontend import compile_kernel, constexpr, Tensor, Float16, UInt8, Int8
from locomp.ir import IRKernel
from locomp.optimizer import optimize
from locomp.backends.metal_codegen import compile_to_metal


# Re-export types for user annotations
constexpr = constexpr
Tensor = Tensor
Float16 = Float16
UInt8 = UInt8
Int8 = Int8


class LocompTensor:
    """A GPU-backed tensor. Wraps a Metal buffer + numpy array."""

    _SUPPORTED_DTYPES = {np.float16, np.float32, np.float64, np.int8, np.uint8}

    def __init__(self, data: np.ndarray, metal_buffer=None):
        if data.dtype.type not in self._SUPPORTED_DTYPES:
            data = data.astype(np.float32)
        self.data = data
        self._metal_buffer = metal_buffer
        self._size = self.data.size
        self._shape = self.data.shape
        self._dtype = self.data.dtype
        self._gpu_dirty = False  # True when GPU has newer data than CPU

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    @property
    def dtype(self):
        return self._dtype

    def to_metal_buffer(self, runtime):
        """Get or create the Metal buffer for this tensor."""
        if self._metal_buffer is None:
            self._metal_buffer = runtime.allocate_buffer(self.data)
        return self._metal_buffer

    def _mark_dirty(self):
        """Mark that GPU may have written newer data."""
        self._gpu_dirty = True

    def sync_from_gpu(self, runtime):
        """Read data back from GPU buffer (only if dirty)."""
        if self._metal_buffer is not None and self._gpu_dirty:
            self.data = runtime.read_buffer(
                self._metal_buffer, self._dtype, self._size
            ).reshape(self._shape)
            self._gpu_dirty = False

    def numpy(self) -> np.ndarray:
        """Get numpy data — syncs from GPU if needed."""
        if self._gpu_dirty and self._metal_buffer is not None:
            from locomp.backends.metal_runtime import get_runtime
            runtime = get_runtime()
            runtime.sync()  # wait for pending async dispatch
            self.data = runtime.read_buffer(
                self._metal_buffer, self._dtype, self._size
            ).reshape(self._shape)
            self._gpu_dirty = False
        return self.data

    def __repr__(self) -> str:
        return f"locomp.Tensor(shape={self._shape}, dtype={self._dtype}, data={self.data})"

    def __str__(self) -> str:
        return str(self.data)


def tensor(data, dtype=np.float32) -> LocompTensor:
    """Create a Locust tensor from a list or numpy array."""
    if isinstance(data, np.ndarray):
        return LocompTensor(data)
    return LocompTensor(np.array(data, dtype=dtype))


def empty(shape, dtype=np.float32) -> LocompTensor:
    """Create an empty Locust tensor."""
    if isinstance(shape, int):
        shape = (shape,)
    return LocompTensor(np.empty(shape, dtype=dtype))


def zeros(shape, dtype=np.float32) -> LocompTensor:
    """Create a zero-filled Locust tensor."""
    if isinstance(shape, int):
        shape = (shape,)
    return LocompTensor(np.zeros(shape, dtype=dtype))


def ones(shape, dtype=np.float32) -> LocompTensor:
    """Create a ones-filled Locust tensor."""
    if isinstance(shape, int):
        shape = (shape,)
    return LocompTensor(np.ones(shape, dtype=dtype))


class KernelLauncher:
    """Handles compiling and launching a kernel on the GPU."""

    def __init__(self, func: Callable, backend: str = "auto"):
        self.func = func
        self.func_name = func.__name__
        self.backend = self._resolve_backend(backend)
        self._ir: IRKernel | None = None
        self._msl_source: str | None = None
        self._pipeline = None
        self._buffer_map: dict | None = None
        self._compiled = False
        # Cache specialized pipelines per constexpr value tuple
        self._specialized: dict[tuple, tuple] = {}  # constexpr_key → (pipeline, buffer_map, msl_source)

    def _resolve_backend(self, backend: str) -> str:
        if backend == "auto":
            if platform.system() == "Darwin":
                return "metal"
            else:
                raise RuntimeError(
                    "No GPU backend available. "
                    "Apple Metal requires macOS."
                )
        return backend

    def _compile(self):
        """Compile the kernel (lazy, happens on first launch)."""
        if self._compiled:
            return

        # Step 1: Python AST → Locust IR
        self._ir = compile_kernel(self.func)

        # Step 2: Optimize IR
        self._ir = optimize(self._ir, target=self.backend)

        # Step 3: IR → MSL source code
        if self.backend == "metal":
            self._msl_source, self._buffer_map = compile_to_metal(self._ir)
        else:
            raise NotImplementedError(f"Backend '{self.backend}' not implemented yet")

        self._compiled = True

    def __getitem__(self, grid) -> _KernelCall:
        """kernel[grid_size](...) or kernel[(grid,), (tg_size,)](...) syntax."""
        if isinstance(grid, tuple) and len(grid) == 2 and isinstance(grid[0], tuple):
            # kernel[(num_groups,), (threads_per_group,)](...)
            return _KernelCall(self, grid[0], threadgroup_size=grid[1])
        if not isinstance(grid, tuple):
            grid = (grid,)
        return _KernelCall(self, grid)

    @property
    def ir(self) -> IRKernel:
        """Get the IR (compiles if needed)."""
        self._compile()
        return self._ir

    @property
    def msl(self) -> str:
        """Get the generated MSL source (compiles if needed)."""
        self._compile()
        return self._msl_source


class _KernelCall:
    """Represents a kernel call with a specific grid size."""

    def __init__(self, launcher: KernelLauncher, grid: tuple,
                 threadgroup_size: tuple = None):
        self.launcher = launcher
        self.grid = grid
        self.threadgroup_size = threadgroup_size

    def __call__(self, *args, **kwargs):
        """Execute the kernel on the GPU."""
        self.launcher._compile()

        if self.launcher.backend == "metal":
            return self._launch_metal(*args, **kwargs)
        else:
            raise NotImplementedError(f"Backend '{self.launcher.backend}' not implemented")

    def _launch_metal(self, *args, **kwargs):
        """Launch kernel on Apple Metal GPU."""
        from locomp.backends.metal_runtime import get_runtime
        from locomp.backends.metal_codegen import compile_to_metal

        runtime = get_runtime()

        # Separate tensor and constexpr args
        all_args = list(args)
        for key, value in kwargs.items():
            all_args.append(value)

        ir_params = self.launcher._ir.params
        constexpr_values = {}
        tensor_args = []

        for i, arg in enumerate(all_args):
            param = ir_params[i]
            if param.is_pointer:
                tensor_args.append(arg)
            else:
                # Preserve int vs float type for correct MSL codegen
                if isinstance(arg, float):
                    constexpr_values[param.name] = float(arg)
                else:
                    constexpr_values[param.name] = int(arg)

        # Get or create specialized pipeline for these constexpr values
        constexpr_key = tuple(sorted(constexpr_values.items()))
        if constexpr_key not in self.launcher._specialized:
            msl_source, buffer_map = compile_to_metal(
                self.launcher._ir, constexpr_values=constexpr_values
            )
            pipeline = runtime.compile_msl(msl_source, self.launcher.func_name)
            self.launcher._specialized[constexpr_key] = (pipeline, buffer_map, msl_source)
            # Update .msl for inspection
            self.launcher._msl_source = msl_source

        pipeline, buffer_map, msl_source = self.launcher._specialized[constexpr_key]

        # Prepare only tensor buffers (constexpr values are inlined in MSL)
        buffers = []
        output_tensors = []

        for arg in tensor_args:
            if isinstance(arg, LocompTensor):
                buf = arg.to_metal_buffer(runtime)
                buffers.append(buf)
                output_tensors.append(arg)
            elif isinstance(arg, np.ndarray):
                t = LocompTensor(arg)
                buf = t.to_metal_buffer(runtime)
                buffers.append(buf)
                output_tensors.append(t)
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        # Determine dispatch dimensions
        if self.threadgroup_size is not None:
            grid = self.grid
            tg_size = self.threadgroup_size
        else:
            grid = self.grid
            tg_size = (1,)

        # Pad to 3D
        while len(grid) < 3:
            grid = grid + (1,)
        while len(tg_size) < 3:
            tg_size = tg_size + (1,)

        # Dispatch
        runtime.dispatch(pipeline, buffers, grid=grid, threadgroup_size=tg_size)

        # Mark tensor args as GPU-dirty
        for t in output_tensors:
            t._mark_dirty()


def kernel(func: Callable = None, *, backend: str = "auto") -> KernelLauncher:
    """Decorator to compile a Python function into a GPU kernel.

    Usage:
        @locomp.kernel
        def my_kernel(X: locomp.Tensor, N: locomp.constexpr):
            ...

        # Launch with grid size
        my_kernel[(grid_size,)](x_tensor, N=1024)
    """
    if func is not None:
        return KernelLauncher(func, backend=backend)

    def wrapper(f):
        return KernelLauncher(f, backend=backend)
    return wrapper


def hardware_info() -> dict:
    """Return GPU hardware capabilities (memory, max buffer, thread limits)."""
    from locomp.backends.metal_runtime import get_runtime
    return get_runtime().hardware_info()


# --- Kernel primitive functions (used inside @kernel functions) ---
# These are never actually called at runtime — the AST compiler intercepts them.
# They exist for IDE autocomplete and type checking.

def program_id(axis: int = 0) -> int:
    """Get the program/block ID for the given axis. Only valid inside @kernel."""
    raise RuntimeError("program_id() can only be used inside a @locomp.kernel function")

def thread_id(axis: int = 0) -> int:
    """Get the global thread ID for the given axis. Only valid inside @kernel."""
    raise RuntimeError("thread_id() can only be used inside a @locomp.kernel function")

def local_id(axis: int = 0) -> int:
    """Get the local thread ID within the threadgroup. Only valid inside @kernel."""
    raise RuntimeError("local_id() can only be used inside a @locomp.kernel function")

def group_size(axis: int = 0) -> int:
    """Get the threadgroup size for the given axis. Only valid inside @kernel."""
    raise RuntimeError("group_size() can only be used inside a @locomp.kernel function")

def num_groups(axis: int = 0) -> int:
    """Get the number of threadgroups for the given axis. Only valid inside @kernel."""
    raise RuntimeError("num_groups() can only be used inside a @locomp.kernel function")

def barrier() -> None:
    """Synchronize all threads in the threadgroup. Only valid inside @kernel."""
    raise RuntimeError("barrier() can only be used inside a @locomp.kernel function")

def shared_memory(size: int, dtype=None) -> Any:
    """Allocate threadgroup shared memory. Only valid inside @kernel."""
    raise RuntimeError("shared_memory() can only be used inside a @locomp.kernel function")

def arange(start_or_end: int, end: int = None) -> Any:
    """Generate a range of indices. Only valid inside @kernel."""
    raise RuntimeError("arange() can only be used inside a @locomp.kernel function")

def load(ptr, mask=None) -> Any:
    """Load data from a pointer. Only valid inside @kernel."""
    raise RuntimeError("load() can only be used inside a @locomp.kernel function")

def store(ptr, value, mask=None) -> None:
    """Store data to a pointer. Only valid inside @kernel."""
    raise RuntimeError("store() can only be used inside a @locomp.kernel function")

def shared_load(arr, idx) -> Any:
    """Load from threadgroup shared memory. Only valid inside @kernel."""
    raise RuntimeError("shared_load() can only be used inside a @locomp.kernel function")

def shared_store(arr, idx, value) -> None:
    """Store to threadgroup shared memory. Only valid inside @kernel."""
    raise RuntimeError("shared_store() can only be used inside a @locomp.kernel function")

def exp(x) -> Any:
    """Compute exp(x). Only valid inside @kernel."""
    raise RuntimeError("exp() can only be used inside a @locomp.kernel function")

def log(x) -> Any:
    """Compute log(x). Only valid inside @kernel."""
    raise RuntimeError("log() can only be used inside a @locomp.kernel function")

def sqrt(x) -> Any:
    """Compute sqrt(x). Only valid inside @kernel."""
    raise RuntimeError("sqrt() can only be used inside a @locomp.kernel function")

def abs(x) -> Any:
    """Compute abs(x). Only valid inside @kernel."""
    raise RuntimeError("abs() can only be used inside a @locomp.kernel function")

def tanh(x) -> Any:
    """Compute tanh(x). Only valid inside @kernel."""
    raise RuntimeError("tanh() can only be used inside a @locomp.kernel function")

def sin(x) -> Any:
    """Compute sin(x). Only valid inside @kernel."""
    raise RuntimeError("sin() can only be used inside a @locomp.kernel function")

def cos(x) -> Any:
    """Compute cos(x). Only valid inside @kernel."""
    raise RuntimeError("cos() can only be used inside a @locomp.kernel function")

def asin(x) -> Any:
    """Compute asin(x). Only valid inside @kernel."""
    raise RuntimeError("asin() can only be used inside a @locomp.kernel function")

def acos(x) -> Any:
    """Compute acos(x). Only valid inside @kernel."""
    raise RuntimeError("acos() can only be used inside a @locomp.kernel function")

def atan(x) -> Any:
    """Compute atan(x). Only valid inside @kernel."""
    raise RuntimeError("atan() can only be used inside a @locomp.kernel function")

def atan2(y, x) -> Any:
    """Compute atan2(y, x). Only valid inside @kernel."""
    raise RuntimeError("atan2() can only be used inside a @locomp.kernel function")

def sinh(x) -> Any:
    """Compute sinh(x). Only valid inside @kernel."""
    raise RuntimeError("sinh() can only be used inside a @locomp.kernel function")

def cosh(x) -> Any:
    """Compute cosh(x). Only valid inside @kernel."""
    raise RuntimeError("cosh() can only be used inside a @locomp.kernel function")

def exp2(x) -> Any:
    """Compute exp2(x) = 2^x. Only valid inside @kernel."""
    raise RuntimeError("exp2() can only be used inside a @locomp.kernel function")

def log2(x) -> Any:
    """Compute log2(x). Only valid inside @kernel."""
    raise RuntimeError("log2() can only be used inside a @locomp.kernel function")

def log10(x) -> Any:
    """Compute log10(x). Only valid inside @kernel."""
    raise RuntimeError("log10() can only be used inside a @locomp.kernel function")

def rsqrt(x) -> Any:
    """Compute 1/sqrt(x). Only valid inside @kernel."""
    raise RuntimeError("rsqrt() can only be used inside a @locomp.kernel function")

def ceil(x) -> Any:
    """Compute ceil(x). Only valid inside @kernel."""
    raise RuntimeError("ceil() can only be used inside a @locomp.kernel function")

def floor(x) -> Any:
    """Compute floor(x). Only valid inside @kernel."""
    raise RuntimeError("floor() can only be used inside a @locomp.kernel function")

def round(x) -> Any:
    """Compute round(x). Only valid inside @kernel."""
    raise RuntimeError("round() can only be used inside a @locomp.kernel function")

def sigmoid(x) -> Any:
    """Compute sigmoid(x) = 1/(1+exp(-x)). Only valid inside @kernel."""
    raise RuntimeError("sigmoid() can only be used inside a @locomp.kernel function")

def fma(a, b, c) -> Any:
    """Compute fma(a,b,c) = a*b+c. Only valid inside @kernel."""
    raise RuntimeError("fma() can only be used inside a @locomp.kernel function")

def pow(x, y) -> Any:
    """Compute pow(x,y). Only valid inside @kernel."""
    raise RuntimeError("pow() can only be used inside a @locomp.kernel function")

def clamp(x, lo, hi) -> Any:
    """Compute clamp(x, lo, hi). Only valid inside @kernel."""
    raise RuntimeError("clamp() can only be used inside a @locomp.kernel function")

def copysign(x, y) -> Any:
    """Copy sign of y to magnitude of x. Only valid inside @kernel."""
    raise RuntimeError("copysign() can only be used inside a @locomp.kernel function")

def fmod(x, y) -> Any:
    """Compute floating-point modulus. Only valid inside @kernel."""
    raise RuntimeError("fmod() can only be used inside a @locomp.kernel function")

def step(edge, x) -> Any:
    """Step function: 0.0 if x < edge, else 1.0. Only valid inside @kernel."""
    raise RuntimeError("step() can only be used inside a @locomp.kernel function")

def where(cond, a, b) -> Any:
    """Select a if cond else b. Only valid inside @kernel."""
    raise RuntimeError("where() can only be used inside a @locomp.kernel function")

def simd_sum(x) -> Any:
    """Sum across SIMD group (32 threads on Apple Silicon). Only valid inside @kernel."""
    raise RuntimeError("simd_sum() can only be used inside a @locomp.kernel function")

def simd_max(x) -> Any:
    """Max across SIMD group. Only valid inside @kernel."""
    raise RuntimeError("simd_max() can only be used inside a @locomp.kernel function")

def simd_min(x) -> Any:
    """Min across SIMD group. Only valid inside @kernel."""
    raise RuntimeError("simd_min() can only be used inside a @locomp.kernel function")

def simd_broadcast(x, lane: int) -> Any:
    """Broadcast value from specified lane across SIMD group. Only valid inside @kernel."""
    raise RuntimeError("simd_broadcast() can only be used inside a @locomp.kernel function")

def simd_shuffle_down(x, delta: int) -> Any:
    """Shift value down by delta lanes in SIMD group. Only valid inside @kernel."""
    raise RuntimeError("simd_shuffle_down() can only be used inside a @locomp.kernel function")

def simd_lane_id() -> int:
    """Get lane index within SIMD group (0-31). Only valid inside @kernel."""
    raise RuntimeError("simd_lane_id() can only be used inside a @locomp.kernel function")

def simd_group_id() -> int:
    """Get SIMD group index within threadgroup. Only valid inside @kernel."""
    raise RuntimeError("simd_group_id() can only be used inside a @locomp.kernel function")


# --- Simdgroup matrix operations (hardware 8×8 matmul) ---

def simdgroup_matrix_load(arr, offset, stride):
    """Load 8×8 matrix from shared memory. Only valid inside @kernel."""
    raise RuntimeError("simdgroup_matrix_load() can only be used inside a @locomp.kernel function")

def simdgroup_matrix_load_device(ptr, stride):
    """Load 8×8 matrix from device memory. Only valid inside @kernel."""
    raise RuntimeError("simdgroup_matrix_load_device() can only be used inside a @locomp.kernel function")

def simdgroup_matrix_store(mat, arr, offset, stride):
    """Store 8×8 matrix to shared memory. Only valid inside @kernel."""
    raise RuntimeError("simdgroup_matrix_store() can only be used inside a @locomp.kernel function")

def simdgroup_matrix_store_device(mat, ptr, stride):
    """Store 8×8 matrix to device memory. Only valid inside @kernel."""
    raise RuntimeError("simdgroup_matrix_store_device() can only be used inside a @locomp.kernel function")

def simdgroup_mac(acc, a, b):
    """Multiply-accumulate: D = A * B + C. Only valid inside @kernel."""
    raise RuntimeError("simdgroup_mac() can only be used inside a @locomp.kernel function")

def simdgroup_matrix(fill_value):
    """Create 8×8 matrix filled with a constant. Only valid inside @kernel."""
    raise RuntimeError("simdgroup_matrix() can only be used inside a @locomp.kernel function")
