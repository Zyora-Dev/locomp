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

from locomp.frontend import compile_kernel, constexpr, Tensor, Float16, UInt8, Int8, Int32, Bool
from locomp.ir import IRKernel
from locomp.optimizer import optimize
from locomp.backends.metal_codegen import compile_to_metal


# Re-export types for user annotations
constexpr = constexpr
Tensor = Tensor
Float16 = Float16
UInt8 = UInt8
Int8 = Int8
Int32 = Int32
Bool = Bool


class LocompTensor:
    """A GPU-backed tensor with shape/stride tracking.
    
    Supports zero-copy operations: reshape, view, transpose, permute.
    Data stays on GPU until .numpy() is called.
    """

    _SUPPORTED_DTYPES = {np.float16, np.float32, np.float64, np.int8, np.uint8, np.int32, np.bool_}

    def __init__(self, data: np.ndarray = None, metal_buffer=None, shape=None,
                 strides=None, dtype=None, offset=0, base=None):
        if data is not None:
            if data.dtype.type not in self._SUPPORTED_DTYPES:
                data = data.astype(np.float32)
            self.data = data
            self._metal_buffer = metal_buffer
            self._size = data.size
            self._shape = data.shape
            self._strides = tuple(s // data.dtype.itemsize for s in data.strides)  # element strides
            self._dtype = data.dtype
        else:
            # View constructor — no data copy, shares metal buffer
            self.data = None
            self._metal_buffer = metal_buffer
            self._shape = tuple(shape)
            self._strides = tuple(strides) if strides else None
            self._dtype = np.dtype(dtype)
            self._size = 1
            for s in self._shape:
                self._size *= s
        self._offset = offset  # element offset into the buffer
        self._base = base  # reference to base tensor (keeps it alive)
        self._gpu_dirty = False
        self._freed = False

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def itemsize(self):
        return self._dtype.itemsize

    def is_contiguous(self):
        """Check if tensor is C-contiguous (row-major)."""
        if self._strides is None:
            return True
        expected = 1
        for i in range(len(self._shape) - 1, -1, -1):
            if self._shape[i] == 1:
                continue
            if self._strides[i] != expected:
                return False
            expected *= self._shape[i]
        return True

    def contiguous(self):
        """Return a contiguous copy if not already contiguous."""
        if self.is_contiguous() and self._offset == 0:
            return self
        # Need to copy — materialize via numpy and re-upload
        np_data = self.numpy().copy()
        return LocompTensor(np_data)

    def reshape(self, *shape):
        """Reshape tensor (zero-copy if contiguous)."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        # Handle -1
        neg_idx = None
        prod = 1
        for i, s in enumerate(shape):
            if s == -1:
                if neg_idx is not None:
                    raise ValueError("Only one -1 allowed in reshape")
                neg_idx = i
            else:
                prod *= s
        if neg_idx is not None:
            shape = list(shape)
            shape[neg_idx] = self._size // prod
            shape = tuple(shape)
        # Verify size match
        new_size = 1
        for s in shape:
            new_size *= s
        if new_size != self._size:
            raise ValueError(f"Cannot reshape {self._shape} to {shape}: size mismatch ({self._size} vs {new_size})")
        if not self.is_contiguous():
            return self.contiguous().reshape(*shape)
        # Zero-copy: compute new strides
        new_strides = []
        stride = 1
        for s in reversed(shape):
            new_strides.append(stride)
            stride *= s
        new_strides.reverse()
        return LocompTensor(
            shape=shape, strides=new_strides, dtype=self._dtype,
            metal_buffer=self._metal_buffer, offset=self._offset,
            base=self._base or self,
        )

    def view(self, *shape):
        """Alias for reshape."""
        return self.reshape(*shape)

    def flatten(self):
        """Flatten to 1D."""
        return self.reshape(self._size)

    def transpose(self, dim0=-2, dim1=-1):
        """Swap two dimensions (zero-copy, just swaps strides)."""
        ndim = len(self._shape)
        if dim0 < 0:
            dim0 += ndim
        if dim1 < 0:
            dim1 += ndim
        new_shape = list(self._shape)
        new_strides = list(self._strides) if self._strides else list(self._compute_strides())
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]
        return LocompTensor(
            shape=tuple(new_shape), strides=tuple(new_strides), dtype=self._dtype,
            metal_buffer=self._metal_buffer, offset=self._offset,
            base=self._base or self,
        )

    def permute(self, *dims):
        """Permute dimensions (zero-copy)."""
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        old_strides = self._strides if self._strides else self._compute_strides()
        new_shape = tuple(self._shape[d] for d in dims)
        new_strides = tuple(old_strides[d] for d in dims)
        return LocompTensor(
            shape=new_shape, strides=new_strides, dtype=self._dtype,
            metal_buffer=self._metal_buffer, offset=self._offset,
            base=self._base or self,
        )

    def unsqueeze(self, dim):
        """Add a dimension of size 1."""
        ndim = len(self._shape)
        if dim < 0:
            dim += ndim + 1
        new_shape = list(self._shape)
        new_shape.insert(dim, 1)
        strides = list(self._strides) if self._strides else list(self._compute_strides())
        # Stride for dim of size 1 doesn't matter, use neighbor's
        if dim < len(strides):
            strides.insert(dim, strides[dim])
        else:
            strides.append(1)
        return LocompTensor(
            shape=tuple(new_shape), strides=tuple(strides), dtype=self._dtype,
            metal_buffer=self._metal_buffer, offset=self._offset,
            base=self._base or self,
        )

    def squeeze(self, dim=None):
        """Remove dimensions of size 1."""
        if dim is not None:
            ndim = len(self._shape)
            if dim < 0:
                dim += ndim
            if self._shape[dim] != 1:
                return self
            new_shape = list(self._shape)
            new_strides = list(self._strides) if self._strides else list(self._compute_strides())
            new_shape.pop(dim)
            new_strides.pop(dim)
        else:
            strides = self._strides if self._strides else self._compute_strides()
            new_shape = []
            new_strides = []
            for s, st in zip(self._shape, strides):
                if s != 1:
                    new_shape.append(s)
                    new_strides.append(st)
        return LocompTensor(
            shape=tuple(new_shape), strides=tuple(new_strides), dtype=self._dtype,
            metal_buffer=self._metal_buffer, offset=self._offset,
            base=self._base or self,
        )

    def _compute_strides(self):
        """Compute C-contiguous strides from shape."""
        strides = []
        stride = 1
        for s in reversed(self._shape):
            strides.append(stride)
            stride *= s
        strides.reverse()
        return tuple(strides)

    def to_metal_buffer(self, runtime):
        """Get or create the Metal buffer for this tensor."""
        if self._metal_buffer is None:
            if self.data is not None:
                self._metal_buffer = runtime.allocate_buffer(self.data)
            else:
                size_bytes = self._size * self._dtype.itemsize
                self._metal_buffer = runtime.allocate_empty_buffer(size_bytes)
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
            total_elems = self._offset + self._size
            # Read the full buffer slice we need
            raw = runtime.read_buffer(self._metal_buffer, self._dtype, total_elems)
            if self._offset > 0:
                raw = raw[self._offset:]
            if self.is_contiguous():
                self.data = raw[:self._size].reshape(self._shape)
            else:
                # Non-contiguous: use stride-based indexing
                self.data = np.lib.stride_tricks.as_strided(
                    raw, shape=self._shape,
                    strides=tuple(s * self._dtype.itemsize for s in self._strides),
                ).copy()
            self._gpu_dirty = False
            return self.data
        if self.data is not None:
            return self.data
        # View with no data yet — read from GPU
        if self._metal_buffer is not None:
            from locomp.backends.metal_runtime import get_runtime
            runtime = get_runtime()
            runtime.sync()
            total_elems = self._offset + self._size
            raw = runtime.read_buffer(self._metal_buffer, self._dtype, total_elems)
            if self._offset > 0:
                raw = raw[self._offset:]
            return raw[:self._size].reshape(self._shape)
        return np.empty(self._shape, dtype=self._dtype)

    def __repr__(self) -> str:
        return f"locomp.Tensor(shape={self._shape}, dtype={self._dtype})"

    def __str__(self) -> str:
        return str(self.numpy())

    def free(self):
        """Release GPU buffer and reclaim memory budget."""
        if self._freed:
            return
        if self._base is not None:
            return  # Views don't own the buffer
        if self._metal_buffer is not None:
            from locomp.backends.metal_runtime import get_runtime
            runtime = get_runtime()
            buf_size = self._metal_buffer.length()
            runtime._allocated = max(0, runtime._allocated - buf_size)
            self._metal_buffer = None
        self._freed = True


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


def atomic_add(ptr, value) -> Any:
    """Atomic add to memory. Only valid inside @kernel."""
    raise RuntimeError("atomic_add() can only be used inside a @locomp.kernel function")

def atomic_max(ptr, value) -> Any:
    """Atomic max on memory. Only valid inside @kernel."""
    raise RuntimeError("atomic_max() can only be used inside a @locomp.kernel function")

def atomic_min(ptr, value) -> Any:
    """Atomic min on memory. Only valid inside @kernel."""
    raise RuntimeError("atomic_min() can only be used inside a @locomp.kernel function")


def set_device(index: int = 0):
    """Select which GPU device to use (for multi-GPU systems)."""
    from locomp.backends.metal_runtime import set_device as _set_device
    _set_device(index)
