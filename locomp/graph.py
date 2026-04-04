"""
locomp Kernel Graph — Chain multiple kernels in a single GPU command buffer.

Without graph:
    kernel_a[(N,)](x, tmp, N=N)   # GPU→CPU sync after each kernel
    kernel_b[(N,)](tmp, out, N=N) # another sync

With graph:
    g = locomp.graph()
    g.add(kernel_a, (N,), x, tmp, N=N)
    g.add(kernel_b, (N,), tmp, out, N=N)
    g.run()   # single command buffer, one sync at the end

Also usable as a context manager:
    with locomp.graph() as g:
        g.add(rms_norm, (N,), x, w, h, N=N, eps=1e-5)
        g.add(matmul,   (M, K), h, w2, out, M=M, N=K, K=N)
    # runs automatically on context exit

Graphs can be re-run with g.run() repeatedly — useful for inference loops
where the same kernel sequence runs on different data already on GPU.
"""

from __future__ import annotations

from typing import Any


class KernelGraph:
    """A sequence of kernel calls dispatched as a single GPU command buffer.

    This eliminates per-kernel CPU↔GPU synchronisation — the GPU executes
    the entire sequence before the CPU resumes.
    """

    def __init__(self):
        self._steps: list[tuple] = []  # (launcher, grid, tg_size, args, kwargs)

    # ── Recording ─────────────────────────────────────────────────────────────

    def add(self, kernel_launcher, grid, *args, **kwargs) -> "KernelGraph":
        """Record a kernel call.

        Args:
            kernel_launcher: A KernelLauncher (result of @locomp.kernel)
            grid: Grid tuple, e.g. (N,) or (M, N)
            *args: Positional kernel arguments (tensors / constexpr values)
            **kwargs: Keyword constexpr arguments

        Returns self for chaining:
            g.add(...).add(...).run()
        """
        # Normalise grid the same way KernelLauncher.__getitem__ does
        if isinstance(grid, tuple) and len(grid) == 2 and isinstance(grid[0], tuple):
            # ((num_groups,), (threads,)) form
            tg_size = grid[1]
            grid = grid[0]
        else:
            tg_size = None
        if not isinstance(grid, tuple):
            grid = (grid,)
        self._steps.append((kernel_launcher, grid, tg_size, args, kwargs))
        return self

    def __len__(self) -> int:
        return len(self._steps)

    def clear(self) -> "KernelGraph":
        """Remove all recorded steps."""
        self._steps.clear()
        return self

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Dispatch all recorded kernels in a single GPU command buffer."""
        if not self._steps:
            return

        from locomp.backends.metal_runtime import get_runtime
        runtime = get_runtime()

        # Ensure all kernels are compiled before entering batch mode
        # (compilation may itself call compile_msl which must run outside batch)
        for launcher, grid, tg_size, args, kwargs in self._steps:
            launcher._compile()
            # Also warm the constexpr specialisation so compile_msl fires now
            all_args = list(args) + list(kwargs.values())
            ir_params = launcher._ir.params
            constexpr_values: dict = {}
            for i, arg in enumerate(all_args):
                if i >= len(ir_params):
                    break
                param = ir_params[i]
                if not param.is_pointer:
                    if isinstance(arg, float):
                        constexpr_values[param.name] = float(arg)
                    else:
                        constexpr_values[param.name] = int(arg)
            constexpr_key = tuple(sorted(constexpr_values.items()))
            if constexpr_key not in launcher._specialized:
                from locomp.backends.metal_codegen import compile_to_metal
                from locomp import cache as _cache
                cached = _cache.get(launcher.func, constexpr_values)
                if cached is not None:
                    msl_source, buffer_map = cached
                else:
                    msl_source, buffer_map = compile_to_metal(
                        launcher._ir, constexpr_values=constexpr_values
                    )
                    _cache.put(launcher.func, constexpr_values, msl_source, buffer_map)
                pipeline = runtime.compile_msl(msl_source, launcher.func_name)
                launcher._specialized[constexpr_key] = (pipeline, buffer_map, msl_source)
                launcher._msl_source = msl_source

        # Now batch all dispatches into one command buffer
        runtime.begin_batch()
        try:
            for launcher, grid, tg_size, args, kwargs in self._steps:
                self._dispatch_one(runtime, launcher, grid, tg_size, args, kwargs)
        finally:
            runtime.end_batch()

    def _dispatch_one(self, runtime, launcher, grid, tg_size, args, kwargs):
        """Dispatch a single recorded step (called inside batch mode)."""
        from locomp.api import LocompTensor
        import numpy as np

        all_args = list(args)
        for value in kwargs.values():
            all_args.append(value)

        ir_params = launcher._ir.params
        constexpr_values: dict = {}
        tensor_args = []

        for i, arg in enumerate(all_args):
            if i >= len(ir_params):
                break
            param = ir_params[i]
            if param.is_pointer:
                tensor_args.append(arg)
            else:
                if isinstance(arg, float):
                    constexpr_values[param.name] = float(arg)
                else:
                    constexpr_values[param.name] = int(arg)

        constexpr_key = tuple(sorted(constexpr_values.items()))
        pipeline, buffer_map, _ = launcher._specialized[constexpr_key]

        buffers = []
        for arg in tensor_args:
            if isinstance(arg, LocompTensor):
                buf = arg.to_metal_buffer(runtime)
                buffers.append(buf)
            elif isinstance(arg, np.ndarray):
                from locomp.api import LocompTensor as LT
                t = LT(arg)
                buffers.append(t.to_metal_buffer(runtime))
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        # Pad grid/tg to 3D
        g = grid
        tg = tg_size if tg_size is not None else (1,)
        while len(g) < 3:
            g = g + (1,)
        while len(tg) < 3:
            tg = tg + (1,)

        runtime.dispatch(pipeline, buffers, grid=g, threadgroup_size=tg)

        # Mark tensors dirty
        for arg in tensor_args:
            if isinstance(arg, LocompTensor):
                arg._mark_dirty()

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "KernelGraph":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.run()
        return False  # don't suppress exceptions

    def __repr__(self) -> str:
        return f"KernelGraph({len(self._steps)} steps)"


def graph() -> KernelGraph:
    """Create a new kernel graph.

    Usage (explicit run):
        g = locomp.graph()
        g.add(kernel_a, (N,), x, y, N=N)
        g.add(kernel_b, (N,), y, z, N=N)
        g.run()

    Usage (context manager):
        with locomp.graph() as g:
            g.add(kernel_a, (N,), x, y, N=N)
            g.add(kernel_b, (N,), y, z, N=N)
        # auto-runs on exit
    """
    return KernelGraph()
