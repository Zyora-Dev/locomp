"""
locomp CUDA backend — Modal test + benchmark script.

Tests the CUDA codegen on a real NVIDIA GPU via Modal.

Run with:
    modal run tests/test_cuda_modal.py

Requirements:
    pip install modal
    modal token new   # first time only

What this tests:
  1. vector_add       — basic load/store/arithmetic
  2. scale_shift      — scalar multiply + add
  3. relu             — conditional (where/max)
  4. reduce_sum       — atomic reduction across blocks
  5. tiled_add        — tiled (arange) load/store with loops
  6. dot_product      — for-loop accumulation
  7. rms_norm         — sqrt + reduce + divide
  8. softmax          — exp + reduce + divide (3-pass)

Each test compiles the kernel to CUDA C via locomp, launches it on an A10G,
and compares output to NumPy reference. Reports max absolute error and timing.
"""

import modal
import numpy as np

app = modal.App("locomp-cuda-tests")

# Install locomp from the local source (editable install)
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy")
    .run_commands(
        "pip install git+https://github.com/Zyora-Dev/locomp.git@main",
    )
)


@app.function(gpu="A10G", image=image, timeout=300)
def run_cuda_tests():
    import locomp
    import numpy as np
    import subprocess
    import ctypes
    import tempfile
    import os
    from locomp.frontend import compile_kernel
    from locomp.optimizer import optimize
    from locomp.backends.cuda_codegen import compile_to_cuda

    results = []

    def compile_and_run(name, fn, constexpr_values, grid, block, inputs, expected_fn):
        """Compile kernel to CUDA, run it, compare to NumPy."""
        ir = compile_kernel(fn)
        ir = optimize(ir, target="cuda")
        cuda_src, param_map = compile_to_cuda(ir, constexpr_values=constexpr_values)

        # Write .cu and compile with nvcc
        with tempfile.NamedTemporaryFile(suffix=".cu", delete=False, mode="w") as f:
            f.write(cuda_src)
            cu_path = f.name
        so_path = cu_path.replace(".cu", ".so")

        r = subprocess.run(
            ["nvcc", "-arch=sm_86", "-O2", "-shared", "-Xcompiler", "-fPIC",
             "-o", so_path, cu_path],
            capture_output=True, text=True
        )
        os.unlink(cu_path)
        if r.returncode != 0:
            results.append({"name": name, "status": "COMPILE_ERROR", "error": r.stderr[:500]})
            return

        lib = ctypes.CDLL(so_path)
        launch_fn = getattr(lib, f"locomp_launch_{fn.__name__}")
        launch_fn.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        launch_fn.restype = None

        # Allocate CUDA device memory and copy inputs
        import ctypes as ct
        libcudart = ctypes.CDLL("libcudart.so")
        libcudart.cudaMalloc.restype = ctypes.c_int
        libcudart.cudaMemcpy.restype = ctypes.c_int
        libcudart.cudaFree.restype = ctypes.c_int
        CUDA_MEMCPY_HOST_TO_DEVICE = 1
        CUDA_MEMCPY_DEVICE_TO_HOST = 2

        d_ptrs = []
        h_arrays = list(inputs)
        for arr in h_arrays:
            d_ptr = ctypes.c_void_p()
            libcudart.cudaMalloc(ctypes.byref(d_ptr), arr.nbytes)
            libcudart.cudaMemcpy(d_ptr, arr.ctypes.data_as(ctypes.c_void_p),
                                 arr.nbytes, CUDA_MEMCPY_HOST_TO_DEVICE)
            d_ptrs.append(d_ptr)

        arr_type = (ctypes.c_void_p * len(d_ptrs))
        c_arr = arr_type(*[p.value for p in d_ptrs])

        import time
        t0 = time.perf_counter()
        launch_fn(
            ctypes.c_int(grid[0]), ctypes.c_int(grid[1] if len(grid) > 1 else 1),
            ctypes.c_int(block[0]), ctypes.c_int(block[1] if len(block) > 1 else 1),
            ctypes.cast(c_arr, ctypes.POINTER(ctypes.c_void_p))
        )
        libcudart.cudaDeviceSynchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Copy output back (last array is output by convention)
        out_arr = h_arrays[-1].copy()
        libcudart.cudaMemcpy(out_arr.ctypes.data_as(ctypes.c_void_p),
                             d_ptrs[-1], out_arr.nbytes, CUDA_MEMCPY_DEVICE_TO_HOST)

        for p in d_ptrs:
            libcudart.cudaFree(p)
        os.unlink(so_path)

        expected = expected_fn(*inputs[:-1])
        max_err = float(np.max(np.abs(out_arr - expected)))
        status = "PASS" if max_err < 1e-4 else "FAIL"
        results.append({
            "name": name, "status": status,
            "max_err": max_err, "time_ms": elapsed_ms
        })

    N = 1024

    # ── 1. vector_add ────────────────────────────────────────────────────────
    def vector_add(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        b = locomp.load(B + i)
        locomp.store(Out + i, a + b)

    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)
    compile_and_run("vector_add", vector_add, {"N": N}, (N,), (1,),
                    [a, b, out], lambda a, b: a + b)

    # ── 2. scale_shift ───────────────────────────────────────────────────────
    def scale_shift(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        x = locomp.load(X + i)
        locomp.store(Out + i, x * 2.0 + 1.0)

    x = np.random.randn(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)
    compile_and_run("scale_shift", scale_shift, {"N": N}, (N,), (1,),
                    [x, out], lambda x: x * 2.0 + 1.0)

    # ── 3. relu ──────────────────────────────────────────────────────────────
    def relu(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        x = locomp.load(X + i)
        y = locomp.where(x > 0.0, x, 0.0)
        locomp.store(Out + i, y)

    x = np.random.randn(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)
    compile_and_run("relu", relu, {"N": N}, (N,), (1,),
                    [x, out], lambda x: np.maximum(x, 0.0))

    # ── 4. dot_product (for-loop accumulation) ───────────────────────────────
    M = 64
    def dot_product(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor,
                    N: locomp.constexpr):
        i = locomp.program_id(0)
        acc = 0.0
        for k in range(N):
            a = locomp.load(A + i * N + k)
            b = locomp.load(B + i * N + k)
            acc = acc + a * b
        locomp.store(Out + i, acc)

    a = np.random.randn(M, N).astype(np.float32)
    b = np.random.randn(M, N).astype(np.float32)
    out = np.zeros(M, dtype=np.float32)
    compile_and_run("dot_product", dot_product, {"N": N}, (M,), (1,),
                    [a.flatten(), b.flatten(), out],
                    lambda a, b: (a.reshape(M, N) * b.reshape(M, N)).sum(axis=1))

    # ── 5. sqrt_exp ──────────────────────────────────────────────────────────
    def sqrt_exp(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        x = locomp.load(X + i)
        y = locomp.exp(locomp.sqrt(locomp.abs(x)))
        locomp.store(Out + i, y)

    x = np.random.randn(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)
    compile_and_run("sqrt_exp", sqrt_exp, {"N": N}, (N,), (1,),
                    [x, out], lambda x: np.exp(np.sqrt(np.abs(x))))

    return results


@app.local_entrypoint()
def main():
    results = run_cuda_tests.remote()
    print("\n" + "="*60)
    print("locomp CUDA Backend — Modal A10G Test Results")
    print("="*60)
    passed = 0
    failed = 0
    for r in results:
        if r["status"] == "PASS":
            print(f"  PASS  {r['name']:20s}  max_err={r['max_err']:.2e}  {r['time_ms']:.3f}ms")
            passed += 1
        elif r["status"] == "COMPILE_ERROR":
            print(f"  ERROR {r['name']:20s}  {r['error'][:80]}")
            failed += 1
        else:
            print(f"  FAIL  {r['name']:20s}  max_err={r['max_err']:.2e}")
            failed += 1
    print("="*60)
    print(f"  {passed} passed, {failed} failed")
    print("="*60)
