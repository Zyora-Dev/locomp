"""
locomp CUDA backend — Modal benchmark + correctness script.

Run with:
    modal run tests/modal_cuda_runner.py

Benchmarks (A10G):
  - 16M-element kernels: vector_add, scale_shift, relu, sqrt_exp
  - dot_product (64 x 1024 accumulation)
  - shared_memory  — __shared__ correctness test
  - float16        — half-precision load/store/arithmetic
  Reports max absolute error, warm kernel time (ms), and memory throughput (GB/s).
"""

import modal
import numpy as np

app = modal.App("locomp-cuda-benchmarks")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("numpy")
    .run_commands(
        "pip install 'git+https://github.com/Zyora-Dev/locomp.git@78f9208'",
    )
)


@app.function(gpu="A10G", image=image, timeout=300)
def run_cuda_benchmarks():
    import locomp
    import numpy as np
    import subprocess
    import ctypes
    import tempfile
    import os
    import time
    from locomp.frontend import compile_kernel
    from locomp.optimizer import optimize
    from locomp.backends.cuda_codegen import compile_to_cuda

    libcudart = ctypes.CDLL("libcudart.so")
    libcudart.cudaMalloc.restype = ctypes.c_int
    libcudart.cudaMemcpy.restype = ctypes.c_int
    libcudart.cudaFree.restype = ctypes.c_int
    libcudart.cudaDeviceSynchronize.restype = ctypes.c_int
    H2D, D2H = 1, 2

    _so_cache = {}

    # Detect actual GPU SM arch once
    sm_arch = "sm_80"
    try:
        r_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if r_smi.returncode == 0:
            cap = r_smi.stdout.strip().split("\n")[0].strip().replace(".", "")
            if cap.isdigit():
                sm_arch = f"sm_{cap}"
    except Exception:
        pass
    print(f"[info] GPU arch: {sm_arch}", flush=True)

    def _build_so(fn, constexpr_values):
        import hashlib, shutil
        ir = compile_kernel(fn)
        ir = optimize(ir, target="cuda")
        cuda_src, param_map = compile_to_cuda(ir, constexpr_values=constexpr_values)
        key = hashlib.sha256(cuda_src.encode()).hexdigest()
        if key in _so_cache:
            return _so_cache[key], param_map, cuda_src
        with tempfile.NamedTemporaryFile(suffix=".cu", delete=False, mode="w") as f:
            f.write(cuda_src); cu = f.name
        so = cu.replace(".cu", ".so")
        r = subprocess.run(
            ["nvcc", f"-arch={sm_arch}", "-O2", "-w", "-shared", "-Xcompiler", "-fPIC", "-o", so, cu],
            capture_output=True, text=True)
        os.unlink(cu)
        if r.returncode != 0:
            print(f"[nvcc STDERR]\n{r.stderr}", flush=True)
            raise RuntimeError(f"nvcc failed:\n{r.stderr[:1200]}")
        _so_cache[key] = so
        return so, param_map, cuda_src

    def alloc_device(arr):
        d = ctypes.c_void_p()
        libcudart.cudaMalloc(ctypes.byref(d), arr.nbytes)
        libcudart.cudaMemcpy(d, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, H2D)
        return d

    def read_device(d, arr):
        out = arr.copy()
        libcudart.cudaMemcpy(out.ctypes.data_as(ctypes.c_void_p), d, arr.nbytes, D2H)
        return out

    def run_kernel(so, fn_name, grid, block, d_ptrs, n_iters=1):
        lib = ctypes.CDLL(so)
        fn = getattr(lib, fn_name)
        fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.POINTER(ctypes.c_void_p)]
        fn.restype = None
        arr_t = (ctypes.c_void_p * len(d_ptrs))
        c_arr = arr_t(*[p.value for p in d_ptrs])
        libcudart.cudaDeviceSynchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fn(grid[0], grid[1] if len(grid) > 1 else 1,
               block[0], block[1] if len(block) > 1 else 1,
               ctypes.cast(c_arr, ctypes.POINTER(ctypes.c_void_p)))
        libcudart.cudaDeviceSynchronize()
        return (time.perf_counter() - t0) * 1000 / n_iters  # ms per call

    results = []

    def bench(name, fn, cv, grid, block, inputs, expected_fn, bytes_rw, dtype=np.float32):
        try:
            so, _, cuda_src = _build_so(fn, cv)
            d_ptrs = [alloc_device(a) for a in inputs]
            # warmup
            run_kernel(so, f"locomp_launch_{fn.__name__}", grid, block, d_ptrs, n_iters=3)
            # timed
            ms = run_kernel(so, f"locomp_launch_{fn.__name__}", grid, block, d_ptrs, n_iters=20)
            out = read_device(d_ptrs[-1], inputs[-1])
            for p in d_ptrs:
                libcudart.cudaFree(p)
            exp = expected_fn(*inputs[:-1])
            tol = 1e-2 if dtype in (np.float16,) else 1e-3
            max_err = float(np.max(np.abs(out.astype(np.float32) - exp.astype(np.float32))))
            status = "PASS" if max_err < tol else "FAIL"
            gbps = (bytes_rw / 1e9) / (ms / 1000)
            results.append({"name": name, "status": status, "max_err": max_err,
                            "ms": ms, "gbps": gbps})
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[ERROR in {name}]\n{tb}", flush=True)
            results.append({"name": name, "status": "ERROR", "error": str(e)})

    # Use 256 threads/block for real GPU occupancy — kernels use block+thread index
    BS = 256
    N = 16 * 1024 * 1024
    BLOCKS = N // BS  # 65536 blocks

    # ── 1. vector_add  16M  (256 threads/block) ───────────────────────────────
    def vector_add(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor,
                   N: locomp.constexpr):
        bid = locomp.program_id(0)
        tid = locomp.local_id(0)
        i = bid * 256 + tid
        locomp.store(Out + i, locomp.load(A + i) + locomp.load(B + i))
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    bench("vector_add (16M f32)", vector_add, {"N": N}, (BLOCKS,), (BS,),
          [a, b, np.zeros(N, dtype=np.float32)],
          lambda a, b: a + b, 3 * N * 4)

    # ── 2. scale_shift  16M ───────────────────────────────────────────────────
    def scale_shift(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        bid = locomp.program_id(0)
        tid = locomp.local_id(0)
        i = bid * 256 + tid
        locomp.store(Out + i, locomp.load(X + i) * 2.0 + 1.0)
    x = np.random.randn(N).astype(np.float32)
    bench("scale_shift (16M f32)", scale_shift, {"N": N}, (BLOCKS,), (BS,),
          [x, np.zeros(N, dtype=np.float32)],
          lambda x: x * 2.0 + 1.0, 2 * N * 4)

    # ── 3. relu  16M ──────────────────────────────────────────────────────────
    def relu(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        bid = locomp.program_id(0)
        tid = locomp.local_id(0)
        i = bid * 256 + tid
        x = locomp.load(X + i)
        locomp.store(Out + i, locomp.where(x > 0.0, x, 0.0))
    x = np.random.randn(N).astype(np.float32)
    bench("relu (16M f32)", relu, {"N": N}, (BLOCKS,), (BS,),
          [x, np.zeros(N, dtype=np.float32)],
          lambda x: np.maximum(x, 0.0), 2 * N * 4)

    # ── 4. sqrt_exp  16M ──────────────────────────────────────────────────────
    def sqrt_exp(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        bid = locomp.program_id(0)
        tid = locomp.local_id(0)
        i = bid * 256 + tid
        x = locomp.load(X + i)
        locomp.store(Out + i, locomp.exp(locomp.sqrt(locomp.abs(x))))
    x = np.random.randn(N).astype(np.float32)
    bench("sqrt_exp (16M f32)", sqrt_exp, {"N": N}, (BLOCKS,), (BS,),
          [x, np.zeros(N, dtype=np.float32)],
          lambda x: np.exp(np.sqrt(np.abs(x))), 2 * N * 4)

    # ── 5. dot_product  64 x 1024 ────────────────────────────────────────────
    M, K = 64, 1024
    def dot_product(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor,
                    N: locomp.constexpr):
        i = locomp.program_id(0)
        acc = 0.0
        for k in range(N):
            acc = acc + locomp.load(A + i * N + k) * locomp.load(B + i * N + k)
        locomp.store(Out + i, acc)
    a2 = np.random.randn(M, K).astype(np.float32)
    b2 = np.random.randn(M, K).astype(np.float32)
    bench("dot_product (64x1024)", dot_product, {"N": K}, (M,), (1,),
          [a2.flatten(), b2.flatten(), np.zeros(M, dtype=np.float32)],
          lambda a, b: (a.reshape(M, K) * b.reshape(M, K)).sum(axis=1),
          (2 * M * K + M) * 4)

    # ── 6. shared_memory correctness ─────────────────────────────────────────
    Ns = BS * 128  # 32768 elements, 128 blocks of 256 threads
    def smem_copy(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        """Copy through shared memory: load to smem, barrier, store from smem."""
        tile = locomp.shared_memory(256, locomp.float32)
        block_id = locomp.program_id(0)
        tid = locomp.local_id(0)
        idx = block_id * 256 + tid
        locomp.shared_store(tile, tid, locomp.load(A + idx))
        locomp.barrier()
        locomp.store(Out + idx, locomp.shared_load(tile, tid))
    a3 = np.random.randn(Ns).astype(np.float32)
    bench("shared_memory copy", smem_copy, {"N": Ns}, (Ns // BS,), (BS,),
          [a3, np.zeros(Ns, dtype=np.float32)],
          lambda a: a, 2 * Ns * 4)

    # ── 7. trig_fused — sin+cos compute throughput ───────────────────────────
    def trig_fused(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        bid = locomp.program_id(0)
        tid = locomp.local_id(0)
        i = bid * 256 + tid
        x = locomp.load(X + i)
        locomp.store(Out + i, locomp.sin(x) + locomp.cos(x))
    x_trig = np.random.randn(N).astype(np.float32)
    bench("trig_fused (16M f32)", trig_fused, {"N": N}, (BLOCKS,), (BS,),
          [x_trig, np.zeros(N, dtype=np.float32)],
          lambda x: np.sin(x) + np.cos(x), 2 * N * 4)

    # ── 8. wmma tensor core matmul  1024×1024×1024 fp16 ─────────────────────
    # One warp (32 threads) per 16×16 output tile. Grid = (N/16, M/16).
    # A[M×K] fp16, B[K×N] fp16, C[M×N] fp32
    TM, TN, TK = 1024, 1024, 1024
    def wmma_matmul(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor,
                    M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
        bx = locomp.program_id(0)   # tile column (N dim)
        by = locomp.program_id(1)   # tile row    (M dim)
        acc = locomp.simdgroup_matrix(0.0)
        for k in range(K // 16):
            a_frag = locomp.simdgroup_matrix_load_device(
                A + by * 16 * K + k * 16, K, role="a")
            b_frag = locomp.simdgroup_matrix_load_device(
                B + k * 16 * N + bx * 16, N, role="b")
            acc = locomp.simdgroup_mac(acc, a_frag, b_frag)
        locomp.simdgroup_matrix_store_device(acc, Out + by * 16 * N + bx * 16, N)

    a_fp16 = np.random.randn(TM, TK).astype(np.float16)
    b_fp16 = np.random.randn(TK, TN).astype(np.float16)
    grid_wmma = (TN // 16, TM // 16)   # (64, 64) = 4096 blocks
    block_wmma = (32,)                  # 1 warp = 32 threads

    def expected_wmma(a, b):
        return (a.astype(np.float32) @ b.astype(np.float32)).flatten()

    # flops = 2 * M * N * K = 2.147e12 for 1024^3
    flops_wmma = 2 * TM * TN * TK
    try:
        so_wmma, _, cuda_src_wmma = _build_so(wmma_matmul, {"M": TM, "N": TN, "K": TK})
        d_a = alloc_device(a_fp16.flatten())
        d_b = alloc_device(b_fp16.flatten())
        d_out = alloc_device(np.zeros(TM * TN, dtype=np.float32))
        # warmup
        run_kernel(so_wmma, "locomp_launch_wmma_matmul", grid_wmma, block_wmma,
                   [d_a, d_b, d_out], n_iters=3)
        ms_wmma = run_kernel(so_wmma, "locomp_launch_wmma_matmul", grid_wmma, block_wmma,
                             [d_a, d_b, d_out], n_iters=20)
        out_wmma = read_device(d_out, np.zeros(TM * TN, dtype=np.float32))
        for p in [d_a, d_b, d_out]:
            libcudart.cudaFree(p)
        exp_wmma = expected_wmma(a_fp16, b_fp16)
        max_err_wmma = float(np.max(np.abs(out_wmma - exp_wmma)))
        status_wmma = "PASS" if max_err_wmma < 0.5 else "FAIL"  # fp16 accumulation tolerance
        tflops = (flops_wmma / 1e12) / (ms_wmma / 1000)
        results.append({"name": "wmma_matmul (1024³ fp16)", "status": status_wmma,
                        "max_err": max_err_wmma, "ms": ms_wmma,
                        "gbps": 0, "tflops": tflops})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR in wmma_matmul]\n{tb}", flush=True)
        results.append({"name": "wmma_matmul (1024³ fp16)", "status": "ERROR", "error": str(e)})

    return results


@app.local_entrypoint()
def main():
    results = run_cuda_benchmarks.remote()
    print(flush=True)
    print("=" * 70, flush=True)
    print("  locomp CUDA Backend — Modal A10G Benchmark Results", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Kernel':<28} {'Status':<8} {'MaxErr':>10} {'Time':>9} {'GB/s':>10}", flush=True)
    print("-" * 70, flush=True)
    passed = failed = 0
    for r in results:
        if r["status"] == "ERROR":
            print(f"  {r['name']:<28} {'ERROR':<8}  {r.get('error', '')[:80]}", flush=True)
            failed += 1
        elif r["status"] == "PASS":
            if r.get("tflops"):
                print(f"  {r['name']:<28} {'PASS':<8} {r['max_err']:>10.2e} {r['ms']:>7.3f}ms {r['tflops']:>7.2f} TFLOPS", flush=True)
            else:
                print(f"  {r['name']:<28} {'PASS':<8} {r['max_err']:>10.2e} {r['ms']:>7.3f}ms {r['gbps']:>8.1f}", flush=True)
            passed += 1
        else:
            print(f"  {r['name']:<28} {'FAIL':<8} {r['max_err']:>10.2e}", flush=True)
            failed += 1
    print("=" * 70, flush=True)
    print(f"  {passed} passed, {failed} failed", flush=True)
    print("=" * 70, flush=True)


