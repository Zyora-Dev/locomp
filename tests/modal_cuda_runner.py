"""
locomp CUDA backend — Modal A100 benchmark + Triton comparison.

Run with:
    modal run tests/modal_cuda_runner.py

Benchmarks (A100-80GB / A100-SXM-140GB):
  - 1B-element bandwidth wall: vector_add, scale_shift, relu        → tests ~2 TB/s HBM2e
  - 16M-element compute:       sqrt_exp, trig_fused, dot_product    → maths throughput
  - Shared memory copy:        correctness + bandwidth
  - Triton comparison:         vector_add, scale_shift, relu side-by-side (same grid/block)
  Reports: status, max absolute error, warm kernel time (ms), memory throughput (GB/s),
           and vs-Triton speedup where applicable.
"""

import modal
import numpy as np

# Triton kernels must be at module level — Triton's JIT parser reads source AST
# and needs `tl` in the module global scope, not in an enclosing function.
try:
    import triton
    import triton.language as tl

    BLOCK_T = 1024

    @triton.jit
    def triton_vector_add(A_ptr, B_ptr, Out_ptr, N, BLOCK: tl.constexpr):
        pid  = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        tl.store(Out_ptr + offs,
                 tl.load(A_ptr + offs, mask=mask) + tl.load(B_ptr + offs, mask=mask),
                 mask=mask)

    @triton.jit
    def triton_scale_shift(X_ptr, Out_ptr, N, BLOCK: tl.constexpr):
        pid  = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        tl.store(Out_ptr + offs, tl.load(X_ptr + offs, mask=mask) * 2.0 + 1.0, mask=mask)

    @triton.jit
    def triton_relu(X_ptr, Out_ptr, N, BLOCK: tl.constexpr):
        pid  = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X_ptr + offs, mask=mask)
        tl.store(Out_ptr + offs, tl.where(x > 0.0, x, 0.0), mask=mask)

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

app = modal.App("locomp-cuda-a100")

# ─── image ────────────────────────────────────────────────────────────────────
# PyTorch + Triton pulled in for the comparison section only.
# CUDA 12.4 devel image → nvcc available at /usr/local/cuda/bin/nvcc
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "numpy",
        # PyTorch cu121 wheel — compatible with CUDA driver 12.0 (Modal A100)
        "torch==2.3.1",
        "triton==2.3.1",
        find_links="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        # Install locomp from latest main — __restrict__ + __ldg + nvcc -O3 (f8ac2ae)
        "pip install 'git+https://github.com/Zyora-Dev/locomp.git@main'",
    )
)


# ─── benchmark function ───────────────────────────────────────────────────────

@app.function(gpu="A100-80GB", image=image, timeout=600)
def run_cuda_benchmarks():
    import locomp
    import numpy as np
    import subprocess
    import ctypes
    import tempfile
    import os
    import hashlib
    import time
    from locomp.backends.cuda_runtime import get_runtime, CUDARuntime
    from locomp.frontend import compile_kernel
    from locomp.optimizer import optimize
    from locomp.backends.cuda_codegen import compile_to_cuda

    # ── Runtime init ──────────────────────────────────────────────────────────
    rt = get_runtime()
    print(f"[info] CUDA devices: {rt.device_count()}", flush=True)

    # ── SM arch ───────────────────────────────────────────────────────────────
    sm_arch = "sm_80"   # A100 = sm_80; A100 SXM4 = sm_80 too
    try:
        r_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if r_smi.returncode == 0:
            first = r_smi.stdout.strip().split("\n")[0]
            cap_str = first.split(",")[0].strip().replace(".", "")
            name_str = first.split(",", 1)[1].strip() if "," in first else ""
            if cap_str.isdigit():
                sm_arch = f"sm_{cap_str}"
            print(f"[info] GPU: {name_str}  arch: {sm_arch}", flush=True)
    except Exception:
        pass

    _so_cache: dict = {}

    def _build_so(fn, constexpr_values: dict):
        ir = compile_kernel(fn)
        ir = optimize(ir, target="cuda")
        cuda_src, param_map = compile_to_cuda(ir, constexpr_values=constexpr_values)
        key = hashlib.sha256(cuda_src.encode()).hexdigest()
        if key in _so_cache:
            return _so_cache[key], param_map, cuda_src
        with tempfile.NamedTemporaryFile(suffix=".cu", delete=False, mode="w") as f:
            f.write(cuda_src)
            cu = f.name
        so = cu.replace(".cu", ".so")
        r = subprocess.run(
            ["nvcc", f"-arch={sm_arch}", "-O3", "-w",
             "-shared", "-Xcompiler", "-fPIC", "-o", so, cu],
            capture_output=True, text=True)
        os.unlink(cu)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{r.stderr[:2000]}")
        _so_cache[key] = so
        return so, param_map, cuda_src

    def _run_timed(so, fn_name, grid, block, d_tensors, n_iters=1):
        """Run locomp_launch_<fn_name> n_iters times. Returns ms/call."""
        lib = ctypes.CDLL(so)
        fn = getattr(lib, fn_name)
        fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.POINTER(ctypes.c_void_p)]
        fn.restype = None
        ptrs = [ctypes.c_void_p(t._cuda_ptr) for t in d_tensors]
        arr_t = (ctypes.c_void_p * len(ptrs))
        c_arr = arr_t(*ptrs)
        vpp = ctypes.cast(c_arr, ctypes.POINTER(ctypes.c_void_p))
        gx = grid[0]; gy = grid[1] if len(grid) > 1 else 1
        bx = block[0]; by = block[1] if len(block) > 1 else 1
        rt.sync()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fn(gx, gy, bx, by, vpp)
        rt.sync()
        return (time.perf_counter() - t0) * 1000 / n_iters  # ms

    results = []

    def bench(name, fn, cv, grid, block, arrays, expected_fn, bytes_rw,
              dtype=np.float32, warmup=10, iters=100):
        """Compile, upload, warm up, time, download, check — single benchmark."""
        try:
            so, _, _ = _build_so(fn, cv)
            d = [rt.upload(a) for a in arrays]
            _run_timed(so, f"locomp_launch_{fn.__name__}", grid, block, d, n_iters=warmup)
            ms = _run_timed(so, f"locomp_launch_{fn.__name__}", grid, block, d, n_iters=iters)
            out = d[-1].numpy().astype(np.float32).flatten()
            for t in d:
                t.free()
            exp = expected_fn(*arrays[:-1]).astype(np.float32).flatten()
            tol = 1e-2 if dtype == np.float16 else 1e-3
            max_err = float(np.max(np.abs(out - exp)))
            status = "PASS" if max_err < tol else "FAIL"
            gbps = (bytes_rw / 1e9) / (ms / 1000)
            results.append({"name": name, "status": status,
                            "max_err": max_err, "ms": ms, "gbps": gbps})
        except Exception as e:
            import traceback
            print(f"[ERROR {name}]\n{traceback.format_exc()}", flush=True)
            results.append({"name": name, "status": "ERROR", "error": str(e)})

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1 — 1B-element bandwidth tests (stress A100 HBM2e ~2 TB/s)
    # ═══════════════════════════════════════════════════════════════════════════
    # Tiled kernels: each thread handles EPT=4 consecutive elements via float4
    # load/store (LDG.128 / STG.128). Grid = N / (BS * EPT).
    BS  = 256    # threads per block
    EPT = 4      # elements per thread  (must be multiple of 4 for float4)
    BLK = BS * EPT  # = 1024 elements per block — mirrors Triton BLOCK=1024
    N1B = 1 << 30   # 1 073 741 824 elements × 4 bytes = 4 GB per array
    G1B = N1B // BLK

    def vector_add(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor,
                   N: locomp.constexpr):
        bid = locomp.program_id(0)
        tid = locomp.local_id(0)
        base = (bid * 256 + tid) * 4
        offs = locomp.arange(0, 4) + base
        locomp.store(Out + offs, locomp.load(A + offs) + locomp.load(B + offs))

    def scale_shift(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        bid = locomp.program_id(0)
        tid = locomp.local_id(0)
        base = (bid * 256 + tid) * 4
        offs = locomp.arange(0, 4) + base
        locomp.store(Out + offs, locomp.load(X + offs) * 2.0 + 1.0)

    def relu(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        bid = locomp.program_id(0)
        tid = locomp.local_id(0)
        base = (bid * 256 + tid) * 4
        offs = locomp.arange(0, 4) + base
        x = locomp.load(X + offs)
        locomp.store(Out + offs, locomp.where(x > 0.0, x, 0.0))

    rng = np.random.default_rng(42)
    a1b = rng.standard_normal(N1B).astype(np.float32)
    b1b = rng.standard_normal(N1B).astype(np.float32)
    z1b = np.zeros(N1B, dtype=np.float32)

    bench("vector_add   (1B f32)", vector_add, {"N": N1B}, (G1B,), (BS,),
          [a1b, b1b, z1b.copy()], lambda a, b: a + b, 3 * N1B * 4)
    bench("scale_shift  (1B f32)", scale_shift, {"N": N1B}, (G1B,), (BS,),
          [a1b, z1b.copy()], lambda x: x * 2.0 + 1.0,  2 * N1B * 4)
    bench("relu         (1B f32)", relu, {"N": N1B}, (G1B,), (BS,),
          [a1b, z1b.copy()], lambda x: np.maximum(x, 0.0), 2 * N1B * 4)

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2 — 16M-element compute-heavy kernels
    # ═══════════════════════════════════════════════════════════════════════════
    N16M = 16 * 1024 * 1024
    G16M = N16M // BS

    def sqrt_exp(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        bid = locomp.program_id(0)
        tid = locomp.local_id(0)
        i = bid * 256 + tid
        x = locomp.load(X + i)
        locomp.store(Out + i, locomp.exp(locomp.sqrt(locomp.abs(x))))

    def trig_fused(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        bid = locomp.program_id(0)
        tid = locomp.local_id(0)
        i = bid * 256 + tid
        x = locomp.load(X + i)
        locomp.store(Out + i, locomp.sin(x) + locomp.cos(x))

    x16 = rng.standard_normal(N16M).astype(np.float32)
    bench("sqrt_exp     (16M f32)", sqrt_exp, {"N": N16M}, (G16M,), (BS,),
          [x16, np.zeros(N16M, dtype=np.float32)],
          lambda x: np.exp(np.sqrt(np.abs(x))), 2 * N16M * 4)
    bench("trig_fused   (16M f32)", trig_fused, {"N": N16M}, (G16M,), (BS,),
          [x16, np.zeros(N16M, dtype=np.float32)],
          lambda x: np.sin(x) + np.cos(x), 2 * N16M * 4)

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3 — Shared memory correctness + bandwidth
    # ═══════════════════════════════════════════════════════════════════════════
    N_smem = BS * 65536   # 16M elements via 65536 blocks
    def smem_copy(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        tile = locomp.shared_memory(256, locomp.float32)
        bid  = locomp.program_id(0)
        tid  = locomp.local_id(0)
        idx  = bid * 256 + tid
        locomp.shared_store(tile, tid, locomp.load(A + idx))
        locomp.barrier()
        locomp.store(Out + idx, locomp.shared_load(tile, tid))

    a_sm = rng.standard_normal(N_smem).astype(np.float32)
    bench("smem_copy    (16M f32)", smem_copy, {"N": N_smem}, (N_smem // BS,), (BS,),
          [a_sm, np.zeros(N_smem, dtype=np.float32)],
          lambda a: a, 2 * N_smem * 4)

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 4 — Dot product (compute-bound, 1M rows × 128 cols)
    # ═══════════════════════════════════════════════════════════════════════════
    M_dp, K_dp = 1024 * 1024, 128

    def dot_product(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor,
                    N: locomp.constexpr):
        row = locomp.program_id(0)
        acc = 0.0
        for k in range(N):
            acc = acc + locomp.load(A + row * N + k) * locomp.load(B + row * N + k)
        locomp.store(Out + row, acc)

    a_dp = rng.standard_normal(M_dp * K_dp).astype(np.float32)
    b_dp = rng.standard_normal(M_dp * K_dp).astype(np.float32)
    bench("dot_product  (1M×128)", dot_product, {"N": K_dp}, (M_dp,), (1,),
          [a_dp, b_dp, np.zeros(M_dp, dtype=np.float32)],
          lambda a, b: (a.reshape(M_dp, K_dp) * b.reshape(M_dp, K_dp)).sum(axis=1),
          (2 * M_dp * K_dp + M_dp) * 4)

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 5 — Triton comparison  (vector_add / scale_shift / relu, 1B elem)
    # ═══════════════════════════════════════════════════════════════════════════
    triton_results = []
    try:
        if not _TRITON_AVAILABLE:
            raise ImportError("triton not available in this environment")
        import torch

        BLOCK_T = 1024

        def _triton_bench(name, kernel_fn, torch_inputs, N, bytes_rw, warmup=10, iters=100):
            grid = lambda _: (triton.cdiv(N, BLOCK_T),)
            torch.cuda.synchronize()
            for _ in range(warmup):
                kernel_fn[grid](*torch_inputs, N, BLOCK=BLOCK_T)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                kernel_fn[grid](*torch_inputs, N, BLOCK=BLOCK_T)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000 / iters
            gbps = (bytes_rw / 1e9) / (ms / 1000)
            triton_results.append({"name": name, "ms": ms, "gbps": gbps})

        a_t = torch.from_numpy(a1b).cuda()
        b_t = torch.from_numpy(b1b).cuda()
        o_t = torch.zeros(N1B, dtype=torch.float32, device="cuda")

        _triton_bench("Triton vector_add  (1B)", triton_vector_add, [a_t, b_t, o_t], N1B, 3*N1B*4)
        _triton_bench("Triton scale_shift (1B)", triton_scale_shift, [a_t, o_t], N1B, 2*N1B*4)
        _triton_bench("Triton relu        (1B)", triton_relu, [a_t, o_t], N1B, 2*N1B*4)

    except Exception as e:
        import traceback
        print(f"[Triton section ERROR]\n{traceback.format_exc()}", flush=True)
        triton_results.append({"name": "Triton section", "ms": 0, "gbps": 0,
                               "error": str(e)})

    return {"locomp": results, "triton": triton_results}


@app.local_entrypoint()
def main():
    data = run_cuda_benchmarks.remote()
    locomp_results  = data["locomp"]
    triton_results  = data["triton"]

    W = 74
    print(flush=True)
    print("=" * W, flush=True)
    print("  locomp CUDA — A100 Benchmark Results", flush=True)
    print("=" * W, flush=True)
    print(f"  {'Kernel':<30} {'Status':<6} {'MaxErr':>10} {'Time':>9} {'GB/s':>10}", flush=True)
    print("-" * W, flush=True)
    locomp_by_name = {}
    passed = failed = 0
    for r in locomp_results:
        if r["status"] == "ERROR":
            print(f"  {r['name']:<30} {'ERROR':<6}  {r.get('error','')[:60]}", flush=True)
            failed += 1
        elif r["status"] == "PASS":
            print(f"  {r['name']:<30} {'PASS':<6} {r['max_err']:>10.2e}"
                  f" {r['ms']:>7.3f}ms {r['gbps']:>8.1f} GB/s", flush=True)
            locomp_by_name[r["name"]] = r
            passed += 1
        else:
            print(f"  {r['name']:<30} {'FAIL':<6} {r['max_err']:>10.2e}", flush=True)
            failed += 1
    print("=" * W, flush=True)
    print(f"  locomp: {passed} passed, {failed} failed", flush=True)

    if triton_results:
        print(flush=True)
        print("=" * W, flush=True)
        print("  locomp vs Triton  (same operation, 1B float32 elements)", flush=True)
        print("=" * W, flush=True)
        print(f"  {'Kernel':<30} {'locomp':>10} {'Triton':>10} {'speedup':>10}", flush=True)
        print("-" * W, flush=True)
        pairs = [
            ("vector_add   (1B f32)", "Triton vector_add  (1B)"),
            ("scale_shift  (1B f32)", "Triton scale_shift (1B)"),
            ("relu         (1B f32)", "Triton relu        (1B)"),
        ]
        for lname, tname in pairs:
            lr = next((r for r in locomp_results if r["name"] == lname), None)
            tr = next((r for r in triton_results if r["name"] == tname), None)
            if lr and tr and tr.get("ms", 0) > 0 and lr.get("ms", 0) > 0:
                speedup = tr["ms"] / lr["ms"]
                tag = f"{speedup:.2f}x {'faster' if speedup >= 1 else 'slower'}"
                print(f"  {lname:<30} {lr['ms']:>8.3f}ms {tr['ms']:>8.3f}ms  {tag:>10}", flush=True)
            elif tr and tr.get("error"):
                print(f"  {lname:<30} {'(Triton error)':>32}", flush=True)
        print("=" * W, flush=True)

