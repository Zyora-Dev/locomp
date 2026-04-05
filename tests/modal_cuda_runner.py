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
        # Pinned to exact commit — forces Modal to rebuild image layer
        "pip install 'git+https://github.com/Zyora-Dev/locomp.git@c69fad2'",
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


# ─── gpu_ag CUDA backward pass ────────────────────────────────────────────────

@app.function(gpu="A100-80GB", image=image, timeout=300)
def run_gpu_ag_cuda():
    """
    Run gpu_ag forward + backward for add/sub/mul/div/exp/log/relu/pow/sigmoid/tanh
    on CUDA backend. Compares gradients against NumPy reference (finite differences).
    First time any gpu_ag CUDA execution has run on a real NVIDIA GPU.
    """
    import locomp
    import locomp.gpu_autograd as gpu_ag
    import numpy as np
    import traceback

    results = []

    def _check(name, got, expected, tol=2e-3):
        got = np.asarray(got, dtype=np.float32).flatten()
        expected = np.asarray(expected, dtype=np.float32).flatten()
        max_err = float(np.max(np.abs(got - expected)))
        status = "PASS" if max_err < tol else "FAIL"
        results.append({"name": name, "status": status, "max_err": max_err})
        print(f"  {name:<35} {status}  max_err={max_err:.2e}", flush=True)

    N = 1024
    rng = np.random.default_rng(7)
    a_np = rng.uniform(0.5, 2.0, N).astype(np.float32)
    b_np = rng.uniform(0.5, 2.0, N).astype(np.float32)

    def _t(arr): return gpu_ag.tensor(arr, requires_grad=True, backend="cuda")

    # ── add ───────────────────────────────────────────────────────────────────
    try:
        a, b = _t(a_np), _t(b_np)
        out = gpu_ag.add(a, b)
        loss = gpu_ag.sum(out)
        gpu_ag.backward(loss)
        _check("add fwd", out.numpy(), a_np + b_np)
        _check("add bwd a.grad", a.grad.numpy(), np.ones(N))
        _check("add bwd b.grad", b.grad.numpy(), np.ones(N))
    except Exception:
        results.append({"name": "add", "status": "ERROR", "max_err": -1})
        print(f"  add ERROR\n{traceback.format_exc()}", flush=True)
    gpu_ag.zero_grad(a, b)

    # ── sub ───────────────────────────────────────────────────────────────────
    try:
        a, b = _t(a_np), _t(b_np)
        out = gpu_ag.sub(a, b)
        loss = gpu_ag.sum(out)
        gpu_ag.backward(loss)
        _check("sub fwd", out.numpy(), a_np - b_np)
        _check("sub bwd a.grad", a.grad.numpy(), np.ones(N))
        _check("sub bwd b.grad", b.grad.numpy(), -np.ones(N))
    except Exception:
        results.append({"name": "sub", "status": "ERROR", "max_err": -1})
        print(f"  sub ERROR\n{traceback.format_exc()}", flush=True)
    gpu_ag.zero_grad(a, b)

    # ── mul ───────────────────────────────────────────────────────────────────
    try:
        a, b = _t(a_np), _t(b_np)
        out = gpu_ag.mul(a, b)
        loss = gpu_ag.sum(out)
        gpu_ag.backward(loss)
        _check("mul fwd", out.numpy(), a_np * b_np)
        _check("mul bwd a.grad", a.grad.numpy(), b_np)
        _check("mul bwd b.grad", b.grad.numpy(), a_np)
    except Exception:
        results.append({"name": "mul", "status": "ERROR", "max_err": -1})
        print(f"  mul ERROR\n{traceback.format_exc()}", flush=True)
    gpu_ag.zero_grad(a, b)

    # ── div ───────────────────────────────────────────────────────────────────
    try:
        a, b = _t(a_np), _t(b_np)
        out = gpu_ag.div(a, b)
        loss = gpu_ag.sum(out)
        gpu_ag.backward(loss)
        _check("div fwd", out.numpy(), a_np / b_np)
        _check("div bwd a.grad", a.grad.numpy(), 1.0 / b_np)
        _check("div bwd b.grad", b.grad.numpy(), -a_np / (b_np ** 2))
    except Exception:
        results.append({"name": "div", "status": "ERROR", "max_err": -1})
        print(f"  div ERROR\n{traceback.format_exc()}", flush=True)
    gpu_ag.zero_grad(a, b)

    # ── exp ───────────────────────────────────────────────────────────────────
    try:
        a = _t(a_np)
        out = gpu_ag.exp(a)
        loss = gpu_ag.sum(out)
        gpu_ag.backward(loss)
        _check("exp fwd", out.numpy(), np.exp(a_np))
        _check("exp bwd a.grad", a.grad.numpy(), np.exp(a_np))
    except Exception:
        results.append({"name": "exp", "status": "ERROR", "max_err": -1})
        print(f"  exp ERROR\n{traceback.format_exc()}", flush=True)

    # ── log ───────────────────────────────────────────────────────────────────
    try:
        a = _t(a_np)
        out = gpu_ag.log(a)
        loss = gpu_ag.sum(out)
        gpu_ag.backward(loss)
        _check("log fwd", out.numpy(), np.log(a_np))
        _check("log bwd a.grad", a.grad.numpy(), 1.0 / a_np)
    except Exception:
        results.append({"name": "log", "status": "ERROR", "max_err": -1})
        print(f"  log ERROR\n{traceback.format_exc()}", flush=True)

    # ── relu ──────────────────────────────────────────────────────────────────
    try:
        x_np = rng.uniform(-1.0, 1.0, N).astype(np.float32)
        a = _t(x_np)
        out = gpu_ag.relu(a)
        loss = gpu_ag.sum(out)
        gpu_ag.backward(loss)
        _check("relu fwd", out.numpy(), np.maximum(x_np, 0.0))
        _check("relu bwd a.grad", a.grad.numpy(), (x_np > 0).astype(np.float32))
    except Exception:
        results.append({"name": "relu", "status": "ERROR", "max_err": -1})
        print(f"  relu ERROR\n{traceback.format_exc()}", flush=True)

    # ── chain: relu(exp(a)) ───────────────────────────────────────────────────
    try:
        a = _t(a_np)
        mid = gpu_ag.exp(a)
        out = gpu_ag.relu(mid)
        loss = gpu_ag.sum(out)
        gpu_ag.backward(loss)
        exp_a = np.exp(a_np)
        expected_grad = (exp_a > 0).astype(np.float32) * exp_a
        _check("chain relu(exp) bwd", a.grad.numpy(), expected_grad)
    except Exception:
        results.append({"name": "chain relu(exp)", "status": "ERROR", "max_err": -1})
        print(f"  chain ERROR\n{traceback.format_exc()}", flush=True)

    return results


# ─── wmma / Tensor Core end-to-end ───────────────────────────────────────────

@app.function(gpu="A100-80GB", image=image, timeout=300)
def run_wmma_gemm():
    """
    Compile a WMMA 16×16×16 fp16 GEMM kernel with nvcc on A100 (sm_80),
    launch it, and verify the result against numpy reference.
    First time any locomp Tensor Core kernel has actually executed.
    """
    import locomp
    import numpy as np
    import subprocess, ctypes, tempfile, os, hashlib
    import traceback
    from locomp.backends.cuda_runtime import get_runtime
    from locomp.frontend import compile_kernel
    from locomp.optimizer import optimize
    from locomp.backends.cuda_codegen import compile_to_cuda

    rt = get_runtime()
    results = []

    sm_arch = "sm_80"
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            cap = r.stdout.strip().split("\n")[0].strip().replace(".", "")
            if cap.isdigit():
                sm_arch = f"sm_{cap}"
        print(f"[info] arch: {sm_arch}", flush=True)
    except Exception:
        pass

    def _build_so(fn, cv):
        ir = compile_kernel(fn)
        ir = optimize(ir, target="cuda")
        src, pmap = compile_to_cuda(ir, constexpr_values=cv)
        h = hashlib.sha256(src.encode()).hexdigest()[:16]
        with tempfile.NamedTemporaryFile(suffix=".cu", delete=False, mode="w") as f:
            f.write(src); cu = f.name
        so = cu.replace(".cu", ".so")
        r = subprocess.run(
            ["nvcc", f"-arch={sm_arch}", "-O3", "-w",
             "-shared", "-Xcompiler", "-fPIC", "-o", so, cu],
            capture_output=True, text=True)
        os.unlink(cu)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc error:\n{r.stderr[:3000]}")
        return so, pmap, src

    def _launch(so, fn_name, grid, block, d_tensors):
        lib = ctypes.CDLL(so)
        fn = getattr(lib, fn_name)
        fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.POINTER(ctypes.c_void_p)]
        fn.restype = None
        ptrs = [ctypes.c_void_p(t._cuda_ptr) for t in d_tensors]
        c_arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        fn(grid[0], 1, block[0], 1,
           ctypes.cast(c_arr, ctypes.POINTER(ctypes.c_void_p)))
        rt.sync()

    # ── Test 1: 16×16×16 single-tile WMMA GEMM ───────────────────────────────
    try:
        # One warp computes one 16×16 output tile: C = A @ B
        def wmma_gemm_16(A: locomp.Float16, B: locomp.Float16, C: locomp.Tensor,
                         M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
            # single warp kernel: warp 0 computes the 16×16 tile
            warp  = locomp.program_id(0)
            acc   = locomp.simdgroup_matrix(0.0)
            a_frag = locomp.simdgroup_matrix_load_device(A, K, role="a")
            b_frag = locomp.simdgroup_matrix_load_device(B, N, role="b")
            acc    = locomp.simdgroup_mac(acc, a_frag, b_frag)
            locomp.simdgroup_matrix_store_device(acc, C, N)

        M, N, K = 16, 16, 16
        rng = np.random.default_rng(42)
        # WMMA load_matrix_sync reads __half* from device memory — upload as fp16.
        # Expected is A@B computed from the fp16-rounded values (matches GPU precision).
        a_np = rng.standard_normal((M, K)).astype(np.float16)
        b_np = rng.standard_normal((K, N)).astype(np.float16)
        expected = (a_np.astype(np.float32) @ b_np.astype(np.float32))

        so, pmap, src = _build_so(wmma_gemm_16, {"M": M, "N": N, "K": K})
        print(f"[wmma] nvcc compile OK  ({sm_arch})", flush=True)
        print(f"[wmma] param_map: {pmap}", flush=True)

        d_a = rt.upload(a_np.flatten())
        d_b = rt.upload(b_np.flatten())
        d_c = rt.zeros(M * N, np.float32)

        # 1 warp = 32 threads; 1 block does the full 16×16 tile
        _launch(so, "locomp_launch_wmma_gemm_16", (1,), (32,), [d_a, d_b, d_c])

        got = d_c.numpy().reshape(M, N)
        max_err = float(np.max(np.abs(got - expected)))
        # fp16 intermediate → fp32 accumulator: expect ~1e-2 max error for random mats
        status = "PASS" if max_err < 0.5 else "FAIL"
        results.append({"name": "wmma_gemm_16x16", "status": status,
                        "max_err": max_err, "note": "fp16 tiles → fp32 acc"})
        print(f"  wmma_gemm 16×16×16   {status}  max_err={max_err:.3e}  "
              f"(fp16 precision expected)", flush=True)
        d_a.free(); d_b.free(); d_c.free()

    except Exception:
        results.append({"name": "wmma_gemm_16x16", "status": "ERROR", "max_err": -1})
        print(f"  wmma_gemm ERROR\n{traceback.format_exc()}", flush=True)

    # ── Test 2: 128×128 tiled WMMA GEMM (8×8 warp grid) ─────────────────────
    try:
        TM, TN = 128, 128
        TK = 64

        def wmma_gemm_tiled(A: locomp.Float16, B: locomp.Float16, C: locomp.Tensor,
                            M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
            warp = locomp.program_id(0)
            row  = warp // (N // 16)
            col  = warp  % (N // 16)
            acc  = locomp.simdgroup_matrix(0.0)
            for k in range(K // 16):
                a_frag = locomp.simdgroup_matrix_load_device(
                    A + (row * 16) * K + k * 16, K, role="a")
                b_frag = locomp.simdgroup_matrix_load_device(
                    B + k * 16 * N + col * 16, N, role="b")
                acc = locomp.simdgroup_mac(acc, a_frag, b_frag)
            locomp.simdgroup_matrix_store_device(
                acc, C + (row * 16) * N + col * 16, N)

        rng2 = np.random.default_rng(99)
        a2   = rng2.standard_normal((TM, TK)).astype(np.float16)
        b2   = rng2.standard_normal((TK, TN)).astype(np.float16)
        exp2 = (a2.astype(np.float32) @ b2.astype(np.float32))

        so2, _, _ = _build_so(wmma_gemm_tiled, {"M": TM, "N": TN, "K": TK})
        print(f"[wmma_tiled] nvcc compile OK  ({sm_arch})", flush=True)

        d_a2 = rt.upload(a2.flatten())
        d_b2 = rt.upload(b2.flatten())
        d_c2 = rt.zeros(TM * TN, np.float32)

        n_warps = (TM // 16) * (TN // 16)  # 8×8 = 64 warps
        _launch(so2, "locomp_launch_wmma_gemm_tiled", (n_warps,), (32,), [d_a2, d_b2, d_c2])

        got2     = d_c2.numpy().reshape(TM, TN)
        max_err2 = float(np.max(np.abs(got2 - exp2)))
        status2  = "PASS" if max_err2 < 1.0 else "FAIL"
        results.append({"name": "wmma_gemm_128x128", "status": status2,
                        "max_err": max_err2, "note": "fp16 tiles → fp32 acc"})
        print(f"  wmma_gemm 128×128×64 {status2}  max_err={max_err2:.3e}", flush=True)
        d_a2.free(); d_b2.free(); d_c2.free()

    except Exception:
        results.append({"name": "wmma_gemm_128x128", "status": "ERROR", "max_err": -1})
        print(f"  wmma_gemm_tiled ERROR\n{traceback.format_exc()}", flush=True)

    return results


# ─── warp intrinsics validation ───────────────────────────────────────────────

@app.function(gpu="A100-80GB", image=image, timeout=300)
def run_warp_intrinsics():
    """Numerically validate all warp intrinsics (SIMD_SUM/MAX/MIN/BROADCAST/SHUFFLE_DOWN)
    on a real A100 GPU.  Each test spawns 32 threads (one full warp) and checks
    the __shfl_down_sync / __shfl_sync output against known expected values."""
    import locomp
    import numpy as np
    import traceback
    import subprocess
    import ctypes
    import tempfile
    import os
    import hashlib
    from locomp.frontend import compile_kernel
    from locomp.optimizer import optimize
    from locomp.backends.cuda_codegen import compile_to_cuda
    from locomp.backends.cuda_runtime import get_runtime

    rt = get_runtime()
    sm_arch = "sm_80"
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            cap = r.stdout.strip().split("\n")[0].strip().replace(".", "")
            if cap.isdigit():
                sm_arch = f"sm_{cap}"
    except Exception:
        pass

    _so_cache: dict = {}

    def _build_so(fn, constexpr_values: dict = {}):
        ir = compile_kernel(fn)
        ir = optimize(ir, target="cuda")
        cuda_src, param_map = compile_to_cuda(ir, constexpr_values=constexpr_values)
        key = hashlib.sha256(cuda_src.encode()).hexdigest()
        if key in _so_cache:
            return _so_cache[key]
        with tempfile.NamedTemporaryFile(suffix=".cu", delete=False, mode="w") as f:
            f.write(cuda_src)
            cu_path = f.name
        so_path = cu_path.replace(".cu", ".so")
        subprocess.run(
            ["nvcc", "-O3", f"-arch={sm_arch}", "--shared", "--compiler-options",
             "-fPIC", "-o", so_path, cu_path],
            check=True, capture_output=True)
        so = ctypes.CDLL(so_path)
        _so_cache[key] = (so, param_map)
        return so, param_map

    def _launch(so, fn_name, grid, block, args):
        launch_fn = getattr(so, fn_name)
        launch_fn.restype = None
        launch_fn.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p]
        ptrs = (ctypes.c_void_p * len(args))(*[a._ptr for a in args])
        launch_fn(grid[0], grid[1] if len(grid) > 1 else 1,
                  block[0], block[1] if len(block) > 1 else 1,
                  ptrs)
        rt.sync()

    results = []

    # ── Test 1: warp_sum ──────────────────────────────────────────────────────
    # Lane i holds float(i).  __shfl_down_sync tree-reduces → lane 0 = sum(0..31) = 496.
    try:
        @locomp.kernel
        def warp_sum_k(OUT: locomp.Tensor):
            tid = locomp.local_id(0)
            val = locomp.cast(tid, "float32")
            reduced = locomp.simd_sum(val)
            if tid == 0:
                locomp.store(OUT + 0, reduced)

        d_out = rt.upload(np.zeros(1, dtype=np.float32))
        so, _ = _build_so(warp_sum_k.func)
        _launch(so, "locomp_launch_warp_sum_k", (1,), (32,), [d_out])
        got = d_out.numpy()[0]
        expected = float(sum(range(32)))   # 496.0
        err = abs(got - expected)
        status = "PASS" if err < 1e-3 else "FAIL"
        results.append({"name": "warp_sum", "status": status,
                        "got": got, "expected": expected, "err": err})
        print(f"  warp_sum      {status}  got={got:.1f}  expected={expected:.1f}  err={err:.1e}",
              flush=True)
        d_out.free()
    except Exception:
        results.append({"name": "warp_sum", "status": "ERROR"})
        print(f"  warp_sum      ERROR\n{traceback.format_exc()}", flush=True)

    # ── Test 2: warp_max ──────────────────────────────────────────────────────
    # Lane i holds float(i).  warp_max → lane 0 = 31.
    try:
        @locomp.kernel
        def warp_max_k(OUT: locomp.Tensor):
            tid = locomp.local_id(0)
            val = locomp.cast(tid, "float32")
            reduced = locomp.simd_max(val)
            if tid == 0:
                locomp.store(OUT + 0, reduced)

        d_out = rt.upload(np.zeros(1, dtype=np.float32))
        so, _ = _build_so(warp_max_k.func)
        _launch(so, "locomp_launch_warp_max_k", (1,), (32,), [d_out])
        got = d_out.numpy()[0]
        expected = 31.0
        err = abs(got - expected)
        status = "PASS" if err < 1e-3 else "FAIL"
        results.append({"name": "warp_max", "status": status,
                        "got": got, "expected": expected, "err": err})
        print(f"  warp_max      {status}  got={got:.1f}  expected={expected:.1f}  err={err:.1e}",
              flush=True)
        d_out.free()
    except Exception:
        results.append({"name": "warp_max", "status": "ERROR"})
        print(f"  warp_max      ERROR\n{traceback.format_exc()}", flush=True)

    # ── Test 3: warp_min ──────────────────────────────────────────────────────
    # Lane i holds float(i+1).  warp_min → lane 0 = 1.
    try:
        @locomp.kernel
        def warp_min_k(OUT: locomp.Tensor):
            tid = locomp.local_id(0)
            val = locomp.cast(tid + 1, "float32")
            reduced = locomp.simd_min(val)
            if tid == 0:
                locomp.store(OUT + 0, reduced)

        d_out = rt.upload(np.zeros(1, dtype=np.float32))
        so, _ = _build_so(warp_min_k.func)
        _launch(so, "locomp_launch_warp_min_k", (1,), (32,), [d_out])
        got = d_out.numpy()[0]
        expected = 1.0
        err = abs(got - expected)
        status = "PASS" if err < 1e-3 else "FAIL"
        results.append({"name": "warp_min", "status": status,
                        "got": got, "expected": expected, "err": err})
        print(f"  warp_min      {status}  got={got:.1f}  expected={expected:.1f}  err={err:.1e}",
              flush=True)
        d_out.free()
    except Exception:
        results.append({"name": "warp_min", "status": "ERROR"})
        print(f"  warp_min      ERROR\n{traceback.format_exc()}", flush=True)

    # ── Test 4: warp_broadcast ────────────────────────────────────────────────
    # Lane i holds float(i) + 42.0.  simd_broadcast from lane 0 → all get 42.0.
    try:
        @locomp.kernel
        def warp_broadcast_k(OUT: locomp.Tensor):
            tid = locomp.local_id(0)
            val = locomp.cast(tid, "float32") + 42.0
            bcast = locomp.simd_broadcast(val, 0)
            locomp.store(OUT + tid, bcast)

        d_out = rt.upload(np.zeros(32, dtype=np.float32))
        so, _ = _build_so(warp_broadcast_k.func)
        _launch(so, "locomp_launch_warp_broadcast_k", (1,), (32,), [d_out])
        got = d_out.numpy()
        expected_val = 42.0
        err = float(np.max(np.abs(got - expected_val)))
        status = "PASS" if err < 1e-3 else "FAIL"
        results.append({"name": "warp_broadcast", "status": status,
                        "got": got.tolist(), "expected": expected_val, "err": err})
        print(f"  warp_broadcast {status}  all=42.0?  max_err={err:.1e}", flush=True)
        d_out.free()
    except Exception:
        results.append({"name": "warp_broadcast", "status": "ERROR"})
        print(f"  warp_broadcast ERROR\n{traceback.format_exc()}", flush=True)

    # ── Test 5: warp_shuffle_down ─────────────────────────────────────────────
    # Lane i holds float(i).  shfl_down(1) → lane i gets val from lane i+1.
    # OUT[i] = i+1 for i=0..30; OUT[31] = 31 (CUDA: out-of-warp returns own value).
    try:
        @locomp.kernel
        def warp_shuffle_k(OUT: locomp.Tensor):
            tid = locomp.local_id(0)
            val = locomp.cast(tid, "float32")
            down = locomp.simd_shuffle_down(val, 1)
            locomp.store(OUT + tid, down)

        d_out = rt.upload(np.zeros(32, dtype=np.float32))
        so, _ = _build_so(warp_shuffle_k.func)
        _launch(so, "locomp_launch_warp_shuffle_k", (1,), (32,), [d_out])
        got = d_out.numpy()
        expected = np.array([float(i + 1) for i in range(31)] + [31.0], dtype=np.float32)
        err = float(np.max(np.abs(got - expected)))
        status = "PASS" if err < 1e-3 else "FAIL"
        results.append({"name": "warp_shuffle_down", "status": status,
                        "got": got.tolist(), "err": err})
        print(f"  warp_shuffle   {status}  max_err={err:.1e}", flush=True)
        d_out.free()
    except Exception:
        results.append({"name": "warp_shuffle_down", "status": "ERROR"})
        print(f"  warp_shuffle   ERROR\n{traceback.format_exc()}", flush=True)

    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "PASS")
    print(f"\n  warp intrinsics: {passed}/{total} passed", flush=True)
    return results


# ─── full CUDA benchmark suite ────────────────────────────────────────────────

@app.function(gpu="A100-80GB", image=image, timeout=600)
def run_full_cuda_benchmark():
    """Comprehensive CUDA benchmark — equivalent of examples/58_benchmark_suite.py
    but running on A100.  Covers all major transformer inference kernels and
    measures A100 GPU vs NumPy, reporting timing, max_err, and speedup."""
    import locomp
    import numpy as np
    import time
    import traceback
    import subprocess
    import ctypes
    import tempfile
    import os
    import hashlib
    from locomp.frontend import compile_kernel
    from locomp.optimizer import optimize
    from locomp.backends.cuda_codegen import compile_to_cuda
    from locomp.backends.cuda_runtime import get_runtime

    rt = get_runtime()
    sm_arch = "sm_80"
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            first = r.stdout.strip().split("\n")[0]
            cap_str = first.split(",")[0].strip().replace(".", "")
            name_str = first.split(",", 1)[1].strip() if "," in first else ""
            if cap_str.isdigit():
                sm_arch = f"sm_{cap_str}"
            print(f"[bench] GPU: {name_str}  arch: {sm_arch}", flush=True)
    except Exception:
        pass

    _so_cache: dict = {}

    def _build_so(fn, constexpr_values: dict = {}):
        ir = compile_kernel(fn)
        ir = optimize(ir, target="cuda")
        cuda_src, param_map = compile_to_cuda(ir, constexpr_values=constexpr_values)
        key = hashlib.sha256(cuda_src.encode()).hexdigest()
        if key in _so_cache:
            return _so_cache[key]
        with tempfile.NamedTemporaryFile(suffix=".cu", delete=False, mode="w") as f:
            f.write(cuda_src)
            cu_path = f.name
        so_path = cu_path.replace(".cu", ".so")
        subprocess.run(
            ["nvcc", "-O3", f"-arch={sm_arch}", "--shared", "--compiler-options",
             "-fPIC", "-o", so_path, cu_path],
            check=True, capture_output=True)
        so = ctypes.CDLL(so_path)
        _so_cache[key] = (so, param_map)
        return so, param_map

    def _launch(so, fn_name, grid, block, args):
        launch_fn = getattr(so, fn_name)
        launch_fn.restype = None
        launch_fn.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p]
        ptrs = (ctypes.c_void_p * len(args))(*[a._ptr for a in args])
        launch_fn(grid[0], grid[1] if len(grid) > 1 else 1,
                  block[0], block[1] if len(block) > 1 else 1,
                  ptrs)
        rt.sync()

    WARMUP = 5
    RUNS = 15

    def bench_gpu(launch_fn):
        for _ in range(WARMUP):
            launch_fn()
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            launch_fn()
            times.append((time.perf_counter() - t0) * 1000)
        return sorted(times)[RUNS // 2]

    def bench_np(fn):
        for _ in range(WARMUP):
            fn()
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
        return sorted(times)[RUNS // 2]

    def row(label, t_gpu, t_np, err, status="PASS"):
        r = t_gpu / t_np
        speedup = f"{1/r:.2f}x faster" if r < 1.0 else f"{r:.2f}x slower"
        print(f"  {label:<44} {status:<5} gpu={t_gpu:7.3f}ms  np={t_np:7.3f}ms  "
              f"{speedup:<18}  err={err:.1e}", flush=True)

    all_results = []
    W = 80

    # ── 1. Softmax ────────────────────────────────────────────────────────────
    # Each row is softmax-reduced.  Uses __shfl_down_sync internally (via simd_sum)
    # → this also numerically validates warp sum reduction in a real workload.
    print(f"\n{'='*W}", flush=True)
    print("  [1] Softmax  (validates warp __shfl_down_sync reduction)", flush=True)
    print(f"{'='*W}", flush=True)

    @locomp.kernel
    def softmax_k(X: locomp.Tensor, O: locomp.Tensor,
                  ROWS: locomp.constexpr, D: locomp.constexpr):
        r = locomp.program_id(0)
        m = -3.4e38
        for j in range(D):
            v = locomp.load(X + r * D + j)
            if v > m:
                m = v
        s = 0.0
        for j in range(D):
            v = locomp.load(X + r * D + j)
            s = s + locomp.exp(v - m)
        for j in range(D):
            v = locomp.load(X + r * D + j)
            locomp.store(O + r * D + j, locomp.exp(v - m) / s)

    for B, D in [(32, 128), (64, 256), (128, 512), (256, 1024)]:
        try:
            data = np.random.randn(B, D).astype(np.float32)
            x_g = rt.upload(data.flatten())
            o_g = rt.upload(np.zeros(B * D, dtype=np.float32))
            so, _ = _build_so(softmax_k.func, {"ROWS": B, "D": D})

            def gpu_fn():
                _launch(so, "locomp_launch_softmax_k", (B,), (1,), [x_g, o_g])

            def np_fn():
                e = np.exp(data - data.max(axis=1, keepdims=True))
                return e / e.sum(axis=1, keepdims=True)

            gpu_fn()
            err = float(np.max(np.abs(o_g.numpy().reshape(B, D) - np_fn())))
            status = "PASS" if err < 1e-4 else "FAIL"
            t_gpu = bench_gpu(gpu_fn)
            t_np = bench_np(np_fn)
            row(f"softmax B={B} D={D}", t_gpu, t_np, err, status)
            all_results.append({"bench": "softmax", "B": B, "D": D,
                                 "status": status, "err": err,
                                 "t_gpu_ms": t_gpu, "t_np_ms": t_np})
            x_g.free(); o_g.free()
        except Exception:
            print(f"  softmax B={B} D={D}  ERROR\n{traceback.format_exc()}", flush=True)
            all_results.append({"bench": "softmax", "B": B, "D": D, "status": "ERROR"})

    # ── 2. RMS Norm ───────────────────────────────────────────────────────────
    print(f"\n{'='*W}", flush=True)
    print("  [2] RMS Norm  (LLaMA-style pre-norm)", flush=True)
    print(f"{'='*W}", flush=True)

    @locomp.kernel
    def rms_norm_k(X: locomp.Tensor, W: locomp.Tensor, O: locomp.Tensor,
                   ROWS: locomp.constexpr, D: locomp.constexpr, eps: locomp.constexpr):
        r = locomp.program_id(0)
        ss = 0.0
        for j in range(D):
            v = locomp.load(X + r * D + j)
            ss = ss + v * v
        rms = locomp.rsqrt(ss / D + eps)
        for j in range(D):
            v = locomp.load(X + r * D + j)
            w = locomp.load(W + j)
            locomp.store(O + r * D + j, v * rms * w)

    for B, D in [(32, 512), (32, 2048), (32, 4096), (128, 4096)]:
        try:
            data = np.random.randn(B, D).astype(np.float32)
            weight = np.ones(D, dtype=np.float32)
            eps = 1e-5
            x_g = rt.upload(data.flatten())
            w_g = rt.upload(weight)
            o_g = rt.upload(np.zeros(B * D, dtype=np.float32))
            so, _ = _build_so(rms_norm_k.func, {"ROWS": B, "D": D, "eps": eps})

            def gpu_fn():
                _launch(so, "locomp_launch_rms_norm_k", (B,), (1,), [x_g, w_g, o_g])

            def np_fn():
                rms = np.sqrt((data ** 2).mean(axis=1, keepdims=True) + eps)
                return data / rms * weight

            gpu_fn()
            err = float(np.max(np.abs(o_g.numpy().reshape(B, D) - np_fn())))
            status = "PASS" if err < 1e-4 else "FAIL"
            t_gpu = bench_gpu(gpu_fn)
            t_np = bench_np(np_fn)
            row(f"rms_norm B={B} D={D}", t_gpu, t_np, err, status)
            all_results.append({"bench": "rms_norm", "B": B, "D": D,
                                 "status": status, "err": err,
                                 "t_gpu_ms": t_gpu, "t_np_ms": t_np})
            x_g.free(); w_g.free(); o_g.free()
        except Exception:
            print(f"  rms_norm B={B} D={D}  ERROR\n{traceback.format_exc()}", flush=True)
            all_results.append({"bench": "rms_norm", "B": B, "D": D, "status": "ERROR"})

    # ── 3. GELU activation ────────────────────────────────────────────────────
    print(f"\n{'='*W}", flush=True)
    print("  [3] GELU activation  (MLP FFN non-linearity)", flush=True)
    print(f"{'='*W}", flush=True)

    @locomp.kernel
    def gelu_k(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        x = locomp.load(X + i)
        c = 0.7978845608028654
        inner = c * (x + 0.044715 * x * x * x)
        locomp.store(O + i, 0.5 * x * (1.0 + locomp.tanh(inner)))

    for N in [65536, 262144, 1048576, 4194304]:
        try:
            data = np.random.randn(N).astype(np.float32)
            x_g = rt.upload(data)
            o_g = rt.upload(np.zeros(N, dtype=np.float32))
            so, _ = _build_so(gelu_k.func, {"N": N})

            def gpu_fn():
                _launch(so, "locomp_launch_gelu_k", (N,), (1,), [x_g, o_g])

            def np_fn():
                c = 0.7978845608028654
                return 0.5 * data * (1 + np.tanh(c * (data + 0.044715 * data ** 3)))

            gpu_fn()
            err = float(np.max(np.abs(o_g.numpy() - np_fn())))
            status = "PASS" if err < 1e-5 else "FAIL"
            t_gpu = bench_gpu(gpu_fn)
            t_np = bench_np(np_fn)
            row(f"gelu N={N}", t_gpu, t_np, err, status)
            all_results.append({"bench": "gelu", "N": N,
                                 "status": status, "err": err,
                                 "t_gpu_ms": t_gpu, "t_np_ms": t_np})
            x_g.free(); o_g.free()
        except Exception:
            print(f"  gelu N={N}  ERROR\n{traceback.format_exc()}", flush=True)
            all_results.append({"bench": "gelu", "N": N, "status": "ERROR"})

    # ── 4. Vector Add ─────────────────────────────────────────────────────────
    print(f"\n{'='*W}", flush=True)
    print("  [4] Vector Add  (elementwise, pure memory-bound)", flush=True)
    print(f"{'='*W}", flush=True)

    @locomp.kernel
    def vec_add_k(X: locomp.Tensor, Y: locomp.Tensor, O: locomp.Tensor,
                  N: locomp.constexpr):
        i = locomp.program_id(0)
        locomp.store(O + i, locomp.load(X + i) + locomp.load(Y + i))

    for N in [1048576, 16777216, 67108864]:
        try:
            a = np.random.randn(N).astype(np.float32)
            b = np.random.randn(N).astype(np.float32)
            x_g = rt.upload(a)
            y_g = rt.upload(b)
            o_g = rt.upload(np.zeros(N, dtype=np.float32))
            so, _ = _build_so(vec_add_k.func, {"N": N})

            def gpu_fn():
                _launch(so, "locomp_launch_vec_add_k", (N,), (1,), [x_g, y_g, o_g])

            def np_fn():
                return a + b

            gpu_fn()
            err = float(np.max(np.abs(o_g.numpy() - np_fn())))
            status = "PASS" if err < 1e-5 else "FAIL"
            t_gpu = bench_gpu(gpu_fn)
            t_np = bench_np(np_fn)
            bw = (3 * N * 4) / (t_gpu / 1000) / 1e9
            row(f"vec_add N={N}  ({bw:.0f}GB/s)", t_gpu, t_np, err, status)
            all_results.append({"bench": "vec_add", "N": N,
                                 "status": status, "err": err,
                                 "t_gpu_ms": t_gpu, "t_np_ms": t_np, "bw_GBs": bw})
            x_g.free(); y_g.free(); o_g.free()
        except Exception:
            print(f"  vec_add N={N}  ERROR\n{traceback.format_exc()}", flush=True)
            all_results.append({"bench": "vec_add", "N": N, "status": "ERROR"})

    # ── 5. Matrix Multiply (naive) ────────────────────────────────────────────
    print(f"\n{'='*W}", flush=True)
    print("  [5] Matrix Multiply  (1 thread/element naive GEMM)", flush=True)
    print(f"{'='*W}", flush=True)

    @locomp.kernel
    def matmul_k(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                 M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
        row = locomp.program_id(0)
        col = locomp.program_id(1)
        acc = 0.0
        for k in range(K):
            acc = acc + locomp.load(A + row * K + k) * locomp.load(B + k * N + col)
        locomp.store(C + row * N + col, acc)

    for M, N, K in [(64, 64, 128), (128, 128, 256), (256, 256, 512)]:
        try:
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)
            a_g = rt.upload(A.flatten())
            b_g = rt.upload(B.flatten())
            c_g = rt.upload(np.zeros(M * N, dtype=np.float32))
            so, _ = _build_so(matmul_k.func, {"M": M, "N": N, "K": K})

            def gpu_fn():
                _launch(so, "locomp_launch_matmul_k", (M, N), (1,), [a_g, b_g, c_g])

            def np_fn():
                return A @ B

            gpu_fn()
            err = float(np.max(np.abs(c_g.numpy().reshape(M, N) - np_fn())))
            status = "PASS" if err < 1e-3 else "FAIL"
            t_gpu = bench_gpu(gpu_fn)
            t_np = bench_np(np_fn)
            row(f"matmul {M}x{K}x{N}", t_gpu, t_np, err, status)
            all_results.append({"bench": "matmul", "M": M, "N": N, "K": K,
                                 "status": status, "err": err,
                                 "t_gpu_ms": t_gpu, "t_np_ms": t_np})
            a_g.free(); b_g.free(); c_g.free()
        except Exception:
            print(f"  matmul {M}x{K}x{N}  ERROR\n{traceback.format_exc()}", flush=True)
            all_results.append({"bench": "matmul", "M": M, "N": N, "K": K, "status": "ERROR"})

    # ── 6. RoPE ───────────────────────────────────────────────────────────────
    print(f"\n{'='*W}", flush=True)
    print("  [6] RoPE  (rotary position embedding)", flush=True)
    print(f"{'='*W}", flush=True)

    @locomp.kernel
    def rope_k(X: locomp.Tensor, O: locomp.Tensor,
               SEQ: locomp.constexpr, HEADS: locomp.constexpr,
               DIM: locomp.constexpr, HALF: locomp.constexpr):
        idx = locomp.program_id(0)
        head = idx // SEQ
        seq = idx % SEQ
        for d in range(HALF):
            base = (head * SEQ + seq) * DIM + d
            x0 = locomp.load(X + base)
            x1 = locomp.load(X + base + HALF)
            theta = locomp.cast(seq, "float32") / locomp.pow(
                10000.0, locomp.cast(d * 2, "float32") / locomp.cast(DIM, "float32"))
            c = locomp.cos(theta)
            s = locomp.sin(theta)
            locomp.store(O + base, x0 * c - x1 * s)
            locomp.store(O + base + HALF, x1 * c + x0 * s)

    for B, H, S, D in [(1, 8, 128, 64), (1, 32, 256, 128), (4, 32, 128, 128)]:
        try:
            data = np.random.randn(B, H, S, D).astype(np.float32)
            HALF = D // 2
            x_g = rt.upload(data.reshape(B * H, S, D).flatten())
            o_g = rt.upload(np.zeros(B * H * S * D, dtype=np.float32))
            so, _ = _build_so(rope_k.func, {"SEQ": S, "HEADS": B * H, "DIM": D, "HALF": HALF})

            def gpu_fn():
                _launch(so, "locomp_launch_rope_k", (B * H * S,), (1,),
                        [x_g, o_g])

            def np_fn():
                angles = np.arange(S)[:, None] / (10000 ** (2 * np.arange(HALF) / D))
                cos_t = np.cos(angles)
                sin_t = np.sin(angles)
                x0 = data[..., :HALF]
                x1 = data[..., HALF:]
                out = np.empty_like(data)
                out[..., :HALF] = x0 * cos_t - x1 * sin_t
                out[..., HALF:] = x1 * cos_t + x0 * sin_t
                return out

            gpu_fn()
            err = float(np.max(np.abs(o_g.numpy().reshape(B, H, S, D) - np_fn())))
            status = "PASS" if err < 1e-4 else "FAIL"
            t_gpu = bench_gpu(gpu_fn)
            t_np = bench_np(np_fn)
            row(f"rope B={B} H={H} S={S} D={D}", t_gpu, t_np, err, status)
            all_results.append({"bench": "rope", "B": B, "H": H, "S": S, "D": D,
                                 "status": status, "err": err,
                                 "t_gpu_ms": t_gpu, "t_np_ms": t_np})
            x_g.free(); o_g.free()
        except Exception:
            print(f"  rope B={B} H={H} S={S} D={D}  ERROR\n{traceback.format_exc()}", flush=True)
            all_results.append({"bench": "rope", "status": "ERROR"})

    # ── 7. FP16 bandwidth ─────────────────────────────────────────────────────
    # Real fp16 tiled copy using new __half2/float4 vectorised load/store path.
    print(f"\n{'='*W}", flush=True)
    print("  [7] FP16 bandwidth  (half2 / float4 vectorised LDG.128 + STG.128)", flush=True)
    print(f"{'='*W}", flush=True)

    @locomp.kernel
    def fp16_scale_k(X: locomp.Float16, O: locomp.Float16, N: locomp.constexpr,
                     TILE: locomp.constexpr):
        i = locomp.program_id(0)
        row = locomp.arange(TILE)
        a = locomp.load(X + i * TILE + row)
        locomp.store(O + i * TILE + row, a)

    for TILE, N_TILES in [(8, 65536), (8, 1048576)]:
        N = N_TILES * TILE
        try:
            data = np.random.randn(N).astype(np.float16)
            x_g = rt.upload(data)
            o_g = rt.upload(np.zeros(N, dtype=np.float16))
            so, _ = _build_so(fp16_scale_k.func, {"N": N, "TILE": TILE})

            def gpu_fn():
                _launch(so, "locomp_launch_fp16_scale_k", (N_TILES,), (1,), [x_g, o_g])

            def np_fn():
                return data.copy()

            gpu_fn()
            err = float(np.max(np.abs(o_g.numpy().astype(np.float32) -
                                      data.astype(np.float32))))
            status = "PASS" if err < 1e-3 else "FAIL"
            t_gpu = bench_gpu(gpu_fn)
            t_np = bench_np(np_fn)
            bw = (2 * N * 2) / (t_gpu / 1000) / 1e9   # 2 bytes per __half
            row(f"fp16_copy N={N} tile={TILE}  ({bw:.0f}GB/s)", t_gpu, t_np, err, status)
            all_results.append({"bench": "fp16_copy", "N": N, "TILE": TILE,
                                 "status": status, "err": err,
                                 "t_gpu_ms": t_gpu, "t_np_ms": t_np, "bw_GBs": bw})
            x_g.free(); o_g.free()
        except Exception:
            print(f"  fp16_copy N={N}  ERROR\n{traceback.format_exc()}", flush=True)
            all_results.append({"bench": "fp16_copy", "status": "ERROR"})

    # ── summary ───────────────────────────────────────────────────────────────
    total = len(all_results)
    passed = sum(1 for r in all_results if r.get("status") == "PASS")
    failed = total - passed
    print(f"\n{'='*W}", flush=True)
    print(f"  Full CUDA benchmark: {passed}/{total} passed, {failed} failed", flush=True)
    print(f"{'='*W}", flush=True)
    return all_results



@app.local_entrypoint()
def main():
    # Fire all five sections in parallel
    bench_handle      = run_cuda_benchmarks.spawn()
    gpu_ag_handle     = run_gpu_ag_cuda.spawn()
    wmma_handle       = run_wmma_gemm.spawn()
    warp_handle       = run_warp_intrinsics.spawn()
    full_bench_handle = run_full_cuda_benchmark.spawn()

    data               = bench_handle.get()
    gpu_ag_results     = gpu_ag_handle.get()
    wmma_results       = wmma_handle.get()
    warp_results       = warp_handle.get()
    full_bench_results = full_bench_handle.get()

    locomp_results = data["locomp"]
    triton_results = data["triton"]

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

    # ── gpu_ag CUDA results ───────────────────────────────────────────────────
    print(flush=True)
    print("=" * W, flush=True)
    print("  gpu_ag CUDA — backward pass on A100", flush=True)
    print("=" * W, flush=True)
    ag_pass = ag_fail = 0
    for r in gpu_ag_results:
        if r["status"] == "ERROR":
            print(f"  {r['name']:<35} ERROR", flush=True)
            ag_fail += 1
        elif r["status"] == "PASS":
            print(f"  {r['name']:<35} PASS   max_err={r['max_err']:.2e}", flush=True)
            ag_pass += 1
        else:
            print(f"  {r['name']:<35} FAIL   max_err={r['max_err']:.2e}", flush=True)
            ag_fail += 1
    print("=" * W, flush=True)
    print(f"  gpu_ag: {ag_pass} passed, {ag_fail} failed", flush=True)

    # ── wmma / Tensor Core results ────────────────────────────────────────────
    print(flush=True)
    print("=" * W, flush=True)
    print("  wmma / Tensor Core — A100 sm_80", flush=True)
    print("=" * W, flush=True)
    wm_pass = wm_fail = 0
    for r in wmma_results:
        note = r.get("note", "")
        if r["status"] == "ERROR":
            print(f"  {r['name']:<35} ERROR", flush=True)
            wm_fail += 1
        elif r["status"] == "PASS":
            print(f"  {r['name']:<35} PASS   max_err={r['max_err']:.3e}  {note}", flush=True)
            wm_pass += 1
        else:
            print(f"  {r['name']:<35} FAIL   max_err={r['max_err']:.3e}  {note}", flush=True)
            wm_fail += 1
    print("=" * W, flush=True)
    print(f"  wmma: {wm_pass} passed, {wm_fail} failed", flush=True)

    # ── warp intrinsics results ───────────────────────────────────────────────
    print(flush=True)
    print("=" * W, flush=True)
    print("  warp intrinsics — __shfl_down_sync / __shfl_sync validation", flush=True)
    print("=" * W, flush=True)
    wp_pass = wp_fail = 0
    for r in warp_results:
        if r["status"] == "ERROR":
            print(f"  {r['name']:<30} ERROR", flush=True)
            wp_fail += 1
        elif r["status"] == "PASS":
            print(f"  {r['name']:<30} PASS   err={r.get('err', 0):.1e}", flush=True)
            wp_pass += 1
        else:
            print(f"  {r['name']:<30} FAIL   err={r.get('err', -1):.1e}", flush=True)
            wp_fail += 1
    print("=" * W, flush=True)
    print(f"  warp: {wp_pass} passed, {wp_fail} failed", flush=True)

    # ── full CUDA benchmark summary ───────────────────────────────────────────
    print(flush=True)
    print("=" * W, flush=True)
    print("  full CUDA benchmark — A100 vs NumPy summary", flush=True)
    print("=" * W, flush=True)
    fb_pass = fb_fail = 0
    for r in full_bench_results:
        if r["status"] == "ERROR":
            fb_fail += 1
        elif r["status"] == "PASS":
            fb_pass += 1
        else:
            fb_fail += 1
    print(f"  full_benchmark: {fb_pass} passed, {fb_fail} failed", flush=True)
    print("=" * W, flush=True)

    # ── grand total ───────────────────────────────────────────────────────────
    grand_pass = (sum(1 for r in data['locomp'] if r.get('status') == 'PASS') +
                  sum(1 for r in gpu_ag_results if r.get('status') == 'PASS') +
                  sum(1 for r in wmma_results if r.get('status') == 'PASS') +
                  wp_pass + fb_pass)
    print(flush=True)
    print(f"  GRAND TOTAL: {grand_pass} checks passed", flush=True)

