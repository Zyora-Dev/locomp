<p align="center">
  <h1 align="center">locomp</h1>
  <p align="center"><strong>A Python GPU Kernel Compiler for Apple Silicon · NVIDIA CUDA · AMD ROCm · RISC-V</strong></p>
  <p align="center">Write kernels once in Python. Compile to Metal, CUDA, HIP, or RISC-V RVV. Run anywhere.</p>
</p>

<p align="center">
  <a href="https://github.com/Zyora-Dev/locomp/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <a href="https://pypi.org/project/locomp/"><img alt="PyPI" src="https://img.shields.io/pypi/v/locomp.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
  <a href="https://github.com/Zyora-Dev/locomp/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/Zyora-Dev/locomp/test.yml?label=CI"></a>
</p>

---

**locomp** is an open-source GPU kernel compiler. Write a GPU kernel as a plain Python function, and locomp compiles it through a SSA intermediate representation into native code for your target hardware — Metal on Apple Silicon, CUDA C on NVIDIA, HIP C on AMD ROCm, or C + RISC-V Vector (RVV) intrinsics on RISC-V.

Think **Triton, but hardware-agnostic** — one `@locomp.kernel` runs on M1, A100, MI300X, and RISC-V without rewriting a line. Triton targets NVIDIA only. locomp targets all four.

```python
import locomp
import numpy as np

@locomp.kernel
def vector_add(X: locomp.Tensor, Y: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    locomp.store(O + i, locomp.load(X + i) + locomp.load(Y + i))

# Apple Silicon (Metal)
x = locomp.tensor(np.ones(4, dtype=np.float32))
y = locomp.tensor(np.ones(4, dtype=np.float32) * 2)
o = locomp.empty(4)
vector_add[(4,)](x, y, o, N=4)
print(o.numpy())  # [3. 3. 3. 3.]

# NVIDIA GPU (CUDA)
x = locomp.tensor(np.ones(4, dtype=np.float32), backend="cuda")
y = locomp.tensor(np.ones(4, dtype=np.float32) * 2, backend="cuda")
o = locomp.empty(4, backend="cuda")
vector_add[(4,)](x, y, o, N=4)
print(o.numpy())  # [3. 3. 3. 3.]
```

## Install

```bash
pip install locomp
```

**PyPI**: https://pypi.org/project/locomp/1.0.0/

- **Apple Silicon**: macOS, M1/M2/M3/M4, Python 3.10+
- **NVIDIA GPU**: Linux or Windows, CUDA 11.0+, Python 3.10+
- **AMD GPU**: Linux, ROCm 5.0+, `hipcc` (tested on MI300X gfx942)
- **RISC-V**: Linux (`gcc-riscv64-linux-gnu` + `qemu-user-static` for emulation)

## How It Works

```
@locomp.kernel (Python function)
        │
        ▼
   Python AST → Locomp IR (SSA, 60+ opcodes)
        │
        ▼
   Optimizer (CSE · DCE · constant folding · type inference · strength reduction)
        │
        ├──── backend="metal" ──→ Metal Shading Language → Apple GPU (M1/M2/M3/M4)
        ├──── backend="cuda"  ──→ CUDA C (nvcc -O3)     → NVIDIA GPU (sm_86, sm_90...)
        ├──── backend="rocm"  ──→ HIP C (hipcc -O3)     → AMD GPU (gfx90a, gfx942...)
        └──── backend="riscv" ──→ C + RVV intrinsics    → RISC-V CPU (rv64gcv)
```

Your Python function is **compiled, not interpreted**. The compiled pipeline is cached per constexpr configuration — repeated calls have near-zero overhead.


## Use Cases

### 1. Standalone Kernel

Write a custom GPU kernel for a single operation — quantization, custom attention, a new activation function, anything NumPy or PyTorch doesn't expose:

```python
import locomp
import numpy as np

@locomp.kernel
def rms_norm(X: locomp.Tensor, W: locomp.Tensor, O: locomp.Tensor,
             N: locomp.constexpr, EPS: locomp.constexpr):
    row = locomp.program_id(0)
    tid = locomp.local_id(0)
    smem = locomp.shared_memory(32)

    local_sum = 0.0
    for i in range(tid, N, 128):
        val = locomp.load(X + row * N + i)
        local_sum = local_sum + val * val
    local_sum = locomp.simd_sum(local_sum)

    if locomp.simd_lane_id() == 0:
        locomp.shared_store(smem, locomp.simd_group_id(), local_sum)
    locomp.barrier()

    if tid == 0:
        total = 0.0
        for g in range(4):
            total = total + locomp.shared_load(smem, g)
        locomp.shared_store(smem, 0, locomp.rsqrt(total / N + EPS))
    locomp.barrier()

    rms = locomp.shared_load(smem, 0)
    for i in range(tid, N, 128):
        val = locomp.load(X + row * N + i)
        locomp.store(O + row * N + i, val * rms * locomp.load(W + i))

# Dispatch: 1 threadgroup per row, 128 threads per group
rows, dim = 128, 4096
x = locomp.tensor(np.random.randn(rows, dim).astype(np.float32))
w = locomp.tensor(np.ones(dim, dtype=np.float32))
o = locomp.empty((rows, dim))
rms_norm[(rows,), (128,)](x, w, o, N=dim, EPS=1e-5)
```

**M1**: 1.2× faster than MLX · **A100**: 4.1× faster than PyTorch.

---

### 2. As the Kernel Layer in a Serving Engine

locomp is designed to drop in as the kernel compilation layer under a GPU inference server. Instead of shipping pre-compiled Metal shaders or CUDA PTX, your server compiles kernels at first request and caches them permanently.

```
Inference Request
      │
      ▼
  Server Engine  (routing · batching · scheduling)
      │
      ▼
  locomp Kernel Layer  ← @locomp.kernel Python functions
      │                   compiled + cached per (kernel, config, GPU model)
      ├── Metal dispatch  → Apple Silicon nodes (M1/M2/M3/M4)
      ├── CUDA dispatch   → NVIDIA GPU nodes  (A10G / A100 / H100)
      ├── ROCm dispatch   → AMD GPU nodes     (MI300X / MI250X)
      └── RISC-V dispatch → RISC-V vector CPU nodes (rv64gcv)
```

**Key properties for serving engines:**

- **Specialization per config** — `constexpr` params (batch size, seq len, head dim) become hardware literals. The compiler generates a separate optimized pipeline per shape — no dynamic dispatch overhead.
- **Persistent pipeline cache** — compiled kernels written to `~/.cache/locomp/` — server restarts are instant after first run.
- **Async dispatch + batch mode** — multiple kernel calls in one command buffer; GPU pipelines work while CPU prepares next batch.
- **Backend-agnostic** — same kernel code, auto-selects Metal/CUDA/ROCm/RISC-V based on available hardware.

```python
import locomp

@locomp.kernel
def fused_rope(Q: locomp.Tensor, Cos: locomp.Tensor, Sin: locomp.Tensor,
               N: locomp.constexpr, D: locomp.constexpr):
    bh = locomp.program_id(0)
    t  = locomp.program_id(1)
    for d in range(0, D // 2):
        qi = locomp.load(Q + bh * N * D + t * D + d)
        qj = locomp.load(Q + bh * N * D + t * D + d + D // 2)
        c  = locomp.load(Cos + t * (D // 2) + d)
        s  = locomp.load(Sin + t * (D // 2) + d)
        locomp.store(Q + bh * N * D + t * D + d,         qi * c - qj * s)
        locomp.store(Q + bh * N * D + t * D + d + D // 2, qi * s + qj * c)

class InferenceServer:
    def __init__(self, backend: str = "auto"):
        self.backend = backend  # "metal" | "cuda" | "riscv" | "auto"

    def run_rope(self, q_np, cos_np, sin_np, heads, seq, dim):
        q   = locomp.tensor(q_np,   backend=self.backend)
        cos = locomp.tensor(cos_np, backend=self.backend)
        sin = locomp.tensor(sin_np, backend=self.backend)
        # compiled once, cached forever — near-zero overhead on all subsequent calls
        fused_rope[(heads, seq)](q, cos, sin, N=seq, D=dim)
        return q.numpy()
```

---

### 3. Quick Kernel Examples

#### GELU (2.6× faster than MLX on M1, 85× on A100)

```python
@locomp.kernel
def gelu(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    x = locomp.load(X + i)
    inner = locomp.clamp(0.7978845608 * (x + 0.044715 * x * x * x), -10.0, 10.0)
    locomp.store(O + i, 0.5 * x * (1.0 + locomp.tanh(inner)))
```

#### INT4 Quantized MatVec (17× faster than NumPy)

```python
@locomp.kernel
def dequant_matvec(W_packed: locomp.Tensor, X: locomp.Tensor, Scales: locomp.Tensor,
                   O: locomp.Tensor, K: locomp.constexpr, GROUP: locomp.constexpr):
    row = locomp.program_id(0)
    acc = 0.0
    for k in range(K // 2):
        packed = locomp.cast(locomp.load(W_packed + row * (K // 2) + k), locomp.UInt8)
        w0 = locomp.cast(packed & 0xF, locomp.Float32) - 8.0
        w1 = locomp.cast((packed >> 4) & 0xF, locomp.Float32) - 8.0
        scale = locomp.load(Scales + row * (K // GROUP) + (2 * k) // GROUP)
        acc = acc + (w0 * scale) * locomp.load(X + 2 * k)
        acc = acc + (w1 * scale) * locomp.load(X + 2 * k + 1)
    locomp.store(O + row, acc)
```

#### Auto-Tuning

```python
from locomp import autotune, Config

@autotune(
    configs=[
        Config(BLOCK=16, num_warps=4),
        Config(BLOCK=32, num_warps=8),
        Config(BLOCK=64, num_warps=16),
    ],
    key=["N", "D"],
)
@locomp.kernel
def softmax(X: locomp.Tensor, O: locomp.Tensor,
            N: locomp.constexpr, D: locomp.constexpr, BLOCK: locomp.constexpr):
    ...
# Benchmarks all configs on first call · caches winner to ~/.cache/locomp/autotune.json
```

---

## Benchmarks

### Apple Silicon (M1) vs MLX 0.31

float32, median of 10 runs after 3 warmup. Ratio < 1.0 = locomp wins.

| Kernel | locomp | MLX | Ratio | Speedup |
|--------|--------|-----|-------|---------|
| Flash Attention GPU kernel (N=64) | 0.032ms | 0.425ms | **0.08×** | **12×** |
| RoPE | — | — | **0.34×** | **2.9×** |
| GELU+bias | — | — | **0.38×** | **2.6×** |
| Reduce sum [32×1024] | 0.258ms | 0.379ms | **0.68×** | **1.5×** |
| Batch norm (N=4, C=128) | 0.246ms | 0.356ms | **0.69×** | **1.4×** |
| SwiGLU | — | — | **0.74×** | **1.4×** |
| LayerNorm | — | — | **0.77×** | **1.3×** |
| Multi-head attention [4×8×256] | 1.705ms | 1.784ms | **0.96×** | 1.04× |
| RMSNorm | — | — | **0.85×** | **1.2×** |
| Simdgroup matmul 1024² | 4.081ms | 4.232ms | **0.96×** | 1.04× |
| Online softmax [256×512] | 0.300ms | 0.305ms | **0.98×** | 1.02× |

**SmolLM2-135M** (30 layers, GQA, RoPE, INT4) — **7.9 tok/s decode** on M1, all kernels pure `@locomp.kernel`.

---

### NVIDIA A100 80GB — Memory Bandwidth

CUDA 12.1, 1B elements, 20 warm runs.

| Kernel | Bandwidth | Notes |
|--------|-----------|-------|
| vector_add (1B f32) | **1,697 GB/s** | Within measurement noise of Triton |
| scale_shift (1B f32) | **1,686 GB/s** | Triton parity |
| relu (1B f32) | **1,685 GB/s** | Triton parity |
| sqrt_exp (16M f32) | 1,453 GB/s | — |
| smem_copy (16M f32) | 1,483 GB/s | — |
| wmma matmul (1024³ fp16) | **13.79 TFLOPS** | CUDA Tensor Cores (`wmma::mma_sync`) |

Memory-bandwidth kernels at **~1,690 GB/s — within measurement error of Triton on A100.**

### NVIDIA A100 — vs PyTorch (large models)

| Kernel | vs PyTorch | Notes |
|--------|-----------|-------|
| Softmax (B=256, D=1024) | **4.7×** faster | — |
| RMSNorm (B=128, D=4096) | **4.1×** faster | — |
| GELU (N=4M) | **85×** faster | Fused, no temp alloc |
| RoPE (N=1M) | **21×** faster | In-place |

---

### RISC-V RVV (rv64gcv, qemu-riscv64-static)

Cross-compiled with `riscv64-linux-gnu-gcc -march=rv64gcv -O2`, executed under QEMU. All match NumPy.

| Kernel | Time | Max Error |
|--------|------|-----------|
| vector_add | 0.59s | 0.00 |
| tiled_add_rvv | 0.51s | 0.00 |
| scale_shift | 0.52s | 0.00 |
| reduce_sum_rvv | 0.46s | 0.00 |
| relu | 0.48s | 0.00 |
| dot_product | 0.49s | 0.00 |
| sqrt_exp | 0.94s | < 1e-6 |
| rms_norm | 0.54s | < 1e-6 |

*(Includes compile + QEMU execution + validation. Real RISC-V hardware will be orders of magnitude faster.)*

---

### AMD MI300X (gfx942) — Memory Bandwidth

HIP C compiled with `hipcc --offload-arch=gfx942 -O3`, measured on real hardware (192 GB HBM3, 5,300 GB/s theoretical peak).

| Kernel | Bandwidth | % of Peak | Config |
|--------|-----------|-----------|--------|
| vector_add | **3,961 GB/s** | **74.7%** | 256M f32, 3 buf |
| scale_shift | **3,440 GB/s** | **64.9%** | 256M f32, 2 buf |
| relu | **3,587 GB/s** | **67.7%** | 256M f32, 2 buf |
| gelu | **3,260 GB/s** | **61.5%** | N=4M (transcendental) |
| rmsnorm | 1,119 GB/s | 21.1% | B=128, D=4096 |
| rope | 739 GB/s | 13.9% | 512×8×64 |
| softmax | 600 GB/s | 11.3% | B=256, D=1024 |
| matvec | 60.9 GB/s | 1.1%† | 4096×4096 f32 |

†Matvec is compute-bound — low bandwidth utilization is expected.

Wavefront intrinsics validated: 64-lane `__shfl_down`, `warp_sum`=2016.0, `warp_max`=63.0 ✓

**locomp is the first and only Python kernel compiler targeting AMD ROCm hardware.** Bandwidth-bound kernels reach 62–75% of HBM3 peak — comparable to Triton's efficiency on A100.

---

## Test Results

| Platform | Tests | Status |
|----------|-------|--------|
| Apple M1 (local) | 227 passed, 22 skipped | ✅ |
| Apple M4 (GitHub Actions macOS-15) | 227 passed, 22 skipped | ✅ |
| NVIDIA A100 (Modal) | 64/64 execution checks | ✅ |
| NVIDIA A10G (Modal) | 17/17 CUDA codegen + runtime | ✅ |
| RISC-V RVV (QEMU, Ubuntu 24.04) | 9/9 execution tests | ✅ |

| Test File | Tests | What Runs |
|-----------|-------|-----------|
| `test_gpu_autograd.py` | 50 | Metal GPU backward passes (M1/M4) |
| `test_cuda_codegen.py` | 17 | CUDA source string checks |
| `test_cuda_runtime.py` | 31 | CUDA runtime (13 skip on macOS) |
| `test_autograd.py` | 34 | CPU NumPy autograd |
| `test_riscv_codegen.py` | 19 | RISC-V codegen string checks |
| `test_riscv_execution.py` | 9 | QEMU execution (skip if no toolchain) |
| `test_autotune.py` | 13 | Config search + disk cache |
| `test_control_flow.py` | 20 | IR + Metal execution |
| `test_hardening.py` | 15 | Edge cases + Metal execution |
| `test_reductions.py` | 10 | Reduce kernels |
| other | ~9 | IR / frontend / optimizer |

```bash
pip install -e ".[dev]"
pytest tests/ -q                             # full suite
pytest tests/ -k "not macos_only"            # no GPU required
pytest tests/test_riscv_execution.py -v      # RISC-V (needs toolchain)
```

---

## Kernel Language Reference

### Dispatch

```python
kernel[(N,)](args...)                          # 1D grid, 1 thread/group
kernel[(N,), (T,)](args...)                    # 1D grid, T threads/group
kernel[(gx, gy)](args...)                      # 2D grid
kernel[(gx, gy, gz), (tx, ty)](args...)        # 3D grid, 2D threadgroup
```

### Built-in Functions

| Category | Functions |
|----------|-----------|
| **Indexing** | `program_id(axis)`, `local_id(axis)`, `arange(start, end)` |
| **Memory** | `load(ptr)`, `store(ptr, val)`, `load(ptr, mask=m)`, `store(ptr, val, mask=m)` |
| **Shared Memory** | `shared_memory(size)`, `shared_load(smem, idx)`, `shared_store(smem, idx, val)` |
| **Sync** | `barrier()` |
| **Math** | `exp` `log` `sqrt` `rsqrt` `abs` `tanh` `sin` `cos` `sigmoid` `asin` `acos` `atan` `atan2` `sinh` `cosh` `exp2` `log2` `log10` `ceil` `floor` `round` `fma` `pow` `clamp` `copysign` `fmod` `step` |
| **Select** | `max(a, b)`, `min(a, b)`, `where(cond, a, b)` |
| **SIMD** | `simd_sum` `simd_max` `simd_min` `simd_broadcast` `simd_shuffle_down` `simd_lane_id()` `simd_group_id()` |
| **Matrix** | `simdgroup_matrix_load` `simdgroup_matrix_store` `simdgroup_mac` `simdgroup_matrix_fill` |
| **Atomics** | `atomic_add(ptr, val)`, `atomic_max(ptr, val)`, `atomic_min(ptr, val)` |
| **Cast** | `cast(val, dtype)` |
| **Control Flow** | `for ... in range()`, `while`, `if / elif / else`, `break`, `continue` |

### Types

| Type | Metal | CUDA | ROCm | RISC-V |
|------|-------|------|------|--------|
| `locomp.Tensor` | `device float*` | `float* __restrict__` | `float*` |
| `locomp.constexpr` | MSL literal | compile-time const | C macro |
| `locomp.Float16` | `half` | `__half` | `_Float16` |
| `locomp.Int8` / `UInt8` | `char` / `uchar` | `int8_t` / `uint8_t` | same |
| `locomp.Int32` | `int` | `int32_t` | same |
| `locomp.Bool` | `bool` | `bool` | `int` |

### Tensor API

```python
t = locomp.tensor(np_array)                   # numpy → GPU
t = locomp.tensor(np_array, backend="cuda")   # to specific backend
t = locomp.empty(shape)                        # uninitialized
t = locomp.zeros(shape)                        # zero-filled
t.numpy()                                      # GPU → numpy (auto-syncs)
t.reshape(new_shape)                           # zero-copy reshape
t.transpose(d0, d1)                            # swap dimensions
t.permute(*dims)                               # arbitrary reorder
t[slices]                                      # numpy-style slicing
```

---

## 63 Kernel Examples

| Range | Category | Kernels |
|-------|----------|---------|
| 01–06 | **Core** | vector_add, dot_product, threaded add, matmul, softmax, float16 |
| 07–13 | **Optimized Matmul** | tiled smem, parallel softmax, SIMD, micro-tiling |
| 14–17 | **Flash Attention** | v1/v2/v3 — online softmax, P-in-shared, constexpr inlined |
| 18–22 | **Simdgroup Matrix** | 1/4/8/16 SIMD groups, simdgroup flash attention |
| 23–24 | **Math + MHA** | 23 math ops, batched MHA [B,H,N,D] |
| 25–28 | **Autotune + Float16** | autotune, float16 simdgroup, conv2d |
| 29–34 | **Transformer Fused** | RMSNorm, LayerNorm, SwiGLU, GELU+bias, RoPE, residual+norm |
| 37–53 | **Production** | Causal attn D=64/128, INT4/INT8, KV-cache, scatter/gather, transpose, pooling, batch_norm, broadcast, concat, cross-attn, dequantize |
| **54** | **LLM Inference** | SmolLM2-135M: 30 layers, GQA, RoPE, INT4, 7.9 tok/s |
| 55–63 | **CUDA** | CUDA runtime, warp intrinsics, wmma Tensor Core, A100 benchmark |

```bash
python examples/01_vector_add.py
python examples/54_smollm2_inference.py       # needs safetensors + huggingface_hub
python examples/59_cuda_benchmark_suite.py    # NVIDIA only
```

---

## SmolLM2-135M End-to-End Inference

A 135M-parameter LLM running entirely on `@locomp.kernel` — no PyTorch, no MLX, no Metal C++:

```
$ python examples/54_smollm2_inference.py

SmolLM2-135M — locomp GPU inference
Loading weights... 272 tensors, 538MB

Prompt: "The meaning of life is"
Output: "to be found in the meaning of the universe."
Decode:  7.9 tok/s

Prompt: "Once upon a time"
Output: ", there was a little girl named Lily..."
Decode:  7.6 tok/s

Prompt: "Python is a programming language that"
Output: "allows you to write programs in a structured way..."
Decode:  7.1 tok/s
```

10 pure `@locomp.kernel` Python functions — the complete inference loop:
`rms_norm` · `matvec` · `silu_mul` · `rope` · `gqa_attn` · `kv_cache_update` · `add_inplace` · `add` · `copy` · `embed`

---

## Autograd

### CPU Autograd (`locomp.ag`)

Tape-based reverse-mode autodiff on NumPy. 15 ops, 34 tests.

```python
a = locomp.ag.tensor(np.random.randn(N), requires_grad=True)
b = locomp.ag.tensor(np.random.randn(N), requires_grad=True)
loss = locomp.ag.sum(locomp.ag.mul(a, b))
locomp.ag.backward(loss)
print(a.grad)  # dL/da = b
```

### GPU Autograd (`locomp.gpu_ag`)

Forward and backward passes execute as real locomp kernels. Gradients stay on-device until `.numpy()`.

```python
ga = locomp.gpu_ag
x  = ga.tensor(np.random.randn(N), requires_grad=True)
y  = ga.relu(ga.exp(x))
ga.backward(ga.sum(y))
print(x.grad.numpy())   # GPU gradient, read back to CPU
```

Validated on Apple M1 (Metal) and NVIDIA A100 (CUDA). 14 ops, 50 tests.

---

## Compiler Architecture

| Stage | File | What It Does |
|-------|------|-------------|
| Frontend | `locomp/frontend.py` | Python AST → Locomp IR (SSA) |
| IR | `locomp/ir.py` | 60+ opcodes, all dtypes, SSA values |
| Optimizer | `locomp/optimizer.py` | CSE, DCE, constant fold, type infer, strength reduce |
| Metal Codegen | `locomp/backends/metal_codegen.py` | IR → Metal Shading Language |
| Metal Runtime | `locomp/backends/metal_runtime.py` | Metal GPU dispatch + buffer management |
| CUDA Codegen | `locomp/backends/cuda_codegen.py` | IR → CUDA C (float4 LDG.128, wmma, warp) |
| CUDA Runtime | `locomp/backends/cuda_runtime.py` | cudaMalloc/Free/Memcpy via ctypes |
| RISC-V Codegen | `locomp/backends/riscv_codegen.py` | IR → C + RVV intrinsics |
| API | `locomp/api.py` | Public API, tensor, kernel launcher, pipeline cache |
| Autotune | `locomp/autotune.py` | Config search + persistent disk cache |
| C Bridge | `locomp/_native/fast_dispatch.m` | Native async dispatch, bypasses PyObjC |

**Optimization passes**: Constexpr Inlining (2× flash attn speedup) · CSE · DCE · Constant Folding · Strength Reduction · Type Inference

**Dispatch optimizations** (Metal): Native C Bridge · Async Dispatch · Batch Mode · Per-specialization pipeline cache · Lazy GPU sync

---

## Competitive Landscape

| | **locomp** | MLX | Triton | PyTorch MPS | Raw Metal/CUDA |
|---|---|---|---|---|---|
| Apple Silicon | ✅ | ✅ | ❌ | ✅ | ✅ |
| NVIDIA CUDA | ✅ | ❌ | ✅ | ✅ | ✅ |
| RISC-V RVV | ✅ | ❌ | ❌ | ❌ | ❌ |
| Kernel language | **Python** | Metal C++ | Python (NVIDIA) | N/A | C++ |
| Full compiler | ✅ | JIT string | ✅ (CUDA) | N/A | Xcode/nvcc |
| Auto-tuning | ✅ | ❌ | ✅ | ❌ | ❌ |
| Autograd | ✅ CPU+GPU | ✅ | ❌ | ✅ | ❌ |
| INT4/INT8 quant | ✅ | ❌ | ❌ | ❌ | Manual |
| LLM inference | ✅ 7.9 tok/s | ✅ | ❌ | ❌ | Manual |

**locomp is the only Python kernel compiler that targets Apple Silicon, NVIDIA CUDA, and RISC-V.**

---

## Development

```bash
git clone https://github.com/Zyora-Dev/locomp.git
cd locomp
pip install -e ".[dev]"

pytest tests/ -q                              # full test suite
pytest tests/ -k "not macos_only"             # no GPU required
python examples/04_matmul.py                  # single kernel
python examples/54_smollm2_inference.py       # LLM inference (Apple Silicon)
python examples/59_cuda_benchmark_suite.py    # CUDA benchmarks (NVIDIA)
```

### Project Structure

```
locomp/
├── locomp/
│   ├── frontend.py              # Python AST → IR
│   ├── ir.py                    # SSA IR (60+ opcodes)
│   ├── optimizer.py             # CSE, DCE, constant folding
│   ├── api.py                   # Public API
│   ├── autotune.py              # Auto-tuning + disk cache
│   ├── autograd.py              # CPU tape-based autograd
│   ├── gpu_autograd.py          # GPU autograd (Metal + CUDA)
│   ├── _native/
│   │   └── fast_dispatch.m      # Native C bridge (macOS)
│   └── backends/
│       ├── metal_codegen.py     # IR → Metal Shading Language
│       ├── metal_runtime.py     # Metal GPU dispatch
│       ├── cuda_codegen.py      # IR → CUDA C
│       ├── cuda_runtime.py      # CUDA runtime via ctypes
│       └── riscv_codegen.py     # IR → C + RVV intrinsics
├── examples/                    # 63 kernel examples
├── benchmarks/                  # Benchmark scripts
└── tests/                       # 227+ unit tests
```

---

## License

Apache-2.0 — [LICENSE](LICENSE)

## Citation

```bibtex
@software{locomp,
  title   = {locomp: A Python GPU Kernel Compiler for Apple Silicon, NVIDIA CUDA, and RISC-V},
  url     = {https://github.com/Zyora-Dev/locomp},
  year    = {2026},
  version = {1.0.0}
}
```
