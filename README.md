<p align="center">
  <h1 align="center">locomp</h1>
  <p align="center"><strong>The GPU Compute Compiler for Apple Silicon</strong></p>
  <p align="center">Write kernels in Python. Compile to Metal. Run on M1/M2/M3/M4.</p>
</p>

<p align="center">
  <a href="https://github.com/Zyora-Dev/locomp/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <a href="https://pypi.org/project/locomp/"><img alt="PyPI" src="https://img.shields.io/pypi/v/locomp.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
  <a href="https://github.com/Zyora-Dev/locomp"><img alt="Apple Silicon" src="https://img.shields.io/badge/Apple%20Silicon-M1%20M2%20M3%20M4-black.svg"></a>
</p>

---

**locomp** is a Python-to-Metal kernel compiler for Apple Silicon. Write GPU kernels as decorated Python functions — locomp compiles them through an SSA intermediate representation to native Metal Shading Language, optimizes them, and dispatches on your Apple GPU.

Think **Triton, but for Apple Silicon**. No Metal C++. No Xcode. No CUDA. Just `@locomp.kernel`.

```python
import locomp
import numpy as np

@locomp.kernel
def vector_add(X: locomp.Tensor, Y: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    locomp.store(O + i, locomp.load(X + i) + locomp.load(Y + i))

x = locomp.tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
y = locomp.tensor(np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
o = locomp.empty(4)

vector_add[(4,)](x, y, o, N=4)
print(o.numpy())  # [6. 8. 10. 12.]
```

## Install

```bash
pip install locomp
```

Requirements: macOS with Apple Silicon (M1/M2/M3/M4), Python 3.10+.

## How It Works

```
@locomp.kernel (Python)
        │
        ▼
   Python AST → Locomp IR (SSA, 60+ opcodes)
        │
        ▼
   Optimization passes (CSE, DCE, constant folding, type inference)
        │
        ▼
   Metal Shading Language (MSL) codegen
        │
        ▼
   Apple Metal GPU dispatch (native C bridge, async, batch mode)
```

Your Python function is **not interpreted** — it's compiled to a Metal shader pipeline that runs natively on the GPU. The compiled pipeline is cached per constexpr configuration, so subsequent calls have near-zero overhead.

## Quick Start

### 1. Element-wise Kernel

```python
import locomp
import numpy as np

@locomp.kernel
def relu(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    val = locomp.load(X + i)
    out = val * (val > 0.0)
    locomp.store(O + i, out)

x = locomp.tensor(np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32))
o = locomp.empty(4)
relu[(4,)](x, o, N=4)
print(o.numpy())  # [0. 2. 0. 4.]
```

### 2. Matrix Multiply with Threadgroups

```python
@locomp.kernel
def matmul(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
           M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
    idx = locomp.program_id(0)
    m = idx // N
    n = idx - m * N
    acc = 0.0
    for k in range(K):
        acc = acc + locomp.load(A + (m * K + k)) * locomp.load(B + (k * N + n))
    locomp.store(C + (m * N + n), acc)

M, N, K = 64, 64, 64
A = locomp.tensor(np.random.randn(M, K).astype(np.float32))
B = locomp.tensor(np.random.randn(K, N).astype(np.float32))
C = locomp.empty((M, N))
matmul[(M * N,)](A, B, C, M=M, N=N, K=K)
```

### 3. SIMD Reductions + Shared Memory

```python
@locomp.kernel
def rms_norm(X: locomp.Tensor, W: locomp.Tensor, O: locomp.Tensor,
             N: locomp.constexpr, EPS: locomp.constexpr):
    row = locomp.program_id(0)
    tid = locomp.local_id(0)

    smem = locomp.shared_memory(32)
    local_sum = 0.0
    for i in range(tid, N, 128):
        val = locomp.load(X + (row * N + i))
        local_sum = local_sum + val * val

    local_sum = locomp.simd_sum(local_sum)
    if locomp.simd_lane_id() == 0:
        locomp.shared_store(smem, locomp.simd_group_id(), local_sum)
    locomp.barrier()

    if tid == 0:
        total = 0.0
        for g in range(4):
            total = total + locomp.shared_load(smem, g)
        rms = locomp.rsqrt(total / N + EPS)
        locomp.shared_store(smem, 0, rms)
    locomp.barrier()

    rms = locomp.shared_load(smem, 0)
    for i in range(tid, N, 128):
        val = locomp.load(X + (row * N + i))
        w = locomp.load(W + i)
        locomp.store(O + (row * N + i), val * rms * w)

# Dispatch: 1 threadgroup per row, 128 threads per group
rms_norm[(rows,), (128,)](x, weights, out, N=dim, EPS=1e-5)
```

### 4. Auto-Tuning

```python
from locomp import autotune, Config

@autotune(
    configs=[
        Config(BLOCK_M=16, BLOCK_N=16, num_warps=4),
        Config(BLOCK_M=32, BLOCK_N=32, num_warps=8),
        Config(BLOCK_M=64, BLOCK_N=64, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@locomp.kernel
def tuned_matmul(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                 M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr,
                 BLOCK_M: locomp.constexpr, BLOCK_N: locomp.constexpr):
    # ... kernel body
    pass
```

Auto-tuning benchmarks each config per GPU, caches the winner to disk.

## Kernel Language Reference

### Dispatch Model

```python
# 1D grid: N threadgroups, 1 thread each
kernel[(N,)](args...)

# 1D grid with threadgroup parallelism: N groups, T threads each
kernel[(N,), (T,)](args...)
```

### Built-in Functions

| Category | Functions |
|----------|-----------|
| **Indexing** | `program_id(axis)`, `local_id(axis)`, `arange(start, end)` |
| **Memory** | `load(ptr)`, `store(ptr, val)`, `load(ptr, mask=m)`, `store(ptr, val, mask=m)` |
| **Shared Memory** | `shared_memory(size)`, `shared_load(smem, idx)`, `shared_store(smem, idx, val)` |
| **Sync** | `barrier()` |
| **Math** | `exp`, `log`, `sqrt`, `rsqrt`, `abs`, `tanh`, `sin`, `cos`, `sigmoid` |
| | `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `exp2`, `log2`, `log10` |
| | `ceil`, `floor`, `round`, `fma`, `pow`, `clamp`, `copysign`, `fmod`, `step` |
| **Comparison** | `max(a, b)`, `min(a, b)`, `where(cond, a, b)` |
| **SIMD** | `simd_sum`, `simd_max`, `simd_min`, `simd_broadcast`, `simd_shuffle_down` |
| | `simd_lane_id()`, `simd_group_id()` |
| **Simdgroup Matrix** | `simdgroup_matrix_load`, `simdgroup_matrix_store` |
| | `simdgroup_matrix_mac`, `simdgroup_matrix_fill` |
| **Atomics** | `atomic_add(ptr, val)`, `atomic_max(ptr, val)`, `atomic_min(ptr, val)` |
| **Cast** | `cast(val, dtype)` |
| **Control Flow** | `for ... in range()`, `while`, `if/else`, `break` |

### Types

| Type | Description |
|------|-------------|
| `locomp.Tensor` | GPU buffer pointer (kernel parameter) |
| `locomp.constexpr` | Compile-time constant (inlined as MSL literal) |
| `locomp.Float16` | 16-bit float (`half` in Metal) |
| `locomp.Int8` / `locomp.UInt8` | 8-bit integer |
| `locomp.Int32` | 32-bit integer |
| `locomp.Bool` | Boolean |

### Tensor API

```python
t = locomp.tensor(numpy_array)    # Create GPU tensor from numpy
t = locomp.empty(shape)           # Allocate uninitialized GPU tensor
t = locomp.zeros(shape)           # Zero-filled GPU tensor
t = locomp.ones(shape)            # Ones-filled GPU tensor

t.numpy()                         # Read back to CPU (auto-syncs)
t.reshape(new_shape)              # Zero-copy reshape
t.transpose(dim0, dim1)           # Transpose dimensions
t.permute(*dims)                  # Arbitrary dimension reorder
t.contiguous()                    # Materialize contiguous copy
t[slices]                         # NumPy-style slicing
```

## 54 Kernel Examples

Full working examples in [`examples/`](examples/):

| # | Kernel | Description |
|---|--------|-------------|
| 01 | Vector Add | Tiled element-wise: arange, mask, load/store |
| 02 | Dot Product | Tiled reduction per threadgroup |
| 03 | Threaded Vector Add | 256 threads/group, `local_id()` |
| 04 | Matrix Multiply | For-loop accumulation, `program_id` |
| 05 | Softmax | 3-pass: max, exp_sum, normalize |
| 06 | Float16 Vector Add | End-to-end `half` type |
| 07 | Tiled Matmul | 16×16 shared memory tiles |
| 08 | Parallel Softmax | Tree-reduction, 256 threads/row |
| 09 | SIMD Softmax | Warp-level SIMD reductions |
| 10 | SIMD Hybrid Softmax | Cross-SIMD-group reductions |
| 11-13 | Optimized Matmul | 4×4 register micro-tiling, multi-level |
| 14 | Online Softmax | 2-pass online (beats MLX 4/5 sizes) |
| 15-17 | Flash Attention v1/v2/v3 | Tiled fused QKV, online softmax, P-in-shared |
| 18-21 | Simdgroup Matmul | Hardware 8×8 AMX: 1/4/8/16 SIMD groups |
| 22 | Simdgroup Flash Attn | Hardware QK + PV matmul |
| 23 | Math Ops | 23 math functions + GELU + LayerNorm |
| 24 | Multi-Head Attention | [B,H,N,D] batched (beats MLX) |
| 25 | Auto-Tune | `locomp.autotune` + disk cache |
| 26-27 | Float16 Simdgroup | Float16 matmul + float16 flash attention |
| 28 | Conv2D | 2D convolution |
| 29-34 | Fused Transformer Ops | RMSNorm, LayerNorm, SwiGLU, GELU+bias, RoPE, residual+norm |
| 37 | Causal Flash Attention | D=64/128, MHA/GQA/MQA |
| 38 | Quantized Matmul | INT4/INT8 with on-the-fly dequantization |
| 39 | Batched Matmul | Batch dimension support |
| 40 | Embedding | Token embedding lookup |
| 41 | Top-K Sampling | GPU top-k for decoding |
| 42 | Fused QKV | Fused query/key/value projection |
| 43 | Paged Attention | Paged KV cache attention |
| 44-53 | Infrastructure | Transpose, reduce, KV cache, scatter/gather, concat/split, batch norm, pooling, cross attention, dequantize, broadcast |
| **54** | **SmolLM2-135M Inference** | **Full LLM running end-to-end on locomp** |

Run any example:

```bash
python examples/01_vector_add.py
python examples/29_rms_norm.py
python examples/54_smollm2_inference.py
```

## End-to-End LLM Inference

locomp can run a real language model using only `@locomp.kernel` Python — no PyTorch, no MLX, no Metal C++:

```
$ python examples/54_smollm2_inference.py

SmolLM2-135M — locomp GPU inference
Loading weights... 272 tensors, 538MB
Uploading to GPU... 538MB uploaded

Prompt: "The meaning of life is"
  Generating:  to be found in the meaning of the universe.

Prompt: "Once upon a time"
  Generating: , there was a little girl named Lily...

Prompt: "Python is a programming language that"
  Generating:  allows you to write programs in a structured way...
```

10 GPU kernels — all pure Python `@locomp.kernel`:

`rms_norm` · `matvec` · `silu_mul` · `rope` · `gqa_attn` · `kv_cache_update` · `add_inplace` · `add` · `copy` · `embed`

### Validated Hardware

| Chip | Status | Method |
|------|--------|--------|
| Apple M1 | **Passed** | Local bare metal |
| Apple M4 | **Passed** | GitHub Actions CI (macOS 15) |

All 55 tests, 5 example kernels, and SmolLM2-135M inference pass on both M1 and M4.

## Benchmarks vs MLX

Apple M1, float32, median of 10 runs. Ratio < 1.0 = locomp wins.

| Kernel | vs MLX | Speedup |
|--------|--------|---------|
| Flash Attention (N=64) | **0.08×** | 12× faster |
| RoPE | **0.34×** | 2.9× faster |
| GELU+bias | **0.38×** | 2.6× faster |
| Reduce sum | **0.68×** | 1.5× faster |
| Batch norm | **0.69×** | 1.4× faster |
| SwiGLU | **0.74×** | 1.4× faster |
| LayerNorm | **0.77×** | 1.3× faster |
| Multi-head attention | **0.77×** | 1.3× faster |
| RMSNorm | **0.85×** | 1.2× faster |
| Simdgroup matmul (128²) | **0.87×** | 1.15× faster |
| Online softmax | **0.87×** | 1.15× faster |
| Gather | **0.93×** | 1.1× faster |
| Cross attention | **0.93×** | 1.1× faster |
| Avg pool 2D | **0.96×** | Parity |
| Simdgroup matmul (1024²) | **0.96×** | Parity |
| Element-wise add | **1.02×** | Parity |

## Compiler Architecture

### Pipeline

| Stage | File | Lines | What It Does |
|-------|------|-------|-------------|
| Frontend | `locomp/frontend.py` | 861 | Python AST → Locomp IR |
| IR | `locomp/ir.py` | 259 | SSA IR: 60+ opcodes, types, values |
| Optimizer | `locomp/optimizer.py` | 286 | CSE, DCE, constant folding, type inference |
| Codegen | `locomp/backends/metal_codegen.py` | 811 | IR → Metal Shading Language |
| Runtime | `locomp/backends/metal_runtime.py` | 435 | Metal GPU dispatch, buffer management |
| API | `locomp/api.py` | 774 | Public API, tensor, kernel launcher |
| Auto-tune | `locomp/autotune.py` | 248 | Config search + persistent disk cache |
| C Bridge | `locomp/_native/fast_dispatch.m` | 162 | Native async dispatch (bypass PyObjC overhead) |

**Total: 3,836 lines** of compiler + runtime.

### Optimization Passes

- **Common Subexpression Elimination** — eliminates redundant computations with cross-scope pre-declaration
- **Dead Code Elimination** — removes unused values while preserving side effects (store, barrier, atomics)
- **Constant Folding** — evaluates constant expressions at compile time
- **Constexpr Inlining** — kernel parameters marked `constexpr` become MSL literals, enabling Metal compiler loop unrolling and constant folding
- **Type Inference** — propagates float/int/half types through the IR

### Dispatch Optimizations

- **Native C Bridge** — bypasses PyObjC overhead via `ctypes` + compiled Objective-C
- **Async Dispatch** — GPU work pipelines while Python prepares the next call
- **Batch Mode** — multiple dispatches in a single command buffer
- **Lazy Sync** — only syncs GPU on `.numpy()` read-back, not on every dispatch

## Competitive Landscape

| | locomp | MLX | Triton | PyTorch MPS | Raw Metal |
|---|---|---|---|---|---|
| **Apple Silicon** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Custom kernels in Python** | ✅ | ❌ | ✅ (NVIDIA only) | ❌ | ❌ |
| **Kernel language** | Python | Metal C++ strings | Python | N/A | Metal C++ |
| **Full compiler** | ✅ | JIT string emit | ✅ (CUDA) | N/A | Xcode |
| **Auto-tuning** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **INT4/INT8 quantization** | ✅ | ❌ | ❌ | ❌ | Manual |
| **SIMD group matrix** | ✅ | Internal | ❌ | ❌ | ✅ |

**locomp is the only Python → Metal kernel compiler for Apple Silicon.**

## Development

```bash
# Clone
git clone https://github.com/Zyora-Dev/locomp.git
cd locomp

# Install in dev mode
pip install -e ".[dev]"

# Run tests (55 tests)
pytest tests/ -q

# Run a kernel example
python examples/04_matmul.py

# Run benchmarks
python benchmarks/bench_new_kernels.py

# Run SmolLM2 inference (requires: pip install safetensors torch huggingface_hub)
python examples/54_smollm2_inference.py
```

### Project Structure

```
locomp/
├── locomp/
│   ├── __init__.py
│   ├── frontend.py              # Python AST → IR
│   ├── ir.py                    # SSA IR definition
│   ├── optimizer.py             # CSE, DCE, constant folding
│   ├── api.py                   # Public API (@kernel, tensor, etc.)
│   ├── autotune.py              # Auto-tuning framework
│   ├── _native/
│   │   ├── fast_dispatch.m      # Native C bridge (Objective-C)
│   │   └── fast_dispatch.dylib  # Compiled dispatch library
│   └── backends/
│       ├── metal_codegen.py     # IR → Metal Shading Language
│       └── metal_runtime.py     # Metal GPU dispatch + buffers
├── examples/                    # 54 kernel examples
├── benchmarks/                  # Performance benchmarks vs MLX
├── tests/                       # 55 unit tests
└── pyproject.toml
```

## License

Apache-2.0 — [LICENSE](LICENSE)

## Citation

```bibtex
@software{locomp,
  title = {locomp: A Python-to-Metal GPU Kernel Compiler for Apple Silicon},
  url = {https://github.com/Zyora-Dev/locomp},
  year = {2026}
}
```
