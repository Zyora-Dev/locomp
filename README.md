# Locomp

**A GPU compute compiler for Apple Silicon — write kernels in Python, compile to native Metal shaders.**

Write GPU kernels in Python. Locomp compiles them to optimized Metal shaders and dispatches on Apple M1/M2/M3/M4.

```python
import locomp

@locomp.kernel
def vector_add(X: locomp.Tensor, Y: locomp.Tensor, OUT: locomp.Tensor, N: locomp.constexpr):
    pid = locomp.program_id(0)
    offsets = pid * 256 + locomp.arange(0, 256)
    mask = offsets < N
    x = locomp.load(X + offsets, mask=mask)
    y = locomp.load(Y + offsets, mask=mask)
    locomp.store(OUT + offsets, x + y, mask=mask)

# Runs on Apple GPU
x = locomp.tensor([1.0, 2.0, 3.0, 4.0])
y = locomp.tensor([5.0, 6.0, 7.0, 8.0])
out = locomp.empty(4)
vector_add[(1,)](x, y, out, N=4)
print(out)  # [6.0, 8.0, 10.0, 12.0]
```

## Architecture

```
@locomp.kernel (Python DSL)
        ↓
    Locomp IR (SSA, tiled compute operations)
        ↓
    Optimization passes (constant folding, DCE, type inference)
        ↓
    Metal codegen → MSL → Apple GPU
```

## What's Working

- **Full compiler pipeline**: Python AST → SSA IR (60+ opcodes) → optimizer → Metal codegen
- **Simdgroup matrix ops**: Hardware 8×8 matmul via `simdgroup_multiply_accumulate`
- **25 math ops**: trig, hyperbolic, exp/log, rounding, fma, clamp, sigmoid, etc.
- **Autotune**: `locomp.autotune` + `locomp.Config` — benchmark configs, cache best
- **Native async dispatch**: C bridge with lazy sync — minimal overhead
- **Constexpr inlining**: Kernel params as MSL literals for compiler optimizations
- **25 examples**: matmul, softmax, flash attention, multi-head attention, GELU, LayerNorm

### Benchmarks vs MLX (Apple M1)

- **Matmul**: beats MLX at 4/6 sizes (0.87×–0.97×)
- **Softmax**: beats MLX at 4/5 sizes (0.87×–0.98×)
- **Multi-head attention**: beats MLX at all 5 tested configs (0.77×–0.96×)
- **Flash attention**: beats MLX at N≤128 with async dispatch

## Install

```bash
pip install locomp
```

## License

Apache-2.0
