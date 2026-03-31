# Locust 🔥

**A universal GPU compute compiler — write once, run on Apple Metal, NVIDIA, AMD.**

Locust lets you write GPU kernels in Python and compiles them to native GPU code for any hardware. Apple Silicon first.

```python
import locust

@locust.kernel
def vector_add(X: locust.Tensor, Y: locust.Tensor, OUT: locust.Tensor, N: locust.constexpr):
    pid = locust.program_id(0)
    offsets = pid * 256 + locust.arange(0, 256)
    mask = offsets < N
    x = locust.load(X + offsets, mask=mask)
    y = locust.load(Y + offsets, mask=mask)
    locust.store(OUT + offsets, x + y, mask=mask)

# Runs on Apple M1/M2/M3/M4 GPU — no CUDA required
x = locust.tensor([1.0, 2.0, 3.0, 4.0])
y = locust.tensor([5.0, 6.0, 7.0, 8.0])
out = locust.empty(4)
vector_add[(1,)](x, y, out, N=4)
print(out)  # [6.0, 8.0, 10.0, 12.0]
```

## Why Locust?

| Feature | Triton | CUDA | Locust |
|---|---|---|---|
| Apple Silicon | ❌ | ❌ | ✅ |
| NVIDIA GPU | ✅ | ✅ | ✅ (planned) |
| AMD GPU | Partial | ❌ | Planned |
| Language | Python DSL | C++ | Python DSL |
| Ownership | OpenAI | NVIDIA | **Community** |

## Architecture

```
@locust.kernel (Python DSL)
        ↓
    Locust IR (tiled compute operations)
        ↓
    Optimization passes (tiling, vectorization, memory coalescing)
        ↓
    Backend codegen:
        ├── Apple Metal → MSL → Apple GPU
        ├── NVIDIA → PTX (planned)
        └── AMD → AMDGPU (planned)
```

## Install

```bash
pip install locust-gpu
```

## Status

🚧 **Phase 1: Apple Metal MVP** — Under active development.

## License

Apache-2.0
