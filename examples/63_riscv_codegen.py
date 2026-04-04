"""
Example 63 — RISC-V RVV Codegen
================================
Demonstrates locomp compiling Python kernels to C + RVV intrinsics.

On macOS (no RISC-V hardware): generates and prints the C source.
On RISC-V Linux with GCC: compiles and runs natively via ctypes.

Run:
    python examples/63_riscv_codegen.py
"""

import sys
import numpy as np

# ── locomp imports ────────────────────────────────────────────────────────────
sys.path.insert(0, ".")
import locomp
from locomp.frontend import compile_kernel
from locomp.optimizer import optimize
from locomp.backends.riscv_codegen import compile_to_riscv


# ─────────────────────────────────────────────────────────────────────────────
# Kernel definitions
# ─────────────────────────────────────────────────────────────────────────────

@locomp.kernel(backend="riscv")
def vector_add(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor,
               N: locomp.constexpr):
    i = locomp.program_id(0)
    a = locomp.load(A + i)
    b = locomp.load(B + i)
    locomp.store(Out + i, a + b)


@locomp.kernel(backend="riscv")
def rms_norm(X: locomp.Tensor, W: locomp.Tensor, Out: locomp.Tensor,
             N: locomp.constexpr):
    row = locomp.program_id(0)
    offs = locomp.arange(0, N)
    x = locomp.load(X + row * N + offs)
    w = locomp.load(W + offs)
    # RMS norm: x / sqrt(mean(x^2) + eps) * w
    x2 = x * x
    mean_x2 = locomp.sum(x2) / N
    rms = locomp.sqrt(mean_x2 + 1e-6)
    normed = (x / rms) * w
    locomp.store(Out + row * N + offs, normed)


@locomp.kernel(backend="riscv")
def gelu(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    offs = locomp.arange(0, N)
    x = locomp.load(X + i * N + offs)
    # GELU approx: 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
    x3 = x * x * x
    inner = 0.7978845608 * (x + 0.044715 * x3)
    out = 0.5 * x * (1.0 + locomp.tanh(inner))
    locomp.store(Out + i * N + offs, out)


# ─────────────────────────────────────────────────────────────────────────────
# Codegen inspection — show the generated C + RVV source
# ─────────────────────────────────────────────────────────────────────────────

def show_codegen(kernel_fn, constexpr_values, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print('='*70)
    ir = compile_kernel(kernel_fn.func)
    ir = optimize(ir, target="riscv")
    c_source, param_map = compile_to_riscv(ir, constexpr_values=constexpr_values)
    print(c_source)
    print(f"\n  param_map: {param_map}")


show_codegen(vector_add, {"N": 1024}, "vector_add  →  C + RVV")
show_codegen(gelu, {"N": 64}, "gelu  →  C + RVV")


# ─────────────────────────────────────────────────────────────────────────────
# Execution test (macOS: compiles with host CC, no RVV instructions)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  Execution test (host C compiler)")
print("="*70)

N = 256
A_np = np.random.rand(N).astype(np.float32)
B_np = np.random.rand(N).astype(np.float32)
Out_np = np.zeros(N, dtype=np.float32)

A = locomp.tensor(A_np.copy())
B = locomp.tensor(B_np.copy())
Out = locomp.tensor(Out_np.copy())

try:
    vector_add[(N,)](A, B, Out, N)
    # For RISC-V backend, data stays in numpy
    result = Out.data if Out.data is not None else Out_np
    expected = A_np + B_np
    # Use the original numpy arrays since RISC-V launch writes directly
    print(f"  vector_add N={N}")
    print(f"  ✓ Compiled and launched with host C compiler")
    print(f"  Note: On RISC-V Linux with -march=rv64gcv, RVV intrinsics activate")
except RuntimeError as e:
    print(f"  ⚠ Launch skipped (no C compiler): {e}")
    print(f"  Install gcc to run on host, or riscv64-linux-gnu-gcc for native RVV")


# ─────────────────────────────────────────────────────────────────────────────
# IR → RVV mapping summary
# ─────────────────────────────────────────────────────────────────────────────

print("""
IR → RVV Mapping
─────────────────────────────────────────────────────────────
locomp IR Op     RVV Intrinsic
─────────────────────────────────────────────────────────────
LOAD (tiled)  →  __riscv_vle32_v_f32m1(ptr, vl)
STORE (tiled) →  __riscv_vse32_v_f32m1(ptr, val, vl)
ADD (tiled)   →  __riscv_vfadd_vv_f32m1(va, vb, vl)
SUB (tiled)   →  __riscv_vfsub_vv_f32m1(va, vb, vl)
MUL (tiled)   →  __riscv_vfmul_vv_f32m1(va, vb, vl)
DIV (tiled)   →  __riscv_vfdiv_vv_f32m1(va, vb, vl)
REDUCE_SUM    →  __riscv_vfredosum_vs_f32m1_f32m1(vd, acc, vl)
REDUCE_MAX    →  __riscv_vfredmax_vs_f32m1_f32m1(vd, acc, vl)
REDUCE_MIN    →  __riscv_vfredmin_vs_f32m1_f32m1(vd, acc, vl)
WHERE (tiled) →  scalar loop with ternary (no bitmask overhead)
─────────────────────────────────────────────────────────────
Parallelism   →  POSIX pthreads (1 thread per grid block)
─────────────────────────────────────────────────────────────
""")
