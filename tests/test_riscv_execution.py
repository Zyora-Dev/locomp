"""
RISC-V Execution Tests
======================
Cross-compiles locomp-generated C + RVV kernels and runs them under QEMU.
Compares numerical output against NumPy reference values.

Requirements (auto-skipped if missing):
    sudo apt install qemu-user-static gcc-riscv64-linux-gnu

How it works:
    1. locomp Python kernel  →  compile_to_riscv()  →  C + RVV source
    2. Inject a C main() that seeds input arrays and prints output
    3. riscv64-linux-gnu-gcc -march=rv64gcv -static -O2 → RISC-V ELF
    4. qemu-riscv64-static ./binary  →  stdout floats
    5. Compare stdout values to NumPy reference  →  pass / fail
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import List

import numpy as np
import pytest

import locomp
from locomp.frontend import compile_kernel
from locomp.optimizer import optimize
from locomp.backends.riscv_codegen import compile_to_riscv


# ─────────────────────────────────────────────────────────────────────────────
# Tool detection
# ─────────────────────────────────────────────────────────────────────────────

def _has_tool(name: str) -> bool:
    try:
        r = subprocess.run([name, "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_TOOLCHAIN_AVAILABLE = _has_tool("riscv64-linux-gnu-gcc") and _has_tool("qemu-riscv64-static")

riscv_only = pytest.mark.skipif(
    not _TOOLCHAIN_AVAILABLE,
    reason="riscv64-linux-gnu-gcc + qemu-riscv64-static not installed. "
           "Install: sudo apt install qemu-user-static gcc-riscv64-linux-gnu"
)


# ─────────────────────────────────────────────────────────────────────────────
# Core executor
# ─────────────────────────────────────────────────────────────────────────────

def riscv_run(
    kernel_fn,
    constexpr_values: dict,
    arrays: List[np.ndarray],
    output_indices: List[int],
    grid_size: int,
) -> List[np.ndarray]:
    """Compile a locomp kernel to RISC-V, run on QEMU, return output arrays.

    Args:
        kernel_fn:        plain Python function (not decorated)
        constexpr_values: {name: value} for constexpr params
        arrays:           list of numpy arrays in kernel param order
        output_indices:   which array indices are outputs (printed + returned)
        grid_size:        number of locomp grid blocks to launch

    Returns:
        List of numpy arrays (one per output_index), filled with QEMU results
    """
    # 1. Generate C source
    ir = compile_kernel(kernel_fn)
    ir = optimize(ir, target="riscv")
    c_src, _param_map = compile_to_riscv(ir, constexpr_values=constexpr_values)

    # 2. Build C main() harness
    main_c = _build_main(c_src, kernel_fn.__name__, arrays, output_indices, grid_size)

    with tempfile.TemporaryDirectory() as tmpdir:
        c_path = os.path.join(tmpdir, "kernel.c")
        bin_path = os.path.join(tmpdir, "kernel")

        with open(c_path, "w") as f:
            f.write(main_c)

        # 3. Cross-compile to static RISC-V ELF
        cc_result = subprocess.run(
            [
                "riscv64-linux-gnu-gcc",
                "-march=rv64gcv_zfh",  # RVV 1.0 + half-float
                "-O2", "-static",
                "-o", bin_path, c_path,
                "-lm", "-lpthread",
            ],
            capture_output=True, timeout=60
        )
        if cc_result.returncode != 0:
            # Retry without _zfh (older toolchains)
            cc_result = subprocess.run(
                [
                    "riscv64-linux-gnu-gcc",
                    "-march=rv64gcv",
                    "-O2", "-static",
                    "-o", bin_path, c_path,
                    "-lm", "-lpthread",
                ],
                capture_output=True, timeout=60
            )
        if cc_result.returncode != 0:
            raise RuntimeError(
                f"RISC-V cross-compilation failed:\n"
                f"{cc_result.stderr.decode()}\n"
                f"--- Generated C source ---\n{main_c}"
            )

        # 4. Run under QEMU
        qemu_result = subprocess.run(
            ["qemu-riscv64-static", bin_path],
            capture_output=True, timeout=30
        )
        if qemu_result.returncode != 0:
            raise RuntimeError(
                f"QEMU execution failed (exit {qemu_result.returncode}):\n"
                f"{qemu_result.stderr.decode()}"
            )

        # 5. Parse stdout — one float per line, outputs separated by "---\n"
        stdout = qemu_result.stdout.decode().strip()
        sections = stdout.split("---")
        results = []
        for i, idx in enumerate(output_indices):
            floats = [float(x) for x in sections[i].strip().split("\n") if x.strip()]
            results.append(np.array(floats, dtype=arrays[idx].dtype))

    return results


def _build_main(
    c_src: str,
    kernel_name: str,
    arrays: List[np.ndarray],
    output_indices: List[int],
    grid_size: int,
) -> str:
    """Append a C main() that initializes arrays, calls the kernel, prints outputs."""
    lines = [c_src, "", "#include <stdio.h>", "", "int main(void) {"]

    # Declare static arrays
    for i, arr in enumerate(arrays):
        arr_flat = arr.flatten()
        ctype = _np_to_c_type(arr.dtype)
        init_vals = ", ".join(f"{v:.8f}f" if arr.dtype.kind == 'f' else str(int(v))
                              for v in arr_flat)
        lines.append(f"    static {ctype} arr_{i}[{arr_flat.size}] = {{{init_vals}}};")

    lines.append("")

    # Build void* args array — each element is the data pointer directly
    n_args = len(arrays)
    lines.append(f"    void* _args[{n_args}] = {{")
    for i in range(n_args):
        comma = "," if i < n_args - 1 else ""
        lines.append(f"        arr_{i}{comma}")
    lines.append("    };")
    lines.append("")

    # Call the kernel
    lines.append(f"    locomp_launch_{kernel_name}({grid_size}, (void**)_args);")
    lines.append("")

    # Print output arrays — one float per line, sections separated by "---"
    for section_idx, out_idx in enumerate(output_indices):
        arr = arrays[out_idx]
        arr_flat = arr.flatten()
        ctype = _np_to_c_type(arr.dtype)
        fmt = "%.8f" if arr.dtype.kind == 'f' else "%d"
        lines.append(f"    for (int _i = 0; _i < {arr_flat.size}; _i++) {{")
        lines.append(f'        printf("{fmt}\\n", ({ctype})arr_{out_idx}[_i]);')
        lines.append("    }")
        if section_idx < len(output_indices) - 1:
            lines.append('    printf("---\\n");')
        lines.append("")

    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines)


def _np_to_c_type(dtype: np.dtype) -> str:
    return {
        np.float32: "float",
        np.float64: "double",
        np.float16: "_Float16",
        np.int32:   "int32_t",
        np.int64:   "int64_t",
        np.uint8:   "uint8_t",
    }.get(dtype.type, "float")


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

@riscv_only
def test_vector_add_execution():
    """Scalar load/store: N blocks, one element per block."""
    N = 8
    A = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    B = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.float32)
    Out = np.zeros(N, dtype=np.float32)

    def vector_add(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor,
                   N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        b = locomp.load(B + i)
        locomp.store(Out + i, a + b)

    results = riscv_run(
        vector_add,
        constexpr_values={"N": N},
        arrays=[A, B, Out],
        output_indices=[2],
        grid_size=N,
    )

    expected = A + B
    np.testing.assert_allclose(results[0], expected, rtol=1e-5,
                               err_msg="vector_add: QEMU output mismatch")


@riscv_only
def test_tiled_add_rvv_execution():
    """Tiled add: single block processes 64 elements via RVV intrinsics."""
    TILE = 64
    A = np.random.rand(TILE).astype(np.float32)
    B = np.random.rand(TILE).astype(np.float32)
    Out = np.zeros(TILE, dtype=np.float32)

    def tiled_add(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor,
                  N: locomp.constexpr):
        i = locomp.program_id(0)
        offs = locomp.arange(0, 64)
        a = locomp.load(A + i * 64 + offs)
        b = locomp.load(B + i * 64 + offs)
        locomp.store(Out + i * 64 + offs, a + b)

    results = riscv_run(
        tiled_add,
        constexpr_values={"N": TILE},
        arrays=[A, B, Out],
        output_indices=[2],
        grid_size=1,
    )

    np.testing.assert_allclose(results[0], A + B, rtol=1e-5,
                               err_msg="tiled_add RVV: QEMU output mismatch")


@riscv_only
def test_scale_shift_execution():
    """Elementwise: out = a * scale + shift — tests scalar broadcast in RVV."""
    N = 16
    A = np.arange(1, N + 1, dtype=np.float32)
    Out = np.zeros(N, dtype=np.float32)

    def scale_shift(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        locomp.store(Out + i, a * 2.0 + 1.0)

    results = riscv_run(
        scale_shift,
        constexpr_values={"N": N},
        arrays=[A, Out],
        output_indices=[1],
        grid_size=N,
    )

    expected = A * 2.0 + 1.0
    np.testing.assert_allclose(results[0], expected, rtol=1e-5,
                               err_msg="scale_shift: QEMU output mismatch")


@riscv_only
def test_tiled_mul_rvv_execution():
    """Tiled elementwise multiply — RVV vfmul path."""
    TILE = 32
    A = np.random.rand(TILE).astype(np.float32) * 2.0
    B = np.random.rand(TILE).astype(np.float32) * 2.0
    Out = np.zeros(TILE, dtype=np.float32)

    def tiled_mul(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor,
                  N: locomp.constexpr):
        i = locomp.program_id(0)
        offs = locomp.arange(0, 32)
        a = locomp.load(A + i * 32 + offs)
        b = locomp.load(B + i * 32 + offs)
        locomp.store(Out + i * 32 + offs, a * b)

    results = riscv_run(
        tiled_mul,
        constexpr_values={"N": TILE},
        arrays=[A, B, Out],
        output_indices=[2],
        grid_size=1,
    )

    np.testing.assert_allclose(results[0], A * B, rtol=1e-5,
                               err_msg="tiled_mul RVV: QEMU output mismatch")


@riscv_only
def test_reduce_sum_rvv_execution():
    """RVV vector reduction: sum of 64 floats via vfredosum."""
    TILE = 64
    A = np.random.rand(TILE).astype(np.float32)
    Out = np.zeros(1, dtype=np.float32)

    def reduce_sum(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        offs = locomp.arange(0, 64)
        a = locomp.load(A + offs)
        s = locomp.sum(a)
        locomp.store(Out + i, s)

    results = riscv_run(
        reduce_sum,
        constexpr_values={"N": TILE},
        arrays=[A, Out],
        output_indices=[1],
        grid_size=1,
    )

    expected = np.sum(A)
    # Ordered sum has stricter tolerance — allow small fp accumulation error
    np.testing.assert_allclose(results[0][0], expected, rtol=1e-4,
                               err_msg="reduce_sum RVV: QEMU output mismatch")


@riscv_only
def test_relu_execution():
    """ReLU: max(0, x) — tests conditional / WHERE path."""
    N = 16
    A = np.array([-3, -2, -1, 0, 1, 2, 3, 4, -5, -0.5, 0.5, 6, -7, 8, -9, 10],
                 dtype=np.float32)
    Out = np.zeros(N, dtype=np.float32)

    def relu(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        mask = a > 0.0
        out = locomp.where(mask, a, 0.0)
        locomp.store(Out + i, out)

    results = riscv_run(
        relu,
        constexpr_values={"N": N},
        arrays=[A, Out],
        output_indices=[1],
        grid_size=N,
    )

    expected = np.maximum(A, 0.0)
    np.testing.assert_allclose(results[0], expected, rtol=1e-5,
                               err_msg="relu: QEMU output mismatch")


@riscv_only
def test_dot_product_execution():
    """Dot product: tiled multiply + reduce — end-to-end RVV test."""
    TILE = 64
    A = np.random.rand(TILE).astype(np.float32)
    B = np.random.rand(TILE).astype(np.float32)
    Out = np.zeros(1, dtype=np.float32)

    def dot_product(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor,
                    N: locomp.constexpr):
        i = locomp.program_id(0)
        offs = locomp.arange(0, 64)
        a = locomp.load(A + offs)
        b = locomp.load(B + offs)
        s = locomp.sum(a * b)
        locomp.store(Out + i, s)

    results = riscv_run(
        dot_product,
        constexpr_values={"N": TILE},
        arrays=[A, B, Out],
        output_indices=[2],
        grid_size=1,
    )

    expected = np.dot(A, B)
    np.testing.assert_allclose(results[0][0], expected, rtol=1e-4,
                               err_msg="dot_product RVV: QEMU output mismatch")


@riscv_only
def test_sqrt_exp_execution():
    """sqrt + exp math ops — verifies math.h linking."""
    N = 8
    A = np.array([1, 4, 9, 16, 0.25, 0.5, 2, 3], dtype=np.float32)
    Out_sqrt = np.zeros(N, dtype=np.float32)
    Out_exp = np.zeros(N, dtype=np.float32)

    def sqrt_kern(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        locomp.store(Out + i, locomp.sqrt(a))

    results = riscv_run(
        sqrt_kern,
        constexpr_values={"N": N},
        arrays=[A, Out_sqrt],
        output_indices=[1],
        grid_size=N,
    )
    np.testing.assert_allclose(results[0], np.sqrt(A), rtol=1e-5,
                               err_msg="sqrt: QEMU output mismatch")

    def exp_kern(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        locomp.store(Out + i, locomp.exp(a))

    results = riscv_run(
        exp_kern,
        constexpr_values={"N": N},
        arrays=[A, Out_exp],
        output_indices=[1],
        grid_size=N,
    )
    np.testing.assert_allclose(results[0], np.exp(A), rtol=1e-5,
                               err_msg="exp: QEMU output mismatch")


@riscv_only
def test_rms_norm_execution():
    """RMS Norm: tiled multiply, reduce_sum, sqrt, division — full pipeline."""
    DIM = 32
    X = np.random.randn(DIM).astype(np.float32)
    W = np.ones(DIM, dtype=np.float32)
    Out = np.zeros(DIM, dtype=np.float32)

    def rms_norm(X: locomp.Tensor, W: locomp.Tensor, Out: locomp.Tensor,
                 N: locomp.constexpr):
        row = locomp.program_id(0)
        offs = locomp.arange(0, 32)
        x = locomp.load(X + row * 32 + offs)
        w = locomp.load(W + offs)
        x2 = x * x
        mean_x2 = locomp.sum(x2) / 32.0
        rms = locomp.sqrt(mean_x2 + 1e-6)
        normed = (x / rms) * w
        locomp.store(Out + row * 32 + offs, normed)

    results = riscv_run(
        rms_norm,
        constexpr_values={"N": DIM},
        arrays=[X, W, Out],
        output_indices=[2],
        grid_size=1,
    )

    # NumPy reference
    rms_ref = np.sqrt(np.mean(X ** 2) + 1e-6)
    expected = (X / rms_ref) * W
    np.testing.assert_allclose(results[0], expected, rtol=1e-4,
                               err_msg="rms_norm RVV: QEMU output mismatch")
