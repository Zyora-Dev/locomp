"""Tests for RISC-V RVV codegen — IR → C + RVV intrinsics."""

import locomp
from locomp.frontend import compile_kernel
from locomp.optimizer import optimize
from locomp.backends.riscv_codegen import compile_to_riscv


def _codegen(fn, constexpr_values=None):
    ir = compile_kernel(fn)
    ir = optimize(ir, target="riscv")
    c_src, param_map = compile_to_riscv(ir, constexpr_values=constexpr_values or {})
    return c_src, param_map


# ─────────────────────────────────────────────────────────────────────────────
# Basic structure tests
# ─────────────────────────────────────────────────────────────────────────────

def test_generated_c_has_headers():
    def vadd(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        b = locomp.load(B + i)
        locomp.store(Out + i, a + b)

    c_src, _ = _codegen(vadd, {"N": 1024})
    assert "#include <riscv_vector.h>" in c_src
    assert "#include <stdint.h>" in c_src
    assert "#include <math.h>" in c_src
    assert "#include <pthread.h>" in c_src


def test_generated_c_has_kernel_function():
    def vadd(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        b = locomp.load(B + i)
        locomp.store(Out + i, a + b)

    c_src, _ = _codegen(vadd, {"N": 64})
    assert "static void vadd_block(" in c_src
    assert "locomp_launch_vadd(" in c_src   # public launch entry point
    assert "pthread_create" in c_src
    assert "pthread_join" in c_src


def test_program_id_maps_to_block_id():
    def mykern(A: locomp.Tensor, B: locomp.Tensor, N: locomp.constexpr):
        pid = locomp.program_id(0)
        a = locomp.load(A + pid)
        locomp.store(B + pid, a)

    c_src, _ = _codegen(mykern, {"N": 128})
    assert "_block_id" in c_src


# ─────────────────────────────────────────────────────────────────────────────
# RVV intrinsic emission tests
# ─────────────────────────────────────────────────────────────────────────────

def test_tiled_load_emits_rvv_vle():
    def tile_load(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        offs = locomp.arange(0, 64)
        a = locomp.load(A + i * 64 + offs)
        locomp.store(Out + i * 64 + offs, a)

    c_src, _ = _codegen(tile_load, {"N": 64})
    # Tiled load must use RVV vector load intrinsic
    assert "__riscv_vle32" in c_src or "__riscv_vle" in c_src


def test_tiled_store_emits_rvv_vse():
    def tile_store(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        offs = locomp.arange(0, 64)
        a = locomp.load(A + i * 64 + offs)
        locomp.store(Out + i * 64 + offs, a)

    c_src, _ = _codegen(tile_store, {"N": 64})
    assert "__riscv_vse32" in c_src or "__riscv_vse" in c_src


def test_tiled_add_emits_rvv_vfadd():
    def tile_add(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        offs = locomp.arange(0, 64)
        a = locomp.load(A + i * 64 + offs)
        b = locomp.load(B + i * 64 + offs)
        locomp.store(Out + i * 64 + offs, a + b)

    c_src, _ = _codegen(tile_add, {"N": 64})
    assert "__riscv_vfadd" in c_src


def test_tiled_mul_emits_rvv_vfmul():
    def tile_mul(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        offs = locomp.arange(0, 32)
        a = locomp.load(A + i * 32 + offs)
        b = locomp.load(B + i * 32 + offs)
        locomp.store(Out + i * 32 + offs, a * b)

    c_src, _ = _codegen(tile_mul, {"N": 32})
    assert "__riscv_vfmul" in c_src


def test_reduce_sum_emits_rvv_vredosum():
    def red_sum(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        offs = locomp.arange(0, 128)  # literal size, not constexpr var
        a = locomp.load(A + offs)
        s = locomp.sum(a)
        locomp.store(Out + i, s)

    c_src, _ = _codegen(red_sum, {"N": 128})
    assert "__riscv_vfredosum" in c_src or "vfredosum" in c_src


def test_reduce_max_emits_rvv_vredmax():
    def red_max(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        offs = locomp.arange(0, 64)  # literal size
        a = locomp.load(A + offs)
        m = locomp.max(a)
        locomp.store(Out + i, m)

    c_src, _ = _codegen(red_max, {"N": 64})
    assert "__riscv_vfredmax" in c_src or "vfredmax" in c_src


# ─────────────────────────────────────────────────────────────────────────────
# Type mapping tests
# ─────────────────────────────────────────────────────────────────────────────

def test_float32_params_use_float_type():
    def kern(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        locomp.store(Out + i, a)

    c_src, _ = _codegen(kern, {"N": 64})
    # float* params in struct and function signature
    assert "float* A" in c_src or "float *A" in c_src


def test_int32_constexpr_is_inlined():
    def kern(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        locomp.store(Out + i, a)

    c_src, param_map = _codegen(kern, {"N": 512})
    # Constexpr is inlined / DCE'd — "N" must NOT appear as a struct field or param
    assert "N" not in param_map
    assert "int32_t N" not in c_src   # not in struct or function signature


# ─────────────────────────────────────────────────────────────────────────────
# Control flow tests
# ─────────────────────────────────────────────────────────────────────────────

def test_for_loop_generates_c_for():
    def loop_kern(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        acc = 0.0
        for k in range(N):
            a = locomp.load(A + i * N + k)
            acc = acc + a
        locomp.store(Out + i, acc)

    c_src, _ = _codegen(loop_kern, {"N": 16})
    assert "for (" in c_src


def test_if_generates_c_if():
    def cond_kern(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        mask = a > 0.0
        b = locomp.where(mask, a, 0.0)
        locomp.store(Out + i, b)

    c_src, _ = _codegen(cond_kern, {"N": 64})
    assert "if (" in c_src or "? " in c_src


# ─────────────────────────────────────────────────────────────────────────────
# Math operator tests
# ─────────────────────────────────────────────────────────────────────────────

def test_sqrt_generates_sqrtf():
    def sqrt_kern(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        locomp.store(Out + i, locomp.sqrt(a))

    c_src, _ = _codegen(sqrt_kern, {"N": 64})
    assert "sqrtf" in c_src


def test_exp_generates_expf():
    def exp_kern(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        locomp.store(Out + i, locomp.exp(a))

    c_src, _ = _codegen(exp_kern, {"N": 64})
    assert "expf" in c_src


def test_tanh_generates_tanhf():
    def tanh_kern(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        locomp.store(Out + i, locomp.tanh(a))

    c_src, _ = _codegen(tanh_kern, {"N": 64})
    assert "tanhf" in c_src


# ─────────────────────────────────────────────────────────────────────────────
# param_map tests
# ─────────────────────────────────────────────────────────────────────────────

def test_param_map_has_all_pointer_params():
    def kern(A: locomp.Tensor, B: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        b = locomp.load(B + i)
        locomp.store(Out + i, a + b)

    _, param_map = _codegen(kern, {"N": 64})
    # All 3 tensor params must appear with base names (no SSA suffix)
    assert "A" in param_map
    assert "B" in param_map
    assert "Out" in param_map


def test_constexpr_not_in_param_map():
    def kern(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        a = locomp.load(A + i)
        locomp.store(Out + i, a)

    _, param_map = _codegen(kern, {"N": 256})
    # Constexpr is inlined, should not be in param_map
    assert "N" not in param_map


# ─────────────────────────────────────────────────────────────────────────────
# vsetvl emission test
# ─────────────────────────────────────────────────────────────────────────────

def test_vsetvl_in_tiled_ops():
    def kern(A: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        offs = locomp.arange(0, 64)
        a = locomp.load(A + i * 64 + offs)
        locomp.store(Out + i * 64 + offs, a)

    c_src, _ = _codegen(kern, {"N": 64})
    assert "__riscv_vsetvl" in c_src
