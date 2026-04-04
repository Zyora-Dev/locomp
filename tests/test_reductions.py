"""Tests for reduction ops — reduce_sum/max/min IR and codegen."""

import locomp
from locomp.frontend import compile_kernel, Tensor, constexpr
from locomp.backends.metal_codegen import compile_to_metal
from locomp.ir import OpCode, IRType
from tests.conftest import macos_only


def test_reduce_sum_global_produces_ir_op():
    """reduce_sum(val, ptr) emits a 2-operand REDUCE_SUM op in IR."""
    def k(X: Tensor, Out: Tensor, N: constexpr):
        i = locomp.program_id(0)
        val = locomp.load(X + i)
        locomp.reduce_sum(val, Out)

    ir = compile_kernel(k)
    reduce_ops = [op for op in ir.ops if op.opcode == OpCode.REDUCE_SUM]
    assert len(reduce_ops) == 1
    assert len(reduce_ops[0].operands) == 2  # val + acc_ptr


def test_reduce_max_global_produces_ir_op():
    """reduce_max(val, ptr) emits a 2-operand REDUCE_MAX op in IR."""
    def k(X: Tensor, Out: Tensor, N: constexpr):
        i = locomp.program_id(0)
        val = locomp.load(X + i)
        locomp.reduce_max(val, Out)

    ir = compile_kernel(k)
    reduce_ops = [op for op in ir.ops if op.opcode == OpCode.REDUCE_MAX]
    assert len(reduce_ops) == 1
    assert len(reduce_ops[0].operands) == 2


def test_reduce_min_global_produces_ir_op():
    """reduce_min(val, ptr) emits a 2-operand REDUCE_MIN op in IR."""
    def k(X: Tensor, Out: Tensor, N: constexpr):
        i = locomp.program_id(0)
        val = locomp.load(X + i)
        locomp.reduce_min(val, Out)

    ir = compile_kernel(k)
    reduce_ops = [op for op in ir.ops if op.opcode == OpCode.REDUCE_MIN]
    assert len(reduce_ops) == 1
    assert len(reduce_ops[0].operands) == 2


@macos_only
def test_reduce_sum_msl_uses_simd_and_atomic():
    """reduce_sum emits simd_sum + atomic_fetch_add in MSL."""
    @locomp.kernel
    def k(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        val = locomp.load(X + i)
        locomp.reduce_sum(val, Out)

    msl = k.msl
    assert "simd_sum(" in msl
    assert "atomic_fetch_add_explicit" in msl
    assert "simd_lid" in msl


@macos_only
def test_reduce_max_msl_uses_simd_and_cas():
    """reduce_max emits simd_max + CAS atomic pattern in MSL."""
    @locomp.kernel
    def k(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        val = locomp.load(X + i)
        locomp.reduce_max(val, Out)

    msl = k.msl
    assert "simd_max(" in msl
    assert "atomic_compare_exchange_weak_explicit" in msl


@macos_only
def test_reduce_min_msl_uses_simd_and_cas():
    """reduce_min emits simd_min + CAS atomic pattern in MSL."""
    @locomp.kernel
    def k(X: locomp.Tensor, Out: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        val = locomp.load(X + i)
        locomp.reduce_min(val, Out)

    msl = k.msl
    assert "simd_min(" in msl
    assert "atomic_compare_exchange_weak_explicit" in msl


@macos_only
def test_tile_sum_still_works():
    """locomp.sum(tile) — 1-operand tile reduction — still works."""
    @locomp.kernel
    def k(X: locomp.Tensor, Out: locomp.Tensor):
        vals = locomp.arange(0, 128)
        x = locomp.load(X + vals)
        total = locomp.sum(x)
        locomp.store(Out, total)

    msl = k.msl
    assert "+=" in msl  # tile reduction loop


@macos_only
def test_tile_max_works():
    """locomp.max(tile) — 1-operand tile max reduction."""
    @locomp.kernel
    def k(X: locomp.Tensor, Out: locomp.Tensor):
        vals = locomp.arange(0, 64)
        x = locomp.load(X + vals)
        m = locomp.max(x)
        locomp.store(Out, m)

    msl = k.msl
    assert "max(" in msl


@macos_only
def test_tile_min_works():
    """locomp.min(tile) — 1-operand tile min reduction."""
    @locomp.kernel
    def k(X: locomp.Tensor, Out: locomp.Tensor):
        vals = locomp.arange(0, 64)
        x = locomp.load(X + vals)
        m = locomp.min(x)
        locomp.store(Out, m)

    msl = k.msl
    assert "min(" in msl


def test_reduce_ops_exported():
    """reduce_sum/max/min are importable from locomp namespace."""
    import locomp
    assert callable(locomp.reduce_sum)
    assert callable(locomp.reduce_max)
    assert callable(locomp.reduce_min)
