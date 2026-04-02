"""Tests for native Float16 kernel support."""

import locomp
from locomp.frontend import compile_kernel, Float16, constexpr
from locomp.backends.metal_codegen import compile_to_metal
from locomp.ir import IRType, OpCode


def test_float16_param_ir_type():
    """Float16 annotation → FLOAT16 pointer param in IR."""
    def k(X: Float16, Y: Float16, O: Float16, N: constexpr):
        i = locomp.program_id(0)

    ir = compile_kernel(k)
    assert ir.params[0].dtype == IRType.FLOAT16
    assert ir.params[0].is_pointer is True
    assert ir.params[1].dtype == IRType.FLOAT16
    assert ir.params[2].dtype == IRType.FLOAT16


def test_float16_load_dtype():
    """Loading from a Float16 pointer produces FLOAT16 values."""
    def k(X: Float16, O: Float16, N: constexpr):
        i = locomp.program_id(0)
        x = locomp.load(X + i)
        locomp.store(O + i, x)

    ir = compile_kernel(k)
    load_ops = [op for op in ir.ops if op.opcode == OpCode.LOAD]
    assert len(load_ops) == 1
    assert load_ops[0].result.dtype == IRType.FLOAT16


def test_float16_msl_uses_half_type():
    """Float16 params compile to MSL 'device half*' buffers."""
    @locomp.kernel
    def k(X: locomp.Float16, O: locomp.Float16, N: locomp.constexpr):
        i = locomp.program_id(0)
        x = locomp.load(X + i)
        locomp.store(O + i, x)

    msl = k.msl
    assert "device half*" in msl


def test_float16_arithmetic_has_explicit_cast():
    """Float16 arithmetic result gets explicit (half)(...) cast in MSL."""
    @locomp.kernel
    def k(X: locomp.Float16, O: locomp.Float16, N: locomp.constexpr):
        i = locomp.program_id(0)
        x = locomp.load(X + i)
        locomp.store(O + i, x + x)

    msl = k.msl
    # MSL half arithmetic needs explicit cast back to half
    assert "(half)" in msl


def test_float16_cast_to_float32_in_msl():
    """locomp.cast(val, 'float32') from Float16 emits (float) cast."""
    @locomp.kernel
    def k(X: locomp.Float16, O: locomp.Tensor, N: locomp.constexpr):
        i = locomp.program_id(0)
        x = locomp.load(X + i)
        locomp.store(O + i, locomp.cast(x, "float32"))

    msl = k.msl
    assert "(float)" in msl


def test_float16_ir_bytewidth():
    assert IRType.FLOAT16.bytewidth == 2
    assert IRType.FLOAT16.is_float is True
    assert IRType.FLOAT16.to_msl() == "half"
