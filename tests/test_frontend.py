"""Tests for the Locust frontend — Python AST → IR compilation."""

import locomp
from locomp.frontend import compile_kernel, constexpr, Tensor
from locomp.ir import OpCode


# Use compile_kernel directly to test frontend without optimizer DCE

def test_simple_kernel_compiles():
    def add_kernel(X: Tensor, Y: Tensor, OUT: Tensor):
        pid = locomp.program_id(0)

    ir = compile_kernel(add_kernel)
    assert ir.name == "add_kernel"
    assert len(ir.params) == 3
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.PROGRAM_ID in opcodes


def test_kernel_with_constexpr():
    def k(X: Tensor, N: constexpr):
        pid = locomp.program_id(0)

    ir = compile_kernel(k)
    assert len(ir.params) == 2
    assert ir.params[0].is_pointer is True
    assert ir.params[1].is_pointer is False


def test_arange_produces_ir():
    def k(X: Tensor):
        offsets = locomp.arange(0, 256)

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.ARANGE in opcodes


def test_arithmetic_ops():
    def k(X: Tensor, Y: Tensor):
        pid = locomp.program_id(0)
        offsets = pid * 256 + locomp.arange(0, 256)

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.MUL in opcodes
    assert OpCode.ADD in opcodes


def test_comparison_op():
    def k(X: Tensor, N: constexpr):
        offsets = locomp.arange(0, 256)
        mask = offsets < N

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.CMP_LT in opcodes


def test_load_store():
    def k(X: Tensor, OUT: Tensor):
        pid = locomp.program_id(0)
        x = locomp.load(X + pid)
        locomp.store(OUT + pid, x)

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.LOAD in opcodes
    assert OpCode.STORE in opcodes


def test_reduction():
    def k(X: Tensor, OUT: Tensor):
        offsets = locomp.arange(0, 256)
        x = locomp.load(X + offsets)
        s = locomp.sum(x)
        locomp.store(OUT, s)

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.REDUCE_SUM in opcodes
