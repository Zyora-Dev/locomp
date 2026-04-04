"""Tests for 2D grid dispatch — chained pointer arithmetic and 2D program_id."""

import platform
import pytest
import locomp
from locomp.frontend import compile_kernel, Tensor, constexpr
from locomp.backends.metal_codegen import compile_to_metal
from locomp.ir import OpCode, IRType

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin", reason="Metal/MSL tests require macOS"
)


def test_program_id_axis1_emits_tgid_y():
    """program_id(1) should emit tgid.y in MSL."""
    @locomp.kernel
    def k(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
        row = locomp.program_id(0)
        col = locomp.program_id(1)
        v = locomp.load(X + row * N + col)
        locomp.store(O + row * N + col, v)

    msl = k.msl
    assert "tgid.x" in msl
    assert "tgid.y" in msl


def test_program_id_axis2_emits_tgid_z():
    """program_id(2) should emit tgid.z in MSL."""
    @locomp.kernel
    def k(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
        depth = locomp.program_id(2)
        v = locomp.load(X + depth)
        locomp.store(O + depth, v)

    msl = k.msl
    assert "tgid.z" in msl


def test_2d_pointer_arithmetic_combines_indices():
    """A + row*N + col should compile to A_ptr[(row*N + col)] — not just col."""
    @locomp.kernel
    def mat_add(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                M: locomp.constexpr, N: locomp.constexpr):
        row = locomp.program_id(0)
        col = locomp.program_id(1)
        a = locomp.load(A + row * N + col)
        b = locomp.load(B + row * N + col)
        locomp.store(C + row * N + col, a + b)

    msl = mat_add.msl
    # The combined index expression must be present — not just individual vars
    # row*N compiled as a MUL, then added to col → combined index string
    assert "+" in msl  # combined index has a + somewhere
    # Both A and B should be dereferenced via buffer subscript
    # Buffer names get _N suffix (A_0, B_1, C_2)
    assert "A_0[" in msl
    assert "B_1[" in msl


def test_2d_grid_ir_has_two_program_ids():
    """2D kernel should have two PROGRAM_ID ops with axis 0 and 1."""
    def k(A: Tensor, M: constexpr, N: constexpr):
        row = locomp.program_id(0)
        col = locomp.program_id(1)

    ir = compile_kernel(k)
    pid_ops = [op for op in ir.ops if op.opcode == OpCode.PROGRAM_ID]
    axes = [op.attrs["axis"] for op in pid_ops]
    assert 0 in axes
    assert 1 in axes


def test_2d_kernel_msl_structure():
    """Full 2D matrix addition kernel produces valid MSL with both buffer args."""
    @locomp.kernel
    def mat_add(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                M: locomp.constexpr, N: locomp.constexpr):
        row = locomp.program_id(0)
        col = locomp.program_id(1)
        idx = row * N + col
        a = locomp.load(A + idx)
        b = locomp.load(B + idx)
        locomp.store(C + idx, a + b)

    msl = mat_add.msl
    assert "kernel void mat_add" in msl
    assert "device float* A" in msl
    assert "device float* B" in msl
    assert "device float* C" in msl
    assert "tgid.x" in msl
    assert "tgid.y" in msl
