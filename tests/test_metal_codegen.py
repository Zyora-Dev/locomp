"""Tests for Metal codegen — IR → MSL compilation."""

import platform
import pytest
import locomp
from locomp.backends.metal_codegen import compile_to_metal

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin", reason="Metal/MSL tests require macOS"
)


def test_vector_add_generates_msl():
    @locomp.kernel
    def vector_add(X: locomp.Tensor, Y: locomp.Tensor, OUT: locomp.Tensor, N: locomp.constexpr):
        pid = locomp.program_id(0)
        offsets = pid * 256 + locomp.arange(0, 256)
        mask = offsets < N
        x = locomp.load(X + offsets, mask=mask)
        y = locomp.load(Y + offsets, mask=mask)
        output = x + y
        locomp.store(OUT + offsets, output, mask=mask)

    msl = vector_add.msl
    assert "#include <metal_stdlib>" in msl
    assert "kernel void vector_add" in msl
    assert "thread_position_in_grid" in msl
    assert "device float*" in msl


def test_dot_product_generates_reduction():
    @locomp.kernel
    def dot_product(A: locomp.Tensor, B: locomp.Tensor, OUT: locomp.Tensor):
        offsets = locomp.arange(0, 128)
        a = locomp.load(A + offsets)
        b = locomp.load(B + offsets)
        dot = locomp.sum(a * b)
        locomp.store(OUT, dot)

    msl = dot_product.msl
    assert "kernel void dot_product" in msl
    # Should have a reduction loop
    assert "+=" in msl


def test_msl_has_correct_structure():
    @locomp.kernel
    def simple(X: locomp.Tensor):
        pid = locomp.program_id(0)

    msl = simple.msl
    # Must have metal headers
    assert "using namespace metal" in msl
    # Must have kernel signature with buffer binding
    assert "[[buffer(0)]]" in msl
    # Must have thread position
    assert "[[thread_position_in_grid]]" in msl


def test_constexpr_param_in_msl():
    @locomp.kernel
    def k(X: locomp.Tensor, N: locomp.constexpr):
        pid = locomp.program_id(0)

    msl = k.msl
    assert "constant int&" in msl
    assert "[[buffer(1)]]" in msl
