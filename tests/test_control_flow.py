"""Tests for compiler control flow: if/else, while, break, continue, tuple unpacking, atomics."""

import numpy as np
import locomp
from locomp.frontend import compile_kernel, constexpr, Tensor, Int32, Bool
from locomp.ir import OpCode
from tests.conftest import macos_only


# --- Frontend IR tests ---

def test_if_else_ir():
    def k(X: Tensor, OUT: Tensor, N: constexpr):
        pid = locomp.program_id(0)
        x = locomp.load(X + pid)
        if x > 0.0:
            y = x * 2.0
        else:
            y = x * -1.0
        locomp.store(OUT + pid, y)

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.IF_START in opcodes
    assert OpCode.ELSE_START in opcodes
    assert OpCode.IF_END in opcodes


def test_while_loop_ir():
    def k(X: Tensor, OUT: Tensor):
        pid = locomp.program_id(0)
        i = 0
        acc = 0.0
        while i < 10:
            acc = acc + locomp.load(X + pid * 10 + i)
            i = i + 1
        locomp.store(OUT + pid, acc)

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.WHILE_START in opcodes
    assert OpCode.WHILE_END in opcodes


def test_break_ir():
    def k(X: Tensor, OUT: Tensor):
        pid = locomp.program_id(0)
        i = 0
        while i < 100:
            val = locomp.load(X + pid * 100 + i)
            if val < 0.0:
                break
            i = i + 1
        locomp.store(OUT + pid, i)

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.BREAK in opcodes


def test_continue_ir():
    def k(X: Tensor, OUT: Tensor):
        pid = locomp.program_id(0)
        acc = 0.0
        for i in range(10):
            val = locomp.load(X + pid * 10 + i)
            if val < 0.0:
                continue
            acc = acc + val
        locomp.store(OUT + pid, acc)

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.CONTINUE in opcodes


def test_tuple_unpacking_ir():
    def k(X: Tensor, Y: Tensor, OUT: Tensor):
        pid = locomp.program_id(0)
        a, b = locomp.load(X + pid), locomp.load(Y + pid)
        locomp.store(OUT + pid, a + b)

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.LOAD in opcodes
    assert OpCode.ADD in opcodes


def test_atomic_add_ir():
    def k(X: Tensor, OUT: Tensor):
        pid = locomp.program_id(0)
        val = locomp.load(X + pid)
        locomp.atomic_add(OUT, val)

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.ATOMIC_ADD in opcodes


def test_atomic_max_ir():
    def k(X: Tensor, OUT: Tensor):
        pid = locomp.program_id(0)
        val = locomp.load(X + pid)
        locomp.atomic_max(OUT, val)

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.ATOMIC_MAX in opcodes


def test_atomic_min_ir():
    def k(X: Tensor, OUT: Tensor):
        pid = locomp.program_id(0)
        val = locomp.load(X + pid)
        locomp.atomic_min(OUT, val)

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.ATOMIC_MIN in opcodes


def test_nested_if_inside_for():
    def k(X: Tensor, OUT: Tensor, N: constexpr):
        pid = locomp.program_id(0)
        acc = 0.0
        for i in range(N):
            val = locomp.load(X + pid * N + i)
            if val > 0.0:
                acc = acc + val
            else:
                acc = acc - val
        locomp.store(OUT + pid, acc)

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.FOR_LOOP_START in opcodes
    assert OpCode.IF_START in opcodes
    assert OpCode.ELSE_START in opcodes
    assert OpCode.IF_END in opcodes
    assert OpCode.FOR_LOOP_END in opcodes


def test_nested_for_inside_if():
    def k(X: Tensor, OUT: Tensor, N: constexpr):
        pid = locomp.program_id(0)
        flag = locomp.load(X + pid)
        if flag > 0.0:
            acc = 0.0
            for i in range(N):
                acc = acc + locomp.load(X + pid * N + i)
            locomp.store(OUT + pid, acc)
        else:
            locomp.store(OUT + pid, 0.0)

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.IF_START in opcodes
    assert OpCode.FOR_LOOP_START in opcodes
    assert OpCode.ELSE_START in opcodes


def test_while_with_break_continue():
    def k(X: Tensor, OUT: Tensor):
        pid = locomp.program_id(0)
        i = 0
        acc = 0.0
        while i < 100:
            i = i + 1
            val = locomp.load(X + pid * 100 + i)
            if val < 0.0:
                continue
            if val > 1000.0:
                break
            acc = acc + val
        locomp.store(OUT + pid, acc)

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.WHILE_START in opcodes
    assert OpCode.CONTINUE in opcodes
    assert OpCode.BREAK in opcodes
    assert OpCode.WHILE_END in opcodes


# --- Int32/Bool type annotation tests ---

def test_int32_param():
    def k(X: Int32, OUT: Tensor):
        pid = locomp.program_id(0)
        val = locomp.load(X + pid)
        locomp.store(OUT + pid, val)

    ir = compile_kernel(k)
    from locomp.ir import IRType
    assert ir.params[0].dtype == IRType.INT32
    assert ir.params[0].is_pointer is True


def test_bool_param():
    def k(Mask: Bool, X: Tensor, OUT: Tensor):
        pid = locomp.program_id(0)

    ir = compile_kernel(k)
    from locomp.ir import IRType
    assert ir.params[0].dtype == IRType.BOOL
    assert ir.params[0].is_pointer is True


# --- Codegen tests (MSL output) ---

def test_if_else_codegen():
    from locomp.backends.metal_codegen import compile_to_metal

    def k(X: Tensor, OUT: Tensor, N: constexpr):
        pid = locomp.program_id(0)
        x = locomp.load(X + pid)
        if x > 0.0:
            y = x * 2.0
        else:
            y = x * -1.0
        locomp.store(OUT + pid, y)

    ir = compile_kernel(k)
    msl, _ = compile_to_metal(ir, constexpr_values={"N": 64})
    assert "if (" in msl
    assert "} else {" in msl


def test_while_codegen():
    from locomp.backends.metal_codegen import compile_to_metal

    def k(X: Tensor, OUT: Tensor):
        pid = locomp.program_id(0)
        i = 0
        acc = 0.0
        while i < 10:
            acc = acc + locomp.load(X + pid * 10 + i)
            i = i + 1
        locomp.store(OUT + pid, acc)

    ir = compile_kernel(k)
    msl, _ = compile_to_metal(ir)
    assert "while (true)" in msl
    assert "break;" in msl  # condition check emits break


def test_atomic_codegen():
    from locomp.backends.metal_codegen import compile_to_metal

    def k(X: Tensor, OUT: Tensor):
        pid = locomp.program_id(0)
        val = locomp.load(X + pid)
        locomp.atomic_add(OUT, val)

    ir = compile_kernel(k)
    msl, _ = compile_to_metal(ir)
    assert "atomic_fetch_add_explicit" in msl


# --- Optimizer tests ---

def test_cse_eliminates_duplicate():
    from locomp.optimizer import common_subexpression_eliminate
    from locomp.ir import IRKernel, IROp, IRValue, IRType

    kernel = IRKernel(name="test", params=[])
    a = kernel.new_value("a", IRType.FLOAT32)
    b = kernel.new_value("b", IRType.FLOAT32)
    c = kernel.new_value("c", IRType.FLOAT32)  # a + b
    d = kernel.new_value("d", IRType.FLOAT32)  # a + b (duplicate)

    kernel.ops = [
        IROp(opcode=OpCode.CONSTANT, result=a, operands=[], attrs={"value": 1.0}),
        IROp(opcode=OpCode.CONSTANT, result=b, operands=[], attrs={"value": 2.0}),
        IROp(opcode=OpCode.ADD, result=c, operands=[a, b], attrs={}),
        IROp(opcode=OpCode.ADD, result=d, operands=[a, b], attrs={}),
    ]

    kernel = common_subexpression_eliminate(kernel)
    # d = a + b should be eliminated (reuses c)
    assert len(kernel.ops) == 3  # only const, const, add


def test_strength_reduce_mul2():
    from locomp.optimizer import strength_reduce
    from locomp.ir import IRKernel, IROp, IRValue, IRType

    kernel = IRKernel(name="test", params=[])
    x = kernel.new_value("x", IRType.FLOAT32)
    two = kernel.new_value("two", IRType.INT32)
    r = kernel.new_value("r", IRType.FLOAT32)

    kernel.ops = [
        IROp(opcode=OpCode.CONSTANT, result=x, operands=[], attrs={"value": 5.0}),
        IROp(opcode=OpCode.CONSTANT, result=two, operands=[], attrs={"value": 2}),
        IROp(opcode=OpCode.MUL, result=r, operands=[x, two], attrs={}),
    ]

    kernel = strength_reduce(kernel)
    # MUL by 2 → ADD(x, x)
    last_op = kernel.ops[-1]
    assert last_op.opcode == OpCode.ADD
    assert last_op.operands[0].id == x.id
    assert last_op.operands[1].id == x.id


# --- End-to-end GPU tests (small sizes only) ---

@macos_only
def test_if_else_end_to_end():
    @locomp.kernel
    def k(X: locomp.Tensor, OUT: locomp.Tensor, N: locomp.constexpr):
        pid = locomp.program_id(0)
        x = locomp.load(X + pid)
        if x > 0.0:
            locomp.store(OUT + pid, x * 2.0)
        else:
            locomp.store(OUT + pid, 0.0)

    x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    x_t = locomp.tensor(x)
    out_t = locomp.tensor(out)
    k[(4,)](x_t, out_t, N=4)
    result = out_t.numpy()
    expected = np.array([2.0, 0.0, 6.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    x_t.free()
    out_t.free()


@macos_only
def test_tensor_free():
    from locomp.backends.metal_runtime import get_runtime
    rt = get_runtime()
    before = rt._allocated
    t = locomp.tensor(np.zeros(1024, dtype=np.float32))
    _ = t.to_metal_buffer(rt)
    after_alloc = rt._allocated
    assert after_alloc > before
    t.free()
    after_free = rt._allocated
    assert after_free < after_alloc
