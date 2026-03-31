"""Tests for the optimizer passes."""

from locomp.ir import IRKernel, IRType, OpCode
from locomp.optimizer import constant_fold, dead_code_eliminate, optimize


def test_constant_folding():
    kernel = IRKernel(name="test", params=[])
    a = kernel.new_value("a", IRType.INT32)
    kernel.add_op(OpCode.CONSTANT, a, attrs={"value": 10})
    b = kernel.new_value("b", IRType.INT32)
    kernel.add_op(OpCode.CONSTANT, b, attrs={"value": 20})
    c = kernel.new_value("c", IRType.INT32)
    kernel.add_op(OpCode.ADD, c, [a, b])

    kernel = constant_fold(kernel)

    # The ADD should be folded into a CONSTANT
    add_ops = [op for op in kernel.ops if op.opcode == OpCode.ADD]
    assert len(add_ops) == 0
    # Should have 3 constants now (a, b, and the folded c)
    const_ops = [op for op in kernel.ops if op.opcode == OpCode.CONSTANT]
    assert len(const_ops) == 3
    # The last constant should be 30
    assert const_ops[-1].attrs["value"] == 30


def test_dead_code_elimination():
    kernel = IRKernel(name="test", params=[])

    # Create a value that's never used
    dead = kernel.new_value("dead", IRType.FLOAT32)
    kernel.add_op(OpCode.CONSTANT, dead, attrs={"value": 42.0})

    # Create a value that IS used in a store
    ptr = kernel.new_value("ptr", IRType.FLOAT32, is_pointer=True)
    kernel.params.append(ptr)
    live = kernel.new_value("live", IRType.FLOAT32)
    kernel.add_op(OpCode.CONSTANT, live, attrs={"value": 1.0})
    store_result = kernel.new_value("store", IRType.BOOL)
    kernel.add_op(OpCode.STORE, store_result, [ptr, live])

    kernel = dead_code_eliminate(kernel)

    # Dead constant should be removed
    values = [op.result.name for op in kernel.ops]
    assert "dead_0" not in values
    # Live constant and store should remain
    assert any("live" in v for v in values)
    assert any("store" in v for v in values)
