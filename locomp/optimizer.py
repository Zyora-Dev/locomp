"""
Locust Optimizer — IR optimization passes.

Applies transformations to the IR before backend codegen:
1. Constant folding — evaluate compile-time expressions
2. Dead code elimination — remove unused values
3. Type inference — propagate types through the graph
4. Tiling analysis — determine optimal tile sizes per backend
"""

from __future__ import annotations

from locomp.ir import IRKernel, IROp, IRValue, OpCode, IRType


def optimize(kernel: IRKernel, target: str = "metal") -> IRKernel:
    """Run all optimization passes on the kernel."""
    kernel = constant_fold(kernel)
    kernel = strength_reduce(kernel)
    kernel = common_subexpression_eliminate(kernel)
    kernel = dead_code_eliminate(kernel)
    kernel = infer_types(kernel)
    return kernel


def constant_fold(kernel: IRKernel) -> IRKernel:
    """Fold constant expressions at compile time."""
    constants: dict[int, any] = {}

    new_ops = []
    for op in kernel.ops:
        if op.opcode == OpCode.CONSTANT:
            constants[op.result.id] = op.attrs["value"]
            new_ops.append(op)
            continue

        # Check if all operands are constants
        if (op.opcode in (OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.MOD)
                and len(op.operands) == 2
                and all(o.id in constants for o in op.operands)):
            a = constants[op.operands[0].id]
            b = constants[op.operands[1].id]
            if op.opcode == OpCode.ADD:
                result_val = a + b
            elif op.opcode == OpCode.SUB:
                result_val = a - b
            elif op.opcode == OpCode.MUL:
                result_val = a * b
            elif op.opcode == OpCode.DIV:
                result_val = a / b if b != 0 else a
            elif op.opcode == OpCode.MOD:
                result_val = a % b if b != 0 else 0
            else:
                new_ops.append(op)
                continue
            constants[op.result.id] = result_val
            folded = IROp(opcode=OpCode.CONSTANT, result=op.result,
                          operands=[], attrs={"value": result_val})
            new_ops.append(folded)
        else:
            new_ops.append(op)

    kernel.ops = new_ops
    return kernel


def strength_reduce(kernel: IRKernel) -> IRKernel:
    """Replace expensive operations with cheaper equivalents."""
    constants: dict[int, any] = {}
    for op in kernel.ops:
        if op.opcode == OpCode.CONSTANT:
            constants[op.result.id] = op.attrs["value"]

    # Remove alias targets from constants — they are mutable accumulators
    # whose initial constant value is not valid for all iterations
    alias_targets = {op.result.aliases for op in kernel.ops if op.result.aliases is not None}
    for target_id in alias_targets:
        constants.pop(target_id, None)

    new_ops = []
    for op in kernel.ops:
        # Skip mutable accumulator updates — they need the alias chain intact
        if op.result.aliases is not None:
            new_ops.append(op)
            continue

        if op.opcode == OpCode.MUL and len(op.operands) == 2:
            a, b = op.operands
            # x * 2 → x + x
            if b.id in constants and constants[b.id] == 2:
                new_ops.append(IROp(opcode=OpCode.ADD, result=op.result,
                                    operands=[a, a], attrs={}))
                continue
            if a.id in constants and constants[a.id] == 2:
                new_ops.append(IROp(opcode=OpCode.ADD, result=op.result,
                                    operands=[b, b], attrs={}))
                continue
            # x * 1 → x (identity via COPY)
            if b.id in constants and constants[b.id] == 1:
                new_ops.append(IROp(opcode=OpCode.COPY, result=op.result,
                                    operands=[a], attrs={}))
                continue
            if a.id in constants and constants[a.id] == 1:
                new_ops.append(IROp(opcode=OpCode.COPY, result=op.result,
                                    operands=[b], attrs={}))
                continue
            # x * 0 → 0
            if (b.id in constants and constants[b.id] == 0) or \
               (a.id in constants and constants[a.id] == 0):
                new_ops.append(IROp(opcode=OpCode.CONSTANT, result=op.result,
                                    operands=[], attrs={"value": 0}))
                continue
        elif op.opcode == OpCode.ADD and len(op.operands) == 2:
            a, b = op.operands
            # x + 0 → x
            if b.id in constants and constants[b.id] == 0:
                new_ops.append(IROp(opcode=OpCode.COPY, result=op.result,
                                    operands=[a], attrs={}))
                continue
            if a.id in constants and constants[a.id] == 0:
                new_ops.append(IROp(opcode=OpCode.COPY, result=op.result,
                                    operands=[b], attrs={}))
                continue
        elif op.opcode == OpCode.SUB and len(op.operands) == 2:
            a, b = op.operands
            # x - 0 → x
            if b.id in constants and constants[b.id] == 0:
                new_ops.append(IROp(opcode=OpCode.COPY, result=op.result,
                                    operands=[a], attrs={}))
                continue

        new_ops.append(op)

    kernel.ops = new_ops
    return kernel


def common_subexpression_eliminate(kernel: IRKernel) -> IRKernel:
    """Eliminate redundant computations — reuse existing results."""
    # Map: (opcode, operand_ids_tuple, frozen_attrs) → first result IRValue
    expr_map: dict[tuple, IRValue] = {}
    # Map: old result id → replacement IRValue
    replacements: dict[int, IRValue] = {}

    # Ops that are safe to CSE (pure, no side effects, no control flow)
    CSE_SAFE = {
        OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.MOD,
        OpCode.BIT_AND, OpCode.BIT_OR, OpCode.BIT_XOR, OpCode.LSHIFT, OpCode.RSHIFT,
        OpCode.SQRT, OpCode.EXP, OpCode.LOG, OpCode.ABS, OpCode.TANH,
        OpCode.SIN, OpCode.COS, OpCode.ASIN, OpCode.ACOS, OpCode.ATAN,
        OpCode.SINH, OpCode.COSH, OpCode.EXP2, OpCode.LOG2, OpCode.LOG10,
        OpCode.RSQRT, OpCode.CEIL, OpCode.FLOOR, OpCode.ROUND,
        OpCode.SIGMOID, OpCode.FMA, OpCode.POW, OpCode.NEG,
        OpCode.CMP_LT, OpCode.CMP_LE, OpCode.CMP_GT, OpCode.CMP_GE,
        OpCode.CMP_EQ, OpCode.CMP_NE,
        OpCode.MAX, OpCode.MIN, OpCode.CLAMP, OpCode.COPYSIGN, OpCode.FMOD, OpCode.STEP,
        OpCode.CAST, OpCode.PTR_ADD,
    }

    new_ops = []
    for op in kernel.ops:
        # Remap operands through replacements
        remapped_operands = []
        for operand in op.operands:
            if operand.id in replacements:
                remapped_operands.append(replacements[operand.id])
            else:
                remapped_operands.append(operand)
        op.operands = remapped_operands

        # Skip aliased ops — mutable accumulators need their own results
        if op.result.aliases is not None:
            new_ops.append(op)
            continue

        if op.opcode in CSE_SAFE:
            operand_ids = tuple(o.id for o in op.operands)
            frozen_attrs = tuple(sorted(op.attrs.items())) if op.attrs else ()
            key = (op.opcode, operand_ids, frozen_attrs)

            if key in expr_map:
                # This expression was already computed — replace with existing
                replacements[op.result.id] = expr_map[key]
                continue  # drop this op
            else:
                expr_map[key] = op.result

        new_ops.append(op)

    kernel.ops = new_ops
    return kernel


def dead_code_eliminate(kernel: IRKernel) -> IRKernel:
    """Remove operations whose results are never used."""
    # Collect all referenced value IDs
    used_ids: set[int] = set()

    # These ops have side effects and are always live
    SIDE_EFFECT_OPS = {
        OpCode.STORE, OpCode.SHARED_STORE, OpCode.BARRIER,
        OpCode.FOR_LOOP_START, OpCode.FOR_LOOP_END,
        OpCode.WHILE_START, OpCode.WHILE_END,
        OpCode.IF_START, OpCode.ELSE_START, OpCode.IF_END,
        OpCode.BREAK, OpCode.CONTINUE,
        OpCode.SIMDGROUP_MATRIX_STORE,
        OpCode.ATOMIC_ADD, OpCode.ATOMIC_MAX, OpCode.ATOMIC_MIN,
    }

    for op in kernel.ops:
        if op.opcode in SIDE_EFFECT_OPS:
            used_ids.add(op.result.id)
            for operand in op.operands:
                used_ids.add(operand.id)
        # Mutable accumulator updates (aliases) are implicit side effects —
        # they update a variable used in subsequent loop iterations via MSL aliasing
        if op.result.aliases is not None:
            used_ids.add(op.result.id)
            used_ids.add(op.result.aliases)  # keep the alias target alive
            for operand in op.operands:
                used_ids.add(operand.id)

    # Walk backwards, marking operands of live ops as live
    changed = True
    while changed:
        changed = False
        for op in reversed(kernel.ops):
            if op.result.id in used_ids:
                for operand in op.operands:
                    if operand.id not in used_ids:
                        used_ids.add(operand.id)
                        changed = True

    # Also keep all param IDs
    for param in kernel.params:
        used_ids.add(param.id)

    kernel.ops = [op for op in kernel.ops if op.result.id in used_ids]
    return kernel


def infer_types(kernel: IRKernel) -> IRKernel:
    """Propagate types through the IR graph."""
    type_map: dict[int, IRType] = {}

    for param in kernel.params:
        type_map[param.id] = param.dtype

    for op in kernel.ops:
        if op.opcode == OpCode.CONSTANT:
            val = op.attrs["value"]
            # Only set dtype if not already set by frontend
            if isinstance(val, float) and op.result.dtype not in (IRType.FLOAT16, IRType.FLOAT32, IRType.FLOAT64):
                op.result.dtype = IRType.FLOAT32
            elif isinstance(val, int) and not op.result.dtype.is_int:
                op.result.dtype = IRType.INT32
            elif isinstance(val, bool):
                op.result.dtype = IRType.BOOL

        elif op.opcode == OpCode.LOAD:
            # Preserve dtype set by frontend (inferred from pointer)
            pass

        elif op.opcode in (OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV):
            # Promote: if any operand is float, result is float (preserve float16 vs float32)
            if any(o.dtype.is_float for o in op.operands):
                float_dtypes = [o.dtype for o in op.operands if o.dtype.is_float]
                # Use the widest float type among operands
                if IRType.FLOAT32 in float_dtypes or IRType.FLOAT64 in float_dtypes:
                    op.result.dtype = IRType.FLOAT32
                else:
                    op.result.dtype = float_dtypes[0]  # FLOAT16
            else:
                op.result.dtype = IRType.INT32

        elif op.opcode in (OpCode.CMP_LT, OpCode.CMP_LE, OpCode.CMP_GT,
                           OpCode.CMP_GE, OpCode.CMP_EQ, OpCode.CMP_NE):
            op.result.dtype = IRType.BOOL

        elif op.opcode in (OpCode.REDUCE_SUM, OpCode.REDUCE_MAX, OpCode.REDUCE_MIN):
            if op.operands:
                op.result.dtype = op.operands[0].dtype

        type_map[op.result.id] = op.result.dtype

    return kernel
