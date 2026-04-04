"""
Locust RISC-V Backend — Compiles IR to C with RVV (RISC-V Vector Extension) intrinsics.

Generates standard C99 + <riscv_vector.h> intrinsics that can be compiled with:
  riscv64-unknown-linux-gnu-gcc -march=rv64gcv -O2 -o kernel kernel.c

The generated C uses:
- POSIX threads (pthreads) for parallel launch — each thread = one locomp block
- RVV 1.0 intrinsics for vectorized inner loops  
- __riscv_vsetvl_e32m1 / __riscv_vsetvlmax_e32m1 for vector length control
- Standard C math.h for scalar math functions

IR → RVV mapping:
  LOAD         → __riscv_vle32_v_f32m1 / vlse32 / vle16 etc.
  STORE        → __riscv_vse32_v_f32m1 etc.
  ADD          → __riscv_vfadd_vv / vadd_vv
  MUL          → __riscv_vfmul_vv / vmul_vv
  REDUCE_SUM   → __riscv_vfredosum_vs
  REDUCE_MAX   → __riscv_vfredmax_vs
  WHERE        → __riscv_vmerge_vvm
  SIMD_SUM     → scalar reduce over RVV result
  BARRIER      → pthread barrier (no-op in single-thread mode)
"""

from __future__ import annotations

from locomp.ir import IRKernel, IROp, IRType, IRValue, OpCode


# ──────────────────────────────────────────────────────────────────────────────
# Type helpers
# ──────────────────────────────────────────────────────────────────────────────

def _c_type(dtype: IRType) -> str:
    """Map IRType → C scalar type."""
    return {
        IRType.FLOAT16:  "_Float16",
        IRType.BFLOAT16: "float",        # C has no __bf16 in mainstream; promote to float
        IRType.FLOAT32:  "float",
        IRType.FLOAT64:  "double",
        IRType.INT8:     "int8_t",
        IRType.INT16:    "int16_t",
        IRType.INT32:    "int32_t",
        IRType.INT64:    "int64_t",
        IRType.UINT8:    "uint8_t",
        IRType.UINT16:   "uint16_t",
        IRType.UINT32:   "uint32_t",
        IRType.UINT64:   "uint64_t",
        IRType.BOOL:     "int",
    }[dtype]


def _rvv_scalar_type(dtype: IRType) -> str:
    """RVV element width letter: e8 / e16 / e32 / e64."""
    return {
        IRType.FLOAT16:  "e16",
        IRType.BFLOAT16: "e32",
        IRType.FLOAT32:  "e32",
        IRType.FLOAT64:  "e64",
        IRType.INT8:     "e8",
        IRType.INT16:    "e16",
        IRType.INT32:    "e32",
        IRType.INT64:    "e64",
        IRType.UINT8:    "e8",
        IRType.UINT16:   "e16",
        IRType.UINT32:   "e32",
        IRType.UINT64:   "e64",
        IRType.BOOL:     "e32",
    }[dtype]


def _rvv_vec_type(dtype: IRType) -> str:
    """RVV vector register type with LMUL=m1."""
    ewidth = _rvv_scalar_type(dtype)
    if dtype.is_float:
        suf = "f"
        bits = {"e16": "16", "e32": "32", "e64": "64"}[ewidth]
        return f"vfloat{bits}m1_t"
    else:
        signed = dtype in (IRType.INT8, IRType.INT16, IRType.INT32, IRType.INT64)
        bits = {"e8": "8", "e16": "16", "e32": "32", "e64": "64"}[ewidth]
        prefix = "vint" if signed else "vuint"
        return f"{prefix}{bits}m1_t"


def _rvv_load_fn(dtype: IRType) -> str:
    """RVV unit-stride vector load intrinsic name."""
    ewidth = _rvv_scalar_type(dtype)
    bits = ewidth[1:]   # "32" from "e32"
    if dtype.is_float:
        return f"__riscv_vle{bits}_v_f{bits}m1"
    signed = dtype in (IRType.INT8, IRType.INT16, IRType.INT32, IRType.INT64)
    sign_char = "i" if signed else "u"
    return f"__riscv_vle{bits}_v_{sign_char}{bits}m1"


def _rvv_store_fn(dtype: IRType) -> str:
    """RVV unit-stride vector store intrinsic name."""
    ewidth = _rvv_scalar_type(dtype)
    bits = ewidth[1:]
    if dtype.is_float:
        return f"__riscv_vse{bits}_v_f{bits}m1"
    signed = dtype in (IRType.INT8, IRType.INT16, IRType.INT32, IRType.INT64)
    sign_char = "i" if signed else "u"
    return f"__riscv_vse{bits}_v_{sign_char}{bits}m1"


def _rvv_add_fn(dtype: IRType) -> str:
    ewidth = _rvv_scalar_type(dtype)
    bits = ewidth[1:]
    if dtype.is_float:
        return f"__riscv_vfadd_vv_f{bits}m1"
    signed = dtype in (IRType.INT8, IRType.INT16, IRType.INT32, IRType.INT64)
    sign_char = "i" if signed else "u"
    return f"__riscv_vadd_vv_{sign_char}{bits}m1"


def _rvv_sub_fn(dtype: IRType) -> str:
    ewidth = _rvv_scalar_type(dtype)
    bits = ewidth[1:]
    if dtype.is_float:
        return f"__riscv_vfsub_vv_f{bits}m1"
    signed = dtype in (IRType.INT8, IRType.INT16, IRType.INT32, IRType.INT64)
    sign_char = "i" if signed else "u"
    return f"__riscv_vsub_vv_{sign_char}{bits}m1"


def _rvv_mul_fn(dtype: IRType) -> str:
    ewidth = _rvv_scalar_type(dtype)
    bits = ewidth[1:]
    if dtype.is_float:
        return f"__riscv_vfmul_vv_f{bits}m1"
    signed = dtype in (IRType.INT8, IRType.INT16, IRType.INT32, IRType.INT64)
    sign_char = "i" if signed else "u"
    return f"__riscv_vmul_vv_{sign_char}{bits}m1"


def _rvv_div_fn(dtype: IRType) -> str:
    ewidth = _rvv_scalar_type(dtype)
    bits = ewidth[1:]
    if dtype.is_float:
        return f"__riscv_vfdiv_vv_f{bits}m1"
    signed = dtype in (IRType.INT8, IRType.INT16, IRType.INT32, IRType.INT64)
    sign_char = "i" if signed else "u"
    return f"__riscv_v{'div' if signed else 'divu'}_vv_{sign_char}{bits}m1"


# ──────────────────────────────────────────────────────────────────────────────
# Codegen class
# ──────────────────────────────────────────────────────────────────────────────

class RISCVCodegen:
    """Generates C + RVV intrinsics from Locust IR.

    The output is a self-contained .c file that:
    1. Declares all kernel parameters as typed pointers / scalars
    2. Uses POSIX threads to parallelise across grid blocks
    3. Uses RVV intrinsics inside each block for vectorised inner loops
    """

    def __init__(self, kernel: IRKernel,
                 constexpr_values: dict[str, int | float] | None = None):
        self.kernel = kernel
        self.indent = "    "
        self._var_names: dict[int, str] = {}
        # constexpr_values may be keyed by base name ("N") OR ssa name ("N_2").
        # Normalise: expand to include both forms so lookup always works.
        raw = constexpr_values or {}
        self._constexpr_values: dict[str, int | float] = {}
        for k, v in raw.items():
            self._constexpr_values[k] = v
            # also register under ssa-name if key is already a base name
        self._param_names: list[str] = []   # ordered list of param variable names
        self._ptr_exprs: dict[int, tuple[str, str]] = {}  # value_id → (base, index)
        self._nesting: int = 0
        # Build base-name lookup: strip trailing _<int> suffix from SSA param names
        self._base_names: dict[int, str] = {}   # param.id → base_name
        for p in kernel.params:
            base = p.name.rsplit("_", 1)[0] if "_" in p.name else p.name
            self._base_names[p.id] = base
            # register ssa→value and base→value in constexpr lookup
            if base in raw:
                self._constexpr_values[p.name] = raw[base]  # ssa name → value
            elif p.name in raw:
                self._constexpr_values[base] = raw[p.name]  # base name → value

    # ── public entry point ────────────────────────────────────────────────────

    def generate(self) -> tuple[str, dict[str, int]]:
        """Return (c_source, param_order_map)."""
        param_map: dict[str, int] = {}
        idx = 0
        for p in self.kernel.params:
            base = self._base_names[p.id]
            # A param is constexpr if its base name OR ssa name is in the dict
            is_constexpr = (base in self._constexpr_values or
                            p.name in self._constexpr_values)
            if is_constexpr:
                val = (self._constexpr_values.get(base)
                       or self._constexpr_values.get(p.name))
                self._var_names[p.id] = str(val)
            else:
                # Use base name as the C variable name (clean, no _0 suffix)
                self._var_names[p.id] = base
                param_map[base] = idx
                idx += 1
            self._param_names.append(self._var_names[p.id])

        src = self._build_source(param_map)
        return src, param_map

    # ── source builder ────────────────────────────────────────────────────────

    def _build_source(self, param_map: dict[str, int]) -> str:
        lines: list[str] = []

        # Headers
        lines += [
            "/* Generated by locomp — RISC-V RVV backend */",
            "#include <stdint.h>",
            "#include <stddef.h>",
            "#include <math.h>",
            "#include <string.h>",
            "#include <riscv_vector.h>",
            "#include <pthread.h>",
            "",
        ]

        # Struct to pass args to each pthread worker
        lines += self._gen_args_struct()
        lines.append("")

        # The inner kernel function (runs one grid block)
        lines += self._gen_kernel_fn()
        lines.append("")

        # pthread worker shim
        lines += self._gen_pthread_worker()
        lines.append("")

        # Public launch function
        lines += self._gen_launch_fn(param_map)

        return "\n".join(lines)

    # ── args struct ───────────────────────────────────────────────────────────

    def _gen_args_struct(self) -> list[str]:
        lines = [f"typedef struct {{", f"    int _block_id;"]
        for p in self.kernel.params:
            base = self._base_names[p.id]
            is_constexpr = (base in self._constexpr_values or
                            p.name in self._constexpr_values)
            if is_constexpr:
                continue
            ctype = _c_type(p.dtype)
            if p.is_pointer:
                lines.append(f"    {ctype}* {base};")
            else:
                lines.append(f"    {ctype} {base};")
        lines.append(f"}} {self.kernel.name}_args_t;")
        return lines

    # ── kernel function ───────────────────────────────────────────────────────

    def _gen_kernel_fn(self) -> list[str]:
        lines = [f"static void {self.kernel.name}_block("]
        # params
        sig_parts = ["int _block_id"]
        for p in self.kernel.params:
            base = self._base_names[p.id]
            is_constexpr = (base in self._constexpr_values or
                            p.name in self._constexpr_values)
            if is_constexpr:
                continue
            ctype = _c_type(p.dtype)
            if p.is_pointer:
                sig_parts.append(f"{ctype}* {base}")
            else:
                sig_parts.append(f"{ctype} {base}")
        lines.append("    " + ", ".join(sig_parts))
        lines.append(") {")
        lines.extend(self._gen_body())
        lines.append("}")
        return lines

    def _gen_body(self) -> list[str]:
        lines: list[str] = []
        self._nesting = 0

        # pre-declare variables defined inside nested scopes
        self._predeclared: set[int] = set()
        depth = 0
        defined_at_depth: dict[int, int] = {}
        NO_RESULT = {OpCode.STORE, OpCode.SHARED_STORE, OpCode.BARRIER,
                     OpCode.FOR_LOOP_END, OpCode.IF_END, OpCode.WHILE_END,
                     OpCode.ELSE_START, OpCode.BREAK, OpCode.CONTINUE,
                     OpCode.ATOMIC_ADD, OpCode.ATOMIC_MAX, OpCode.ATOMIC_MIN}
        for op in self.kernel.ops:
            if op.opcode in (OpCode.FOR_LOOP_END, OpCode.IF_END, OpCode.WHILE_END):
                depth -= 1
            if op.opcode not in NO_RESULT and op.result is not None:
                defined_at_depth[op.result.id] = depth
            if op.opcode in (OpCode.FOR_LOOP_START, OpCode.IF_START, OpCode.WHILE_START):
                depth += 1

        referenced: set[int] = set()
        for op in self.kernel.ops:
            for o in op.operands:
                referenced.add(o.id)

        for op in self.kernel.ops:
            if op.result is None:
                continue
            rid = op.result.id
            if (defined_at_depth.get(rid, 0) > 0
                    and rid in referenced
                    and op.result.aliases is None
                    and op.opcode not in (OpCode.FOR_LOOP_START, OpCode.CONSTANT)
                    and not op.result.is_pointer          # never pre-declare pointers
                    and op.opcode != OpCode.PTR_ADD):     # PTR_ADD result is a pointer
                var = self._vname(op.result)
                ctype = _c_type(op.result.dtype)
                lines.append(f"{self.indent}{ctype} {var};")
                self._predeclared.add(rid)

        if self._predeclared:
            lines.append("")

        for op in self.kernel.ops:
            if op.opcode in (OpCode.FOR_LOOP_END, OpCode.IF_END, OpCode.WHILE_END):
                self._nesting -= 1
                lines.append(self.indent * (1 + self._nesting) + "}")
                continue
            if op.opcode == OpCode.ELSE_START:
                self._nesting -= 1
                lines.append(self.indent * (1 + self._nesting) + "} else {")
                self._nesting += 1
                continue

            ind = self.indent * (1 + self._nesting)
            emitted = self._gen_op(op)
            if emitted is None:
                pass
            elif isinstance(emitted, list):
                for l in emitted:
                    full = f"{ind}{l}"
                    if op.result is not None:
                        full = self._strip_type(full, op.result.id)
                    lines.append(full)
            else:
                full = f"{ind}{emitted}"
                if op.result is not None:
                    full = self._strip_type(full, op.result.id)
                lines.append(full)

            if op.opcode in (OpCode.FOR_LOOP_START, OpCode.IF_START, OpCode.WHILE_START):
                self._nesting += 1

        return lines

    # ── pthread worker ────────────────────────────────────────────────────────

    def _gen_pthread_worker(self) -> list[str]:
        lines = [
            f"static void* {self.kernel.name}_worker(void* _arg) {{",
            f"    {self.kernel.name}_args_t* _a = ({self.kernel.name}_args_t*)_arg;",
        ]
        call_args = ["_a->_block_id"]
        for p in self.kernel.params:
            base = self._base_names[p.id]
            is_constexpr = (base in self._constexpr_values or
                            p.name in self._constexpr_values)
            if not is_constexpr:
                call_args.append(f"_a->{base}")
        lines.append(f"    {self.kernel.name}_block({', '.join(call_args)});")
        lines.append("    return NULL;")
        lines.append("}")
        return lines

    # ── launch function ───────────────────────────────────────────────────────

    def _gen_launch_fn(self, param_map: dict[str, int]) -> list[str]:
        """Public launch: locomp_launch_<name>(int grid_size, void** args)"""
        lines = [
            f"void locomp_launch_{self.kernel.name}(int _grid_size, void** _args) {{",
        ]
        # unpack void** args into typed pointers
        idx = 0
        for p in self.kernel.params:
            base = self._base_names[p.id]
            is_constexpr = (base in self._constexpr_values or
                            p.name in self._constexpr_values)
            if is_constexpr:
                continue
            ctype = _c_type(p.dtype)
            if p.is_pointer:
                lines.append(f"    {ctype}* {base} = ({ctype}*)_args[{idx}];")
            else:
                lines.append(f"    {ctype} {base} = *({ctype}*)_args[{idx}];")
            idx += 1

        lines += [
            f"    pthread_t* _threads = (pthread_t*)__builtin_alloca(sizeof(pthread_t) * _grid_size);",
            f"    {self.kernel.name}_args_t* _arg_arr = ({self.kernel.name}_args_t*)__builtin_alloca(sizeof({self.kernel.name}_args_t) * _grid_size);",
            f"    for (int _b = 0; _b < _grid_size; _b++) {{",
            f"        _arg_arr[_b]._block_id = _b;",
        ]
        for p in self.kernel.params:
            base = self._base_names[p.id]
            is_constexpr = (base in self._constexpr_values or
                            p.name in self._constexpr_values)
            if not is_constexpr:
                lines.append(f"        _arg_arr[_b].{base} = {base};")
        lines += [
            f"        pthread_create(&_threads[_b], NULL, {self.kernel.name}_worker, &_arg_arr[_b]);",
            f"    }}",
            f"    for (int _b = 0; _b < _grid_size; _b++) {{",
            f"        pthread_join(_threads[_b], NULL);",
            f"    }}",
            "}",
        ]
        return lines

    # ── op codegen ────────────────────────────────────────────────────────────

    def _gen_op(self, op: IROp) -> str | list[str] | None:
        rv = self._vname(op.result)

        # ── thread indexing ──────────────────────────────────────────────────
        if op.opcode == OpCode.PROGRAM_ID:
            return f"int {rv} = _block_id;"

        elif op.opcode == OpCode.THREAD_ID:
            return f"int {rv} = _block_id;"

        elif op.opcode == OpCode.LOCAL_ID:
            return f"int {rv} = 0;"

        elif op.opcode == OpCode.GROUP_SIZE:
            return f"int {rv} = 1;"

        elif op.opcode == OpCode.NUM_GROUPS:
            return f"int {rv} = 0;  /* set at launch */"

        # ── arange ──────────────────────────────────────────────────────────
        elif op.opcode == OpCode.ARANGE:
            start = op.attrs["start"]
            end = op.attrs["end"]
            size = end - start
            ct = _c_type(op.result.dtype)
            return [
                f"{ct} {rv}[{size}];",
                f"for (int _i = 0; _i < {size}; _i++) {{ {rv}[_i] = {start} + _i; }}",
            ]

        # ── constants ────────────────────────────────────────────────────────
        elif op.opcode == OpCode.CONSTANT:
            if "shared_mem" in op.attrs:
                smem_name = op.attrs["shared_mem"]
                self._var_names[op.result.id] = smem_name
                return None
            val = op.attrs["value"]
            ct = _c_type(op.result.dtype)
            if isinstance(val, bool):
                return f"{ct} {rv} = {1 if val else 0};"
            elif isinstance(val, float):
                return f"{ct} {rv} = {val}f;"
            else:
                return f"{ct} {rv} = {val};"

        elif op.opcode == OpCode.CONSTEXPR:
            val = op.attrs.get("value", 0)
            return f"int {rv} = {val};"

        # ── pointer arithmetic ───────────────────────────────────────────────
        elif op.opcode == OpCode.PTR_ADD:
            base = self._vname(op.operands[0])
            offset = self._vname(op.operands[1])
            # Track for later load/store resolution
            self._ptr_exprs[op.result.id] = (base, offset)
            ct = _c_type(op.result.dtype)
            return f"{ct}* {rv} = {base} + {offset};"
        # ── memory ───────────────────────────────────────────────────────────
        elif op.opcode == OpCode.LOAD:
            return self._gen_load(op)

        elif op.opcode == OpCode.STORE:
            return self._gen_store(op)

        elif op.opcode == OpCode.SHARED_LOAD:
            arr = self._vname(op.operands[0])
            idx = self._vname(op.operands[1])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = {arr}[{idx}];"

        elif op.opcode == OpCode.SHARED_STORE:
            arr = self._vname(op.operands[0])
            idx = self._vname(op.operands[1])
            val = self._vname(op.operands[2])
            return f"{arr}[{idx}] = {val};"

        elif op.opcode == OpCode.BARRIER:
            return "/* barrier — pthread implicit via join */;"

        # ── arithmetic ───────────────────────────────────────────────────────
        elif op.opcode in (OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.MOD):
            return self._gen_arith_binop(op)

        elif op.opcode in (OpCode.BIT_AND, OpCode.BIT_OR, OpCode.BIT_XOR,
                           OpCode.LSHIFT, OpCode.RSHIFT):
            sym = {OpCode.BIT_AND: "&", OpCode.BIT_OR: "|", OpCode.BIT_XOR: "^",
                   OpCode.LSHIFT: "<<", OpCode.RSHIFT: ">>"}[op.opcode]
            a = self._vname(op.operands[0])
            b = self._vname(op.operands[1])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = {a} {sym} {b};"

        elif op.opcode == OpCode.NEG:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = -{a};"

        # ── math functions ────────────────────────────────────────────────────
        elif op.opcode in (OpCode.SQRT, OpCode.EXP, OpCode.LOG, OpCode.ABS,
                           OpCode.TANH, OpCode.SIN, OpCode.COS,
                           OpCode.ASIN, OpCode.ACOS, OpCode.ATAN,
                           OpCode.SINH, OpCode.COSH,
                           OpCode.EXP2, OpCode.LOG2, OpCode.LOG10,
                           OpCode.RSQRT, OpCode.CEIL, OpCode.FLOOR, OpCode.ROUND):
            fn = {
                OpCode.SQRT: "sqrtf", OpCode.EXP: "expf", OpCode.LOG: "logf",
                OpCode.ABS: "fabsf", OpCode.TANH: "tanhf", OpCode.SIN: "sinf",
                OpCode.COS: "cosf", OpCode.ASIN: "asinf", OpCode.ACOS: "acosf",
                OpCode.ATAN: "atanf", OpCode.SINH: "sinhf", OpCode.COSH: "coshf",
                OpCode.EXP2: "exp2f", OpCode.LOG2: "log2f", OpCode.LOG10: "log10f",
                OpCode.RSQRT: "rsqrtf", OpCode.CEIL: "ceilf", OpCode.FLOOR: "floorf",
                OpCode.ROUND: "roundf",
            }[op.opcode]
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            if op.result.shape:
                size = op.result.shape[0]
                return [
                    f"{ct} {rv}[{size}];",
                    f"for (int _i = 0; _i < {size}; _i++) {{ {rv}[_i] = {fn}({a}[_i]); }}",
                ]
            return f"{ct} {rv} = {fn}({a});"

        elif op.opcode == OpCode.SIGMOID:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = 1.0f / (1.0f + expf(-{a}));"

        elif op.opcode in (OpCode.POW, OpCode.ATAN2, OpCode.COPYSIGN,
                           OpCode.FMOD, OpCode.STEP, OpCode.MAX, OpCode.MIN):
            fn = {OpCode.POW: "powf", OpCode.ATAN2: "atan2f",
                  OpCode.COPYSIGN: "copysignf", OpCode.FMOD: "fmodf",
                  OpCode.STEP: None,
                  OpCode.MAX: "fmaxf", OpCode.MIN: "fminf"}[op.opcode]
            a = self._vname(op.operands[0])
            b = self._vname(op.operands[1])
            ct = _c_type(op.result.dtype)
            if op.opcode == OpCode.STEP:
                return f"{ct} {rv} = ({a} < {b}) ? 0.0f : 1.0f;"
            return f"{ct} {rv} = {fn}({a}, {b});"

        elif op.opcode == OpCode.FMA:
            a = self._vname(op.operands[0])
            b = self._vname(op.operands[1])
            c = self._vname(op.operands[2])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = fmaf({a}, {b}, {c});"

        elif op.opcode == OpCode.CLAMP:
            a = self._vname(op.operands[0])
            lo = self._vname(op.operands[1])
            hi = self._vname(op.operands[2])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = fmaxf({lo}, fminf({hi}, {a}));"

        # ── comparisons ───────────────────────────────────────────────────────
        elif op.opcode in (OpCode.CMP_LT, OpCode.CMP_LE, OpCode.CMP_GT,
                           OpCode.CMP_GE, OpCode.CMP_EQ, OpCode.CMP_NE):
            sym = {OpCode.CMP_LT: "<", OpCode.CMP_LE: "<=", OpCode.CMP_GT: ">",
                   OpCode.CMP_GE: ">=", OpCode.CMP_EQ: "==", OpCode.CMP_NE: "!="}[op.opcode]
            a = self._vname(op.operands[0])
            b = self._vname(op.operands[1])
            if op.result.shape:
                size = op.result.shape[0]
                a_tile = bool(op.operands[0].shape)
                b_tile = bool(op.operands[1].shape)
                return [
                    f"int {rv}[{size}];",
                    f"for (int _i = 0; _i < {size}; _i++) {{",
                    f"    {rv}[_i] = ({a if not a_tile else a+'[_i]'} {sym} {b if not b_tile else b+'[_i]'}) ? 1 : 0;",
                    "}",
                ]
            return f"int {rv} = ({a} {sym} {b}) ? 1 : 0;"

        # ── reductions ────────────────────────────────────────────────────────
        elif op.opcode == OpCode.REDUCE_SUM:
            return self._gen_reduce(op, "sum")

        elif op.opcode == OpCode.REDUCE_MAX:
            return self._gen_reduce(op, "max")

        elif op.opcode == OpCode.REDUCE_MIN:
            return self._gen_reduce(op, "min")

        # ── SIMD (map to scalar on RISC-V CPU — RVV handles vectorisation) ───
        elif op.opcode == OpCode.SIMD_SUM:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = {a};  /* simd_sum → scalar passthrough on RISC-V */"

        elif op.opcode == OpCode.SIMD_MAX:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = {a};"

        elif op.opcode == OpCode.SIMD_MIN:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = {a};"

        elif op.opcode == OpCode.SIMD_BROADCAST:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = {a};"

        elif op.opcode == OpCode.SIMD_SHUFFLE_DOWN:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = {a};"

        elif op.opcode == OpCode.SIMD_LANE_ID:
            return f"int {rv} = 0;"

        elif op.opcode == OpCode.SIMD_GROUP_ID:
            return f"int {rv} = 0;"

        # ── cast ──────────────────────────────────────────────────────────────
        elif op.opcode == OpCode.CAST:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = ({ct}){a};"

        # ── where / select ────────────────────────────────────────────────────
        elif op.opcode == OpCode.WHERE:
            return self._gen_where(op)

        # ── copy ──────────────────────────────────────────────────────────────
        elif op.opcode == OpCode.COPY:
            a = self._vname(op.operands[0])
            if op.result.aliases is not None:
                return f"{rv} = {a};"
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = {a};"

        # ── atomics ───────────────────────────────────────────────────────────
        elif op.opcode in (OpCode.ATOMIC_ADD, OpCode.ATOMIC_MAX, OpCode.ATOMIC_MIN):
            return self._gen_atomic(op)

        # ── control flow ─────────────────────────────────────────────────────
        elif op.opcode == OpCode.FOR_LOOP_START:
            start = self._vname(op.operands[0])
            end = self._vname(op.operands[1])
            step = self._vname(op.operands[2])
            loop_var = self._vname(op.result)
            return f"for (int {loop_var} = {start}; {loop_var} < {end}; {loop_var} += {step}) {{"

        elif op.opcode == OpCode.WHILE_START:
            return "while (1) {"

        elif op.opcode == OpCode.IF_START:
            cond = self._vname(op.operands[0])
            return f"if ({cond}) {{"

        elif op.opcode in (OpCode.FOR_LOOP_END, OpCode.IF_END, OpCode.WHILE_END,
                           OpCode.ELSE_START):
            return None  # handled in _gen_body

        elif op.opcode == OpCode.BREAK:
            return "break;"

        elif op.opcode == OpCode.CONTINUE:
            return "continue;"

        # ── simdgroup matrix — not applicable on RISC-V, emit scalar fallback ─
        elif op.opcode in (OpCode.SIMDGROUP_MATRIX_LOAD, OpCode.SIMDGROUP_MATRIX_STORE,
                           OpCode.SIMDGROUP_MATRIX_MAC, OpCode.SIMDGROUP_MATRIX_FILL):
            return f"/* simdgroup matrix op not supported on RISC-V */"

        else:
            return f"/* unhandled op: {op.opcode.name} */"

    # ── load / store helpers ─────────────────────────────────────────────────

    def _gen_load(self, op: IROp) -> str | list[str]:
        rv = self._vname(op.result)
        ptr_val = op.operands[0]
        ct = _c_type(op.result.dtype)

        # Resolve pointer expression
        if ptr_val.id in self._ptr_exprs:
            base, idx = self._ptr_exprs[ptr_val.id]
            ptr_expr = f"({base} + {idx})"
        elif ptr_val.is_pointer:
            ptr_expr = self._vname(ptr_val)  # already a float* variable
        else:
            ptr_expr = self._vname(ptr_val)

        # Tiled load → use RVV vector load
        if op.result.shape:
            size = op.result.shape[0]
            vtype = _rvv_vec_type(op.result.dtype)
            load_fn = _rvv_load_fn(op.result.dtype)
            return [
                f"{ct} {rv}[{size}];",
                f"{{",
                f"    size_t _vl;",
                f"    size_t _rem = {size};",
                f"    {ct}* _src = {ptr_expr};",
                f"    {ct}* _dst = {rv};",
                f"    while (_rem > 0) {{",
                f"        _vl = __riscv_vsetvl_e{op.result.dtype.bytewidth * 8}m1(_rem);",
                f"        {vtype} _vdata = {load_fn}(_src, _vl);",
                f"        {_rvv_store_fn(op.result.dtype)}(_dst, _vdata, _vl);",
                f"        _src += _vl; _dst += _vl; _rem -= _vl;",
                f"    }}",
                f"}}",
            ]

        # Scalar load
        mask = op.attrs.get("mask")
        other = op.attrs.get("other")
        if mask is not None:
            mask_var = self._vname(mask) if isinstance(mask, IRValue) else str(mask)
            other_val = self._vname(other) if isinstance(other, IRValue) else str(other)
            return [
                f"{ct} {rv};",
                f"if ({mask_var}) {{ {rv} = *{ptr_expr}; }}",
                f"else {{ {rv} = ({ct}){other_val}; }}",
            ]
        return f"{ct} {rv} = *{ptr_expr};"

    def _gen_store(self, op: IROp) -> str | list[str]:
        ptr_val = op.operands[0]
        val_val = op.operands[1]

        if ptr_val.id in self._ptr_exprs:
            base, idx = self._ptr_exprs[ptr_val.id]
            ptr_expr = f"({base} + {idx})"
        else:
            ptr_expr = self._vname(ptr_val)

        val = self._vname(val_val)

        # Tiled store → RVV vector store
        if val_val.shape:
            size = val_val.shape[0]
            ct = _c_type(val_val.dtype)
            vtype = _rvv_vec_type(val_val.dtype)
            store_fn = _rvv_store_fn(val_val.dtype)
            load_fn = _rvv_load_fn(val_val.dtype)
            return [
                f"{{",
                f"    size_t _vl;",
                f"    size_t _rem = {size};",
                f"    {ct}* _src = {val};",
                f"    {ct}* _dst = {ptr_expr};",
                f"    while (_rem > 0) {{",
                f"        _vl = __riscv_vsetvl_e{val_val.dtype.bytewidth * 8}m1(_rem);",
                f"        {vtype} _vdata = {load_fn}(_src, _vl);",
                f"        {store_fn}(_dst, _vdata, _vl);",
                f"        _src += _vl; _dst += _vl; _rem -= _vl;",
                f"    }}",
                f"}}",
            ]

        # Scalar store
        mask = op.attrs.get("mask")
        if mask is not None:
            mask_var = self._vname(mask) if isinstance(mask, IRValue) else str(mask)
            return [f"if ({mask_var}) {{ *{ptr_expr} = {val}; }}"]
        return f"*{ptr_expr} = {val};"

    # ── arithmetic helpers ───────────────────────────────────────────────────

    def _gen_arith_binop(self, op: IROp) -> str | list[str]:
        rv = self._vname(op.result)
        a = self._vname(op.operands[0])
        b = self._vname(op.operands[1])
        ct = _c_type(op.result.dtype)

        # Pointer arithmetic: if either operand is a pointer, result is a pointer
        a_ptr = op.operands[0].is_pointer or op.operands[0].id in self._ptr_exprs
        b_ptr = op.operands[1].is_pointer or op.operands[1].id in self._ptr_exprs
        if (op.opcode == OpCode.ADD) and (a_ptr or b_ptr):
            self._ptr_exprs[op.result.id] = (a if a_ptr else b,
                                             b if a_ptr else a)
            return f"{ct}* {rv} = {a} + {b};"

        sym_map = {OpCode.ADD: "+", OpCode.SUB: "-", OpCode.MUL: "*",
                   OpCode.DIV: "/", OpCode.MOD: "%"}
        sym = sym_map[op.opcode]

        # Tiled op → RVV vector arithmetic
        if op.result.shape:
            size = op.result.shape[0]
            a_tile = bool(op.operands[0].shape)
            b_tile = bool(op.operands[1].shape)
            vtype = _rvv_vec_type(op.result.dtype)
            ewidth_bits = op.result.dtype.bytewidth * 8

            # Choose RVV intrinsic
            rvv_fn = {
                OpCode.ADD: _rvv_add_fn(op.result.dtype),
                OpCode.SUB: _rvv_sub_fn(op.result.dtype),
                OpCode.MUL: _rvv_mul_fn(op.result.dtype),
                OpCode.DIV: _rvv_div_fn(op.result.dtype),
                OpCode.MOD: None,   # no RVV fmod — fall back to scalar loop
            }[op.opcode]

            if rvv_fn is None or (not a_tile and not b_tile):
                # Scalar fallback loop
                return [
                    f"{ct} {rv}[{size}];",
                    f"for (int _i = 0; _i < {size}; _i++) {{",
                    f"    {rv}[_i] = {a if not a_tile else a+'[_i]'} {sym} {b if not b_tile else b+'[_i]'};",
                    "}",
                ]

            # RVV vectorised path
            lines = [f"{ct} {rv}[{size}];", "{"]
            lines += [
                f"    size_t _vl;",
                f"    size_t _rem = {size};",
                f"    int _off = 0;",
                f"    while (_rem > 0) {{",
                f"        _vl = __riscv_vsetvl_e{ewidth_bits}m1(_rem);",
            ]
            load_fn = _rvv_load_fn(op.result.dtype)
            store_fn = _rvv_store_fn(op.result.dtype)
            if a_tile:
                lines.append(f"        {vtype} _va = {load_fn}({a} + _off, _vl);")
                va = "_va"
            else:
                # broadcast scalar — use vfmv intrinsic
                if op.result.dtype.is_float:
                    bits = ewidth_bits
                    lines.append(f"        {vtype} _va = __riscv_vfmv_v_f_f{bits}m1({a}, _vl);")
                else:
                    bits = ewidth_bits
                    signed = op.result.dtype in (IRType.INT8, IRType.INT16, IRType.INT32, IRType.INT64)
                    sc = "i" if signed else "u"
                    lines.append(f"        {vtype} _va = __riscv_vmv_v_x_{sc}{bits}m1({a}, _vl);")
                va = "_va"
            if b_tile:
                lines.append(f"        {vtype} _vb = {load_fn}({b} + _off, _vl);")
                vb = "_vb"
            else:
                if op.result.dtype.is_float:
                    bits = ewidth_bits
                    lines.append(f"        {vtype} _vb = __riscv_vfmv_v_f_f{bits}m1({b}, _vl);")
                else:
                    bits = ewidth_bits
                    signed = op.result.dtype in (IRType.INT8, IRType.INT16, IRType.INT32, IRType.INT64)
                    sc = "i" if signed else "u"
                    lines.append(f"        {vtype} _vb = __riscv_vmv_v_x_{sc}{bits}m1({b}, _vl);")
                vb = "_vb"
            lines += [
                f"        {vtype} _vr = {rvv_fn}({va}, {vb}, _vl);",
                f"        {store_fn}({rv} + _off, _vr, _vl);",
                f"        _off += _vl; _rem -= _vl;",
                f"    }}",
                "}",
            ]
            return lines

        # Scalar arithmetic
        return f"{ct} {rv} = {a} {sym} {b};"

    # ── reduce helpers ───────────────────────────────────────────────────────

    def _gen_reduce(self, op: IROp, kind: str) -> str | list[str]:
        rv = self._vname(op.result)
        arg = self._vname(op.operands[0])
        ct = _c_type(op.result.dtype)

        # Tiled reduce → RVV vredsum / vredmax / vredmin
        if op.operands[0].shape:
            size = op.operands[0].shape[0]
            ewidth_bits = op.result.dtype.bytewidth * 8
            vtype = _rvv_vec_type(op.result.dtype)
            load_fn = _rvv_load_fn(op.result.dtype)

            if op.result.dtype.is_float:
                bits = ewidth_bits
                init = {"sum": f"__riscv_vfmv_v_f_f{bits}m1(0.0f, 1)",
                        "max": f"__riscv_vfmv_v_f_f{bits}m1(-1e38f, 1)",
                        "min": f"__riscv_vfmv_v_f_f{bits}m1(1e38f, 1)"}[kind]
                rvv_red = {"sum": f"__riscv_vfredosum_vs_f{bits}m1_f{bits}m1",
                           "max": f"__riscv_vfredmax_vs_f{bits}m1_f{bits}m1",
                           "min": f"__riscv_vfredmin_vs_f{bits}m1_f{bits}m1"}[kind]
                ext_fn = f"__riscv_vfmv_f_s_f{bits}m1_f{bits}"
            else:
                signed = op.result.dtype in (IRType.INT8, IRType.INT16, IRType.INT32, IRType.INT64)
                sc = "i" if signed else "u"
                bits = ewidth_bits
                init = {"sum": f"__riscv_vmv_v_x_{sc}{bits}m1(0, 1)",
                        "max": f"__riscv_vmv_v_x_{sc}{bits}m1(-2147483648, 1)",
                        "min": f"__riscv_vmv_v_x_{sc}{bits}m1(2147483647, 1)"}[kind]
                rvv_red = {"sum": f"__riscv_vredsum_vs_{sc}{bits}m1_{sc}{bits}m1",
                           "max": f"__riscv_vredmax_vs_{sc}{bits}m1_{sc}{bits}m1" if signed else f"__riscv_vredmaxu_vs_{sc}{bits}m1_{sc}{bits}m1",
                           "min": f"__riscv_vredmin_vs_{sc}{bits}m1_{sc}{bits}m1" if signed else f"__riscv_vredminu_vs_{sc}{bits}m1_{sc}{bits}m1"}[kind]
                ext_fn = f"__riscv_vmv_x_s_{sc}{bits}m1_{sc}{bits}"

            return [
                f"{ct} {rv};",
                f"{{",
                f"    {vtype} _acc = {init};",
                f"    size_t _vl;",
                f"    size_t _rem = {size};",
                f"    int _off = 0;",
                f"    while (_rem > 0) {{",
                f"        _vl = __riscv_vsetvl_e{ewidth_bits}m1(_rem);",
                f"        {vtype} _vd = {load_fn}({arg} + _off, _vl);",
                f"        _acc = {rvv_red}(_vd, _acc, _vl);",
                f"        _off += _vl; _rem -= _vl;",
                f"    }}",
                f"    {rv} = {ext_fn}(_acc);",
                f"}}",
            ]

        # Scalar passthrough
        return f"{ct} {rv} = {arg};"

    # ── where helper ─────────────────────────────────────────────────────────

    def _gen_where(self, op: IROp) -> str | list[str]:
        rv = self._vname(op.result)
        cond = self._vname(op.operands[0])
        a = self._vname(op.operands[1])
        b = self._vname(op.operands[2])
        ct = _c_type(op.result.dtype)

        if op.result.shape:
            size = op.result.shape[0]
            cond_tile = bool(op.operands[0].shape)
            a_tile = bool(op.operands[1].shape)
            b_tile = bool(op.operands[2].shape)
            return [
                f"{ct} {rv}[{size}];",
                f"for (int _i = 0; _i < {size}; _i++) {{",
                f"    {rv}[_i] = ({cond if not cond_tile else cond+'[_i]'}) ? "
                f"({a if not a_tile else a+'[_i]'}) : ({b if not b_tile else b+'[_i]'});",
                "}",
            ]
        return f"{ct} {rv} = {cond} ? {a} : {b};"

    # ── atomic helpers ───────────────────────────────────────────────────────

    def _gen_atomic(self, op: IROp) -> str | list[str]:
        # RISC-V has AMO (atomic memory operations) — use GCC __sync builtins
        ptr_val = op.operands[0]
        val_val = op.operands[1]
        if ptr_val.id in self._ptr_exprs:
            base, idx = self._ptr_exprs[ptr_val.id]
            ptr_expr = f"({base} + {idx})"
        else:
            ptr_expr = self._vname(ptr_val)
        val = self._vname(val_val)

        if op.opcode == OpCode.ATOMIC_ADD:
            return f"__sync_fetch_and_add({ptr_expr}, {val});"
        elif op.opcode == OpCode.ATOMIC_MAX:
            # No hardware AMO for max on float — CAS loop
            ct = _c_type(op.operands[1].dtype)
            return [
                f"{{ {ct} _old, _new;",
                f"  do {{ _old = *{ptr_expr};",
                f"       _new = (_old > {val}) ? _old : {val};",
                f"  }} while (!__sync_bool_compare_and_swap({ptr_expr}, _old, _new)); }}",
            ]
        elif op.opcode == OpCode.ATOMIC_MIN:
            ct = _c_type(op.operands[1].dtype)
            return [
                f"{{ {ct} _old, _new;",
                f"  do {{ _old = *{ptr_expr};",
                f"       _new = (_old < {val}) ? _old : {val};",
                f"  }} while (!__sync_bool_compare_and_swap({ptr_expr}, _old, _new)); }}",
            ]
        return f"/* atomic op unhandled */"

    # ── variable name helpers ────────────────────────────────────────────────

    def _vname(self, value: IRValue) -> str:
        if value.aliases is not None and value.aliases in self._var_names:
            self._var_names[value.id] = self._var_names[value.aliases]
            return self._var_names[value.id]
        if value.id not in self._var_names:
            base = value.name.replace(".", "_")
            self._var_names[value.id] = base
        return self._var_names[value.id]

    def _strip_type(self, line: str, result_id: int) -> str:
        if result_id not in self._predeclared:
            return line
        stripped = line.lstrip()
        # Never strip pointer declarations — they can't be pre-declared as scalars
        if '* ' in stripped.split('=')[0] or stripped.split('=')[0].rstrip().endswith('*'):
            return line
        for prefix in ("float ", "int ", "double ", "int8_t ", "int16_t ",
                       "int32_t ", "int64_t ", "uint8_t ", "uint16_t ",
                       "uint32_t ", "uint64_t ", "_Float16 "):
            if stripped.startswith(prefix):
                indent = line[:len(line) - len(stripped)]
                return indent + stripped[len(prefix):]
        return line


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point (mirrors compile_to_metal signature)
# ──────────────────────────────────────────────────────────────────────────────

def compile_to_riscv(kernel: IRKernel,
                     constexpr_values: dict[str, int | float] | None = None
                     ) -> tuple[str, dict[str, int]]:
    """Compile IR kernel to C + RVV source.

    Returns:
        (c_source, param_map)  — param_map: {param_name → arg_index}
    """
    cg = RISCVCodegen(kernel, constexpr_values=constexpr_values)
    return cg.generate()
