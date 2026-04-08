"""
locomp ROCm Backend — Compiles IR to HIP C kernels for AMD GPUs.

HIP (Heterogeneous-compute Interface for Portability) is AMD's CUDA-compatible
GPU programming model.  Generated HIP C is compiled with hipcc:

  hipcc --amdgpu-target=gfx90a -O2 -shared -fPIC -o kernel.so kernel.hip

Key differences from CUDA backend:
  - Header:     #include <hip/hip_runtime.h>  (not cuda_runtime.h)
  - bfloat16:   hip_bfloat16  (not __nv_bfloat16)
  - shfl mask:  __shfl_down(v, d)  — no mask argument
  - __ldg:      direct pointer deref (HIP __ldg works but not needed)
  - wmma:       rocwmma headers (not nvcuda/mma.h) — disabled for now
  - Kernel launch triple-chevrons work identically in HIP C
"""

from __future__ import annotations

from locomp.ir import IRKernel, IROp, IRType, IRValue, OpCode


# ──────────────────────────────────────────────────────────────────────────────
# Type helpers
# ──────────────────────────────────────────────────────────────────────────────

def _c_type(dtype: IRType) -> str:
    return {
        IRType.FLOAT16:  "__half",
        IRType.BFLOAT16: "hip_bfloat16",   # ROCm: hip_bfloat16, not __nv_bfloat16
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


def _math_fn(op: OpCode, dtype: IRType) -> str:
    """Return math function name for a unary op."""
    use_f = dtype == IRType.FLOAT32
    use_h = dtype in (IRType.FLOAT16, IRType.BFLOAT16)
    if use_h:
        return {
            OpCode.SQRT:  "hsqrt",
            OpCode.EXP:   "hexp",
            OpCode.LOG:   "hlog",
            OpCode.ABS:   "__habs",
            OpCode.EXP2:  "hexp2",
            OpCode.LOG2:  "hlog2",
            OpCode.CEIL:  "hceil",
            OpCode.FLOOR: "hfloor",
            OpCode.ROUND: "hrint",
            OpCode.RSQRT: "hrsqrt",
            OpCode.TANH:  "tanhf",
            OpCode.SIN:   "sinf",
            OpCode.COS:   "cosf",
            OpCode.ASIN:  "asinf",
            OpCode.ACOS:  "acosf",
            OpCode.ATAN:  "atanf",
            OpCode.SINH:  "sinhf",
            OpCode.COSH:  "coshf",
            OpCode.LOG10: "log10f",
        }[op]
    return {
        OpCode.SQRT:  "sqrtf" if use_f else "sqrt",
        OpCode.EXP:   "expf"  if use_f else "exp",
        OpCode.LOG:   "logf"  if use_f else "log",
        OpCode.ABS:   "fabsf" if use_f else "fabs",
        OpCode.TANH:  "tanhf" if use_f else "tanh",
        OpCode.SIN:   "sinf"  if use_f else "sin",
        OpCode.COS:   "cosf"  if use_f else "cos",
        OpCode.ASIN:  "asinf" if use_f else "asin",
        OpCode.ACOS:  "acosf" if use_f else "acos",
        OpCode.ATAN:  "atanf" if use_f else "atan",
        OpCode.SINH:  "sinhf" if use_f else "sinh",
        OpCode.COSH:  "coshf" if use_f else "cosh",
        OpCode.EXP2:  "exp2f" if use_f else "exp2",
        OpCode.LOG2:  "log2f" if use_f else "log2",
        OpCode.LOG10: "log10f" if use_f else "log10",
        OpCode.RSQRT: "rsqrtf" if use_f else "rsqrt",
        OpCode.CEIL:  "ceilf" if use_f else "ceil",
        OpCode.FLOOR: "floorf" if use_f else "floor",
        OpCode.ROUND: "roundf" if use_f else "round",
    }[op]


# ──────────────────────────────────────────────────────────────────────────────
# Codegen class
# ──────────────────────────────────────────────────────────────────────────────

class ROCmCodegen:
    """Generates HIP C source from locomp IR for AMD ROCm GPUs.

    Output is a self-contained .hip file exposing:
      extern "C" void locomp_launch_<name>(int grid_x, int grid_y,
                                           int block_x, int block_y, void** args)
    compiled to a .so with hipcc and loaded via ctypes.
    """

    def __init__(self, kernel: IRKernel,
                 constexpr_values: dict[str, int | float] | None = None):
        self.kernel = kernel
        self.indent = "    "
        self._var_names: dict[int, str] = {}
        raw = constexpr_values or {}
        self._constexpr_values: dict[str, int | float] = {}
        for k, v in raw.items():
            self._constexpr_values[k] = v
        self._ptr_exprs: dict[int, tuple[str, str, bool]] = {}
        self._ptr_elem_dtype: dict[int, "IRType"] = {}
        self._nesting: int = 0
        self._predeclared: set[int] = set()

        self._base_names: dict[int, str] = {}
        for p in kernel.params:
            base = p.name.rsplit("_", 1)[0] if "_" in p.name else p.name
            self._base_names[p.id] = base
            if base in raw:
                self._constexpr_values[p.name] = raw[base]
            elif p.name in raw:
                self._constexpr_values[base] = raw[p.name]

    def generate(self) -> tuple[str, dict[str, int]]:
        """Return (hip_source, param_map)."""
        param_map: dict[str, int] = {}
        idx = 0
        for p in self.kernel.params:
            base = self._base_names[p.id]
            is_constexpr = (base in self._constexpr_values or
                            p.name in self._constexpr_values)
            if is_constexpr:
                val = (self._constexpr_values.get(base)
                       or self._constexpr_values.get(p.name))
                self._var_names[p.id] = str(val)
            else:
                self._var_names[p.id] = base
                param_map[base] = idx
                idx += 1

        src = self._build_source(param_map)
        return src, param_map

    def _build_source(self, param_map: dict[str, int]) -> str:
        lines: list[str] = []
        lines += [
            "/* Generated by locomp — ROCm/HIP backend */",
            "#include <hip/hip_runtime.h>",
            "#include <hip/hip_fp16.h>",
            "#include <hip/hip_bfloat16.h>",
            "#include <stdint.h>",
            "#include <stddef.h>",
            "#include <math.h>",
            "",
        ]
        lines += self._gen_kernel_fn()
        lines.append("")
        lines += self._gen_launch_fn(param_map)
        return "\n".join(lines)

    def _gen_kernel_fn(self) -> list[str]:
        sig_parts = []
        for p in self.kernel.params:
            base = self._base_names[p.id]
            is_constexpr = (base in self._constexpr_values or
                            p.name in self._constexpr_values)
            if is_constexpr:
                continue
            ct = _c_type(p.dtype)
            if p.is_pointer:
                sig_parts.append(f"{ct}* __restrict__ {base}")
            else:
                sig_parts.append(f"{ct} {base}")

        lines = [
            f"__global__ void {self.kernel.name}_kernel(",
            "    " + ", ".join(sig_parts) if sig_parts else "    void",
            ") {",
        ]
        lines.extend(self._gen_body())
        lines.append("}")
        return lines

    def _gen_body(self) -> list[str]:
        lines: list[str] = []
        self._nesting = 0
        self._ptr_exprs = {}

        for p in self.kernel.params:
            if p.is_pointer:
                self._ptr_elem_dtype[p.id] = p.dtype
        _pe_changed = True
        while _pe_changed:
            _pe_changed = False
            for _op in self.kernel.ops:
                if _op.result is None:
                    continue
                if _op.opcode in (OpCode.PTR_ADD, OpCode.ADD, OpCode.COPY):
                    for _o in _op.operands:
                        if _o.id in self._ptr_elem_dtype:
                            if _op.result.id not in self._ptr_elem_dtype:
                                self._ptr_elem_dtype[_op.result.id] = self._ptr_elem_dtype[_o.id]
                                _pe_changed = True
                            break

        self._predeclared = set()
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

        _ptr_result_ids: set[int] = {p.id for p in self.kernel.params}
        _changed = True
        while _changed:
            _changed = False
            for op in self.kernel.ops:
                if op.result is None:
                    continue
                if op.opcode in (OpCode.PTR_ADD, OpCode.ADD, OpCode.COPY):
                    if any(o.id in _ptr_result_ids or getattr(o, 'is_pointer', False)
                           for o in op.operands):
                        if op.result.id not in _ptr_result_ids:
                            _ptr_result_ids.add(op.result.id)
                            _changed = True

        for op in self.kernel.ops:
            if op.result is None:
                continue
            rid = op.result.id
            if (defined_at_depth.get(rid, 0) > 0
                    and rid in referenced
                    and op.result.aliases is None
                    and op.opcode not in (OpCode.FOR_LOOP_START, OpCode.CONSTANT)
                    and not op.result.is_pointer
                    and op.opcode != OpCode.PTR_ADD
                    and rid not in _ptr_result_ids
                    and not getattr(op.result, 'is_simdgroup_matrix', False)):
                var = self._vname(op.result)
                ct = _c_type(op.result.dtype)
                lines.append(f"{self.indent}{ct} {var};")
                self._predeclared.add(rid)

        if self._predeclared:
            lines.append("")

        for smem_name, (smem_dtype, smem_size) in self.kernel.shared_mem.items():
            ct = _c_type(smem_dtype)
            lines.append(f"{self.indent}__shared__ {ct} {smem_name}[{smem_size}];")
        if self.kernel.shared_mem:
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

    def _gen_launch_fn(self, param_map: dict[str, int]) -> list[str]:
        lines = [
            'extern "C" void locomp_launch_' + self.kernel.name + '(',
            "    int _grid_x, int _grid_y, int _block_x, int _block_y,",
            "    void** _args",
            ") {",
        ]
        idx = 0
        for p in self.kernel.params:
            base = self._base_names[p.id]
            is_constexpr = (base in self._constexpr_values or
                            p.name in self._constexpr_values)
            if is_constexpr:
                continue
            ct = _c_type(p.dtype)
            if p.is_pointer:
                lines.append(f"    {ct}* {base} = ({ct}*)_args[{idx}];")
            else:
                lines.append(f"    {ct} {base} = *({ct}*)_args[{idx}];")
            idx += 1

        call_args = []
        for p in self.kernel.params:
            base = self._base_names[p.id]
            is_constexpr = (base in self._constexpr_values or
                            p.name in self._constexpr_values)
            if not is_constexpr:
                call_args.append(base)

        lines += [
            "    dim3 _grid(_grid_x, _grid_y, 1);",
            "    dim3 _block(_block_x, _block_y, 1);",
            f"    {self.kernel.name}_kernel<<<_grid, _block>>>({', '.join(call_args)});",
            "    // caller is responsible for synchronization",
            "}",
        ]
        return lines

    # ── op codegen ────────────────────────────────────────────────────────────

    def _gen_op(self, op: IROp) -> str | list[str] | None:
        rv = self._vname(op.result) if op.result else "_"

        if op.opcode == OpCode.PROGRAM_ID:
            axis = op.attrs.get("axis", 0)
            dim = ["x", "y", "z"][axis]
            return f"int {rv} = blockIdx.{dim};"

        elif op.opcode == OpCode.THREAD_ID:
            return f"int {rv} = blockIdx.x * blockDim.x + threadIdx.x;"

        elif op.opcode == OpCode.LOCAL_ID:
            axis = op.attrs.get("axis", 0)
            dim = ["x", "y", "z"][axis]
            return f"int {rv} = threadIdx.{dim};"

        elif op.opcode == OpCode.GROUP_SIZE:
            return f"int {rv} = blockDim.x;"

        elif op.opcode == OpCode.NUM_GROUPS:
            return f"int {rv} = gridDim.x;"

        elif op.opcode == OpCode.ARANGE:
            start = op.attrs["start"]
            end = op.attrs["end"]
            size = end - start
            ct = _c_type(op.result.dtype)
            return [
                f"{ct} {rv}[{size}];",
                f"for (int _i = 0; _i < {size}; _i++) {{ {rv}[_i] = {start} + _i; }}",
            ]

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

        elif op.opcode == OpCode.PTR_ADD:
            base = self._vname(op.operands[0])
            offset = self._vname(op.operands[1])
            self._ptr_exprs[op.result.id] = (base, offset, False)
            ct = _c_type(op.result.dtype)
            return f"{ct}* {rv} = {base} + {offset};"

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
            return "__syncthreads();"

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

        elif op.opcode in (OpCode.SQRT, OpCode.EXP, OpCode.LOG, OpCode.ABS,
                           OpCode.TANH, OpCode.SIN, OpCode.COS,
                           OpCode.ASIN, OpCode.ACOS, OpCode.ATAN,
                           OpCode.SINH, OpCode.COSH,
                           OpCode.EXP2, OpCode.LOG2, OpCode.LOG10,
                           OpCode.RSQRT, OpCode.CEIL, OpCode.FLOOR, OpCode.ROUND):
            fn = _math_fn(op.opcode, op.result.dtype)
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
            fn_map = {OpCode.POW: "powf", OpCode.ATAN2: "atan2f",
                      OpCode.COPYSIGN: "copysignf", OpCode.FMOD: "fmodf",
                      OpCode.MAX: "fmaxf", OpCode.MIN: "fminf"}
            a = self._vname(op.operands[0])
            b = self._vname(op.operands[1])
            ct = _c_type(op.result.dtype)
            if op.opcode == OpCode.STEP:
                return f"{ct} {rv} = ({a} < {b}) ? 0.0f : 1.0f;"
            return f"{ct} {rv} = {fn_map[op.opcode]}({a}, {b});"

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

        elif op.opcode == OpCode.REDUCE_SUM:
            return self._gen_reduce(op, "sum")

        elif op.opcode == OpCode.REDUCE_MAX:
            return self._gen_reduce(op, "max")

        elif op.opcode == OpCode.REDUCE_MIN:
            return self._gen_reduce(op, "min")

        # ── SIMD / wavefront ops (ROCm wavefront = 64 lanes on GFX9, 32 on GFX10+) ──
        # We use warp size 64 for GFX9 (MI series) and 32 for GFX10+ (RX series).
        # At runtime we use 64 as safe default — __shfl_down is correct for both.
        elif op.opcode == OpCode.SIMD_SUM:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            # ROCm __shfl_down does NOT take a mask argument (unlike CUDA)
            return [
                f"{ct} {rv} = {a};",
                f"#pragma unroll",
                f"for (int _off = 32; _off > 0; _off >>= 1)",
                f"    {rv} += __shfl_down({rv}, _off);",
            ]

        elif op.opcode == OpCode.SIMD_MAX:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return [
                f"{ct} {rv} = {a};",
                f"#pragma unroll",
                f"for (int _off = 32; _off > 0; _off >>= 1)",
                f"    {rv} = fmaxf({rv}, __shfl_down({rv}, _off));",
            ]

        elif op.opcode == OpCode.SIMD_MIN:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return [
                f"{ct} {rv} = {a};",
                f"#pragma unroll",
                f"for (int _off = 32; _off > 0; _off >>= 1)",
                f"    {rv} = fminf({rv}, __shfl_down({rv}, _off));",
            ]

        elif op.opcode == OpCode.SIMD_BROADCAST:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = __shfl({a}, 0);"

        elif op.opcode == OpCode.SIMD_SHUFFLE_DOWN:
            a = self._vname(op.operands[0])
            delta = self._vname(op.operands[1]) if len(op.operands) > 1 else "1"
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = __shfl_down({a}, {delta});"

        elif op.opcode == OpCode.SIMD_LANE_ID:
            return f"int {rv} = threadIdx.x % 64;"   # wavefront of 64 for GFX9

        elif op.opcode == OpCode.SIMD_GROUP_ID:
            return f"int {rv} = threadIdx.x / 64;"

        elif op.opcode == OpCode.CAST:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = ({ct}){a};"

        elif op.opcode == OpCode.WHERE:
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

        elif op.opcode == OpCode.COPY:
            a = self._vname(op.operands[0])
            if op.result.aliases is not None:
                return f"{rv} = {a};"
            src_id = op.operands[0].id
            if src_id in self._ptr_elem_dtype:
                elem_ct = _c_type(self._ptr_elem_dtype[src_id])
                self._ptr_elem_dtype[op.result.id] = self._ptr_elem_dtype[src_id]
                return f"{elem_ct}* {rv} = {a};"
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = {a};"

        elif op.opcode in (OpCode.ATOMIC_ADD, OpCode.ATOMIC_MAX, OpCode.ATOMIC_MIN):
            return self._gen_atomic(op)

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
            return None

        elif op.opcode == OpCode.BREAK:
            return "break;"

        elif op.opcode == OpCode.CONTINUE:
            return "continue;"

        # wmma/simdgroup matrix ops — not supported on ROCm (use rocWMMA separately)
        elif op.opcode in (OpCode.SIMDGROUP_MATRIX_FILL, OpCode.SIMDGROUP_MATRIX_LOAD,
                           OpCode.SIMDGROUP_MATRIX_STORE, OpCode.SIMDGROUP_MATRIX_MAC):
            return f"/* ROCm: simdgroup matrix op {op.opcode.name} requires rocWMMA — skipped */"

        else:
            return f"/* unhandled op: {op.opcode.name} */"

    # ── load / store helpers ─────────────────────────────────────────────────

    def _gen_load(self, op: IROp) -> str | list[str]:
        rv = self._vname(op.result)
        ptr_val = op.operands[0]
        _elem_dtype = self._ptr_elem_dtype.get(ptr_val.id, op.result.dtype)
        ct = _c_type(_elem_dtype)

        is_tiled = False
        pb = None
        pi = None
        if ptr_val.id in self._ptr_exprs:
            pb, pi, is_tiled = self._ptr_exprs[ptr_val.id]
            ptr_expr = pb if is_tiled else f"({pb} + {pi})"
        else:
            ptr_expr = self._vname(ptr_val)

        if op.result.shape:
            size = op.result.shape[0]
            if is_tiled and pi is not None:
                if ct == "float" and size % 4 == 0:
                    n4 = size // 4
                    ls = [f"float {rv}[{size}];",
                          f"{{ float4* _src4 = (float4*)({pb} + {pi}[0]);"]
                    ls.append(f"  for (int _j = 0; _j < {n4}; _j++) {{")
                    ls.append(f"    float4 _v = _src4[_j];")
                    ls.append(f"    {rv}[_j*4+0]=_v.x; {rv}[_j*4+1]=_v.y;")
                    ls.append(f"    {rv}[_j*4+2]=_v.z; {rv}[_j*4+3]=_v.w; }}")
                    ls.append(f"}}")
                    return ls
                return [
                    f"{ct} {rv}[{size}];",
                    f"for (int _i = 0; _i < {size}; _i++) {{ {rv}[_i] = {pb}[{pi}[_i]]; }}",
                ]
            # Non-indexed tiled
            if ct == "float" and size % 4 == 0:
                n4 = size // 4
                ls = [f"float {rv}[{size}];",
                      f"{{ float4* _src4 = (float4*)({ptr_expr});"]
                ls.append(f"  for (int _j = 0; _j < {n4}; _j++) {{")
                ls.append(f"    float4 _v = _src4[_j];")
                ls.append(f"    {rv}[_j*4+0]=_v.x; {rv}[_j*4+1]=_v.y;")
                ls.append(f"    {rv}[_j*4+2]=_v.z; {rv}[_j*4+3]=_v.w; }}")
                ls.append(f"}}")
                return ls
            return [
                f"{ct} {rv}[{size}];",
                f"for (int _i = 0; _i < {size}; _i++) {{ {rv}[_i] = {ptr_expr}[_i]; }}",
            ]

        mask = op.attrs.get("mask")
        other = op.attrs.get("other")
        if mask is not None:
            from locomp.ir import IRValue as _IRV
            mask_var = self._vname(mask) if isinstance(mask, _IRV) else str(mask)
            other_val = self._vname(other) if isinstance(other, _IRV) else str(other)
            return [
                f"{ct} {rv};",
                f"if ({mask_var}) {{ {rv} = *{ptr_expr}; }}",
                f"else {{ {rv} = ({ct}){other_val}; }}",
            ]
        return f"{ct} {rv} = *{ptr_expr};"

    def _gen_store(self, op: IROp) -> str | list[str]:
        ptr_val = op.operands[0]
        val_val = op.operands[1]

        is_tiled = False
        pb = None
        pi = None
        if ptr_val.id in self._ptr_exprs:
            pb, pi, is_tiled = self._ptr_exprs[ptr_val.id]
            ptr_expr = pb if is_tiled else f"({pb} + {pi})"
        else:
            ptr_expr = self._vname(ptr_val)

        val = self._vname(val_val)

        if val_val.shape:
            size = val_val.shape[0]
            _store_dtype = self._ptr_elem_dtype.get(ptr_val.id, val_val.dtype)
            ct = _c_type(_store_dtype)
            if is_tiled and pi is not None:
                if ct == "float" and size % 4 == 0:
                    n4 = size // 4
                    ls = [f"{{ float4* _dst4 = (float4*)({pb} + {pi}[0]);"]
                    ls.append(f"  for (int _j = 0; _j < {n4}; _j++)")
                    ls.append(f"    _dst4[_j] = make_float4({val}[_j*4+0],{val}[_j*4+1],"
                              f"{val}[_j*4+2],{val}[_j*4+3]);")
                    ls.append(f"}}")
                    return ls
                return [
                    f"for (int _i = 0; _i < {size}; _i++) {{ {pb}[{pi}[_i]] = {val}[_i]; }}",
                ]
            if ct == "float" and size % 4 == 0:
                n4 = size // 4
                ls = [f"{{ float4* _dst4 = (float4*)({ptr_expr});"]
                ls.append(f"  for (int _j = 0; _j < {n4}; _j++)")
                ls.append(f"    _dst4[_j] = make_float4({val}[_j*4+0],{val}[_j*4+1],"
                          f"{val}[_j*4+2],{val}[_j*4+3]);")
                ls.append(f"}}")
                return ls
            return [
                f"for (int _i = 0; _i < {size}; _i++) {{ {ptr_expr}[_i] = {val}[_i]; }}",
            ]

        mask = op.attrs.get("mask")
        if mask is not None:
            from locomp.ir import IRValue as _IRV
            mask_var = self._vname(mask) if isinstance(mask, _IRV) else str(mask)
            return [f"if ({mask_var}) {{ *{ptr_expr} = {val}; }}"]
        return f"*{ptr_expr} = {val};"

    def _gen_arith_binop(self, op: IROp) -> str | list[str]:
        rv = self._vname(op.result)
        a = self._vname(op.operands[0])
        b = self._vname(op.operands[1])
        ct = _c_type(op.result.dtype)

        a_ptr = op.operands[0].is_pointer or (op.operands[0].id in self._ptr_exprs)
        b_ptr = op.operands[1].is_pointer or (op.operands[1].id in self._ptr_exprs)
        if (op.opcode == OpCode.ADD) and (a_ptr or b_ptr):
            ptr_op = op.operands[0] if a_ptr else op.operands[1]
            idx_op = op.operands[1] if a_ptr else op.operands[0]
            ptr_var = a if a_ptr else b
            idx_var = b if a_ptr else a
            if ptr_op.id in self._ptr_exprs:
                pb, pi, pt = self._ptr_exprs[ptr_op.id]
                base_expr = pb if pt else f"({pb} + {pi})"
            else:
                base_expr = ptr_var
            if idx_op.shape:
                self._ptr_exprs[op.result.id] = (base_expr, idx_var, True)
                return None
            else:
                self._ptr_exprs[op.result.id] = (base_expr, idx_var, False)
                ptr_ct = _c_type(self._ptr_elem_dtype.get(ptr_op.id, op.result.dtype))
                return f"{ptr_ct}* {rv} = {base_expr} + {idx_var};"

        sym = {OpCode.ADD: "+", OpCode.SUB: "-", OpCode.MUL: "*",
               OpCode.DIV: "/", OpCode.MOD: "%"}.get(op.opcode, "+")

        if op.result.shape:
            size = op.result.shape[0]
            a_tile = bool(op.operands[0].shape)
            b_tile = bool(op.operands[1].shape)
            return [
                f"{ct} {rv}[{size}];",
                f"for (int _i = 0; _i < {size}; _i++) {{",
                f"    {rv}[_i] = {a if not a_tile else a+'[_i]'} {sym} {b if not b_tile else b+'[_i]'};",
                "}",
            ]
        return f"{ct} {rv} = {a} {sym} {b};"

    def _gen_reduce(self, op: IROp, kind: str) -> str | list[str]:
        rv = self._vname(op.result)
        arg = self._vname(op.operands[0])
        ct = _c_type(op.result.dtype)

        if op.operands[0].shape:
            size = op.operands[0].shape[0]
            init = {"sum": "0.0f", "max": "-1e38f", "min": "1e38f"}[kind]
            op_expr = {"sum": f"{rv} += {arg}[_i]",
                       "max": f"{rv} = fmaxf({rv}, {arg}[_i])",
                       "min": f"{rv} = fminf({rv}, {arg}[_i])"}[kind]
            return [
                f"{ct} {rv} = {init};",
                f"for (int _i = 0; _i < {size}; _i++) {{ {op_expr}; }}",
            ]

        if len(op.operands) == 2:
            val = arg
            out_ptr = self._vname(op.operands[1])
            if kind == "sum":
                return f"atomicAdd({out_ptr}, {val});"
            elif kind == "max":
                return [
                    f"{{ int* _addr = (int*){out_ptr};",
                    f"  int _old = *_addr, _assumed;",
                    f"  float _val = {val};",
                    f"  do {{ _assumed = _old;",
                    f"       _old = atomicCAS(_addr, _assumed,",
                    f"           __float_as_int(fmaxf(__int_as_float(_assumed), _val)));",
                    f"  }} while (_assumed != _old); }}",
                ]
            elif kind == "min":
                return [
                    f"{{ int* _addr = (int*){out_ptr};",
                    f"  int _old = *_addr, _assumed;",
                    f"  float _val = {val};",
                    f"  do {{ _assumed = _old;",
                    f"       _old = atomicCAS(_addr, _assumed,",
                    f"           __float_as_int(fminf(__int_as_float(_assumed), _val)));",
                    f"  }} while (_assumed != _old); }}",
                ]

        return f"{ct} {rv} = {arg};"

    def _gen_atomic(self, op: IROp) -> str | list[str]:
        ptr_val = op.operands[0]
        val_val = op.operands[1]
        if ptr_val.id in self._ptr_exprs:
            pb, pi, _ = self._ptr_exprs[ptr_val.id]
            ptr_expr = f"({pb} + {pi})"
        else:
            ptr_expr = self._vname(ptr_val)
        val = self._vname(val_val)

        if op.opcode == OpCode.ATOMIC_ADD:
            return f"atomicAdd({ptr_expr}, {val});"
        elif op.opcode == OpCode.ATOMIC_MAX:
            if val_val.dtype == IRType.INT32:
                return f"atomicMax((int*){ptr_expr}, (int){val});"
            return [
                f"{{ int* _addr = (int*){ptr_expr};",
                f"  int _old = *_addr, _assumed;",
                f"  do {{ _assumed = _old;",
                f"       _old = atomicCAS(_addr, _assumed,",
                f"           __float_as_int(fmaxf(__int_as_float(_assumed), {val})));",
                f"  }} while (_assumed != _old); }}",
            ]
        elif op.opcode == OpCode.ATOMIC_MIN:
            if val_val.dtype == IRType.INT32:
                return f"atomicMin((int*){ptr_expr}, (int){val});"
            return [
                f"{{ int* _addr = (int*){ptr_expr};",
                f"  int _old = *_addr, _assumed;",
                f"  do {{ _assumed = _old;",
                f"       _old = atomicCAS(_addr, _assumed,",
                f"           __float_as_int(fminf(__int_as_float(_assumed), {val})));",
                f"  }} while (_assumed != _old); }}",
            ]
        return "/* atomic op unhandled */;"

    def _vname(self, value: IRValue) -> str:
        if value is None:
            return "_"
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
        if '* ' in stripped.split('=')[0] or stripped.split('=')[0].rstrip().endswith('*'):
            return line
        for prefix in ("float ", "int ", "double ", "int8_t ", "int16_t ",
                       "int32_t ", "int64_t ", "uint8_t ", "uint16_t ",
                       "uint32_t ", "uint64_t ", "__half ", "hip_bfloat16 "):
            if stripped.startswith(prefix):
                indent = line[:len(line) - len(stripped)]
                return indent + stripped[len(prefix):]
        return line


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def compile_to_rocm(kernel: IRKernel,
                    constexpr_values: dict[str, int | float] | None = None
                    ) -> tuple[str, dict[str, int]]:
    """Compile IR kernel to HIP C source for AMD ROCm.

    Returns:
        (hip_source, param_map)  — param_map: {param_name → arg_index}
    """
    cg = ROCmCodegen(kernel, constexpr_values=constexpr_values)
    return cg.generate()
