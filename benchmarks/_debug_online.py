import locomp, numpy as np
from locomp.frontend import compile_kernel
from locomp.optimizer import optimize
from locomp.backends.metal_codegen import compile_to_metal

def online_softmax_test(X, OUT, ROWS, D, ELEMS):
    row = locomp.program_id(0)
    lid = locomp.local_id(0)
    guard = row < ROWS
    local_max = locomp.load(X + (row * D + lid))
    local_sum = 1.0
    for j in range(1, ELEMS):
        idx = lid + j * 32
        val = locomp.load(X + (row * D + idx))
        new_max = locomp.where(val > local_max, val, local_max)
        local_sum = local_sum * locomp.exp(local_max - new_max) + locomp.exp(val - new_max)
        local_max = new_max
    row_max = locomp.simd_max(local_max)
    local_sum = local_sum * locomp.exp(local_max - row_max)
    total_sum = locomp.simd_sum(local_sum)
    for j in range(ELEMS):
        idx = lid + j * 32
        val = locomp.load(X + (row * D + idx))
        result = locomp.exp(val - row_max) / total_sum
        locomp.store(OUT + (row * D + idx), result, mask=guard)

from locomp.api import Tensor, constexpr

online_softmax_test.__annotations__ = {
    'X': Tensor, 'OUT': Tensor,
    'ROWS': constexpr, 'D': constexpr, 'ELEMS': constexpr
}
ir = compile_kernel(online_softmax_test)
optimize(ir)
msl, _ = compile_to_metal(ir)
print(msl)
