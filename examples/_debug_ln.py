"""Quick debug: inspect generated MSL for LayerNorm."""
import numpy as np
import locomp

@locomp.kernel
def k_layernorm(X: locomp.Tensor, O: locomp.Tensor, D: locomp.constexpr):
    row = locomp.program_id(0)
    tid = locomp.thread_id(0)
    smem = locomp.shared_memory(256)
    stats = locomp.shared_memory(2)

    val = locomp.load(X + (row * D + tid))
    locomp.shared_store(smem, tid, val)
    locomp.barrier()

    if tid == 0:
        s = 0.0
        for j in range(D):
            s = s + locomp.shared_load(smem, j)
        locomp.shared_store(stats, 0, s / 256.0)
    locomp.barrier()

    mean = locomp.shared_load(stats, 0)
    diff = val - mean
    locomp.shared_store(smem, tid, diff * diff)
    locomp.barrier()

    if tid == 0:
        v = 0.0
        for j in range(D):
            v = v + locomp.shared_load(smem, j)
        locomp.shared_store(stats, 1, v / 256.0)
    locomp.barrier()

    var = locomp.shared_load(stats, 1)
    norm = (val - mean) * locomp.rsqrt(var + 0.00001)
    locomp.store(O + (row * D + tid), norm)

D = 256
O_t = locomp.empty(D)
k_layernorm[(1,), (256,)](locomp.tensor(np.ones(D, dtype=np.float32)), O_t, D)
print(k_layernorm._msl_source)
