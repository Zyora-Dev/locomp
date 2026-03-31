import locomp
@locomp.kernel
def parallel_softmax(X: locomp.Tensor, OUT: locomp.Tensor,
                     ROWS: locomp.constexpr, D: locomp.constexpr,
                     THREADS: locomp.constexpr, LOG_T: locomp.constexpr,
                     ELEMS: locomp.constexpr):
    row = locomp.program_id(0)
    lid = locomp.local_id(0)
    guard = row < ROWS
    smem = locomp.shared_memory(256)
    local_max = locomp.load(X + (row * D + lid))
    for j in range(1, ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (row * D + idx))
        local_max = locomp.where(val > local_max, val, local_max)
    locomp.shared_store(smem, lid, local_max)
    locomp.barrier()
    stride = THREADS / 2
    for s in range(LOG_T):
        if lid < stride:
            a = locomp.shared_load(smem, lid)
            b = locomp.shared_load(smem, lid + stride)
            mx = locomp.where(b > a, b, a)
            locomp.shared_store(smem, lid, mx)
        locomp.barrier()
        stride = stride / 2
    row_max = locomp.shared_load(smem, 0)
    locomp.barrier()
    local_sum = 0.0
    for j in range(ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (row * D + idx))
        e = locomp.exp(val - row_max)
        local_sum = local_sum + e
    locomp.shared_store(smem, lid, local_sum)
    locomp.barrier()
    stride2 = THREADS / 2
    for s in range(LOG_T):
        if lid < stride2:
            a = locomp.shared_load(smem, lid)
            b = locomp.shared_load(smem, lid + stride2)
            locomp.shared_store(smem, lid, a + b)
        locomp.barrier()
        stride2 = stride2 / 2
    total_sum = locomp.shared_load(smem, 0)
    locomp.barrier()
    for j in range(ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (row * D + idx))
        e = locomp.exp(val - row_max)
        result = e / total_sum
        locomp.store(OUT + (row * D + idx), result, mask=guard)
print(parallel_softmax.msl)
