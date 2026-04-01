"""
Example 43: Paged Attention — variable-length KV cache with block table.

Core serving primitive: multiple requests with different sequence lengths
share a pool of KV cache blocks. Each request has a block table mapping
logical block → physical block.

Architecture:
  KV cache is stored as physical blocks: cache_k[NUM_BLOCKS, BLOCK_SIZE, D],
  cache_v[NUM_BLOCKS, BLOCK_SIZE, D].
  
  Block table: block_table[MAX_SEQ_BLOCKS] — maps logical block index to
  physical block index for the current request.
  
  Query: q[D] — single query vector (decode step).
  Output: o[D] — attention output.

  Kernel: one threadgroup per query head.
  Threads cooperatively compute attention over all KV blocks.
  Online softmax across blocks for numerical stability.
"""

import time
import numpy as np
import locomp


@locomp.kernel
def paged_attention_d128(
    Q: locomp.Tensor,            # [H, D] query vectors (D=128)
    K_cache: locomp.Tensor,      # [num_phys_blocks * BLOCK_SIZE * D] flat
    V_cache: locomp.Tensor,      # [num_phys_blocks * BLOCK_SIZE * D] flat
    block_table: locomp.Tensor,  # [max_blocks] physical block indices (as floats)
    O: locomp.Tensor,            # [H, D] output
    NUM_BLOCKS: locomp.constexpr,    # number of blocks in this request's sequence
    BLOCK_SIZE: locomp.constexpr,    # tokens per block (e.g. 16)
    D: locomp.constexpr,             # head dim (128)
):
    head = locomp.program_id(0)
    tid = locomp.local_id(0)  # 0..127
    lane = locomp.simd_lane_id()
    sg = locomp.simd_group_id()

    # Shared memory for partial results
    partial_sum = locomp.shared_memory(4)    # per-simd-group
    partial_max = locomp.shared_memory(4)
    q_shared = locomp.shared_memory(128)     # D=128

    # Load query into shared memory (128 threads load 1 element each for D=128)
    scale = 0.0883883  # 1/sqrt(128) ≈ 0.08839
    q_shared_val = locomp.load(Q + (head * D + tid))
    locomp.shared_store(q_shared, tid, q_shared_val * scale)
    locomp.barrier()

    # Each thread accumulates a partial output vector element
    # Thread tid is responsible for output[tid] (one element of D=128)
    acc_out = 0.0
    global_max = -1000000.0
    global_sum = 0.0

    for blk in range(NUM_BLOCKS):
        # Look up physical block
        phys_block_f = locomp.load(block_table + blk)

        # For each token in this block, compute attention score
        for t in range(BLOCK_SIZE):
            # k_offset = phys_block * BLOCK_SIZE * D + t * D
            k_base = phys_block_f * BLOCK_SIZE * D + t * D

            # Dot product q · k: each thread loads one q and one k element,
            # then SIMD reduce
            q_val = locomp.shared_load(q_shared, tid)
            k_val = locomp.load(K_cache + (k_base + tid))
            dot_partial = q_val * k_val

            # Reduce across all 128 threads (4 SIMD groups)
            dot_sg = locomp.simd_sum(dot_partial)
            if lane == 0:
                locomp.shared_store(partial_sum, sg, dot_sg)
            locomp.barrier()

            # Thread 0 computes full dot product
            if tid == 0:
                score = locomp.shared_load(partial_sum, 0) + locomp.shared_load(partial_sum, 1) + locomp.shared_load(partial_sum, 2) + locomp.shared_load(partial_sum, 3)
                locomp.shared_store(partial_sum, 0, score)
            locomp.barrier()
            score_all = locomp.shared_load(partial_sum, 0)

            # Online softmax update
            old_max = global_max
            global_max = locomp.where(score_all > global_max, score_all, global_max)
            correction = locomp.exp(old_max - global_max)

            # Rescale running accumulator and sum
            acc_out = acc_out * correction
            global_sum = global_sum * correction

            # Add this token's contribution
            weight = locomp.exp(score_all - global_max)
            global_sum = global_sum + weight

            v_val = locomp.load(V_cache + (k_base + tid))
            acc_out = acc_out + weight * v_val

        locomp.barrier()

    # Final normalize
    inv_sum = 1.0 / global_sum
    locomp.store(O + (head * D + tid), acc_out * inv_sum)


# =============================================================================
# Dispatch
# =============================================================================

def gpu_paged_attention(q, k_cache, v_cache, block_table, num_blocks,
                        block_size=16, D=128):
    """Paged attention for decode step.
    q: [H, D], k_cache/v_cache: [num_phys_blocks, block_size, D],
    block_table: [max_blocks] int → physical block indices.
    """
    H = q.shape[0]

    Q_g = locomp.tensor(q.reshape(-1))
    K_g = locomp.tensor(k_cache.reshape(-1))
    V_g = locomp.tensor(v_cache.reshape(-1))
    BT_g = locomp.tensor(block_table.astype(np.float32))
    O_g = locomp.empty(H * D)

    paged_attention_d128[(H,), (128,)](Q_g, K_g, V_g, BT_g, O_g,
                                        num_blocks, block_size, D)
    return O_g.numpy().reshape(H, D)


def numpy_paged_attention(q, k_cache, v_cache, block_table, num_blocks,
                          block_size=16, D=128):
    """Reference paged attention."""
    H = q.shape[0]
    scale = 1.0 / np.sqrt(D)
    output = np.zeros((H, D), dtype=np.float32)

    for h in range(H):
        # Gather all K, V from block table
        seq_len = num_blocks * block_size
        all_k = np.zeros((seq_len, D), dtype=np.float32)
        all_v = np.zeros((seq_len, D), dtype=np.float32)
        for b in range(num_blocks):
            phys = int(block_table[b])
            all_k[b * block_size:(b+1) * block_size] = k_cache[phys]
            all_v[b * block_size:(b+1) * block_size] = v_cache[phys]

        scores = (q[h] @ all_k.T) * scale
        scores -= scores.max()
        weights = np.exp(scores)
        weights /= weights.sum()
        output[h] = weights @ all_v

    return output


if __name__ == "__main__":
    WARMUP = 5
    RUNS = 15
    D = 128
    BLOCK_SIZE = 16

    print(f"\n{'='*70}")
    print(f"Paged Attention (D={D}, block_size={BLOCK_SIZE})")
    print(f"{'='*70}")
    print(f"{'Config':>30} | {'GPU':>8} | {'NumPy':>8} | GPU/NP | {'Error':>8}")
    print("-" * 70)

    configs = [
        (8, 4, "H=8 seq=64"),
        (8, 8, "H=8 seq=128"),
        (8, 16, "H=8 seq=256"),
        (32, 8, "H=32 seq=128"),
        (32, 16, "H=32 seq=256"),
        (32, 32, "H=32 seq=512"),
    ]

    for H, num_blocks, label in configs:
        np.random.seed(42)
        num_phys_blocks = num_blocks + 8  # extra physical blocks (fragmented pool)
        q = np.random.randn(H, D).astype(np.float32) * 0.1
        k_cache = np.random.randn(num_phys_blocks, BLOCK_SIZE, D).astype(np.float32) * 0.1
        v_cache = np.random.randn(num_phys_blocks, BLOCK_SIZE, D).astype(np.float32) * 0.1

        # Random block table (simulates fragmented allocation)
        all_phys = np.random.permutation(num_phys_blocks)
        block_table = all_phys[:num_blocks].astype(np.int32)

        ref = numpy_paged_attention(q, k_cache, v_cache, block_table, num_blocks,
                                     BLOCK_SIZE, D)
        gpu_out = gpu_paged_attention(q, k_cache, v_cache, block_table, num_blocks,
                                       BLOCK_SIZE, D)
        err = np.max(np.abs(gpu_out - ref))

        for _ in range(WARMUP):
            gpu_paged_attention(q, k_cache, v_cache, block_table, num_blocks,
                                BLOCK_SIZE, D)
        times_gpu = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            gpu_paged_attention(q, k_cache, v_cache, block_table, num_blocks,
                                BLOCK_SIZE, D)
            times_gpu.append((time.perf_counter() - t0) * 1000)
        t_gpu = sorted(times_gpu)[RUNS // 2]

        for _ in range(WARMUP):
            numpy_paged_attention(q, k_cache, v_cache, block_table, num_blocks,
                                   BLOCK_SIZE, D)
        times_np = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            numpy_paged_attention(q, k_cache, v_cache, block_table, num_blocks,
                                   BLOCK_SIZE, D)
            times_np.append((time.perf_counter() - t0) * 1000)
        t_np = sorted(times_np)[RUNS // 2]

        r = t_gpu / t_np
        print(f"{label:>30} | {t_gpu:>6.3f}ms | {t_np:>6.3f}ms | {r:>5.2f}x | {err:.2e}")
