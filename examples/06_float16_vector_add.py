"""
Example: Float16 Vector Addition on Apple GPU using Locust.

Demonstrates end-to-end float16 support:
- Float16 type annotation for kernel parameters
- Half-precision Metal buffers (2 bytes per element vs 4)
- MSL `half` type codegen

Apple M1/M2/M3/M4 GPUs have native half-precision ALUs — float16 runs
at 2x throughput and uses half the memory bandwidth.
"""

import locomp
import numpy as np


@locomp.kernel
def vector_add_f16(X: locomp.Float16, Y: locomp.Float16, OUT: locomp.Float16,
                   N: locomp.constexpr):
    tid = locomp.thread_id(0)
    mask = tid < N
    x = locomp.load(X + tid, mask=mask)
    y = locomp.load(Y + tid, mask=mask)
    out = x + y
    locomp.store(OUT + tid, out, mask=mask)


def main():
    print("=" * 60)
    print("LOCUST IR (float16 vector_add):")
    print("=" * 60)
    print(vector_add_f16.ir.dump())

    print("\n" + "=" * 60)
    print("GENERATED MSL:")
    print("=" * 60)
    print(vector_add_f16.msl)

    print("\n" + "=" * 60)
    print("EXECUTING ON GPU (float16):")
    print("=" * 60)

    N = 1024
    x = locomp.tensor(np.random.randn(N).astype(np.float16))
    y = locomp.tensor(np.random.randn(N).astype(np.float16))
    out = locomp.empty(N, dtype=np.float16)

    threads_per_group = 256
    num_groups = (N + threads_per_group - 1) // threads_per_group
    vector_add_f16[(num_groups,), (threads_per_group,)](x, y, out, N)

    expected = (x.numpy() + y.numpy())
    result = out.numpy()
    max_error = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))

    print(f"Dtype: {result.dtype}")
    print(f"Buffer size: {N * 2} bytes (vs {N * 4} for float32)")
    print(f"Max error: {max_error}")
    print(f"Result (first 10): {result[:10]}")
    print(f"Expected (first 10): {expected[:10]}")
    assert max_error < 0.01, f"Error too large: {max_error}"
    print("PASSED!")


if __name__ == "__main__":
    main()
