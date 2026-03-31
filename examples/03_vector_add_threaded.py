"""
Example: Threaded Vector Addition on Apple GPU using Locust.

Unlike 01_vector_add.py which uses 1 thread per threadgroup (Triton-style tiling),
this uses real GPU threading: each thread handles one element.

Demonstrates: thread_id(), threadgroup dispatch, multi-thread parallelism.
"""

import locomp
import numpy as np


@locomp.kernel
def vector_add_threaded(X: locomp.Tensor, Y: locomp.Tensor, OUT: locomp.Tensor, N: locomp.constexpr):
    # Each thread processes exactly one element
    tid = locomp.thread_id(0)
    mask = tid < N
    x = locomp.load(X + tid, mask=mask)
    y = locomp.load(Y + tid, mask=mask)
    out = x + y
    locomp.store(OUT + tid, out, mask=mask)


def main():
    print("=" * 60)
    print("LOCUST IR:")
    print("=" * 60)
    print(vector_add_threaded.ir.dump())

    print("\n" + "=" * 60)
    print("GENERATED MSL (Metal Shading Language):")
    print("=" * 60)
    print(vector_add_threaded.msl)

    print("\n" + "=" * 60)
    print("EXECUTING ON GPU (threaded):")
    print("=" * 60)

    N = 1024
    x = locomp.tensor(np.random.randn(N).astype(np.float32))
    y = locomp.tensor(np.random.randn(N).astype(np.float32))
    out = locomp.empty(N)

    # Launch: 4 threadgroups × 256 threads each = 1024 threads total
    threads_per_group = 256
    num_groups = (N + threads_per_group - 1) // threads_per_group
    vector_add_threaded[(num_groups,), (threads_per_group,)](x, y, out, N)

    # Verify
    expected = x.numpy() + y.numpy()
    result = out.numpy()
    max_error = np.max(np.abs(result - expected))
    print(f"Max error: {max_error}")
    print(f"Result (first 10): {result[:10]}")
    print(f"Expected (first 10): {expected[:10]}")
    assert max_error < 1e-5, f"Error too large: {max_error}"
    print("PASSED!")


if __name__ == "__main__":
    main()
