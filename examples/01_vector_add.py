"""
Example: Vector Addition on Apple GPU using Locust.

This is the "hello world" of GPU computing — add two arrays element-wise.
Demonstrates @kernel, program_id, arange, load, store, and masked operations.
"""

import locomp
import numpy as np


@locomp.kernel
def vector_add(X: locomp.Tensor, Y: locomp.Tensor, OUT: locomp.Tensor, N: locomp.constexpr):
    pid = locomp.program_id(0)
    offsets = pid * 256 + locomp.arange(0, 256)
    mask = offsets < N
    x = locomp.load(X + offsets, mask=mask)
    y = locomp.load(Y + offsets, mask=mask)
    output = x + y
    locomp.store(OUT + offsets, output, mask=mask)


def main():
    # --- Inspect the compilation pipeline ---

    # 1. Show the IR
    print("=" * 60)
    print("LOCUST IR:")
    print("=" * 60)
    print(vector_add.ir.dump())

    # 2. Show the generated Metal Shading Language
    print("\n" + "=" * 60)
    print("GENERATED MSL (Metal Shading Language):")
    print("=" * 60)
    print(vector_add.msl)

    # --- Run on GPU ---
    print("\n" + "=" * 60)
    print("EXECUTING ON GPU:")
    print("=" * 60)

    N = 1024
    x = locomp.tensor(np.random.randn(N).astype(np.float32))
    y = locomp.tensor(np.random.randn(N).astype(np.float32))
    out = locomp.empty(N)

    # Launch with enough blocks to cover N elements (N/256 = 4 blocks)
    grid_size = (N + 255) // 256
    vector_add[(grid_size,)](x, y, out, N)

    # Verify
    expected = x.numpy() + y.numpy()
    actual = out.numpy()
    max_diff = np.max(np.abs(expected - actual))
    print(f"Max difference from CPU result: {max_diff}")
    print(f"First 10 results: {actual[:10]}")
    print(f"Correct: {max_diff < 1e-5}")


if __name__ == "__main__":
    main()
