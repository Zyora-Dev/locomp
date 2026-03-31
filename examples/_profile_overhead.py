"""Profile where the dispatch overhead actually lives."""
import time
import numpy as np
import ctypes
import locomp
from locomp.backends.metal_runtime import get_runtime

N, d = 128, 32
Br, Bc = 16, 16

import importlib
mod = importlib.import_module("examples.22_simdgroup_flash_attn")
flash_attn_simd = mod.flash_attn_simd

Q_t = locomp.tensor(np.random.randn(N * d).astype(np.float32))
K_t = locomp.tensor(np.random.randn(N * d).astype(np.float32))
V_t = locomp.tensor(np.random.randn(N * d).astype(np.float32))
O_t = locomp.empty(N * d)
nkv = N // Bc

# Warmup (compiles pipeline)
flash_attn_simd[(N // Br,), (32, 4)](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)

rt = get_runtime()
import objc

# Get cached pipeline
keys = list(flash_attn_simd._specialized.keys())
pipeline, bmap, msl = flash_attn_simd._specialized[keys[0]]

# Pre-resolve Metal buffers
bufs = [Q_t.to_metal_buffer(rt), K_t.to_metal_buffer(rt),
        V_t.to_metal_buffer(rt), O_t.to_metal_buffer(rt)]

# Pre-resolve ctypes pointers
pp = objc.pyobjc_id(pipeline)
bp = (ctypes.c_void_p * 4)(*[objc.pyobjc_id(b) for b in bufs])
qp = rt._queue_ptr

# 1) Time just C bridge dispatch (no Python overhead)
times_c = []
for _ in range(50):
    t0 = time.perf_counter()
    rt._fast_lib.locust_dispatch(qp, pp, bp, 4, N // Br, 1, 1, 32, 4, 1)
    times_c.append((time.perf_counter() - t0) * 1000)
t_c = sorted(times_c)[25]

# 2) Time full Python call path
times_py = []
for _ in range(50):
    t0 = time.perf_counter()
    flash_attn_simd[(N // Br,), (32, 4)](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)
    times_py.append((time.perf_counter() - t0) * 1000)
t_py = sorted(times_py)[25]

print(f"C bridge dispatch only: {t_c:.3f}ms")
print(f"Full Python call path:  {t_py:.3f}ms")
print(f"Python arg overhead:    {t_py - t_c:.3f}ms")
