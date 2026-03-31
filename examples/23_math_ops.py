"""
Math ops — verify all 24 new math functions compile + run correctly on GPU.
Tests every unary, binary, and ternary math op against NumPy reference.
"""
import numpy as np
import locomp

N = 256


# ── Unary ops ──

@locomp.kernel
def k_tanh(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.tanh(locomp.load(X + i)))

@locomp.kernel
def k_sin(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.sin(locomp.load(X + i)))

@locomp.kernel
def k_cos(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.cos(locomp.load(X + i)))

@locomp.kernel
def k_asin(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.asin(locomp.load(X + i)))

@locomp.kernel
def k_acos(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.acos(locomp.load(X + i)))

@locomp.kernel
def k_atan(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.atan(locomp.load(X + i)))

@locomp.kernel
def k_sinh(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.sinh(locomp.load(X + i)))

@locomp.kernel
def k_cosh(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.cosh(locomp.load(X + i)))

@locomp.kernel
def k_exp2(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.exp2(locomp.load(X + i)))

@locomp.kernel
def k_log2(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.log2(locomp.load(X + i)))

@locomp.kernel
def k_log10(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.log10(locomp.load(X + i)))

@locomp.kernel
def k_rsqrt(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.rsqrt(locomp.load(X + i)))

@locomp.kernel
def k_ceil(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.ceil(locomp.load(X + i)))

@locomp.kernel
def k_floor(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.floor(locomp.load(X + i)))

@locomp.kernel
def k_round(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.round(locomp.load(X + i)))

@locomp.kernel
def k_sigmoid(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.sigmoid(locomp.load(X + i)))


# ── Binary ops ──

@locomp.kernel
def k_pow(X: locomp.Tensor, Y: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.pow(locomp.load(X + i), locomp.load(Y + i)))

@locomp.kernel
def k_atan2(X: locomp.Tensor, Y: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.atan2(locomp.load(X + i), locomp.load(Y + i)))

@locomp.kernel
def k_copysign(X: locomp.Tensor, Y: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.copysign(locomp.load(X + i), locomp.load(Y + i)))

@locomp.kernel
def k_fmod(X: locomp.Tensor, Y: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.fmod(locomp.load(X + i), locomp.load(Y + i)))

@locomp.kernel
def k_step(X: locomp.Tensor, Y: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.step(locomp.load(X + i), locomp.load(Y + i)))


# ── Ternary ops ──

@locomp.kernel
def k_fma(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.fma(locomp.load(A + i), locomp.load(B + i), locomp.load(C + i)))

@locomp.kernel
def k_clamp(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    locomp.store(O + i, locomp.clamp(locomp.load(X + i), -0.5, 0.5))


# ── GELU kernel (uses tanh + math) ──

@locomp.kernel
def k_gelu(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0) * 256 + locomp.thread_id(0)
    x = locomp.load(X + i)
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    inner = 0.7978845608 * (x + 0.044715 * x * x * x)
    locomp.store(O + i, 0.5 * x * (1.0 + locomp.tanh(inner)))


# ── LayerNorm kernel (uses rsqrt) ──

@locomp.kernel
def k_layernorm(X: locomp.Tensor, O: locomp.Tensor, D: locomp.constexpr):
    row = locomp.program_id(0)
    tid = locomp.local_id(0)
    smem = locomp.shared_memory(256)
    stats = locomp.shared_memory(2)  # [0]=mean, [1]=variance

    # Load value
    val = locomp.load(X + (row * D + tid))
    locomp.shared_store(smem, tid, val)
    locomp.barrier()

    # Mean (thread 0)
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

    # Variance (thread 0)
    if tid == 0:
        v = 0.0
        for j in range(D):
            v = v + locomp.shared_load(smem, j)
        locomp.shared_store(stats, 1, v / 256.0)
    locomp.barrier()

    var = locomp.shared_load(stats, 1)
    norm = (val - mean) * locomp.rsqrt(var + 0.00001)
    locomp.store(O + (row * D + tid), norm)


def run_unary(name, kernel_fn, x_np, ref_fn):
    X = locomp.tensor(x_np)
    O = locomp.empty(N)
    kernel_fn[(1,), (256,)](X, O, N)
    result = O.numpy()
    expected = ref_fn(x_np)
    err = np.max(np.abs(result - expected))
    status = "✅" if err < 1e-5 else "❌"
    print(f"  {name:>12}: err={err:.2e} {status}")
    return err < 1e-5


def run_binary(name, kernel_fn, x_np, y_np, ref_fn):
    X = locomp.tensor(x_np)
    Y = locomp.tensor(y_np)
    O = locomp.empty(N)
    kernel_fn[(1,), (256,)](X, Y, O, N)
    result = O.numpy()
    expected = ref_fn(x_np, y_np)
    err = np.max(np.abs(result - expected))
    status = "✅" if err < 1e-5 else "❌"
    print(f"  {name:>12}: err={err:.2e} {status}")
    return err < 1e-5


if __name__ == "__main__":
    np.random.seed(42)
    x = np.random.randn(N).astype(np.float32)
    x_pos = np.abs(x) + 0.01  # positive values for log/rsqrt
    x_unit = np.clip(x * 0.3, -0.99, 0.99)  # in (-1,1) for asin/acos
    x_frac = x * 1.5  # for ceil/floor/round
    y = np.random.randn(N).astype(np.float32)
    y_pos = np.abs(y) + 0.5  # positive for pow

    passed = 0
    total = 0

    print("=== Unary Math Ops ===")
    tests = [
        ("tanh", k_tanh, x, np.tanh),
        ("sin", k_sin, x, np.sin),
        ("cos", k_cos, x, np.cos),
        ("asin", k_asin, x_unit, np.arcsin),
        ("acos", k_acos, x_unit, np.arccos),
        ("atan", k_atan, x, np.arctan),
        ("sinh", k_sinh, x, np.sinh),
        ("cosh", k_cosh, x, np.cosh),
        ("exp2", k_exp2, x, np.exp2),
        ("log2", k_log2, x_pos, np.log2),
        ("log10", k_log10, x_pos, np.log10),
        ("rsqrt", k_rsqrt, x_pos, lambda v: 1.0 / np.sqrt(v)),
        ("ceil", k_ceil, x_frac, np.ceil),
        ("floor", k_floor, x_frac, np.floor),
        ("round", k_round, x_frac, np.rint),
        ("sigmoid", k_sigmoid, x, lambda v: 1.0 / (1.0 + np.exp(-v))),
    ]
    for name, fn, inp, ref in tests:
        if run_unary(name, fn, inp, ref):
            passed += 1
        total += 1

    print("\n=== Binary Math Ops ===")
    bin_tests = [
        ("pow", k_pow, x_pos, y_pos, np.power),
        ("atan2", k_atan2, x, y, np.arctan2),
        ("copysign", k_copysign, x, y, np.copysign),
        ("fmod", k_fmod, x, y_pos, np.fmod),
        ("step", k_step, np.zeros(N, dtype=np.float32), x,
         lambda e, v: np.where(v < e, 0.0, 1.0).astype(np.float32)),
    ]
    for name, fn, ix, iy, ref in bin_tests:
        if run_binary(name, fn, ix, iy, ref):
            passed += 1
        total += 1

    print("\n=== Ternary Math Ops ===")
    # fma
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    c = np.random.randn(N).astype(np.float32)
    A_t = locomp.tensor(a); B_t = locomp.tensor(b); C_t = locomp.tensor(c)
    O_t = locomp.empty(N)
    k_fma[(1,), (256,)](A_t, B_t, C_t, O_t, N)
    fma_err = np.max(np.abs(O_t.numpy() - (a * b + c)))
    ok = fma_err < 1e-5
    print(f"  {'fma':>12}: err={fma_err:.2e} {'✅' if ok else '❌'}")
    if ok: passed += 1
    total += 1

    # clamp
    O_t = locomp.empty(N)
    k_clamp[(1,), (256,)](locomp.tensor(x), O_t, N)
    clamp_err = np.max(np.abs(O_t.numpy() - np.clip(x, -0.5, 0.5)))
    ok = clamp_err < 1e-5
    print(f"  {'clamp':>12}: err={clamp_err:.2e} {'✅' if ok else '❌'}")
    if ok: passed += 1
    total += 1

    print("\n=== Composite Kernels ===")
    # GELU
    O_t = locomp.empty(N)
    k_gelu[(1,), (256,)](locomp.tensor(x), O_t, N)
    gelu_ref = 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))
    gelu_err = np.max(np.abs(O_t.numpy() - gelu_ref))
    ok = gelu_err < 1e-5
    print(f"  {'GELU':>12}: err={gelu_err:.2e} {'✅' if ok else '❌'}")
    if ok: passed += 1
    total += 1

    # LayerNorm
    D = 256
    ROWS = 4
    x_ln = np.random.randn(ROWS, D).astype(np.float32)
    O_ln = locomp.empty(ROWS * D)
    k_layernorm[(ROWS,), (256,)](locomp.tensor(x_ln.flatten()), O_ln, D)
    result_ln = O_ln.numpy().reshape(ROWS, D)
    mean = x_ln.mean(axis=1, keepdims=True)
    var = x_ln.var(axis=1, keepdims=True)
    expected_ln = (x_ln - mean) / np.sqrt(var + 1e-5)
    ln_err = np.max(np.abs(result_ln - expected_ln))
    ok = ln_err < 1e-4
    print(f"  {'LayerNorm':>12}: err={ln_err:.2e} {'✅' if ok else '❌'}")
    if ok: passed += 1
    total += 1

    print(f"\n{'='*40}")
    print(f"  {passed}/{total} passed")
