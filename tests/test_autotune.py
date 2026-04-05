"""
Unit tests for locomp.autotune — backend-agnostic config search + disk cache.

Uses mock launchers to avoid needing a real GPU, so these run on any platform.
Metal-dispatch tests are gated by macos_only.
"""
import time
import pytest
import locomp
from locomp.autotune import AutotunedKernel, Config, _get_gpu_name, clear_cache
from tests.conftest import macos_only


# ─── helpers ────────────────────────────────────────────────────────────────

class _FakeLauncher:
    """Minimal stand-in for a KernelLauncher — no real GPU dispatch."""

    def __init__(self, name="fake_k"):
        self.func_name = name
        self._call_log = []

        import inspect
        # Fake signature: (X, N, BLOCK_SIZE)
        def _fn(X, N, BLOCK_SIZE):
            pass
        self.func = _fn

    def __getitem__(self, grid_tg):
        return self

    def __call__(self, *args, **kwargs):
        self._call_log.append(args)
        return None


# ─── _get_gpu_name ──────────────────────────────────────────────────────────

def test_get_gpu_name_returns_string():
    name = _get_gpu_name()
    assert isinstance(name, str)
    assert len(name) > 0


# ─── Config ─────────────────────────────────────────────────────────────────

def test_config_static_constexprs():
    cfg = Config(BLOCK_SIZE=128, grid=lambda N, BLOCK_SIZE, **kw: (N // BLOCK_SIZE,), tg=(128,))
    resolved = cfg.resolve({"N": 1024})
    assert resolved["BLOCK_SIZE"] == 128
    assert resolved["N"] == 1024


def test_config_derived_constexprs():
    cfg = Config(
        BLOCK_SIZE=128,
        GRID_SIZE=lambda N, BLOCK_SIZE, **kw: N // BLOCK_SIZE,
        grid=lambda GRID_SIZE, **kw: (GRID_SIZE,),
        tg=(128,),
    )
    resolved = cfg.resolve({"N": 1024})
    assert resolved["GRID_SIZE"] == 8  # 1024 // 128


def test_config_get_grid_callable():
    cfg = Config(BLOCK_SIZE=256, grid=lambda N, BLOCK_SIZE, **kw: (N // BLOCK_SIZE,), tg=(256,))
    grid = cfg.get_grid({"N": 512, "BLOCK_SIZE": 256})
    assert grid == (2,)


def test_config_get_grid_static():
    cfg = Config(BLOCK_SIZE=64, grid=(16,), tg=(64,))
    assert cfg.get_grid({}) == (16,)


def test_config_repr():
    cfg = Config(BLOCK_SIZE=64, grid=(1,), tg=(64,))
    r = repr(cfg)
    assert "BLOCK_SIZE=64" in r
    assert "tg=(64,)" in r


# ─── AutotunedKernel — in-memory cache ──────────────────────────────────────

def test_autotune_selects_config():
    """AutotunedKernel benchmarks all configs and picks one."""
    launcher = _FakeLauncher("sel_k")
    configs = [
        Config(BLOCK_SIZE=64,  grid=lambda N, BLOCK_SIZE, **kw: (max(1, N // BLOCK_SIZE),), tg=(64,)),
        Config(BLOCK_SIZE=128, grid=lambda N, BLOCK_SIZE, **kw: (max(1, N // BLOCK_SIZE),), tg=(128,)),
    ]
    ak = AutotunedKernel(launcher, configs, key=["N"], warmup=1, rep=2)
    ak(None, 256)  # X=None (fake tensor), N=256
    assert 0 in ak._cache or (256,) in ak._cache or len(ak._cache) > 0


def test_autotune_uses_memory_cache_on_second_call():
    """Second call with same key must not re-benchmark (no extra launcher calls)."""
    launcher = _FakeLauncher("cache_k")
    configs = [
        Config(BLOCK_SIZE=128, grid=lambda N, BLOCK_SIZE, **kw: (max(1, N // BLOCK_SIZE),), tg=(128,)),
    ]
    ak = AutotunedKernel(launcher, configs, key=["N"], warmup=1, rep=1)
    ak(None, 128)   # first call — benchmarks
    n_calls_after_first = len(launcher._call_log)
    ak(None, 128)   # second call — must use cache
    n_calls_after_second = len(launcher._call_log)
    # Second call should dispatch exactly once (no benchmark loop)
    assert n_calls_after_second - n_calls_after_first == 1


def test_autotune_different_keys_benchmarked_separately():
    """Different N values are separate cache entries."""
    launcher = _FakeLauncher("diff_k")
    configs = [
        Config(BLOCK_SIZE=64, grid=lambda N, BLOCK_SIZE, **kw: (max(1, N // BLOCK_SIZE),), tg=(64,)),
    ]
    ak = AutotunedKernel(launcher, configs, key=["N"], warmup=1, rep=1)
    ak(None, 64)
    ak(None, 128)
    assert len(ak._cache) == 2


# ─── Disk cache ─────────────────────────────────────────────────────────────

def test_autotune_disk_cache_round_trip(tmp_path, monkeypatch):
    """Best config index is persisted and reloaded."""
    import sys
    import locomp  # ensure module is loaded
    _at = sys.modules["locomp.autotune"]
    cache_file = tmp_path / "autotune.json"
    monkeypatch.setattr(_at, "_CACHE_FILE", cache_file)
    monkeypatch.setattr(_at, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_at, "_disk_cache", None)

    launcher = _FakeLauncher("disk_k")
    configs = [
        Config(BLOCK_SIZE=64,  grid=lambda N, BLOCK_SIZE, **kw: (max(1, N // BLOCK_SIZE),), tg=(64,)),
        Config(BLOCK_SIZE=128, grid=lambda N, BLOCK_SIZE, **kw: (max(1, N // BLOCK_SIZE),), tg=(128,)),
    ]
    ak = AutotunedKernel(launcher, configs, key=["N"], warmup=1, rep=1)
    ak(None, 256)

    assert cache_file.exists()

    # New instance — should load from disk, not re-benchmark
    monkeypatch.setattr(_at, "_disk_cache", None)
    launcher2 = _FakeLauncher("disk_k")
    ak2 = AutotunedKernel(launcher2, configs, key=["N"], warmup=1, rep=1)
    ak2(None, 256)
    # Only 1 call (final dispatch), no benchmark loop
    assert len(launcher2._call_log) == 1


def test_clear_cache(tmp_path, monkeypatch):
    """clear_cache() removes disk file and resets in-memory cache."""
    import sys
    import locomp
    _at = sys.modules["locomp.autotune"]
    cache_file = tmp_path / "autotune.json"
    cache_file.write_text('{"some_key": 0}')
    monkeypatch.setattr(_at, "_CACHE_FILE", cache_file)
    monkeypatch.setattr(_at, "_disk_cache", {"some_key": 0})

    clear_cache()

    assert not cache_file.exists()
    assert _at._disk_cache == {}


# ─── float constexpr not truncated ──────────────────────────────────────────

def test_autotune_float_constexpr_preserved():
    """Float constexpr values should NOT be truncated to int."""
    captured = {}

    class _FloatFakeLauncher(_FakeLauncher):
        def __call__(self, *args, **kwargs):
            captured["args"] = args
            return None

    fl = _FloatFakeLauncher("float_k")
    # Signature: (X, SCALE) — only 2 params so Config can provide SCALE
    import inspect
    def _fn2(X, SCALE):
        pass
    fl.func = _fn2

    configs = [
        Config(SCALE=0.5, grid=(1,), tg=(1,)),
    ]
    ak = AutotunedKernel(fl, configs, key=[], warmup=0, rep=1)
    ak(None)  # X=None
    # SCALE should come through as 0.5, not 0
    scale_val = captured["args"][1] if len(captured.get("args", ())) > 1 else None
    assert scale_val == 0.5


# ─── Metal integration (macOS only) ─────────────────────────────────────────

@macos_only
def test_autotune_metal_gelu():
    """Full autotune round-trip on M1: benchmark configs, verify output correctness."""
    import numpy as np
    locomp.clear_cache()

    @locomp.autotune(
        configs=[
            locomp.Config(BLOCK_SIZE=64,
                          grid=lambda N, BLOCK_SIZE, **kw: (N // BLOCK_SIZE,),
                          tg=(64,)),
            locomp.Config(BLOCK_SIZE=128,
                          grid=lambda N, BLOCK_SIZE, **kw: (N // BLOCK_SIZE,),
                          tg=(128,)),
        ],
        key=["N"],
        warmup=1, rep=2,
    )
    @locomp.kernel
    def gelu_at(X: locomp.Tensor, O: locomp.Tensor,
                N: locomp.constexpr, BLOCK_SIZE: locomp.constexpr):
        pid = locomp.program_id(0)
        tid = locomp.local_id(0)
        idx = pid * BLOCK_SIZE + tid
        x = locomp.load(X + idx)
        inner = locomp.clamp(0.7978845608 * (x + 0.044715 * x * x * x), -10.0, 10.0)
        locomp.store(O + idx, 0.5 * x * (1.0 + locomp.tanh(inner)))

    N = 512
    x = np.random.randn(N).astype(np.float32)
    X_t = locomp.tensor(x)
    O_t = locomp.empty(N)

    gelu_at(X_t, O_t, N)
    result = O_t.numpy()

    # NumPy reference GELU
    inner = np.clip(0.7978845608 * (x + 0.044715 * x**3), -10.0, 10.0)
    expected = 0.5 * x * (1.0 + np.tanh(inner))
    assert np.max(np.abs(result - expected)) < 1e-4

    # Second call — must use cached config (fast)
    gelu_at(X_t, O_t, N)
    result2 = O_t.numpy()
    assert np.max(np.abs(result2 - expected)) < 1e-4
