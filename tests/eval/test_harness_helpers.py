"""Tests for helper functions in pedal_model.eval.harness."""
from pathlib import Path

import numpy as np
import pytest
import torch

from pedal_model.eval.harness import (
    EvalConfig,
    _bad_block_set,
    _filter_seq_blocks,
    _resolve_device,
    _select_seq_blocks,
)
from pedal_model.signals.manifest import Section


# ── Fake manifest duck-type for _bad_block_set ──────────────────────────── #

class _FakeManifest:
    def __init__(self, sections: list[Section]) -> None:
        self.sections = sections

    def sections_of_type(self, type_str: str) -> list[Section]:
        return [s for s in self.sections if s.type == type_str]


def _sec(type_str: str, start: int, end: int, label: str = "s") -> Section:
    return Section(
        label=label, type=type_str, index=0,
        start_sample=start, end_sample=end,
        start_s=0.0, end_s=1.0,
        amplitude_dbfs=-12.0, params={},
    )


def test_select_seq_blocks_total_length():
    np.random.seed(0)
    dry = np.random.randn(100_000).astype(np.float32)
    wet = np.random.randn(100_000).astype(np.float32)
    d, w = _select_seq_blocks(dry, wet, max_samples=8192, block_size=512)
    expected_blocks = 8192 // 512  # 16
    assert len(d) == expected_blocks * 512
    assert len(w) == len(d)


def test_select_seq_blocks_blocks_are_contiguous():
    """Each block must match the corresponding contiguous region of the source."""
    np.random.seed(1)
    N = 200_000
    dry = np.arange(N, dtype=np.float32)  # identity signal: dry[i] = i
    wet = np.arange(N, dtype=np.float32)
    block_size = 256
    max_samples = 2048
    d, _ = _select_seq_blocks(dry, wet, max_samples=max_samples, block_size=block_size)

    n_blocks = max_samples // block_size
    stride = N // n_blocks
    for i in range(n_blocks):
        src_start = i * stride
        blk_start = i * block_size
        # Each block should be an unbroken run from the source
        expected = dry[src_start : src_start + block_size]
        actual = d[blk_start : blk_start + block_size]
        np.testing.assert_array_equal(actual, expected, err_msg=f"Block {i} is not contiguous")


def test_select_seq_blocks_covers_full_signal_range():
    """Block start indices should span from near 0 to near N-block_size."""
    np.random.seed(2)
    N = 50_000
    dry = np.arange(N, dtype=np.float32)
    wet = np.zeros(N, dtype=np.float32)
    block_size = 512
    n_blocks = 8
    max_samples = n_blocks * block_size
    d, _ = _select_seq_blocks(dry, wet, max_samples=max_samples, block_size=block_size)

    # First block should start near index 0
    assert d[0] < block_size
    # Last block should start well into the signal (not just the beginning)
    last_block_first_val = d[-block_size]
    assert last_block_first_val > N // 2


def test_select_seq_blocks_when_signal_fits_entirely():
    """When N <= max_samples, return full signal trimmed to block boundary."""
    np.random.seed(3)
    dry = np.random.randn(10_000).astype(np.float32)
    wet = np.random.randn(10_000).astype(np.float32)
    d, w = _select_seq_blocks(dry, wet, max_samples=50_000, block_size=4096)
    expected_len = (10_000 // 4096) * 4096
    assert len(d) == expected_len
    np.testing.assert_array_equal(d, dry[:expected_len])


def test_select_seq_blocks_dtype_preserved():
    dry = np.ones(20_000, dtype=np.float32)
    wet = np.ones(20_000, dtype=np.float32)
    d, w = _select_seq_blocks(dry, wet, max_samples=4096, block_size=512)
    assert d.dtype == np.float32
    assert w.dtype == np.float32


# ── _bad_block_set ───────────────────────────────────────────────────────── #


def test_bad_block_set_all_active_returns_empty():
    n, cs = 16384, 4096
    m = _FakeManifest([_sec("guitar", 0, n)])
    bad = _bad_block_set(m, cs, n)
    assert len(bad) == 0


def test_bad_block_set_fully_silent_returns_all():
    n, cs = 16384, 4096
    m = _FakeManifest([])  # no sections → everything is silence
    bad = _bad_block_set(m, cs, n)
    assert bad == frozenset(range(n // cs))


def test_bad_block_set_marks_trailing_silence():
    n, cs = 16384, 4096
    # Only first two blocks covered; blocks 2 and 3 are silence
    m = _FakeManifest([_sec("guitar", 0, 2 * cs)])
    bad = _bad_block_set(m, cs, n)
    assert 2 in bad
    assert 3 in bad
    assert 0 not in bad
    assert 1 not in bad


def test_bad_block_set_partially_covered_block_is_bad():
    n, cs = 16384, 4096
    # Block 1 only has its first half covered → bad
    m = _FakeManifest([_sec("guitar", 0, cs + cs // 2)])
    bad = _bad_block_set(m, cs, n)
    assert 1 in bad
    assert 0 not in bad


def test_bad_block_set_marks_impulse_train_blocks():
    n, cs = 16384, 4096
    # Entire signal active, but block 1 contains an impulse_train
    m = _FakeManifest([
        _sec("guitar", 0, n),
        _sec("impulse_train", cs, 2 * cs, label="imp"),
    ])
    bad = _bad_block_set(m, cs, n)
    assert 1 in bad
    assert 0 not in bad
    assert 2 not in bad


def test_bad_block_set_impulse_train_spanning_two_blocks():
    n, cs = 16384, 4096
    m = _FakeManifest([
        _sec("guitar", 0, n),
        _sec("impulse_train", cs // 2, cs + cs // 2, label="imp"),
    ])
    bad = _bad_block_set(m, cs, n)
    assert 0 in bad  # partial overlap with block 0
    assert 1 in bad  # partial overlap with block 1


# ── _filter_seq_blocks ───────────────────────────────────────────────────── #


def test_filter_seq_blocks_no_bad_returns_unchanged():
    np.random.seed(10)
    n, cs = 8192, 4096
    dry = np.random.randn(n).astype(np.float32)
    wet = np.random.randn(n).astype(np.float32)
    d, w = _filter_seq_blocks(dry, wet, cs, frozenset())
    np.testing.assert_array_equal(d, dry)
    np.testing.assert_array_equal(w, wet)


def test_filter_seq_blocks_removes_bad_block():
    n, cs = 12288, 4096  # 3 blocks
    dry = np.arange(n, dtype=np.float32)
    wet = np.arange(n, dtype=np.float32) * 2.0
    d, w = _filter_seq_blocks(dry, wet, cs, frozenset([1]))
    assert len(d) == 2 * cs
    np.testing.assert_array_equal(d[:cs], dry[:cs])
    np.testing.assert_array_equal(d[cs:], dry[2 * cs:])


def test_filter_seq_blocks_removes_multiple_bad_blocks():
    n, cs = 16384, 4096  # 4 blocks, remove 0 and 2
    dry = np.arange(n, dtype=np.float32)
    wet = np.zeros(n, dtype=np.float32)
    d, w = _filter_seq_blocks(dry, wet, cs, frozenset([0, 2]))
    assert len(d) == 2 * cs
    np.testing.assert_array_equal(d[:cs], dry[cs: 2 * cs])
    np.testing.assert_array_equal(d[cs:], dry[3 * cs:])


def test_filter_seq_blocks_output_is_contiguous_float32():
    n, cs = 8192, 4096
    dry = np.random.randn(n).astype(np.float32)
    wet = np.random.randn(n).astype(np.float32)
    d, w = _filter_seq_blocks(dry, wet, cs, frozenset([0]))
    assert d.dtype == np.float32
    assert d.flags["C_CONTIGUOUS"]


# ── EvalConfig ────────────────────────────────────────────────────────────── #


def test_eval_config_defaults():
    cfg = EvalConfig()
    assert isinstance(cfg.klon_sim_dir, Path)
    assert isinstance(cfg.output_dir, Path)
    assert cfg.neural_epochs == 50
    assert cfg.device == "auto"
    assert cfg.run_classical is True
    assert cfg.run_neural is True
    assert cfg.dry_wav is None
    assert cfg.wet_wav is None
    assert cfg.dry_val_wav is None
    assert cfg.wet_val_wav is None
    assert cfg.experiment_name is None


def test_eval_config_wav_overrides_coerced_to_path(tmp_path):
    p = tmp_path / "dry.wav"
    p.touch()
    cfg = EvalConfig(
        dry_wav=str(p),
        wet_wav=str(p),
        dry_val_wav=str(p),
        wet_val_wav=str(p),
    )
    assert isinstance(cfg.dry_wav, Path)
    assert cfg.wet_wav == Path(str(p))


def test_eval_config_experiment_name():
    cfg = EvalConfig(experiment_name="medley_run")
    assert cfg.experiment_name == "medley_run"


def test_eval_config_skip_models_default_empty():
    cfg = EvalConfig()
    assert cfg.skip_models == []


# ── _resolve_device ───────────────────────────────────────────────────────── #


def test_resolve_device_explicit_cpu():
    cfg = EvalConfig(device="cpu")
    dev = _resolve_device(cfg)
    assert dev == torch.device("cpu")


def test_resolve_device_auto_returns_torch_device():
    cfg = EvalConfig(device="auto")
    dev = _resolve_device(cfg)
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_resolve_device_auto_returns_cuda_when_available():
    cfg = EvalConfig(device="auto")
    dev = _resolve_device(cfg)
    assert dev == torch.device("cuda")


def test_resolve_device_explicit_cuda_string():
    # Only check the type attribute — don't actually move tensors to GPU
    cfg = EvalConfig(device="cpu")
    dev = _resolve_device(cfg)
    assert dev.type == "cpu"
