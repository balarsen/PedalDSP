"""Tests for helper functions in pedal_model.eval.harness."""
import numpy as np
import pytest

from pedal_model.eval.harness import _select_seq_blocks


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
