"""Tests for pedal_model.data.dataset."""
import numpy as np
import pytest
import torch

from pedal_model.data.dataset import ChunkDataset, PedalDataset

SR = 48000


def _make_pair(n: int = SR * 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dry = rng.standard_normal(n).astype(np.float32) * 0.3
    wet = np.tanh(dry * 2.0).astype(np.float32)
    return dry, wet


class TestPedalDataset:
    def test_length(self):
        dry, wet = _make_pair(n=1000)
        rf, hop = 32, 1
        ds = PedalDataset(dry, wet, receptive_field=rf, hop=hop)
        expected = len(range(0, len(wet) - rf + 1, hop))
        assert len(ds) == expected

    def test_item_shapes(self):
        dry, wet = _make_pair(n=1000)
        rf = 64
        ds = PedalDataset(dry, wet, receptive_field=rf)
        window, target = ds[0]
        assert window.shape == (rf,)
        assert target.shape == ()

    def test_item_dtype(self):
        dry, wet = _make_pair(n=1000)
        ds = PedalDataset(dry, wet, receptive_field=32)
        window, target = ds[0]
        assert window.dtype == torch.float32
        assert target.dtype == torch.float32

    def test_hop_reduces_length(self):
        dry, wet = _make_pair(n=4000)
        rf = 32
        ds_hop1 = PedalDataset(dry, wet, receptive_field=rf, hop=1)
        ds_hop8 = PedalDataset(dry, wet, receptive_field=rf, hop=8)
        assert len(ds_hop8) < len(ds_hop1)

    def test_mismatched_lengths_raise(self):
        dry = np.zeros(1000, dtype=np.float32)
        wet = np.zeros(900, dtype=np.float32)
        with pytest.raises(AssertionError):
            PedalDataset(dry, wet, receptive_field=32)


class TestChunkDataset:
    def test_length(self):
        dry, wet = _make_pair(n=SR * 2)
        chunk_size = 2048
        ds = ChunkDataset(dry, wet, chunk_size=chunk_size)
        expected = len(dry) // chunk_size
        assert len(ds) == expected

    def test_item_shapes(self):
        dry, wet = _make_pair(n=SR)
        chunk_size = 512
        ds = ChunkDataset(dry, wet, chunk_size=chunk_size)
        d, w = ds[0]
        assert d.shape == (chunk_size, 1)
        assert w.shape == (chunk_size, 1)

    def test_item_dtype(self):
        dry, wet = _make_pair(n=SR)
        ds = ChunkDataset(dry, wet, chunk_size=256)
        d, w = ds[0]
        assert d.dtype == torch.float32
        assert w.dtype == torch.float32
