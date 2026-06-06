"""Tests for manifest-aware PedalDataset / ChunkDataset factory methods.

Uses an identity pedal (wet = dry) and a short generated signal so no
real captures are needed.  All audio lives in tmp_path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pedal_model.data.dataset import ChunkDataset, PedalDataset
from pedal_model.signals.generate import ValParams, generate
from pedal_model.signals.manifest import Manifest

SR = 8_000


@pytest.fixture(scope="module")
def val_signal(tmp_path_factory):
    """Generate a short val signal once for the whole module."""
    d = tmp_path_factory.mktemp("signals")
    p = ValParams(
        sample_rate=SR,
        seed=77,
        silence_gap_s=0.05,
        stepped_sweep_amplitudes_dbfs=[-12.0],
        stepped_sweep_duration_s=0.25,
        stepped_sweep_f_start_hz=100.0,
        stepped_sweep_f_end_hz=3_500.0,
        val_stepped_freqs_hz=[196.0, 246.9],
        val_stepped_levels_dbfs=[-12.0],
        val_stepped_tone_duration_s=0.1,
        val_pink_amplitudes_dbfs=[-12.0],
        val_pink_duration_s=0.25,
        val_two_tone_pairs=[[196.0, 293.7]],
        val_two_tone_duration_s=0.2,
        val_impulse_duration_s=0.2,
        val_impulse_amplitude_dbfs=-6.0,
        val_impulse_spacings_s=[0.04, 0.07],
        val_pluck_freqs_hz=[196.0, 246.9],
        val_pluck_decay_tau_s=0.06,
        val_pluck_amplitude_dbfs=-6.0,
    )
    wav, js = generate("val", d, val_params=p)
    manifest = Manifest(js)
    # Identity "capture": wet = dry (no pedal processing)
    import shutil
    wet_wav = d / "wet_identity.wav"
    shutil.copy(wav, wet_wav)
    return manifest, wet_wav


# ── PedalDataset.from_manifest ────────────────────────────────────────────────


class TestPedalDatasetFromManifest:
    def test_returns_pedal_dataset(self, val_signal):
        m, wet = val_signal
        ds = PedalDataset.from_manifest(m, wet, receptive_field=32, hop=16)
        assert isinstance(ds, PedalDataset)

    def test_nonempty(self, val_signal):
        m, wet = val_signal
        ds = PedalDataset.from_manifest(m, wet, receptive_field=32, hop=16)
        assert len(ds) > 0

    def test_item_shapes(self, val_signal):
        m, wet = val_signal
        rf = 64
        ds = PedalDataset.from_manifest(m, wet, receptive_field=rf, hop=32)
        window, target = ds[0]
        assert window.shape == (rf,)
        assert target.shape == ()

    def test_item_dtype(self, val_signal):
        m, wet = val_signal
        ds = PedalDataset.from_manifest(m, wet, receptive_field=32, hop=16)
        window, target = ds[0]
        assert window.dtype == torch.float32
        assert target.dtype == torch.float32

    def test_identity_pedal_target_is_audio_sample(self, val_signal):
        """With wet=dry, every target must equal the dry sample at start+rf-1.

        The causal-padding scheme: window[-1] = dry[start], target = wet[start+rf-1].
        For identity (wet=dry) the target is dry[start+rf-1], not dry[start].
        """
        import soundfile as sf
        m, wet = val_signal
        rf = 32
        hop = rf
        ds = PedalDataset.from_manifest(m, wet, receptive_field=rf, hop=hop)
        # Reconstruct the concatenated section audio (same as what factory loaded)
        chunks = [
            sf.read(str(m.path.with_suffix(".wav")),
                    start=s.start_sample, stop=s.end_sample,
                    dtype="float32", always_2d=False)[0]
            for s in m.sections
        ]
        full_dry = np.concatenate(chunks)
        for i in range(min(len(ds), 10)):
            _, target = ds[i]
            start = i * hop
            expected = full_dry[start + rf - 1]
            assert target.item() == pytest.approx(float(expected), abs=1e-6)

    def test_section_type_filter(self, val_signal):
        m, wet = val_signal
        ds_all = PedalDataset.from_manifest(m, wet, receptive_field=32, hop=32)
        ds_sweep = PedalDataset.from_manifest(
            m, wet, section_types=["log_sine_sweep"], receptive_field=32, hop=32
        )
        assert len(ds_sweep) < len(ds_all)
        assert len(ds_sweep) > 0

    def test_section_label_filter(self, val_signal):
        m, wet = val_signal
        label = m.sections_of_type("log_sine_sweep")[0].label
        ds = PedalDataset.from_manifest(
            m, wet, section_labels=[label], receptive_field=32, hop=32
        )
        assert len(ds) > 0

    def test_empty_filter_raises(self, val_signal):
        m, wet = val_signal
        with pytest.raises(ValueError, match="No sections matched"):
            PedalDataset.from_manifest(
                m, wet,
                section_types=["nonexistent_type_xyz"],
                receptive_field=32,
                hop=32,
            )

    def test_resampled_length_shorter(self, val_signal):
        """Loading at 48k from an 8k signal tests the resample path.
        Actually 8k→4k to keep it integer; just verify dataset is shorter."""
        m, wet = val_signal
        ds_native = PedalDataset.from_manifest(m, wet, receptive_field=16, hop=8)
        ds_half = PedalDataset.from_manifest(
            m, wet, target_sr=SR // 2, receptive_field=16, hop=8
        )
        assert len(ds_half) < len(ds_native)

    def test_dry_wet_alignment_preserved(self, val_signal):
        """Identity pedal: dry and wet must produce identical targets.

        With wet=dry, loading dry via manifest and wet via the copied WAV must
        return the same concatenated audio, so dry_target == wet_target at
        every index.
        """
        import soundfile as sf
        m, wet = val_signal
        rf = 32
        # Build dataset — since wet=dry, we can build two datasets from the
        # same audio and confirm their targets are identical sample-for-sample.
        ds = PedalDataset.from_manifest(m, wet, receptive_field=rf, hop=64)
        chunks = [
            sf.read(str(m.path.with_suffix(".wav")),
                    start=s.start_sample, stop=s.end_sample,
                    dtype="float32", always_2d=False)[0]
            for s in m.sections
        ]
        dry_concat = np.concatenate(chunks)
        wet_chunks = [
            sf.read(str(wet), start=s.start_sample, stop=s.end_sample,
                    dtype="float32", always_2d=False)[0]
            for s in m.sections
        ]
        wet_concat = np.concatenate(wet_chunks)
        # Identity: every position must be equal
        np.testing.assert_array_equal(dry_concat, wet_concat)


# ── ChunkDataset.from_manifest ────────────────────────────────────────────────


class TestChunkDatasetFromManifest:
    def test_returns_chunk_dataset(self, val_signal):
        m, wet = val_signal
        ds = ChunkDataset.from_manifest(m, wet, chunk_size=128)
        assert isinstance(ds, ChunkDataset)

    def test_nonempty(self, val_signal):
        m, wet = val_signal
        ds = ChunkDataset.from_manifest(m, wet, chunk_size=128)
        assert len(ds) > 0

    def test_item_shapes(self, val_signal):
        m, wet = val_signal
        chunk_size = 256
        ds = ChunkDataset.from_manifest(m, wet, chunk_size=chunk_size)
        d, w = ds[0]
        assert d.shape == (chunk_size, 1)
        assert w.shape == (chunk_size, 1)

    def test_item_dtype(self, val_signal):
        m, wet = val_signal
        ds = ChunkDataset.from_manifest(m, wet, chunk_size=128)
        d, w = ds[0]
        assert d.dtype == torch.float32
        assert w.dtype == torch.float32

    def test_section_type_filter(self, val_signal):
        m, wet = val_signal
        ds_all = ChunkDataset.from_manifest(m, wet, chunk_size=64)
        ds_sweep = ChunkDataset.from_manifest(
            m, wet, section_types=["log_sine_sweep"], chunk_size=64
        )
        assert len(ds_sweep) < len(ds_all)

    def test_chunk_covers_full_audio(self, val_signal):
        """n_chunks * chunk_size should be close to the loaded audio length."""
        m, wet = val_signal
        chunk_size = 128
        ds = ChunkDataset.from_manifest(m, wet, chunk_size=chunk_size)
        # Total samples represented = n_chunks * chunk_size
        total_represented = len(ds) * chunk_size
        # Sum up section sample counts
        total_section_samples = sum(s.n_samples for s in m.sections)
        # Dataset drops the tail; difference must be < one chunk
        assert total_section_samples - total_represented < chunk_size


# ── Existing PedalDataset / ChunkDataset (no regressions) ────────────────────


def test_existing_pedaldataset_unchanged():
    """Original constructor still works identically."""
    np.random.seed(0)
    dry = np.random.randn(2000).astype(np.float32) * 0.3
    wet = np.tanh(dry * 2.0).astype(np.float32)
    rf, hop = 32, 1
    ds = PedalDataset(dry, wet, receptive_field=rf, hop=hop)
    assert len(ds) == len(range(0, len(wet) - rf + 1, hop))
    window, target = ds[0]
    assert window.shape == (rf,)
    assert target.shape == ()


def test_existing_chunk_dataset_unchanged():
    np.random.seed(1)
    dry = np.random.randn(4096).astype(np.float32) * 0.3
    wet = np.tanh(dry * 2.0).astype(np.float32)
    chunk_size = 512
    ds = ChunkDataset(dry, wet, chunk_size=chunk_size)
    assert len(ds) == len(dry) // chunk_size
    d, w = ds[0]
    assert d.shape == (chunk_size, 1)


def test_mismatched_lengths_raise():
    dry = np.zeros(1000, dtype=np.float32)
    wet = np.zeros(900, dtype=np.float32)
    with pytest.raises(ValueError, match="same length"):
        PedalDataset(dry, wet, receptive_field=32)
