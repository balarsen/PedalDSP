"""Tests for pedal_model.capture.verify."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from pedal_model.capture.verify import check_capture, plot_alignment

SR = 48_000


def _pair(n: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """Return clean dry/wet pair with guaranteed zero DC and peak well below 0.99."""
    t = np.arange(n, dtype=np.float32) / SR
    dry = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wet = (0.4 * np.sin(2 * np.pi * 440 * t) * 0.7).astype(np.float32)
    return dry, wet


class TestCheckCapture:
    def test_returns_dict_with_required_keys(self):
        dry, wet = _pair()
        stats = check_capture(dry, wet, SR)
        assert set(stats.keys()) == {"dry_peak", "wet_peak", "dry_dc", "wet_dc", "duration_s"}

    def test_peak_values_are_correct(self):
        dry = np.full(SR, 0.5, dtype=np.float32)
        wet = np.full(SR, 0.25, dtype=np.float32)
        stats = check_capture(dry, wet, SR)
        assert stats["dry_peak"] == pytest.approx(0.5, abs=1e-5)
        assert stats["wet_peak"] == pytest.approx(0.25, abs=1e-5)

    def test_dc_values_are_correct(self):
        dry = np.full(SR, 0.01, dtype=np.float32)
        wet = np.full(SR, -0.02, dtype=np.float32)
        stats = check_capture(dry, wet, SR)
        assert stats["dry_dc"] == pytest.approx(0.01, abs=1e-5)
        assert stats["wet_dc"] == pytest.approx(-0.02, abs=1e-5)

    def test_duration_matches_sr(self):
        dry, wet = _pair(n=SR * 3)
        stats = check_capture(dry, wet, SR)
        assert stats["duration_s"] == pytest.approx(3.0, abs=1e-3)

    def test_clipping_warning_printed(self, capsys):
        dry = np.full(SR, 0.99, dtype=np.float32)
        wet = np.full(SR, 0.99, dtype=np.float32)
        check_capture(dry, wet, SR)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_dc_warning_printed(self, capsys):
        dry = np.full(SR, 0.01, dtype=np.float32)
        wet = np.full(SR, 0.1, dtype=np.float32)  # DC > 1e-3
        check_capture(dry, wet, SR)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_clean_signal_no_warning(self, capsys):
        dry, wet = _pair()
        check_capture(dry, wet, SR)
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out

    def test_values_are_float(self):
        dry, wet = _pair()
        stats = check_capture(dry, wet, SR)
        for v in stats.values():
            assert isinstance(v, float)


class TestPlotAlignment:
    def test_saves_to_file(self, tmp_path):
        dry, wet = _pair()
        save = tmp_path / "align.png"
        plot_alignment(dry, wet, SR, seconds=0.5, save_path=save)
        assert save.exists()
        assert save.stat().st_size > 0

    def test_short_signal_no_crash(self, tmp_path):
        dry = np.zeros(SR, dtype=np.float32)
        wet = np.zeros(SR, dtype=np.float32)
        save = tmp_path / "short.png"
        plot_alignment(dry, wet, SR, seconds=0.5, save_path=save)
        assert save.exists()

    def test_custom_seconds(self, tmp_path):
        dry, wet = _pair(n=SR * 5)
        save = tmp_path / "align_short.png"
        plot_alignment(dry, wet, SR, seconds=0.5, save_path=save)
        assert save.exists()
