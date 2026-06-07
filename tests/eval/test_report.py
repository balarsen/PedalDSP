"""Tests for pedal_model.eval.report."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import pytest

from pedal_model.eval.report import render_all, render_heatmap


def _sample_results() -> dict[str, dict[str, float]]:
    return {
        "fir": {"ESR": 0.3, "MSE": 0.05, "HP_sim": 0.8},
        "lstm": {"ESR": 0.1, "MSE": 0.01, "HP_sim": 0.95},
        "tcn": {"ESR": 0.15, "MSE": 0.02, "HP_sim": 0.92},
    }


class TestRenderHeatmap:
    def test_save_creates_file(self, tmp_path):
        save = tmp_path / "heatmap.png"
        render_heatmap(_sample_results(), pedal_name="klon_test", save_path=save)
        assert save.exists()
        assert save.stat().st_size > 0

    def test_no_crash_without_save(self):
        """When save_path=None the function calls plt.show() — must not raise."""
        import matplotlib.pyplot as plt
        render_heatmap(_sample_results(), pedal_name="klon")
        plt.close("all")

    def test_single_model_no_crash(self, tmp_path):
        save = tmp_path / "single.png"
        render_heatmap({"fir": {"ESR": 0.5}}, pedal_name="test", save_path=save)
        assert save.exists()

    def test_error_metrics_are_inverted_lower_is_greener(self, tmp_path):
        """Save a figure with only ESR (error metric) — just check no exception."""
        save = tmp_path / "esr_only.png"
        render_heatmap(
            {"a": {"ESR": 0.1}, "b": {"ESR": 0.9}},
            pedal_name="inversion_test",
            save_path=save,
        )
        assert save.exists()

    def test_similarity_metric_not_inverted(self, tmp_path):
        """HP_sim is higher-is-better — should render without inversion crash."""
        save = tmp_path / "sim.png"
        render_heatmap(
            {"a": {"HP_sim": 0.9}, "b": {"HP_sim": 0.5}},
            pedal_name="sim_test",
            save_path=save,
        )
        assert save.exists()

    def test_all_models_same_value_no_crash(self, tmp_path):
        """Identical values → span=0 → should not divide by zero."""
        save = tmp_path / "flat.png"
        render_heatmap(
            {"a": {"ESR": 0.5}, "b": {"ESR": 0.5}},
            pedal_name="flat",
            save_path=save,
        )
        assert save.exists()


class TestRenderAll:
    def test_creates_one_file_per_capture(self, tmp_path):
        all_results = {
            "capture_A": _sample_results(),
            "capture_B": _sample_results(),
        }
        render_all(all_results, output_dir=tmp_path)
        files = list(tmp_path.glob("comparison_*.png"))
        assert len(files) == 2

    def test_filenames_match_capture_names(self, tmp_path):
        render_all({"klon_run1": _sample_results()}, output_dir=tmp_path)
        assert (tmp_path / "comparison_klon_run1.png").exists()

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "new_dir" / "sub"
        render_all({"cap": _sample_results()}, output_dir=nested)
        assert nested.exists()

    def test_empty_all_results_no_crash(self, tmp_path):
        render_all({}, output_dir=tmp_path)
