"""Tests for pedal_model.eval.compare.compare_models."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


class _FakeModel:
    def __init__(self, name: str) -> None:
        self.name = name

    def predict(self, dry: np.ndarray) -> np.ndarray:
        return dry.copy()


def test_compare_models_returns_nested_dict():
    from pedal_model.eval.compare import compare_models

    models = [_FakeModel("fir")]
    fake_metrics = {"ESR": 0.12, "STFT": 0.05}

    with patch("pedal_model.eval.compare.evaluate_model", return_value=fake_metrics):
        result = compare_models(models, ["cap_a.wav", "cap_b.wav"])

    assert "cap_a" in result
    assert "cap_b" in result
    assert result["cap_a"]["fir"]["ESR"] == pytest.approx(0.12)
    assert result["cap_b"]["fir"]["STFT"] == pytest.approx(0.05)


def test_compare_models_uses_path_stem_as_key():
    from pedal_model.eval.compare import compare_models

    models = [_FakeModel("lstm")]
    with patch("pedal_model.eval.compare.evaluate_model", return_value={}):
        result = compare_models(
            models,
            [Path("/some/dir/my_capture.wav")],
        )
    assert "my_capture" in result


def test_compare_models_multiple_models_called_for_each_capture():
    from pedal_model.eval.compare import compare_models

    models = [_FakeModel("fir"), _FakeModel("tcn")]
    call_log: list[str] = []

    def _fake_eval(model, path, *args, **kwargs):
        call_log.append(model.name)
        return {"ESR": 0.1}

    with patch("pedal_model.eval.compare.evaluate_model", side_effect=_fake_eval):
        result = compare_models(models, ["a.wav", "b.wav"])

    # 2 models × 2 captures = 4 total calls
    assert len(call_log) == 4
    assert set(result["a"].keys()) == {"fir", "tcn"}
    assert set(result["b"].keys()) == {"fir", "tcn"}


def test_compare_models_failed_model_returns_empty_dict():
    from pedal_model.eval.compare import compare_models

    models = [_FakeModel("fir"), _FakeModel("lstm")]

    def _fail_lstm(model, path, *args, **kwargs):
        if model.name == "lstm":
            raise RuntimeError("training failed")
        return {"ESR": 0.5}

    with patch("pedal_model.eval.compare.evaluate_model", side_effect=_fail_lstm):
        result = compare_models(models, ["cap.wav"])

    assert result["cap"]["fir"]["ESR"] == pytest.approx(0.5)
    assert result["cap"]["lstm"] == {}


def test_compare_models_empty_captures_returns_empty():
    from pedal_model.eval.compare import compare_models

    models = [_FakeModel("fir")]
    with patch("pedal_model.eval.compare.evaluate_model", return_value={}):
        result = compare_models(models, [])
    assert result == {}


def test_compare_models_empty_models_list():
    from pedal_model.eval.compare import compare_models

    with patch("pedal_model.eval.compare.evaluate_model", return_value={}) as mock:
        result = compare_models([], ["a.wav"])
    mock.assert_not_called()
    assert result["a"] == {}
