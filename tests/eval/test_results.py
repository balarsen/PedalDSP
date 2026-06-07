"""Tests for pedal_model.eval.results — ModelResult and ZooResults."""
import numpy as np
import pytest

from pedal_model.eval.results import ModelResult, ZooResults


def _make_zoo(n_sections: int = 3) -> ZooResults:
    np.random.seed(0)
    zoo = ZooResults(
        config={"epochs": 10},
        date_run="2026-01-01T00:00:00Z",
        sr=48_000,
        dry_val=np.zeros(100, dtype=np.float32),
        wet_val=np.zeros(100, dtype=np.float32),
    )
    for model_name in ("fir", "lstm"):
        sections = {
            f"sec_{i}": {"ESR": float(i + 1) * 0.1, "STFT": float(i) * 0.5}
            for i in range(n_sections)
        }
        zoo.models[model_name] = ModelResult(
            name=model_name,
            model_type="classical" if model_name == "fir" else "neural",
            prediction=np.random.randn(100).astype(np.float32),
            fit_time_s=1.0,
            predict_time_s=0.1,
            metrics_per_section=sections,
        )
    return zoo


def test_model_result_stores_fields():
    pred = np.random.randn(50).astype(np.float32)
    mr = ModelResult(
        name="test",
        model_type="classical",
        prediction=pred,
        fit_time_s=2.5,
        predict_time_s=0.3,
        metrics_per_section={"s1": {"ESR": 0.1}},
    )
    assert mr.name == "test"
    assert mr.model_type == "classical"
    np.testing.assert_array_equal(mr.prediction, pred)
    assert mr.fit_time_s == 2.5
    assert mr.ddsp_analysis is None


def test_zoo_results_save_load_roundtrip(tmp_path):
    zoo = _make_zoo()
    pkl = tmp_path / "results.pkl"
    zoo.save(pkl)
    loaded = ZooResults.load(pkl)
    assert loaded.sr == zoo.sr
    assert loaded.date_run == zoo.date_run
    assert set(loaded.models.keys()) == set(zoo.models.keys())
    np.testing.assert_array_equal(loaded.dry_val, zoo.dry_val)


def test_zoo_results_save_creates_parent_dir(tmp_path):
    zoo = _make_zoo()
    nested = tmp_path / "deep" / "nested" / "results.pkl"
    zoo.save(nested)
    assert nested.exists()


def test_zoo_results_load_wrong_type_raises(tmp_path):
    import pickle
    bad_pkl = tmp_path / "bad.pkl"
    with open(bad_pkl, "wb") as f:
        pickle.dump({"not": "a ZooResults"}, f)
    with pytest.raises(TypeError):
        ZooResults.load(bad_pkl)


def test_zoo_results_to_dataframe_shape():
    zoo = _make_zoo(n_sections=4)
    df = zoo.to_dataframe()
    assert set(df.index) == {"fir", "lstm"}
    assert "ESR" in df.columns
    assert "STFT" in df.columns
    assert "fit_s" in df.columns


def test_zoo_results_to_dataframe_mean_is_correct():
    zoo = _make_zoo(n_sections=2)
    df = zoo.to_dataframe(aggregate="mean")
    # fir sections: ESR = [0.1, 0.2] → mean = 0.15
    assert df.loc["fir", "ESR"] == pytest.approx(0.15, abs=1e-6)


def test_zoo_results_to_dataframe_empty_models():
    zoo = ZooResults(config={}, date_run="", sr=48000,
                     dry_val=np.zeros(10, dtype=np.float32),
                     wet_val=np.zeros(10, dtype=np.float32))
    df = zoo.to_dataframe()
    assert df.empty


def test_zoo_results_to_csv(tmp_path):
    zoo = _make_zoo()
    csv_path = tmp_path / "out.csv"
    zoo.to_csv(csv_path)
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "fir" in content
    assert "ESR" in content
