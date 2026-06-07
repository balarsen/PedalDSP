"""ZooResults: structured storage for model zoo eval runs."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ModelResult:
    """Per-model results from one eval run.

    Attributes:
        name: Model display name.
        model_type: 'classical' or 'neural'.
        prediction: Full-signal val prediction, shape (N,), float32.
        fit_time_s: Time to fit/train the model in seconds.
        predict_time_s: Time to run inference on the val signal.
        metrics_per_section: Section label → metric name → float value.
        train_loss: Final epoch average training loss (neural models only).
        ddsp_analysis: Optional DDSP fraction analysis for KlonDDSP models.
    """

    name: str
    model_type: str
    prediction: np.ndarray
    fit_time_s: float
    predict_time_s: float
    metrics_per_section: dict[str, dict[str, float]] = field(default_factory=dict)
    train_loss: float | None = None
    ddsp_analysis: dict[str, float] | None = None
    topology: dict[str, float] | None = None


@dataclass
class ZooResults:
    """Full eval run results for a model zoo on one dataset.

    Attributes:
        config: Dict of eval config parameters (paths, epochs, etc.).
        date_run: ISO-8601 timestamp of when the run started.
        sr: Sample rate used for eval in Hz.
        dry_val: Dry validation signal, shape (N,), float32.
        wet_val: Wet (target) validation signal, shape (N,), float32.
        models: Mapping from model name to ModelResult.
    """

    config: dict
    date_run: str
    sr: int
    dry_val: np.ndarray
    wet_val: np.ndarray
    models: dict[str, ModelResult] = field(default_factory=dict)

    def to_dataframe(self, aggregate: str = "mean") -> pd.DataFrame:
        """Aggregate per-section metrics into a model × metric DataFrame.

        Args:
            aggregate: Aggregation function name accepted by pandas groupby.
                'mean', 'median', 'min', 'max'.

        Returns:
            DataFrame indexed by model name; columns are metric names.
            Rows ordered by model insertion order.
        """
        rows = []
        for model_name, result in self.models.items():
            section_rows = list(result.metrics_per_section.values())
            if not section_rows:
                continue
            df_sec = pd.DataFrame(section_rows)
            agg_row = getattr(df_sec, aggregate)(numeric_only=True).to_dict()
            agg_row["model"] = model_name
            agg_row["type"] = result.model_type
            agg_row["fit_s"] = result.fit_time_s
            agg_row["predict_s"] = result.predict_time_s
            rows.append(agg_row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("model")
        # Put timing columns last
        cols = [c for c in df.columns if c not in ("type", "fit_s", "predict_s")]
        df = df[cols + ["fit_s", "predict_s"]]
        return df

    def save(self, path: Path) -> None:
        """Pickle the full results to *path*.

        Args:
            path: Destination file. Parent directory is created if absent.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "ZooResults":
        """Load pickled results from *path*.

        Args:
            path: Path to a .pkl file saved by :meth:`save`.

        Returns:
            Deserialized :class:`ZooResults`.
        """
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ZooResults, got {type(obj).__name__}")
        return obj

    def to_csv(self, path: Path) -> None:
        """Write the aggregated metric table to a CSV file.

        Args:
            path: Destination .csv path. Parent directory is created if absent.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path)
