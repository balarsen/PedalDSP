"""Render model comparison results as a colour-coded seaborn heatmap."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Metrics where lower = better (normalise as 1 - normalised)
_ERROR_METRICS = {"ESR", "MSE", "DC_err", "RMS_err", "STFT", "FR_err_dB", "THD_err", "MCD"}
# Metrics where higher = better
_SIMILARITY_METRICS = {"HP_sim"}


def render_heatmap(
    results: dict[str, float],
    pedal_name: str,
    save_path: Path | str | None = None,
) -> None:
    """Render a green/yellow/red comparison heatmap.

    Args:
        results: Dict mapping model_name → metrics dict (from compare.py).
        pedal_name: Label shown in the plot title.
        save_path: If given, save the PNG here; otherwise display interactively.
    """
    df = pd.DataFrame(results).T  # rows = models, cols = metrics

    normed = df.copy()
    for col in df.columns:
        col_min, col_max = df[col].min(), df[col].max()
        span = col_max - col_min + 1e-10
        if col in _ERROR_METRICS:
            normed[col] = 1.0 - (df[col] - col_min) / span
        else:
            normed[col] = (df[col] - col_min) / span

    fig, ax = plt.subplots(figsize=(14, max(4, len(results) * 0.6 + 2)))
    sns.heatmap(
        normed,
        annot=df.round(4),
        fmt="",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Model Comparison — {pedal_name}", fontsize=15, pad=16)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Model")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150)
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def render_all(
    all_results: dict[str, dict[str, dict[str, float]]],
    output_dir: Path | str = ".",
) -> None:
    """Render one heatmap per capture in all_results.

    Args:
        all_results: Output of compare.compare_models().
        output_dir: Directory to save PNG files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for capture_name, model_results in all_results.items():
        save = output_dir / f"comparison_{capture_name}.png"
        render_heatmap(model_results, pedal_name=capture_name, save_path=save)
