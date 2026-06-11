"""Stateful streaming inference for all PedalDSP model types.

Each model processes a full audio buffer in one contiguous pass, preserving
hidden state from start to finish — simulating real-time guitar playback.

For audio longer than 10 minutes at 96 kHz (57.6 M samples), a chunked
overlap-save fallback is used automatically to avoid VRAM exhaustion.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np

_LOG = logging.getLogger(__name__)

# 10 minutes at 96 kHz — above this, fall back to chunked processing
_CHUNK_THRESHOLD = 10 * 60 * 96_000  # 57_600_000 samples
_CHUNK_SIZE = 262_144
_CHUNK_OVERLAP = _CHUNK_SIZE // 2  # 131_072 — first half is state warmup

# Soft limiter activates when peak exceeds this level after RMS normalization.
# -0.1 dBFS = 10^(-0.1/20) ≈ 0.9886. Gives a tiny margin before digital clipping.
_SOFT_LIMIT_THRESHOLD = 10.0 ** (-0.1 / 20.0)


def _is_neural(model: object) -> bool:
    """Return True if model is a torch.nn.Module (neural), False for classical."""
    try:
        import torch.nn as nn
        return isinstance(model, nn.Module)
    except ImportError:
        return False


def _select_device() -> "torch.device":
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StreamRunner:
    """Runs a single model over a full audio buffer in one stateful pass.

    For neural models (LSTM, GRU, TCN) hidden state is initialized to zeros
    and carried through the entire signal. For classical models (FIR, IIR,
    Hammerstein, etc.) the model's own stateful filter implementation is used.

    Memory note: at 96 kHz a 3-minute guitar take is 17.3 M samples / ~66 MB.
    This fits comfortably in an RTX 2070's 8 GB VRAM for recurrent models
    (LSTM, GRU) that chunk internally. TCN processes the full tensor at once;
    for very long takes use a GPU with sufficient VRAM or accept the CPU fallback.

    If the signal exceeds 10 minutes, a warning is emitted and overlap-save
    chunked processing is used automatically (chunk = 262 144 samples, 50 %
    overlap for state warmup; output taken from the non-overlap region only).

    RMS normalization: when training_rms is provided, the input is scaled so
    its RMS matches the training distribution before inference. The output is
    un-scaled by the same factor. This prevents gain-staging issues when the
    input recording level differs from the training data. A soft tanh limiter
    fires as a last resort if the output still exceeds -0.1 dBFS after un-scaling.
    """

    def __init__(
        self,
        model: object,
        model_name: str,
        sr: int = 96_000,
        training_rms: float | None = None,
    ) -> None:
        """
        Args:
            model: A fitted PedalModel instance (classical or neural). Must
                implement a ``predict(audio: np.ndarray) -> np.ndarray`` method.
            model_name: Display name used in log messages.
            sr: Sample rate of the audio that will be passed to run(). Default 96 000.
            training_rms: RMS of the dry training signal. When set, the input is
                RMS-normalised to match before inference and the output is un-scaled.
                Use ``np.sqrt(np.mean(zoo.dry_val ** 2))`` to compute this value.
        """
        self.model = model
        self.model_name = model_name
        self.sr = sr
        self.training_rms = training_rms
        self._device = None

        if _is_neural(model):
            self._device = _select_device()
            model.to(self._device)  # type: ignore[attr-defined]
            _LOG.debug("StreamRunner[%s]: moved to %s", model_name, self._device)

    def run(self, audio: np.ndarray) -> np.ndarray:
        """Run inference over the full audio buffer.

        Applies RMS normalization if training_rms was provided, then runs the
        model, then un-scales the output. A soft tanh limiter is applied as a
        last resort if the output peak exceeds -0.1 dBFS after un-scaling.

        Args:
            audio: Float32 mono array at self.sr.

        Returns:
            Float32 mono array, same length as input.
        """
        scale = 1.0
        if self.training_rms is not None and self.training_rms > 0.0:
            input_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) + 1e-8
            scale = float(self.training_rms) / input_rms
            audio = (audio * scale).astype(np.float32)
            _LOG.info(
                "StreamRunner[%s]: RMS normalization input=%.4f training=%.4f scale=%.3f",
                self.model_name,
                input_rms,
                self.training_rms,
                scale,
            )

        n = len(audio)
        if n > _CHUNK_THRESHOLD:
            mins = n / self.sr / 60.0
            _LOG.warning(
                "StreamRunner[%s]: audio %.1f min exceeds 10-min threshold; "
                "using chunked overlap-save processing.",
                self.model_name,
                mins,
            )
            output = self._run_chunked(audio)
        else:
            output = self._predict(audio)

        if scale != 1.0:
            output = (output / scale).astype(np.float32)

        peak = float(np.max(np.abs(output)))
        if peak > _SOFT_LIMIT_THRESHOLD:
            peak_db = 20.0 * np.log10(peak + 1e-9)
            _LOG.warning(
                "StreamRunner[%s]: output peak %.1f dBFS after normalization — "
                "applying soft limiter. Check gain staging.",
                self.model_name,
                peak_db,
            )
            output = np.tanh(output).astype(np.float32)

        return output

    # ── Internal helpers ──────────────────────────────────────────────────── #

    def _predict(self, audio: np.ndarray) -> np.ndarray:
        """Single-call inference, no chunking."""
        import torch
        with torch.no_grad() if _is_neural(self.model) else _null_ctx():
            out = self.model.predict(audio)  # type: ignore[attr-defined]
        return np.asarray(out, dtype=np.float32)

    def _run_chunked(self, audio: np.ndarray) -> np.ndarray:
        """Overlap-save chunked inference for very long audio (> 10 min).

        Each step advances by CHUNK_SIZE - CHUNK_OVERLAP = CHUNK_OVERLAP samples.
        The leading CHUNK_OVERLAP samples of each chunk are model warmup;
        only the trailing half is written to the output.
        """
        n = len(audio)
        output = np.zeros(n, dtype=np.float32)
        step = _CHUNK_OVERLAP  # advance by half-chunk each iteration

        pos = 0
        while pos < n:
            buf_start = max(0, pos - _CHUNK_OVERLAP)
            buf_end = min(pos + step, n)

            chunk = audio[buf_start:buf_end]
            warmup_len = pos - buf_start  # 0 for first chunk, CHUNK_OVERLAP after
            pred = self._predict(chunk)

            out_len = buf_end - pos
            output[pos:buf_end] = pred[warmup_len: warmup_len + out_len]
            pos = buf_end

        return output


class _null_ctx:
    """No-op context manager for classical (non-torch) models."""
    def __enter__(self) -> "_null_ctx":
        return self

    def __exit__(self, *_: object) -> None:
        pass


class _CallableAdapter:
    """Wrap a bare callable as a predict()-method model for StreamRunner."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self._fn = fn

    def predict(self, audio: np.ndarray) -> np.ndarray:
        return np.asarray(self._fn(audio), dtype=np.float32)


# ── KlonSim loader ────────────────────────────────────────────────────────── #


def load_klonsim(results_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    """Load the KlonSim circuit simulation from a ZooResults directory.

    Reads params.json via the path stored in results.pkl config, instantiates
    KlonSim with the original pedal parameters, and returns a callable
    ``f(dry_audio) -> wet_audio`` that applies the full signal chain.

    This produces the same output as KlonSim.process() called during dataset
    creation — it is NOT a trained model approximation.

    Args:
        results_path: Directory containing results.pkl
            (e.g. results/data/klon_sim/klon_sim_v1).

    Returns:
        Callable that accepts a float32 mono numpy array and returns float32.

    Raises:
        FileNotFoundError: If results.pkl or params.json cannot be located.
        RuntimeError: If params.json path stored in results.pkl is invalid.
    """
    from pedal_model.eval.results import ZooResults
    from pedal_model.synthetic.pedal import KlonSim, KlonSimParams

    results_path = Path(results_path)
    pkl_path = results_path / "results.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"results.pkl not found at {pkl_path}")

    zoo = ZooResults.load(pkl_path)
    sr = zoo.sr

    params_path = Path(zoo.config.get("params_path", ""))
    if not params_path.exists():
        raise RuntimeError(
            f"params.json path '{params_path}' from results.pkl does not exist. "
            "Ensure the klon_sim data directory is accessible."
        )

    with open(params_path) as fh:
        d = json.load(fh)

    sim = KlonSim(KlonSimParams(**d["params"]))
    _LOG.info("load_klonsim: loaded KlonSim v%s from %s", sim.VERSION, params_path)

    def _process(audio: np.ndarray) -> np.ndarray:
        result = sim.process(audio, sr)
        peak_db = 20.0 * np.log10(float(np.max(np.abs(result))) + 1e-9)
        _LOG.info("load_klonsim: raw output peak = %.2f dBFS", peak_db)
        return result

    return _process
