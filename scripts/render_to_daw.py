#!/usr/bin/env python3
"""render_to_daw.py — DAW-ready inference from a dry guitar take.

Takes a mono dry WAV (any sample rate), upsamples to 96 kHz, runs it through
KlonSim and each selected model, then downsamples back to 44.1 kHz / 24-bit
and writes organised WAV files ready to drag into Studio One (or any DAW).

Classical model caching:
    Fitted parameters are cached to {results}/fitted_params/{model}.pkl on the
    first run. Subsequent runs load from cache instead of re-fitting (Volterra
    and WienerHammerstein can each take several minutes to fit). Use
    --refit-classical to force re-fitting and update the cache.

Neural model loading:
    Prefers {results}/{model}_best.pt (saved by retrain_neural.py with
    epoch/val_esr metadata). Falls back to {results}/{model}_weights.pt with a
    warning. If neither exists the model is skipped.

    If results.pkl contains model_state (saved by a recent fit_models.py), neural
    models are loaded directly from the pkl without needing .pt files at all.

Sample rate handling:
    44.1 kHz: native — expected input rate.
    48.0 kHz: informational log only — valid first-class input.
    Any other: WARNING — will still work but results may be affected.
    All output files are written at 44.1 kHz / 24-bit regardless of input rate.

Usage:
    uv run python scripts/render_to_daw.py --input path/to/dry.wav \\
        --results results/data/klon_sim/klon_sim_v1

    # Select specific models:
    uv run python scripts/render_to_daw.py --input dry.wav \\
        --results results/data/klon_sim/klon_sim_v1 \\
        --models fir tcn

    # Force re-fitting classical models (clears cache):
    uv run python scripts/render_to_daw.py --input dry.wav \\
        --results results/data/klon_sim/klon_sim_v1 --refit-classical
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf

from pedal_model.eval.results import ZooResults
from pedal_model.inference.stream_runner import StreamRunner, _CallableAdapter, load_klonsim
from pedal_model.utils.resampling import load_wav_to_float32, write_wav_44k

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
_LOG = logging.getLogger(__name__)

# ── Model registries (mirrors pedal_model.eval.harness registries) ─────────── #

_CLASSICAL_MODELS: dict[str, tuple[str, str, dict]] = {
    "fir": ("pedal_model.models.classical.fir", "FIRModel", {"n_taps": 1024}),
    "iir": ("pedal_model.models.classical.iir", "IIRModel", {"order": 8}),
    "hammerstein": (
        "pedal_model.models.classical.hammerstein",
        "HammersteinModel",
        {"poly_order": 5},
    ),
    "wiener_hammerstein": (
        "pedal_model.models.classical.wiener_hammerstein",
        "WienerHammersteinModel",
        {"poly_order": 5},
    ),
    "volterra": (
        "pedal_model.models.classical.volterra",
        "VolterraModel",
        {"memory": 20},
    ),
    "circuit_dict": (
        "pedal_model.models.classical.circuit_dictionary",
        "CircuitDictionary",
        {"n_epochs": 200, "window_size": 480_000},
    ),
}

_NEURAL_MODELS: dict[str, tuple[str, str, dict]] = {
    "mlp": (
        "pedal_model.models.neural.mlp",
        "MLPModel",
        {"receptive_field_samples": 1024},
    ),
    "lstm": ("pedal_model.models.neural.lstm", "LSTMModel", {"hidden_size": 32}),
    "gru": ("pedal_model.models.neural.gru", "GRUModel", {"hidden_size": 32}),
    "tcn": ("pedal_model.models.neural.tcn", "TCNModel", {"channels": 32, "n_blocks": 10}),
    "ddsp": ("pedal_model.models.neural.ddsp", "DDSPModel", {"n_fir_taps": 64}),
    "klon_ddsp": (
        "pedal_model.models.neural.ddsp_klon",
        "KlonDDSP",
        {"gru_hidden": 16},
    ),
}

_ALL_MODEL_KEYS = list(_CLASSICAL_MODELS) + list(_NEURAL_MODELS)


# ── Helpers ────────────────────────────────────────────────────────────────── #


def _db(amplitude: float) -> str:
    val = 20.0 * np.log10(float(amplitude) + 1e-9)
    return f"{val:+.1f} dBFS"


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _load_training_data(
    params_path: Path,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load dry and wet training signals from params.json paths.

    Args:
        params_path: Absolute path to params.json.

    Returns:
        (dry_train, wet_train, sr)
    """
    klon_sim_dir = params_path.parent
    with open(params_path) as fh:
        params = json.load(fh)

    sr = params["captures"]["train"]["sample_rate"]
    dry_manifest_rel = params["captures"]["train"]["dry_manifest"]
    dry_wav = (klon_sim_dir / dry_manifest_rel).resolve().with_suffix(".wav")
    wet_wav = klon_sim_dir / params["captures"]["train"]["wet_wav"]

    _LOG.info("Loading training data…")
    dry, _ = sf.read(str(dry_wav), dtype="float32", always_2d=False)
    wet, _ = sf.read(str(wet_wav), dtype="float32", always_2d=False)
    return dry, wet, int(sr)


def _fit_classical_model(
    key: str,
    dry_train: np.ndarray,
    wet_train: np.ndarray,
    sr: int,
) -> object | None:
    """Instantiate and fit a classical model. Returns None on failure."""
    module_path, class_name, kwargs = _CLASSICAL_MODELS[key]
    try:
        mod = importlib.import_module(module_path)
        model = getattr(mod, class_name)(**kwargs)
        model.fit(dry_train, wet_train, sr)
        return model
    except Exception as exc:
        _LOG.error("  [%s] classical fit failed: %s", key, exc)
        return None


def _get_fitted_classical_model(
    key: str,
    results_path: Path,
    zoo: ZooResults,
    dry_train: np.ndarray | None,
    wet_train: np.ndarray | None,
    sr_model: int,
    refit: bool = False,
) -> object | None:
    """Return a fitted classical model, using cache or pkl when available.

    Priority order:
    1. model_state from zoo.models[key] (saved by recent fit_models.py)
    2. Cache pkl at {results_path}/fitted_params/{key}.pkl
    3. Re-fit from training data (slow for Volterra/WH)

    Args:
        key: Model key (e.g. 'fir', 'volterra').
        results_path: Directory containing results.pkl and fitted_params/.
        zoo: Loaded ZooResults — checked for pre-fitted model_state.
        dry_train: Dry training signal (needed only if no cache/pkl state).
        wet_train: Wet training signal (needed only if no cache/pkl state).
        sr_model: Sample rate used for fitting.
        refit: When True, ignore cache and pkl, always re-fit.

    Returns:
        Fitted model instance, or None on failure.
    """
    # 1. Model state embedded in results.pkl (newest runs include this)
    if not refit and key in zoo.models:
        ms = getattr(zoo.models[key], "model_state", None)
        if ms is not None:
            _LOG.info("[%s] using model state from results.pkl", key)
            return ms

    cache_path = results_path / "fitted_params" / f"{key}.pkl"

    # 2. On-disk parameter cache
    if cache_path.exists() and not refit:
        _LOG.info("[%s] loading fitted parameters from cache (%s)", key, cache_path)
        try:
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)
        except Exception as exc:
            _LOG.warning("[%s] cache load failed (%s) — will refit", key, exc)

    # 3. Fit from scratch
    if dry_train is None or wet_train is None:
        _LOG.error("[%s] no training data available and no cache — skipping", key)
        return None

    _LOG.info("[%s] fitting classical model (first run — result will be cached)…", key)
    t0 = time.perf_counter()
    model = _fit_classical_model(key, dry_train, wet_train, sr_model)
    if model is None:
        return None
    elapsed = time.perf_counter() - t0
    _LOG.info("[%s] fitted in %.1fs", key, elapsed)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, "wb") as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
        _LOG.info("[%s] fitted parameters cached → %s", key, cache_path)
    except Exception as exc:
        _LOG.warning("[%s] could not save cache: %s", key, exc)

    return model


def _load_neural_model(
    key: str,
    results_path: Path,
    zoo: ZooResults,
    device: "torch.device",
) -> object | None:
    """Load a neural model, preferring best checkpoint over final weights.

    Priority order:
    1. model_state from zoo.models[key] (saved by recent fit_models.py)
    2. {results_path}/{key}_best.pt — best val-ESR checkpoint with metadata
    3. {results_path}/{key}_weights.pt — final-epoch weights (warning logged)

    Returns None if no weights are found.
    """
    import torch

    module_path, class_name, kwargs = _NEURAL_MODELS[key]

    def _make_model() -> object:
        mod = importlib.import_module(module_path)
        m = getattr(mod, class_name)(**kwargs)
        m.to(device)
        return m

    # 1. Embedded model state in pkl
    if key in zoo.models:
        ms = getattr(zoo.models[key], "model_state", None)
        if ms is not None and isinstance(ms, dict):
            try:
                model = _make_model()
                model.load_state_dict(ms)
                model.eval()
                _LOG.info("[%s] loaded model state from results.pkl", key)
                return model
            except Exception as exc:
                _LOG.warning("[%s] pkl model_state load failed (%s) — trying .pt files", key, exc)

    # 2. Best checkpoint file
    best_path = results_path / f"{key}_best.pt"
    weights_path = results_path / f"{key}_weights.pt"

    if best_path.exists():
        try:
            model = _make_model()
            ckpt = torch.load(str(best_path), map_location=device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                epoch = ckpt.get("epoch", "?")
                val_esr = ckpt.get("val_esr", float("nan"))
                _LOG.info(
                    "[%s] loaded best checkpoint (epoch %s, val_esr=%.4f)",
                    key, epoch, val_esr,
                )
            else:
                model.load_state_dict(ckpt)
                _LOG.info("[%s] loaded best checkpoint (legacy format)", key)
            model.eval()
            return model
        except Exception as exc:
            _LOG.error("[%s] failed to load %s: %s", key, best_path, exc)

    # 3. Final-epoch weights
    if weights_path.exists():
        _LOG.warning(
            "[%s] best checkpoint not found — loading final-epoch weights. "
            "Re-run retrain_neural.py to save the best checkpoint.",
            key,
        )
        try:
            model = _make_model()
            state = torch.load(str(weights_path), map_location=device)
            model.load_state_dict(state)
            model.eval()
            return model
        except Exception as exc:
            _LOG.error("[%s] failed to load %s: %s", key, weights_path, exc)
            return None

    _LOG.warning(
        "[%s] no checkpoint found at %s or %s — skipping. "
        "Run retrain_neural.py or re-run fit_models.py to generate weights.",
        key, best_path, weights_path,
    )
    return None


# ── Main ───────────────────────────────────────────────────────────────────── #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render a dry take through PedalDSP models and write DAW-ready WAVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="WAV",
        help="Mono dry guitar take (any sample rate; 44.1 kHz and 48 kHz recommended).",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/data/klon_sim/klon_sim_v1"),
        metavar="DIR",
        help="Directory containing results.pkl from fit_models.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Root directory for rendered WAVs. "
            "Default: {results}/daw_renders/{dry_stem}/ so all artifacts for a "
            "session live under the same results directory."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        metavar="MODEL",
        help=(
            "Models to render. Default: all. Classical models use fitted-param cache "
            f"(see --refit-classical). Choices: {', '.join(_ALL_MODEL_KEYS)}"
        ),
    )
    parser.add_argument(
        "--no-klonsim",
        action="store_true",
        help="Skip the KlonSim circuit-simulation reference track.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="PyTorch device for neural models.",
    )
    parser.add_argument(
        "--refit-classical",
        action="store_true",
        help=(
            "Force re-fitting of all classical models even if cached parameters exist. "
            "Use after changing training data or model hyperparameters."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without writing any files.",
    )
    args = parser.parse_args(argv)

    # ── Resolve device ────────────────────────────────────────────────────── #
    try:
        import torch
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
    except ImportError:
        device = None

    # ── Validate & load ZooResults ────────────────────────────────────────── #
    results_path = args.results.resolve()
    pkl_path = results_path / "results.pkl"
    if not pkl_path.exists():
        _LOG.error("results.pkl not found at %s", pkl_path)
        sys.exit(1)

    zoo = ZooResults.load(pkl_path)
    sr_model = zoo.sr  # 96 000

    params_path = Path(zoo.config.get("params_path", ""))
    if not params_path.exists():
        _LOG.error(
            "params.json '%s' referenced in results.pkl does not exist.", params_path
        )
        sys.exit(1)

    # RMS of training dry signal — used for gain-staging normalization in StreamRunner
    training_rms = float(np.sqrt(np.mean(zoo.dry_val.astype(np.float64) ** 2)))
    _LOG.info("Training dry RMS: %.4f  (%.1f dBFS)", training_rms,
              20.0 * np.log10(training_rms + 1e-9))

    # ── Determine which models to run ─────────────────────────────────────── #
    requested = set(args.models if args.models else _ALL_MODEL_KEYS)
    unknown = requested - set(_ALL_MODEL_KEYS)
    if unknown:
        _LOG.error("Unknown model(s): %s. Valid: %s", unknown, _ALL_MODEL_KEYS)
        sys.exit(1)

    classical_keys = [k for k in _CLASSICAL_MODELS if k in requested]
    neural_keys = [k for k in _NEURAL_MODELS if k in requested]
    input_path = args.input.resolve()
    dry_stem = input_path.stem.lower().replace(" ", "_")
    if args.output_dir is not None:
        out_dir = args.output_dir.resolve() / dry_stem
    else:
        out_dir = results_path / "daw_renders" / dry_stem

    if args.dry_run:
        print("\n--- DRY RUN ---")
        print(f"input       : {input_path}")
        print(f"results dir : {results_path}")
        print(f"output dir  : {out_dir}")
        print(f"model sr    : {sr_model} Hz")
        print(f"training rms: {training_rms:.4f}")
        print(f"klonsim     : {'yes' if not args.no_klonsim else 'no'}")
        print(f"classical   : {classical_keys}  (cache: {results_path}/fitted_params/)")
        print(f"neural      : {neural_keys} (prefer {results_path}/{{key}}_best.pt)")
        print(f"device      : {device}")
        return

    # ── Load & validate input WAV ─────────────────────────────────────────── #
    try:
        dry_44k, input_sr = load_wav_to_float32(input_path)
    except ValueError as exc:
        _LOG.error("Input WAV error: %s", exc)
        sys.exit(1)

    if input_sr == 44_100:
        pass  # expected
    elif input_sr == 48_000:
        _LOG.info(
            "Input sample rate is 48000 Hz — will resample to %d Hz for inference. "
            "Output files will be written at 44100 Hz.",
            sr_model,
        )
    else:
        _LOG.warning(
            "Input sample rate is %d Hz (expected 44100). The pipeline will resample "
            "to %d Hz for inference. For best results, record at 44100 Hz or 48000 Hz "
            "and ensure your DAW session rate matches your interface rate. "
            "Output files will be written at 44100 Hz regardless of input rate.",
            input_sr,
            sr_model,
        )

    duration_s = len(dry_44k) / input_sr

    _LOG.info(
        "Input: %s  (%.1f s, peak %s, RMS %s)",
        input_path.name,
        duration_s,
        _db(np.max(np.abs(dry_44k))),
        _db(np.sqrt(np.mean(dry_44k**2))),
    )

    # ── Upsample input to model sr ────────────────────────────────────────── #
    from pedal_model.utils.resampling import resample_audio

    _LOG.info("Resampling %d Hz → %d Hz…", input_sr, sr_model)
    dry_96k = resample_audio(dry_44k, input_sr, sr_model, quality="HQ")
    _LOG.info(
        "Upsampled: %d samples (%.1f s at %d Hz), peak %s",
        len(dry_96k),
        len(dry_96k) / sr_model,
        sr_model,
        _db(np.max(np.abs(dry_96k))),
    )

    # ── Clear classical cache if --refit-classical ────────────────────────── #
    if args.refit_classical:
        cache_dir = results_path / "fitted_params"
        for key in classical_keys:
            cache_path = cache_dir / f"{key}.pkl"
            if cache_path.exists():
                cache_path.unlink()
                _LOG.info("[%s] deleted cached params — will refit", key)

    # ── Write dry passthrough ─────────────────────────────────────────────── #
    out_dir.mkdir(parents=True, exist_ok=True)
    dry_out = out_dir / f"{dry_stem}__dry.wav"
    write_wav_44k(dry_out, dry_96k, src_sr=sr_model)
    _LOG.info("Wrote: %s", dry_out)

    # Track summary rows: model → (duration_s, peak_dBFS, inference_s)
    summary: list[tuple[str, float, float, float]] = []

    # ── KlonSim reference ─────────────────────────────────────────────────── #
    if not args.no_klonsim:
        _LOG.info("[klonsim] running circuit simulation…")
        t0 = time.perf_counter()
        try:
            klonsim_fn = load_klonsim(results_path)
            runner = StreamRunner(
                _CallableAdapter(klonsim_fn),
                "klonsim",
                sr=sr_model,
                training_rms=training_rms,
            )
            wet_ks = runner.run(dry_96k)
            inf_s = time.perf_counter() - t0
            ks_path = out_dir / f"{dry_stem}__klonsim.wav"
            write_wav_44k(ks_path, wet_ks, src_sr=sr_model)
            peak = float(np.max(np.abs(wet_ks)))
            _log_model_result("klonsim", peak, inf_s)
            summary.append(("klonsim", duration_s, peak, inf_s))
            _LOG.info("Wrote: %s", ks_path)
        except Exception as exc:
            _LOG.error("[klonsim] failed: %s", exc)

    # ── Load training data (for classical models that need re-fitting) ──────── #
    dry_train: np.ndarray | None = None
    wet_train: np.ndarray | None = None
    if classical_keys:
        # Check if all classical models can be served from cache/pkl
        need_training_data = any(
            (
                getattr(zoo.models.get(k), "model_state", None) is None
                and not (results_path / "fitted_params" / f"{k}.pkl").exists()
            )
            for k in classical_keys
        ) or args.refit_classical
        if need_training_data:
            try:
                dry_train, wet_train, _ = _load_training_data(params_path)
            except Exception as exc:
                _LOG.error("Could not load training data: %s", exc)

    from tqdm import tqdm

    # ── Classical models ──────────────────────────────────────────────────── #
    for key in tqdm(classical_keys, desc="Classical models", unit="model"):
        model = _get_fitted_classical_model(
            key, results_path, zoo, dry_train, wet_train, sr_model,
            refit=args.refit_classical,
        )
        if model is None:
            continue

        _LOG.info("[%s] running inference…", key)
        t0 = time.perf_counter()
        try:
            runner = StreamRunner(model, key, sr=sr_model, training_rms=training_rms)
            wet = runner.run(dry_96k)
            inf_s = time.perf_counter() - t0

            out_path = out_dir / f"{dry_stem}__{key}.wav"
            write_wav_44k(out_path, wet, src_sr=sr_model)
            peak = float(np.max(np.abs(wet)))
            _log_model_result(key, peak, inf_s)
            summary.append((key, duration_s, peak, inf_s))
            _LOG.info("Wrote: %s", out_path)
        except Exception as exc:
            _LOG.error("[%s] inference failed: %s", key, exc)

    # ── Neural models ─────────────────────────────────────────────────────── #
    for key in tqdm(neural_keys, desc="Neural models", unit="model"):
        if device is None:
            _LOG.warning("[%s] torch not available — skipping neural model", key)
            continue

        model = _load_neural_model(key, results_path, zoo, device)
        if model is None:
            continue

        _LOG.info("[%s] running neural inference on %s…", key, device)
        t0 = time.perf_counter()
        try:
            runner = StreamRunner(model, key, sr=sr_model, training_rms=training_rms)
            wet = runner.run(dry_96k)
            inf_s = time.perf_counter() - t0

            out_path = out_dir / f"{dry_stem}__{key}.wav"
            write_wav_44k(out_path, wet, src_sr=sr_model)
            peak = float(np.max(np.abs(wet)))
            _log_model_result(key, peak, inf_s)
            summary.append((key, duration_s, peak, inf_s))
            _LOG.info("Wrote: %s", out_path)
        except Exception as exc:
            _LOG.error("[%s] inference failed: %s", key, exc)

    # ── Summary table ─────────────────────────────────────────────────────── #
    if summary:
        print(f"\n{'model':<22}  {'duration':>8}  {'peak_out':>10}  {'inference':>10}")
        print("-" * 58)
        for model_name, dur, peak, inf_s in summary:
            peak_db = 20.0 * np.log10(peak + 1e-9)
            print(
                f"  {model_name:<20}  {_fmt_time(dur):>8}  "
                f"{peak_db:>+8.1f} dBFS  {inf_s:>8.1f}s"
            )
        print()

    _LOG.info("Done. Output directory: %s", out_dir)


def _log_model_result(key: str, peak: float, inf_s: float) -> None:
    peak_db = 20.0 * np.log10(peak + 1e-9)
    if peak < 1e-6:
        _LOG.error("[%s] output is silent (peak = %.1f dBFS)", key, peak_db)
    elif peak_db > -0.1:
        _LOG.warning("[%s] output is near clipping (peak = %.1f dBFS)", key, peak_db)
    else:
        _LOG.info("[%s] inference done in %.1fs, peak = %.1f dBFS", key, inf_s, peak_db)


if __name__ == "__main__":
    main()
