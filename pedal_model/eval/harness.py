"""Eval harness for the klon_sim model zoo.

Classical models fit in parallel (ProcessPoolExecutor); neural models train
sequentially on the available device (CPU or CUDA). Results are stored in a
ZooResults object and pickled to disk for later analysis.

The klon_sim capture is identified by its params.json, which records the path
to the dry signal manifest and both WAV files.

Usage (typically via run_eval.py):
    from pedal_model.eval.harness import EvalConfig, run_zoo
    cfg = EvalConfig(klon_sim_dir=Path("data/klon_sim/klon_sim_v1"))
    results = run_zoo(cfg)
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

# ── Eval config ──────────────────────────────────────────────────────────── #


@dataclass
class EvalConfig:
    """All options that control one model-zoo eval run.

    Attributes:
        klon_sim_dir: Directory containing params.json and WAV files.
        output_dir: Where results.pkl and results.csv are written.
        max_classical_samples: Max training samples for Hammerstein/WH
            (None = use full signal; FIR/IIR always use the full signal).
        max_volterra_samples: Max training samples for Volterra (much smaller
            to keep the feature matrix tractable, ≤ 100 k recommended).
        max_train_samples: Cap on training data fed to neural models (samples
            at the native sample rate). None = use full train signal.
        neural_epochs: Training epochs for each neural model.
        neural_batch_size: Chunk batch size for recurrent models.
        neural_chunk_size: TBPTT chunk length in samples.
        neural_lr: Initial learning rate.
        device: 'cuda', 'cpu', or 'auto' (picks CUDA if available).
        run_classical: Include classical model fits.
        run_neural: Include neural model training.
        classical_workers: Max parallel workers for classical fits. 0 = cpu_count.
        skip_models: Model names to exclude (e.g. ['volterra'] to skip slow fit).
    """

    klon_sim_dir: Path = Path("data/klon_sim/klon_sim_v1")
    output_dir: Path = Path("results/klon_sim/klon_sim_v1")
    max_classical_samples: int | None = 200_000
    max_volterra_samples: int = 100_000
    max_train_samples: int | None = 500_000
    neural_epochs: int = 30
    neural_batch_size: int = 32
    neural_chunk_size: int = 4096
    neural_lr: float = 3e-4
    device: str = "auto"
    run_classical: bool = True
    run_neural: bool = True
    classical_workers: int = 0
    skip_models: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.klon_sim_dir = Path(self.klon_sim_dir)
        self.output_dir = Path(self.output_dir)


# ── Classical model registry ─────────────────────────────────────────────── #

# Each entry: model_key → (module_path, class_name, kwargs, max_samples_key)
_CLASSICAL_REGISTRY: dict[str, tuple[str, str, dict]] = {
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
}

# ── Neural model registry ─────────────────────────────────────────────────── #

_NEURAL_REGISTRY: dict[str, tuple[str, str, dict]] = {
    "mlp": (
        "pedal_model.models.neural.mlp",
        "MLPModel",
        {"receptive_field_samples": 1024},
    ),
    "lstm": (
        "pedal_model.models.neural.lstm",
        "LSTMModel",
        {"hidden_size": 32},
    ),
    "gru": (
        "pedal_model.models.neural.gru",
        "GRUModel",
        {"hidden_size": 32},
    ),
    "tcn": (
        "pedal_model.models.neural.tcn",
        "TCNModel",
        {"channels": 32, "n_blocks": 10},
    ),
    "ddsp": (
        "pedal_model.models.neural.ddsp",
        "DDSPModel",
        {"n_fir_taps": 64},
    ),
    "klon_ddsp": (
        "pedal_model.models.neural.ddsp_klon",
        "KlonDDSP",
        {"gru_hidden": 16},
    ),
}


# ── Worker function (top-level for pickle compatibility) ──────────────────── #


def _classical_worker(args: tuple) -> tuple[str, np.ndarray, float, float]:
    """Fit and predict one classical model; runs in a subprocess.

    Args:
        args: (key, module_path, class_name, kwargs, dry_train_path,
               wet_train_path, dry_val_path, sr, max_samples, seed)

    Returns:
        (key, val_prediction, fit_time_s, predict_time_s)
    """
    (
        key,
        module_path,
        class_name,
        kwargs,
        dry_train_path,
        wet_train_path,
        dry_val_path,
        sr,
        max_samples,
        seed,
    ) = args

    # Lazy imports inside the subprocess to avoid heavy torch startup cost
    import importlib

    import numpy as np
    import soundfile as sf

    dry_train, _ = sf.read(str(dry_train_path), dtype="float32", always_2d=False)
    wet_train, _ = sf.read(str(wet_train_path), dtype="float32", always_2d=False)
    dry_val, _ = sf.read(str(dry_val_path), dtype="float32", always_2d=False)

    if max_samples and len(dry_train) > max_samples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(dry_train), max_samples, replace=False))
        dry_fit = dry_train[idx]
        wet_fit = wet_train[idx]
    else:
        dry_fit, wet_fit = dry_train, wet_train

    mod = importlib.import_module(module_path)
    model = getattr(mod, class_name)(**kwargs)

    t0 = time.perf_counter()
    model.fit(dry_fit, wet_fit, sr)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred = model.predict(dry_val)
    pred_s = time.perf_counter() - t1

    # Ensure prediction length matches val signal
    N = len(dry_val)
    if len(pred) < N:
        pred = np.concatenate([np.zeros(N - len(pred), dtype=np.float32), pred])
    elif len(pred) > N:
        pred = pred[:N]

    return key, pred.astype(np.float32), float(fit_s), float(pred_s)


# ── Classical parallel fitting ─────────────────────────────────────────────  #


def fit_classical_parallel(
    cfg: EvalConfig,
    dry_train_path: Path,
    wet_train_path: Path,
    dry_val_path: Path,
    sr: int,
) -> dict[str, tuple[np.ndarray, float, float]]:
    """Fit all classical models in parallel.

    Args:
        cfg: Eval configuration.
        dry_train_path: Path to dry training WAV.
        wet_train_path: Path to wet training WAV.
        dry_val_path: Path to dry val WAV.
        sr: Sample rate in Hz.

    Returns:
        Mapping from model key to (val_prediction, fit_time_s, predict_time_s).
    """
    tasks = []
    seed = 0
    for key, (module_path, class_name, kwargs) in _CLASSICAL_REGISTRY.items():
        if key in cfg.skip_models:
            continue
        max_s = cfg.max_volterra_samples if key == "volterra" else cfg.max_classical_samples
        tasks.append((
            key,
            module_path,
            class_name,
            kwargs,
            str(dry_train_path),
            str(wet_train_path),
            str(dry_val_path),
            sr,
            max_s,
            seed,
        ))
        seed += 1

    n_workers = cfg.classical_workers or min(len(tasks), os.cpu_count() or 4)
    results: dict[str, tuple[np.ndarray, float, float]] = {}

    print(f"  fitting {len(tasks)} classical model(s) with {n_workers} worker(s)…")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_classical_worker, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            key_done = futures[fut]
            try:
                key_r, pred, fit_s, pred_s = fut.result()
                results[key_r] = (pred, fit_s, pred_s)
                print(f"    {key_r}: fit {fit_s:.1f}s, predict {pred_s:.1f}s")
            except Exception as exc:
                print(f"    {key_done}: FAILED — {exc}")

    return results


# ── Neural sequential training ────────────────────────────────────────────── #


def _resolve_device(cfg: EvalConfig) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def _load_module_class(module_path: str, class_name: str) -> type:
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _train_recurrent(
    model: Any,
    dry_train: np.ndarray,
    wet_train: np.ndarray,
    cfg: EvalConfig,
    device: torch.device,
) -> None:
    """Train LSTM, GRU, KlonDDSP, or DDSP models (chunk-based TBPTT).

    Models must accept (batch, time, 1) input and return either a tensor of
    the same shape or a (tensor, hidden) tuple.
    """
    from pedal_model.data.dataset import ChunkDataset
    from pedal_model.train.losses import CombinedLoss

    dataset = ChunkDataset(dry_train, wet_train, chunk_size=cfg.neural_chunk_size)
    loader = DataLoader(dataset, batch_size=cfg.neural_batch_size, shuffle=True, drop_last=True)
    loss_fn = CombinedLoss().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.neural_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg.neural_epochs
    )
    model.to(device).train()

    for epoch in range(1, cfg.neural_epochs + 1):
        epoch_loss = 0.0
        for dry_chunk, wet_chunk in loader:
            dry_chunk = dry_chunk.to(device)
            wet_chunk = wet_chunk.to(device)
            optimiser.zero_grad()
            out = model(dry_chunk)
            # Models may return (output, hidden) — unwrap
            if isinstance(out, tuple):
                out = out[0]
            # Handle channels-first models (batch, 1, time) → (batch, time, 1)
            if out.dim() == 3 and out.shape[1] == 1 and out.shape[2] != 1:
                out = out.transpose(1, 2)
            loss = loss_fn(out, wet_chunk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()
        scheduler.step()
        if epoch % max(1, cfg.neural_epochs // 5) == 0:
            avg = epoch_loss / max(1, len(loader))
            print(f"      epoch {epoch}/{cfg.neural_epochs}  loss={avg:.4f}")


def _train_tcn(
    model: Any,
    dry_train: np.ndarray,
    wet_train: np.ndarray,
    cfg: EvalConfig,
    device: torch.device,
) -> None:
    """Train TCN model (expects channels-first (batch, 1, time) input)."""
    from pedal_model.data.dataset import ChunkDataset
    from pedal_model.train.losses import CombinedLoss

    dataset = ChunkDataset(dry_train, wet_train, chunk_size=cfg.neural_chunk_size)
    loader = DataLoader(dataset, batch_size=cfg.neural_batch_size, shuffle=True, drop_last=True)
    loss_fn = CombinedLoss().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.neural_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg.neural_epochs
    )
    model.to(device).train()

    for epoch in range(1, cfg.neural_epochs + 1):
        epoch_loss = 0.0
        for dry_chunk, wet_chunk in loader:
            # dry_chunk/wet_chunk: (batch, time, 1) from ChunkDataset
            dry_cf = dry_chunk.to(device).transpose(1, 2)  # (batch, 1, time)
            wet_cf = wet_chunk.to(device).transpose(1, 2)
            optimiser.zero_grad()
            out = model(dry_cf)  # (batch, 1, time)
            loss = loss_fn(out, wet_cf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()
        scheduler.step()
        if epoch % max(1, cfg.neural_epochs // 5) == 0:
            avg = epoch_loss / max(1, len(loader))
            print(f"      epoch {epoch}/{cfg.neural_epochs}  loss={avg:.4f}")


def _train_mlp(
    model: Any,
    dry_train: np.ndarray,
    wet_train: np.ndarray,
    cfg: EvalConfig,
    device: torch.device,
) -> None:
    """Train MLP on (window, scalar) pairs from PedalDataset."""
    from pedal_model.data.dataset import PedalDataset
    from pedal_model.train.losses import ESRLoss

    dataset = PedalDataset(dry_train, wet_train, receptive_field=model.receptive_field, hop=32)
    loader = DataLoader(dataset, batch_size=cfg.neural_batch_size * 16, shuffle=True, drop_last=True)
    loss_fn = ESRLoss().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.neural_lr)
    model.to(device).train()

    for epoch in range(1, cfg.neural_epochs + 1):
        epoch_loss = 0.0
        for windows, targets in loader:
            windows = windows.to(device)
            targets = targets.to(device)
            optimiser.zero_grad()
            pred = model(windows).squeeze(-1)  # (batch,)
            loss = loss_fn(pred.unsqueeze(0), targets.unsqueeze(0))
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
        if epoch % max(1, cfg.neural_epochs // 5) == 0:
            avg = epoch_loss / max(1, len(loader))
            print(f"      epoch {epoch}/{cfg.neural_epochs}  loss={avg:.4f}")


def train_neural_sequential(
    cfg: EvalConfig,
    dry_train: np.ndarray,
    wet_train: np.ndarray,
    dry_val: np.ndarray,
    wet_val: np.ndarray,
    sr: int,
) -> dict[str, tuple[np.ndarray, float, float, dict | None]]:
    """Train neural models one at a time on the available device.

    Args:
        cfg: Eval configuration.
        dry_train: Dry training signal.
        wet_train: Wet training signal.
        dry_val: Dry validation signal for prediction.
        wet_val: Wet validation signal (used for klon_ddsp fraction analysis).
        sr: Sample rate in Hz.

    Returns:
        Mapping from model key to (val_prediction, train_time_s, predict_time_s,
        ddsp_analysis_or_None).
    """
    device = _resolve_device(cfg)
    print(f"  neural models on device: {device}")

    results: dict[str, tuple[np.ndarray, float, float]] = {}

    for key, (module_path, class_name, kwargs) in _NEURAL_REGISTRY.items():
        if key in cfg.skip_models:
            continue
        print(f"  training {key}…")
        cls = _load_module_class(module_path, class_name)
        model = cls(**kwargs)

        t0 = time.perf_counter()
        if key in ("tcn", "ddsp"):
            # Both expect (batch, 1, time) channels-first input
            _train_tcn(model, dry_train, wet_train, cfg, device)
        elif key == "mlp":
            _train_mlp(model, dry_train, wet_train, cfg, device)
        else:
            _train_recurrent(model, dry_train, wet_train, cfg, device)
        train_s = time.perf_counter() - t0

        model.eval()
        model.cpu()
        t1 = time.perf_counter()
        pred = model.predict(dry_val)
        pred_s = time.perf_counter() - t1

        # Ensure length matches
        N = len(dry_val)
        if len(pred) < N:
            pred = np.concatenate([np.zeros(N - len(pred), dtype=np.float32), pred])
        elif len(pred) > N:
            pred = pred[:N]

        ddsp_analysis: dict | None = None
        if key == "klon_ddsp":
            try:
                ddsp_analysis = model.analyse_ddsp_fraction(dry_val, wet_val)
                print(f"    klon_ddsp: ddsp_fraction={ddsp_analysis['ddsp_fraction']:.1%}")
            except Exception as exc:
                print(f"    klon_ddsp: analyse_ddsp_fraction failed — {exc}")

        results[key] = (pred.astype(np.float32), float(train_s), float(pred_s), ddsp_analysis)
        print(f"    {key}: train {train_s:.1f}s, predict {pred_s:.2f}s")

    return results


# ── Metrics ───────────────────────────────────────────────────────────────── #


def _compute_metrics(
    wet_val: np.ndarray,
    prediction: np.ndarray,
    dry_val: np.ndarray,
    manifest: Any,
    sr: int,
) -> dict[str, dict[str, float]]:
    """Run compute_per_section for one model's prediction."""
    from pedal_model.metrics.suite import compute_per_section
    return compute_per_section(manifest, dry_val, wet_val, prediction, sr)


# ── Main entry point ──────────────────────────────────────────────────────── #


def run_zoo(cfg: EvalConfig) -> "ZooResults":  # noqa: F821
    """Run the full model zoo on klon_sim data.

    Loads the params.json from cfg.klon_sim_dir, resolves dry/wet WAV paths,
    fits classical models in parallel, trains neural models sequentially,
    runs compute_per_section for each, and returns a ZooResults object.

    Args:
        cfg: Eval configuration.

    Returns:
        :class:`~pedal_model.eval.results.ZooResults` with predictions and
        per-section metrics for every model that ran successfully.
    """
    from pedal_model.eval.results import ModelResult, ZooResults
    from pedal_model.signals.manifest import Manifest

    # ── Resolve paths from params.json ──────────────────────────────── #
    params_path = cfg.klon_sim_dir / "params.json"
    params = json.loads(params_path.read_text())
    sr = params["captures"]["val"]["sample_rate"]

    # Dry paths from the signal generator manifest
    dry_manifest_rel = params["captures"]["train"]["dry_manifest"]
    dry_manifest_path = (cfg.klon_sim_dir / dry_manifest_rel).resolve()
    dry_val_rel = params["captures"]["val"]["dry_manifest"]
    dry_val_manifest_path = (cfg.klon_sim_dir / dry_val_rel).resolve()

    dry_train_wav = dry_manifest_path.with_suffix(".wav")
    dry_val_wav = dry_val_manifest_path.with_suffix(".wav")
    wet_train_wav = cfg.klon_sim_dir / params["captures"]["train"]["wet_wav"]
    wet_val_wav = cfg.klon_sim_dir / params["captures"]["val"]["wet_wav"]

    print(f"Loading val signal ({params['captures']['val']['duration_s']:.1f}s @ {sr}Hz)…")
    dry_val, _ = sf.read(str(dry_val_wav), dtype="float32", always_2d=False)
    wet_val, _ = sf.read(str(wet_val_wav), dtype="float32", always_2d=False)

    manifest = Manifest(dry_val_manifest_path)

    # ── Training data (possibly subsampled) ──────────────────────────── #
    print(f"Loading train signal…")
    dry_train, _ = sf.read(str(dry_train_wav), dtype="float32", always_2d=False)
    wet_train, _ = sf.read(str(wet_train_wav), dtype="float32", always_2d=False)

    if cfg.max_train_samples and len(dry_train) > cfg.max_train_samples:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(dry_train), cfg.max_train_samples, replace=False))
        dry_train_neural = dry_train[idx]
        wet_train_neural = wet_train[idx]
        print(f"  subsampled train: {len(dry_train)} → {len(dry_train_neural)} samples")
    else:
        dry_train_neural = dry_train
        wet_train_neural = wet_train

    date_run = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results = ZooResults(
        config={"cfg": str(cfg), "params_path": str(params_path)},
        date_run=date_run,
        sr=sr,
        dry_val=dry_val,
        wet_val=wet_val,
    )

    # ── Classical models ─────────────────────────────────────────────── #
    if cfg.run_classical:
        print("\nClassical models:")
        classical = fit_classical_parallel(
            cfg, dry_train_wav, wet_train_wav, dry_val_wav, sr
        )
        for key, (pred, fit_s, pred_s) in classical.items():
            metrics = _compute_metrics(wet_val, pred, dry_val, manifest, sr)
            results.models[key] = ModelResult(
                name=key,
                model_type="classical",
                prediction=pred,
                fit_time_s=fit_s,
                predict_time_s=pred_s,
                metrics_per_section=metrics,
            )

    # ── Neural models ─────────────────────────────────────────────────── #
    if cfg.run_neural:
        print("\nNeural models:")
        neural = train_neural_sequential(
            cfg, dry_train_neural, wet_train_neural, dry_val, wet_val, sr
        )
        for key, (pred, train_s, pred_s, ddsp_analysis) in neural.items():
            metrics = _compute_metrics(wet_val, pred, dry_val, manifest, sr)

            results.models[key] = ModelResult(
                name=key,
                model_type="neural",
                prediction=pred,
                fit_time_s=train_s,
                predict_time_s=pred_s,
                metrics_per_section=metrics,
                ddsp_analysis=ddsp_analysis,
            )

    # ── Save ──────────────────────────────────────────────────────────── #
    out_pkl = cfg.output_dir / "results.pkl"
    out_csv = cfg.output_dir / "results.csv"
    results.save(out_pkl)
    results.to_csv(out_csv)
    print(f"\nSaved:\n  {out_pkl}\n  {out_csv}")

    return results
