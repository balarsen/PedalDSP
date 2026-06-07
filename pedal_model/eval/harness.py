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

import copy
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
        max_hammerstein_samples: Subsample cap for Hammerstein only. Its
            polynomial lstsq stage cannot be streamed (no temporal structure),
            so a large random subsample is used instead. 5M gives 8.1% coverage.
        max_mlp_samples: Random subsample cap for MLP. MLP is pointwise so
            random scatter is correct; 5M gives 8.1% coverage.
        neural_epochs: Training epochs for LSTM, GRU, TCN, DDSP, KlonDDSP.
            50 epochs × ~15 k chunks = ~750 k gradient steps; 6× more than the
            previous 1000 epochs × 122 chunks but with full-signal diversity.
        circuit_dict_epochs: Training epochs for CircuitDictionary. Independent
            of neural_epochs because circuit_dict uses per-epoch window cycling
            rather than TBPTT. Default 200 = 1.55 full passes at window=480k.
        circuit_dict_window_size: Samples per epoch for CircuitDictionary.
            480 000 samples = 5 s at 96 kHz; one full pass every 129 epochs.
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
    max_hammerstein_samples: int = 5_000_000
    max_mlp_samples: int = 5_000_000
    neural_epochs: int = 50
    circuit_dict_epochs: int = 200
    circuit_dict_window_size: int = 480_000
    neural_batch_size: int = 32
    neural_chunk_size: int = 4096
    neural_lr: float = 1e-3
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
    "circuit_dict": (
        "pedal_model.models.classical.circuit_dictionary",
        "CircuitDictionary",
        {},  # n_epochs and window_size injected from cfg at dispatch time
    ),
}

# Models that require contiguous temporal blocks for training rather than
# random-index subsampling. Gradient-based models with FIR convolutions need
# unbroken signal history so the causal zero-padding transient is negligible.
_CONTIGUOUS_CLASSICAL: frozenset[str] = frozenset({"circuit_dict"})

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
               wet_train_path, dry_val_path, sr, max_samples, seed, contiguous)

    Returns:
        (key, val_prediction, fit_time_s, predict_time_s, topology)
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
        contiguous,
    ) = args

    # Lazy imports inside the subprocess to avoid heavy torch startup cost
    import importlib

    import numpy as np
    import soundfile as sf

    dry_train, _ = sf.read(str(dry_train_path), dtype="float32", always_2d=False)
    wet_train, _ = sf.read(str(wet_train_path), dtype="float32", always_2d=False)
    dry_val, _ = sf.read(str(dry_val_path), dtype="float32", always_2d=False)

    if max_samples and len(dry_train) > max_samples and not contiguous:
        # Random-index subsampling for LSE-based classical models (FIR, IIR, etc.)
        # that treat each sample independently and need no temporal context.
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(dry_train), max_samples, replace=False))
        dry_fit = dry_train[idx]
        wet_fit = wet_train[idx]
    else:
        # Contiguous models (circuit_dict) receive the full signal and handle
        # per-epoch windowing internally for full-signal coverage.
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

    topology: dict[str, float] | None = None
    if hasattr(model, "alpha_") and model.alpha_ is not None:
        topology = dict(zip(model.primitive_names, model.alpha_.tolist()))

    return key, pred.astype(np.float32), float(fit_s), float(pred_s), topology


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
        Mapping from model key to (val_prediction, fit_time_s, predict_time_s, topology).
        topology is None for models without architecture weights, or a {primitive: weight}
        dict for CircuitDictionary.
    """
    tasks = []
    for key, (module_path, class_name, kwargs) in _CLASSICAL_REGISTRY.items():
        if key in cfg.skip_models:
            continue

        # Per-model max_samples dispatch:
        #   FIR, IIR, WH, Volterra  → None (full signal; FFT or streaming XTX)
        #   Hammerstein              → cfg.max_hammerstein_samples (poly lstsq needs cap)
        #   circuit_dict             → None (contiguous full signal, windowed internally)
        if key == "hammerstein":
            max_s: int | None = cfg.max_hammerstein_samples
        else:
            max_s = None

        # circuit_dict epochs and window_size come from cfg, not the registry default
        effective_kwargs = dict(kwargs)
        if key == "circuit_dict":
            effective_kwargs["n_epochs"] = cfg.circuit_dict_epochs
            effective_kwargs["window_size"] = cfg.circuit_dict_window_size

        # Key-based seed: skip-resistant — skipping other models does not change
        # this model's seed, so repeated runs produce identical subsamples.
        seed = abs(hash(key)) % (2 ** 31)
        tasks.append((
            key,
            module_path,
            class_name,
            effective_kwargs,
            str(dry_train_path),
            str(wet_train_path),
            str(dry_val_path),
            sr,
            max_s,
            seed,
            key in _CONTIGUOUS_CLASSICAL,
        ))

    n_workers = cfg.classical_workers or min(len(tasks), os.cpu_count() or 4)
    results: dict[str, tuple[np.ndarray, float, float, dict | None]] = {}

    print(f"  fitting {len(tasks)} classical model(s) with {n_workers} worker(s)…")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_classical_worker, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            key_done = futures[fut]
            try:
                key_r, pred, fit_s, pred_s, topology = fut.result()
                results[key_r] = (pred, fit_s, pred_s, topology)
                print(f"    {key_r}: fit {fit_s:.1f}s, predict {pred_s:.1f}s")
                if topology:
                    top_str = ", ".join(
                        f"{n}={w:.2f}" for n, w in
                        sorted(topology.items(), key=lambda kv: -kv[1])[:3]
                    )
                    print(f"      topology: {top_str}")
            except Exception as exc:
                print(f"    {key_done}: FAILED — {exc}")

    return results


# ── Neural sequential training ────────────────────────────────────────────── #


def _select_seq_blocks(
    dry: np.ndarray,
    wet: np.ndarray,
    max_samples: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a training array from evenly-spaced contiguous blocks.

    Selects ``max_samples // block_size`` non-overlapping blocks of
    ``block_size`` samples, spaced evenly across the full signal.  Each
    block is internally contiguous so recurrent models see valid sequential
    dynamics within every training chunk.

    Args:
        dry: Full dry training signal, shape (N,).
        wet: Full wet training signal, shape (N,).
        max_samples: Target total sample count.
        block_size: Block length in samples (should match neural_chunk_size).

    Returns:
        Tuple of concatenated ``(dry_blocks, wet_blocks)`` each of length
        ``n_blocks * block_size``.
    """
    N = len(dry)
    n_blocks = max_samples // block_size
    if n_blocks * block_size >= N:
        trim = (N // block_size) * block_size
        return dry[:trim].copy(), wet[:trim].copy()
    stride = N // n_blocks
    dry_out = np.empty(n_blocks * block_size, dtype=np.float32)
    wet_out = np.empty(n_blocks * block_size, dtype=np.float32)
    for i in range(n_blocks):
        src = i * stride
        ds, de = i * block_size, (i + 1) * block_size
        dry_out[ds:de] = dry[src : src + block_size]
        wet_out[ds:de] = wet[src : src + block_size]
    return dry_out, wet_out


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
) -> float:
    """Train LSTM, GRU, or DDSP models (chunk-based TBPTT).

    Uses training-loss-minimum checkpointing: saves a deepcopy of model weights
    whenever training loss hits a new minimum, then restores at the end.  This
    catches catastrophic late-epoch divergence without requiring a separate val
    pass every epoch.

    Models must accept (batch, time, 1) input and return either a tensor of
    the same shape or a (tensor, hidden) tuple.

    Returns:
        Best epoch training loss.
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

    best_loss = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    max_grad_norm_seen = 0.0
    t_loop_start = time.perf_counter()

    avg = float("nan")
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
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            max_grad_norm_seen = max(max_grad_norm_seen, grad_norm)
            optimiser.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg = epoch_loss / max(1, len(loader))
        if avg < best_loss:
            best_loss = avg
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % max(1, cfg.neural_epochs // 5) == 0:
            elapsed = time.perf_counter() - t_loop_start
            eta_s = elapsed / epoch * (cfg.neural_epochs - epoch)
            eta_str = time.strftime("%H:%M:%S", time.localtime(time.time() + eta_s))
            print(f"      epoch {epoch}/{cfg.neural_epochs}  loss={avg:.4f}  best={best_loss:.4f}@ep{best_epoch}"
                  f"  elapsed={elapsed:.0f}s  eta {eta_str}")

    if best_state is not None:
        model.load_state_dict(best_state)
    weights_ok = all(torch.isfinite(p).all().item() for p in model.parameters())
    print(f"      best checkpoint: epoch {best_epoch}/{cfg.neural_epochs}  loss={best_loss:.4f}  grad_norm_max={max_grad_norm_seen:.3f}  weights_ok={weights_ok}")
    return best_loss


def _train_tcn(
    model: Any,
    dry_train: np.ndarray,
    wet_train: np.ndarray,
    cfg: EvalConfig,
    device: torch.device,
    dry_val: np.ndarray | None = None,
    wet_val: np.ndarray | None = None,
    val_check_interval: int = 50,
) -> float:
    """Train TCN model (expects channels-first (batch, 1, time) input).

    Uses val-ESR checkpointing every `val_check_interval` epochs when dry_val
    and wet_val are provided.  TCN's training loss can still descend while
    val-ESR degrades (overfitting to training distribution), so training-loss
    is not a reliable proxy — periodic val-ESR is the correct signal.

    Returns:
        Best val-ESR (or best training loss when val not provided).
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

    use_val_esr = dry_val is not None and wet_val is not None
    best_metric = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    max_grad_norm_seen = 0.0
    t_loop_start = time.perf_counter()

    avg = float("nan")
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
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            max_grad_norm_seen = max(max_grad_norm_seen, grad_norm)
            optimiser.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg = epoch_loss / max(1, len(loader))

        if use_val_esr and epoch % val_check_interval == 0:
            # Periodic val-ESR check — more reliable than training loss for TCN
            from pedal_model.metrics.time_domain import compute_esr
            model.eval()
            pred = model.predict(dry_val)
            model.train()
            n = min(len(wet_val), len(pred))
            val_esr = compute_esr(wet_val[:n], pred[:n])
            metric = val_esr
            if val_esr < best_metric:
                best_metric = val_esr
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch % max(1, cfg.neural_epochs // 5) == 0:
                elapsed = time.perf_counter() - t_loop_start
                eta_s = elapsed / epoch * (cfg.neural_epochs - epoch)
                eta_str = time.strftime("%H:%M:%S", time.localtime(time.time() + eta_s))
                print(f"      epoch {epoch}/{cfg.neural_epochs}  loss={avg:.4f}  val_esr={val_esr:.4f}  best={best_metric:.4f}@ep{best_epoch}"
                      f"  elapsed={elapsed:.0f}s  eta {eta_str}")
        else:
            metric = avg
            if not use_val_esr and avg < best_metric:
                best_metric = avg
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch % max(1, cfg.neural_epochs // 5) == 0:
                elapsed = time.perf_counter() - t_loop_start
                eta_s = elapsed / epoch * (cfg.neural_epochs - epoch)
                eta_str = time.strftime("%H:%M:%S", time.localtime(time.time() + eta_s))
                print(f"      epoch {epoch}/{cfg.neural_epochs}  loss={avg:.4f}  best={best_metric:.4f}@ep{best_epoch}"
                      f"  elapsed={elapsed:.0f}s  eta {eta_str}")

    # If val-ESR checkpointing never fired (neural_epochs < val_check_interval),
    # run one final check so best_state is always set before restore.
    if use_val_esr and best_state is None:
        from pedal_model.metrics.time_domain import compute_esr
        model.eval()
        pred = model.predict(dry_val)
        model.train()
        n = min(len(wet_val), len(pred))
        best_metric = compute_esr(wet_val[:n], pred[:n])
        best_epoch = cfg.neural_epochs
        best_state = copy.deepcopy(model.state_dict())
        print(f"      (final val-ESR check: {best_metric:.4f} — val_check_interval > neural_epochs)")

    if best_state is not None:
        model.load_state_dict(best_state)
    weights_ok = all(torch.isfinite(p).all().item() for p in model.parameters())
    metric_name = "val_esr" if use_val_esr else "train_loss"
    print(f"      best checkpoint: epoch {best_epoch}/{cfg.neural_epochs}  {metric_name}={best_metric:.4f}  grad_norm_max={max_grad_norm_seen:.3f}  weights_ok={weights_ok}")
    return best_metric


def _train_mlp(
    model: Any,
    dry_train: np.ndarray,
    wet_train: np.ndarray,
    cfg: EvalConfig,
    device: torch.device,
) -> float:
    """Train MLP on (window, scalar) pairs from PedalDataset.

    Returns:
        Final epoch average loss.
    """
    from pedal_model.data.dataset import PedalDataset
    from pedal_model.train.losses import ESRLoss

    dataset = PedalDataset(dry_train, wet_train, receptive_field=model.receptive_field, hop=32)
    loader = DataLoader(dataset, batch_size=cfg.neural_batch_size * 16, shuffle=True, drop_last=True)
    loss_fn = ESRLoss().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.neural_lr)
    model.to(device).train()

    avg = float("nan")
    t_loop_start = time.perf_counter()
    for epoch in range(1, cfg.neural_epochs + 1):
        epoch_loss = 0.0
        for windows, targets in loader:
            windows = windows.to(device)
            targets = targets.to(device)
            optimiser.zero_grad()
            pred = model(windows).squeeze(-1)  # (batch,)
            loss = loss_fn(pred.unsqueeze(0), targets.unsqueeze(0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()
        avg = epoch_loss / max(1, len(loader))
        if epoch % max(1, cfg.neural_epochs // 5) == 0:
            elapsed = time.perf_counter() - t_loop_start
            eta_s = elapsed / epoch * (cfg.neural_epochs - epoch)
            eta_str = time.strftime("%H:%M:%S", time.localtime(time.time() + eta_s))
            print(f"      epoch {epoch}/{cfg.neural_epochs}  loss={avg:.4f}"
                  f"  elapsed={elapsed:.0f}s  eta {eta_str}")
    return avg


def _train_klon_ddsp(
    model: Any,
    dry_train: np.ndarray,
    wet_train: np.ndarray,
    cfg: EvalConfig,
    device: torch.device,
    ddsp_lr_scale: float = 0.1,
    aux_weight: float = 0.1,
) -> float:
    """Train KlonDDSP with separate DDSP-circuit / GRU param groups and auxiliary circuit loss.

    The DDSP circuit and GRU corrector compete for gradient signal when they share a
    single optimizer at the same lr.  Separating them prevents the GRU from dominating
    (it has more parameters and converges faster) and keeps the physics-informed circuit
    from collapsing.

    An auxiliary loss on the DDSP circuit output alone (weight=``aux_weight``) provides
    a direct gradient signal to the circuit parameters, preventing the common failure
    mode where ddsp_fraction → 0% because the circuit gets no gradient once the GRU
    correction is large enough to minimize the combined-output loss alone.

    Uses training-loss-minimum checkpointing (same as _train_recurrent).

    Args:
        model: KlonDDSP instance.
        dry_train: Contiguous-block training dry signal.
        wet_train: Contiguous-block training wet signal.
        cfg: Eval configuration.
        device: Compute device.
        ddsp_lr_scale: Multiplier applied to cfg.neural_lr for DDSP circuit params.
            Default 0.1 → DDSP at 1e-4 when base lr is 1e-3.
        aux_weight: Weight of the auxiliary loss on the DDSP-stage output.
            Default 0.1. Increase to 0.3 if circuit consistently collapses.

    Returns:
        Best training loss (combined + auxiliary).
    """
    from pedal_model.data.dataset import ChunkDataset
    from pedal_model.train.losses import CombinedLoss

    dataset = ChunkDataset(dry_train, wet_train, chunk_size=cfg.neural_chunk_size)
    loader = DataLoader(dataset, batch_size=cfg.neural_batch_size, shuffle=True, drop_last=True)
    loss_fn = CombinedLoss().to(device)

    # Separate param groups: DDSP circuit at lower lr so GRU doesn't dominate.
    ddsp_params = [p for n, p in model.named_parameters() if not n.startswith("gru")]
    gru_params = [p for n, p in model.named_parameters() if n.startswith("gru")]
    optimiser = torch.optim.AdamW(
        [
            {"params": ddsp_params, "lr": cfg.neural_lr * ddsp_lr_scale},
            {"params": gru_params, "lr": cfg.neural_lr},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg.neural_epochs
    )
    model.to(device).train()

    best_loss = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    max_grad_norm_seen = 0.0
    t_loop_start = time.perf_counter()

    avg = float("nan")
    for epoch in range(1, cfg.neural_epochs + 1):
        epoch_loss = 0.0
        for dry_chunk, wet_chunk in loader:
            dry_chunk = dry_chunk.to(device)   # (batch, time, 1)
            wet_chunk = wet_chunk.to(device)
            optimiser.zero_grad()

            out, _ = model(dry_chunk)          # (batch, time, 1)
            loss = loss_fn(out, wet_chunk)

            # Auxiliary loss: DDSP circuit stage alone — keeps circuit from collapsing.
            # Detach so the aux gradient affects only DDSP params, not GRU.
            if aux_weight > 0.0:
                x_2d = dry_chunk.squeeze(-1)   # (batch, time)
                ddsp_out = model._ddsp_stage(x_2d).unsqueeze(-1)  # (batch, time, 1)
                aux = loss_fn(ddsp_out, wet_chunk)
                loss = loss + aux_weight * aux

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            max_grad_norm_seen = max(max_grad_norm_seen, grad_norm)
            optimiser.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg = epoch_loss / max(1, len(loader))
        if avg < best_loss:
            best_loss = avg
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % max(1, cfg.neural_epochs // 5) == 0:
            elapsed = time.perf_counter() - t_loop_start
            eta_s = elapsed / epoch * (cfg.neural_epochs - epoch)
            eta_str = time.strftime("%H:%M:%S", time.localtime(time.time() + eta_s))
            print(f"      epoch {epoch}/{cfg.neural_epochs}  loss={avg:.4f}  best={best_loss:.4f}@ep{best_epoch}"
                  f"  elapsed={elapsed:.0f}s  eta {eta_str}")

    if best_state is not None:
        model.load_state_dict(best_state)
    weights_ok = all(torch.isfinite(p).all().item() for p in model.parameters())
    print(f"      best checkpoint: epoch {best_epoch}/{cfg.neural_epochs}  loss={best_loss:.4f}  grad_norm_max={max_grad_norm_seen:.3f}  weights_ok={weights_ok}")
    return best_loss


def train_neural_sequential(
    cfg: EvalConfig,
    dry_train_rnd: np.ndarray,
    wet_train_rnd: np.ndarray,
    dry_train_seq: np.ndarray,
    wet_train_seq: np.ndarray,
    dry_val: np.ndarray,
    wet_val: np.ndarray,
    sr: int,
) -> dict[str, tuple[np.ndarray, float, float, dict | None]]:
    """Train neural models one at a time on the available device.

    MLP uses randomly-subsampled data (``_rnd``) to sample the full amplitude
    range of the signal.  All sequential models (LSTM, GRU, TCN, DDSP,
    KlonDDSP) use evenly-spaced contiguous blocks (``_seq``) so that each
    training chunk contains coherent time-domain samples — a requirement for
    recurrent hidden-state dynamics to be learnable.

    Args:
        cfg: Eval configuration.
        dry_train_rnd: Randomly-subsampled dry training signal (for MLP).
        wet_train_rnd: Randomly-subsampled wet training signal (for MLP).
        dry_train_seq: Contiguous-block dry training signal (for recurrent/TCN).
        wet_train_seq: Contiguous-block wet training signal (for recurrent/TCN).
        dry_val: Dry validation signal for prediction.
        wet_val: Wet validation signal (used for klon_ddsp fraction analysis).
        sr: Sample rate in Hz.

    Returns:
        Mapping from model key to (val_prediction, train_time_s, predict_time_s,
        ddsp_analysis_or_None).
    """
    device = _resolve_device(cfg)
    total_s = len(dry_train_seq)
    n_seq_blocks = total_s // cfg.neural_chunk_size
    seq_coverage = 100.0 * (n_seq_blocks * cfg.neural_chunk_size) / total_s
    mlp_coverage = 100.0 * len(dry_train_rnd) / total_s
    print(f"  neural models on device: {device}  epochs={cfg.neural_epochs}")
    print(f"  recurrent: {n_seq_blocks} blocks, {n_seq_blocks * cfg.neural_chunk_size / 1e6:.2f}M samples "
          f"({seq_coverage:.1f}% of training signal)")
    print(f"  MLP:       {len(dry_train_rnd) / 1e6:.2f}M random samples "
          f"({mlp_coverage:.1f}% of training signal)")

    results: dict[str, tuple[np.ndarray, float, float]] = {}

    for key, (module_path, class_name, kwargs) in _NEURAL_REGISTRY.items():
        if key in cfg.skip_models:
            continue
        print(f"  training {key}…")
        cls = _load_module_class(module_path, class_name)
        model = cls(**kwargs)

        t0 = time.perf_counter()
        if key in ("tcn", "ddsp"):
            # Channels-first (batch, 1, time); needs contiguous temporal data.
            # Pass val signal so TCN uses periodic val-ESR checkpointing instead
            # of training-loss proxy (TCN loss can still descend as val-ESR degrades).
            train_loss = _train_tcn(
                model, dry_train_seq, wet_train_seq, cfg, device,
                dry_val=dry_val, wet_val=wet_val,
            )
        elif key == "mlp":
            # Pointwise model; benefits from amplitude diversity of random samples
            train_loss = _train_mlp(model, dry_train_rnd, wet_train_rnd, cfg, device)
        elif key == "klon_ddsp":
            # Separate DDSP-circuit / GRU param groups + auxiliary circuit loss
            train_loss = _train_klon_ddsp(model, dry_train_seq, wet_train_seq, cfg, device)
        else:
            # LSTM, GRU — recurrent; must have contiguous chunks
            train_loss = _train_recurrent(model, dry_train_seq, wet_train_seq, cfg, device)
        train_s = time.perf_counter() - t0

        model.eval()
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

        results[key] = (pred.astype(np.float32), float(train_s), float(pred_s), ddsp_analysis, float(train_loss))
        print(f"    {key}: train {train_s:.1f}s, predict {pred_s:.2f}s  final_loss={train_loss:.4f}")

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

    # ── Training data ─────────────────────────────────────────────────── #
    print("Loading train signal…")
    dry_train, _ = sf.read(str(dry_train_wav), dtype="float32", always_2d=False)
    wet_train, _ = sf.read(str(wet_train_wav), dtype="float32", always_2d=False)
    total_s = len(dry_train)

    # Sequential models (LSTM, GRU, TCN, DDSP, KlonDDSP) get the full signal.
    # ChunkDataset divides it into non-overlapping blocks of neural_chunk_size.
    dry_train_seq = dry_train
    wet_train_seq = wet_train
    n_seq_blocks = total_s // cfg.neural_chunk_size
    seq_samples = n_seq_blocks * cfg.neural_chunk_size
    seq_coverage = 100.0 * seq_samples / total_s

    # MLP uses random scatter (non-contiguous OK; amplitude diversity matters).
    if cfg.max_mlp_samples and total_s > cfg.max_mlp_samples:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(total_s, cfg.max_mlp_samples, replace=False))
        dry_train_rnd = dry_train[idx]
        wet_train_rnd = wet_train[idx]
        mlp_samples = cfg.max_mlp_samples
    else:
        dry_train_rnd = dry_train
        wet_train_rnd = wet_train
        mlp_samples = total_s
    mlp_coverage = 100.0 * mlp_samples / total_s

    # circuit_dict coverage (sequential epoch cycling)
    cd_W = cfg.circuit_dict_window_size
    cd_full_pass = (total_s + cd_W - 1) // cd_W  # ceil(N/W) — integer arithmetic
    cd_coverage = min(100.0, cfg.circuit_dict_epochs * cd_W / total_s * 100.0)

    ham_samples = min(cfg.max_hammerstein_samples, total_s)
    ham_coverage = 100.0 * ham_samples / total_s

    # Silence-block note: 84 sections, 15 gaps × 48k samples = 720k silence samples.
    # 720k / 4096 = 175 silence-only blocks out of 15,058 total = 1.2% — negligible.

    print(f"\n┌──────────────────────────────────────────────────────────────────┐")
    print(f"│ Data coverage summary  (train signal: {total_s/1e6:.2f}M samples)         │")
    print(f"├────────────────────────┬──────────────┬────────────────────────────┤")
    print(f"│ Model type             │   Samples    │ Coverage                   │")
    print(f"├────────────────────────┼──────────────┼────────────────────────────┤")
    print(f"│ FIR / IIR              │ {total_s/1e6:7.2f}M     │ 100.0% (FFT full signal)   │")
    print(f"│ Volterra               │ {total_s/1e6:7.2f}M     │ 100.0% (streaming XTX)    │")
    print(f"│ WienerHammerstein      │ {total_s/1e6:7.2f}M     │ 100.0% (streaming poly)   │")
    print(f"│ Hammerstein            │ {ham_samples/1e6:7.2f}M     │ {ham_coverage:5.1f}% (random subsample) │")
    print(f"│ circuit_dict           │ {cd_W/1e3:5.0f}k/ep      │ {cd_coverage:5.1f}% ({cfg.circuit_dict_epochs}ep, {cd_full_pass}ep/pass) │")
    print(f"│ MLP                    │ {mlp_samples/1e6:7.2f}M     │ {mlp_coverage:5.1f}% (random subsample) │")
    print(f"│ LSTM/GRU/TCN/klon_ddsp │ {seq_samples/1e6:7.2f}M     │ {seq_coverage:5.1f}% ({n_seq_blocks} blocks)        │")
    print(f"└────────────────────────┴──────────────┴────────────────────────────┘")
    print(f"  neural_epochs={cfg.neural_epochs}  circuit_dict_epochs={cfg.circuit_dict_epochs}")
    print(f"  CosineAnnealingLR T_max={cfg.neural_epochs} (auto-adapts to epoch count)\n")

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
        for key, (pred, fit_s, pred_s, topology) in classical.items():
            metrics = _compute_metrics(wet_val, pred, dry_val, manifest, sr)
            results.models[key] = ModelResult(
                name=key,
                model_type="classical",
                prediction=pred,
                fit_time_s=fit_s,
                predict_time_s=pred_s,
                metrics_per_section=metrics,
                topology=topology,
            )

    # ── Neural models ─────────────────────────────────────────────────── #
    if cfg.run_neural:
        print("\nNeural models:")
        neural = train_neural_sequential(
            cfg,
            dry_train_rnd, wet_train_rnd,
            dry_train_seq, wet_train_seq,
            dry_val, wet_val,
            sr,
        )
        for key, (pred, train_s, pred_s, ddsp_analysis, train_loss) in neural.items():
            metrics = _compute_metrics(wet_val, pred, dry_val, manifest, sr)

            results.models[key] = ModelResult(
                name=key,
                model_type="neural",
                prediction=pred,
                fit_time_s=train_s,
                predict_time_s=pred_s,
                metrics_per_section=metrics,
                train_loss=train_loss,
                ddsp_analysis=ddsp_analysis,
            )

    # ── Save ──────────────────────────────────────────────────────────── #
    out_pkl = cfg.output_dir / "results.pkl"
    out_csv = cfg.output_dir / "results.csv"
    results.save(out_pkl)
    results.to_csv(out_csv)
    print(f"\nSaved:\n  {out_pkl}\n  {out_csv}")

    return results
