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
    tcn_epochs: int = 0  # 0 = use neural_epochs; override just for TCN/DDSP
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
    # Optional WAV path overrides — bypass the paths derived from params.json when set.
    dry_wav: Path | None = None
    wet_wav: Path | None = None
    dry_val_wav: Path | None = None
    wet_val_wav: Path | None = None
    experiment_name: str | None = None

    def __post_init__(self) -> None:
        self.klon_sim_dir = Path(self.klon_sim_dir)
        self.output_dir = Path(self.output_dir)
        if self.dry_wav is not None:
            self.dry_wav = Path(self.dry_wav)
        if self.wet_wav is not None:
            self.wet_wav = Path(self.wet_wav)
        if self.dry_val_wav is not None:
            self.dry_val_wav = Path(self.dry_val_wav)
        if self.wet_val_wav is not None:
            self.wet_val_wav = Path(self.wet_val_wav)


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


# ── Block filter helpers ──────────────────────────────────────────────────── #


def _bad_block_set(
    manifest: Any, chunk_size: int, total_samples: int
) -> frozenset[int]:
    """Block indices to exclude from recurrent training.

    Excludes every block that contains at least one silence sample (i.e. a
    sample not within any manifest section's [start, end) range), plus all
    blocks that overlap an impulse_train section.

    Silence blocks cause ESR loss to divide by ≈1e-8 once the model has
    non-zero bias-driven output, producing gradients ~1e8× amplified that
    corrupt LSTM/GRU parameters over successive epochs.  Impulse_train blocks
    contain sharp high-amplitude transients that spike the loss even without
    the silence problem.

    Args:
        manifest: Loaded Manifest for the training signal.
        chunk_size: Block length in samples (neural_chunk_size).
        total_samples: Total training signal length in samples.

    Returns:
        frozenset of 0-indexed block indices to exclude.
    """
    n_blocks = total_samples // chunk_size

    # Mark every sample that belongs to an active section.
    active = np.zeros(total_samples, dtype=bool)
    for sec in manifest.sections:
        s = max(0, sec.start_sample)
        e = min(total_samples, sec.end_sample)
        if e > s:
            active[s:e] = True

    # A block is bad if it contains any silence sample.
    block_active = active[:n_blocks * chunk_size].reshape(n_blocks, chunk_size)
    bad = set(int(i) for i in np.where(~block_active.all(axis=1))[0])

    # All blocks that overlap any impulse_train section.
    for sec in manifest.sections_of_type("impulse_train"):
        first = sec.start_sample // chunk_size
        last = min((sec.end_sample - 1) // chunk_size, n_blocks - 1)
        bad.update(range(first, last + 1))

    return frozenset(bad)


def _filter_seq_blocks(
    dry: np.ndarray, wet: np.ndarray, chunk_size: int, bad_blocks: frozenset[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Remove bad blocks and return filtered contiguous arrays.

    Args:
        dry: Full dry training signal, shape (N,), float32.
        wet: Full wet training signal, same shape.
        chunk_size: Block length in samples.
        bad_blocks: Block indices to exclude (from _bad_block_set).

    Returns:
        Filtered (dry, wet) as contiguous float32 arrays.
    """
    n_blocks = len(dry) // chunk_size
    good = [i for i in range(n_blocks) if i not in bad_blocks]
    if len(good) == n_blocks:
        return dry[:n_blocks * chunk_size], wet[:n_blocks * chunk_size]
    d = np.concatenate([dry[i * chunk_size : (i + 1) * chunk_size] for i in good])
    w = np.concatenate([wet[i * chunk_size : (i + 1) * chunk_size] for i in good])
    return d, w


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

    return key, pred.astype(np.float32), float(fit_s), float(pred_s), topology, model


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
    results: dict[str, tuple[np.ndarray, float, float, dict | None, Any]] = {}

    print(f"  fitting {len(tasks)} classical model(s) with {n_workers} worker(s)…")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_classical_worker, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            key_done = futures[fut]
            try:
                key_r, pred, fit_s, pred_s, topology, model = fut.result()
                results[key_r] = (pred, fit_s, pred_s, topology, model)
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
    bad_blocks: frozenset[int] | None = None,
) -> float:
    """Stateful batched TBPTT for LSTM/GRU.

    Divides good blocks into B contiguous segments (B = neural_batch_size).
    Each step processes one chunk from every segment simultaneously, so GPU
    utilisation matches the old shuffled-batch approach.  Hidden state carries
    between consecutive blocks within a segment and resets at gaps caused by
    bad-block filtering (section boundaries, silence regions).

    This matches inference (predict() carries state continuously), eliminating
    the train–val distribution mismatch seen with shuffled stateless training.

    Args:
        dry_train: Full dry training signal (unfiltered), shape (N,), float32.
        wet_train: Full wet training signal, same shape.
        bad_blocks: Block indices to skip.  State resets wherever consecutive
            good blocks are non-adjacent in the original signal.

    Returns:
        Best epoch average training loss.
    """
    from pedal_model.train.losses import CombinedLoss

    chunk_size = cfg.neural_chunk_size
    n_blocks = len(dry_train) // chunk_size
    bad = bad_blocks or frozenset()
    good_blocks = [i for i in range(n_blocks) if i not in bad]

    if not good_blocks:
        raise ValueError("_train_recurrent: all blocks filtered — nothing to train on.")

    # Split good_blocks into B contiguous segments for batched stateful TBPTT.
    B = cfg.neural_batch_size
    n_good = len(good_blocks)
    seg_size = max(n_good // min(B, n_good), 1)
    segments: list[list[int]] = []
    for b in range(min(B, n_good)):
        s0 = b * seg_size
        s1 = (b + 1) * seg_size if b < min(B, n_good) - 1 else n_good
        if s0 < n_good:
            segments.append(good_blocks[s0:s1])
    B_actual = len(segments)
    n_steps = min(len(seg) for seg in segments)

    is_lstm = hasattr(model, "lstm")
    hid_layers: int = model.num_layers
    hid_size: int = model.hidden_size

    loss_fn = CombinedLoss().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.neural_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg.neural_epochs
    )
    model.to(device).train()

    def _fresh_hidden() -> Any:
        h = torch.zeros(hid_layers, B_actual, hid_size, device=device)
        return (h, h.clone()) if is_lstm else h

    best_loss = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    max_grad_norm_seen = 0.0
    t_loop_start = time.perf_counter()
    total_nan_skips = 0

    for epoch in range(1, cfg.neural_epochs + 1):
        epoch_loss = 0.0
        n_valid = 0
        epoch_max_grad_norm = 0.0
        hidden = _fresh_hidden()

        for step in range(n_steps):
            # Reset hidden for any segment whose next block is non-adjacent in the
            # original signal (gap due to a filtered bad block).
            if step > 0:
                for b, seg in enumerate(segments):
                    if seg[step] != seg[step - 1] + 1:
                        if is_lstm:
                            hidden[0][:, b, :] = 0.0
                            hidden[1][:, b, :] = 0.0
                        else:
                            hidden[:, b, :] = 0.0

            chunks_d = [dry_train[seg[step] * chunk_size : (seg[step] + 1) * chunk_size]
                        for seg in segments]
            chunks_w = [wet_train[seg[step] * chunk_size : (seg[step] + 1) * chunk_size]
                        for seg in segments]
            dry_t = torch.tensor(
                np.stack(chunks_d), dtype=torch.float32
            ).unsqueeze(-1).to(device)  # (B, T, 1)
            wet_t = torch.tensor(
                np.stack(chunks_w), dtype=torch.float32
            ).unsqueeze(-1).to(device)

            optimiser.zero_grad()
            out, new_h = model(dry_t, hidden)
            # TBPTT: detach so gradient doesn't flow across chunk boundaries.
            if is_lstm:
                hidden = (new_h[0].detach(), new_h[1].detach())
            else:
                hidden = new_h.detach()

            if isinstance(out, tuple):
                out = out[0]
            if out.dim() == 3 and out.shape[1] == 1 and out.shape[2] != 1:
                out = out.transpose(1, 2)

            loss = loss_fn(out, wet_t)
            if not loss.isfinite():
                total_nan_skips += 1
                hidden = _fresh_hidden()
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            if not np.isfinite(grad_norm):
                optimiser.zero_grad()
                if best_state is not None:
                    model.load_state_dict(best_state)
                hidden = _fresh_hidden()
                total_nan_skips += 1
                continue

            epoch_max_grad_norm = max(epoch_max_grad_norm, grad_norm)
            max_grad_norm_seen = max(max_grad_norm_seen, grad_norm)
            optimiser.step()
            epoch_loss += loss.item()
            n_valid += 1

        scheduler.step()
        avg = epoch_loss / max(1, n_valid)
        if avg < best_loss:
            best_loss = avg
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        # Print every epoch for the first 10 (early grad_norm diagnostic),
        # then at 20% intervals with grad_norm included throughout.
        periodic = epoch % max(1, cfg.neural_epochs // 5) == 0
        if epoch <= 10 or periodic:
            elapsed = time.perf_counter() - t_loop_start
            eta_s = elapsed / epoch * (cfg.neural_epochs - epoch)
            eta_str = time.strftime("%H:%M:%S", time.localtime(time.time() + eta_s))
            skip_str = f"  nan_skips={total_nan_skips}" if total_nan_skips else ""
            print(
                f"      epoch {epoch}/{cfg.neural_epochs}  loss={avg:.4f}  "
                f"best={best_loss:.4f}@ep{best_epoch}  "
                f"grad_norm={epoch_max_grad_norm:.3e}"
                f"  elapsed={elapsed:.0f}s  eta {eta_str}{skip_str}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    weights_ok = all(torch.isfinite(p).all().item() for p in model.parameters())
    skip_str = f"  nan_skips={total_nan_skips}" if total_nan_skips else ""
    print(
        f"      best checkpoint: epoch {best_epoch}/{cfg.neural_epochs}  "
        f"loss={best_loss:.4f}  grad_norm_max={max_grad_norm_seen:.3f}  "
        f"weights_ok={weights_ok}{skip_str}"
    )
    if max_grad_norm_seen > 100.0:
        print(
            f"      WARNING: grad_norm_max={max_grad_norm_seen:.1f} at best checkpoint. "
            "Consider reducing learning rate or increasing gradient clip threshold. "
            "Training may be unstable."
        )
    return best_loss, {"epoch": best_epoch, "grad_norm_max": max_grad_norm_seen}


def _train_tcn(
    model: Any,
    dry_train: np.ndarray,
    wet_train: np.ndarray,
    cfg: EvalConfig,
    device: torch.device,
    dry_val: np.ndarray | None = None,
    wet_val: np.ndarray | None = None,
    val_check_interval: int = 50,
    n_epochs: int = 0,
) -> float:
    """Train TCN model (expects channels-first (batch, 1, time) input).

    Uses val-ESR checkpointing every `val_check_interval` epochs when dry_val
    and wet_val are provided.  TCN's training loss can still descend while
    val-ESR degrades (overfitting to training distribution), so training-loss
    is not a reliable proxy — periodic val-ESR is the correct signal.

    Args:
        n_epochs: Override epoch count (0 = use cfg.neural_epochs). Use this
            to run TCN longer than other neural models without changing cfg.

    Returns:
        Best val-ESR (or best training loss when val not provided).
    """
    from pedal_model.data.dataset import ChunkDataset
    from pedal_model.train.losses import CombinedLoss

    n_ep = n_epochs if n_epochs > 0 else cfg.neural_epochs

    dataset = ChunkDataset(dry_train, wet_train, chunk_size=cfg.neural_chunk_size)
    loader = DataLoader(dataset, batch_size=cfg.neural_batch_size, shuffle=True, drop_last=True)
    loss_fn = CombinedLoss().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.neural_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_ep)
    model.to(device).train()

    use_val_esr = dry_val is not None and wet_val is not None
    best_metric = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    max_grad_norm_seen = 0.0
    t_loop_start = time.perf_counter()

    avg = float("nan")
    for epoch in range(1, n_ep + 1):
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
            if val_esr < best_metric:
                best_metric = val_esr
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch % 100 == 0 or epoch == n_ep:
                elapsed = time.perf_counter() - t_loop_start
                eta_s = elapsed / epoch * (n_ep - epoch)
                eta_str = time.strftime("%H:%M:%S", time.localtime(time.time() + eta_s))
                print(f"      epoch {epoch}/{n_ep}  loss={avg:.4f}  val_esr={val_esr:.4f}  best={best_metric:.4f}@ep{best_epoch}"
                      f"  grad_norm={max_grad_norm_seen:.3e}  elapsed={elapsed:.0f}s  eta {eta_str}")
        else:
            if not use_val_esr and avg < best_metric:
                best_metric = avg
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch % max(1, n_ep // 5) == 0:
                elapsed = time.perf_counter() - t_loop_start
                eta_s = elapsed / epoch * (n_ep - epoch)
                eta_str = time.strftime("%H:%M:%S", time.localtime(time.time() + eta_s))
                print(f"      epoch {epoch}/{n_ep}  loss={avg:.4f}  best={best_metric:.4f}@ep{best_epoch}"
                      f"  grad_norm={max_grad_norm_seen:.3e}  elapsed={elapsed:.0f}s  eta {eta_str}")

    # If val-ESR checkpointing never fired (n_ep < val_check_interval),
    # run one final check so best_state is always set before restore.
    if use_val_esr and best_state is None:
        from pedal_model.metrics.time_domain import compute_esr
        model.eval()
        pred = model.predict(dry_val)
        model.train()
        n = min(len(wet_val), len(pred))
        best_metric = compute_esr(wet_val[:n], pred[:n])
        best_epoch = n_ep
        best_state = copy.deepcopy(model.state_dict())
        print(f"      (final val-ESR check: {best_metric:.4f} — val_check_interval > n_epochs)")

    if best_state is not None:
        model.load_state_dict(best_state)
    weights_ok = all(torch.isfinite(p).all().item() for p in model.parameters())
    metric_name = "val_esr" if use_val_esr else "train_loss"
    print(f"      best checkpoint: epoch {best_epoch}/{n_ep}  {metric_name}={best_metric:.4f}  grad_norm_max={max_grad_norm_seen:.3f}  weights_ok={weights_ok}")
    if max_grad_norm_seen > 100.0:
        print(
            f"      WARNING: grad_norm_max={max_grad_norm_seen:.1f} at best checkpoint. "
            "Consider reducing learning rate or increasing gradient clip threshold. "
            "Training may be unstable."
        )
    return best_metric, {"epoch": best_epoch, "grad_norm_max": max_grad_norm_seen}


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
    return avg, {"epoch": cfg.neural_epochs, "grad_norm_max": 0.0}


def _train_klon_ddsp(
    model: Any,
    dry_train: np.ndarray,
    wet_train: np.ndarray,
    cfg: EvalConfig,
    device: torch.device,
    ddsp_lr_scale: float = 0.1,
    aux_weight: float = 0.1,
    bad_blocks: frozenset[int] | None = None,
) -> float:
    """Stateful batched TBPTT for KlonDDSP (DDSP circuit + GRU corrector).

    Uses the same segment-based stateful TBPTT as _train_recurrent so that the
    GRU corrector's hidden state carries between consecutive signal chunks rather
    than being discarded at every step.  This eliminates the train–val mismatch
    that caused ESR=13 in the stateless version.

    Keeps separate DDSP-circuit / GRU param groups (GRU dominates with shared
    lr) and the auxiliary DDSP-stage loss (prevents DDSP collapse to zero).

    Args:
        dry_train: Full dry training signal (unfiltered), shape (N,), float32.
        wet_train: Full wet training signal, same shape.
        cfg: Eval configuration.
        device: Compute device.
        ddsp_lr_scale: LR multiplier for DDSP circuit params (default 0.1).
        aux_weight: Weight of auxiliary loss on DDSP-stage output (default 0.1).
        bad_blocks: Block indices to skip; state resets at gaps in original order.

    Returns:
        Best training loss (combined + auxiliary).
    """
    from pedal_model.train.losses import CombinedLoss

    chunk_size = cfg.neural_chunk_size
    n_blocks = len(dry_train) // chunk_size
    bad = bad_blocks or frozenset()
    good_blocks = [i for i in range(n_blocks) if i not in bad]

    if not good_blocks:
        raise ValueError("_train_klon_ddsp: all blocks filtered — nothing to train on.")

    B = cfg.neural_batch_size
    n_good = len(good_blocks)
    seg_size = max(n_good // min(B, n_good), 1)
    segments: list[list[int]] = []
    for b in range(min(B, n_good)):
        s0 = b * seg_size
        s1 = (b + 1) * seg_size if b < min(B, n_good) - 1 else n_good
        if s0 < n_good:
            segments.append(good_blocks[s0:s1])
    B_actual = len(segments)
    n_steps = min(len(seg) for seg in segments)

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

    def _fresh_hidden() -> torch.Tensor:
        # KlonDDSP GRU: 1 layer, hidden shape (1, batch, gru_hidden)
        return torch.zeros(1, B_actual, model.gru_hidden, device=device)

    best_loss = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    max_grad_norm_seen = 0.0
    t_loop_start = time.perf_counter()
    total_nan_skips = 0

    for epoch in range(1, cfg.neural_epochs + 1):
        epoch_loss = 0.0
        n_valid = 0
        hidden = _fresh_hidden()

        for step in range(n_steps):
            # Reset GRU state at section boundaries (non-adjacent good blocks).
            if step > 0:
                for b, seg in enumerate(segments):
                    if seg[step] != seg[step - 1] + 1:
                        hidden[:, b, :] = 0.0

            chunks_d = [dry_train[seg[step] * chunk_size : (seg[step] + 1) * chunk_size]
                        for seg in segments]
            chunks_w = [wet_train[seg[step] * chunk_size : (seg[step] + 1) * chunk_size]
                        for seg in segments]
            dry_t = torch.tensor(
                np.stack(chunks_d), dtype=torch.float32
            ).unsqueeze(-1).to(device)  # (B, T, 1)
            wet_t = torch.tensor(
                np.stack(chunks_w), dtype=torch.float32
            ).unsqueeze(-1).to(device)

            optimiser.zero_grad()
            out, new_h = model(dry_t, hidden)
            # TBPTT: detach GRU state so gradient doesn't flow across chunk boundaries.
            hidden = new_h.detach()

            loss = loss_fn(out, wet_t)

            # Auxiliary loss on DDSP stage alone — keeps DDSP from collapsing.
            if aux_weight > 0.0:
                x_2d = dry_t.squeeze(-1)   # (B, T)
                ddsp_out = model._ddsp_stage(x_2d).unsqueeze(-1)   # (B, T, 1)
                aux = loss_fn(ddsp_out, wet_t)
                loss = loss + aux_weight * aux

            if not loss.isfinite():
                total_nan_skips += 1
                hidden = _fresh_hidden()
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            if not np.isfinite(grad_norm):
                optimiser.zero_grad()
                if best_state is not None:
                    model.load_state_dict(best_state)
                hidden = _fresh_hidden()
                total_nan_skips += 1
                continue

            max_grad_norm_seen = max(max_grad_norm_seen, grad_norm)
            optimiser.step()
            epoch_loss += loss.item()
            n_valid += 1

        scheduler.step()
        avg = epoch_loss / max(1, n_valid)
        if avg < best_loss:
            best_loss = avg
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % max(1, cfg.neural_epochs // 5) == 0:
            elapsed = time.perf_counter() - t_loop_start
            eta_s = elapsed / epoch * (cfg.neural_epochs - epoch)
            eta_str = time.strftime("%H:%M:%S", time.localtime(time.time() + eta_s))
            skip_str = f"  nan_skips={total_nan_skips}" if total_nan_skips else ""
            print(
                f"      epoch {epoch}/{cfg.neural_epochs}  loss={avg:.4f}  "
                f"best={best_loss:.4f}@ep{best_epoch}"
                f"  elapsed={elapsed:.0f}s  eta {eta_str}{skip_str}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    weights_ok = all(torch.isfinite(p).all().item() for p in model.parameters())
    skip_str = f"  nan_skips={total_nan_skips}" if total_nan_skips else ""
    print(
        f"      best checkpoint: epoch {best_epoch}/{cfg.neural_epochs}  "
        f"loss={best_loss:.4f}  grad_norm_max={max_grad_norm_seen:.3f}  "
        f"weights_ok={weights_ok}{skip_str}"
    )
    if max_grad_norm_seen > 100.0:
        print(
            f"      WARNING: grad_norm_max={max_grad_norm_seen:.1f} at best checkpoint. "
            "Consider reducing learning rate or increasing gradient clip threshold. "
            "Training may be unstable."
        )
    return best_loss, {"epoch": best_epoch, "grad_norm_max": max_grad_norm_seen}


def train_neural_sequential(
    cfg: EvalConfig,
    dry_train_rnd: np.ndarray,
    wet_train_rnd: np.ndarray,
    dry_train_seq: np.ndarray,
    wet_train_seq: np.ndarray,
    dry_val: np.ndarray,
    wet_val: np.ndarray,
    sr: int,
    bad_blocks: frozenset[int] | None = None,
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
        bad_blocks: Block indices to exclude from sequential training (silence,
            impulse_train, boundary blocks). Computed from train manifest by
            run_zoo via _bad_block_set. None = no filtering.

    Returns:
        Mapping from model key to (val_prediction, train_time_s, predict_time_s,
        ddsp_analysis_or_None).
    """
    device = _resolve_device(cfg)
    total_s = len(dry_train_seq)
    n_seq_blocks_raw = total_s // cfg.neural_chunk_size

    # Keep the original (unfiltered) arrays for recurrent models, which use
    # stateful TBPTT and need the full signal to know where section gaps are.
    # Filtered arrays are used for TCN/DDSP where filtering is correct (no state).
    dry_seq_orig = dry_train_seq
    wet_seq_orig = wet_train_seq
    if bad_blocks:
        dry_train_seq, wet_train_seq = _filter_seq_blocks(
            dry_seq_orig, wet_seq_orig, cfg.neural_chunk_size, bad_blocks
        )
        n_removed = n_seq_blocks_raw - len(dry_train_seq) // cfg.neural_chunk_size
    else:
        n_removed = 0

    n_seq_blocks = len(dry_train_seq) // cfg.neural_chunk_size
    seq_samples = n_seq_blocks * cfg.neural_chunk_size
    seq_coverage = 100.0 * seq_samples / total_s
    mlp_coverage = 100.0 * len(dry_train_rnd) / total_s

    tcn_n_ep = cfg.tcn_epochs if cfg.tcn_epochs > 0 else cfg.neural_epochs

    print(f"  neural models on device: {device}  epochs={cfg.neural_epochs}"
          + (f"  tcn={tcn_n_ep}" if tcn_n_ep != cfg.neural_epochs else ""))
    print(f"  recurrent: {n_seq_blocks} blocks, {seq_samples / 1e6:.2f}M samples "
          f"({seq_coverage:.1f}% of training signal)")
    if bad_blocks:
        print(f"    → removed {n_removed} silence/impulse/boundary blocks "
              f"({len(bad_blocks)} bad of {n_seq_blocks_raw} total)")
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
            n_ep = tcn_n_ep if key == "tcn" else cfg.neural_epochs
            train_loss, train_meta = _train_tcn(
                model, dry_train_seq, wet_train_seq, cfg, device,
                dry_val=dry_val, wet_val=wet_val,
                n_epochs=n_ep,
            )
        elif key == "mlp":
            # Pointwise model; benefits from amplitude diversity of random samples
            train_loss, train_meta = _train_mlp(model, dry_train_rnd, wet_train_rnd, cfg, device)
        elif key == "klon_ddsp":
            # Stateful TBPTT: GRU state carries between consecutive chunks.
            # Uses original (unfiltered) signal so segment gaps are tracked correctly.
            train_loss, train_meta = _train_klon_ddsp(
                model, dry_seq_orig, wet_seq_orig, cfg, device,
                bad_blocks=bad_blocks,
            )
        else:
            # LSTM, GRU, conditioned_lstm — stateful TBPTT with original signal + gap tracking.
            train_loss, train_meta = _train_recurrent(
                model, dry_seq_orig, wet_seq_orig, cfg, device,
                bad_blocks=bad_blocks,
            )
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

        model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
        results[key] = (pred.astype(np.float32), float(train_s), float(pred_s), ddsp_analysis, float(train_loss), model_state, train_meta)
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

    # WAV path overrides: when set, use caller-supplied paths instead of params.json paths.
    if cfg.dry_wav is not None:
        dry_train_wav = cfg.dry_wav
        print(f"  dry_wav override: {dry_train_wav}")
    if cfg.wet_wav is not None:
        wet_train_wav = cfg.wet_wav
        print(f"  wet_wav override: {wet_train_wav}")
    if cfg.dry_val_wav is not None:
        dry_val_wav = cfg.dry_val_wav
        print(f"  dry_val_wav override: {dry_val_wav}")
    if cfg.wet_val_wav is not None:
        wet_val_wav = cfg.wet_val_wav
        print(f"  wet_val_wav override: {wet_val_wav}")

    print(f"Loading val signal ({params['captures']['val']['duration_s']:.1f}s @ {sr}Hz)…")
    dry_val, _ = sf.read(str(dry_val_wav), dtype="float32", always_2d=False)
    wet_val, _ = sf.read(str(wet_val_wav), dtype="float32", always_2d=False)
    # Clip to matching length — DAW recordings can differ by a handful of frames.
    _min_val = min(len(dry_val), len(wet_val))
    dry_val, wet_val = dry_val[:_min_val], wet_val[:_min_val]

    manifest = Manifest(dry_val_manifest_path)
    train_manifest = Manifest(dry_manifest_path)

    # ── Training data ─────────────────────────────────────────────────── #
    print("Loading train signal…")
    dry_train, _ = sf.read(str(dry_train_wav), dtype="float32", always_2d=False)
    wet_train, _ = sf.read(str(wet_train_wav), dtype="float32", always_2d=False)
    _min_train = min(len(dry_train), len(wet_train))
    dry_train, wet_train = dry_train[:_min_train], wet_train[:_min_train]
    total_s = len(dry_train)

    # Sequential models (LSTM, GRU, TCN, DDSP, KlonDDSP) get the full signal.
    # ChunkDataset divides it into non-overlapping blocks of neural_chunk_size.
    dry_train_seq = dry_train
    wet_train_seq = wet_train
    n_seq_blocks_raw = total_s // cfg.neural_chunk_size
    seq_samples = n_seq_blocks_raw * cfg.neural_chunk_size
    seq_coverage = 100.0 * seq_samples / total_s

    # Block filter: exclude silence-containing blocks (ESR ÷ 1e-8 blowup) and
    # impulse_train blocks from the sequential training pool.
    bad_blocks = _bad_block_set(train_manifest, cfg.neural_chunk_size, total_s)
    n_seq_blocks = n_seq_blocks_raw - len(bad_blocks)

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

    seq_filt_samples = n_seq_blocks * cfg.neural_chunk_size
    seq_filt_coverage = 100.0 * seq_filt_samples / total_s

    tcn_n_ep = cfg.tcn_epochs if cfg.tcn_epochs > 0 else cfg.neural_epochs

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
    print(f"│ LSTM/GRU/TCN/klon_ddsp │ {seq_filt_samples/1e6:7.2f}M     │ {seq_filt_coverage:5.1f}% ({n_seq_blocks} blocks, -{len(bad_blocks)} filtered) │")
    print(f"└────────────────────────┴──────────────┴────────────────────────────┘")
    ep_line = f"  neural_epochs={cfg.neural_epochs}  circuit_dict_epochs={cfg.circuit_dict_epochs}"
    if tcn_n_ep != cfg.neural_epochs:
        ep_line += f"  tcn_epochs={tcn_n_ep}"
    print(ep_line)
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
        for key, (pred, fit_s, pred_s, topology, model_obj) in classical.items():
            metrics = _compute_metrics(wet_val, pred, dry_val, manifest, sr)
            results.models[key] = ModelResult(
                name=key,
                model_type="classical",
                prediction=pred,
                fit_time_s=fit_s,
                predict_time_s=pred_s,
                metrics_per_section=metrics,
                topology=topology,
                model_state=model_obj,
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
            bad_blocks=bad_blocks,
        )
        for key, (pred, train_s, pred_s, ddsp_analysis, train_loss, model_state, _train_meta) in neural.items():
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
                model_state=model_state,
            )

    # ── Save ──────────────────────────────────────────────────────────── #
    out_pkl = cfg.output_dir / "results.pkl"
    out_csv = cfg.output_dir / "results.csv"
    results.save(out_pkl)
    results.to_csv(out_csv)
    print(f"\nSaved:\n  {out_pkl}\n  {out_csv}")

    return results
