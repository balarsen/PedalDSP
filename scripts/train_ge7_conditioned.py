#!/usr/bin/env python3
"""train_ge7_conditioned.py — train a conditioned LSTM on all GE-7 preset captures.

One model covers the full slider parameter space. Each preset's 8-dim slider
vector is embedded into the LSTM input at every time step, so the network
learns how gain, not just a single static EQ, changes with the knobs.

Usage:
    uv run python scripts/train_ge7_conditioned.py
    uv run python scripts/train_ge7_conditioned.py --epochs 200 --hidden 64 --device cuda
    uv run python scripts/train_ge7_conditioned.py --presets bright bass_boost flat
    uv run python scripts/train_ge7_conditioned.py --exclude all_up all_up_2 all_up_3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pedal_model.metrics.time_domain import compute_esr
from pedal_model.models.neural.conditioned_lstm import ConditionedLSTMModel
from pedal_model.models.neural.conditioned_tcn import ConditionedTCNModel
from pedal_model.train.losses import CombinedLoss

# ── Constants ─────────────────────────────────────────────────────────────── #

ROOT = PROJECT_ROOT
DRY_WAV = ROOT / "recordings/data/signals/train_signal_v2_48k.wav"
PRESETS_DIR = ROOT / "recordings/data/ge7/presets"
CAPTURES_DIR = ROOT / "recordings/data/ge7/captures"

SLIDER_KEYS: list[str] = [
    "hz_100", "hz_200", "hz_400", "hz_800",
    "hz_1600", "hz_3200", "hz_6400", "level",
]
SLIDER_MAX = 15.0       # physical range -15 to +15 dB, normalised to [-1, 1]
REF_FADER_DB = -10.0   # DAW fader level used for the majority of captures

# Captures excluded by default: ADC clipping or hardware saturation
DEFAULT_EXCLUDE: frozenset[str] = frozenset({"all_up", "all_up_2"})


# ── Preset loading ────────────────────────────────────────────────────────── #

def _find_wet_wav(capture_dir: Path) -> Path:
    """Return the wet WAV in a capture dir (canonical name, or the only WAV).

    Args:
        capture_dir: Directory under recordings/data/ge7/captures/preset_<name>/.

    Returns:
        Path to the wet WAV file.
    """
    canonical = capture_dir / "wet_train_signal_v2.wav"
    if canonical.exists():
        return canonical
    wavs = [f for f in capture_dir.glob("*.wav")]
    if len(wavs) == 1:
        return wavs[0]
    raise FileNotFoundError(
        f"Cannot determine wet WAV in {capture_dir}. Found: {[w.name for w in wavs]}"
    )


def load_preset_meta(
    exclude: frozenset[str] = DEFAULT_EXCLUDE,
    only: list[str] | None = None,
) -> list[dict]:
    """Load metadata for all usable GE-7 presets.

    Args:
        exclude: Preset names to skip.
        only: If provided, restrict to this list of preset names.

    Returns:
        List of dicts: name, knobs (np.ndarray (8,) in [-1,1]),
        wet_wav (Path), fader_correction (float linear multiplier).
    """
    result = []
    for json_path in sorted(PRESETS_DIR.glob("preset_*.json")):
        meta = json.loads(json_path.read_text())
        name = meta["preset_name"]
        if name in exclude:
            continue
        if only is not None and name not in only:
            continue

        capture_dir = CAPTURES_DIR / f"preset_{name}"
        if not capture_dir.exists():
            print(f"  [skip] {name}: no capture directory")
            continue
        try:
            wet_wav = _find_wet_wav(capture_dir)
        except FileNotFoundError as e:
            print(f"  [skip] {name}: {e}")
            continue

        knobs = np.array(
            [meta["sliders"][k] for k in SLIDER_KEYS], dtype=np.float32
        ) / SLIDER_MAX

        preset_fader_db = float(meta.get("fader_db", REF_FADER_DB))
        fader_correction = 10.0 ** ((REF_FADER_DB - preset_fader_db) / 20.0)

        result.append(
            {"name": name, "knobs": knobs, "wet_wav": wet_wav,
             "fader_correction": fader_correction}
        )
    return result


# ── Dataset ───────────────────────────────────────────────────────────────── #

class ConditionedChunkDataset(Dataset):
    """Chunk pool from multiple GE-7 preset captures.

    Each item is a (dry_chunk, wet_chunk, knobs) triple:
      - dry_chunk / wet_chunk: shape (chunk_size, 1), float32 tensors
      - knobs: shape (n_knobs,), float32 tensor, values in [-1, 1]

    All presets share the same dry signal and contribute equal numbers of
    chunks, so the model sees each slider setting equally often.

    Args:
        dry: Shared dry signal, shape (N,), float32.
        wet_arrays: One wet array per preset, each shape (N,), float32.
        knobs_list: Normalised knob vector per preset, each shape (n_knobs,).
        chunk_size: TBPTT chunk length in samples. Hidden state is reset
            between chunks (no carry across batch items).
        n_samples: Use only the first n_samples of the signal (train split).
    """

    def __init__(
        self,
        dry: np.ndarray,
        wet_arrays: list[np.ndarray],
        knobs_list: list[np.ndarray],
        chunk_size: int = 4096,
        n_samples: int | None = None,
    ) -> None:
        n = n_samples if n_samples is not None else len(dry)
        n = (n // chunk_size) * chunk_size
        self._dry = dry[:n]
        self._wet = [w[:n] for w in wet_arrays]
        self._knobs = knobs_list
        self._chunk_size = chunk_size
        n_chunks = n // chunk_size
        self._items: list[tuple[int, int]] = [
            (p, c) for p in range(len(wet_arrays)) for c in range(n_chunks)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p, c = self._items[idx]
        s, e = c * self._chunk_size, (c + 1) * self._chunk_size
        dry_chunk = torch.from_numpy(self._dry[s:e]).unsqueeze(-1)     # (T, 1)
        wet_chunk = torch.from_numpy(self._wet[p][s:e]).unsqueeze(-1)  # (T, 1)
        knobs = torch.from_numpy(self._knobs[p])                        # (n_knobs,)
        return dry_chunk, wet_chunk, knobs


# ── Validation ────────────────────────────────────────────────────────────── #

def eval_preset_esr(
    model: ConditionedLSTMModel | ConditionedTCNModel,
    dry_val: np.ndarray,
    wet_val: np.ndarray,
    knobs: np.ndarray,
    device: torch.device,
    chunk_size: int = 4096,
) -> float:
    """ESR for one preset on the validation split.

    Delegates to each model's own predict() method, which handles the
    correct context strategy — LSTM carries hidden state between chunks,
    TCN prefixes each chunk with receptive_field context samples.

    Args:
        model: Trained model.
        dry_val: Dry validation signal, shape (N,), float32.
        wet_val: Wet ground truth, same shape.
        knobs: Normalised knob vector, shape (n_knobs,), float32.
        device: Torch device (unused — model uses its own parameter device).
        chunk_size: Unused — each model selects its own optimal chunk size.

    Returns:
        ESR in [0, ∞).
    """
    pred = model.predict(dry_val, knobs)
    return compute_esr(wet_val, pred)


# ── Training ──────────────────────────────────────────────────────────────── #

def train(
    n_epochs: int = 100,
    hidden_size: int = 32,
    chunk_size: int = 4096,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    device_str: str = "auto",
    exclude: frozenset[str] = DEFAULT_EXCLUDE,
    only: list[str] | None = None,
    model_type: str = "lstm",
    checkpoint_dir: Path = ROOT / "checkpoints" / "ge7_conditioned",
) -> None:
    """Train a conditioned model on all usable GE-7 preset captures.

    Args:
        n_epochs: Training epochs.
        hidden_size: LSTM hidden units (lstm) or conv channels (tcn).
        chunk_size: TBPTT chunk length in samples at 48 kHz.
        batch_size: Items per gradient step.
        lr: Initial AdamW learning rate (cosine-annealed to lr/100).
        val_frac: Fraction of each capture held out as validation.
        device_str: 'auto', 'cuda', or 'cpu'.
        exclude: Preset names to skip.
        only: If set, train on only these presets.
        model_type: 'lstm' or 'tcn'.
        checkpoint_dir: Root directory for checkpoint subdirectories.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # ── Presets ───────────────────────────────────────────────────────────── #
    print("\nLoading preset metadata…")
    presets = load_preset_meta(exclude=exclude, only=only)
    if not presets:
        raise RuntimeError("No usable presets found after filtering.")
    for p in presets:
        db_vals = (p["knobs"] * SLIDER_MAX).round(1).tolist()
        corr = p["fader_correction"]
        corr_str = f"  fader_corr={corr:.3f}x" if abs(corr - 1.0) > 0.01 else ""
        print(f"  {p['name']:20s}  {db_vals}{corr_str}")

    # ── Dry signal ────────────────────────────────────────────────────────── #
    print(f"\nLoading {DRY_WAV.name}…")
    dry, sr = sf.read(str(DRY_WAV), dtype="float32", always_2d=False)
    total_n = len(dry)
    print(f"  {total_n/sr:.1f}s @ {sr} Hz  ({total_n:,} samples)")

    val_n = (int(total_n * val_frac) // chunk_size) * chunk_size
    train_n = total_n - val_n
    print(f"  train={train_n/sr:.1f}s  val={val_n/sr:.1f}s")

    dry_train = dry[:train_n]
    dry_val = dry[train_n : train_n + val_n]

    # ── Wet signals ───────────────────────────────────────────────────────── #
    print("\nLoading wet WAVs…")
    wet_train_arrays: list[np.ndarray] = []
    wet_val_arrays: list[np.ndarray] = []
    for p in presets:
        wet, _ = sf.read(str(p["wet_wav"]), dtype="float32", always_2d=False)
        wet = wet[:total_n]
        if p["fader_correction"] != 1.0:
            wet = (wet * p["fader_correction"]).astype(np.float32)
        wet_train_arrays.append(wet[:train_n].copy())
        wet_val_arrays.append(wet[train_n : train_n + val_n].copy())
        print(f"  {p['name']:20s}  {p['wet_wav'].name}")

    # ── Dataset ───────────────────────────────────────────────────────────── #
    knobs_list = [p["knobs"] for p in presets]
    dataset = ConditionedChunkDataset(
        dry_train, wet_train_arrays, knobs_list, chunk_size=chunk_size
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    chunks_per_preset = train_n // chunk_size
    print(
        f"\nDataset: {len(presets)} presets × {chunks_per_preset:,} chunks"
        f" = {len(dataset):,} items  (chunk={chunk_size}, batch={batch_size})"
    )

    # ── Model ─────────────────────────────────────────────────────────────── #
    if model_type == "tcn":
        model: nn.Module = ConditionedTCNModel(
            n_knobs=len(SLIDER_KEYS),
            channels=hidden_size,
            n_blocks=10,
            film_hidden=64,
        ).to(device)
    else:
        model = ConditionedLSTMModel(
            n_knobs=len(SLIDER_KEYS),
            knob_embedding_dim=16,
            hidden_size=hidden_size,
            num_layers=2,
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.name}  params={n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────── #
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_epochs, eta_min=lr * 0.01
    )
    criterion = CombinedLoss(alpha=0.75, beta=0.25, preemphasis_coef=0.85)

    # ── Checkpoint dir ────────────────────────────────────────────────────── #
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    ckpt_dir = checkpoint_dir / ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    best_val_esr = float("inf")

    print(f"Checkpoints → {ckpt_dir}\n")
    print(f"{'Epoch':>6}  {'Loss':>9}  {'Val ESR':>9}  {'LR':>8}  {'s/ep':>5}")
    print("─" * 46)

    # ── Loop ──────────────────────────────────────────────────────────────── #
    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for dry_batch, wet_batch, knobs_batch in loader:
            dry_batch = dry_batch.to(device)       # (B, T, 1)
            wet_batch = wet_batch.to(device)       # (B, T, 1)
            knobs_batch = knobs_batch.to(device)   # (B, 8)

            optimiser.zero_grad()
            # hidden=None: state is zeroed each batch; no carry across presets
            pred, _ = model(dry_batch, knobs_batch, hidden=None)
            loss = criterion(pred, wet_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Val ESR: mean across all presets (sequential inference per preset)
        val_esrs = [
            eval_preset_esr(model, dry_val, wv, kn, device, chunk_size)
            for wv, kn in zip(wet_val_arrays, knobs_list)
        ]
        mean_val_esr = float(np.mean(val_esrs))
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.perf_counter() - t0

        print(
            f"{epoch:>6d}  {epoch_loss/n_batches:>9.5f}  {mean_val_esr:>9.5f}"
            f"  {lr_now:>8.6f}  {elapsed:>5.1f}s"
        )

        state = {
            "epoch": epoch,
            "val_esr": mean_val_esr,
            "model_state": model.state_dict(),
            "optimiser_state": optimiser.state_dict(),
            "hidden_size": hidden_size,
            "n_knobs": len(SLIDER_KEYS),
            "slider_keys": SLIDER_KEYS,
            "slider_max": SLIDER_MAX,
            "presets": [p["name"] for p in presets],
        }
        torch.save(state, last_path)
        if mean_val_esr < best_val_esr:
            best_val_esr = mean_val_esr
            torch.save(state, best_path)

    # ── Per-preset breakdown ──────────────────────────────────────────────── #
    print(f"\nPer-preset val ESR (best checkpoint, epoch {torch.load(best_path)['epoch']}):")
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    print(f"  {'Preset':20s}  {'ESR':>9}")
    print("  " + "─" * 34)
    for p, wv, kn in zip(presets, wet_val_arrays, knobs_list):
        esr = eval_preset_esr(model, dry_val, wv, kn, device, chunk_size)
        print(f"  {p['name']:20s}  {esr:9.5f}")
    print(f"\nBest checkpoint : {best_path}")
    print(f"Best val ESR    : {best_val_esr:.5f}")


# ── CLI ───────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a conditioned LSTM on all GE-7 preset captures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs.")
    parser.add_argument("--hidden", type=int, default=32,
                        help="LSTM hidden units per layer (2 layers).")
    parser.add_argument("--chunk-size", type=int, default=4096,
                        help="TBPTT chunk length in samples.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Items per gradient step.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial AdamW learning rate.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model", default="lstm", choices=["lstm", "tcn"],
                        help="Model architecture: lstm or tcn.")
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                        metavar="NAME",
                        help="Preset names to exclude from training.")
    parser.add_argument("--presets", nargs="*", default=None,
                        metavar="NAME",
                        help="Train on only these presets (default: all usable).")
    args = parser.parse_args()

    train(
        n_epochs=args.epochs,
        hidden_size=args.hidden,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        exclude=frozenset(args.exclude),
        only=args.presets,
        model_type=args.model,
    )


if __name__ == "__main__":
    main()
