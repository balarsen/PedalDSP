"""Three-stage KlonSim parameter recovery fitter."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import soundfile as sf
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from pedal_model.ode.identifiability import (
    compute_transfer_function,
    select_amplitude_sections,
    tanh_linearity_error,
)
from pedal_model.ode.losses import (
    composite_loss_stage3,
    log_magnitude_loss,
    make_frequency_weights,
    nl_smoothness_loss,
    time_domain_mse,
)
from pedal_model.ode.wh_model import WHODEModel


# ── Ground-truth values for diagnostic reporting ──────────────────────────────

_GT: dict[str, float] = {
    "fc1_hz": 80.0,
    "fc2_hz": 350.0,
    "shelf_freq_hz": 3000.0,
    "shelf_gain_db": 2.0,
    "drive": 4.0,
    "output_level": 0.7,
}

def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


# Stage boundary sentinel printed before each stage transition.
_STAGE_BOUNDARY = "=" * 72 + "\n  STAGE_BOUNDARY\n" + "=" * 72


def _pretrain_nl_to_tanh(
    nl: "Any",
    device: str,
    n_steps: int = 300,
    lr: float = 1e-3,
) -> None:
    """Pre-train an NL model to approximate tanh before Stage 2.

    Ensures the NL starts nonlinear so drive has a meaningful gradient from step 0.
    A purely linear NL makes ∂(G/drive · NL(drive·x))/∂drive ≈ 0, stalling drive.
    """
    opt = torch.optim.Adam(nl.parameters(), lr=lr)
    x = torch.linspace(-3.0, 3.0, 1000, device=device)
    y_target = torch.tanh(x).detach()
    for _ in range(n_steps):
        opt.zero_grad()
        time_domain_mse(nl(x), y_target).backward()
        opt.step()


class KlonSimFitter:
    """Three-stage physics-informed fitter for KlonSim parameter recovery.

    Stage 1 — Linear regime: fit filter frequencies + combined gain via
        frequency-domain loss on the log sine sweep at −24 dBFS.
    Stage 2 — Nonlinear regime: fit drive, output_level, and NL shape via
        direct NL input/output mapping using stepped-tone sections at −3 dBFS.
    Stage 3 — Joint refinement: end-to-end time+spectral loss on full signal
        with physics penalties.

    Args:
        model: WHODEModel to fit in place.
        dry: Full dry audio array, shape (N,), float32.
        wet: Full wet audio array, shape (N,).
        manifest_sections: List of section dicts from the JSON manifest.
        sr: Sample rate in Hz.
        results_dir: Directory to save plots and JSON parameter files.
        device: Torch device for training.
        verbose: Print parameter snapshots and loss values.
        debug: Print extra diagnostic info (tabbed in) without extra forward passes.
    """

    # Stage 3 section exclusions (held out for generalization testing)
    _STAGE3_EXCLUDE_LABELS: frozenset[str] = frozenset({
        "val_pink_l-18",
        "val_pink_l-6",
        "val_impulse_train",
    })

    def __init__(
        self,
        model: WHODEModel,
        dry: np.ndarray,
        wet: np.ndarray,
        manifest_sections: list[dict],
        sr: int,
        results_dir: Path,
        device: torch.device | str = "cpu",
        verbose: bool = True,
        debug: bool = False,
    ) -> None:
        self.model = model
        self.dry = dry.astype(np.float32)
        self.wet = wet.astype(np.float32)
        self.sections = manifest_sections
        self.sr = sr
        self.dt = 1.0 / sr
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)
        self.device = torch.device(device)
        self.verbose = verbose
        self.debug = debug
        self.model.to(self.device)

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(f"    {msg}")

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        stage1_steps: int = 2000,
        stage2_steps: int = 3000,
        stage3_steps: int = 5000,
        stage3_window: int = 4096,
        stall_patience: int = 200,
    ) -> dict[str, Any]:
        """Execute all three stages and return the final parameter recovery report.

        Args:
            stage1_steps: Optimisation steps for Stage 1.
            stage2_steps: Optimisation steps for Stage 2.
            stage3_steps: Optimisation steps for Stage 3.
            stage3_window: Audio window length (samples) for Stage 3 mini-batches.
            stall_patience: Stop a stage early if loss doesn't improve for this many steps.

        Returns:
            Dict with keys 'stage1_params', 'stage2_params', 'final_params',
            'esr', 'nl_r2', 'generalization_esr', 'pass_fail'.
        """
        self._print_startup_diagnostics()

        self._run_stage1(n_steps=stage1_steps, stall_patience=stall_patience)
        print(_STAGE_BOUNDARY)

        if getattr(self.model, "is_linear", False):
            print(f"\n{_ts()} ── Stage 2 skipped — model.is_linear=True (no NL to identify) ──")
        else:
            self._run_stage2(n_steps=stage2_steps, stall_patience=stall_patience)
        print(_STAGE_BOUNDARY)

        self._run_stage3(
            n_steps=stage3_steps,
            window=stage3_window,
            stall_patience=stall_patience,
        )
        print(_STAGE_BOUNDARY)

        return self._build_report()

    # ── Startup diagnostics ───────────────────────────────────────────────────

    def _print_startup_diagnostics(self) -> None:
        """Log linearisation error at Stage 1 data amplitude."""
        true_drive = _GT["drive"]
        # Only the val_sweep_l-24 section qualifies (amplitude < -20 dBFS)
        stage1_secs = [s for s in self.sections if float(s.get("amplitude_dbfs", 0)) <= -20.0]
        if stage1_secs:
            amp = float(stage1_secs[0]["amplitude_dbfs"])
            err = tanh_linearity_error(true_drive, amp)
            print(
                f"{_ts()} [startup] Stage 1 section: '{stage1_secs[0]['label']}' "
                f"at {amp:.0f} dBFS\n"
                f"  At drive={true_drive}, tanh input amplitude ≈ "
                f"{true_drive * 10**(amp/20):.3f}, "
                f"nonlinearity error ≈ {err*100:.1f}%  (< 5% → linear regime ✓)"
            )
        print(f"{_ts()} [startup] Signal: {len(self.dry)} samples at {self.sr} Hz "
              f"({len(self.dry)/self.sr:.1f} s)")

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def _run_stage1(self, n_steps: int, stall_patience: int) -> None:
        """Stage 1: frequency-domain linear-regime fitting."""
        print(f"\n{_ts()} ── Stage 1: Linear-regime transfer function fitting ──")

        # Select only sections at or below -20 dBFS (val_sweep_l-24)
        dry_s1, wet_s1 = select_amplitude_sections(
            self.dry, self.wet, self.sections,
            amplitude_threshold_dbfs=-20.0,
            above=False,
        )
        if self.verbose:
            print(f"  Stage 1 data: {len(dry_s1)/self.sr:.1f} s")

        # Estimate measured transfer function
        n_fft_s1, hop_s1 = 8192, 4096
        freqs_hz, H_meas = compute_transfer_function(
            dry_s1, wet_s1, self.sr, n_fft=n_fft_s1, hop=hop_s1
        )
        freqs_rad = torch.tensor(
            2.0 * math.pi * freqs_hz, dtype=torch.float32, device=self.device
        )
        log_H_meas = torch.tensor(
            np.log(np.abs(H_meas) + 1e-6), dtype=torch.float32, device=self.device
        )
        freq_weights = make_frequency_weights(
            torch.tensor(freqs_hz, dtype=torch.float32, device=self.device),
            self.sr,
        )
        n_frames = max((len(dry_s1) - n_fft_s1) // hop_s1, 0)
        passband_H = float(np.abs(H_meas[(freqs_hz > 500) & (freqs_hz < 8000)]).mean())
        # passband_H includes the shelf boost; true G (without shelf) is lower
        self._dbg(f"TF estimate: {n_frames} frames, passband |H| = {passband_H:.3f} "
                  f"({20*np.log10(passband_H+1e-9):.1f} dB, includes shelf)")

        # Free in Stage 1: filter cutoffs + combined gain (drive×level).
        # NL is frozen — shelf is also frozen for WHODEModel (its ±2 dB effect is
        # too small to identify reliably from the linear-regime sweep and causes the
        # optimizer to absorb the bulk of the passband gain into a spurious shelf
        # boost instead of G). Stage 3 refines the shelf. For LinearBiquadModel,
        # freeze_nl() is a no-op and stage1_parameters() returns all biquad params.
        self.model.freeze_nl()
        stage1_params = self.model.stage1_parameters()
        opt = torch.optim.Adam(stage1_params, lr=1e-3)

        best_loss = float("inf")
        stall = 0
        for step in range(n_steps):
            opt.zero_grad()
            log_H_model = self.model.combined_log_magnitude(freqs_rad, self.dt)
            loss = log_magnitude_loss(log_H_model, log_H_meas, freq_weights)
            loss.backward()
            opt.step()

            l_val = float(loss.detach())
            if l_val < best_loss - 1e-8:
                best_loss = l_val
                stall = 0
            else:
                stall += 1

            if self.verbose and (step % 200 == 0 or step == n_steps - 1):
                self.model.log_params(f"S1 step {step:4d}  loss={l_val:.5f}")

            if stall >= stall_patience:
                print(f"{_ts()}   [Stage 1] Early stop at step {step}: no improvement for {stall_patience} steps.")
                break

        g_s1 = float(torch.exp(self.model.log_drive + self.model.log_output_level).item())
        print(f"{_ts()}   Stage 1 final loss: {best_loss:.6f} "
              f"({'PASS' if best_loss < 0.01 else 'FAIL (> 0.01)'})")
        self._dbg(f"G = drive×level = {g_s1:.4f}  ({20*math.log10(g_s1+1e-9):.1f} dB)")
        self._save_stage_params("stage1_params.json")
        self._plot_stage1_tf(freqs_hz, H_meas)

        # Unfreeze NL for Stage 2 (no-op for LinearBiquadModel)
        self.model.unfreeze_nl()

    # ── Stage 2 ───────────────────────────────────────────────────────────────

    def _run_stage2(self, n_steps: int, stall_patience: int) -> None:
        """Stage 2: nonlinearity + drive identification via forward model (FFT convolution).

        The previous inverse-filter approach inverted HPF2 (a highpass), which introduced
        a large phase shift (~77–142°) at the tone frequencies. A static NL cannot match a
        phase-shifted target, so the MSE minimum was NL→0. This forward-model approach
        applies the frozen post-filter in the forward direction via FFT convolution and
        compares wet_model = (G_s1/drive) * HPF2(shelf(NL(drive·x_pre))) with measured wet.
        Both sides go through the same causal filter so phases cancel.
        """
        print(f"\n{_ts()} ── Stage 2: Nonlinearity and drive identification ──")

        # Select only stepped sine tones at -3 dBFS
        dry_s2, wet_s2 = select_amplitude_sections(
            self.dry, self.wet, self.sections,
            amplitude_threshold_dbfs=-4.0,
            above=True,
            section_types=["stepped_sine_tone"],
        )
        if self.verbose:
            print(f"  Stage 2 data: {len(dry_s2)/self.sr:.1f} s "
                  f"({len(dry_s2)} samples, stepped_sine_tone at ≥ -4 dBFS)")

        # Freeze filter parameters; only drive and NL are free.
        self.model.freeze_filters()

        dry_t = torch.tensor(dry_s2, device=self.device)
        wet_t = torch.tensor(wet_s2, device=self.device)

        with torch.no_grad():
            x_pre = self.model.hpf1(dry_t, self.dt)  # (N,) pre-filtered dry, no drive

        # Stage 1 product constraint: G = drive × output_level is conserved.
        # As drive changes during Stage 2, output_level is adjusted to maintain G_s1,
        # so the overall gain stays consistent with Stage 1's TF measurement.
        G_s1 = float(torch.exp(self.model.log_drive + self.model.log_output_level).item())

        rms_x_pre = float(x_pre.pow(2).mean().sqrt())
        rms_wet_s2 = float(wet_t.pow(2).mean().sqrt())
        self._dbg(f"x_pre RMS = {rms_x_pre:.4f}  wet RMS = {rms_wet_s2:.4f}  G_s1 = {G_s1:.4f}")

        # Precompute frozen post-filter (HPF2 × shelf) frequency response for FFT convolution.
        # 50 ms captures the full impulse response (HPF2 at 332 Hz decays in ~3 ms).
        ir_len = min(int(self.sr * 0.05), 4800)
        ir_in = torch.zeros(ir_len, device=self.device)
        ir_in[0] = 1.0
        with torch.no_grad():
            ir_post_t = self.model.hpf2(ir_in, self.dt)
            ir_post_t = self.model.shelf(ir_post_t, self.dt)

        N = len(dry_s2)
        n_fft = 1 << (N + ir_len - 1).bit_length()  # next power-of-2 ≥ N + ir_len − 1
        H_post_fft = torch.fft.rfft(ir_post_t, n=n_fft).detach()  # frozen complex spectrum

        # Pre-warm NL to approximate tanh so drive has a meaningful gradient from step 0.
        # For a linear NL (NL≈c·x), ∂[(G/drive)·NL(drive·x)]/∂drive = (G/drive)·c·0 = 0 —
        # the drive gradient cancels algebraically and drive cannot move.
        self._dbg("Pre-warming NL to tanh shape (300 steps)…")
        _pretrain_nl_to_tanh(self.model.nl, device=self.device, n_steps=300, lr=1e-3)

        # Stage 2 optimizer: drive + NL. output_level is NOT a free parameter here —
        # it is derived after each step via G_s1/drive to maintain the Stage 1 constraint.
        stage2_params = [self.model.log_drive] + list(self.model.nl.parameters())
        opt = torch.optim.Adam(stage2_params, lr=5e-4)

        lambda_smooth_start, lambda_smooth_end = 1e-4, 1e-6
        best_loss = float("inf")
        stall = 0

        for step in range(n_steps):
            lambda_smooth = lambda_smooth_start * (
                lambda_smooth_end / lambda_smooth_start
            ) ** (step / max(n_steps - 1, 1))

            opt.zero_grad()
            drive_t = torch.exp(self.model.log_drive)
            # Enforce G_s1 = drive × output_level: output_level tracks drive inversely
            output_level_t = G_s1 / drive_t

            z = x_pre.detach() * drive_t
            nl_out = self.model.nl(z)  # (N,) with gradient

            # Forward model: apply frozen post-filter via FFT convolution.
            # gradient flows through nl_out; H_post_fft is detached (frozen filter).
            nl_fft = torch.fft.rfft(nl_out, n=n_fft)
            nl_post = torch.fft.irfft(nl_fft * H_post_fft, n=n_fft)[:N]
            wet_model = nl_post * output_level_t

            l_mse = time_domain_mse(wet_model, wet_t.detach())
            l_sm = nl_smoothness_loss(self.model.nl)
            loss = l_mse + lambda_smooth * l_sm
            loss.backward()
            opt.step()

            # Keep log_output_level in sync with G_s1/drive so log_params is accurate
            with torch.no_grad():
                drive_val = float(torch.exp(self.model.log_drive))
                new_level = float(np.clip(G_s1 / max(drive_val, 1e-3), 0.01, 20.0))
                self.model.log_output_level.copy_(torch.tensor(math.log(new_level)))

            l_val = float(l_mse.detach())
            if l_val < best_loss - 1e-8:
                best_loss = l_val
                stall = 0
            else:
                stall += 1

            if self.verbose and (step % 300 == 0 or step == n_steps - 1):
                self.model.log_params(f"S2 step {step:4d}  wet-MSE={l_val:.5f}")

            if stall >= stall_patience:
                print(f"{_ts()}   [Stage 2] Early stop at step {step}: no improvement for {stall_patience} steps.")
                break

        with torch.no_grad():
            drive_final = float(torch.exp(self.model.log_drive).item())
            output_level_final = float(np.clip(G_s1 / max(drive_final, 1e-3), 0.01, 20.0))
            self.model.log_output_level.copy_(torch.tensor(math.log(output_level_final)))

        self._dbg(f"drive={drive_final:.4f}  output_level={output_level_final:.4f}  "
                  f"G={drive_final * output_level_final:.4f}")

        print(f"{_ts()}   Stage 2 final wet-MSE: {best_loss:.6f} "
              f"({'PASS' if best_loss < 0.01 else 'FAIL (> 0.01)'})")
        self.model.log_params("S2 final")
        self._save_stage_params("stage2_params.json")
        self._plot_stage2_nl()

        # Unfreeze for Stage 3
        self.model.unfreeze_all()

    # ── Stage 3 ───────────────────────────────────────────────────────────────

    # Warm-up samples prepended to each Stage 3 mini-batch window so that the
    # causal biquad filter state (time constant ~12 ms for HPF1@80 Hz) is
    # initialised before the loss window begins. Without warm-up, every window
    # starts with zero filter state → ~1200-sample transient that the spectral
    # FFT loss amplifies into large opposing gradients.
    _STAGE3_WARMUP: int = 2048  # ~21 ms at 96 kHz (> 1 HPF1 time constant)

    def _run_stage3(
        self, n_steps: int, window: int, stall_patience: int
    ) -> None:
        """Stage 3: joint refinement on full signal (excluding held-out sections)."""
        print(f"\n{_ts()} ── Stage 3: Joint refinement (full signal, biquad forward pass) ──")

        # Exclude held-out sections from Stage 3 training
        stage3_sections = [
            s for s in self.sections
            if s.get("label", "") not in self._STAGE3_EXCLUDE_LABELS
        ]
        dry_s3, wet_s3 = self._concat_sections(stage3_sections)
        if self.verbose:
            print(f"  Stage 3 data: {len(dry_s3)/self.sr:.1f} s "
                  f"({len(dry_s3)} samples, {len(stage3_sections)} sections)")

        dry_t = torch.tensor(dry_s3, device=self.device)
        wet_t = torch.tensor(wet_s3, device=self.device)
        N = len(dry_s3)

        # Adam with cosine annealing
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-6)

        param_history: list[dict[str, float]] = []
        loss_history: list[tuple[int, float]] = []
        best_loss = float("inf")
        stall = 0
        warm_up = self._STAGE3_WARMUP

        for step in range(n_steps):
            # Random loss window with warm-up prefix to initialise filter state.
            # Process [warm_start : start+window] through the model, then crop
            # the output to [start : start+window] before computing the loss.
            start = np.random.randint(0, max(N - window, 1))
            warm_start = max(0, start - warm_up)
            actual_warm = start - warm_start  # may be < warm_up near signal start

            d_ext = dry_t[warm_start : start + window]
            w_win = wet_t[start : start + window]

            opt.zero_grad()
            y_ext = self.model(d_ext, self.dt)
            y_hat = y_ext[actual_warm:]  # discard warm-up output, keep loss window

            fc1 = torch.exp(self.model.hpf1.log_omega) / (2.0 * math.pi)
            fc2 = torch.exp(self.model.hpf2.log_omega) / (2.0 * math.pi)
            drive_t = torch.exp(self.model.log_drive)
            level_t = torch.exp(self.model.log_output_level)

            # beta=0: spectral FFT loss removed — it amplifies warm-up transient
            # residuals and fights the time-domain gradient. Time-domain MSE +
            # physics penalty is sufficient for joint refinement.
            loss, components = composite_loss_stage3(
                y_hat, w_win,
                self.model.nl,
                fc1, fc2, drive_t, level_t,
                self.sr,
                beta=0.0,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()

            l_val = float(loss.detach())
            loss_history.append((step, l_val))

            if l_val < best_loss - 1e-8:
                best_loss = l_val
                stall = 0
            else:
                stall += 1

            if step % 50 == 0:
                param_history.append({**self.model.get_params_dict(), "step": step})

            if self.verbose and (step % 500 == 0 or step == n_steps - 1):
                self.model.log_params(
                    f"S3 step {step:4d}  "
                    f"L={l_val:.5f} time={components['time']:.5f}"
                )
                self._dbg(
                    f"  smooth={components.get('smooth', 0):.5f}  "
                    f"physics={components.get('physics', 0):.5f}  "
                    f"lr={opt.param_groups[0]['lr']:.2e}"
                )

            if stall >= stall_patience:
                # Capture final state before breaking so the plot has an endpoint.
                param_history.append({**self.model.get_params_dict(), "step": step})
                print(f"{_ts()}   [Stage 3] Early stop at step {step}: no improvement for {stall_patience} steps.")
                break

        print(f"{_ts()}   Stage 3 best loss: {best_loss:.6f}")
        self.model.log_params("S3 final")
        self._save_stage_params("stage3_params.json")
        pt_path = self.results_dir / "model_final.pt"
        torch.save(self.model.state_dict(), pt_path)
        if self.verbose:
            print(f"  Saved: {pt_path}")
        self._plot_stage3_convergence(param_history, loss_history)
        self._plot_waveform_overlay()

    # ── Diagnostics and report ─────────────────────────────────────────────────

    def _build_report(self) -> dict[str, Any]:
        """Compute ESR, NL R², generalization ESR, and pass/fail table."""
        print("\n── Parameter Recovery Report ──")

        # Full validation ESR
        with torch.no_grad():
            dry_t = torch.tensor(self.dry, device=self.device)
            y_hat = self.model(dry_t, self.dt).cpu().numpy()
        N = min(len(y_hat), len(self.wet))
        esr = float(
            np.sum((self.wet[:N] - y_hat[:N]) ** 2)
            / (np.sum(self.wet[:N] ** 2) + 1e-8)
        )

        # nl_r2 only meaningful for WHODEModel (LinearBiquadModel has no NL)
        nl_obj = getattr(self.model, "nl", None)
        nl_r2 = float(nl_obj.r2_vs_tanh()) if nl_obj is not None else 1.0

        # Save audio comparison files (wet_model, residual, stereo A/B)
        self._save_audio_comparison(y_hat[:N])

        # Generalization ESR on held-out sections
        gen_esr: dict[str, float] = {}
        for label in ("val_pink_l-18", "val_pink_l-6", "val_impulse_train"):
            secs = [s for s in self.sections if s.get("label") == label]
            if not secs:
                continue
            d_heldout, w_heldout = self._concat_sections(secs)
            with torch.no_grad():
                d_t = torch.tensor(d_heldout, device=self.device)
                y_h = self.model(d_t, self.dt).cpu().numpy()
            n = min(len(y_h), len(w_heldout))
            gen_esr[label] = float(
                np.sum((w_heldout[:n] - y_h[:n]) ** 2)
                / (np.sum(w_heldout[:n] ** 2) + 1e-8)
            )

        # IMD generalization (two-tone sections)
        two_tone_secs = [s for s in self.sections if s.get("type") == "two_tone"]
        if two_tone_secs:
            d_tt, w_tt = self._concat_sections(two_tone_secs)
            with torch.no_grad():
                d_t = torch.tensor(d_tt, device=self.device)
                y_h = self.model(d_t, self.dt).cpu().numpy()
            n = min(len(y_h), len(w_tt))
            gen_esr["val_two_tone_all"] = float(
                np.sum((w_tt[:n] - y_h[:n]) ** 2)
                / (np.sum(w_tt[:n] ** 2) + 1e-8)
            )

        params = self.model.get_params_dict()
        tolerances = {
            "fc1_hz": (80.0, 8.0),
            "fc2_hz": (350.0, 35.0),
            "shelf_freq_hz": (3000.0, 300.0),
            "shelf_gain_db": (2.0, 0.3),
            "drive": (4.0, 0.6),
            "output_level": (0.7, 0.1),
        }
        rows = []
        for param_name, (gt, tol) in tolerances.items():
            if param_name not in params:
                continue  # LinearBiquadModel has different param keys
            val = params[param_name]
            err = abs(val - gt)
            err_pct = err / abs(gt) * 100.0
            passed = err <= tol
            rows.append({
                "parameter": param_name,
                "ground_truth": gt,
                "recovered": round(val, 4),
                "error": round(err, 4),
                "error_pct": round(err_pct, 2),
                "within_tolerance": "PASS" if passed else "FAIL",
            })
        rows.append({
            "parameter": "nl_r2",
            "ground_truth": 1.0,
            "recovered": round(nl_r2, 6),
            "error": round(1.0 - nl_r2, 6),
            "error_pct": round((1.0 - nl_r2) * 100, 4),
            "within_tolerance": "PASS" if nl_r2 > 0.999 else "FAIL",
        })
        rows.append({
            "parameter": "esr",
            "ground_truth": 0.0,
            "recovered": round(esr, 6),
            "error": round(esr, 6),
            "error_pct": "-",
            "within_tolerance": "PASS" if esr < 0.01 else "FAIL",
        })
        for label, g_esr in gen_esr.items():
            rows.append({
                "parameter": f"generalization_esr_{label}",
                "ground_truth": "-",
                "recovered": round(g_esr, 6),
                "error": "-",
                "error_pct": "-",
                "within_tolerance": "PASS" if g_esr < 0.05 else "INFO",
            })

        # Print table
        header = f"{'parameter':<30} {'gt':>10} {'recovered':>12} {'err%':>8} {'status':>6}"
        print(header)
        print("-" * len(header))
        for r in rows:
            gt_str = f"{r['ground_truth']:.4g}" if isinstance(r['ground_truth'], float) else str(r['ground_truth'])
            ep_str = f"{r['error_pct']:.2f}" if isinstance(r['error_pct'], float) else str(r['error_pct'])
            print(f"  {r['parameter']:<28} {gt_str:>10} {r['recovered']:>12.6g} {ep_str:>8} {r['within_tolerance']:>6}")

        # Save CSV
        import csv
        csv_path = self.results_dir / "param_recovery.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Saved: {csv_path}")

        # IMD spectrum plot (uses val_two_tone_f196_f294 if available)
        imd_sec = next(
            (s for s in self.sections if s.get("label") == "val_two_tone_f196_f294"), None
        )
        if imd_sec:
            start, end = int(imd_sec["start_sample"]), int(imd_sec["end_sample"])
            self._plot_imd_spectrum(
                self.dry[start:end], self.wet[start:end],
                f1_hz=196.0, f2_hz=294.0,
            )

        return {
            "final_params": params,
            "esr": esr,
            "nl_r2": nl_r2,
            "generalization_esr": gen_esr,
            "pass_fail_rows": rows,
        }

    # ── Audio export ──────────────────────────────────────────────────────────

    def _save_audio_comparison(self, y_hat: np.ndarray) -> None:
        """Save wet_model, residual, and stereo A/B WAV files for audible comparison.

        Files written to <results_dir>/audio/:
            dry.wav                    — original dry input (normalised copy)
            wet_klonsim.wav            — target wet signal
            wet_model.wav              — model output ŷ
            residual_boosted.wav       — (wet − ŷ) × boost for null-test listening
            stereo_AB.wav              — stereo: L = KlonSim wet, R = model ŷ

        Args:
            y_hat: Model output aligned to self.wet, shape (N,), float32.
        """
        audio_dir = self.results_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        N = len(y_hat)
        wet = self.wet[:N].astype(np.float32)
        dry = self.dry[:N].astype(np.float32)
        y_hat = y_hat.astype(np.float32)

        residual = wet - y_hat
        # Boost residual so a -40 dB null test (-40 dB ESR ≈ 0.0001) is audible
        res_peak = float(np.max(np.abs(residual)) + 1e-9)
        # Target -6 dBFS peak for the boosted residual
        boost = min(10.0 ** (-6.0 / 20.0) / res_peak, 50.0)
        residual_boosted = np.clip(residual * boost, -1.0, 1.0)

        stereo_AB = np.stack([wet, y_hat], axis=1)  # L=KlonSim, R=model

        sf.write(str(audio_dir / "dry.wav"), dry, self.sr, subtype="FLOAT")
        sf.write(str(audio_dir / "wet_klonsim.wav"), wet, self.sr, subtype="FLOAT")
        sf.write(str(audio_dir / "wet_model.wav"), y_hat, self.sr, subtype="FLOAT")
        sf.write(str(audio_dir / "residual_boosted.wav"), residual_boosted, self.sr, subtype="FLOAT")
        sf.write(str(audio_dir / "stereo_AB.wav"), stereo_AB, self.sr, subtype="FLOAT")

        esr = float(np.sum(residual ** 2) / (np.sum(wet ** 2) + 1e-8))
        print(f"  Audio saved to: {audio_dir}/")
        print(f"    dry.wav  wet_klonsim.wav  wet_model.wav")
        print(f"    residual_boosted.wav  (×{boost:.1f} gain,  ESR={esr:.5f})")
        print(f"    stereo_AB.wav  (L=KlonSim  R=model — open in DAW for A/B)")

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _plot_stage1_tf(
        self, freqs_hz: np.ndarray, H_meas: np.ndarray
    ) -> None:
        """Plot Stage 1: measured vs fitted transfer function magnitude."""
        freqs_rad = torch.tensor(
            2.0 * math.pi * freqs_hz, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            log_H_model = self.model.combined_log_magnitude(freqs_rad, self.dt).cpu().numpy()

        log_H_meas = np.log(np.abs(H_meas) + 1e-6)
        residual = log_H_model - log_H_meas

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].semilogx(freqs_hz, 20 * log_H_meas / math.log(10), label="Measured", alpha=0.8)
        axes[0].semilogx(freqs_hz, 20 * log_H_model / math.log(10), "--", label="Fitted", alpha=0.8)
        axes[0].set_ylabel("Magnitude (dB)")
        axes[0].set_title("Stage 1 — Transfer Function Fit")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(20, 15_000)  # clipped to freq-weight taper endpoint; spike above 15k is noise
        axes[0].set_ylim(-60, 20)

        axes[1].semilogx(freqs_hz, 20 * residual / math.log(10), color="red", alpha=0.7)
        axes[1].axhline(0, color="gray", linewidth=0.8)
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Residual (dB)")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(20, 15_000)  # clipped to freq-weight taper endpoint
        axes[1].set_ylim(-20, 20)

        plt.tight_layout()
        path = self.results_dir / "plots" / "stage1_tf.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        if self.verbose:
            print(f"  Saved: {path}")

    def _plot_stage2_nl(self) -> None:
        """Plot Stage 2: learned NL(x) vs tanh(drive·x) and tanh(x)."""
        drive = self.model.drive
        x_np = np.linspace(-3.0, 3.0, 400, dtype=np.float32)
        x_t = torch.tensor(x_np, device=self.device)
        with torch.no_grad():
            y_nl = self.model.nl(x_t).cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(x_np, y_nl, label="NL(x) fitted", linewidth=2)
        axes[0].plot(x_np, np.tanh(x_np), "--", label="tanh(x)", alpha=0.7)
        axes[0].plot(
            x_np,
            np.tanh(drive * x_np),
            ":",
            label=f"tanh({drive:.2f}·x)",
            alpha=0.7,
        )
        axes[0].set_xlabel("x (NL input)")
        axes[0].set_ylabel("NL(x)")
        axes[0].set_title(f"Stage 2 — Nonlinearity Shape (drive={drive:.3f})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Derivative (finite differences)
        dx = x_np[1] - x_np[0]
        dydx = np.gradient(y_nl, dx)
        axes[1].plot(x_np, dydx, label="NL′(x)")
        tanh_prime = 1.0 - np.tanh(x_np) ** 2
        axes[1].plot(x_np, tanh_prime, "--", label="tanh′(x)", alpha=0.7)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("NL′(x)")
        axes[1].set_title("Derivative")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.results_dir / "plots" / "stage2_nl.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        if self.verbose:
            print(f"  Saved: {path}")

    def _plot_stage3_convergence(
        self,
        history: list[dict[str, float]],
        loss_history: list[tuple[int, float]],
    ) -> None:
        """Plot Stage 3: loss curve + parameter trajectories vs ground truth.

        Layout: loss curve on top (full width), 6 param traces below in 2×3 grid.
        The loss curve is the primary convergence diagnostic; param traces show
        whether parameters drifted from their Stage 2 starting point.
        """
        if not history and not loss_history:
            return

        param_names = [
            ("fc1_hz", "fc₁ (Hz)", 80.0),
            ("fc2_hz", "fc₂ (Hz)", 350.0),
            ("shelf_freq_hz", "shelf freq (Hz)", 3000.0),
            ("shelf_gain_db", "shelf gain (dB)", 2.0),
            ("drive", "drive", 4.0),
            ("output_level", "output level", 0.7),
        ]

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.45, wspace=0.35)
        ax_loss = fig.add_subplot(gs[0, :])
        param_axes = [
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]),
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1]),
            fig.add_subplot(gs[2, 2]),
        ]

        # ── Loss curve ────────────────────────────────────────────────────────
        if loss_history:
            ls, lv = zip(*loss_history)
            ls_arr = np.array(ls)
            lv_arr = np.array(lv)
            ax_loss.plot(ls_arr, lv_arr, color="steelblue", linewidth=0.7, alpha=0.4, label="per-step")
            # Smoothed overlay (running mean, window = 1% of total or 20, whichever is larger)
            w = max(20, len(lv_arr) // 100)
            if len(lv_arr) >= w:
                kernel = np.ones(w) / w
                smooth = np.convolve(lv_arr, kernel, mode="valid")
                ax_loss.plot(ls_arr[w - 1:], smooth, color="steelblue", linewidth=2.0, label=f"smoothed (w={w})")
            ax_loss.set_xlabel("Step")
            ax_loss.set_ylabel("Loss")
            ax_loss.set_title("Stage 3 — Loss Curve")
            ax_loss.legend(fontsize=8)
            ax_loss.grid(True, alpha=0.3)
            # Use log scale if the loss spans more than one order of magnitude
            if len(lv_arr) > 1:
                finite_vals = lv_arr[np.isfinite(lv_arr) & (lv_arr > 0)]
                if len(finite_vals) > 1 and (np.max(finite_vals) / np.min(finite_vals)) > 10:
                    ax_loss.set_yscale("log")

        # ── Parameter trajectories ────────────────────────────────────────────
        if history:
            steps = [h["step"] for h in history]
            for ax, (key, label, gt) in zip(param_axes, param_names):
                if key not in history[0]:
                    ax.set_visible(False)
                    continue
                vals = [h[key] for h in history]
                ax.plot(steps, vals, linewidth=1.5, color="steelblue")
                ax.axhline(gt, color="red", linestyle="--", alpha=0.7, label=f"GT={gt}")
                # Show percentage error at final step in title
                final_err_pct = abs(vals[-1] - gt) / (abs(gt) + 1e-9) * 100.0
                ax.set_title(f"{label}  (err={final_err_pct:.2f}%)", fontsize=9)
                ax.set_xlabel("Step", fontsize=8)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)

        plt.suptitle("Stage 3 — Convergence", fontsize=13, y=1.01)
        path = self.results_dir / "plots" / "stage3_convergence.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if self.verbose:
            print(f"  Saved: {path}")

    def _plot_waveform_overlay(self) -> None:
        """Plot a 40 ms window of wet vs ŷ vs residual from the highest-drive section."""
        # Use the first -3 dBFS section for the overlay
        secs_hi = [s for s in self.sections if float(s.get("amplitude_dbfs", -99)) >= -4.0
                   and s.get("type") == "stepped_sine_tone"]
        if not secs_hi:
            return

        sec = secs_hi[0]
        start = int(sec["start_sample"])
        n_samples = int(0.04 * self.sr)  # 40 ms
        end = min(start + n_samples, int(sec["end_sample"]))

        d_win = self.dry[start:end]
        w_win = self.wet[start:end]

        with torch.no_grad():
            d_t = torch.tensor(d_win, device=self.device)
            y_hat = self.model(d_t, self.dt).cpu().numpy()

        t_ms = np.arange(len(w_win)) * 1000.0 / self.sr
        n = min(len(y_hat), len(w_win))

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(t_ms[:n], w_win[:n], label="KlonSim wet", alpha=0.85)
        axes[0].plot(t_ms[:n], y_hat[:n], "--", label="Model ŷ", alpha=0.85)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title(f"Waveform Overlay — section: {sec.get('label', '')}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_ms[:n], w_win[:n] - y_hat[:n], color="red", alpha=0.75, label="Residual")
        axes[1].axhline(0, color="gray", linewidth=0.8)
        axes[1].set_xlabel("Time (ms)")
        axes[1].set_ylabel("Residual")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.results_dir / "plots" / "waveform_overlay.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        if self.verbose:
            print(f"  Saved: {path}")

    def _plot_imd_spectrum(
        self,
        dry_section: np.ndarray,
        wet_section: np.ndarray,
        f1_hz: float,
        f2_hz: float,
    ) -> None:
        """Plot IMD spectrum: model vs KlonSim, marking fundamentals + IMD products.

        Args:
            dry_section: Dry audio for the two-tone section.
            wet_section: KlonSim-processed wet audio.
            f1_hz: First fundamental frequency in Hz.
            f2_hz: Second fundamental frequency in Hz.
        """
        with torch.no_grad():
            d_t = torch.tensor(dry_section, device=self.device)
            y_hat = self.model(d_t, self.dt).cpu().numpy()

        n = min(len(y_hat), len(wet_section))
        Y_hat = np.abs(np.fft.rfft(y_hat[:n]))
        Y_wet = np.abs(np.fft.rfft(wet_section[:n]))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sr)

        fig, ax = plt.subplots(figsize=(14, 6))
        db_floor = -120.0
        Y_hat_db = np.maximum(20 * np.log10(Y_hat + 1e-12), db_floor)
        Y_wet_db = np.maximum(20 * np.log10(Y_wet + 1e-12), db_floor)
        ax.plot(freqs, Y_wet_db, label="KlonSim", alpha=0.8, linewidth=0.8)
        ax.plot(freqs, Y_hat_db, "--", label="Model ŷ", alpha=0.8, linewidth=0.8)

        # Mark frequencies of interest
        markers = {
            f"f₁={f1_hz:.0f}Hz": f1_hz,
            f"f₂={f2_hz:.0f}Hz": f2_hz,
            f"2f₁={2*f1_hz:.0f}Hz": 2 * f1_hz,
            f"2f₂={2*f2_hz:.0f}Hz": 2 * f2_hz,
            f"3f₁={3*f1_hz:.0f}Hz": 3 * f1_hz,
            f"f₂-f₁={f2_hz-f1_hz:.0f}Hz": f2_hz - f1_hz,
            f"2f₁-f₂={2*f1_hz-f2_hz:.0f}Hz": 2 * f1_hz - f2_hz,
            f"2f₂-f₁={2*f2_hz-f1_hz:.0f}Hz": 2 * f2_hz - f1_hz,
        }
        for label, freq in markers.items():
            if freq > 0:
                ax.axvline(freq, color="gray", alpha=0.4, linewidth=0.8)
                ax.text(freq, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else -20,
                        label, fontsize=6, rotation=90, ha="right")

        ax.set_xscale("log")
        ax.set_xlim(20, self.sr / 2)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"IMD Spectrum — f₁={f1_hz:.0f} Hz, f₂={f2_hz:.0f} Hz")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = self.results_dir / "plots" / "imd_spectrum.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        if self.verbose:
            print(f"  Saved: {path}")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _concat_sections(self, sections: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate dry/wet audio from a list of manifest sections."""
        dry_parts = [self.dry[int(s["start_sample"]) : int(s["end_sample"])] for s in sections]
        wet_parts = [self.wet[int(s["start_sample"]) : int(s["end_sample"])] for s in sections]
        return np.concatenate(dry_parts), np.concatenate(wet_parts)

    def _save_stage_params(self, filename: str) -> None:
        """Save current model parameters to a JSON file in results_dir."""
        p = self.model.get_params_dict()
        path = self.results_dir / filename
        with open(path, "w") as f:
            json.dump(p, f, indent=2)
        if self.verbose:
            print(f"  Saved: {path}")
