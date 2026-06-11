# PedalODE — KlonSim Parameter Recovery Experiment
## Specification v1.0

---

## Purpose

This experiment validates the PedalODE physics-informed fitting approach on a
known-answer target before applying it to real hardware. KlonSim is a fully
deterministic synthetic Klon-style overdrive pedal whose exact parameters are
known. Success means recovering those parameters — not just fitting the
input-output behavior.

This is the nonlinear equivalent of the Boss GE-7 known-answer linear test in
the PedalODE ladder. If the approach cannot recover KlonSim's six physical
parameters from dry/wet audio alone, it will not work on the real Notaklon.

**Two things are being tested simultaneously:**

1. Can differentiable ODE-based fitting recover physically meaningful parameters
   from a Wiener-Hammerstein system given only dry/wet audio?
2. Does the three-stage fitting strategy correctly exploit signal amplitude
   stratification to produce a well-conditioned optimization problem?

---

## KlonSim Ground Truth

The target system is `pedal_model/pedals/klon_sim.py`, `KlonSim v1.0`.

### Signal chain (exact)

```
dry input
    │
    ▼
[1] Input HPF     2nd-order Butterworth highpass   fc = 80 Hz
    │
    ▼
[2] Drive gain    scalar multiply                  drive = 4.0
    │
    ▼
[3] tanh clip     symmetric soft saturation        (static nonlinearity)
    │
    ▼
[4] Post-clip HPF 2nd-order Butterworth highpass   fc = 350 Hz
    │
    ▼
[5] High shelf    Audio EQ Cookbook biquad         presence_hz = 3000 Hz
                                                   presence_db = +2 dB
    │
    ▼
[6] Output level  scalar multiply                  output_level = 0.7
    │
    ▼
[7] Noise floor   seeded Gaussian                  −70 dBFS  (not fitted)
    │
    ▼
wet output
```

### Topological classification

KlonSim is a **Wiener-Hammerstein (WH) system**: a linear subsystem followed
by a static nonlinearity followed by a second linear subsystem.

```
PreFilter(fc₁) → (×drive) → NL(·) → PostFilter(fc₂, presence) → (×level)
└──────────── Linear ──────┘└─NL─┘└────────────── Linear ──────────────┘
```

The drive gain is absorbed into the pre-filter linear path. The identifiable
parameters are: fc₁, fc₂, presence_hz, presence_db, drive, output_level,
and the nonlinearity shape.

### Known ground-truth values

| Parameter | Ground truth | Stage fitted | Notes |
|-----------|-------------|--------------|-------|
| `fc₁` (input HPF) | 80 Hz | Stage 1 | Filter frequency |
| `fc₂` (post HPF) | 350 Hz | Stage 1 | Filter frequency |
| `presence_hz` | 3000 Hz | Stage 1 | Shelf frequency |
| `presence_db` | +2.0 dB | Stage 1 | Shelf gain |
| `drive` | 4.0 | Stage 2 | Pre-gain scalar |
| `output_level` | 0.7 | Stage 2 | Post-gain scalar |
| NL shape | tanh | Stage 2 | Should emerge without assuming tanh |

---

## Mathematical Formulation

### Continuous-time state-space ODE for biquad filters

Each second-order filter section is represented as a continuous-time ODE
rather than as discrete IIR coefficients. This allows fitting in continuous
time and discretizing at any sample rate.

For a second-order highpass filter with cutoff ω₀ = 2πfc and damping ζ:

```
State vector:  s = [s₁, s₂]ᵀ

ODE:
  ṡ₁ = s₂
  ṡ₂ = -ω₀² s₁  -  2ζω₀ s₂  +  ω₀² u(t)

Output (highpass):
  y(t) = u(t) - s₁(t)
```

For a Butterworth filter, ζ = √2/2 ≈ 0.707 (fixed). The only free parameter
is ω₀, which maps to fc = ω₀ / (2π). This is a 1-parameter fit per filter.

For the high-shelf biquad the state-space form follows the Audio EQ Cookbook
but re-parameterized in continuous time with free parameters ω₀ (shelf
frequency) and A = 10^(presence_db/40) (shelf gain).

### Full system ODE

The complete system has 6 state variables (2 per biquad × 3 filters):

```
State:  [s₁_hpf1, s₂_hpf1, s₁_hpf2, s₂_hpf2, s₁_shelf, s₂_shelf]

Parameters θ = {ω₁, ω₂, ω_shelf, A_shelf, drive, output_level, NL_params}

Forward pass:
  1. Integrate HPF1 ODE with input u(t) → pre-filtered signal x_pre(t)
  2. Apply drive:  x_driven(t) = drive · x_pre(t)
  3. Apply NL:     x_nl(t) = NL(x_driven(t))        [pointwise, no ODE]
  4. Integrate HPF2 ODE with input x_nl(t) → x_post(t)
  5. Integrate shelf ODE with input x_post(t) → x_shelf(t)
  6. Apply level:  ŷ(t) = output_level · x_shelf(t)
```

### ODE solver

- **Training:** `torchdiffeq.odeint` with `method='rk4'`, fixed step size
  `dt = 1/sr` (one step per audio sample). Fast, deterministic, differentiable.
- **Validation:** `torchdiffeq.odeint` with `method='dopri5'` (adaptive
  Runge-Kutta). More accurate, used only for final parameter validation.
- **Adjoint method:** Use `odeint_adjoint` for backpropagation through the
  ODE solve. Required for sequences of >100k samples — direct backprop
  through unrolled RK4 at 96 kHz is infeasible in memory.

### Nonlinearity representation

The nonlinearity is represented as a **small MLP with odd-symmetry enforced
by construction**:

```python
NL(x) = x · σ_mlp(x²)
```

where σ_mlp is a 3-layer MLP with input x², hidden size 8, output scalar,
positive output constraint (softplus activation on final layer). This
parameterization guarantees NL(0)=0, NL(-x)=-NL(x) (odd symmetry), and
NL'(0) > 0 (gain at origin). tanh satisfies all three — if the data is
generated by tanh, this MLP should converge to tanh-like behavior.

The odd-symmetry constraint is separate from the symmetry regularization
penalty — it is structural, not a soft prior. This is deliberate: for the
real Notaklon with asymmetric germanium diode clipping, we will switch to
an unconstrained MLP and let asymmetry emerge from data.

---

## Three-Stage Fitting Strategy

The three-stage approach exploits the fact that different amplitude regimes
of the training signal provide different information about different parameters.

### Signal amplitude stratification

The training signal `val_signal_v1.wav` contains stepped sine tone sections
at drive levels l-24, l-12, l-3 (approximately -24, -12, -3 dBFS). At
drive=4.0:

| Input level | Signal at tanh input | tanh error vs linear |
|-------------|---------------------|---------------------|
| -24 dBFS (0.063) | 0.25 | 2.1% — effectively linear |
| -12 dBFS (0.25) | 1.0 | 23.8% — clearly nonlinear |
| -3 dBFS (0.71) | 2.83 | clipping hard |

**Stage 1 uses only the lowest-amplitude sections** where the system is
approximately linear. **Stage 2 uses the highest-amplitude sections** where
the nonlinearity is fully engaged. **Stage 3 uses the full signal.**

---

### Stage 1 — Pre-filter and post-filter identification (linear regime)

**Goal:** Recover fc₁, fc₂, presence_hz, presence_db, and the combined
gain product (drive × output_level) from the approximately linear regime.

**Data:** Lowest-amplitude sections only (l-24 sections, or equivalent
amplitude threshold < -20 dBFS RMS per section).

**Method:** Frequency-domain loss on log-magnitude transfer function.

For each windowed segment of the low-amplitude sections, compute:
```
H_measured(f) = FFT(wet_segment) / FFT(dry_segment)
H_model(f)    = G · HPF1(f; ω₁) · HPF2(f; ω₂) · Shelf(f; ω_s, A_s)
```
where G = drive × output_level is a single combined gain scalar.

**Loss:**
```
L₁ = Σ_f  w(f) · [log|H_measured(f)| - log|H_model(f)|]²
```

Frequency weighting w(f):
- Zero weight below 20 Hz (DC/subsonic, unreliable)
- Unit weight 20 Hz – 10 kHz
- Tapered to zero above 15 kHz (noise floor influence)

**Free parameters at Stage 1:** ω₁, ω₂, ω_shelf, A_shelf, G (5 parameters)
**Frozen:** drive, output_level individually, NL_params

**Optimizer:** Adam, lr=1e-3, 2000 steps on the frequency-domain loss.
No ODE solver at this stage — transfer function is computed analytically
from the continuous-time pole-zero representation.

**Convergence criterion:** L₁ < 0.01 (mean squared log-mag error < 0.1 dB
per frequency bin).

**Diagnostic:** Plot measured vs fitted transfer function magnitude. If
residual has harmonic structure (energy at 2f, 3f for a sine input), the
sections chosen are not sufficiently in the linear regime — tighten the
amplitude threshold.

---

### Stage 2 — Nonlinearity and drive/level identification (nonlinear regime)

**Goal:** Recover drive, output_level, and NL shape. Freeze Stage 1 results.

**Data:** Highest-amplitude sections only (l-3 sections, or equivalent
amplitude threshold > -6 dBFS RMS per section).

**Method:** Two sub-steps.

**Sub-step 2a — Construct intermediate signals**

Using Stage 1's fitted pre-filter, compute the signal at the tanh input:
```
z[n] = PreFilter_fitted(dry[n]) · drive_candidate
```

Invert the post-filter to estimate the nonlinearity output:
```
ñ[n] = PostFilter_fitted⁻¹(wet[n]) / output_level_candidate
```

Post-filter inversion uses a minimum-phase inverse filter (Wiener filter
with regularization β=1e-4 to avoid noise amplification). This gives
paired samples (z[n], ñ[n]) that are approximately input-output pairs of
the nonlinearity alone.

**Sub-step 2b — Identifiability anchor**

drive and output_level are not separately identifiable from the nonlinearity
shape alone (scaling degeneracy). Apply the energy-balance anchor:

```
output_level = RMS(PostFilter_fitted⁻¹(wet)) / RMS(NL(PreFilter_fitted(dry)))
```

evaluated at convergence. This is the unique output_level that makes the
energy balance correct given the nonlinearity shape. Drive is then determined
by the amplitude at which NL transitions from linear to saturating.

**Loss:**
```
L₂ = MSE(NL(z[n]), ñ[n])
   + λ_smooth · Σ_n (NL''(z_n))²
```

λ_smooth = 1e-4 initially, annealed to 1e-6 over training.
No symmetry penalty here — symmetry is enforced structurally by the MLP
parameterization (odd-symmetry by construction).

**Free parameters at Stage 2:** drive, output_level, NL_params
**Frozen:** ω₁, ω₂, ω_shelf, A_shelf (from Stage 1)

**Optimizer:** Adam, lr=5e-4, 3000 steps.

**Convergence criterion:** MSE(NL(z), ñ) < 0.005.

**Diagnostic:** Plot learned NL(x) vs tanh(drive·x)/drive (the true
nonlinearity after absorbing drive into the axis). If the recovered shape
matches tanh but the x-axis is scaled differently, the drive/level anchor
has not converged — recheck energy balance.

---

### Stage 3 — Joint refinement (full signal, full ODE)

**Goal:** Fine-tune all parameters jointly. Stage 1+2 provides initialization
close enough to the global minimum that joint optimization is well-conditioned.

**Data:** Full training signal.

**Method:** End-to-end ODE solve with composite loss. This is the only
stage that uses torchdiffeq.

**Loss:**
```
L₃ = α · MSE_time(ŷ[n], wet[n])
   + β · MSE_logmag(FFT(ŷ), FFT(wet))
   + γ · L_smooth(NL)
   + δ · L_physics(θ)
```

where:
- MSE_time: standard time-domain mean squared error
- MSE_logmag: log-magnitude spectral loss (same as L₁ but on full signal)
- L_smooth: smoothness regularizer on NL second derivative
- L_physics: soft physics constraints (see below)

**Physics constraint terms:**
```
L_physics = penalty(fc₁ < 20 Hz)      # HPF below bass range → unphysical
           + penalty(fc₁ > 500 Hz)     # HPF above midrange → wrong character
           + penalty(fc₂ < 100 Hz)     # Post HPF below bass → unphysical
           + penalty(fc₂ > 2000 Hz)    # Post HPF above mid → wrong character
           + penalty(drive < 1.0)      # Sub-unity drive → not a gain stage
           + penalty(output_level > 1.0)  # Unity gain exceeded → unphysical
```

Each penalty is a soft ReLU: penalty(condition) = ReLU(violation)².

**Loss weights:**
```
α = 1.0   (time MSE — primary)
β = 0.1   (spectral — regularizer)
γ = 1e-5  (NL smoothness — weak prior)
δ = 0.01  (physics — hard enough to prevent unphysical solutions)
```

**Free parameters at Stage 3:** all (ω₁, ω₂, ω_shelf, A_shelf, drive,
output_level, NL_params)

**Optimizer:** Adam with cosine annealing.
  - lr_max = 1e-4 (lower than Stage 1/2 — near optimum, fine-tune only)
  - lr_min = 1e-6
  - T_max = 5000 steps
  - Gradient clipping: max_norm = 1.0

**Convergence criterion:** ESR < 0.01 on full validation signal.

**Diagnostic:** Parameter trajectory plot — all 6 physical parameters vs
training step. Should converge smoothly from Stage 1/2 initialization
without large excursions.

---

## Success Criteria

### Primary (both must pass)

**1. ESR threshold:**
```
ESR(ŷ, wet) < 0.01   on full validation signal
```
This matches TCN performance on KlonSim (ESR=0.0072) and is substantially
better than the classical WH baseline (ESR=0.65). The physics-informed model
should outperform a structural-prior-matched classical model because it has
the exact topology, not just the topology class.

**2. Parameter recovery:**

| Parameter | Ground truth | Tolerance | Basis |
|-----------|-------------|-----------|-------|
| fc₁ | 80 Hz | ±8 Hz (10%) | Cramér-Rao motivated |
| fc₂ | 350 Hz | ±35 Hz (10%) | Cramér-Rao motivated |
| presence_hz | 3000 Hz | ±300 Hz (10%) | Cramér-Rao motivated |
| presence_db | 2.0 dB | ±0.3 dB | Perceptual JND |
| drive | 4.0 | ±0.6 (15%) | Degeneracy-aware |
| output_level | 0.7 | ±0.1 (15%) | Degeneracy-aware |
| NL shape | tanh | R² > 0.999 vs tanh | Functional match |

Wider tolerance on drive/output_level reflects the inherent scaling
degeneracy — even with anchoring, the product drive×output_level is
better-conditioned than either individually.

### Secondary (informative, not pass/fail)

- **Stage 1 transfer function fit:** log-magnitude RMSE < 0.5 dB across
  20 Hz – 10 kHz
- **Stage 2 NL shape:** recovered shape matches tanh(4·x)/tanh(4) with
  R² > 0.999 on x ∈ [-1, 1]
- **No local minimum capture:** Stage 3 parameter trajectory shows
  monotone convergence from Stage 1/2 initialization
- **Noise floor correct:** residual (wet - ŷ) RMS matches -70 dBFS
  Gaussian noise level (within 3 dB)

---

## Non-Uniqueness and Identifiability

The Wiener-Hammerstein decomposition is unique only up to certain scaling
ambiguities. Specifically:

**Drive/level degeneracy:** Multiplying drive by k and dividing output_level
by k changes the operating point of the tanh but not the combined gain in
the linear regime. This is resolved by the energy-balance anchor in Stage 2.

**Filter frequency ambiguity:** In principle, some of the post-clip HPF
character could be absorbed into a different nonlinearity shape. This is
resolved by the Stage 1 linear-regime identification which constrains fc₂
before the nonlinearity is fitted.

**Implication for real Notaklon:** If the optimizer converges to a solution
with fc₁ ≠ 80 Hz but ESR is excellent, this is not necessarily a failure —
it may be a genuinely equivalent parameterization. The correct response is
to design a targeted low-amplitude excitation sweep around the suspected
frequency range (new data) and re-optimize. Do not perturb the solution
without new data.

---

## Diagnostic Outputs

The experiment must produce the following plots and tables automatically
on completion:

### Plots

1. **Stage 1: Transfer function fit** — measured H(f) vs fitted H(f),
   log-magnitude, 20 Hz to 20 kHz. Include residual panel.

2. **Stage 2: Nonlinearity shape** — learned NL(x) vs tanh(x) vs
   tanh(drive·x) on x ∈ [-2, 2]. Include derivative NL'(x).

3. **Stage 3: Parameter convergence** — 6-panel plot, one per physical
   parameter, showing value vs training step from Stage 3 initialization
   to convergence. Draw horizontal dashed line at ground truth.

4. **Full signal: waveform overlay** — 40ms window of wet vs ŷ vs residual,
   selected from highest-drive section (most diagnostic).

5. **Full signal: ESR vs drive level** — ESR per section (same x-axis as
   notebook plot 9) for the ODE model vs TCN vs WH classical.

6. **Nonlinearity spectrum** — harmonic amplitudes (H1-H8) of the fitted
   model vs KlonSim at drive=4.0, 1 kHz sine input. Tests whether the
   recovered NL produces the correct harmonic distribution.

### Parameter recovery table

Printed to stdout and saved as `results/pedalode/klonsim_v1/param_recovery.csv`:

```
parameter    | ground_truth | recovered | error_pct | within_tolerance
fc1_hz       | 80.0         | ?         | ?         | PASS/FAIL
fc2_hz       | 350.0        | ?         | ?         | PASS/FAIL
presence_hz  | 3000.0       | ?         | ?         | PASS/FAIL
presence_db  | 2.0          | ?         | ?         | PASS/FAIL
drive        | 4.0          | ?         | ?         | PASS/FAIL
output_level | 0.7          | ?         | ?         | PASS/FAIL
nl_r2        | 1.0          | ?         | —         | PASS/FAIL
esr          | 0.0          | ?         | —         | PASS/FAIL
```

---

## Project Structure

```
pedal_model/
    ode/
        __init__.py
        filters.py          # Continuous-time state-space ODE filter classes
        nonlinearity.py     # Odd-symmetric MLP nonlinearity
        wh_model.py         # Full Wiener-Hammerstein ODE model
        fitting.py          # Three-stage fitting orchestrator
        losses.py           # All loss functions (L1, L2, L3, physics penalties)
        identifiability.py  # Energy-balance anchor, Cramér-Rao utilities

scripts/
    fit_klonsim_ode.py      # Entry point for this experiment

results/
    pedalode/
        klonsim_v1/
            stage1_params.json
            stage2_params.json
            stage3_params.json       # Final fitted parameters
            param_recovery.csv       # Success criterion table
            plots/                   # All diagnostic plots (PNG)

notebooks/
    pedalode/
        01_klonsim_ode_diagnostics.py   # Marimo notebook for visual inspection
```

---

## Dependencies

New dependencies required (add to pyproject.toml):
```
torchdiffeq >= 0.2.3
```

All other dependencies (torch, numpy, scipy, matplotlib) are already present.

---

## Relation to PedalODE Roadmap

This experiment is **Rung 1.5** in the PedalODE ladder — between the Boss GE-7
linear known-answer test and the full Notaklon nonlinear challenge:

```
Rung 1:   Boss GE-7         Linear system, known topology, fit parameters
Rung 1.5: KlonSim           Nonlinear (WH), known topology class,
                            recover parameters + nonlinearity shape  ← HERE
Rung 2:   Notaklon (real)   Nonlinear (parallel paths), unknown topology,
                            discover structure + fit parameters
```

The key capability being validated at Rung 1.5 that Rung 1 did not test:
- Differentiable ODE backprop through a nonlinear system
- Three-stage amplitude-stratified fitting
- Nonlinearity shape recovery without structural assumptions
- Identifiability anchoring for scaling degeneracy

Success at Rung 1.5 is a prerequisite for Rung 2. If parameter recovery
fails here on a known-answer system, the approach needs revision before
being applied to real hardware where ground truth is unavailable.

---

## Skeptical-Colleague Notes

Per PedalODE SKILL.md, the following decisions are flagged as worth
challenging before implementation:

**Decision: odd-symmetry enforced structurally in the MLP.**
Rationale: KlonSim uses tanh (odd-symmetric). But the real Notaklon has
asymmetric clipping. Building in odd-symmetry now creates a structural
assumption that must be removed later. Counter-argument for keeping it:
this is a validation experiment, and the symmetry constraint makes the
Stage 2 problem better-conditioned. Resolution: implement both versions
(symmetric and unconstrained MLP), run both, report which recovers
better parameters. This is one experiment with two conditions, not two
separate experiments.

**Decision: three separate optimization stages rather than joint from
random initialization.**
Rationale: Joint optimization from random init on a WH system is known
to suffer from local minima (the pre-filter and post-filter can swap
roles). The staged approach prevents this. Counter-argument: if Stage 1
or 2 gives a poor initialization, Stage 3 may converge to a wrong local
minimum anyway. Resolution: run Stage 3 from both the staged initialization
and from 5 random initializations. Report all results. If staged init
consistently outperforms random, the strategy is validated.

**Decision: rk4 fixed-step at 1/sr for training.**
Rationale: Simple, fast, exactly matches the sample rate. Counter-argument:
audio signals have energy up to 48 kHz (at 96 kHz sr) and the rk4 step
size must satisfy the Nyquist criterion for the highest-frequency state
variable. For fc₂=350 Hz the state evolves at ~2.2 krad/s, and rk4 is
stable for step sizes up to ~2.8/ω₀ ≈ 1.3ms. At 96 kHz, dt=10.4μs, which
is far inside the stability region. This is safe.
