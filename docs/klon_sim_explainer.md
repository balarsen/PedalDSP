# KlonSim: Synthetic Klon-Style Overdrive Pedal

## What It Is

KlonSim is a purely synthetic signal processor that approximates the behavior
of a Klon Centaur-style overdrive pedal. It exists as a development and
testing tool — a known-answer target for training and evaluating pedal
emulation models before real hardware captures are available. Because it is
synthetic, its exact transfer function is fully known and every parameter is
deterministic and reproducible.

It is **not** a circuit simulation. It does not model individual components,
solve node voltages, or implement Wave Digital Filters. Instead it implements
a hand-designed signal chain whose qualitative behavior resembles the Klon
at a functional level.

---

## Signal Chain

The pedal processes audio through seven sequential stages:

```
dry input
    │
    ▼
[1] Input HPF  ─── 2nd-order Butterworth highpass, fc = 80 Hz
    │
    ▼
[2] Drive gain ─── scalar multiply by `drive` parameter
    │
    ▼
[3] tanh clip  ─── symmetric soft saturation
    │
    ▼
[4] Post-clip HPF ─ 2nd-order Butterworth highpass, fc = 350 Hz
    │
    ▼
[5] High shelf ─── +2 dB presence boost at 3 kHz (Audio EQ Cookbook biquad)
    │
    ▼
[6] Output level ─ scalar multiply by `output_level` parameter
    │
    ▼
[7] Noise floor ── seeded Gaussian at −70 dBFS
    │
    ▼
wet output (clipped to [−1, 1])
```

---

## Stage-by-Stage Explanation

### Stage 1 — Input HPF

A second-order Butterworth highpass filter with cutoff at 80 Hz. This models
the coupling capacitor at the pedal input, which blocks DC and very low
frequencies from reaching the gain stage. In the real Klon the input capacitor
forms an RC highpass with the input resistor; 80 Hz is a reasonable approximation
of that roll-off.

The filter is implemented using `scipy.signal.butter` with `output='sos'`
(second-order sections) for numerical stability, then applied with `sosfilt`.

### Stage 2 — Drive Gain

A simple scalar multiplication by the `drive` parameter. This represents the
pre-amplification stage in the Klon's gain circuit — boosting the signal level
before it hits the clipping stage. Higher drive pushes more signal into the
nonlinearity in Stage 3.

Default drive is 4.0 (≈ 12 dB). At 9 o'clock knob position the drive is
approximately 1.85×; at fully clockwise it reaches 12.0×.

### Stage 3 — Soft Clip (tanh)

The driven signal passes through `tanh(x)`, a smooth symmetric saturation
function. This is the primary nonlinearity in the chain and is what makes
KlonSim a nonlinear system rather than a pure filter.

`tanh` models the combined soft-clipping behavior of the op-amp and diodes in
the gain stage. At low drive levels tanh(x) ≈ x (linear regime). As drive
increases, the output saturates toward ±1, producing harmonic distortion. The
symmetry of tanh means it produces primarily odd harmonics (3rd, 5th, 7th...),
consistent with symmetric clipping.

This is a *static* (memoryless) nonlinearity — the output at any sample depends
only on the current input sample, not on past samples. This, combined with the
linear filters before and after it, makes KlonSim a **Wiener–Hammerstein
system**: a linear filter followed by a static nonlinearity followed by another
linear filter.

### Stage 4 — Post-Clip HPF

A second-order Butterworth highpass at 350 Hz, applied after the clipping stage.
This is the character-defining filter of the KlonSim. In the real Klon, the gain
stage output goes through a capacitor network that rolls off the bass before the
summing amplifier; the 350 Hz cutoff gives the output its "mid-forward" voice —
the low end is attenuated, emphasizing the midrange presence that the Klon is
known for.

Note that this filter operates on the *clipped* signal, not the dry signal. The
ordering matters: filtering after clipping is different from filtering before it.
A 350 Hz highpass on a clipped signal removes bass harmonics as well as bass
fundamentals.

### Stage 5 — High Shelf (Presence Boost)

A second-order high-shelf biquad filter with transition frequency 3000 Hz and
gain +2 dB, computed using the Audio EQ Cookbook formulas. This adds a gentle
presence boost in the upper midrange and treble, brightening the output relative
to a flat response.

The shelf is implemented as a direct-form II biquad via `scipy.signal.lfilter`.
The Q factor is fixed at 0.707 (Butterworth/maximally flat shelf slope).

At 9 o'clock treble position the shelf gain is approximately −0.8 dB (slight
cut); at fully clockwise it reaches +12 dB (substantial boost).

### Stage 6 — Output Level

A scalar multiplication by `output_level`, which controls the final loudness
of the processed signal. This is separate from the drive gain — drive controls
how hard the signal is pushed into the saturation, while output level simply
scales the result.

Default output level is 0.7. At 9 o'clock volume position it is approximately
0.35; at fully clockwise it reaches 1.0.

### Stage 7 — Noise Floor

Seeded Gaussian white noise at −70 dBFS RMS is added to the output. The RNG
is seeded from `noise_seed` (default 42), so the noise pattern is fully
reproducible for a given set of parameters. At −70 dBFS the noise is well
below the audible threshold for most signals but is present in the signal
for realism and to prevent the output from being exactly zero in silence.

The final output is hard-clipped to [−1, 1] with `np.clip`.

---

## Topological Classification

KlonSim has the following structure:

```
HPF₁ → (×drive) → tanh → HPF₂ → HighShelf → (×level) → + noise
└─────────────── Linear ──────┘ └─── NL ──┘ └──────── Linear ────────────┘
```

More precisely: two linear filters sandwich the nonlinearity, making this
a **Wiener–Hammerstein** system. The `drive` gain is absorbed into the
pre-filter linear path (scaling a linear system is still linear).

This topology has a direct implication for the model zoo: among classical
black-box models, only Wiener–Hammerstein has the right structural prior to
fit this exactly. IIR, FIR, and Hammerstein are all wrong by construction.
Volterra can approximate it but requires high order. Neural models (TCN, LSTM)
should outperform all classical baselines because they can represent the
composed linear–nonlinear–linear structure without being constrained to any
specific topology.

---

## What KlonSim Is Missing Relative to the Real Klon

KlonSim deliberately omits several features of the real Klon Centaur circuit
to keep it tractable as a development target:

**No parallel clean blend.** The real Klon splits the signal into three
parallel paths — the gain/clip stage, a lowpass feed-forward network (FF1)
passing clean bass below ~106 Hz, and a gain-compensating blend network (FF2).
All three are summed at the output. KlonSim is a purely serial chain with no
parallel paths. This is the single largest structural difference.

**No asymmetric germanium diode clipping.** The real Klon uses germanium diodes
in the feedback loop of the gain stage, producing soft asymmetric clipping
(stronger clipping on one polarity than the other). KlonSim uses symmetric
`tanh`, which clips both polarities equally and produces only odd harmonics.
The real pedal produces even harmonics as well.

**No op-amp slew rate limiting.** The real Klon's second op-amp stage is
powered from a charge-pumped ±18V supply, giving it very high slew rate and
headroom for fast transients. KlonSim has no slew rate model.

**No gain-dependent tonal balance.** In the real Klon, the FF2 network works
against the gain pot so that as gain increases, the clean blend decreases,
maintaining a consistent overall tonal balance. KlonSim's `drive` parameter
simply scales into the saturation with no compensating network.

**No output buffer.** The real Klon has a dedicated output buffer stage.
KlonSim's output stage is just a scalar.

---

## Parameters and Knob Mapping

### KlonSimParams (internal physical parameters)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `drive` | 4.0 | Pre-gain multiplier into tanh (linear scale) |
| `input_hpf_hz` | 80.0 | Input coupling-cap HPF cutoff in Hz |
| `clip_hpf_hz` | 350.0 | Post-clip HPF cutoff in Hz |
| `presence_hz` | 3000.0 | High-shelf transition frequency in Hz |
| `presence_db` | 2.0 | High-shelf gain in dB |
| `output_level` | 0.7 | Output scalar in (0, 1] |
| `noise_floor_dbfs` | −70.0 | Noise RMS in dBFS |
| `noise_seed` | 42 | RNG seed for reproducibility |

### KlonKnobs (user-facing knob positions)

Knob positions are normalised to [0, 1] where 0 = fully counterclockwise
(6 o'clock) and 1 = fully clockwise (5 o'clock next day). The conversion
from knob position to physical parameter is:

| Knob | Parameter | Mapping |
|------|-----------|---------|
| `gain` | `drive` | Logarithmic: 1.5× to 12.0× (log₁₀ interpolation) |
| `treble` | `presence_db` | Linear: −3 dB to +12 dB |
| `volume` | `output_level` | Linear: 0.2 to 1.0 |

The default training knob position is **9 o'clock** (≈ 0.273), corresponding
to drive ≈ 1.85×, presence ≈ −0.8 dB, output level ≈ 0.35.

The standard training configuration used in `klon_sim_v1` is:
- Gain: 11 o'clock (≈ 0.45) → drive ≈ 4.0
- Treble: 12 o'clock (0.545) → presence ≈ +2 dB
- Volume: 9 o'clock (0.273) → output level ≈ 0.35

---

## Reproducibility

KlonSim is fully deterministic for a fixed set of parameters and sample rate.
The noise floor uses `np.random.default_rng(noise_seed)` which is isolated from
the global numpy RNG state. All filters are applied with initial conditions of
zero (steady-state from silence). The `to_dict` / `from_dict` round-trip
preserves all parameters exactly.

The VERSION string (`"1.0"`) is embedded in serialised results so that future
changes to the simulation can be tracked.

---

## Usage

```python
from pedal_model.pedals.klon_sim import KlonSim, KlonSimParams, KlonKnobs

# Default parameters
pedal = KlonSim()
wet = pedal.process(dry_audio, sr=96000)

# Custom knob positions
knobs = KlonKnobs(gain=0.45, treble=0.545, volume=0.273)
params = KlonSimParams(**knobs.to_params_delta())
pedal = KlonSim(params)
wet = pedal.process(dry_audio, sr=96000)

# Serialise / deserialise
d = pedal.to_dict()
pedal2 = KlonSim.from_dict(d)
```
