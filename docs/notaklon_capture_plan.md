# JHS Notaklon — Hardware Capture Plan

Parallel plan to `ge7_plan.md`. Covers physical routing, level calibration, knob
positions, directory structure, capture procedure, post-capture validation, and the
combined Notaklon → GE-7 chain plan for song-specific settings.

**Status: PARTIAL — bypass done, level verified. Preset captures pending.**

---

## 1. Pedal Overview

The JHS Notaklon is a Klon Centaur clone.

**Dial travel: 6 o'clock (min/CCW) → 4 o'clock (max/CW) = 10 hours total.**

| Control | Range | Noon | Effect |
|---|---|---|---|
| **GAIN** | 6 o'c → 4 o'c | 12 o'c | Drive + clean blend (coupled). Fully CCW = clean boost. CW = max overdrive. Mid presence increases with gain. |
| **TREBLE** | 6 o'c → 4 o'c | 12 o'c | Boost/cut shelf. **Noon = neutral (0 dB flat).** CCW = treble cut (to −8 dB), CW = treble boost (to +8 dB). |
| **VOLUME** | 6 o'c → 4 o'c | 12 o'c | Output level. Noon ≈ −0.8 dB. Max (4 o'clock) = +16 dB. |
| **Shamrock Mod** | Off / On | — | +4 dB gain, mid boost (~800 Hz), additional hard-edge clipping stage. |

### Normalised position reference

```
 6 o'c   7     8     9    10    11    12    1     2     3    4 o'c
  0.0   0.1   0.2   0.3  0.4   0.5   0.6  0.7   0.8   0.9   1.0
  min                    ← songs land here →     noon          max
```

Songs are all **9–12 o'clock = normalised 0.3–0.6** on the gain knob.

---

## 2. Hardware Routing

```
DAW playback (fader 0 dB)
    │
    ▼
Reamp box  →  −12.4 dB insertion loss  →  −14 dBFS peak at pedal input
    │
    ▼
Notaklon INPUT  ──→  Notaklon OUTPUT
                            │
                            ▼
                   Interface Input  ← RECORD THIS, not DAW loopback
```

**Chain routing (Notaklon → GE-7):**
```
Reamp out → Notaklon in → Notaklon out → GE-7 in → GE-7 out → Interface
```

---

## 3. Level Calibration — VERIFIED

**DAW fader 0 dB is confirmed correct. Do not change it.**

Measured from hardware test captures:

| Capture | Setting | Output peak | Headroom | Clipped |
|---|---|---|---|---|
| bypass | pedal off | −14.0 dBFS | — | 0 |
| V12-T12-G12 | G12, noon vol, Shamrock off | −17.5 dBFS | +17.4 dB | 0 |
| V4-T4-G4 + Shamrock | ALL MAX | **−12.6 dBFS** | **+12.5 dB** | 0 |

Worst case (all-max + Shamrock) still has 12.5 dB of headroom. The DAW fader needs
no adjustment for any capture within the 9–12 o'clock range.

Record the calibration:

```json
{
  "date": "YYYY-MM-DD",
  "reamp_box": "Radial ProRMP",
  "playback_fader_db": 0.0,
  "input_peak_at_interface_dbfs": -14.0,
  "reamp_insertion_loss_db": -12.4,
  "worst_case_output_dbfs": -12.6,
  "worst_case_setting": "V4-T4-G4-Shamrock",
  "headroom_db": 12.5
}
```

---

## 4. Capture Positions

### Coverage rationale

Songs span **9–12 o'clock** on the gain knob. The plan is to start with **4 presets
at 1-hour steps** and test the model quality before adding more. If interpolation
between captures looks smooth on the val set, no further captures are needed for the
song range. If there are audible artefacts between positions, add the half-steps.

```
Position    Norm   Drive   Blend  Status    Songs
──────────────────────────────────────────────────────────────────────
bypass      —      —       —      DONE      Reference
G9  noon    0.30   2.11×   0.68   pending   Hey Joe, Fire on the Mountain
G10 noon    0.40   2.70×   0.59   pending   The Thrill Is Gone
G11 noon    0.50   3.46×   0.50   pending   Flagpole Sitta, TNT, ILRNR, Crossroads
G12 noon    0.60   4.44×   0.41   pending   High-gain headroom test
```

All at **Treble noon (12 o'clock), Volume noon (12 o'clock), Shamrock off**.

### Optional additions (capture only if model quality requires it)

```
G9.5  noon  0.35   —      —      optional  half-step for finer interp
G10.5 noon  0.45   —      —      optional
G11.5 noon  0.55   —      —      optional
G11   noon + Shamrock               optional  Shamrock variant
```

### Preset JSON format (v2.2, schema with 6→4 dial)

```json
{
  "schema_version": "2.2",
  "pedal": "jhs_notaklon",
  "knobs": { "gain_oclock": 11.0, "treble_oclock": 12.0, "volume_oclock": 12.0, "shamrock": false },
  "klonsim_approx": {
    "drive": 3.464, "clean_blend": 0.500, "treble_db": 0.0,
    "output_level": 0.911, "clip_asymmetry": 0.15, "shamrock": false
  }
}
```

`treble_db`: exactly 0.0 at noon with the corrected 6→4 dial range.
`output_level`: 0.911 (≈ −0.8 dB) at noon volume — close to unity.

---

## 5. Song → Chain Mapping

| Song | Notaklon | GE-7 preset | GE-7 slider notes |
|---|---|---|---|
| Fire on the Mountain | gain9_noon | ??? | |
| Flagpole Sitta | gain11_noon | ??? | |
| Hey Joe | gain9_noon | ??? | |
| I Love Rock 'N Roll | gain11_noon | ??? | |
| T.N.T. | gain11_noon | ??? | |
| The Thrill Is Gone | gain10_noon | ??? | |

---

## 6. Directory Structure

```
recordings/data/notaklon/
    presets/
        gain9_noon.json     [DONE — v2.2]
        gain10_noon.json    [DONE — v2.2]
        gain11_noon.json    [DONE — v2.2]
        gain12_noon.json    [DONE — v2.2]
    captures/
        calibration.json    ← fill with values from §3
        bypass/
            wet_train_signal_v2_48k.wav    [DONE — valid]
            params.json
        gain9_noon/         ← pending
        gain10_noon/        ← pending
        gain11_noon/        ← pending
        gain12_noon/        ← pending
    Notaklon - bypass.wav              [VALID — move to captures/bypass/]
    Notaklon - V12-T12-G12.wav         [level test capture — G12 noon vol]
    Notaklon - V4-T4-G4-True.wav       [level test capture — all-max + Shamrock]
    Notaklon - train_signal_v2_48k.wav [dry reference copy]
```

---

## 7. Capture Procedure

1. **Routing check first** — 5-second test, verify correlation with dry < 0.95:
   ```bash
   uv run python - << 'EOF'
   import soundfile as sf, numpy as np
   dry, _ = sf.read("recordings/data/signals/train_signal_v2_48k.wav", dtype="float32")
   wet, _ = sf.read("recordings/data/notaklon/captures/gain11_noon/test_5s.wav", dtype="float32")
   n = min(len(dry), len(wet))
   corr = float(np.corrcoef(dry[:n], wet[:n])[0, 1])
   peak = float(20 * np.log10(abs(wet).max() + 1e-12))
   print(f"corr={corr:.4f}  peak={peak:+.1f} dBFS")
   print("OK" if corr < 0.95 else "ROUTING ERROR — recording DAW loopback")
   EOF
   ```
2. **Knob positions**: set from preset JSON. Gain, Treble noon, Volume noon, Shamrock off.
3. **Record full 642.5 s** of `train_signal_v2_48k.wav`. Save to `captures/<preset>/wet_train_signal_v2_48k.wav`.
4. **Verify** immediately (§8).
5. Repeat for next preset. **Do not change the DAW fader.**

---

## 8. Post-Capture Validation

| Check | Pass | Flag |
|---|---|---|
| Not loopback | corr with dry < 0.95 | > 0.99 = routing error |
| No ADC clipping | peak < −0.1 dBFS | any sample > −0.05 dBFS |
| Full duration | > 640 s of active content | < 600 s = connection dropped |
| Gain vs bypass | ±6 dB of expected | > 6 dB off |
| Noise floor | < −80 dBFS in silent gaps | > −70 dBFS |
| 2nd harmonic | H2 at −30 to −20 dBc @ 440 Hz | H2 < −50 dBc |

```bash
uv run python scripts/validate_notaklon_capture.py --preset gain11_noon
```

---

## 9. Bypass Level Analysis

| File | Status | Peak | Lag | Corr vs dry |
|---|---|---|---|---|
| `Notaklon - bypass.wav` | VALID | −14.0 dBFS | 10 samples | 0.979 |
| `Notaklon - V12-T12-G12.wav` | Level test | −17.5 dBFS | — | 0.679 |
| `Notaklon - V4-T4-G4-True.wav` | Level test (worst case) | −12.6 dBFS | — | — |

Reamp insertion loss: −12.4 dB. Signal at pedal input: −14.0 dBFS peak. Correct.

---

## 10. Synthetic Captures

```bash
uv run python scripts/make_klonsim_captures.py --gain 9  --treble 12 --volume 12
uv run python scripts/make_klonsim_captures.py --gain 10 --treble 12 --volume 12 --name klon_sim_v2_gain10_t12_v12
uv run python scripts/make_klonsim_captures.py --gain 11 --treble 12 --volume 12
uv run python scripts/make_klonsim_captures.py --gain 12 --treble 12 --volume 12 --name klon_sim_v2_gain12_t12_v12
```

---

## 11. Implementation Status

```
[✓] pedal_model/synthetic/pedal.py      KlonSim v2.1 — dial corrected to 6→4 (10hr)
[✓] recordings/data/notaklon/presets/gain*_noon.json  v2.2 — treble_db=0.0 at noon, correct range
[✓] Level verified: all-max + Shamrock peaks at −12.6 dBFS, 12.5 dB headroom

CAPTURES — start with 4, test model, add half-steps if needed:
[✓] bypass           DONE (valid)
[ ] gain9_noon       pending
[ ] gain10_noon      pending
[ ] gain11_noon      pending
[ ] gain12_noon      pending

OPTIONAL (only if model quality requires):
[ ] gain9.5_noon     half-step
[ ] gain10.5_noon    half-step
[ ] gain11.5_noon    half-step
[ ] gain11_shamrock  Shamrock variant

BLOCKED on hardware:
[ ] recordings/data/chain/*/hardware/    chained Notaklon → GE-7 captures
[ ] scripts/synthesize_chain.py (GE-7 per-song settings still ???)
```
