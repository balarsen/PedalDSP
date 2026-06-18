# Pedal Capture Plan
## Guitar → Pedal → Audio Interface → Training Data

---

## 1. Hardware Hookup

```
Guitar
  │
  ├──[Y-splitter / DI box]────────────────────► Interface Input 1  (DRY)
  │
  └────► Pedal Input
            │
         Pedal Output ───────────────────────► Interface Input 2  (WET)
```

### What You Need
- **Passive Y-splitter** (TS to 2x TS) — cheapest option, slight loading on the signal but fine for capture
- **OR a DI box with a thru/parallel output** — cleaner, more accurate to how the pedal "hears" the guitar
- **Two TS/TRS cables** from the split to your interface inputs
- Your audio interface set to **48kHz, 24-bit**

### Input Gain Setting
- Plug in and strum hard
- Set gain on both channels so peaks sit around **-12 dBFS**
- Leave at least 6dB of headroom — you do NOT want any clipping in your training data

### Pedal Power
- Use the pedal's normal power supply, not batteries if avoidable
- Let the pedal warm up for 5 minutes before capture (analog circuits drift slightly when cold)

---

## 2. DAW Setup

Open your DAW (Reaper, Ableton, Logic, etc.) and:

1. Create a **stereo or two-mono-track session**
2. Input 1 → Track 1 (label: DRY)
3. Input 2 → Track 2 (label: WET)
4. **Disable all input monitoring effects / plugins** on both tracks — completely clean capture
5. Set buffer size to anything comfortable (doesn't affect recorded audio)
6. Record both tracks simultaneously — they will share the same clock

### Export Settings
When done: export both tracks as a **single stereo interleaved WAV**
- Channel 1 = DRY (left)
- Channel 2 = WET (right)
- 48kHz, 24-bit, no dithering, no normalization

---

## 3. Sweep Signal Generation

You'll generate a Python script that outputs a WAV file. Play this file through your interface into the guitar/pedal chain.

### What to Generate
A **logarithmic sine sweep** (also called an exponential sweep or Farina sweep). It spends more time at low frequencies where guitar energy lives.

```python
import numpy as np
import soundfile as sf

def generate_sweep(
    f_start=20,       # Hz
    f_end=20000,      # Hz
    duration=10.0,    # seconds
    sr=48000,
    amplitude=0.5     # keep below clipping
):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Logarithmic (exponential) sweep
    sweep = amplitude * np.sin(
        2 * np.pi * f_start * duration / np.log(f_end / f_start) *
        (np.exp(t / duration * np.log(f_end / f_start)) - 1)
    )
    return sweep.astype(np.float32)

sr = 48000

# Build the full capture signal
silence  = np.zeros(sr)                         # 1 sec silence at start
click    = np.zeros(sr // 4)                    # alignment click
click[0] = 0.9                                  # sharp single sample spike
sweep1   = generate_sweep(20,    20000, 10, sr) # full range sweep
sweep2   = generate_sweep(80,    8000,  10, sr) # guitar-focused sweep
sweep3   = generate_sweep(20,    20000, 10, sr) # repeat for averaging
noise    = (np.random.randn(sr * 5) * 0.3).astype(np.float32)  # white noise burst
silence2 = np.zeros(sr * 2)                     # trailing silence

signal = np.concatenate([
    silence,
    click,
    sweep1,
    silence,
    sweep2,
    silence,
    sweep3,
    silence,
    noise,
    silence2
])

sf.write('capture_sweep.wav', signal, sr, subtype='PCM_24')
print(f"Generated {len(signal)/sr:.1f} seconds of capture signal")
print("Output: capture_sweep.wav")
```

Run this once. It produces `capture_sweep.wav` — a ~45 second file.

---

## 4. Playing the Sweep Into the Pedal

You have two options:

### Option A — Through the Guitar (Most Accurate)
Feed the sweep out of your interface's headphone/line output into a **re-amping box** (like a Radial X-Amp or similar), then into the guitar's input on the pedal. This accurately replicates the impedance the pedal expects.

```
Interface Output → Re-amp Box → Pedal Input
```

### Option B — Direct Line Level (Easier for a Demo)
If you don't have a re-amp box, feed the interface line output directly into the pedal. The impedance won't be exactly right, but for a demo it's fine. Turn the output level down to instrument level (~100–200mV peak).

### Routing for Playback + Record Simultaneously
In your DAW:
- Load `capture_sweep.wav` on a playback track, routed to Interface Output 1
- Track 1 (DRY) recording from Interface Input 1
- Track 2 (WET) recording from Interface Input 2
- Hit record, then play — both channels capture while the sweep plays

---

## 5. Capture Session Order

Run these in one session without changing any gain settings:

| Step | Signal | Duration | Purpose |
|------|--------|----------|---------|
| 1 | Alignment click | 0.25 sec | Time-align channels |
| 2 | Log sweep 20–20kHz | 10 sec | Full frequency response |
| 3 | Log sweep 80–8kHz | 10 sec | Guitar-range detail |
| 4 | White noise | 5 sec | Transient response |
| 5 | Log sweep repeat | 10 sec | Averaging / consistency check |
| 6 | Real guitar playing | 2–3 min | Generalization data |

For step 6, play:
- Single notes, slow and fast
- Power chords, big strums
- Muted/palm-muted notes
- Natural harmonics
- Let notes fully decay to silence

---

## 6. Verification Before Training

```python
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

audio, sr = sf.read('my_capture.wav')  # stereo file
dry = audio[:, 0]
wet = audio[:, 1]

# Check alignment using the click
corr = np.correlate(dry[:sr], wet[:sr], mode='full')
lag = corr.argmax() - (sr - 1)
print(f"Channel lag: {lag} samples = {lag/sr*1000:.2f} ms")

# Quick level check
print(f"DRY peak: {np.max(np.abs(dry)):.3f}")
print(f"WET peak: {np.max(np.abs(wet)):.3f}")

# Plot first 2 seconds to visually confirm alignment
plt.figure(figsize=(12, 4))
t = np.arange(sr * 2) / sr
plt.plot(t, dry[:sr*2], label='DRY', alpha=0.7)
plt.plot(t, wet[:sr*2], label='WET', alpha=0.7)
plt.legend()
plt.title('DRY vs WET — first 2 seconds')
plt.xlabel('Time (s)')
plt.savefig('capture_check.png', dpi=150)
plt.show()
```

### What Good Data Looks Like
- Lag should be under 50 samples (< 1ms)
- WET signal should be clearly related to DRY but shaped/colored
- No clipping (peaks < 0.99)
- No DC offset (waveform centered at zero)

---

## 7. File Checklist Before Moving to Training

- [ ] Stereo WAV, 48kHz, 24-bit
- [ ] Channel 1 = DRY, Channel 2 = WET
- [ ] No clipping on either channel
- [ ] Alignment click present at start
- [ ] Both sweeps recorded
- [ ] 2–3 minutes of real guitar playing included
- [ ] Pedal settings noted (write them down — you'll want to reproduce them)

---

## Next Step

Once you have this WAV file, the next phase is:
1. **Alignment** — auto-correct the sample offset using the click
2. **Windowing** — slice into training pairs (e.g. 1–4 second chunks)
3. **Model** — train a small LSTM or TCN on (DRY → WET) pairs
4. **Inference** — run new dry guitar audio through the model in real time

That's Phase 2. Get the capture done first and we'll build from there.
