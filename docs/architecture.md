# PedalDSP — Architecture Notes

## Audio resampling

PedalDSP uses **soxr** for all sample-rate conversion. soxr is the
reference-quality resampler used by SoX, ffmpeg, and most professional audio
tools. It is preferred over scipy or torchaudio resamplers for any audio that
will be evaluated by ear. All 44.1 kHz ↔ 96 kHz conversion in this project
goes through `pedal_model/utils/resampling.py` using soxr's `'HQ'` quality
preset.

`'VHQ'` is available but approximately 2× slower with no audible difference
for 44.1 ↔ 96 kHz conversion at guitar-signal amplitudes. Use `'HQ'` unless
you have a specific reason for the extra CPU cost.

The existing `pedal_model/data/resample.py` (used for Daisy Seed 96 k → 48 k
export) uses scipy polyphase resampling, which is adequate for embedded
deployment but not for DAW bounce quality. Do not mix the two resampling paths.

## DAW inference pipeline

`scripts/render_to_daw.py` implements the full 44.1 kHz → 96 kHz → model → 44.1 kHz
pipeline:

1. Load mono 44.1 kHz dry WAV from Studio 1824c.
2. Upsample to 96 kHz via soxr HQ (`resample_audio`).
3. Run KlonSim circuit simulation and/or each selected model through `StreamRunner`.
4. Downsample back to 44.1 kHz, quantize to 24-bit PCM, and write WAV (`write_wav_44k`).

`StreamRunner` (`pedal_model/inference/stream_runner.py`) provides a unified
interface: it moves neural models to the available device, runs the full signal
as one contiguous pass, and falls back to 262 144-sample overlap-save chunking
for audio longer than 10 minutes.

## Sample rates

| Stage | Rate |
|---|---|
| Studio 1824c capture / DAW project | 44 100 Hz |
| Training data and model inference | 96 000 Hz |
| Daisy Seed embedded deployment | 48 000 Hz |

All in-memory audio in the `pedal_model` library is float32. Quantization
to int24 or int16 happens only at the output stage.
