# Session 2026-10-22_1625_48k (code 606f705): pre-flight for the record run

Return on In 3, Sweeps only. C1-C4 ok (C2 and C3 run twice; final references band coherence
>= 0.9996). C2: limit on the send side at -9 dBFS sent (1 kHz and 60 Hz agree in sent level, 5 dB
apart in return level). Auto-level sent flat at -12 dB (train) and -15 dB (val), not -23 as at
16:00: the quietest sweep is 57 / 47 dB over the noise. No retakes; every QC item ok.

Repeatability, untouched (flat train, flat val, C4 over ~5 min): within 0.035 dB, 0.3 degrees.
Pre-flight passed: the 0.35 dB val shift of session 1600 did not recur.

Across sessions flat differs by up to 0.5 dB in level and shape (engaged / bypassed at 1 kHz:
-1.65 on 10-21, -1.95 at 16:00, -1.29 at 16:25). The chain cancels in that ratio, so the likely
cause is where the sliders sit in their centre detents each time flat is set: a setting error,
which the record run's re-plugged takes (flat_replug, smile_replug) measure.
