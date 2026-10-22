# Session 2026-10-22_1600_48k: the In 3 test (code 10c9bca)

Return on **In 3** (ProDI XLR moved from In 1), quick signal, flat only (train + val). A test
session, not part of the record.

## Findings

1. **In 3 is at least as good as In 1.** Same send (-9 dBFS at 1 kHz): In 1 compressed 1.0 dB
   (yesterday), In 3 0.33 dB with a 2.6 dB hotter return. Keep In 3.
2. **The limit is on the send side, not the input.** C2's tone steps start compressing at the same
   sent level at 1 kHz and 60 Hz (0.3/0.2 dB at -9 dBFS sent, 2.3/2.1 at -6, 4.7/4.6 at -3) while
   the 60 Hz return is 5 dB lower. An input limit would trip at the same return level. It matches the
   old profile note "the passive reamp path saturates above about -12 dBFS". The old code took the
   60 Hz return level (-19 dBFS) as a return ceiling and sent flat at -23 dB; with the limit located
   it sends -14 dB (9 dB more signal). Fixed after this session (`calibrate.limit_kind`).
3. **This session's C3 (bypass) reference was disturbed** while its sweep passed 2.9 kHz (band
   coherence 0.972-0.984; Welch's averaged coherence read 0.9997 and missed it). It adds a 0.06 dB
   bump near 3 kHz to every ratio here. C2-C4 now warn below 0.995 and ask for the step again.
4. **flat train matches C4 within 0.02 dB; flat val is 0.1-0.35 dB higher** (broad, peaking near
   1 kHz). Nothing was touched between the two (one Ready press plays both). Each take is linear on
   its own (three sweep levels agree within 0.02 dB). Unexplained so far: the repeatability
   pre-flight below decides whether the record run can go ahead.
5. **Across sessions, flat differs by ~0.3 dB** (yesterday In 1 vs today In 3, engaged ÷ bypassed,
   which cancels the chain). Slider detents that do not return exactly would do it; the record
   run's repeats measure it.

## Pre-flight before the record run (owner: "I don't want to run hours then toss it")

Code: 2026-10-22 commit after 10c9bca (send/return limit, retakes, reference coherence check).

1. Double-click the Desktop shortcut. **New session**: 48 kHz, **Sweeps only**.
2. Setup page: return channel 3; note the return input's gain knob position ("Return input gain").
3. Calibrate C1-C4. Go on only if:
   - C2 says "Limit on the send side ... about -9 dBFS sent" (or reports a different limit: then
     stop and send it to me);
   - C2, C3 and C4 all read "band coherence >= 0.99x" (no "Disturbed": redo that step).
4. **Repeatability check (5 min):** Plan page -> Load record run -> run only the first three steps:
   flat, flat_r2, flat_r3, then stop (do not press Ready for the next step). Then run
   `uv run python scripts/ge7_dataset.py <session>` (or ask me).
   - flat, flat_r2 and flat_r3 within ~0.05 dB of each other: go, run the whole record run.
   - spread ~0.3 dB like today's train vs val: stop. The repeatability floor would swamp the fit;
     we find the cause first (warm-up drift, the bypass/engage switch, slider contacts).
5. Optional, not needed for the GE-7: the ProRMP level-knob test (C2 with the knob 6 dB lower)
   to learn whether the send limit is before the knob (1824c Out 1 / ProRMP input) or after it
   (ProDI input). It matters for gain staging later (the Klon), not for this run.

Expect "Signal-to-noise" to warn on deep-cut presets (all_down): their quietest sweep sits near the
noise; the louder sweeps carry the estimate (the dataset weights each sweep by its energy).
