"""Capture engine: play one signal through the rig and record the return and the loopback."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

PRE_ROLL_S = (
    0.5  # silence before the signal: lets the converters settle and gives a noise-floor window
)
TAIL_S = 0.5  # silence after: catches the round-trip latency and any pedal decay


@dataclass
class Take:
    """One recorded pass of a signal.

    Attributes:
        sr: Sample rate in Hz.
        play: What was sent, shape (n, 2): [:, 0] to the pedal (Out 1), [:, 1] to the loopback
            (Out 2).
        rec: What came back, shape (n, 2): [:, 0] the pedal return (In 1), [:, 1] the loopback
            (In 2).
        pre_roll: Samples of silence before the signal starts.
        signal_len: Samples of signal (the dry WAV's length).
        send_level_db: Digital gain applied to the signal before it was sent.
        xruns: Blocks with a driver under/overflow (0 = no dropouts).
        aborted: True if the take was stopped early.
        elapsed_s: Wall-clock duration.
    """

    sr: int
    play: np.ndarray
    rec: np.ndarray
    pre_roll: int
    signal_len: int
    send_level_db: float
    xruns: int
    aborted: bool
    elapsed_s: float


class CaptureEngine:
    """Owns the play/record buffers of the take in progress; the GUI reads them for its live views.

    All buffers are allocated before the stream starts. The audio callback only copies slices, so it
    does not allocate arrays or take locks (Python still creates small view objects; that is cheap
    and bounded).
    """

    def __init__(self, backend, block_size: int = 1024) -> None:
        self.backend = backend
        self.block_size = block_size
        self.play = np.zeros((0, 2), dtype=np.float32)
        self.rec = np.zeros((0, 2), dtype=np.float32)
        self.sr = 48_000
        self.pre_roll = 0
        self._pos = 0
        self._abort = False
        self.running = False

    @property
    def position(self) -> int:
        """Frames played and recorded so far in the current take."""
        return self._pos

    @property
    def total(self) -> int:
        return len(self.play)

    def abort(self) -> None:
        """Stop the take at the next block."""
        self._abort = True

    def _callback(self, out: np.ndarray, inp: np.ndarray) -> bool:
        p = self._pos
        end = min(p + len(out), len(self.play))
        k = end - p
        out[:k] = self.play[p:end]
        out[k:] = 0.0
        self.rec[p:end] = inp[:k]
        self._pos = end
        return end >= len(self.play) or self._abort

    def run(
        self,
        signal: np.ndarray,
        sr: int,
        send_level_db: float,
        route: str = "both",
    ) -> Take:
        """Play *signal* and record. Blocks until the take ends (run it off the GUI thread).

        Args:
            signal: Dry signal, shape (N,), float32, in [-1, 1].
            sr: Sample rate in Hz.
            send_level_db: Digital gain applied before sending (the profile's send level).
            route: "both" sends to the pedal and the loopback, "loop" to the loopback only,
                "send" to the pedal only.

        Returns:
            The recorded take.
        """
        pre = int(round(PRE_ROLL_S * sr))
        tail = int(round(TAIL_S * sr))
        n = pre + len(signal) + tail
        gain = np.float32(10 ** (send_level_db / 20))
        play = np.zeros((n, 2), dtype=np.float32)
        if route in ("both", "send"):
            play[pre : pre + len(signal), 0] = signal * gain
        if route in ("both", "loop"):
            play[pre : pre + len(signal), 1] = signal * gain
        self.play = play
        self.rec = np.zeros((n, 2), dtype=np.float32)
        self.sr = sr
        self.pre_roll = pre
        self._pos = 0
        self._abort = False
        self.running = True
        t0 = time.perf_counter()
        try:
            xruns = self.backend.run(sr, self.play, self._callback, self.block_size)
        finally:
            self.running = False
        return Take(
            sr=sr,
            play=self.play,
            rec=self.rec,
            pre_roll=pre,
            signal_len=len(signal),
            send_level_db=send_level_db,
            xruns=xruns,
            aborted=self._abort,
            elapsed_s=time.perf_counter() - t0,
        )
