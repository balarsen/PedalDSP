"""Audio backends: the real interface over ASIO, and a simulated rig for demos and tests.

Both backends run one take with the same contract: the engine's ``callback(out, inp)`` is called
once per
block with an output block to fill (2 channels: send, loop) and the input block just recorded (2
channels:
return, loop). The callback returns True when the take is finished. ``run`` blocks until then and
returns
the number of blocks with a driver under/overflow (0 means no dropouts).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pedal_model.rig.config import ChannelMap

BlockCallback = Callable[[np.ndarray, np.ndarray], bool]
PedalFn = Callable[[np.ndarray, int], np.ndarray]


def _identity_pedal(x: np.ndarray, sr: int) -> np.ndarray:
    return x


@dataclass
class SimulatedRig:
    """Analog behaviour of the simulated chain. Defaults are the real rig's measured numbers.

    Attributes:
        latency_ms: Round trip out → in. 0.98 ms = the 47 samples at 48 kHz in
        outputs/notaklon_delay.
        reamp_db: ProRMP insertion loss (docs/notaklon_capture_plan.md: −12.4 dB).
        return_gain_db: ProDI + In 1 preamp gain, the knob calibration asks you to set.
        loop_gain_db: Out 2 → In 2 cable with In 2 gain at minimum.
        noise_dbfs: RMS noise floor of each input.
        clip_dbfs: Converter full scale; anything above is hard-clipped, as a real ADC would.
    """

    latency_ms: float = 0.98
    reamp_db: float = -12.4
    return_gain_db: float = 12.0
    loop_gain_db: float = -0.5
    noise_dbfs: float = -95.0
    clip_dbfs: float = 0.0


class SimulatedBackend:
    """A rig that exists only in software.

    It never opens a sound card, so it is safe to run anywhere.

    The whole return is computed when the take starts (the pedal model is applied to the complete
    send),
    then streamed block by block at ``speed`` times real time so the GUI's live views behave as they
    will
    with hardware.
    """

    name = "Simulated rig (no audio is played)"
    is_hardware = False

    def __init__(self, rig: SimulatedRig | None = None, speed: float = 8.0, seed: int = 7) -> None:
        self.rig = rig or SimulatedRig()
        self.speed = speed  # <= 0 means as fast as possible (tests)
        self.pedal: PedalFn = _identity_pedal
        self._rng = np.random.default_rng(seed)

    def set_pedal(self, fn: PedalFn | None) -> None:
        """Model of the pedal as currently set (the demo's stand-in for turning the knobs)."""
        self.pedal = fn or _identity_pedal

    def _simulate(self, play: np.ndarray, sr: int) -> np.ndarray:
        r = self.rig
        n = len(play)
        lag = int(round(r.latency_ms * 1e-3 * sr))
        noise = 10 ** (r.noise_dbfs / 20)
        ret = self.pedal(play[:, 0] * 10 ** (r.reamp_db / 20), sr) * 10 ** (r.return_gain_db / 20)
        loop = play[:, 1] * 10 ** (r.loop_gain_db / 20)
        inp = np.zeros((n, 2), dtype=np.float32)
        inp[lag:, 0] = ret[: n - lag]
        inp[lag:, 1] = loop[: n - lag]
        inp += (noise * self._rng.standard_normal((n, 2))).astype(np.float32)
        full_scale = 10 ** (r.clip_dbfs / 20)
        np.clip(inp, -full_scale, full_scale, out=inp)
        return inp

    def run(self, sr: int, play: np.ndarray, callback: BlockCallback, block_size: int) -> int:
        """Stream one take. ``play`` is the full (n, 2) output the engine will send."""
        inp = self._simulate(play, sr)
        out = np.zeros((block_size, 2), dtype=np.float32)
        t0 = time.perf_counter()
        pos = 0
        while True:
            k = min(block_size, len(inp) - pos)
            if k <= 0:
                block_in = np.zeros((block_size, 2), dtype=np.float32)
                done = callback(out, block_in)
            else:
                done = callback(out[:k], inp[pos : pos + k])
            pos += max(k, 0)
            if done:
                return 0
            if self.speed > 0:
                due = t0 + pos / (sr * self.speed)
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)


def _sounddevice():
    """Import sounddevice with its ASIO-enabled PortAudio (env var set before import)."""
    os.environ.setdefault("SD_ENABLE_ASIO", "1")
    import sounddevice as sd

    return sd


def list_asio_devices() -> list[tuple[int, str, int, int]]:
    """(index, name, max inputs, max outputs) per ASIO device. Opens nothing, plays nothing."""
    sd = _sounddevice()
    out = []
    for i, d in enumerate(sd.query_devices()):
        if sd.query_hostapis(d["hostapi"])["name"] == "ASIO":
            out.append((i, d["name"], d["max_input_channels"], d["max_output_channels"]))
    return out


class AsioBackend:
    """The real interface: one full-duplex ASIO stream, so send and return share one clock."""

    is_hardware = True

    def __init__(self, device_name: str, channels: ChannelMap) -> None:
        self.device_name = device_name
        self.channels = channels
        self.name = f"ASIO: {device_name}"

    def device_index(self) -> int:
        for idx, name, _, _ in list_asio_devices():
            if name == self.device_name:
                return idx
        raise RuntimeError(
            f"ASIO device {self.device_name!r} not found. Is the 1824c connected, and is Studio "
            "One "
            "(or another ASIO program) closed?"
        )

    def run(self, sr: int, play: np.ndarray, callback: BlockCallback, block_size: int) -> int:
        sd = _sounddevice()
        idx = self.device_index()
        ch = self.channels
        in_sel = [ch.return_in - 1, ch.loop_in - 1]
        out_sel = [ch.send_out - 1, ch.loop_out - 1]
        finished = threading.Event()
        xruns = 0

        def cb(indata, outdata, frames, time_info, status) -> None:
            nonlocal xruns
            if status:
                xruns += 1
            if callback(outdata, indata):
                raise sd.CallbackStop

        with sd.Stream(
            device=(idx, idx),
            samplerate=sr,
            blocksize=block_size,
            channels=(2, 2),
            dtype="float32",
            extra_settings=(
                sd.AsioSettings(channel_selectors=in_sel),
                sd.AsioSettings(channel_selectors=out_sel),
            ),
            callback=cb,
            finished_callback=finished.set,
        ):
            finished.wait()
        return xruns
