"""Audio backends: the real interface over ASIO, and a simulated rig for demos and tests.

Both backends run one take with the same contract: the engine's ``callback(out, inp)`` is called
once per block with an output block to fill (2 channels: send, loop) and the input block just
recorded (2 channels:
return, loop). The callback returns True when the take is finished. ``run`` blocks until then and
returns the number of blocks with a driver under/overflow (0 means no dropouts).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from pedal_model.rig.config import ChannelMap

BlockCallback = Callable[[np.ndarray, np.ndarray], bool]
PedalFn = Callable[[np.ndarray, int], np.ndarray]


def _identity_pedal(x: np.ndarray, sr: int) -> np.ndarray:
    return x


PEDAL_STATES = ("cable", "bypass", "engaged")  # pedal out of the chain / in, bypassed / engaged


@dataclass
class SimulatedRig:
    """Analog behaviour of the simulated chain. Defaults are the real rig's measured numbers where
    one exists; the rest are marked illustrative.

    Attributes:
        latency_ms: Round trip out → in. 0.98 ms = the 47 samples at 48 kHz in
            outputs/notaklon_delay.
        reamp_db: ProRMP insertion loss (docs/notaklon_capture_plan.md: −12.4 dB).
        reamp_tilt_db: High-frequency rise of the chain, a first-order shelf centred on
            ``reamp_tilt_hz``. Size from the Studio One-era re-amp-only captures (about +4 dB); the
            shape is illustrative. It is what a scalar chain-gain normalisation gets wrong.
        return_gain_db: ProDI + In 1 preamp gain, the knob calibration asks you to set.
        loop_gain_db: Out 2 → In 2 cable with In 2 gain at minimum.
        noise_dbfs: RMS noise floor of each input.
        clip_dbfs: Converter full scale; anything above is hard-clipped, as a real ADC would.
        bypass_gain_db, bypass_hp_hz: The pedal's buffers when bypassed (illustrative).
        engaged_gain_db: Engaged path vs bypassed at the flat setting (illustrative).
        dropout_prob: Chance that each 1024-sample output block is silently lost (the driver
            plays silence and reports nothing), as seen on the 1824c on 2026-10-18.
    """

    latency_ms: float = 0.98
    reamp_db: float = -12.4
    reamp_tilt_db: float = 4.0
    reamp_tilt_hz: float = 3000.0
    return_gain_db: float = 12.0
    loop_gain_db: float = -0.5
    noise_dbfs: float = -95.0
    clip_dbfs: float = 0.0
    bypass_gain_db: float = -0.3
    bypass_hp_hz: float = 8.0
    engaged_gain_db: float = 0.4
    dropout_prob: float = 0.0


def _first_order(b_s: tuple[float, float], a_s: tuple[float, float], x: np.ndarray, sr: int):
    """Filter x by the analog first-order section b_s / a_s (bilinear transform)."""
    from scipy.signal import bilinear, lfilter  # lazy: keeps the audio child process light

    b, a = bilinear(b_s, a_s, fs=sr)
    return lfilter(b, a, x).astype(np.float32)


class SimulatedBackend:
    """A rig that exists only in software.

    It never opens a sound card, so it is safe to run anywhere.

    The whole return is computed when the take starts (the pedal model is applied to the complete
    send), then streamed block by block at ``speed`` times real time so the GUI's live views behave
    as they will with hardware.
    """

    name = "Simulated rig (no audio is played)"
    is_hardware = False

    def __init__(self, rig: SimulatedRig | None = None, speed: float = 8.0, seed: int = 7) -> None:
        self.rig = rig or SimulatedRig()
        self.speed = speed  # <= 0 means as fast as possible (tests)
        self.pedal: PedalFn = _identity_pedal
        self.pedal_state = "engaged"
        self._rng = np.random.default_rng(seed)

    def set_pedal(self, fn: PedalFn | None) -> None:
        """Model of the pedal as currently set (the demo's stand-in for turning the knobs)."""
        self.pedal = fn or _identity_pedal

    def set_pedal_state(self, state: str) -> None:
        """The demo's stand-in for moving cables and pressing the footswitch (``PEDAL_STATES``)."""
        if state not in PEDAL_STATES:
            raise ValueError(f"pedal state must be one of {PEDAL_STATES}, not {state!r}")
        self.pedal_state = state

    def _chain_in(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Send → ProRMP output: insertion loss and the high-frequency tilt."""
        r = self.rig
        x = x * np.float32(10 ** (r.reamp_db / 20))
        if r.reamp_tilt_db:
            g = 10 ** (r.reamp_tilt_db / 20)
            w = 2 * np.pi * r.reamp_tilt_hz
            wz, wp = w / np.sqrt(g), w * np.sqrt(g)
            x = _first_order((wp / wz, wp), (1.0, wp), x, sr)  # 1 at DC, g at high frequency
        return x

    def _through_pedal(self, x: np.ndarray, sr: int) -> np.ndarray:
        r = self.rig
        if self.pedal_state == "cable":
            return x
        if r.bypass_hp_hz > 0:
            w = 2 * np.pi * r.bypass_hp_hz
            x = _first_order((1.0, 0.0), (1.0, w), x, sr)
        x = x * np.float32(10 ** (r.bypass_gain_db / 20))
        if self.pedal_state == "engaged":
            x = self.pedal(x, sr) * np.float32(10 ** (r.engaged_gain_db / 20))
        return x

    def _simulate(self, play: np.ndarray, sr: int) -> np.ndarray:
        r = self.rig
        n = len(play)
        if r.dropout_prob > 0:
            play = play.copy()
            for a in range(0, n, 1024):
                if self._rng.random() < r.dropout_prob:
                    play[a : a + 1024] = 0.0
        lag = int(round(r.latency_ms * 1e-3 * sr))
        noise = 10 ** (r.noise_dbfs / 20)
        ret = self._through_pedal(self._chain_in(play[:, 0], sr), sr)
        ret = ret * 10 ** (r.return_gain_db / 20)
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


@contextmanager
def _com_apartment():
    """Initialise COM on this thread for the duration (Windows only).

    An ASIO driver is a COM object, and PortAudio loads it on the thread that opens the stream. The
    GUI opens streams on a worker thread, where COM is not initialised, and the driver then fails
    with "Failed to load ASIO driver" (seen on the 1824c, 2026-10-18; the same open works from the
    main thread).
    """
    if sys.platform != "win32":
        yield
        return
    import ctypes

    hr = ctypes.windll.ole32.CoInitializeEx(None, 0x2)  # COINIT_APARTMENTTHREADED
    try:
        yield
    finally:
        if hr in (0, 1):  # S_OK / S_FALSE: this call owns an initialisation, so undo it
            ctypes.windll.ole32.CoUninitialize()


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

        with (
            _com_apartment(),
            sd.Stream(
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
            ),
        ):
            finished.wait()
        return xruns


# ── Out-of-process streaming ───────────────────────────────────────────────── #
#
# The audio callback is Python, so it needs the GIL. In the GUI process the live views hold it often
# enough that the callback misses the driver's deadline: on 2026-10-18 about 30 % of the blocks of
# every Run-page take came back silent, with no dropout reported, while the short calibration takes
# (no live views) were perfect. A child process owns the stream instead and copies straight between
# shared-memory play/record buffers, so nothing in the GUI process can delay it.


def _raise_priority() -> None:
    if sys.platform == "win32":
        import ctypes

        HIGH_PRIORITY_CLASS = 0x80
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), HIGH_PRIORITY_CLASS)


def _child_main(factory, kwargs, sr, play_name, rec_name, n, block_size, pos, stop, conn) -> None:
    """Child process: open the shared buffers, build the backend, stream one take."""
    import traceback
    from multiprocessing import shared_memory

    shm_p = shared_memory.SharedMemory(name=play_name)
    shm_r = shared_memory.SharedMemory(name=rec_name)
    try:
        _raise_priority()
        play = np.ndarray((n, 2), dtype=np.float32, buffer=shm_p.buf)
        rec = np.ndarray((n, 2), dtype=np.float32, buffer=shm_r.buf)
        backend = factory(**kwargs)
        p = 0

        def cb(out: np.ndarray, inp: np.ndarray) -> bool:
            nonlocal p
            end = min(p + len(out), n)
            k = end - p
            out[:k] = play[p:end]
            out[k:] = 0.0
            rec[p:end] = inp[:k]
            p = end
            pos.value = end
            return end >= n or bool(stop.value)

        xruns = backend.run(sr, play, cb, block_size)
        conn.send(("ok", xruns))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        play = rec = None  # release the views before closing the mappings
        shm_p.close()
        shm_r.close()


class ProcessBackend:
    """Runs another backend's stream in a child process, on shared-memory play/record buffers.

    ``factory(**kwargs)`` builds the real backend inside the child, so both must be picklable (a
    class and plain settings). The engine sees the same buffers through shared memory, so its
    position and live views work as before.
    """

    out_of_process = True

    def __init__(self, factory, kwargs: dict, name: str, is_hardware: bool) -> None:
        self.factory = factory
        self.kwargs = kwargs
        self.name = name
        self.is_hardware = is_hardware

    def run_shared(
        self,
        sr: int,
        play_name: str,
        rec_name: str,
        n: int,
        block_size: int,
        progress: Callable[[int], None],
        aborted: Callable[[], bool],
    ) -> int:
        """Stream one take in a child process; returns its dropout count.

        Args:
            play_name, rec_name: Shared-memory blocks holding (n, 2) float32 play / record buffers.
            progress: Called with the frame position while the take runs (from this thread).
            aborted: Polled; when it returns True the child stops at its next block.
        """
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        pos = ctx.Value("q", 0, lock=False)
        stop = ctx.Value("b", 0, lock=False)
        recv, send = ctx.Pipe(duplex=False)
        args = (self.factory, self.kwargs, sr, play_name, rec_name, n, block_size, pos, stop, send)
        child = ctx.Process(target=_child_main, args=args, daemon=True)
        child.start()
        result = None
        try:
            while result is None:
                progress(int(pos.value))
                if aborted():
                    stop.value = 1
                if recv.poll(0.02):
                    result = recv.recv()
                elif not child.is_alive():
                    if recv.poll(0.2):
                        result = recv.recv()
                    break
        finally:
            child.join(5.0)
            if child.is_alive():
                child.kill()
        progress(int(pos.value))
        if result is None:
            raise RuntimeError(f"The audio process exited with no result (code {child.exitcode}).")
        if result[0] == "error":
            raise RuntimeError("The audio process failed:\n" + result[1])
        return int(result[1])
