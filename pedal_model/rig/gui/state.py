"""Shared state of the GUI: rig config, profile, backend, engine, session, calibration and plan."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from pedal_model.rig.backends import AsioBackend, ProcessBackend, SimulatedBackend
from pedal_model.rig.config import RigConfig
from pedal_model.rig.engine import CaptureEngine
from pedal_model.rig.profiles import Preset, Profile, load_profile
from pedal_model.rig.session import Session

DEMO_SPEEDS = (1.0, 4.0, 16.0, 64.0)


class AppState(QObject):
    session_changed = Signal()
    calibration_changed = Signal()
    rig_changed = Signal()
    plan_changed = Signal()
    cables_changed = Signal(dict)  # cable id -> "ok" / "fail" / "idle"

    def __init__(self) -> None:
        super().__init__()
        self.rig = RigConfig.load()
        self.profile: Profile = load_profile("ge7")
        self.simulated = True
        self.demo_speed = 16.0
        self.sample_rate = 48_000
        self.signal_kind = "quick"
        self.session: Session | None = None
        self.plan: list[Preset] = self.profile.presets()
        self.cables: dict[str, str] = {}
        self.hardware_confirmed = False  # user OK'd playing test signals out of Out 1/2
        self.engine = CaptureEngine(
            self._make_backend(), self.rig.block_size, self.rig.loop_send_db
        )

    def _make_backend(self):
        if self.simulated:
            return SimulatedBackend(speed=self.demo_speed)
        # The stream runs in its own process so the GUI can never starve the audio callback.
        return ProcessBackend(
            AsioBackend,
            {"device_name": self.rig.asio_device, "channels": self.rig.channels},
            f"ASIO: {self.rig.asio_device}",
            is_hardware=True,
        )

    def set_mode(self, simulated: bool, demo_speed: float) -> None:
        self.simulated, self.demo_speed = simulated, demo_speed
        self.engine = CaptureEngine(
            self._make_backend(), self.rig.block_size, self.rig.loop_send_db
        )
        self.hardware_confirmed = False
        self.rig_changed.emit()

    def set_session(self, session: Session | None) -> None:
        self.session = session
        if session is not None:
            self.sample_rate = session.sample_rate
            self.signal_kind = session.data["signal_kind"]
            self.plan = (
                session.presets()
            )  # the session's own copies: map and custom steps resume too
            cal = session.data.get("calibration") or {}
            self.set_cables_from_calibration(cal)
        self.session_changed.emit()
        self.plan_changed.emit()

    def set_cables_from_calibration(self, cal: dict) -> None:
        loop, chain = cal.get("loopback"), cal.get("chain")
        cables = {}
        if loop is not None:
            ok = all(a["status"] != "fail" for a in loop["advice"])
            cables.update(usb="ok", loop="ok" if ok else "fail")
        if chain is not None:
            ok = all(a["status"] != "fail" for a in chain["advice"])
            for c in ("send", "inst_in", "inst_out", "mic"):
                cables[c] = "ok" if ok else "fail"
        self.cables = cables
        self.cables_changed.emit(cables)

    @property
    def calibration(self) -> dict | None:
        return self.session.data.get("calibration") if self.session else None
