@echo off
rem Start the PedalDSP capture rig GUI in hardware (ASIO) mode.
rem Double-click it, or the "PedalDSP Rig (hardware)" shortcut on the Desktop that points here.
rem The console window stays open while the GUI runs: errors and tracebacks show up in it.
title PedalDSP capture rig (hardware)
cd /d "%~dp0.."
uv run --extra rig python -m pedal_model.rig --hardware
if errorlevel 1 (
    echo.
    echo The rig exited with an error. Read the messages above, then press a key to close.
    pause >nul
)
