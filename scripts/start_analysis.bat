@echo off
rem Open the PedalDSP GE-7 analysis GUI: trains the block model on a capture session, live.
rem Double-click it, or the "PedalDSP Analysis" shortcut on the Desktop that points here.
title PedalDSP analysis
cd /d "%~dp0.."
uv run --extra rig python -m pedal_model.analysis
if errorlevel 1 (
    echo.
    echo The analysis exited with an error. Read the messages above, then press a key to close.
    pause >nul
)
