"""``uv run python -m pedal_model.rig`` launches the capture GUI (demo unless --hardware)."""

import sys

# Guarded: the audio stream runs in a child process (spawn), which imports this module again.
if __name__ == "__main__":
    from pedal_model.rig.gui.app import main

    sys.exit(main())
