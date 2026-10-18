"""``uv run python -m pedal_model.rig`` launches the capture GUI (demo unless --hardware)."""

import sys

from pedal_model.rig.gui.app import main

sys.exit(main())
