"""``uv run python -m pedal_model.analysis`` opens the GE-7 analysis (training) GUI."""

import sys

if __name__ == "__main__":
    from pedal_model.analysis.app import main

    sys.exit(main())
