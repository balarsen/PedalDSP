#!/usr/bin/env python3
"""generate_signal.py — thin CLI wrapper around pedal_model.signals.generate.

Examples:
  python generate_signal.py --signal both --output data/signals/
  python generate_signal.py --signal train --sr 96000 --seed-train 1234
  python generate_signal.py --signal val   --output data/signals/
  python generate_signal.py --from-manifest data/signals/train_signal_v1.json
  python generate_signal.py --from-manifest data/signals/val_signal_v1.json --output /tmp/
"""

from pedal_model.signals.generate import main

if __name__ == "__main__":
    main()
