#!/usr/bin/env python3
"""check_links — thin wrapper around `tools.check_docs`.

Kept as a separate entry point so `scripts/check_links.py` can be run with the
system `python3` (no venv, no third-party deps) while the logic lives in the
importable `tools.check_docs` module (which the test suite also uses).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.check_docs import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
