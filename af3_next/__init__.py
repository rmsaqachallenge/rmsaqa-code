"""Experiment entry points using the canonical spatial_af3 package."""

import sys
from pathlib import Path

_code_root = str(Path(__file__).resolve().parents[1])
if _code_root not in sys.path:
    sys.path.insert(0, _code_root)
