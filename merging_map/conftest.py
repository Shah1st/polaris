"""Lets the tests import fusion_carte from this folder (pytest runs in importlib mode)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
