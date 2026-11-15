"""Lets the tests import polaris_devices and their helpers (pytest runs in importlib mode)."""

import sys
from pathlib import Path

HERE = Path(__file__).parent
for folder in (HERE, HERE / 'tests'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
