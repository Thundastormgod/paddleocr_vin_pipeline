"""
Pytest configuration - repository root.

The test suite imports the package as ``src.vin_ocr.*`` (absolute, from the
repository root). That resolves only when the repo root is on ``sys.path``,
which happens implicitly with ``python -m pytest`` (Python prepends the CWD)
but NOT with a bare ``pytest`` call.

Without this file, `pytest` and `python -m pytest` behave differently and CI
fails with ModuleNotFoundError while local runs pass. Inserting the root here
makes both invocations equivalent.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
