"""pytest configuration: make the `matcher` package importable and load fixtures.

Mirrors the runtime convention used by the step scripts (which append `src/` to
sys.path), so tests import `from matcher.hybrid_core import ...`.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture
def frames():
    """Return the list of synthetic frames (each a list of node dicts)."""
    path = os.path.join(ROOT, "tests", "fixtures", "synthetic_frames.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["frames"]
