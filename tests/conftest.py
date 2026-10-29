"""Tests run against this checkout (not an installed copy elsewhere)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

# Never let a developer's own ~/.compositional_suite/libraries add-ons leak into tests.
import tempfile  # noqa: E402
os.environ["COMPOSITIONAL_SUITE_LIBRARY_DIR"] = tempfile.mkdtemp(prefix="pf-lib-")
