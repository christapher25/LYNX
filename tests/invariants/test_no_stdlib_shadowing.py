"""
tests/invariants/test_no_stdlib_shadowing.py

Invariant: no project module may share a name with a standard-library module.

`python data/ingest_x.py` puts data/ first on sys.path, so a data/calendar.py silently replaces the stdlib `calendar` for
everything imported afterwards, including the standard library's own `email` and pandas. On 2026-09-24 that made every
ingester fail with "No module named 'config'" (data/calendar.py imports config, which is only on the path once the script has
started). The commands the project documents (`python data/ingest_fno_bhavcopy.py --days 3`, the DII capture) would all have
broken, and a scheduled capture would have failed silently.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
# Every directory a script in this repo can be launched from puts its own path first on sys.path.
SCRIPT_DIRS = [ROOT, ROOT / "data", ROOT / "scripts", ROOT / "features", ROOT / "features" / "blocks", ROOT / "signals",
               ROOT / "eval", ROOT / "costs"]


def _modules(directory: Path):
    if not directory.is_dir():
        return []
    return [p.stem for p in directory.glob("*.py")] + [p.name for p in directory.iterdir() if p.is_dir() and (p / "__init__.py").exists()]


@pytest.mark.parametrize("directory", SCRIPT_DIRS, ids=lambda p: str(p.relative_to(ROOT)) or ".")
def test_no_module_shadows_the_standard_library(directory):
    shadows = sorted(name for name in _modules(directory) if name in sys.stdlib_module_names)
    assert not shadows, f"{directory.relative_to(ROOT)} contains {shadows}, which would replace the stdlib module of that name"


INGESTERS = sorted((ROOT / "data").glob("ingest_*.py"))


_LAUNCH_ENVIRONMENT = """
import runpy, sys
sys.path.insert(0, sys.argv[1])            # what `python data/x.py` does: the script's own directory goes first
runpy.run_path(sys.argv[2], run_name="imported_not_main")   # module top level only: imports and constants, never main()
"""


@pytest.mark.parametrize("script", INGESTERS, ids=lambda p: p.name)
def test_every_ingester_imports_cleanly_under_script_launch_conditions(script):
    """
    The documented usage is `python data/<script>.py`. Reproduce its import environment WITHOUT running it (some ingesters
    have no --help and would start a real crawl), so anything shadowed by the script's directory fails here.
    """
    result = subprocess.run([sys.executable, "-c", _LAUNCH_ENVIRONMENT, str(script.parent), str(script)],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-400:]
