"""
The experiment harness must still run.

These scripts are not exercised by CI - they are slow and they hit the network -
so nothing else would notice them rotting. The failure mode is quiet and
annoying: a question that was already settled has to be re-argued from scratch
because the code that settled it no longer imports.

So this checks the cheap half: they parse, they import, their arguments still
resolve, and the report can read a sweep's output. It does NOT check the
measurement itself; that lives in `test_hit_price.py`, which pins the numbers
and the reasoning.
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ["hit_cap_sweep.py", "hit_cap_report.py", "validate_odds_source.py"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_script_is_present_and_parses(name):
    import py_compile

    path = ROOT / "experiments" / name
    assert path.exists(), f"{name} has gone missing"
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_script_can_be_invoked(name):
    """`--help` imports the module and builds the parser, which is where a
    stale keyword argument or a renamed import shows up."""
    r = subprocess.run([sys.executable, str(ROOT / "experiments" / name), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "usage:" in r.stdout


def test_the_sweep_and_the_report_agree_on_the_format(tmp_path):
    """
    The sweep writes and the report reads; nothing type-checks the seam. A
    minimal handmade run exercises it without replaying a season.
    """
    history = [
        {"gw": gw, "points": 50, "hits": 0, "net": 50, "cumulative": 50 * gw,
         "transfers": 0, "bank": 1.0, "captain": 1, "leader": 1, "autosubs": 0,
         "squad_value": 100.0, "in": [], "out": [], "squad": list(range(1, 16))}
        for gw in range(1, 6)
    ]
    blob = {"label": "cap", "runs": {
        f"2024-25|{cap}": {"season": "2024-25", "cap": cap, "total": 250.0,
                           "hits": cap, "transfers": cap, "history": history}
        for cap in (0, 2)
    }}
    p = tmp_path / "runs.json"
    p.write_text(json.dumps(blob))

    r = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "hit_cap_report.py"),
         "--results", str(p)],
        capture_output=True, text=True, timeout=600, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    # The validity section has to come first, and has to actually check.
    assert "1. VALIDITY" in r.stdout
    assert "GW1 squad identical" in r.stdout
    assert "2. HEADLINE" in r.stdout


def test_the_harness_records_which_arm_it_describes():
    """
    Sections 3 and 4 describe one arm. They first defaulted to the most
    permissive arm in the sweep rather than the engine's shipped setting, which
    silently produced numbers that disagreed with the write-up. The arm must be
    stated in the output.
    """
    src = (ROOT / "experiments" / "hit_cap_report.py").read_text()
    assert "sections 3 and 4 examine" in src
    assert "NOT the shipped" in src
