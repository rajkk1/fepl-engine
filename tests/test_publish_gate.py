"""
Whether a new plan should replace the published one.

Two failures have to be avoided at once. Publishing is destructive, because
actions-gh-pages replaces the whole directory - so a fixture-blind plan deletes
the good one rather than sitting beside it. But refusing to publish silently
serves whatever is already there, and the published plan carries no expiry: that
is how a GW3 plan came to be read as current days after GW3 had been played.

So the rule is by gameweek. A good plan stays visible only while it belongs to
the gameweek still in play.
"""
import json

import pytest

from publish_gate import should_publish, _load, main


def _plan(gw, degraded=False):
    return {"gameweek": gw, "degraded": degraded}


# ------------------------------------------------------------- the core rule


def test_a_sound_plan_always_publishes():
    for published in (None, _plan(3), _plan(4), _plan(9, degraded=True)):
        ok, _ = should_publish(_plan(4), published)
        assert ok


def test_a_good_plan_for_the_current_gameweek_is_kept():
    """The requested behaviour: still this gameweek, so still usable."""
    ok, why = should_publish(_plan(4, degraded=True), _plan(4))
    assert ok is False
    assert "GW4" in why


def test_a_good_plan_for_a_past_gameweek_is_replaced():
    """
    Once the published plan belongs to a gameweek that has gone, it is worse
    than an honest degraded one - a degraded plan says so, a stale one does not.
    """
    ok, why = should_publish(_plan(4, degraded=True), _plan(3))
    assert ok is True
    assert "stale" in why


def test_a_degraded_plan_publishes_when_nothing_is_live():
    ok, _ = should_publish(_plan(4, degraded=True), None)
    assert ok is True


def test_a_degraded_plan_replaces_another_degraded_one():
    """No good plan is being protected, and a rerun may have more results."""
    ok, _ = should_publish(_plan(4, degraded=True), _plan(4, degraded=True))
    assert ok is True


def test_a_missing_gameweek_does_not_hold_back_publishing():
    """Better to publish than to protect a plan we cannot date."""
    ok, _ = should_publish({"degraded": True}, _plan(4))
    assert ok is True
    ok, _ = should_publish(_plan(4, degraded=True), {"degraded": False})
    assert ok is True


# ------------------------------------------------------------------ plumbing


def test_an_unreadable_published_plan_counts_as_nothing_live(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert _load(str(bad)) is None
    assert _load(str(tmp_path / "absent.json")) is None
    assert _load(None) is None


def test_a_json_list_is_not_treated_as_a_plan(tmp_path):
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]")
    assert _load(str(p)) is None


def test_cli_emits_github_output_pairs(tmp_path, monkeypatch, capsys):
    new = tmp_path / "new.json"
    new.write_text(json.dumps(_plan(4, degraded=True)))
    old = tmp_path / "old.json"
    old.write_text(json.dumps(_plan(4)))

    monkeypatch.setattr("sys.argv", ["publish_gate.py", str(new), str(old)])
    assert main() == 0
    out = capsys.readouterr().out
    assert "publish=false" in out
    assert "reason=" in out


def test_cli_refuses_to_publish_when_no_plan_was_generated(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv",
                        ["publish_gate.py", str(tmp_path / "missing.json")])
    assert main() == 0
    out = capsys.readouterr().out
    assert "publish=false" in out
    assert "no plan was generated" in out


# --------------------------------------------------------- staleness is visible


def test_the_exported_plan_is_dated():
    """
    Without a timestamp, "we kept the last good plan" and "we are serving a
    gameweek that has been played" are indistinguishable to a reader.
    """
    import inspect

    import weekly_manager

    src = inspect.getsource(weekly_manager.main)
    assert '"generated_at"' in src
    assert '"gameweek"' in src


def test_the_workflow_uses_the_gate():
    import pathlib

    wf = pathlib.Path(".github/workflows/weekly_optimizer.yml").read_text()
    assert "publish_gate.py" in wf
    assert "steps.quality.outputs.publish == 'true'" in wf
    gate = wf.index("Should this replace the published plan?")
    assert gate < wf.index("Send Discord notification")
    assert gate < wf.index("Deploy JSON to GitHub Pages")
