"""
The pre-deadline forecast archive, and the scorer that reads it.

The archive exists because two things cannot be recovered once a deadline
passes: what the engine forecast for every player, and FPL's own `ep_next`. So
the tests here are mostly about what may and may not be written - above all,
that nothing overwrites a gameweek whose deadline has gone - and about the
scorer measuring exactly what `backtest` measures, so the live numbers can be
read against the README's. Everything is synthetic; nothing touches the network.
"""
import datetime
import json
import os

import numpy as np
import pytest

import backtest
import forecast_archive as fa
import score_archive

from tests.test_smoke import mini_league  # noqa: F401  (fixture)

UTC = datetime.timezone.utc
DEADLINE = "2026-11-13T18:30:00Z"


def _t(raw):
    return fa.parse_time(raw)


def _bootstrap(gw=11, n_players=6, deadline=DEADLINE, is_next=None, checked=()):
    events = []
    for eid in range(1, 39):
        if eid == gw:
            d = deadline
        else:
            d = (datetime.datetime(2026, 8, 14, 17, 30, tzinfo=UTC)
                 + datetime.timedelta(days=7 * (eid - 1))).isoformat().replace("+00:00", "Z")
        events.append({"id": eid, "deadline_time": d,
                       "is_next": eid == (is_next if is_next is not None else gw),
                       "finished": eid in checked, "data_checked": eid in checked})
    elements = [{
        "id": pid, "web_name": f"P{pid}", "team": 1 + pid % 4,
        "element_type": 1 + pid % 4, "now_cost": 45 + 5 * pid,
        "status": "a" if pid != 3 else "d",
        "chance_of_playing_next_round": None if pid != 3 else 50,
        "selected_by_percent": f"{30.0 - pid:.1f}", "ep_next": f"{pid * 0.5:.1f}",
    } for pid in range(1, n_players + 1)]
    return {"events": events, "elements": elements}


def _matrix(n_players=6, gws=(11, 12, 13)):
    return {pid: {**{g: round(1.0 + pid * 0.4 + 0.1 * (g - gws[0]), 2) for g in gws},
                  **{f"{g}_p_play": 0.9 for g in gws}}
            for pid in range(1, n_players + 1)}


def _history(pid, upto=11):
    return [{"round": r, "total_points": (pid + r) % 7, "minutes": 90 if pid % 2 else 30}
            for r in range(1, upto + 2)]  # includes rounds >= upto, which must be ignored


def _snapshot(forecast_at="2026-11-12T12:00:00Z", deadline=DEADLINE, degraded=False,
              **kw):
    bs = _bootstrap(deadline=deadline, **kw)
    minutes = {(pid, 11): {"p_play": 0.9, "p_60": 0.8, "xmin": 75.0, "n_fixtures": 1}
               for pid in range(1, 7)}
    history = {pid: _history(pid) for pid in range(1, 7)}
    return fa.build_snapshot(bs, _matrix(), 11, [11, 12, 13], _t(forecast_at),
                             minutes=minutes, history=history, degraded=degraded,
                             fixture_model="market", commit="abc1234",
                             run={"id": "42"})


# ------------------------------------------------------------ snapshot format


def test_the_snapshot_records_everything_the_scorer_needs():
    snap = _snapshot()
    for key in ("format", "season", "gameweek", "deadline_time", "forecast_at",
                "commit", "run", "horizon_gws", "degraded", "fixture_model",
                "ep_next_gw", "config", "columns", "rows"):
        assert key in snap, key
    assert snap["season"] == "2026-27"
    assert snap["gameweek"] == 11
    assert snap["deadline_time"] == DEADLINE
    assert snap["forecast_at"] == "2026-11-12T12:00:00Z"
    assert snap["commit"] == "abc1234"
    assert snap["horizon_gws"] == [11, 12, 13]
    assert snap["columns"] == list(fa.SNAPSHOT_COLUMNS)

    rows = {r["id"]: r for r in fa.rows(snap)}
    assert len(rows) == 6
    p = rows[3]
    assert p["xp"] == _matrix()[3][11]
    assert p["xp_horizon"] == [_matrix()[3][g] for g in (11, 12, 13)]
    assert (p["p_play"], p["p_60"], p["xmin"], p["n_fixtures"]) == (0.9, 0.8, 75.0, 1)
    assert p["ep_next"] == 1.5
    assert p["status"] == "d" and p["chance_of_playing_next_round"] == 50
    assert p["selected_by_percent"] == 27.0


def test_trailing_form_only_reads_rounds_before_the_gameweek():
    hist = _history(5)
    ppg, n, mins = fa.trailing_form(hist, 11)
    before = [h for h in hist if h["round"] < 11]
    assert n == len(before) == 10
    assert ppg == pytest.approx(np.mean([h["total_points"] for h in before]))
    assert mins == 5 * 90  # rounds 6..10


def test_trailing_form_matches_the_backtest_baseline(gw_frame):
    """Same definition as `backtest`, or the live numbers are not comparable."""
    target = 6
    expected_ppg = backtest.build_baselines(gw_frame, target)["ppg"]
    expected_pop = backtest.build_eval_population(gw_frame, target, top_n=0)
    for pid, g in gw_frame.groupby("element"):
        hist = [{"round": int(r.GW), "total_points": r.total_points, "minutes": r.minutes}
                for r in g.itertuples()]
        ppg, _, mins = fa.trailing_form(hist, target)
        assert ppg == pytest.approx(expected_ppg[pid])
        assert (mins >= 180) == (pid in expected_pop)


def test_ep_next_is_null_when_fpl_means_another_gameweek():
    """Never let a projection for one gameweek be scored against another."""
    snap = _snapshot(is_next=12)
    assert snap["ep_next_gw"] == 12
    assert all(r["ep_next"] is None for r in fa.rows(snap))


def test_the_file_round_trips_with_one_player_per_line(tmp_path):
    snap = _snapshot()
    ok, _ = fa.write_snapshot(snap, str(tmp_path), now=_t("2026-11-12T12:05:00Z"))
    assert ok
    path = fa.snapshot_path(str(tmp_path), "2026-27", 11)
    assert path.endswith(os.path.join("2026-27", "gw11_forecast.json"))
    text = open(path).read()
    assert json.loads(text) == snap
    player_lines = [ln for ln in text.splitlines() if ln.startswith("    [")]
    assert len(player_lines) == len(snap["rows"])


def test_season_label_comes_from_gw1():
    assert fa.season_label(_bootstrap()) == "2026-27"
    assert fa.season_label({"events": []}, now=datetime.datetime(2027, 3, 1, tzinfo=UTC)) == "2026-27"
    assert fa.season_label({"events": []}, now=datetime.datetime(2027, 8, 1, tzinfo=UTC)) == "2027-28"


def test_the_engine_reports_its_minutes_without_changing_its_forecast(mini_league):  # noqa: F811
    """The archive's hooks into `generate_merv_matrix` must be read-only."""
    from xp_model import generate_merv_matrix

    bootstrap, fixtures, hist = mini_league
    kwargs = dict(bootstrap=bootstrap, fixtures=fixtures, all_history=hist,
                  calibrate=False)
    plain = generate_merv_matrix([5, 6], **kwargs)
    minutes, history = {}, {}
    seen = generate_merv_matrix([5, 6], minutes_out=minutes, history_out=history, **kwargs)
    assert seen == plain
    assert history == hist

    bootstrap = dict(bootstrap, events=[
        {"id": g, "deadline_time": f"2025-09-{10 + g:02d}T10:00:00Z", "is_next": g == 5}
        for g in range(1, 7)])
    snap = fa.build_snapshot(bootstrap, seen, 5, [5, 6],
                             _t("2025-09-14T09:00:00Z"), minutes=minutes, history=history)
    rows = fa.rows(snap)
    assert len(rows) == len(bootstrap["elements"])
    assert any(r["xp"] > 0 for r in rows)
    for r in rows:
        assert r["xp"] == seen[r["id"]][5]
        assert 0.0 <= r["p_60"] <= r["p_play"] <= 1.0
        assert 0.0 <= r["xmin"] <= 90.0 * max(1, r["n_fixtures"])
        assert len(r["xp_horizon"]) == 2


# ---------------------------------------------------- never overwrite a past GW


def test_a_pre_deadline_snapshot_is_written_and_the_latest_one_wins(tmp_path):
    d = str(tmp_path)
    assert fa.write_snapshot(_snapshot("2026-11-11T12:00:00Z"), d,
                             now=_t("2026-11-11T12:01:00Z"))[0]
    ok, why = fa.write_snapshot(_snapshot("2026-11-13T12:00:00Z"), d,
                                now=_t("2026-11-13T12:01:00Z"))
    assert ok and "replaced" in why
    assert fa.load(fa.snapshot_path(d, "2026-27", 11))["forecast_at"] == "2026-11-13T12:00:00Z"


def test_nothing_overwrites_a_gameweek_once_its_deadline_has_passed(tmp_path):
    d = str(tmp_path)
    fa.write_snapshot(_snapshot("2026-11-13T12:00:00Z"), d, now=_t("2026-11-13T12:01:00Z"))
    path = fa.snapshot_path(d, "2026-27", 11)
    before = open(path).read()

    # Even a forecast that was itself completed before the deadline: the file
    # is frozen from the deadline on, whoever writes.
    ok, why = fa.write_snapshot(_snapshot("2026-11-13T18:00:00Z"), d,
                                now=_t("2026-11-13T18:31:00Z"))
    assert not ok and "deadline" in why
    assert open(path).read() == before


def test_a_deadline_moved_later_does_not_reopen_a_frozen_gameweek(tmp_path):
    d = str(tmp_path)
    fa.write_snapshot(_snapshot("2026-11-13T12:00:00Z"), d, now=_t("2026-11-13T12:01:00Z"))
    path = fa.snapshot_path(d, "2026-27", 11)
    before = open(path).read()

    moved = _snapshot("2026-11-14T09:00:00Z", deadline="2026-11-14T11:00:00Z")
    ok, _ = fa.write_snapshot(moved, d, now=_t("2026-11-14T09:30:00Z"))
    assert not ok
    assert open(path).read() == before


def test_a_forecast_completed_after_the_deadline_is_never_written(tmp_path):
    d = str(tmp_path)
    ok, why = fa.write_snapshot(_snapshot("2026-11-13T18:30:00Z"), d,
                                now=_t("2026-11-13T18:31:00Z"))
    assert not ok and "pre-deadline" in why
    assert not os.path.exists(fa.snapshot_path(d, "2026-27", 11))


def test_a_first_snapshot_from_before_the_deadline_is_still_kept(tmp_path):
    """A run that straddles the deadline forecast from pre-deadline data."""
    ok, _ = fa.write_snapshot(_snapshot("2026-11-13T18:00:00Z"), str(tmp_path),
                              now=_t("2026-11-13T18:40:00Z"))
    assert ok


def test_an_unreadable_file_is_not_overwritten_after_the_deadline(tmp_path):
    d = str(tmp_path)
    path = fa.snapshot_path(d, "2026-27", 11)
    os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        f.write("{not json")
    ok, _ = fa.write_snapshot(_snapshot("2026-11-13T18:00:00Z"), d,
                              now=_t("2026-11-13T19:00:00Z"))
    assert not ok
    assert open(path).read() == "{not json"


def test_a_degraded_forecast_never_replaces_a_sound_one(tmp_path):
    d = str(tmp_path)
    fa.write_snapshot(_snapshot("2026-11-11T12:00:00Z"), d, now=_t("2026-11-11T12:01:00Z"))
    ok, why = fa.write_snapshot(_snapshot("2026-11-12T12:00:00Z", degraded=True), d,
                                now=_t("2026-11-12T12:01:00Z"))
    assert not ok and "degraded" in why
    assert not fa.load(fa.snapshot_path(d, "2026-27", 11))["degraded"]


def test_a_sound_forecast_replaces_a_degraded_one(tmp_path):
    d = str(tmp_path)
    fa.write_snapshot(_snapshot("2026-11-11T12:00:00Z", degraded=True), d,
                      now=_t("2026-11-11T12:01:00Z"))
    assert fa.write_snapshot(_snapshot("2026-11-12T12:00:00Z"), d,
                             now=_t("2026-11-12T12:01:00Z"))[0]


def test_an_older_forecast_never_replaces_a_newer_one(tmp_path):
    d = str(tmp_path)
    fa.write_snapshot(_snapshot("2026-11-12T12:00:00Z"), d, now=_t("2026-11-12T12:01:00Z"))
    ok, why = fa.write_snapshot(_snapshot("2026-11-11T12:00:00Z"), d,
                                now=_t("2026-11-12T13:00:00Z"))
    assert not ok and "newer" in why


# ---------------------------------------------------------------- outcomes


def _live(n_players=6, pts=lambda pid: pid % 5, mins=lambda pid: 90 if pid % 3 else 0):
    return {"elements": [{"id": pid, "stats": {"total_points": pts(pid),
                                               "minutes": mins(pid), "starts": int(mins(pid) > 0)}}
                         for pid in range(1, n_players + 1)]}


def test_outcomes_are_saved_once_fpl_has_checked_them_and_never_rewritten(tmp_path):
    d = str(tmp_path)
    fa.write_snapshot(_snapshot("2026-11-12T12:00:00Z"), d, now=_t("2026-11-12T12:01:00Z"))
    calls = []

    def fetch(gw):
        calls.append(gw)
        return _live()

    # Played but not yet checked: bonus may still move.
    assert fa.record_actuals(d, "2026-27", _bootstrap(), fetch) == []
    assert calls == []

    checked = _bootstrap(gw=12, checked={11})
    assert fa.record_actuals(d, "2026-27", checked, fetch) == [11]
    doc = fa.load(fa.actual_path(d, "2026-27", 11))
    assert doc["gameweek"] == 11
    assert {r["id"]: r["total_points"] for r in fa.rows(doc)} == {p: p % 5 for p in range(1, 7)}

    assert fa.record_actuals(d, "2026-27", checked, fetch) == []
    assert calls == [11]


def test_a_failed_outcome_fetch_is_left_for_the_next_run(tmp_path):
    d = str(tmp_path)
    fa.write_snapshot(_snapshot("2026-11-12T12:00:00Z"), d, now=_t("2026-11-12T12:01:00Z"))

    def broken(gw):
        raise OSError("down")

    assert fa.record_actuals(d, "2026-27", _bootstrap(gw=12, checked={11}), broken) == []
    assert not os.path.exists(fa.actual_path(d, "2026-27", 11))


def test_outcomes_are_only_read_for_the_live_season(tmp_path):
    d = str(tmp_path)
    fa.write_snapshot(_snapshot("2026-11-12T12:00:00Z"), d, now=_t("2026-11-12T12:01:00Z"))
    next_season = _bootstrap(gw=12, checked={11})
    next_season["events"][0]["deadline_time"] = "2027-08-13T17:30:00Z"
    assert fa.record_actuals(d, "2026-27", next_season, lambda gw: _live()) == []


def test_archive_forecast_writes_the_gameweek_ahead(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("GITHUB_RUN_ID", "7")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    ok, why, recorded = fa.archive_forecast(
        str(tmp_path), bootstrap=_bootstrap(), xp_matrix=_matrix(),
        horizon_gws=[11, 12, 13], forecast_at=_t("2026-11-12T12:00:00Z"),
        now=_t("2026-11-12T12:00:30Z"))
    assert ok, why
    snap = fa.load(fa.snapshot_path(str(tmp_path), "2026-27", 11))
    assert snap["commit"] == "deadbeef"
    assert snap["run"] == {"id": "7", "url": "https://github.com/o/r/actions/runs/7"}


def test_the_weekly_job_survives_an_archive_failure(monkeypatch, capsys):
    """A week's plan must never be lost to the archive."""
    import argparse

    import weekly_manager

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(fa, "archive_forecast", boom)
    args = argparse.Namespace(risk_aversion=0.0, archive_dir="x", calibrate=False)
    weekly_manager._archive_forecast(args, {}, {}, [11], None, {}, {}, False, {}, {})

    # Rank-adjusted values are not expected points, so they are not archived.
    called = []
    monkeypatch.setattr(fa, "archive_forecast", lambda *a, **k: called.append(1))
    args.risk_aversion = 0.05
    weekly_manager._archive_forecast(args, {}, {}, [11], None, {}, {}, False, {}, {})
    assert called == []


# ------------------------------------------------------------------- scorer


def _scorable(gw, actual_pts, xp, ep_next, ppg, mins_last5=None, owned=None,
              ep_next_gw=None, degraded=False, season="2026-27"):
    """A hand-built snapshot and outcome pair, with full control of every column."""
    n = len(actual_pts)
    ids = list(range(1, n + 1))
    snap_rows = []
    for i, pid in enumerate(ids):
        row = dict.fromkeys(fa.SNAPSHOT_COLUMNS)
        row.update(id=pid, element_type=1 + i % 4, now_cost=50 + i,
                   selected_by_percent=(owned[i] if owned else 50.0 - i),
                   ep_next=ep_next[i], xp=xp[i], p_play=0.8, ppg=ppg[i],
                   mins_last5=(mins_last5[i] if mins_last5 else 450.0),
                   xp_horizon=[xp[i]])
        snap_rows.append([row[c] for c in fa.SNAPSHOT_COLUMNS])
    snap = {"season": season, "gameweek": gw, "commit": "c0ffee0", "degraded": degraded,
            "ep_next_gw": gw if ep_next_gw is None else ep_next_gw,
            "forecast_at": "2026-11-12T12:00:00Z",
            "columns": list(fa.SNAPSHOT_COLUMNS), "rows": snap_rows}
    actual = {"season": season, "gameweek": gw, "columns": list(fa.ACTUAL_COLUMNS),
              "rows": [[pid, actual_pts[i], 90 if actual_pts[i] else 0, 1]
                       for i, pid in enumerate(ids)]}
    return snap, actual


def _write_pair(d, snap, actual):
    path = fa.snapshot_path(d, snap["season"], snap["gameweek"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(fa.dumps(snap))
    with open(fa.actual_path(d, actual["season"], actual["gameweek"]), "w") as f:
        f.write(fa.dumps(actual))


def _gameweek(gw, seed):
    rng = np.random.default_rng(seed)
    actual = rng.poisson(3.0, 40).astype(float)
    xp = actual + rng.normal(0, 0.5, 40)            # the good forecast
    ep = actual + rng.normal(0, 2.0, 40)            # a noisier one
    ppg = np.full(40, actual.mean())                # no information at all
    return _scorable(gw, list(actual), list(xp), list(ep), list(ppg))


def test_the_scorer_uses_the_backtest_metrics_unchanged():
    snap, actual = _gameweek(11, 0)
    res = score_archive.score_gameweek(snap, actual)
    assert set(res["models"]) == {"fepl", "ep_next", "ppg"}

    pts = [r[1] for r in actual["rows"]]
    for name, col in (("fepl", "xp"), ("ep_next", "ep_next"), ("ppg", "ppg")):
        pred = [r[col] for r in fa.rows(snap)]
        ids = [r["id"] for r in fa.rows(snap)]
        m = res["models"][name]
        assert m["rmse"] == pytest.approx(np.sqrt(np.mean((np.array(pts) - pred) ** 2)))
        assert m["captured_at_15"] == pytest.approx(backtest._points_captured_at_k(pts, pred, ids, 15))
        assert m["ndcg_at_15"] == pytest.approx(backtest._ndcg_at_k(pts, pred, ids, 15))
        assert m["captain_regret"] == pytest.approx(backtest._captain_regret(pts, pred, ids))
    assert res["models"]["fepl"]["rmse"] < res["models"]["ep_next"]["rmse"]
    assert "p_play_auc" in res


def test_a_perfect_forecast_scores_perfectly():
    pts = [2, 9, 0, 1, 6, 3, 12, 1]
    snap, actual = _scorable(11, pts, pts, [1.0] * 8, [2.0] * 8)
    m = score_archive.score_gameweek(snap, actual)["models"]["fepl"]
    assert m["rmse"] == 0.0
    assert m["captured_at_15"] == 1.0
    assert m["captain_regret"] == 0.0


def test_ep_next_is_not_scored_when_it_was_for_another_gameweek():
    snap, actual = _scorable(11, [1, 2, 3], [1, 2, 3], [None] * 3, [1, 1, 1], ep_next_gw=12)
    assert set(score_archive.score_gameweek(snap, actual)["models"]) == {"fepl", "ppg"}


def test_the_population_is_backtests_rule_and_ignores_every_forecast():
    owned = [40.0, 30.0, 20.0, 10.0, 5.0]
    mins = [0.0, 0.0, 0.0, 0.0, 200.0]
    snap, _ = _scorable(11, [1] * 5, [9, 0, 0, 0, 0], [0] * 5, [0] * 5,
                        mins_last5=mins, owned=owned)
    # Top 2 by ownership, plus anyone with 180+ recent minutes.
    assert score_archive.eval_population(fa.rows(snap), top_n=2) == {1, 2, 5}


def test_players_without_outcomes_are_not_scored():
    snap, actual = _scorable(11, [1, 2, 3, 4], [1, 2, 3, 4], [1] * 4, [1] * 4)
    actual["rows"] = actual["rows"][:3]
    assert score_archive.score_gameweek(snap, actual)["n"] == 3


def test_mismatched_gameweeks_are_refused():
    snap, actual = _scorable(11, [1, 2, 3], [1, 2, 3], [1] * 3, [1] * 3)
    actual["gameweek"] = 12
    with pytest.raises(ValueError):
        score_archive.score_gameweek(snap, actual)


def test_paired_intervals_need_three_gameweeks(tmp_path):
    d = str(tmp_path)
    for gw in (11, 12):
        _write_pair(d, *_gameweek(gw, gw))
    per_gw = score_archive.score(d)
    assert [g["gw"] for g in per_gw] == [11, 12]
    assert score_archive.comparisons(per_gw) == []

    _write_pair(d, *_gameweek(13, 13))
    per_gw = score_archive.score(d)
    rows = score_archive.comparisons(per_gw)
    rmse = {(r["model"], r["baseline"]): r for r in rows if r["metric"] == "rmse"}
    assert set(rmse) == set(score_archive.COMPARISONS)
    # Identical to calling the backtest's own interval directly.
    direct = backtest.paired_ci(per_gw, "fepl", "ep_next", "rmse")
    assert rmse[("fepl", "ep_next")]["diff"] == pytest.approx(direct["diff"])
    assert rmse[("fepl", "ep_next")]["diff"] < 0


def test_gameweeks_without_outcomes_or_with_a_degraded_forecast_are_skipped(tmp_path):
    d = str(tmp_path)
    _write_pair(d, *_gameweek(11, 1))
    snap, actual = _gameweek(12, 2)
    snap["degraded"] = True
    _write_pair(d, snap, actual)
    snap13, _ = _gameweek(13, 3)
    path = fa.snapshot_path(d, "2026-27", 13)
    with open(path, "w") as f:
        f.write(fa.dumps(snap13))

    assert [g["gw"] for g in score_archive.score(d)] == [11]
    assert [g["gw"] for g in score_archive.score(d, include_degraded=True)] == [11, 12]


def test_the_scorer_runs_end_to_end(tmp_path, capsys):
    d = str(tmp_path / "archive")
    for gw in (11, 12, 13, 14):
        _write_pair(d, *_gameweek(gw, gw))
    out = str(tmp_path / "scores.json")
    assert score_archive.main(["--archive-dir", d, "--json-out", out]) == 0
    text = capsys.readouterr().out
    assert "BACKTEST SUMMARY" in text and "ep_next" in text and "significant" in text
    saved = json.load(open(out))
    assert len(saved["per_gameweek"]) == 4
    assert {r["baseline"] for r in saved["comparisons"]} == {"ep_next", "ppg"}


def test_an_empty_archive_is_not_an_error(tmp_path, capsys):
    assert score_archive.main(["--archive-dir", str(tmp_path)]) == 0
    assert "Nothing to score" in capsys.readouterr().out
