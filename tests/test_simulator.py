"""
Season replay.

This is the only thing in the repo that measures the objective the engine exists
to serve - points on the board - so its own correctness matters more than the
model's, exactly as with the backtest harness.
"""
import pytest

from simulator import (apply_autosubs, score_gameweek, _formation_ok,
                       _baseline_matrix, FORMATION_LIMITS, XP_SOURCES,
                       POS_GKP, POS_DEF, POS_MID, POS_FWD)

# A legal FPL squad: 2 GK, 5 DEF, 5 MID, 3 FWD, starting 1-4-4-2.
POSITION = {1: POS_GKP, 12: POS_GKP,
            2: POS_DEF, 3: POS_DEF, 4: POS_DEF, 5: POS_DEF, 13: POS_DEF,
            6: POS_MID, 7: POS_MID, 8: POS_MID, 9: POS_MID, 14: POS_MID,
            10: POS_FWD, 11: POS_FWD, 15: POS_FWD}
STARTERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
BENCH = [12, 13, 14, 15]
ALL_PLAYED = {i: True for i in range(1, 16)}


def _absent(**ids):
    p = dict(ALL_PLAYED)
    p.update({int(k[1:]): False for k in ids})
    return p


def _shape(eleven):
    pos = [POSITION[p] for p in eleven]
    return {q: pos.count(q) for q in (POS_GKP, POS_DEF, POS_MID, POS_FWD)}


def test_no_absences_means_no_substitutions():
    assert apply_autosubs(STARTERS, BENCH, ALL_PLAYED, POSITION) == STARTERS


def test_only_a_keeper_replaces_a_keeper():
    """An outfielder must never come on for the goalkeeper."""
    out = apply_autosubs(STARTERS, BENCH, _absent(p1=1), POSITION)
    assert 12 in out and 1 not in out
    assert _shape(out)[POS_GKP] == 1


def test_bench_order_is_the_priority():
    out = apply_autosubs(STARTERS, BENCH, _absent(p2=1), POSITION)
    assert [p for p in out if p not in STARTERS] == [13]


@pytest.mark.parametrize("absent", [
    {"p10": 1, "p11": 1},              # both forwards
    {"p6": 1, "p7": 1},                # two midfielders
    {"p1": 1, "p10": 1},               # keeper and a forward
    {"p2": 1, "p6": 1, "p10": 1},      # one of each
    {"p2": 1, "p3": 1, "p4": 1},       # three defenders
])
def test_autosubs_always_leave_a_legal_formation(absent):
    """
    The regression this guards: substitutions were taken greedily in bench
    order, so with both forwards absent the bench defender *and* midfielder came
    on and the side finished with no forward at all. Declining a substitution can
    be the only way to stay legal, so the choice needs lookahead.
    """
    out = apply_autosubs(STARTERS, BENCH, _absent(**absent), POSITION)
    if len(out) == 11:
        assert _formation_ok([POSITION[p] for p in out])
    for q, (_, hi) in FORMATION_LIMITS.items():
        assert _shape(out)[q] <= hi


def test_a_side_can_finish_short_when_the_bench_cannot_cover():
    played = _absent(p10=1)
    for b in BENCH:
        played[b] = False
    out = apply_autosubs(STARTERS, BENCH, played, POSITION)
    assert len(out) == 10 and 10 not in out


def test_captain_keeps_the_armband_after_a_blank():
    """
    A captain who played and scored nothing still wears it. The previous version
    handed the armband to the vice-captain whenever the captain blanked, which
    silently inflated every score.
    """
    pts = {i: 2.0 for i in range(1, 16)}
    pts[6] = 0.0
    res = score_gameweek(STARTERS, BENCH, 6, 7, pts, ALL_PLAYED, POSITION)
    assert res["leader"] == 6


def test_armband_moves_only_when_the_captain_does_not_appear():
    pts = {i: 2.0 for i in range(1, 16)}
    pts[7] = 9.0
    res = score_gameweek(STARTERS, BENCH, 6, 7, pts, _absent(p6=1), POSITION)
    assert res["leader"] == 7
    # 10 survivors at 2 plus the 9-point vice, plus his 9 again for the armband.
    assert res["points"] == pytest.approx(2.0 * 9 + 9.0 + 2.0 + 9.0)


def test_armband_is_dropped_when_captain_and_vice_both_miss():
    res = score_gameweek(STARTERS, BENCH, 6, 7, {i: 2.0 for i in range(1, 16)},
                         _absent(p6=1, p7=1), POSITION)
    assert res["leader"] is None


def test_triple_captain_pays_twice_over():
    pts = {i: 0.0 for i in range(1, 16)}
    pts[6] = 10.0
    normal = score_gameweek(STARTERS, BENCH, 6, 7, pts, ALL_PLAYED, POSITION)
    tripled = score_gameweek(STARTERS, BENCH, 6, 7, pts, ALL_PLAYED, POSITION,
                             triple=True)
    assert normal["points"] == 20.0 and tripled["points"] == 30.0


def test_autosubbed_players_score():
    pts = {i: 0.0 for i in range(1, 16)}
    pts[13] = 7.0
    res = score_gameweek(STARTERS, BENCH, 6, 7, pts, _absent(p2=1), POSITION)
    assert 13 in res["eleven"] and res["points"] == 7.0


def test_baseline_matrix_is_shaped_like_the_engines(gw_frame):
    """
    The optimiser must not be able to tell a baseline forecast from the engine's,
    or the comparison measures the plumbing instead of the forecast.
    """
    m = _baseline_matrix(gw_frame, [5, 6], "ppg", [1, 2, 3])
    assert set(m) == {1, 2, 3}
    for row in m.values():
        assert set(row) == {5, 6, "5_p_play", "6_p_play"}
        assert 0.0 <= row["5_p_play"] <= 1.0


def test_every_declared_source_can_build_a_matrix(gw_frame):
    for src in XP_SOURCES:
        if src == "engine":
            continue
        assert _baseline_matrix(gw_frame, [5], src, [1, 2])


# --- chips -----------------------------------------------------------------
#
# The replay used to pass no `active_chip` at all, so a wildcard, free hit,
# bench boost and triple captain were never played in any season it reported.
# The chip machinery was therefore measured by nothing, which is how optimiser
# rules keyed to a loop index rather than a gameweek survived in it.

def test_bench_boost_scores_all_fifteen():
    points = {i: 2.0 for i in range(1, 16)}
    plain = score_gameweek(STARTERS, BENCH, 1, 2, points, ALL_PLAYED, POSITION)
    boosted = score_gameweek(STARTERS, BENCH, 1, 2, points, ALL_PLAYED, POSITION,
                             bench_boost=True)
    # Eleven plus the armband, against fifteen plus the armband.
    assert plain["points"] == 12 * 2.0
    assert boosted["points"] == 16 * 2.0
    assert set(boosted["eleven"]) == set(STARTERS) | set(BENCH)


def test_bench_boost_has_nothing_for_an_autosub_to_do():
    """
    Every player is already counted, so a bench player coming on for an absent
    starter must not be counted twice.
    """
    points = {i: 2.0 for i in range(1, 16)}
    boosted = score_gameweek(STARTERS, BENCH, 1, 2, points, _absent(p10=True),
                             POSITION, bench_boost=True)
    assert boosted["autosubs"] == []
    assert len(boosted["eleven"]) == len(set(boosted["eleven"])) == 15


def test_bench_boost_still_pays_the_armband_once():
    points = {i: 0.0 for i in range(1, 16)}
    points[1] = 10.0
    boosted = score_gameweek(STARTERS, BENCH, 1, 2, points, ALL_PLAYED, POSITION,
                             bench_boost=True)
    assert boosted["points"] == 20.0


def test_triple_captain_and_bench_boost_are_independent():
    points = {i: 1.0 for i in range(1, 16)}
    both = score_gameweek(STARTERS, BENCH, 1, 2, points, ALL_PLAYED, POSITION,
                          triple=True, bench_boost=True)
    assert both["points"] == 15 + 2.0


# --- the policy ------------------------------------------------------------

def test_a_chip_is_spent_for_the_half_it_was_played_in():
    from chip_policy import available_chips

    assert set(available_chips({}, 5)) == {"wc", "fh", "bb", "tc"}
    assert "wc" not in available_chips({"wc": 2}, 5)
    # ...but the second-half set is a fresh one.
    assert "wc" in available_chips({"wc": 2}, 25)
    assert "wc" not in available_chips({"wc": 25}, 30)


def test_the_bar_for_a_chip_falls_as_the_season_runs_out():
    from chip_policy import chip_thresholds

    early, late = chip_thresholds(2), chip_thresholds(35)
    for chip in ("tc", "bb", "fh", "wc"):
        assert late[chip] < early[chip], chip


def test_choose_chip_holds_a_chip_for_the_gameweek_that_wants_it():
    """
    Searching gameweeks as well as chips is the whole point: a chip worth
    playing in GW+2 must not be reported as a chip for this week.
    """
    from chip_policy import choose_chip

    def solve(chip, gw):
        # Nothing is worth a chip except a bench boost two weeks out.
        gain = 60.0 if (chip == "bb" and gw == 12) else 0.0
        return {"total_xp": 100.0 + gain}

    best = choose_chip(solve, [10, 11, 12, 13, 14], 10, ["wc", "fh", "bb", "tc"])
    assert best["chip"] == "bb"
    assert best["gw"] == 12


def test_choose_chip_declines_a_gain_under_the_threshold():
    from chip_policy import choose_chip

    def solve(chip, gw):
        return {"total_xp": 100.0 + (1.0 if chip else 0.0)}

    best = choose_chip(solve, [10, 11], 10, ["wc", "fh", "bb", "tc"])
    assert best["chip"] == ""


def test_a_chip_already_spent_is_never_searched():
    from chip_policy import choose_chip

    asked = []

    def solve(chip, gw):
        asked.append(chip)
        return {"total_xp": 100.0 + (99.0 if chip else 0.0)}

    best = choose_chip(solve, [10], 10, ["tc"])
    assert set(a for a in asked if a) == {"tc"}
    assert best["chip"] == "tc"


# --- the free hit hands the squad back --------------------------------------
#
# A free hit squad is borrowed for one week. Nothing here tested that, because
# nothing here played a chip at all, and the same blind spot in the optimiser
# meant a free hit planned for a later gameweek was solved as a permanent
# rebuild. This drives a whole (small) season to check the bookkeeping: the
# squad, the bank and the purchase prices all come back.

@pytest.fixture
def tiny_season():
    """
    Four gameweeks, 20 clubs, 60 players: enough for a legal squad under the
    3-per-club cap, small enough to solve in a moment.
    """
    import pandas as pd

    n_teams, per_team = 20, 3
    players, teams = [], [{"id": t, "name": f"T{t}", "short_name": f"T{t}"}
                          for t in range(1, n_teams + 1)]
    pid = 1
    for t in range(1, n_teams + 1):
        for k in range(per_team):
            players.append({
                "id": pid, "web_name": f"P{pid}",
                # 1 GK, 1 DEF, 1 MID per club, plus forwards from the last few.
                "element_type": [1, 2, 3][k] if t <= 14 else [1, 2, 4][k],
                "team": t, "now_cost": 45,
                "penalties_order": None,
                "corners_and_indirect_freekicks_order": None,
                "direct_freekicks_order": None,
            })
            pid += 1

    gw_rows, fixtures = [], []
    for gw in range(1, 5):
        for t in range(1, n_teams, 2):
            fixtures.append({
                "event": gw, "team_h": t, "team_a": t + 1,
                "team_h_score": 1, "team_a_score": 1,
                "kickoff_time": f"2024-08-{10 + gw:02d}T14:00:00Z",
                "team_h_difficulty": 3, "team_a_difficulty": 3,
            })
        for p in players:
            gw_rows.append({
                "GW": gw, "element": p["id"], "minutes": 90,
                "total_points": 2, "value": 45, "was_home": True,
                "opponent_team": 1, "selected": 1000, "xP": 2.0,
            })
    return (pd.DataFrame(gw_rows), pd.DataFrame(players),
            pd.DataFrame(teams), pd.DataFrame(fixtures))


def test_a_free_hit_squad_is_handed_back_the_following_week(tiny_season, monkeypatch):
    import chip_policy
    import simulator

    df_gw, df_players, _, _ = tiny_season
    all_ids = list(df_players["id"])
    # Players 31+ are worth everything in GW3 and nothing otherwise, so the
    # free hit squad must differ sharply and must not survive the week.
    spike = set(all_ids[30:])

    def fake_matrix(df_gw_, horizon_gws, source, population):
        out = {}
        for p in population:
            out[p] = {gw: (9.0 if (gw == 3 and p in spike) else
                           (0.5 if p in spike else 3.0))
                      for gw in horizon_gws}
        return out

    monkeypatch.setattr(simulator, "_baseline_matrix", fake_matrix)

    def force_free_hit(solve, horizon_gws, current_gw, available):
        if current_gw == 3:
            return {"chip": "fh", "gw": 3, "gain": 99.0, "res": solve("fh", 3)}
        return {"chip": "", "gw": horizon_gws[0], "gain": 0.0, "res": solve(None, None)}

    monkeypatch.setattr(chip_policy, "choose_chip", force_free_hit)

    res = simulator.run_season_simulation(
        "2024-25", horizon=2, xp_source="ppg", from_gw=1, to_gw=4,
        data=tiny_season, verbose=False)

    by_gw = {h["gw"]: h for h in res["history"]}
    assert res["chips_played"] == [{"chip": "fh", "gw": 3}]

    before = set(by_gw[2]["squad"])
    during = set(by_gw[3]["squad"])
    after = set(by_gw[4]["squad"])

    # The chip actually bought a different squad...
    assert len(during & spike) > len(before & spike)
    # ...and it was handed back, not kept. One free transfer may then move a
    # single player, so allow exactly that much drift and no more.
    assert len(after - before) <= 1, "the free hit squad survived its week"
    assert by_gw[4]["hits"] == 0, "paid to undo a free hit"


def test_a_free_hit_does_not_spend_the_free_transfer_bank(tiny_season, monkeypatch):
    """
    The chip pays for its own transfers and saved free transfers survive it, so
    the week after must not find the bank emptied by the rebuild.
    """
    import chip_policy
    import simulator

    def force_free_hit(solve, horizon_gws, current_gw, available):
        if current_gw == 3:
            return {"chip": "fh", "gw": 3, "gain": 99.0, "res": solve("fh", 3)}
        return {"chip": "", "gw": horizon_gws[0], "gain": 0.0, "res": solve(None, None)}

    monkeypatch.setattr(chip_policy, "choose_chip", force_free_hit)

    res = simulator.run_season_simulation(
        "2024-25", horizon=2, xp_source="ppg", from_gw=1, to_gw=4,
        data=tiny_season, verbose=False)

    by_gw = {h["gw"]: h for h in res["history"]}
    # A free hit week makes many transfers but none of them are charged, and
    # none of them come out of the bank.
    assert by_gw[3]["hits"] == 0
    assert by_gw[4]["hits"] == 0


# --- pooling across seasons -------------------------------------------------
#
# The README quotes a pooled figure over 114 gameweeks, which is the one worth
# quoting - a single season is ~38 paired samples of a very noisy difference -
# but nothing in the repo produced it, so the number could not be re-derived.

def _season(net_by_source, hits=0, transfers=0):
    return {src: {"history": [{"net": v} for v in nets],
                  "total_hits": hits, "total_transfers": transfers,
                  "total_points": float(sum(nets)),
                  "points_per_gw": float(sum(nets)) / max(1, len(nets)),
                  "season": "x", "gameweeks": len(nets), "chips_played": []}
            for src, nets in net_by_source.items()}


def test_pooling_sums_the_seasons_it_was_given(capsys):
    from simulator import _print_pooled

    _print_pooled({
        "2023-24": _season({"engine": [10, 20], "ppg": [5, 5]}, hits=1, transfers=3),
        "2024-25": _season({"engine": [30, 40], "ppg": [5, 5]}, hits=2, transfers=4),
    })
    out = capsys.readouterr().out

    assert "4 gameweeks" in out
    # 10+20+30+40 against 5*4, and the hits/transfers add up too.
    assert " 100" in out and " 20" in out
    assert "+80" in out, out           # engine minus ppg, pooled total


def test_pooling_pairs_by_gameweek_not_by_season_total(capsys):
    """
    A season total is one sample. The paired test needs every gameweek, so the
    pooled win rate must be computed over all of them, not over seasons.
    """
    from simulator import _print_pooled

    # Engine wins three gameweeks out of four, narrowly, and loses one heavily.
    _print_pooled({
        "a": _season({"engine": [11, 11], "ppg": [10, 10]}),
        "b": _season({"engine": [11, 0], "ppg": [10, 40]}),
    })
    out = capsys.readouterr().out
    assert "0.750" in out, out          # 3 of 4 gameweeks won
    # ...but the mean difference is negative, which is the point of reporting both.
    assert "-9.25" in out, out


def test_pooling_is_skipped_for_a_single_season(capsys):
    from simulator import compare_seasons
    import simulator

    calls = []

    def fake_compare(season, **kw):
        calls.append(season)
        return _season({"engine": [1, 2]})

    original = simulator.compare_sources
    simulator.compare_sources = fake_compare
    try:
        compare_seasons(["2024-25"])
    finally:
        simulator.compare_sources = original

    assert calls == ["2024-25"]
    assert "POOLED" not in capsys.readouterr().out


# --- protecting the table ---------------------------------------------------
#
# The README's end-to-end table drifted and nothing caught it, because nothing
# re-ran it. These cover the mechanism that now does.

def _hist(nets, squad=None, armband=None, hits=None):
    n = len(nets)
    squad = squad if squad is not None else [v for v in nets]
    armband = armband if armband is not None else [0.0] * n
    hits = hits if hits is not None else [0] * n
    return [{"net": nets[i], "squad_points": squad[i],
             "armband_points": armband[i], "hits": hits[i]} for i in range(n)]


def _pooled(**by_source):
    return {"s1": {src: {"history": _hist(nets), "total_points": float(sum(nets)),
                         "points_per_gw": 0.0, "total_hits": 0,
                         "total_transfers": 0, "season": "s1",
                         "gameweeks": len(nets), "chips_played": []}
                   for src, nets in by_source.items()}}


def test_the_replay_gate_passes_an_unchanged_run():
    from simulator import check_replay

    run = _pooled(engine=[50, 60], ppg=[40, 40])
    ok, msg = check_replay(run, {"totals": {"engine": 110, "ppg": 80}})
    assert ok, msg


def test_the_replay_gate_catches_the_drift_that_actually_happened():
    """
    The recorded table said 2069/2218/2090 and the replay returned
    2022/2247/2040 - between 1.1% and 2.4%. The gate has to catch that or it is
    not worth having.
    """
    from simulator import check_replay

    run = _pooled(engine=[2022 + 2247 + 2040])
    ok, msg = check_replay(run, {"totals": {"engine": 2069 + 2218 + 2090}})
    assert not ok
    assert "engine" in msg and "re-record" in msg


def test_the_replay_gate_tolerates_drift_below_its_bar():
    from simulator import check_replay

    # 0.5% low: upstream wobble, let it through.
    run = _pooled(engine=[6277])
    assert check_replay(run, {"totals": {"engine": 6309}})[0]
    # 2% low: the table no longer says what it says.
    run = _pooled(engine=[6183])
    assert not check_replay(run, {"totals": {"engine": 6309}})[0]


def test_the_replay_gate_watches_the_baselines_too():
    """
    A run where the engine holds but `ppg` moves is still a table that no longer
    reproduces - and it is the *margin* that the README quotes.
    """
    from simulator import check_replay

    run = _pooled(engine=[6309], ppg=[6300])
    ok, msg = check_replay(run, {"totals": {"engine": 6309, "ppg": 6007}})
    assert not ok
    assert "ppg" in msg


def test_a_configuration_with_nothing_recorded_does_not_fail():
    from simulator import check_replay

    ok, msg = check_replay(_pooled(engine=[1]), {})
    assert ok
    assert "nothing to hold" in msg


def test_the_replay_key_separates_chips_from_no_chips():
    """
    Chips on and chips off are different questions with different totals;
    scoring one against the other's record would fail for no reason.
    """
    from simulator import replay_key

    on = replay_key(["2023-24"], 1, 38, True)
    off = replay_key(["2023-24"], 1, 38, False)
    assert on != off
    assert replay_key(["b", "a"], 1, 38, True) == replay_key(["a", "b"], 1, 38, True)


def test_recording_then_gating_the_same_run_passes(tmp_path):
    from simulator import record_replay_baseline, load_replay_baseline, check_replay

    path = str(tmp_path / "replay_baseline.json")
    run = _pooled(engine=[50, 60], ppg=[40, 40])
    entry = record_replay_baseline(run, "k", path)

    assert entry["totals"] == {"engine": 110, "ppg": 80}
    assert entry["_gameweeks"] == 2
    # The margin is recorded for a reader, not gated on.
    assert entry["_margin_per_gw"]["ppg"] == 15.0

    ok, _ = check_replay(run, load_replay_baseline(path)["k"])
    assert ok


def test_a_missing_or_corrupt_replay_baseline_is_not_fatal(tmp_path):
    from simulator import load_replay_baseline

    assert load_replay_baseline(str(tmp_path / "nope.json")) == {}
    p = tmp_path / "replay_baseline.json"
    p.write_text("{not json")
    assert load_replay_baseline(str(p)) == {}


# --- attribution ------------------------------------------------------------

def test_the_attribution_sums_to_the_margin(capsys):
    """
    squad + armband - 4 x hits must equal the net difference exactly, or the
    decomposition is telling a story the points do not support.
    """
    from simulator import _print_attribution

    per_source = {
        "engine": {"history": _hist([46, 46], squad=[40, 40],
                                    armband=[10, 10], hits=[1, 1])},
        "ppg": {"history": _hist([30, 30], squad=[25, 25],
                                 armband=[5, 5], hits=[0, 0])},
    }
    _print_attribution(per_source)
    out = capsys.readouterr().out

    # squad +30, armband +10, hits -8 -> net +32, which is 2 x (46 - 30).
    assert "+30" in out and "+10" in out and "-8" in out and "+32" in out


def test_the_attribution_shows_a_margin_handed_back_at_the_armband(capsys):
    """
    The case the decomposition exists for: better at picking a squad, worse at
    picking a captain, and a season total that cannot tell you so.
    """
    from simulator import _print_attribution

    per_source = {
        "engine": {"history": _hist([50], squad=[45], armband=[5], hits=[0])},
        "ppg": {"history": _hist([50], squad=[30], armband=[20], hits=[0])},
    }
    _print_attribution(per_source)
    out = capsys.readouterr().out
    assert "+15" in out and "-15" in out
