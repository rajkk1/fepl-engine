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
