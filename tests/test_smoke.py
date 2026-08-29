"""
End-to-end wiring.

The offline test drives the whole pipeline on a synthetic league so CI never
depends on the FPL API being up. The live check is marked `network` and is
skipped unless you ask for it with `-m network`.
"""
import pytest

from xp_model import generate_xp_matrix, generate_merv_matrix
from optimizer import solve_fpl_optimization


@pytest.fixture
def mini_league():
    """Six clubs, a full squad's worth of players, two gameweeks of history."""
    teams = [{"id": i, "name": f"Club {i}", "short_name": f"C{i}"} for i in range(1, 7)]

    elements, all_history = [], {}
    pid = 1
    for team in range(1, 7):
        # 2 GKP, 5 DEF, 5 MID, 3 FWD per club
        for et, count in ((1, 2), (2, 5), (3, 5), (4, 3)):
            for _ in range(count):
                elements.append({
                    "id": pid, "web_name": f"P{pid}", "element_type": et,
                    "team": team, "now_cost": 40 + (pid % 60),
                    "status": "a", "chance_of_playing_next_round": None, "news": "",
                    "selected_by_percent": str(1.0 + pid % 30),
                    "penalties_order": 1 if pid % 15 == 0 else None,
                    "history_past": [],
                })
                all_history[pid] = [
                    {"round": gw, "minutes": 90 if pid % 5 else 0,
                     "total_points": 2 + (pid % 5), "value": 40 + (pid % 60),
                     "was_home": gw % 2 == 0, "opponent_team": (pid % 6) + 1,
                     "starts": 1 if pid % 5 else 0,
                     "expected_goals": 0.1 * (et == 4), "expected_assists": 0.05,
                     "saves": 2.0 if et == 1 else 0.0, "yellow_cards": 0.0,
                     "clearances_blocks_interceptions": 4.0, "tackles": 1.0,
                     "recoveries": 5.0}
                    for gw in range(1, 5)
                ]
                pid += 1

    fixtures = []
    for gw in (1, 2, 3, 4, 5, 6):
        pairs = [(1, 2), (3, 4), (5, 6)] if gw % 2 else [(2, 3), (4, 5), (6, 1)]
        for h, a in pairs:
            fixtures.append({
                "event": gw, "team_h": h, "team_a": a,
                "finished": gw < 5,
                "team_h_score": 1 if gw < 5 else None,
                "team_a_score": 1 if gw < 5 else None,
                "kickoff_time": f"2025-09-{10 + gw:02d}T14:00:00Z",
            })

    return {"teams": teams, "elements": elements}, fixtures, all_history


def test_xp_matrix_end_to_end(mini_league):
    bootstrap, fixtures, hist = mini_league
    matrix = generate_xp_matrix([5, 6], bootstrap=bootstrap, fixtures=fixtures,
                                all_history=hist, calibrate=True)

    assert len(matrix) == len(bootstrap["elements"])
    for pid, row in matrix.items():
        for gw in (5, 6):
            assert row[gw] >= 0.0, "expected points must be non-negative"
            assert row[gw] < 30.0, "expected points must be plausible"
            assert 0.0 <= row[f"{gw}_p_play"] <= 1.0
    assert max(r[5] for r in matrix.values()) > 0.0, "everyone scored zero"


def test_merv_equals_xp_when_risk_aversion_is_zero(mini_league):
    bootstrap, fixtures, hist = mini_league
    kwargs = dict(bootstrap=bootstrap, fixtures=fixtures, all_history=hist, calibrate=False)
    xp = generate_xp_matrix([5], **kwargs)
    merv = generate_merv_matrix([5], risk_aversion=0.0, **kwargs)
    for pid in xp:
        assert xp[pid][5] == pytest.approx(merv[pid][5], abs=0.01)


def test_merv_diverges_from_xp_with_risk_aversion(mini_league):
    bootstrap, fixtures, hist = mini_league
    kwargs = dict(bootstrap=bootstrap, fixtures=fixtures, all_history=hist, calibrate=False)
    xp = generate_xp_matrix([5], **kwargs)
    merv = generate_merv_matrix([5], risk_aversion=0.08, **kwargs)
    assert any(abs(xp[pid][5] - merv[pid][5]) > 0.01 for pid in xp), \
        "risk aversion had no effect - the rank-aware layer is inert"


def test_optimizer_consumes_the_matrix(mini_league):
    bootstrap, fixtures, hist = mini_league
    matrix = generate_xp_matrix([5, 6], bootstrap=bootstrap, fixtures=fixtures,
                                all_history=hist, calibrate=False)
    res = solve_fpl_optimization(bootstrap, matrix, [5, 6], initial_bank=100.0)
    assert res["status"] == "Optimal"
    gw = res["gameweeks"][5]
    assert len(gw["starters"]) == 11
    assert len(gw["bench"]) == 4
    assert gw["captain_id"] is not None


def test_weekly_manager_state_arity():
    """The CLI unpacks five values; a change here breaks the daily job."""
    import weekly_manager
    res = weekly_manager.get_manager_team_state(-1, 1)
    assert len(res) == 5
    _, _, _, _, chips = res
    assert isinstance(chips, list)


@pytest.mark.network
def test_live_api_shape():
    from fpl_api import get_bootstrap_static, get_fixtures, get_current_gameweek

    bs = get_bootstrap_static()
    assert len(bs["elements"]) > 300
    assert len(bs["teams"]) == 20
    assert get_fixtures()
    assert 1 <= get_current_gameweek(bs) <= 38
