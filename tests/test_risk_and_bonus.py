"""
Match simulation (bonus + correlated risk), effective ownership, and MERV.
"""
import numpy as np
import pytest

from match_sim import MatchSimulator, squad_score_draws
from monte_carlo import calculate_merv
from ownership_model import project_effective_ownership, build_eo_matrix


def _player(pid, team, et=3, **kw):
    base = {
        "id": pid, "team": team, "element_type": et,
        "p_play": 0.9, "p_60": 0.8, "xmin": 80.0,
        "xg_cond": 0.3, "xa_cond": 0.2, "saves90": 0.0, "yc90": 0.1,
        "recoveries90": 5.0, "tackles90": 1.5, "key_passes90": 1.0,
        "defcon_mu": 6.0, "defcon_dispersion": 1.85,
    }
    base.update(kw)
    return base


@pytest.fixture
def squad():
    return ([_player(i, 1) for i in range(1, 12)] +
            [_player(i, 2) for i in range(12, 23)])


# ------------------------------------------------------------------- bonus


def test_bonus_is_awarded_by_rank_within_the_match(squad):
    """Bonus is 3/2/1 to the top three BPS scorers, so it cannot exceed 6 a match."""
    sim = MatchSimulator(n_sims=400, seed=1)
    out = sim.simulate(squad, {1: 1.2, 2: 1.4})
    total = sum(out["bonus"].values())
    assert 0.0 < total <= 6.0 + 1e-6, f"total expected bonus {total} is not rank-consistent"


def test_better_players_earn_more_bonus():
    strong = _player(1, 1, 4, xg_cond=1.2, xa_cond=0.5)
    weak = [_player(i, 2, 2, xg_cond=0.01, xa_cond=0.01, defcon_mu=1.0)
            for i in range(2, 14)]
    sim = MatchSimulator(n_sims=600, seed=2)
    out = sim.simulate([strong] + weak, {1: 1.0, 2: 1.6})
    assert out["bonus"][1] > max(out["bonus"][i] for i in range(2, 14))


def test_a_player_who_cannot_play_scores_nothing(squad):
    squad = list(squad)
    squad[0] = _player(1, 1, p_play=0.0, p_60=0.0, xmin=0.0)
    sim = MatchSimulator(n_sims=300, seed=3)
    out = sim.simulate(squad, {1: 1.2, 2: 1.4})
    assert out["bonus"][1] == pytest.approx(0.0)
    assert out["mean_points"][1] == pytest.approx(0.0)


# ------------------------------------------------------- correlated variance


def test_teammates_clean_sheets_are_correlated():
    """
    The whole point of simulating a fixture jointly: stacking three defenders
    from one club must be riskier than one from each of three clubs.
    """
    sim = MatchSimulator(n_sims=4000, seed=5)
    same = [_player(i, 1, 2, xg_cond=0.05, xa_cond=0.05) for i in (1, 2, 3)]
    others = [_player(i, 2, 3) for i in range(10, 21)]
    out = sim.simulate(same + others, {1: 1.1, 2: 1.3})

    stacked = squad_score_draws(out["points"], [1, 2, 3])
    independent_var = sum(np.var(out["points"][i]) for i in (1, 2, 3))
    assert np.var(stacked) > independent_var * 1.05, \
        "team-mates must not behave as independent bets"


def test_squad_draws_double_the_captain():
    sim = MatchSimulator(n_sims=500, seed=7)
    squad = [_player(i, 1) for i in range(1, 12)] + [_player(i, 2) for i in range(12, 23)]
    out = sim.simulate(squad, {1: 1.2, 2: 1.4})
    plain = squad_score_draws(out["points"], [1, 2, 3])
    capped = squad_score_draws(out["points"], [1, 2, 3], captain_id=1)
    assert capped.mean() > plain.mean()


# ---------------------------------------------------------------- MERV / EO


def test_merv_rewards_template_and_penalises_differentials():
    """
    Regression: `eo_bounded = min(0.5, eo)` made (1 - 2*EO) non-negative, so the
    template-as-variance-shield branch the docstring described was unreachable.
    """
    xp, var, ra = 6.0, 20.0, 0.05
    differential = calculate_merv(xp, var, eo=0.05, risk_aversion=ra)
    neutral = calculate_merv(xp, var, eo=0.50, risk_aversion=ra)
    template = calculate_merv(xp, var, eo=1.40, risk_aversion=ra)

    assert differential < xp, "a differential must be penalised"
    assert neutral == pytest.approx(xp), "EO 0.5 is the break-even point"
    assert template > xp, "high ownership must act as a variance shield"


def test_merv_is_identity_without_risk_aversion():
    for eo in (0.0, 0.3, 0.8, 1.5):
        assert calculate_merv(5.0, 30.0, eo, risk_aversion=0.0) == pytest.approx(5.0)


def test_effective_ownership_is_monotone_and_bounded():
    values = [project_effective_ownership({"selected_by_percent": p})
              for p in ("0.0", "5.0", "20.0", "50.0", "80.0")]
    assert values == sorted(values)
    assert all(0.0 <= v < 2.0 for v in values)


def test_eo_handles_missing_and_malformed_input():
    assert project_effective_ownership({}) == pytest.approx(0.0)
    assert project_effective_ownership({"selected_by_percent": None}) == pytest.approx(0.0)
    assert project_effective_ownership({"selected_by_percent": "n/a"}) == pytest.approx(0.0)
    assert build_eo_matrix([{"id": 1, "selected_by_percent": "10.0"}])[1] > 0
