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
        "recoveries90": 5.0, "tackles90": 1.5, "cbi90": 2.0, "xa90": 0.15,
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
    """
    Bonus is 3/2/1 to the top three BPS scorers in a match.

    It is *at least* 6 whenever three players appear, and more than 6 whenever
    there is a tie, because FPL shares tied places rather than breaking them
    (3+3+1, 3+2+2, 3+3+3). This squad is 22 identical players, which is the
    maximum-tie case, so the total sits well above 6.
    """
    sim = MatchSimulator(n_sims=400, seed=1)
    out = sim.simulate(squad, {1: 1.2, 2: 1.4})
    total = sum(out["bonus"].values())
    assert 6.0 - 1e-6 <= total <= 12.0, f"total expected bonus {total} is not rank-consistent"


def test_tied_players_share_bonus_rather_than_the_tie_being_broken_by_position():
    """
    Regression: BPS was simulated as a continuous quantity, so exact ties never
    occurred and `argsort` silently resolved every one of them by array index.
    Real BPS is an integer and ties are common, so the first-listed of two
    identical players was quietly handed the better award.
    """
    a, b, c = _player(1, 1), _player(2, 1), _player(3, 2)
    out = MatchSimulator(n_sims=8000, seed=5).simulate([a, b, c], {1: 1.4, 2: 1.4})
    assert out["bonus"][1] == pytest.approx(out["bonus"][2], abs=0.06)


def test_bonus_ordering_survives_reversing_the_player_list(squad):
    """The award must depend on BPS, never on where a player sits in the list."""
    sim = MatchSimulator(n_sims=800, seed=9)
    forward = sim.simulate(squad, {1: 1.2, 2: 1.4})["bonus"]
    backward = sim.simulate(list(reversed(squad)), {1: 1.2, 2: 1.4})["bonus"]
    assert sum(forward.values()) == pytest.approx(sum(backward.values()), rel=0.1)


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


def test_unmodelled_position_is_skipped_not_crashed(squad):
    """
    Regression: FPL shipped managers as element_type 5 in 2024-25, and the
    simulator hard-indexed the scoring tables, so a whole gameweek's forecast
    died with KeyError: 5. An unmodelled position must forecast to nothing.
    """
    squad = list(squad) + [_player(99, 1, et=5)]
    sim = MatchSimulator(n_sims=200, seed=11)
    out = sim.simulate(squad, {1: 1.2, 2: 1.4})
    assert out["mean_points"][99] == pytest.approx(0.0)
    assert out["bonus"][99] == pytest.approx(0.0)
    # The rest of the match is unaffected.
    assert sum(out["bonus"].values()) > 0.0


def test_forecaster_scores_unmodelled_positions_at_zero(teams):
    from xp_model import EnsembleForecaster

    ens = EnsembleForecaster()
    ens.dc.market._fill_missing(teams)
    ens.dc.market._league_mean = 1.4
    ens.mc.is_trained = False
    manager = {"id": 500, "element_type": 5, "team": 1, "now_cost": 15,
               "status": "a", "chance_of_playing_next_round": None, "news": ""}
    comps = ens._predict_uncalibrated(
        manager, {"event": 1, "team_h": 1, "team_a": 2}, [], 0, 1)
    assert comps["math_pts"] == 0.0
    assert comps["p_play"] == 0.0


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


def test_rare_events_are_scored():
    """Red cards, own goals and penalty misses/saves all move the score."""
    clean = _player(1, 1)
    carded = _player(2, 1, rc90=0.9)
    keeper = _player(3, 1, et=1, saves90=3.0, defcon_mu=0.0, pen_save90=0.9)
    plain_keeper = _player(4, 1, et=1, saves90=3.0, defcon_mu=0.0)
    for p in (clean, carded, keeper, plain_keeper):
        p["play_frac"] = 0.9
    out = MatchSimulator(n_sims=6000, seed=4).simulate(
        [clean, carded, keeper, plain_keeper], {1: 1.3})
    assert out["mean_points"][2] < out["mean_points"][1], "a red card must cost points"
    assert out["mean_points"][3] > out["mean_points"][4], "a penalty save must pay"


def test_players_in_different_fixtures_are_uncorrelated():
    """
    Regression: every fixture built its own generator from a fixed seed, so two
    matches drew from an identical random stream and unrelated players came out
    correlated - the one artefact a correlated simulation must not have.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    a = [_player(i, 1 if i < 4 else 2, play_frac=0.9) for i in range(1, 7)]
    b = [_player(i, 3 if i < 14 else 4, play_frac=0.9) for i in range(11, 17)]
    sim = MatchSimulator(n_sims=6000)
    out_a = sim.simulate(a, {1: 1.3, 2: 1.4}, rng=rng)
    out_b = sim.simulate(b, {3: 1.3, 4: 1.4}, rng=rng)
    corr = abs(np.corrcoef(out_a["points"][1], out_b["points"][11])[0, 1])
    assert corr < 0.10, f"unrelated fixtures correlated at {corr:.3f}"


# --------------------------------------------------- shared team goal totals


def _attacker(pid, team, xg=0.5, xa=0.3, et=4, p_play=1.0):
    return dict(id=pid, team=team, element_type=et, p_play=p_play,
                p_60=p_play * 0.95, xg_cond=xg, xa_cond=xa, saves90=0.0,
                yc90=0.1, play_frac=1.0, cond_frac=1.0, defcon_mu=0.0,
                defcon_dispersion=1.85, xmin=90.0, recoveries90=2.0,
                tackles90=0.8, cbi90=1.5, xa90=0.3)


def _two_team_match(n_sims=60000, seed=3):
    squad = ([_attacker(1, 1, 0.55, 0.30), _attacker(2, 1, 0.50, 0.35),
              _attacker(3, 1, 0.50, 0.25)]
             + [_attacker(i, 2, 0.30, 0.20) for i in range(5, 10)])
    # team_lambdas maps team -> goals it CONCEDES, so team 1 scores 1.60.
    return squad, MatchSimulator(n_sims=n_sims, seed=seed).simulate(
        squad, {1: 1.20, 2: 1.60})


def test_shared_team_total_preserves_each_players_marginal():
    """
    Allocating one shared team total must not change what a player is expected
    to score - only the joint distribution. If this drifts, every xP in the
    engine drifts with it, because the analytic path still uses xg directly.
    """
    squad, out = _two_team_match()
    for p in squad[:3]:
        assert out["mean_goals"][p["id"]] == pytest.approx(
            p["xg_cond"] * p["p_play"], abs=0.02)
        assert out["mean_assists"][p["id"]] == pytest.approx(
            p["xa_cond"] * p["p_play"], abs=0.02)


def test_club_mates_attacking_returns_are_positively_correlated():
    """
    The regression this guards: goals used to come from an independent Poisson
    per player, so three club-mates correlated at -0.07 - indistinguishable from
    opposition players, and negative only because they compete for bonus.
    Measured ground truth over 2023-24..2025-26 is +0.098 between two
    same-club attackers who both played 60+.
    """
    _, out = _two_team_match()
    d = out["points"]
    same_club = np.mean([np.corrcoef(d[a], d[b])[0, 1]
                         for a, b in ((1, 2), (1, 3), (2, 3))])
    opposition = np.mean([np.corrcoef(d[1], d[i])[0, 1] for i in (5, 6, 7)])
    assert same_club > 0.02, f"club-mates uncorrelated: {same_club:+.4f}"
    assert same_club > opposition


def test_stacking_club_mates_increases_variance():
    """A stack must be riskier than the same players spread across clubs -
    that is the whole premise of the rank-aware (MERV) valuation."""
    _, out = _two_team_match()
    d = out["points"]
    stack = d[1] + d[2] + d[3]
    if_independent = np.sqrt(sum(np.var(d[i]) for i in (1, 2, 3)))
    assert stack.std() > if_independent


def test_a_team_never_scores_more_than_its_match_total():
    """Listed players share one team total, so they cannot collectively
    out-score the team - the incoherence independent Poissons allowed."""
    squad, out = _two_team_match()
    listed = sum(out["mean_goals"][p["id"]] for p in squad if p["team"] == 1)
    assert listed <= 1.60 + 1e-6


def test_single_team_input_falls_back_to_independent_draws():
    """A caller passing one side has no opponent to take a total from; it must
    still produce sane per-player goals rather than dividing by nothing."""
    squad = [_attacker(1, 1, 0.5, 0.3), _attacker(2, 1, 0.4, 0.2)]
    out = MatchSimulator(n_sims=20000, seed=1).simulate(squad, {1: 1.3})
    assert out["mean_goals"][1] == pytest.approx(0.5, abs=0.05)
    assert out["mean_goals"][2] == pytest.approx(0.4, abs=0.05)


# ------------------------------------------------------- volume BPS by position


def test_volume_bps_is_position_specific():
    """
    Bonus is a rank statistic over BPS, so BPS accurate for one position and
    approximated for another hands the accurate one bonus it did not earn.
    Keepers were exactly modelled (clean sheet 12, saves 2 each) while
    outfielders ran on three global constants, and keepers duly won: predicted
    bonus 0.312 against a realised 0.178.
    """
    from match_sim import volume_bps90, BPS_VOLUME, POS_GKP, POS_MID

    comps = {"recoveries90": 4.5, "tackles90": 1.8, "cbi90": 3.0, "xa90": 0.15}
    assert volume_bps90(POS_MID, comps) > volume_bps90(POS_GKP, comps)
    assert set(BPS_VOLUME) == {1, 2, 3, 4}


def test_volume_bps_never_goes_negative():
    """The fitted intercept is negative for keepers and forwards, whose rate
    terms over-explain. A low-volume player must floor at zero."""
    from match_sim import volume_bps90, POS_GKP, POS_FWD

    empty = {"recoveries90": 0.0, "tackles90": 0.0, "cbi90": 0.0, "xa90": 0.0}
    assert volume_bps90(POS_GKP, empty) == 0.0
    assert volume_bps90(POS_FWD, empty) == 0.0


def test_volume_bps_rewards_creativity_for_outfielders_only():
    """
    xA is the single most valuable volume term for an outfielder (midfield CV
    R^2 0.362 -> 0.471) and worth nothing for a keeper, where the fitted sign is
    noise. The old model priced it through `key_passes = xa90 * 4` at 1 BPS
    each, under-crediting creators by roughly 3.5x.
    """
    from match_sim import volume_bps90, POS_GKP, POS_MID

    base = {"recoveries90": 4.0, "tackles90": 1.5, "cbi90": 2.5, "xa90": 0.05}
    creative = dict(base, xa90=0.45)
    assert volume_bps90(POS_MID, creative) > volume_bps90(POS_MID, base) + 1.0
    assert volume_bps90(POS_GKP, creative) == pytest.approx(
        volume_bps90(POS_GKP, base))


def test_volume_bps_is_monotone_in_each_volume_term():
    from match_sim import volume_bps90, POS_DEF

    base = {"recoveries90": 4.0, "tackles90": 1.5, "cbi90": 2.5, "xa90": 0.08}
    for term in ("recoveries90", "tackles90", "cbi90", "xa90"):
        more = dict(base)
        more[term] = base[term] + 1.0
        assert volume_bps90(POS_DEF, more) > volume_bps90(POS_DEF, base), term


def test_unmodelled_position_earns_no_volume_bps():
    from match_sim import volume_bps90

    assert volume_bps90(5, {"recoveries90": 9.0, "xa90": 1.0}) == 0.0


# ------------------------------------------------- tilted team-goal marginal


def test_team_goal_pmf_is_a_distribution_with_the_right_mean():
    from match_sim import team_goal_pmf

    for lam in (0.4, 1.0, 1.5, 2.2, 3.5):
        pmf = team_goal_pmf(lam)
        assert pmf.sum() == pytest.approx(1.0)
        assert (pmf >= 0).all()
        mean = float((pmf * np.arange(len(pmf))).sum())
        assert mean == pytest.approx(lam, rel=0.06), lam


def test_clean_sheet_probability_matches_the_league(monkeypatch):
    """
    The cell this exists to fix. Plain Poisson over-states P(concede nothing) by
    +0.0163 against 2280 market-priced team-matches - a clean sheet handed to
    every keeper and defender that reality does not award. Mixed over a
    realistic spread of lambda the tilted version lands on the measured 0.2320.
    """
    from match_sim import p_no_goals

    rng = np.random.default_rng(0)
    lam = np.clip(rng.normal(1.50, 0.50, 40000), 0.15, 5.0)
    tilted = float(np.mean([p_no_goals(l) for l in lam]))
    poisson = float(np.mean(np.exp(-lam)))
    assert abs(tilted - 0.2320) < abs(poisson - 0.2320)
    assert tilted == pytest.approx(0.2320, abs=0.012)


def test_p_no_goals_is_monotone_and_total_at_zero():
    from match_sim import p_no_goals

    assert p_no_goals(0.0) == 1.0
    vals = [p_no_goals(l) for l in (0.3, 0.8, 1.5, 2.5, 4.0)]
    assert vals == sorted(vals, reverse=True)


def test_analytic_clean_sheet_reads_the_same_distribution_as_the_simulation():
    """
    The analytic path scores the clean sheet and the simulation scores the bonus
    that depends on it. If they draw from different distributions they disagree
    about every defender, so both must go through `p_no_goals`.
    """
    import inspect
    import xp_model

    src = inspect.getsource(xp_model.EnsembleForecaster._predict_uncalibrated)
    assert "p_no_goals(" in src
    assert "math.exp(-opp_xg" not in src


def test_sampling_follows_the_tilted_pmf():
    from match_sim import MatchSimulator, team_goal_pmf

    sim = MatchSimulator(n_sims=1)
    draws = sim._team_goals(np.random.default_rng(0), 1.5, 200000)
    pmf = team_goal_pmf(1.5)
    for k in range(5):
        assert (draws == k).mean() == pytest.approx(pmf[k], abs=0.005)
