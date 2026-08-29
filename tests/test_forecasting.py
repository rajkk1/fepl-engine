"""
Behavioural tests for the forecast layer.

Each test here pins a defect found in the teardown, so a regression fails the
build rather than silently degrading the forecast.
"""
import math

import numpy as np
import pytest

from xp_model import (
    GammaPoissonFilter, MinutesClassifier, MarketOddsPredictor, EnsembleForecaster,
    POINTS_GOAL, POINTS_CLEAN_SHEET,
)


# --------------------------------------------------------------- minutes model


def test_minutes_train_and_predict_use_the_same_cost_scale(player, history):
    """
    Regression: `train` read history["value"] (tenths, e.g. 155) while
    `predict` passed now_cost/10 (15.5), so every prediction fell below the
    lowest bin boundary the model had ever seen on that feature.
    """
    mc = MinutesClassifier()
    window = {"starts": [1, 1, 1], "mins": [90, 90, 90]}

    train_row = mc._features(window, cost_tenths=155.0, element_type=3,
                             prior_season={}, flagged=0.0)
    predict_row, _, _ = mc.build_row({**player, "now_cost": 155}, history)

    cost_idx = 2
    assert train_row[cost_idx] == pytest.approx(15.5)
    assert predict_row[cost_idx] == pytest.approx(15.5)


def test_minutes_features_have_declared_width(player, history):
    mc = MinutesClassifier()
    row, _, _ = mc.build_row(player, history)
    assert len(row) == MinutesClassifier.N_FEATURES


def test_injured_player_cannot_play(player, history, fixture):
    mc = MinutesClassifier()
    probs = mc.predict_proba({**player, "status": "i"}, history, fixture)
    assert probs[0] == pytest.approx(1.0)
    assert sum(probs[1:]) == pytest.approx(0.0)


def test_minutes_probabilities_are_a_distribution(player, history, fixture):
    mc = MinutesClassifier()
    for chance in (None, 100, 75, 25, 0):
        probs = mc.predict_proba({**player, "chance_of_playing_next_round": chance},
                                 history, fixture)
        assert sum(probs) == pytest.approx(1.0)
        assert all(p >= 0 for p in probs)


def test_lineup_override_shifts_start_probability(player, history, fixture):
    """
    Integration point for a predicted-lineups feed. No such feed ships with the
    repo, so this pins the contract rather than any particular source.
    """
    mc = MinutesClassifier()
    baseline = mc.predict_proba(player, history, fixture)

    mc.lineup_overrides = {player["id"]: 1.0}
    starting = mc.predict_proba(player, history, fixture)
    mc.lineup_overrides = {player["id"]: 0.0}
    benched = mc.predict_proba(player, history, fixture)

    assert starting[3] > baseline[3] >= 0.0
    assert benched[3] < baseline[3]
    assert all(sum(p) == pytest.approx(1.0) for p in (baseline, starting, benched))


def test_lineup_override_cannot_start_an_injured_player(player, history, fixture):
    mc = MinutesClassifier()
    mc.lineup_overrides = {player["id"]: 1.0}
    probs = mc.predict_proba({**player, "status": "i"}, history, fixture)
    assert probs[0] == pytest.approx(1.0)


def test_batched_and_single_minutes_predictions_agree(player, history, fixture):
    mc = MinutesClassifier()
    rows = [mc.build_row(player, history, fixture) for _ in range(3)]
    batched = mc.predict_proba_batch(rows)
    single = mc.predict_proba(player, history, fixture)
    assert batched[0] == pytest.approx(single)


# ------------------------------------------------------------ rate estimation


def _market(teams):
    dc = MarketOddsPredictor()
    dc.market._flat_ratings(teams)
    dc.market.status = {"source": "test", "degraded": True}
    return dc


def test_missing_stat_is_not_an_observed_zero(player, teams):
    """
    Regression: absent columns were fed to the filter as 0, so the posterior
    decayed toward nothing instead of holding at the prior. 2024-25 has no
    CBIT/tackles/recoveries columns, which silently disabled DefCon.
    """
    gpf = GammaPoissonFilter()
    dc = _market(teams)

    with_zeros = [{"round": gw, "minutes": 90, "was_home": True, "opponent_team": 2,
                   "clearances_blocks_interceptions": 0, "tackles": 0, "recoveries": 0}
                  for gw in range(1, 10)]
    without_field = [{"round": gw, "minutes": 90, "was_home": True, "opponent_team": 2}
                     for gw in range(1, 10)]

    observed = gpf.predict_match({**player, "element_type": 2}, with_zeros, dc, 10)
    missing = gpf.predict_match({**player, "element_type": 2}, without_field, dc, 10)

    prior = gpf.pos_priors[2]["cbit"]
    assert observed["cbit90"] < 0.5 * prior, "observed zeros must pull the rate down"
    assert missing["cbit90"] == pytest.approx(prior, rel=0.02), \
        "a missing field must leave the prior untouched"


def test_rates_shrink_toward_prior_with_little_evidence(player, teams):
    gpf = GammaPoissonFilter()
    dc = _market(teams)
    one_game = [{"round": 9, "minutes": 90, "was_home": True, "opponent_team": 2,
                 "expected_goals": 2.0, "expected_assists": 0.0}]
    out = gpf.predict_match({**player, "element_type": 4}, one_game, dc, 10)
    assert out["xg90"] < 2.0, "a single outlier match must not become the estimate"
    assert out["xg90"] > gpf.pos_priors[4]["xg"], "but it must move the estimate up"


def test_high_volume_player_retains_most_of_their_rate(player, teams):
    gpf = GammaPoissonFilter()
    dc = _market(teams)
    hist = [{"round": gw, "minutes": 90, "was_home": True, "opponent_team": 2,
             "expected_goals": 0.80, "expected_assists": 0.10} for gw in range(1, 16)]
    out = gpf.predict_match({**player, "element_type": 4}, hist, dc, 16)
    assert out["xg90"] > 0.55, "sustained evidence should survive shrinkage"


# ------------------------------------------------------------- fixture effects


def test_fixture_context_favours_the_stronger_side(teams):
    dc = MarketOddsPredictor()
    dc.market._fill_missing(teams)
    dc.market.team_ratings[1].update(att_home=2.2, att_away=1.9, def_home=0.8, def_away=1.0)
    dc.market.team_ratings[2].update(att_home=1.0, att_away=0.8, def_home=1.8, def_away=2.0)
    dc.market._league_mean = 1.4

    strong = dc.fixture_context(1, {"event": 1, "team_h": 1, "team_a": 2})
    weak = dc.fixture_context(2, {"event": 1, "team_h": 1, "team_a": 2})
    assert strong["team_xg"] > weak["team_xg"]
    assert strong["att_mult"] > weak["att_mult"]
    # The weaker side faces more threat, so does more defensive work.
    assert weak["def_mult"] > strong["def_mult"]


def test_defcon_is_opponent_adjusted(player, teams, fixture):
    """DefCon volume must respond to the fixture, as attacking rates already do."""
    ens = EnsembleForecaster()
    ens.dc.market._fill_missing(teams)
    ens.dc.market._league_mean = 1.4
    ens.mc.is_trained = False

    hist = [{"round": gw, "minutes": 90, "was_home": True, "opponent_team": 2,
             "starts": 1, "clearances_blocks_interceptions": 6, "tackles": 3,
             "recoveries": 5, "expected_goals": 0.05, "expected_assists": 0.05}
            for gw in range(1, 10)]
    defender = {**player, "element_type": 2}

    ens.dc.market.team_ratings[2].update(att_home=2.4, att_away=2.2)
    tough = ens._predict_uncalibrated(defender, fixture, hist, 0, 10)
    ens.clear_minutes_cache()
    ens.dc.market.team_ratings[2].update(att_home=0.7, att_away=0.6)
    easy = ens._predict_uncalibrated(defender, fixture, hist, 0, 10)

    assert tough["defcon_mu"] > easy["defcon_mu"], \
        "defensive volume should rise against a stronger attack"


def test_penalty_duty_raises_expected_goals(player, teams, fixture):
    ens = EnsembleForecaster()
    ens.dc.market._fill_missing(teams)
    ens.dc.market._league_mean = 1.4
    ens.mc.is_trained = False

    hist = [{"round": gw, "minutes": 90, "was_home": True, "opponent_team": 2,
             "starts": 1, "expected_goals": 0.25, "expected_assists": 0.10}
            for gw in range(1, 10)]

    taker = ens._predict_uncalibrated({**player, "element_type": 4, "penalties_order": 1},
                                      fixture, hist, 0, 10)
    ens.clear_minutes_cache()
    non_taker = ens._predict_uncalibrated({**player, "element_type": 4, "penalties_order": None},
                                          fixture, hist, 0, 10)
    assert taker["xg"] > non_taker["xg"]


def test_assist_fixture_multiplier_is_symmetric(player, teams):
    """
    Regression: history was normalised by the full attacking multiplier while
    predictions were scaled by a damped one, shaving 6-13% off the assist rate of
    every strong-attack team -- precisely where premium players are, and the
    single largest contributor to their under-prediction.

    A player whose past fixtures match the upcoming one must come back with the
    rate they actually demonstrated.
    """
    from xp_model import assist_multiplier

    gpf = GammaPoissonFilter()
    dc = MarketOddsPredictor()
    dc.market._fill_missing(teams)
    dc.market._league_mean = 1.4
    # A strong attack: every fixture carries att_mult > 1.
    for tid in dc.market.team_ratings:
        dc.market.team_ratings[tid].update(att_home=2.2, att_away=2.0,
                                           def_home=1.2, def_away=1.2)

    hist = [{"round": gw, "minutes": 90, "was_home": True, "opponent_team": 2,
             "expected_assists": 0.40, "expected_goals": 0.10} for gw in range(1, 13)]
    rates = gpf.predict_match({**player, "team": 1}, hist, dc, 13)

    ctx = dc.fixture_context(1, {"event": 13, "team_h": 1, "team_a": 2})
    recovered = rates["xa90"] * assist_multiplier(ctx["att_mult"])
    assert recovered == pytest.approx(0.40, rel=0.12), (
        f"demonstrated 0.40 xA90 came back as {recovered:.3f}"
    )


def test_assist_multiplier_is_softer_than_the_goal_multiplier():
    from xp_model import assist_multiplier

    for m in (1.2, 1.5):
        assert 1.0 < assist_multiplier(m) < m
    for m in (0.7, 0.9):
        assert m < assist_multiplier(m) < 1.0
    assert assist_multiplier(1.0) == pytest.approx(1.0)


# ------------------------------------------------------------------ integrity


def test_components_are_finite_and_sane(player, history, fixture, teams):
    ens = EnsembleForecaster()
    ens.dc.market._fill_missing(teams)
    ens.dc.market._league_mean = 1.4
    ens.mc.is_trained = False

    for et in (1, 2, 3, 4):
        c = ens._predict_uncalibrated({**player, "element_type": et}, fixture, history, 0, 10)
        ens.clear_minutes_cache()
        assert all(math.isfinite(v) for v in c.values() if isinstance(v, float))
        assert 0.0 <= c["p_play"] <= 1.0
        assert 0.0 <= c["p_60"] <= c["p_play"] + 1e-9
        assert 0.0 <= c["p_cs"] <= 1.0
        assert -5.0 < c["math_pts"] < 25.0


def test_second_match_of_a_double_gameweek_is_discounted(player, history, fixture, teams):
    ens = EnsembleForecaster()
    ens.dc.market._fill_missing(teams)
    ens.dc.market._league_mean = 1.4
    ens.mc.is_trained = False

    first = ens._predict_uncalibrated(player, fixture, history, 0, 10)
    second = ens._predict_uncalibrated(player, fixture, history, 1, 10)
    assert second["xmin"] < first["xmin"]
    # p_play, p_60 and xMin must stay mutually consistent after the discount.
    assert second["p_60"] <= second["p_play"] + 1e-9
    assert second["xmin"] <= 90.0


def test_scoring_tables_match_fpl_rules():
    assert POINTS_GOAL == {1: 10, 2: 6, 3: 5, 4: 4}
    assert POINTS_CLEAN_SHEET == {1: 4, 2: 4, 3: 1, 4: 0}


def test_scoring_tables_agree_across_modules():
    """Regression: monte_carlo scored a keeper goal as 6 while the table said 10."""
    import match_sim
    assert match_sim.POINTS_GOAL[1] == POINTS_GOAL[1] == 10
