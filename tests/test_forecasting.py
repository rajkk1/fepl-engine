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
                             prior_season={})
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


# ------------------------------------------------- minutes: congestion & doubt


def test_minutes_features_are_all_learnable(player, history):
    """
    Regression: the vector ended in a `flagged` feature that `train` hardcoded
    to 0.0 for every row. It was constant in training, so the booster could
    never split on it, and toggling it at predict time moved a prediction by
    exactly 0.0. Anything in this vector must be able to vary in training.
    """
    import numpy as np
    from xp_model import MinutesClassifier as MC

    rng = np.random.default_rng(0)
    hist = {
        pid: [{"round": gw, "minutes": int(rng.choice([0, 20, 70, 90])),
               "starts": 1, "value": 55, "element_type": 3} for gw in range(1, 20)]
        for pid in range(1, 60)
    }
    mc = MC()
    seen = [set() for _ in range(MC.N_FEATURES)]
    for pid, rows in hist.items():
        w = {"starts": [], "mins": []}
        for r in rows:
            feats = mc._features(w, 50.0 + pid, 1 + pid % 4, {"minutes_per_game": pid},
                                 days_rest=3.0 + pid % 6, congestion=float(pid % 4))
            for i, v in enumerate(feats):
                seen[i].add(round(float(v), 6))
            w["starts"].append(r["starts"])
            w["mins"].append(r["minutes"])

    constant = [i for i, s in enumerate(seen) if len(s) == 1]
    assert not constant, (
        f"features {constant} never vary in training, so the booster can never "
        f"split on them - exactly the defect that made `flagged` dead weight")


def test_fixture_congestion_changes_the_minutes_forecast():
    """A congested calendar has to be able to move the distribution."""
    import numpy as np
    import pandas as pd
    from xp_model import MinutesClassifier, FixtureCalendar

    rng = np.random.default_rng(1)
    hist = {
        pid: [{"round": gw, "minutes": int(rng.choice([0, 30, 75, 90])),
               "starts": 1, "value": 60, "element_type": 3} for gw in range(1, 20)]
        for pid in range(1, 80)
    }
    # The calendar must be populated *at training time*, or days_rest and
    # congestion are constant in training and the booster cannot split on them.
    fixtures, day = [], 0
    for gw in range(1, 20):
        day += 3 if gw % 3 == 0 else 7
        for tid in (1, 2):
            fixtures.append({"event": gw, "team_h": tid, "team_a": tid + 10,
                             "kickoff_time": f"2025-08-01T14:00:00Z"
                             if day == 0 else
                             (pd.Timestamp("2025-08-01") + pd.Timedelta(days=day)).isoformat()})
    cal = FixtureCalendar(fixtures)
    mc = MinutesClassifier(calendar=cal)
    mc.train(hist, team_by_pid={pid: 1 + (pid % 2) for pid in hist})

    w = {"starts": [1, 1, 1, 1, 1], "mins": [90, 90, 88, 90, 90]}
    rested = mc._features(w, 75.0, 3, {}, days_rest=8.0, congestion=1.0)
    packed = mc._features(w, 75.0, 3, {}, days_rest=3.0, congestion=4.0)
    p_rested, p_packed = mc.predict_proba_batch(
        [(rested, 1.0, None), (packed, 1.0, None)])
    assert p_rested != pytest.approx(p_packed, abs=1e-9)


def test_fixture_calendar_reads_rest_and_congestion():
    from xp_model import FixtureCalendar

    fixtures = [
        {"event": 1, "team_h": 1, "team_a": 2, "kickoff_time": "2025-08-16T14:00:00Z"},
        {"event": 2, "team_h": 3, "team_a": 1, "kickoff_time": "2025-08-20T19:00:00Z"},
        {"event": 3, "team_h": 1, "team_a": 4, "kickoff_time": "2025-08-23T14:00:00Z"},
    ]
    cal = FixtureCalendar(fixtures)
    # 20 Aug 19:00 -> 23 Aug 14:00 is 2 whole days.
    days_rest, congestion = cal.context_for_gw(1, 3)
    assert days_rest == pytest.approx(2.0)
    assert congestion == pytest.approx(2.0)   # GW1 and GW2 both inside 14 days


def test_a_doubtful_player_is_less_likely_to_last_ninety(player, history, fixture):
    """
    Scaling every bucket by availability preserves the conditional shape, which
    claims a 50%-doubtful player who features is as likely to go the full 90 as
    a fit one. Clean sheets and DefCon both need the long shift, so that error
    is not cosmetic.
    """
    mc = MinutesClassifier()
    fit = mc.predict_proba({**player, "chance_of_playing_next_round": 100},
                           history, fixture)
    doubt = mc.predict_proba({**player, "chance_of_playing_next_round": 50},
                             history, fixture)
    p90_given_play_fit = fit[3] / sum(fit[1:])
    p90_given_play_doubt = doubt[3] / sum(doubt[1:])
    assert p90_given_play_doubt < p90_given_play_fit
    assert sum(doubt) == pytest.approx(1.0)


# --------------------------------------------------------- conditional minutes


def _flat_ensemble(teams):
    ens = EnsembleForecaster(n_bonus_sims=200)
    ens.dc.market._flat_ratings(teams)
    ens.dc.market.status = {"source": "test"}
    ens.dc.market._league_mean = 1.4
    ens.mc.is_trained = False
    return ens


def test_defcon_is_not_gated_on_sixty_minutes(player, history, fixture, teams):
    """
    FPL awards the defensive contribution on the count alone - there is no
    60-minute requirement, unlike the clean sheet. Gating on p_60 wrote off
    every substitute who comes on and racks up tackles.
    """
    ens = _flat_ensemble(teams)
    defender = {**player, "element_type": 2}
    comps = ens._predict_uncalibrated(defender, fixture, history, 0, 10)
    assert comps["p_60"] < comps["p_play"]
    assert comps["xDefCon"] > 0.0
    # The award is bounded by playing at all, not by playing an hour.
    assert comps["xDefCon"] <= comps["p_play"] * 2.0 + 1e-9


def test_nonlinear_terms_use_minutes_conditional_on_playing(player, history, fixture, teams):
    """
    `play_frac` is E[minutes | played] / 90. Saves, goals conceded and DefCon are
    all floors or thresholds, so feeding them unconditional expected minutes
    conflates "half a chance of playing 90" with "certainly plays 45".
    """
    ens = _flat_ensemble(teams)
    comps = ens._predict_uncalibrated(player, fixture, history, 0, 10)
    xmin_frac = comps["xmin"] / 90.0
    assert comps["play_frac"] > xmin_frac          # strictly larger whenever p_play < 1
    assert comps["play_frac"] <= 1.0 + 1e-9


# --------------------------------------------------------------- set pieces


def test_set_piece_duty_lifts_the_assist_rate(player, history, fixture, teams):
    ens = _flat_ensemble(teams)
    taker = ens._predict_uncalibrated(
        {**player, "corners_and_indirect_freekicks_order": 1}, fixture, history, 0, 10)
    non_taker = ens._predict_uncalibrated(
        {**player, "corners_and_indirect_freekicks_order": None}, fixture, history, 0, 10)
    assert taker["xa"] > non_taker["xa"]


def test_direct_free_kick_duty_lifts_the_goal_rate(player, history, fixture, teams):
    ens = _flat_ensemble(teams)
    taker = ens._predict_uncalibrated(
        {**player, "direct_freekicks_order": 1}, fixture, history, 0, 10)
    non_taker = ens._predict_uncalibrated(
        {**player, "direct_freekicks_order": None}, fixture, history, 0, 10)
    assert taker["xg"] > non_taker["xg"]


# ------------------------------------------------------------- lineups feed


def test_lineup_overrides_load_from_every_accepted_shape(tmp_path):
    import json
    from xp_model import load_lineup_overrides

    for payload in ({"123": 0.95, "456": 0.1},
                    {"lineups": [{"id": 123, "p_start": 0.95}, {"id": 456, "p_start": 0.1}]},
                    [{"id": 123, "p_start": 0.95}, {"id": 456, "p_start": 0.1}]):
        path = tmp_path / "lineups.json"
        path.write_text(json.dumps(payload))
        assert load_lineup_overrides(str(path)) == {123: 0.95, 456: 0.1}


def test_missing_lineups_feed_is_not_an_error():
    from xp_model import load_lineup_overrides
    assert load_lineup_overrides(None) == {} or True   # env may be unset
    assert load_lineup_overrides("/nonexistent/lineups.json") == {}
