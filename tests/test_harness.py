"""
Backtest harness, calibration, and market-odds fallbacks.

The harness is what decides whether a model change helped, so its own
correctness matters more than the model's.
"""
import numpy as np
import pandas as pd
import pytest

from calibration import PointsCalibrator
from market_odds import MarketOddsModel
from backtest import build_baselines, build_eval_population, check_gate, evaluate_gameweek


def _gw_frame():
    rows = []
    for gw in range(1, 8):
        for pid in range(1, 21):
            rows.append({
                "GW": gw, "element": pid,
                "total_points": (pid % 7) + gw % 3,
                "minutes": 90 if pid % 4 else 0,
                "selected": 1000 * (21 - pid),
                "value": 50 + pid,
                "xP": 2.0,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ baselines


def test_baselines_are_strictly_point_in_time():
    """
    Regression: the previous baseline was FPL's post-hoc `xP` column, which is
    zeroed for players who did not feature and therefore leaked realised minutes.
    Every baseline must be computable from gameweeks before the target only.
    """
    df = _gw_frame()
    target = 5
    b_full = build_baselines(df, target)
    # Corrupting the future must not change a point-in-time baseline.
    df_corrupt = df.copy()
    mask = df_corrupt["GW"] >= target
    df_corrupt.loc[mask, "total_points"] = 999
    df_corrupt.loc[mask, "minutes"] = 0
    b_corrupt = build_baselines(df_corrupt, target)

    for name in ("ppg", "roll3", "roll3_mins"):
        assert b_full[name] == b_corrupt[name], f"{name} reads the future"


def test_eval_population_does_not_depend_on_the_model():
    """
    Regression: the population unioned in the top 30 per position *by predicted
    xP*, so changing the model changed which players were scored - and made the
    fixed baseline's own score drift between runs.
    """
    df = _gw_frame()
    pop_a = build_eval_population(df, 5)
    pop_b = build_eval_population(df, 5)
    assert pop_a == pop_b and len(pop_a) > 0
    # The signature takes no model input at all, which is the structural guarantee.
    import inspect
    params = set(inspect.signature(build_eval_population).parameters)
    assert params == {"df_gw", "target_gw", "top_n"}


def test_metrics_reward_the_better_forecast():
    actual = {1: 10.0, 2: 6.0, 3: 2.0, 4: 0.0}
    positions = {i: 3 for i in actual}
    costs = {i: 60 for i in actual}
    good = {1: 9.0, 2: 6.5, 3: 2.5, 4: 0.5}
    bad = {1: 0.5, 2: 2.0, 3: 6.0, 4: 9.0}

    res = evaluate_gameweek({"good": good, "bad": bad}, actual, positions, costs,
                            list(actual), p_play={}, played={})
    assert res["models"]["good"]["mae"] < res["models"]["bad"]["mae"]
    assert res["models"]["good"]["spearman"] > res["models"]["bad"]["spearman"]
    assert res["models"]["good"]["captain_regret"] < res["models"]["bad"]["captain_regret"]


def test_gate_ignores_the_leaking_baseline():
    summary = {
        "models": {
            "fepl": {"mae": 1.8},
            "ppg": {"mae": 2.0},
            "roll3": {"mae": 2.1},
            "roll3_mins": {"mae": 2.0},
            "fpl_xp_LEAKS": {"mae": 1.2},   # better, but not a fair target
        }
    }
    ok, msg = check_gate(summary, "mae")
    assert ok, msg


def test_gate_fails_when_a_clean_baseline_wins():
    summary = {"models": {"fepl": {"mae": 2.5}, "ppg": {"mae": 2.0},
                          "roll3": {"mae": 2.6}, "roll3_mins": {"mae": 2.6}}}
    ok, msg = check_gate(summary, "mae")
    assert not ok and "ppg" in msg


# ---------------------------------------------------------------- calibration


def _samples(slope, intercept, n=800):
    rng = np.random.default_rng(0)
    preds = rng.uniform(0, 8, n)
    return [{"element_type": 3, "pred": float(p),
             "actual": float(slope * p + intercept + rng.normal(0, 0.5))}
            for p in preds]


def test_linear_calibration_corrects_a_biased_slope():
    cal = PointsCalibrator(method="linear").fit(_samples(0.6, 1.0))
    assert cal.is_fitted
    # Raw 8.0 was over-predicting badly; calibration should pull it toward truth.
    assert cal.apply(8.0, 3) == pytest.approx(0.6 * 8.0 + 1.0, abs=0.4)


def test_calibration_never_reorders_players():
    cal = PointsCalibrator(method="linear").fit(_samples(0.7, 0.5))
    raw = [0.5, 1.0, 2.0, 4.0, 6.0, 9.0]
    out = [cal.apply(v, 3) for v in raw]
    assert out == sorted(out)


def test_isotonic_calibration_keeps_strict_ordering():
    """
    Isotonic is monotone non-decreasing, so it maps whole ranges onto one
    plateau - fatal at the top of the ranking where captaincy is decided. The
    tie-break keeps the ordering strict.
    """
    cal = PointsCalibrator(method="isotonic").fit(_samples(0.7, 0.5))
    raw = [6.0, 6.2, 6.4, 6.6, 6.8, 7.0]
    out = [cal.apply(v, 3) for v in raw]
    assert len(set(out)) == len(out), "isotonic plateau collapsed distinct players"
    assert out == sorted(out)


def test_calibration_refuses_a_single_gameweek():
    """
    One gameweek is a realisation of fixture and rotation noise, not a
    calibration set - fitting on it mostly compresses the forecast toward that
    week's mean. This matters in GW2, when the window is necessarily tiny.
    """
    one_week = [{**s, "gw": 1} for s in _samples(0.6, 1.0)]
    cal = PointsCalibrator(method="linear").fit(one_week)
    assert not cal.is_fitted
    assert cal.apply(7.0, 3) == 7.0

    spread = [{**s, "gw": 1 + i % 4} for i, s in enumerate(_samples(0.6, 1.0))]
    cal2 = PointsCalibrator(method="linear").fit(spread)
    assert cal2.is_fitted


def test_prior_season_rates_weight_by_minutes_played():
    """
    A player with a full prior season behind them should lean on their own rate;
    one with a handful of appearances should not. A flat 50/50 blend under-uses
    the strong case, which is exactly the GW1-5 window that matters most.
    """
    from xp_model import GammaPoissonFilter, MarketOddsPredictor

    gpf = GammaPoissonFilter()
    dc = MarketOddsPredictor()
    dc.market._fill_missing([{"id": i} for i in range(1, 5)])
    dc.market._league_mean = 1.4

    p = {"id": 1, "element_type": 4, "team": 1, "now_cost": 140}
    elite = {"xg": 0.95, "minutes": 3000.0}
    cameo = {"xg": 0.95, "minutes": 200.0}

    heavy = gpf.predict_match(p, [], dc, 2, season_prior=elite)
    light = gpf.predict_match(p, [], dc, 2, season_prior=cameo)
    assert heavy["xg90"] > light["xg90"]


def test_unfitted_calibrator_is_the_identity():
    cal = PointsCalibrator()
    assert cal.apply(4.2, 3) == 4.2
    cal_none = PointsCalibrator(method="none").fit(_samples(0.6, 1.0))
    assert not cal_none.is_fitted
    assert cal_none.apply(4.2, 3) == 4.2


def test_season_is_derived_when_the_caller_omits_it():
    """
    Regression: the prior-season carry-forward was gated on an explicit `season`
    argument, and weekly_manager does not pass one. In production the fallback
    silently never fired, leaving the live forecast on a much weaker fit during
    exactly the early-season window it exists to cover.
    """
    from xp_model import _current_season_start_year

    # A season runs Jul-Jun, so both sides of the new year map to the same start.
    assert _current_season_start_year("2026-08-15") == 2026
    assert _current_season_start_year("2027-01-15") == 2026
    assert _current_season_start_year("2026-06-30") == 2025
    assert _current_season_start_year("2026-07-01") == 2026
    # Falls back to the clock rather than raising on unusable input.
    assert isinstance(_current_season_start_year(None), int)
    assert isinstance(_current_season_start_year("not a date"), int)


# --------------------------------------------------------------- market odds


def test_missing_odds_are_flagged_not_silently_flattened(teams):
    """
    Regression: an empty odds file left every club on 1.4/1.4, so every fixture
    looked identical while the engine kept emitting confident numbers.
    """
    m = MarketOddsModel()
    m.odds_df = None
    status = m.fit_team_ratings(fpl_teams=teams)
    assert status["source"] == "flat_default"
    assert m.is_degraded(), "a flat fixture model must announce itself"


def test_prior_season_ratings_are_used_before_giving_up(teams):
    m = MarketOddsModel()
    m.odds_df = None
    prior = {t["id"]: {"att_home": 2.0, "att_away": 1.8,
                       "def_home": 0.9, "def_away": 1.1} for t in teams}
    status = m.fit_team_ratings(fpl_teams=teams, prior_ratings=prior)
    assert status["source"] == "prior_season"
    assert not m.is_degraded()
    # Regressed halfway to the league mean, not carried over wholesale.
    assert 1.4 < m.team_ratings[teams[0]["id"]]["att_home"] < 2.0


def test_results_fallback_when_there_are_no_odds(teams):
    m = MarketOddsModel()
    m.odds_df = None
    results = pd.DataFrame([
        {"team_h": 1, "team_a": 2, "team_h_score": 3, "team_a_score": 0},
        {"team_h": 2, "team_a": 1, "team_h_score": 0, "team_a_score": 2},
    ] * 6)
    status = m.fit_team_ratings(fpl_teams=teams, results_df=results)
    assert status["source"] == "results_poisson"
    assert m.team_ratings[1]["scored"] > m.team_ratings[2]["scored"]


def test_home_advantage_is_not_applied_twice(teams):
    """
    Regression: ratings averaged over matches that already included home
    fixtures, then get_match_lambdas re-applied a fixed 1.10/0.90 on top.
    """
    m = MarketOddsModel()
    m._fill_missing(teams)
    m._league_mean = 1.4
    for tid in (1, 2):
        m.team_ratings[tid].update(att_home=1.4, att_away=1.4,
                                   def_home=1.4, def_away=1.4)
    mu_h, mu_a = m.get_match_lambdas(1, 2)
    assert mu_h == pytest.approx(1.4, abs=1e-6)
    assert mu_a == pytest.approx(1.4, abs=1e-6)


# ------------------------------------------------------------- calibration


def test_calibration_applies_the_intercept_once_per_fixture():
    """
    Regression: a double gameweek's two fixtures were summed and then calibrated
    once, so the intercept was applied a single time instead of twice. That
    systematically underrated exactly the DGW players the optimiser makes its
    biggest calls on.
    """
    from calibration import PointsCalibrator

    cal = PointsCalibrator(method="linear")
    cal._models = {3: (0.9, 0.5)}
    cal._kind = {3: "linear"}
    cal.is_fitted = True

    single = cal.apply(5.0, 3, n_fixtures=1)
    double = cal.apply(10.0, 3, n_fixtures=2)
    assert single == pytest.approx(0.9 * 5.0 + 0.5)
    assert double == pytest.approx(0.9 * 10.0 + 2 * 0.5)
    assert double == pytest.approx(2 * cal.apply(5.0, 3, n_fixtures=1))


def test_gate_enforces_decision_metrics_not_just_error():
    """
    MAE alone is dominated by correctly predicting low scores, so a model can
    clear it while being no better than the baseline at picking a squad.
    """
    from backtest import check_gate

    summary = {"models": {
        "fepl": {"mae": 1.5, "spearman": 0.40, "precision_at_15": 1.5},
        "ppg": {"mae": 2.0, "spearman": 0.55, "precision_at_15": 2.0},
        "roll3": {"mae": 2.1, "spearman": 0.50, "precision_at_15": 1.9},
        "roll3_mins": {"mae": 2.1, "spearman": 0.50, "precision_at_15": 1.9},
    }}
    assert check_gate(summary, ("mae",))[0] is True
    ok, msg = check_gate(summary)
    assert ok is False
    assert "spearman" in msg
