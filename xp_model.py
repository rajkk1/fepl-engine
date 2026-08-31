"""
Expected-points forecasting.

Structure:
    MarketOddsPredictor  - turns team-level market lambdas into player-level rates
    GammaPoissonFilter   - shrinks noisy per-90 rates toward informed priors
    MinutesClassifier    - P(0 / 1-59 / 60-89 / 90 minutes)
    EnsembleForecaster   - combines the above, then calibrates
    MatchSimulator       - (match_sim.py) supplies rank-based bonus and correlated risk

Notable corrections against the previous version:
  * minutes classifier trained and predicted on different units for `cost`
  * defensive contributions were not opponent-adjusted, unlike attacking rates
  * bonus was a linear map of E[BPS]; it is a rank statistic and is now simulated
  * missing stat columns were treated as observed zeros, collapsing rates
  * penalties were invisible; now an explicit term driven by `penalties_order`
  * no calibration existed despite the `_predict_uncalibrated` name
  * every player-fixture was forecast twice
"""
import logging
import math
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from functools import lru_cache

from fpl_api import get_bootstrap_static, get_fixtures, get_all_element_summaries
from market_odds import MarketOddsModel, LEAGUE_MEAN_GOALS
from match_sim import (
    MatchSimulator, POINTS_GOAL, POINTS_CLEAN_SHEET, POINTS_ASSIST,
    POINTS_RED, POINTS_OWN_GOAL, POINTS_PENALTY_MISS, POINTS_PENALTY_SAVE,
    POS_GKP, POS_DEF, POS_MID, POS_FWD, DEFCON_THRESHOLD, MODELLED_POSITIONS,
    defcon_active, BUCKET_MINUTES, p_no_goals,
)
from calibration import PointsCalibrator

logger = logging.getLogger(__name__)

# Penalties: league rate is roughly one every four matches, converted at ~0.79.
PEN_XG = 0.79
TEAM_PEN_RATE = 0.26
# P(this player takes a given penalty) by `penalties_order`.
PEN_ORDER_SHARE = {1: 0.90, 2: 0.12, 3: 0.04}
PEN_ORDER_DEFAULT = 0.01
# Effective 90s of history after which we assume the observed rate already
# reflects the player's current penalty duty.
PEN_EVIDENCE_K = 6.0
# Penalties respond to fixture strength, but less than open play does.
PEN_FIXTURE_DAMPING = 0.4
# Share of penalties that are not converted, and the share of those a keeper
# actually saves (the rest go off target and score the keeper nothing).
PEN_SAVE_PROB = 0.13

# Set-piece duty. FPL publishes the designated takers in `bootstrap-static`,
# and it is the strongest freely-available predictor of assist rate that a
# player's own xA history cannot see until it has already happened. That gap is
# widest exactly where it hurts most: a new signing, or a player who has just
# taken the duty over from someone who left.
#
# Handled the same way as penalties -- an explicit term, with the portion the
# history has already had a chance to observe subtracted back out first, so an
# established taker is not counted twice.
SP_ORDER_SHARE = {1: 0.85, 2: 0.15, 3: 0.05}
SP_ORDER_DEFAULT = 0.0
# Expected assists per match accruing to a team's primary corner/indirect
# free-kick taker, and expected goals to its primary direct free-kick taker.
SET_PIECE_XA_PER_MATCH = 0.105
DIRECT_FK_XG_PER_MATCH = 0.035
SP_EVIDENCE_K = 6.0
# A corner is a corner whoever the opponent is, so set pieces respond to
# fixture strength even more softly than penalties do.
SP_FIXTURE_DAMPING = 0.3

# BPS credits key passes, which FPL does not report per player. Assists are
# the only observable proxy, and roughly four key passes precede one expected
# assist. This only ever shifts a player within a bonus race, never their base
# Below this many priced matches, this season's odds cannot support ratings
# on their own and we fall back to carrying the previous season forward.
MIN_MATCHES_FOR_RATINGS = 40
# Prior-season minutes at which that season's rates carry half the weight of
# the blended prior. ~1200 minutes is roughly a third of a season.
PRIOR_SEASON_HALF_MINUTES = 1200.0


# Assists track chance creation, which responds to fixture strength more softly
# than goals do. The damping must be applied *symmetrically*: the same factor
# that scales a prediction up must be the one that normalises history back down.
# Dividing history by the full attacking multiplier while multiplying predictions
# by a damped one shaved 6-13% off the assist rate of every strong-attack team --
# which is exactly where premium players are.
ASSIST_FIXTURE_DAMPING = 0.65


def assist_multiplier(att_mult: float) -> float:
    return 1.0 + ASSIST_FIXTURE_DAMPING * (att_mult - 1.0)


def _current_season_start_year(reference_date=None) -> int:
    """The year a Premier League season starting in July/August belongs to."""
    import datetime

    if reference_date is not None:
        try:
            ref = pd.to_datetime(reference_date)
            return int(ref.year if ref.month >= 7 else ref.year - 1)
        except Exception:
            pass
    now = datetime.datetime.now()
    return now.year if now.month >= 7 else now.year - 1


def _f(value, default=0.0) -> float:
    """Coerce possibly-None / possibly-string numerics without lying about missing."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


class MarketOddsPredictor:
    """Fixture-conditioned player rates from market-implied team strength."""

    def __init__(self):
        self.market = MarketOddsModel()

    def fit_team_ratings(self, teams, current_gw_date=None, season_str=None,
                         prior_ratings=None, results_df=None):
        self.market.fetch_odds(season_str=season_str)
        return self.market.fit_team_ratings(
            fpl_teams=teams, current_gw_date=current_gw_date,
            prior_ratings=prior_ratings, results_df=results_df,
        )

    def fixture_context(self, team_id: int, fixture: Dict[str, Any]) -> Dict[str, float]:
        """Attacking and defensive multipliers for one team in one fixture."""
        is_home = fixture.get("team_h") == team_id
        home_id = fixture.get("team_h")
        away_id = fixture.get("team_a")
        lam, mu = self.market.get_match_lambdas(home_id, away_id)
        team_xg = lam if is_home else mu
        opp_xg = mu if is_home else lam

        att_baseline = self.market.team_attack_baseline(team_id)
        att_mult = team_xg / att_baseline if att_baseline > 0 else 1.0
        # Defensive workload scales with how much threat the opponent carries.
        def_mult = opp_xg / LEAGUE_MEAN_GOALS if LEAGUE_MEAN_GOALS > 0 else 1.0

        return {
            "team_xg": team_xg,
            "opp_xg": opp_xg,
            "att_mult": float(np.clip(att_mult, 0.5, 1.8)),
            "def_mult": float(np.clip(def_mult, 0.6, 1.7)),
            "is_home": is_home,
        }


class GammaPoissonFilter:
    """
    Time-decayed Gamma-Poisson shrinkage of per-90 rates.

    Two corrections over the original:
      * a statistic absent from the data source is *missing*, not an observed
        zero. Observed zeros pull the rate down; missing data must leave it at
        the prior. Feeding zeros for absent columns silently disabled DefCon on
        any season whose dataset lacks those columns.
      * half-life and prior strength are per-statistic. Minutes-like counts
        stabilise quickly; expected goals do not.
    """

    # (half_life_in_gws, prior_weight) per statistic.
    STAT_SETTINGS = {
        "xg":     (8.0, 2.0),
        "xa":     (8.0, 2.0),
        "cbit":   (5.0, 1.5),
        "cbirt":  (5.0, 1.5),
        "saves":  (6.0, 1.5),
        "yc":     (12.0, 2.0),
        # Red cards and own goals are rare enough that one occurrence would
        # dominate a short window, so they decay slowly and shrink hard toward
        # the positional prior.
        "rc":     (16.0, 6.0),
        "og":     (16.0, 6.0),
        "recoveries": (5.0, 1.5),
        "tackles":    (5.0, 1.5),
    }

    def __init__(self, half_life: float = 5.0, prior_weight: float = 1.0,
                 stat_settings: Optional[Dict[str, Tuple[float, float]]] = None):
        self.half_life = half_life
        self.prior_weight = prior_weight
        self.stat_settings = dict(self.STAT_SETTINGS)
        if stat_settings:
            self.stat_settings.update(stat_settings)

        self.pos_priors = {
            POS_GKP: {"xg": 0.00, "xa": 0.01, "cbit": 0.5, "cbirt": 0.5,
                      "saves": 2.5, "yc": 0.076, "rc": 0.003, "og": 0.002,
                      "recoveries": 2.0, "tackles": 0.1},
            POS_DEF: {"xg": 0.05, "xa": 0.08, "cbit": 7.45, "cbirt": 7.45,
                      "saves": 0.0, "yc": 0.165, "rc": 0.006, "og": 0.006,
                      "recoveries": 5.5, "tackles": 1.6},
            POS_MID: {"xg": 0.15, "xa": 0.15, "cbit": 1.0, "cbirt": 7.86,
                      "saves": 0.0, "yc": 0.134, "rc": 0.005, "og": 0.002,
                      "recoveries": 5.0, "tackles": 1.5},
            POS_FWD: {"xg": 0.40, "xa": 0.15, "cbit": 0.5, "cbirt": 4.09,
                      "saves": 0.0, "yc": 0.088, "rc": 0.004, "og": 0.001,
                      "recoveries": 3.0, "tackles": 0.9},
        }

    def _settings(self, stat: str) -> Tuple[float, float]:
        return self.stat_settings.get(stat, (self.half_life, self.prior_weight))

    def predict_match(self, player, history, market_predictor, current_gw,
                      season_prior: Optional[Dict[str, float]] = None):
        pos = player.get("element_type", POS_MID)
        prior = dict(self.pos_priors.get(pos, self.pos_priors[POS_MID]))
        # Blend the positional prior with the player's own prior-season rates,
        # weighted by how much prior-season evidence there actually is. A flat
        # 50/50 under-weights a player with a full season behind them, which
        # matters most in GW1-5 when there is nothing else to go on.
        if season_prior:
            prior_mins = _f(season_prior.get("minutes"), 0.0)
            if prior_mins <= 0:
                prior_mins = _f(season_prior.get("minutes_per_game"), 0.0) * 38.0
            w = prior_mins / (prior_mins + PRIOR_SEASON_HALF_MINUTES)
            for k, v in season_prior.items():
                if k in prior and v is not None and math.isfinite(v):
                    prior[k] = (1.0 - w) * prior[k] + w * v

        player_team = player.get("team")
        history = sorted(history, key=lambda x: x.get("round", 0))

        stats = list(self.pos_priors[POS_MID].keys())
        num = {s: 0.0 for s in stats}
        wsum = {s: 0.0 for s in stats}

        for h in history:
            mins = _f(h.get("minutes"), 0.0)
            if mins <= 0:
                continue
            gw = h.get("round", current_gw - 1)
            age = max(1, current_gw - gw)

            was_home = h.get("was_home")
            opp_id = h.get("opponent_team")
            home_id = player_team if was_home else opp_id
            away_id = opp_id if was_home else player_team
            lam, mu = market_predictor.market.get_match_lambdas(home_id, away_id)
            team_xg = lam if was_home else mu
            opp_xg = mu if was_home else lam

            att_base = market_predictor.market.team_attack_baseline(player_team)
            att_mult = float(np.clip(team_xg / att_base if att_base > 0 else 1.0, 0.5, 1.8))
            def_mult = float(np.clip(opp_xg / LEAGUE_MEAN_GOALS, 0.6, 1.7))

            # Per-stat observation, with its own decay weight. `None` means the
            # source did not report the field, which is not the same as zero.
            obs = {
                "xg": (_maybe(h, "expected_goals"), att_mult),
                "xa": (_maybe(h, "expected_assists"), assist_multiplier(att_mult)),
                "saves": (_maybe(h, "saves"), def_mult),
                "yc": (_maybe(h, "yellow_cards"), 1.0),
                "rc": (_maybe(h, "red_cards"), 1.0),
                "og": (_maybe(h, "own_goals"), 1.0),
                "recoveries": (_maybe(h, "recoveries"), def_mult),
                "tackles": (_maybe(h, "tackles"), def_mult),
            }
            cbi = _maybe(h, "clearances_blocks_interceptions")
            tk = _maybe(h, "tackles")
            rc = _maybe(h, "recoveries")
            cbit = None if (cbi is None and tk is None) else (_f(cbi) + _f(tk))
            cbirt = None if (cbit is None and rc is None) else (_f(cbit) + _f(rc))
            obs["cbit"] = (cbit, def_mult)
            obs["cbirt"] = (cbirt, def_mult)

            for stat, (value, condition) in obs.items():
                if value is None:
                    continue  # missing -> contributes nothing, prior holds
                hl, _ = self._settings(stat)
                w = 0.5 ** (age / hl)
                denom = condition if condition and condition > 0 else 1.0
                num[stat] += w * (_f(value) / denom)
                wsum[stat] += w * (mins / 90.0)

        out = {}
        for stat in stats:
            _, pw = self._settings(stat)
            a0 = prior.get(stat, 0.0) * pw
            out[f"{stat}90"] = (a0 + num[stat]) / (pw + wsum[stat])
        # Back-compat aliases used elsewhere in the engine.
        out["xg"] = out["xg90"]
        out["xa"] = out["xa90"]
        out["effective_n"] = float(wsum["xg"])
        return out


def load_lineup_overrides(source: Optional[str] = None) -> Dict[int, float]:
    """
    Load predicted-lineup start probabilities from a JSON file.

    `source` is a path, or None to read the `FPL_LINEUPS` environment variable.
    Returns {} when nothing is configured, which is the normal case: no lineups
    feed ships with this repo, and none is invented here. What ships is the
    ingestion path, so a feed you trust can be plugged in without touching the
    model. This is the single largest accuracy gain available to the engine -
    minutes dominate FPL point error, and p_play AUC is ~0.88 without it.

    Accepted shapes:
        {"123": 0.95, "456": 0.10}
        {"lineups": [{"id": 123, "p_start": 0.95}, ...]}
        [{"id": 123, "p_start": 0.95}, ...]

    Values are clamped to [0, 1]. Availability is applied *after* this, so a
    feed cannot start a player FPL has flagged as injured.
    """
    import json
    import os

    path = source or os.getenv("FPL_LINEUPS")
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as e:
        logger.warning("Could not read lineups feed %s: %s", path, e)
        return {}

    if isinstance(raw, dict) and "lineups" in raw:
        raw = raw["lineups"]

    out: Dict[int, float] = {}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = [(r.get("id"), r.get("p_start", r.get("probability")))
                 for r in raw if isinstance(r, dict)]
    else:
        logger.warning("Lineups feed %s has an unrecognised shape.", path)
        return {}

    for pid, p in items:
        try:
            out[int(pid)] = float(np.clip(float(p), 0.0, 1.0))
        except (TypeError, ValueError):
            continue
    logger.info("Loaded %d predicted-lineup entries from %s", len(out), path)
    return out


# Minutes a team actually plays in one match. Not 11 x 90 = 990: red cards cost
# a team the balance, and the measured mean over 2025-26 single fixtures is 985.
TEAM_MATCH_MINUTES = 985.0
# Players lasting 60+ minutes in a team-match, measured over 2025-26 single
# fixtures. Not 11: three to five substitutions a match means roughly one
# starter is routinely withdrawn before the hour.
TEAM_LONG_APPEARANCES = 10.28
# Bucket midpoints live in match_sim so both paths cannot drift apart.
_BUCKET_MINUTES = BUCKET_MINUTES


def _appearance_minutes(probs: List[float]) -> float:
    return sum(p * m for p, m in zip(probs[1:], _BUCKET_MINUTES))


def _tilt_one(probs: List[float], lam: float) -> List[float]:
    """
    Scale a player's odds of appearing by `lam`, keeping the shape of the
    appearance intact.

    Working in odds rather than probability is what makes this self-targeting:
    a player at 0.99 has odds of 99 and is essentially immovable, while one at
    0.50 has odds of 1 and takes the correction. Rescaling probabilities
    directly would drag the nailed starters down with everyone else.
    """
    _, p1, p2, p3 = probs
    m = min(1.0 - 1e-9, max(1e-9, p1 + p2 + p3))
    odds = (m / (1.0 - m)) * lam
    m2 = odds / (1.0 + odds)
    scale = m2 / m
    return [1.0 - m2, p1 * scale, p2 * scale, p3 * scale]


def _tilt_to_team_minutes(dists: List[List[float]],
                          target: float = TEAM_MATCH_MINUTES) -> List[List[float]]:
    """
    One tilt making a team's expected minutes sum to `target`.

    Kept as the fallback for `_reconcile_team_minutes`, which controls the same
    total plus the number of long appearances. One knob cannot do both, and
    which way it errs is not a detail: it over-shaves the fringe and leaves
    nailed starters over-allocated, which lands as a *per-position* bias.
    """
    if not dists:
        return dists
    total = sum(_appearance_minutes(d) for d in dists)
    # Nothing to solve for, and nothing to solve *with* if no one can play.
    if total <= 0 or abs(total - target) < 1.0:
        return dists
    # The bracket has to span the odds themselves. A squad of certain starters
    # sits at odds ~1e9 after the clamp in `_tilt_one`, and pulling it down to a
    # real team-match needs a lambda near 1e-9 - outside a 1e-6 bracket, where
    # the search would quietly converge on the wrong end and leave the total
    # uncorrected.
    lo, hi = 1e-12, 1e12
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if sum(_appearance_minutes(_tilt_one(d, mid)) for d in dists) > target:
            hi = mid
        else:
            lo = mid
    lam = math.sqrt(lo * hi)
    return [_tilt_one(d, lam) for d in dists]


def _reconcile_team_minutes(dists: List[List[float]],
                            minutes_target: float = TEAM_MATCH_MINUTES,
                            long_target: float = TEAM_LONG_APPEARANCES
                            ) -> List[List[float]]:
    """
    Reconcile a team against *both* things a team-match is known to contain:
    985 minutes, and 10.28 players lasting 60 minutes or more.

    A single tilt on "does this player appear" hits the minutes total but lets
    the composition drift, because it cannot tell a starter's minutes from a
    substitute's. Measured, it over-corrected the fringe (-473 minutes) while
    leaving nailed starters over-allocated (+307) - a bias that falls unevenly
    across positions, which is worse than an even one for a solver whose whole
    job is choosing between positions.

    The four buckets already separate the two groups that need separate
    treatment: 60+ is a long appearance, 1-59 is a cameo. So tilt them
    independently, multinomial-logit style, with `p0` as the fixed reference
    category. Each player keeps their own 60-89 / 90 split, and a player with
    no chance of appearing stays at no chance.
    """
    if not dists:
        return dists
    arr = np.asarray(dists, dtype=float)
    p0, cameo = arr[:, 0], arr[:, 1]
    long = arr[:, 2] + arr[:, 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        share90 = np.where(long > 1e-12, arr[:, 3] / np.maximum(long, 1e-12), 0.0)
    long_minutes = share90 * _BUCKET_MINUTES[2] + (1.0 - share90) * _BUCKET_MINUTES[1]

    # Infeasible if nobody can start or nobody can come on: there is no tilt
    # that reaches the target, so leave the team to the one-knob fallback.
    if long.sum() <= 0 or cameo.sum() <= 0:
        return _tilt_to_team_minutes(dists, minutes_target)

    def _apply(theta):
        lc, ll = math.exp(theta[0]), math.exp(theta[1])
        denom = p0 + cameo * lc + long * ll
        denom = np.where(denom > 1e-12, denom, 1.0)
        return cameo * lc / denom, long * ll / denom

    def _resid(theta):
        c2, l2 = _apply(theta)
        return [c2.sum() * _BUCKET_MINUTES[0] + float((l2 * long_minutes).sum())
                - minutes_target,
                l2.sum() - long_target]

    try:
        from scipy.optimize import root

        sol = root(_resid, [0.0, 0.0], method="hybr")
        if not sol.success or not np.all(np.isfinite(sol.x)):
            return _tilt_to_team_minutes(dists, minutes_target)
        c2, l2 = _apply(sol.x)
    except Exception:  # a solver failure must not lose a gameweek
        return _tilt_to_team_minutes(dists, minutes_target)

    if not (np.all(np.isfinite(c2)) and np.all(np.isfinite(l2))):
        return _tilt_to_team_minutes(dists, minutes_target)

    out = np.column_stack([
        np.clip(1.0 - c2 - l2, 0.0, 1.0),
        c2,
        l2 * (1.0 - share90),
        l2 * share90,
    ])
    totals = out.sum(axis=1, keepdims=True)
    out = out / np.where(totals > 0, totals, 1.0)
    return [list(map(float, row)) for row in out]


def _maybe(row: Dict[str, Any], key: str):
    """Return a float, or None when the source genuinely lacks the field."""
    if key not in row:
        return None
    v = row[key]
    if v is None:
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


@lru_cache(maxsize=8192)
def _parse_kickoff(value: str):
    """
    Parse a kickoff timestamp, memoised on the string.

    A gameweek holds ten distinct kickoff times and the forecast asks for one
    per player-fixture, so this was 4,700 identical `pd.to_datetime` calls and
    1.6 of 11 seconds. Memoising on the string rather than reading the
    calendar's cached value keeps the semantics exact: the calendar is keyed by
    (team, gameweek) and holds only one kickoff per key, so it would hand back
    the wrong time for the second fixture of a double gameweek.
    """
    ts = pd.to_datetime(value)
    return ts.tz_convert(None) if getattr(ts, "tzinfo", None) else ts


class FixtureCalendar:
    """
    Kickoff times per (team, gameweek), and the rest/congestion they imply.

    Rolling means over a player's last five appearances cannot see that those
    five came in eighteen days. Rotation is driven by the calendar as much as by
    form, so the calendar has to be a feature rather than something the model is
    left to infer from noisy minutes alone.
    """

    DEFAULT_DAYS_REST = 7.0
    DEFAULT_CONGESTION = 2.0
    CONGESTION_WINDOW_DAYS = 14

    def __init__(self, fixtures=None):
        self._by_team: Dict[int, List[Any]] = {}
        self._kickoff: Dict[Tuple[int, int], Any] = {}
        for f in (fixtures or []):
            ko = f.get("kickoff_time")
            if not ko:
                continue
            try:
                ts = _parse_kickoff(ko)
            except Exception:
                continue
            if pd.isna(ts):
                continue
            gw = f.get("event")
            for tid in (f.get("team_h"), f.get("team_a")):
                if tid is None:
                    continue
                self._by_team.setdefault(int(tid), []).append(ts)
                if gw is not None:
                    self._kickoff[(int(tid), int(gw))] = ts
        for tid in self._by_team:
            self._by_team[tid].sort()

    def kickoff(self, team_id, gw):
        if team_id is None or gw is None:
            return None
        return self._kickoff.get((int(team_id), int(gw)))

    def context(self, team_id, kickoff_ts) -> Tuple[float, float]:
        """(days since the team's previous match, matches in the prior 14 days)."""
        if team_id is None or kickoff_ts is None:
            return self.DEFAULT_DAYS_REST, self.DEFAULT_CONGESTION
        times = self._by_team.get(int(team_id))
        if not times:
            return self.DEFAULT_DAYS_REST, self.DEFAULT_CONGESTION
        prior = [t for t in times if t < kickoff_ts]
        if not prior:
            # Opening fixture of the season: fully rested, nothing behind it.
            return 14.0, 0.0
        days_rest = float((kickoff_ts - prior[-1]).days)
        window_start = kickoff_ts - pd.Timedelta(days=self.CONGESTION_WINDOW_DAYS)
        congestion = float(sum(1 for t in prior if t >= window_start))
        return float(np.clip(days_rest, 0.0, 30.0)), congestion

    def context_for_gw(self, team_id, gw) -> Tuple[float, float]:
        return self.context(team_id, self.kickoff(team_id, gw))


class MinutesClassifier:
    """
    Four-class minutes model: 0, 1-59, 60-89, 90.

    The single most consequential fix here is that `cost` is now on the same
    scale at train and predict time. Previously training saw FPL's tenths
    (155) while prediction passed pounds (15.5), so every prediction fell below
    the lowest bin boundary the model had ever seen on that feature.

    The `flagged` feature that used to sit at the end of this vector was dead:
    `train` hardcoded it to 0.0 for every row, so it was constant in training
    and the booster could never split on it. Toggling it at predict time moved
    a prediction by exactly 0.0. Historical injury flags are not recoverable
    from FPL's gameweek history, so it could not be learned and has been
    removed rather than left in place looking load-bearing. Availability still
    enters through `_availability` / `_blend`, which is where it belongs.
    """

    N_FEATURES = 12
    IDX_DAYS_REST = 10
    IDX_CONGESTION = 11

    # Optional external override of start probability, keyed by player id, e.g.
    # from a predicted-lineups feed in the final 24h before a deadline. This is
    # where most of the human edge lives; no such feed ships with this repo, so
    # the hook is left explicit rather than faked.
    lineup_overrides: Dict[int, float]

    def __init__(self, calibrate: bool = False, calendar: Optional["FixtureCalendar"] = None):
        self.lineup_overrides = {}
        self.calendar = calendar or FixtureCalendar()
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.model = HistGradientBoostingClassifier(
            max_iter=120, max_depth=6, learning_rate=0.10, random_state=42
        )
        self.calibrated_model = None
        self.calibrate = calibrate
        self.is_trained = False
        self.classes_: List[int] = []

    @staticmethod
    def _get_class(mins) -> int:
        mins = _f(mins, 0.0)
        if mins == 0:
            return 0
        if mins < 60:
            return 1
        if mins < 90:
            return 2
        return 3

    @staticmethod
    def _features(window: Dict[str, list], cost_tenths: float, element_type: int,
                  prior_season: Dict[str, float],
                  days_rest: float = FixtureCalendar.DEFAULT_DAYS_REST,
                  congestion: float = FixtureCalendar.DEFAULT_CONGESTION) -> List[float]:
        starts = window["starts"]
        mins = window["mins"]
        avg_starts = float(np.mean(starts)) if starts else 0.0
        avg_mins = float(np.mean(mins)) if mins else 0.0
        last_mins = float(mins[-1]) if mins else 0.0

        consec_starts = 0
        for s in reversed(starts):
            if s >= 1:
                consec_starts += 1
            else:
                break
        consec_absent = 0
        for m in reversed(mins):
            if m == 0:
                consec_absent += 1
            else:
                break

        return [
            avg_starts,
            avg_mins,
            cost_tenths / 10.0,          # <- pounds, matching prediction
            float(element_type),
            float(consec_starts),
            float(consec_absent),
            last_mins,
            float(np.std(mins)) if len(mins) > 1 else 0.0,
            float(len(mins)),
            float(prior_season.get("minutes_per_game", 0.0)),
            float(days_rest),
            float(congestion),
        ]

    def train(self, all_history, prior_seasons: Optional[Dict[int, Dict[str, float]]] = None,
              extra_rows: Optional[List[Tuple[List[float], int]]] = None,
              team_by_pid: Optional[Dict[int, int]] = None):
        prior_seasons = prior_seasons or {}
        team_by_pid = team_by_pid or {}
        X, y = [], []

        for pid, history in all_history.items():
            window = {"starts": [], "mins": []}
            ps = prior_seasons.get(pid, {})
            team_id = team_by_pid.get(pid)
            history = sorted(history, key=lambda h: h.get("round", 0))
            for h in history:
                # Rest and congestion as they stood for THAT gameweek, so the
                # feature means the same thing at train and predict time.
                days_rest, congestion = self.calendar.context_for_gw(
                    team_id, h.get("round"))
                X.append(self._features(
                    window,
                    cost_tenths=_f(h.get("value"), 50.0),
                    element_type=_f(h.get("element_type"), 3.0),
                    prior_season=ps,
                    days_rest=days_rest,
                    congestion=congestion,
                ))
                y.append(self._get_class(h.get("minutes")))
                window["starts"].append(_f(h.get("starts"), 0.0))
                window["mins"].append(_f(h.get("minutes"), 0.0))
                if len(window["starts"]) > 5:
                    window["starts"].pop(0)
                if len(window["mins"]) > 5:
                    window["mins"].pop(0)

        if extra_rows:
            for feats, label in extra_rows:
                X.append(feats)
                y.append(label)

        if not X or len(set(y)) < 2:
            self.is_trained = False
            return self

        # A feature that never varies in training is one the booster can never
        # split on, so setting it at predict time does nothing at all. That is
        # precisely how the old `flagged` feature came to be dead weight while
        # looking load-bearing, so say so loudly rather than letting it repeat.
        varying = np.ptp(np.asarray(X, dtype=float), axis=0)
        inert = [i for i, spread in enumerate(varying)
                 if spread == 0 and i in (self.IDX_DAYS_REST, self.IDX_CONGESTION)]
        if inert:
            logger.warning(
                "Minutes features %s are constant in training and will have no "
                "effect on any prediction. The fixture calendar is probably "
                "empty - check that fixtures carry `kickoff_time`.", inert,
            )

        X_arr, y_arr = np.array(X, dtype=float), np.array(y, dtype=int)
        self.model.fit(X_arr, y_arr)
        self.classes_ = list(self.model.classes_)

        # Probability calibration, when there is enough data to hold some out.
        if self.calibrate and len(X_arr) >= 2000 and len(set(y)) >= 3:
            try:
                from sklearn.calibration import CalibratedClassifierCV
                from sklearn.ensemble import HistGradientBoostingClassifier

                base = HistGradientBoostingClassifier(
                    max_iter=120, max_depth=6, learning_rate=0.10, random_state=42
                )
                cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
                cal.fit(X_arr, y_arr)
                self.calibrated_model = cal
            except Exception as e:  # calibration is a bonus, never a blocker
                logger.debug("Minutes calibration skipped: %s", e)
                self.calibrated_model = None

        self.is_trained = True
        return self

    def _predictor(self):
        return self.calibrated_model if self.calibrated_model is not None else self.model

    def _availability(self, player, fixture) -> float:
        status = player.get("status", "a")
        chance_raw = player.get("chance_of_playing_next_round")

        if status in ("i", "s", "u", "n") or chance_raw == 0:
            return 0.0
        chance = _f(chance_raw, 100.0) if chance_raw is not None else 100.0

        news = player.get("news", "") or ""
        if news and fixture and fixture.get("kickoff_time"):
            import re
            import datetime

            m = re.search(r"Expected back (\d{1,2} [a-zA-Z]{3})", news)
            if m:
                try:
                    curr_year = datetime.datetime.now().year
                    ret = datetime.datetime.strptime(f"{m.group(1)} {curr_year}", "%d %b %Y")
                    kickoff = pd.to_datetime(fixture["kickoff_time"]).tz_localize(None)
                    if ret.month < 6 and kickoff.month > 7:
                        ret = ret.replace(year=curr_year + 1)
                    if kickoff >= ret:
                        days_past = (kickoff - ret).days
                        chance = max(chance, min(100.0, 50.0 + days_past * 10.0))
                except Exception:
                    pass
        return chance / 100.0

    def build_row(self, player, history, fixture=None,
                  prior_season: Optional[Dict[str, float]] = None):
        """Feature vector plus availability for one (player, fixture) pair."""
        p_available = self._availability(player, fixture)
        window = {"starts": [], "mins": []}
        for h in sorted(history, key=lambda x: x.get("round", 0)):
            window["starts"].append(_f(h.get("starts"), 0.0))
            window["mins"].append(_f(h.get("minutes"), 0.0))
            if len(window["starts"]) > 5:
                window["starts"].pop(0)
            if len(window["mins"]) > 5:
                window["mins"].pop(0)
        kickoff = None
        if fixture is not None and fixture.get("kickoff_time"):
            try:
                kickoff = _parse_kickoff(fixture["kickoff_time"])
            except Exception:
                kickoff = None
        if kickoff is None and fixture is not None:
            kickoff = self.calendar.kickoff(player.get("team"), fixture.get("event"))
        days_rest, congestion = self.calendar.context(player.get("team"), kickoff)

        feats = self._features(
            window,
            cost_tenths=_f(player.get("now_cost"), 50.0),
            element_type=_f(player.get("element_type"), 3.0),
            prior_season=prior_season or {},
            days_rest=days_rest,
            congestion=congestion,
        )
        return feats, p_available, player

    def _apply_lineup_override(self, base, player):
        """
        Fold a known/likely starting XI into the minutes distribution.

        `p_start` shifts mass between "starts and plays long" and "does not
        start". It is applied before availability so an injured player still
        cannot play, whatever a lineup feed claims.
        """
        p_start = self.lineup_overrides.get(player.get("id"))
        if p_start is None:
            return base
        p_start = float(min(1.0, max(0.0, p_start)))
        starter_mass = base[2] + base[3]
        bench_mass = base[0] + base[1]
        if starter_mass <= 0 or bench_mass <= 0:
            return base
        out = [
            base[0] / bench_mass * (1.0 - p_start),
            base[1] / bench_mass * (1.0 - p_start),
            base[2] / starter_mass * p_start,
            base[3] / starter_mass * p_start,
        ]
        return out

    # A player carrying a fitness doubt who *does* play is more likely to be
    # eased back in - benched, or withdrawn early. Scaling every outcome by the
    # same availability factor preserves the conditional shape exactly, which
    # asserts that a 50%-doubtful player is as likely to last 90 minutes given
    # he features as a fully fit one. He is not, and that error lands straight
    # on clean sheets and defensive contributions, which need the long shift.
    DOUBT_SHORTENING = 0.35

    def _blend(self, base, p_available):
        total = sum(base)
        if total <= 0:
            base = [0.25, 0.25, 0.25, 0.25]
            total = 1.0
        base = [b / total for b in base]

        doubt = 1.0 - min(1.0, max(0.0, p_available))
        if doubt > 0:
            shift = self.DOUBT_SHORTENING * doubt
            moved_90 = base[3] * shift
            moved_60 = base[2] * shift
            base = [
                base[0] + moved_90 * 0.3 + moved_60 * 0.4,
                base[1] + moved_60 * 0.6,
                base[2] + moved_90 * 0.7 - moved_60,
                base[3] - moved_90,
            ]

        # Fold in availability: the mass removed goes to "did not play".
        return [
            (1.0 - p_available) + p_available * base[0],
            p_available * base[1],
            p_available * base[2],
            p_available * base[3],
        ]

    UNTRAINED_PRIOR = [0.15, 0.12, 0.13, 0.60]

    def predict_proba_batch(self, rows):
        """
        Minutes distributions for many (features, availability) pairs at once.

        Batching is not an optimisation detail here - single-row `predict_proba`
        on a boosted ensemble costs ~65ms because per-call overhead dominates,
        which made a full gameweek take minutes. One batched call over every
        player is several orders of magnitude cheaper.
        """
        if not rows:
            return []
        if not self.is_trained:
            return [
                self._blend(
                    self._apply_lineup_override(list(self.UNTRAINED_PRIOR), r[2])
                    if len(r) > 2 and r[2] is not None else list(self.UNTRAINED_PRIOR),
                    r[1],
                )
                for r in rows
            ]

        X = np.array([r[0] for r in rows], dtype=float)
        raw = self._predictor().predict_proba(X)
        out = []
        for i, row in enumerate(rows):
            avail = row[1]
            base = [0.0, 0.0, 0.0, 0.0]
            for cls, p in zip(self.classes_, raw[i]):
                base[int(cls)] = float(p)
            if len(row) > 2 and row[2] is not None:
                base = self._apply_lineup_override(base, row[2])
            out.append(self._blend(base, avail))
        return out

    def predict_proba(self, player, history, fixture=None, current_gw=1,
                      prior_season: Optional[Dict[str, float]] = None):
        row = self.build_row(player, history, fixture, prior_season)
        return self.predict_proba_batch([row])[0]


class EnsembleForecaster:
    def __init__(self, half_life: float = 5.0, prior_weight: float = 1.0,
                 stat_settings=None, n_bonus_sims: int = 800,
                 calibration_method: str = "linear", defcon_enabled: bool = True,
                 normalise_team_minutes: bool = True):
        self.dc = MarketOddsPredictor()
        self.gpf = GammaPoissonFilter(half_life=half_life, prior_weight=prior_weight,
                                      stat_settings=stat_settings)
        self.mc = MinutesClassifier()
        self.sim = MatchSimulator(n_sims=n_bonus_sims)
        self.calibrator = PointsCalibrator(method=calibration_method)
        self.prior_seasons: Dict[int, Dict[str, float]] = {}
        self.defcon_dispersion: Dict[int, float] = {POS_DEF: 1.85, POS_MID: 1.85, POS_FWD: 1.85}
        # False for seasons before FPL awarded defensive contribution points.
        self.defcon_enabled = bool(defcon_enabled)
        # Constrain each team's predicted minutes to a real team-match.
        self.normalise_team_minutes = bool(normalise_team_minutes)
        self._minutes_cache: Dict[tuple, List[float]] = {}
        self.calendar = FixtureCalendar()
        self._fit_teams = None
        self._fit_season_str = None
        self._fit_fixtures = None

    def set_lineup_overrides(self, overrides: Optional[Dict[int, float]]):
        """
        Install per-player start probabilities from a predicted-lineups feed.

        Keys are FPL element ids, values are P(starts) in [0, 1]. Applied before
        availability, so an injured player still cannot play whatever the feed
        claims. See `load_lineup_overrides` for the file/env ingestion path.
        """
        self.mc.lineup_overrides = dict(overrides or {})
        self.clear_minutes_cache()
        return self.mc.lineup_overrides

    @staticmethod
    def _minutes_key(player, fixture, dgw_idx):
        return (player.get("id"), fixture.get("event"),
                fixture.get("team_h"), fixture.get("team_a"), dgw_idx)

    def prime_minutes(self, jobs):
        """
        Pre-compute minutes distributions for many (player, fixture, history,
        dgw_idx) jobs in a single batched model call, then serve them from cache.
        """
        rows, keys = [], []
        for player, fixture, history, dgw_idx in jobs:
            key = self._minutes_key(player, fixture, dgw_idx)
            if key in self._minutes_cache:
                continue
            rows.append(self.mc.build_row(
                player, history, fixture, self.prior_seasons.get(player.get("id"))))
            keys.append(key)
        if rows:
            for key, probs in zip(keys, self.mc.predict_proba_batch(rows)):
                self._minutes_cache[key] = probs
        if self.normalise_team_minutes:
            self._normalise_team_minutes(jobs)

    def _normalise_team_minutes(self, jobs):
        """
        Make each team's predicted minutes add up to a team-match.

        The classifier scores players one at a time and has no idea that a team
        fields eleven of them. Measured on 2025-26 it hands out 1059 minutes per
        team-match against a true 985 - a 7% over-allocation. It is not inventing
        extra starters (10.50 players predicted to reach 60+ against a true
        10.28); the excess is cameo mass smeared across fringe players who will
        not actually get on.

        Tilting the odds of appearing by one shared factor per team fixes the
        total while preserving the ordering, and it self-targets: a nailed
        starter's odds are enormous and barely move, while a fringe player at
        even money absorbs almost all of the correction. That is exactly the
        0.6-0.8 band where P(plays) was most over-confident.
        """
        groups: Dict[tuple, List[tuple]] = {}
        for player, fixture, history, dgw_idx in jobs:
            key = self._minutes_key(player, fixture, dgw_idx)
            if key not in self._minutes_cache:
                continue
            groups.setdefault(
                (fixture.get("event"), fixture.get("team_h"), fixture.get("team_a"),
                 player.get("team"), dgw_idx), []).append(key)

        for keys in groups.values():
            dists = [self._minutes_cache[k] for k in keys]
            for k, d in zip(keys, _reconcile_team_minutes(dists)):
                self._minutes_cache[k] = d

    def clear_minutes_cache(self):
        self._minutes_cache.clear()

    # ------------------------------------------------------------------ fit

    def fit(self, teams, past_fixtures, current_gw, all_history, fpl_players=None,
            season=None, prior_ratings=None, results_df=None, all_fixtures=None):
        # The calendar needs every fixture, not just the played ones: rest and
        # congestion for the gameweek being forecast are properties of fixtures
        # that have not happened yet.
        self.calendar = FixtureCalendar(all_fixtures if all_fixtures is not None
                                        else past_fixtures)
        self.mc.calendar = self.calendar
        current_gw_date = None
        if past_fixtures:
            times = [pd.to_datetime(f.get("kickoff_time")) for f in past_fixtures
                     if f.get("kickoff_time")]
            times = [t.tz_convert(None) if getattr(t, "tzinfo", None) else t for t in times]
            current_gw_date = max(times) if times else None

        # Callers that omit `season` (the CLI does) must still get the
        # prior-season fallback, so derive it from the calendar rather than
        # letting the carry-forward silently switch itself off.
        if not season:
            season = _current_season_start_year(current_gw_date)
        season_str = f"{str(season)[2:]}{str(season + 1)[2:]}"
        # Kept so calibration can roll the fitted state back to an earlier
        # gameweek instead of scoring itself with a model that has already seen
        # the answers.
        self._fit_teams = teams
        self._fit_season_str = season_str
        self._fit_fixtures = all_fixtures if all_fixtures is not None else past_fixtures

        status = self.dc.fit_team_ratings(
            teams, current_gw_date=current_gw_date, season_str=season_str,
            prior_ratings=prior_ratings, results_df=results_df,
        )

        # Early in a season the odds file holds only matches already played, so
        # there is nothing to fit and every fixture would look identical. Carry
        # last season's ratings forward rather than going flat - this is exactly
        # the window (wildcard, initial squad) where fixture signal matters most.
        thin = status.get("degraded") or status.get("n_matches", 0) < MIN_MATCHES_FOR_RATINGS
        if thin and prior_ratings is None and season:
            carried = self._prior_season_ratings(teams, season)
            if carried:
                logger.info(
                    "Only %d priced matches this season; seeding team ratings from %d-%d.",
                    status.get("n_matches", 0), season - 1, season,
                )
                status = self.dc.fit_team_ratings(
                    teams, current_gw_date=current_gw_date, season_str=season_str,
                    prior_ratings=carried, results_df=results_df,
                )

        if status.get("degraded"):
            logger.error(
                "Fixture model is degraded (%s): expected points carry no fixture "
                "signal this run.", status,
            )

        self.prior_seasons = self._build_prior_seasons(fpl_players)
        team_by_pid = {p["id"]: p.get("team") for p in (fpl_players or [])}
        self.mc.train(all_history, prior_seasons=self.prior_seasons,
                      team_by_pid=team_by_pid)
        self._fit_defcon_dispersion(all_history, fpl_players)
        return status

    @staticmethod
    def _prior_season_ratings(teams, season: int):
        """Fit the previous season's odds and return its final team ratings."""
        try:
            prev = MarketOddsModel()
            prev_str = f"{str(season - 1)[2:]}{str(season)[2:]}"
            if not prev.fetch_odds(season_str=prev_str):
                return None
            st = prev.fit_team_ratings(fpl_teams=teams)
            if st.get("degraded") or st.get("n_matches", 0) < MIN_MATCHES_FOR_RATINGS:
                return None
            return prev.export_ratings()
        except Exception as e:
            logger.warning("Could not load prior-season ratings: %s", e)
            return None

    @staticmethod
    def _build_prior_seasons(fpl_players) -> Dict[int, Dict[str, float]]:
        """Per-90 rates from `history_past`, used to inform this season's priors."""
        out: Dict[int, Dict[str, float]] = {}
        for p in (fpl_players or []):
            past = p.get("history_past") or []
            if not past:
                continue
            last = past[-1]
            mins = _f(last.get("minutes"), 0.0)
            if mins < 450:
                continue
            per90 = lambda k: _f(last.get(k), 0.0) / mins * 90.0
            out[p["id"]] = {
                "xg": per90("expected_goals"),
                "xa": per90("expected_assists"),
                "saves": per90("saves"),
                "yc": per90("yellow_cards"),
                "minutes": mins,
                "minutes_per_game": mins / 38.0,
            }
        return out

    def _fit_defcon_dispersion(self, all_history, fpl_players):
        """Estimate variance/mean for defensive-contribution counts, per position."""
        pos_by_id = {p["id"]: p.get("element_type", POS_MID) for p in (fpl_players or [])}
        buckets: Dict[int, List[float]] = {POS_DEF: [], POS_MID: [], POS_FWD: []}
        for pid, hist in (all_history or {}).items():
            pos = pos_by_id.get(pid)
            if pos not in buckets:
                continue
            for h in hist:
                if _f(h.get("minutes"), 0.0) < 60:
                    continue
                cbi = _maybe(h, "clearances_blocks_interceptions")
                tk = _maybe(h, "tackles")
                rc = _maybe(h, "recoveries")
                if cbi is None and tk is None:
                    continue
                val = _f(cbi) + _f(tk)
                if pos != POS_DEF:
                    val += _f(rc)
                buckets[pos].append(val)
        for pos, vals in buckets.items():
            if len(vals) >= 200:
                arr = np.array(vals, dtype=float)
                mu, var = arr.mean(), arr.var()
                if mu > 0 and var > mu:
                    self.defcon_dispersion[pos] = float(np.clip(var / mu, 1.05, 4.0))
        logger.debug("DefCon dispersion: %s", self.defcon_dispersion)

    def refit_as_of(self, gw: int, players, past_history):
        """
        Roll the fitted state back to what it would have been before `gw`.

        Cutting the *history* fed to a forecast is not enough to make a
        retrospective forecast honest: the minutes classifier's weights and the
        market team ratings are themselves fitted quantities. Leaving those at
        their end-of-window values let the model score gameweeks whose outcomes
        it had already been trained on, which flattered it and biased the
        calibration slope toward 1.0 - i.e. toward concluding, wrongly, that
        there was nothing to correct.
        """
        if self._fit_teams is None:
            return
        fixtures = self._fit_fixtures or []
        times = []
        for f in fixtures:
            ev, ko = f.get("event"), f.get("kickoff_time")
            if ev is None or ko is None or ev >= gw:
                continue
            try:
                times.append(_parse_kickoff(ko))
            except Exception:
                continue
        self.dc.fit_team_ratings(
            self._fit_teams, current_gw_date=max(times) if times else None,
            season_str=self._fit_season_str,
        )
        team_by_pid = {p["id"]: p.get("team") for p in (players or [])}
        self.mc.train(past_history, prior_seasons=self.prior_seasons,
                      team_by_pid=team_by_pid)
        self.clear_minutes_cache()

    def fit_calibration(self, players, fixtures, all_history, target_gw,
                        window_gws: int = 5, honest: bool = True):
        """
        Re-forecast recent completed gameweeks and pair predictions with outcomes.

        This deliberately runs the *same* fixture-level pipeline the live path
        uses, including the match simulation that supplies bonus. Fitting on
        `_predict_uncalibrated` alone would train the calibrator on totals that
        are missing bonus points, and it would then scale up predictions that do
        include them.

        The time cut is strict: for gameweek g the forecaster only ever sees
        history with round < g, and with `honest=True` its fitted parameters are
        rolled back to that point too (see `refit_as_of`).
        """
        samples = build_calibration_samples(
            self, players, fixtures, all_history, target_gw,
            window_gws=window_gws, honest=honest,
        )
        if samples:
            self.calibrator.fit(samples)
        return self.calibrator

    # ---------------------------------------------------------- single player

    def _minutes_distribution(self, player, fixture, history, dgw_idx, current_gw):
        key = self._minutes_key(player, fixture, dgw_idx)
        p_states = self._minutes_cache.get(key)
        if p_states is None:
            p_states = self.mc.predict_proba(
                player, history, fixture, current_gw,
                prior_season=self.prior_seasons.get(player.get("id")),
            )
            self._minutes_cache[key] = p_states
        p_0, p_1_59, p_60_89, p_90 = p_states

        if dgw_idx > 0:
            # Second match of a double gameweek: rotation risk. Shift mass from
            # a full 90 toward a partial appearance and toward not playing, so
            # p_play, p_60 and xMin stay mutually consistent - the previous
            # version subtracted a flat 22 minutes and recomputed p_60 by a
            # separate formula that could disagree with p_play.
            rest = 0.18
            moved_90 = p_90 * rest
            moved_60 = p_60_89 * rest
            p_90 -= moved_90
            p_60_89 += moved_90 * 0.6 - moved_60
            p_1_59 += moved_90 * 0.25 + moved_60 * 0.6
            p_0 += moved_90 * 0.15 + moved_60 * 0.4

        p_play = p_1_59 + p_60_89 + p_90
        p_60 = p_60_89 + p_90
        xMin = (p_1_59 * BUCKET_MINUTES[0] + p_60_89 * BUCKET_MINUTES[1]
                + p_90 * BUCKET_MINUTES[2])
        return p_0, p_1_59, p_60_89, p_90, p_play, p_60, xMin

    def _predict_uncalibrated(self, player, fixture, history, dgw_idx=0, current_gw=1):
        element_type = player.get("element_type", POS_MID)
        # FPL has shipped positions this engine does not model (managers were
        # element_type 5 in 2024-25). Forecast them to zero so they are never
        # selected, rather than scoring them as if they were outfield players.
        if element_type not in MODELLED_POSITIONS:
            return _empty_components(player, element_type)
        p_0, p_1_59, p_60_89, p_90, p_play, p_60, xMin = self._minutes_distribution(
            player, fixture, history, dgw_idx, current_gw
        )

        if xMin <= 0 or p_play <= 0:
            return _empty_components(player, element_type)

        ctx = self.dc.fixture_context(player.get("team"), fixture)
        gw = fixture.get("event", current_gw)
        rates = self.gpf.predict_match(
            player, history, self.dc, gw,
            season_prior=self.prior_seasons.get(player.get("id")),
        )

        min_frac = xMin / 90.0
        # E[minutes | played 60+] / 90, for quantities that require a long shift.
        cond_frac = ((p_60_89 * BUCKET_MINUTES[1] + p_90 * BUCKET_MINUTES[2])
                     / (p_60 * 90.0)) if p_60 > 0 else 0.0
        # E[minutes | played at all] / 90. Every *nonlinear* term below has to
        # use this rather than min_frac: a floor or a threshold applied to a
        # Poisson mean scaled by unconditional minutes conflates "half a chance
        # of playing 90" with "certainly plays 45", and those are not the same
        # distribution. Linear terms (goals, assists) are unaffected either way.
        play_frac = (xMin / (p_play * 90.0)) if p_play > 0 else 0.0

        # ---- attacking returns, penalties split out ----
        pen_share = PEN_ORDER_SHARE.get(player.get("penalties_order"), PEN_ORDER_DEFAULT)

        # The observed xG90 already contains whatever penalties this player took,
        # but only to the extent we have actually seen them play. `confidence`
        # is how much of their current duty the history can already account for:
        # an established taker's rate has it baked in (subtract it back out
        # before re-adding), while a new signing or a player who has just taken
        # over spot-kicks has not yet earned it in the data.
        evidence = rates.get("effective_n", 0.0)
        confidence = evidence / (evidence + PEN_EVIDENCE_K)
        hist_pen_xg90 = TEAM_PEN_RATE * PEN_XG * pen_share * confidence
        open_play_xg90 = max(0.0, rates["xg90"] - hist_pen_xg90)

        # Penalties are won by getting into the box, not by general attacking
        # dominance, so they scale with the fixture more softly than open play.
        pen_mult = 1.0 + PEN_FIXTURE_DAMPING * (ctx["att_mult"] - 1.0)
        pen_xg90 = TEAM_PEN_RATE * PEN_XG * pen_share * pen_mult

        # ---- set-piece duty, handled exactly like penalties ----
        # Corner / indirect free-kick duty drives assists; direct free-kick duty
        # drives goals. Both are published by FPL and neither is visible in a
        # player's own xA/xG until it has already paid off, so the explicit term
        # is what carries a new taker before the history catches up. The share
        # the history has already had a chance to observe is subtracted first.
        sp_share = SP_ORDER_SHARE.get(
            player.get("corners_and_indirect_freekicks_order"), SP_ORDER_DEFAULT)
        fk_share = SP_ORDER_SHARE.get(
            player.get("direct_freekicks_order"), SP_ORDER_DEFAULT)
        sp_confidence = evidence / (evidence + SP_EVIDENCE_K)
        sp_mult = 1.0 + SP_FIXTURE_DAMPING * (ctx["att_mult"] - 1.0)

        hist_sp_xa90 = SET_PIECE_XA_PER_MATCH * sp_share * sp_confidence
        open_play_xa90 = max(0.0, rates["xa90"] - hist_sp_xa90)
        sp_xa90 = SET_PIECE_XA_PER_MATCH * sp_share * sp_mult

        hist_fk_xg90 = DIRECT_FK_XG_PER_MATCH * fk_share * sp_confidence
        open_play_xg90 = max(0.0, open_play_xg90 - hist_fk_xg90)
        fk_xg90 = DIRECT_FK_XG_PER_MATCH * fk_share * sp_mult

        xg = (open_play_xg90 * ctx["att_mult"] + pen_xg90 + fk_xg90) * min_frac
        # Assists track chance creation, which scales more softly than goals.
        xa = (open_play_xa90 * assist_multiplier(ctx["att_mult"]) + sp_xa90) * min_frac

        xg_cond = xg / max(1e-6, p_play)
        xa_cond = xa / max(1e-6, p_play)

        # ---- penalties missed and saved ----
        # Both fall out of the penalty machinery already fitted above rather
        # than needing their own rate: a taker's misses are the unconverted
        # share of the penalties they take, and a keeper faces the league rate.
        pen_miss90 = TEAM_PEN_RATE * pen_share * (1.0 - PEN_XG)
        pen_save90 = TEAM_PEN_RATE * PEN_SAVE_PROB if element_type == POS_GKP else 0.0

        # ---- clean sheet ----
        # Poisson P(opponent scores 0) over the share of the match the player is
        # on the pitch. The Dixon-Coles low-score adjustment shapes the *joint*
        # scoreline distribution used to split the match total in `market_odds`;
        # it does not alter this marginal, so this is a plain Poisson tail.
        opp_xg = ctx["opp_xg"]
        # Same distribution the simulation draws from, so the analytic clean
        # sheet and the simulated one cannot disagree. Plain Poisson over-states
        # P(concede nothing) by +0.016 against 2280 measured team-matches.
        p_cs_player = p_no_goals(opp_xg * cond_frac) if cond_frac > 0 else 0.0

        xG_pts = xg * POINTS_GOAL.get(element_type, 4)
        xA_pts = xa * POINTS_ASSIST
        xCS_pts = p_60 * p_cs_player * POINTS_CLEAN_SHEET.get(element_type, 0)

        # ---- goals conceded ----
        # Conditional on appearing, then weighted by p_play: the deduction is
        # E[floor(C/2)], which is convex, so feeding it an unconditional mean
        # understated it for anyone carrying rotation risk.
        player_opp_xg = opp_xg * play_frac
        if element_type in (POS_GKP, POS_DEF) and player_opp_xg > 0:
            xConc_penalty = p_play * sum(
                math.exp(-player_opp_xg) * player_opp_xg ** k / math.factorial(k) * (k // 2)
                for k in range(12)
            )
        else:
            xConc_penalty = 0.0

        # ---- cards, own goals, penalties missed ----
        xCard_penalty = (rates["yc90"] * min_frac
                         + rates["rc90"] * min_frac * abs(POINTS_RED)
                         + rates["og90"] * min_frac * abs(POINTS_OWN_GOAL)
                         + pen_miss90 * min_frac * abs(POINTS_PENALTY_MISS))

        # ---- saves ----
        expected_saves = rates["saves90"] * ctx["def_mult"] * play_frac
        if element_type == POS_GKP and expected_saves > 0:
            xSaves = p_play * sum(
                math.exp(-expected_saves) * expected_saves ** k / math.factorial(k) * (k // 3)
                for k in range(20)
            )
        else:
            xSaves = 0.0
        xPenSaves = pen_save90 * min_frac * POINTS_PENALTY_SAVE

        # ---- defensive contributions, now opponent-adjusted ----
        # Defensive volume rises when the opponent carries more threat; the
        # previous model applied no fixture conditioning to these at all.
        xDefCon, defcon_mu = 0.0, 0.0
        if self.defcon_enabled and element_type in DEFCON_THRESHOLD:
            base = rates["cbit90"] if element_type == POS_DEF else rates["cbirt90"]
            defcon_mu = base * ctx["def_mult"] * play_frac
            if defcon_mu > 0:
                from scipy.stats import nbinom

                disp = self.defcon_dispersion.get(element_type, 1.85)
                var = max(defcon_mu * 1.0001, disp * defcon_mu)
                p_nb = defcon_mu / var
                n_nb = defcon_mu ** 2 / max(1e-9, var - defcon_mu)
                threshold = DEFCON_THRESHOLD[element_type]
                p_hit = max(0.0, 1.0 - float(nbinom.cdf(threshold - 1, n_nb, p_nb)))
                # FPL awards the defensive contribution on the count alone -
                # there is no 60-minute requirement, unlike the clean sheet.
                # Gating on p_60 wrote off every substitute who comes on and
                # racks up tackles, and double-counted the minutes adjustment
                # already carried by `play_frac` inside defcon_mu.
                xDefCon = p_play * p_hit * 2.0

        xApp = p_60 * 2.0 + (p_play - p_60) * 1.0

        # Bonus is filled in by the match simulator; a per-player placeholder is
        # kept so single-player callers still get a sane total.
        xBonus = 0.0

        math_pts = (xApp + xCS_pts + xG_pts + xA_pts + xSaves + xPenSaves + xDefCon
                    + xBonus - xConc_penalty - xCard_penalty)

        return {
            "id": player.get("id"),
            "team": player.get("team"),
            "element_type": element_type,
            "p_play": p_play,
            "p_60": p_60,
            "xmin": xMin,
            "play_frac": play_frac,
            "cond_frac": cond_frac,
            "xApp": xApp,
            "xg": xg,
            "xa": xa,
            "xg_cond": xg_cond,
            "xa_cond": xa_cond,
            "p_cs": p_cs_player,
            "opp_xg": opp_xg,
            "player_opp_xg": player_opp_xg,
            "saves90": rates["saves90"] * ctx["def_mult"],
            "yc90": rates["yc90"],
            "rc90": rates["rc90"],
            "og90": rates["og90"],
            "pen_miss90": pen_miss90,
            "pen_save90": pen_save90,
            "recoveries90": rates["recoveries90"] * ctx["def_mult"],
            "tackles90": rates["tackles90"] * ctx["def_mult"],
            # Per-90 volume inputs for the BPS model. CBI is clearances,
            # blocks and interceptions *alone*: the filter tracks cbit = CBI +
            # tackles, and tackles are priced separately below, so passing cbit
            # here would charge them twice.
            "cbi90": max(0.0, rates["cbit90"] - rates["tackles90"]) * ctx["def_mult"],
            "xa90": rates["xa90"] * assist_multiplier(ctx["att_mult"]),
            "defcon_mu": defcon_mu,
            "defcon_dispersion": self.defcon_dispersion.get(element_type, 1.85),
            "cbit": rates["cbit90"] * ctx["def_mult"] * min_frac,
            "cbirt": rates["cbirt90"] * ctx["def_mult"] * min_frac,
            "xBonus": xBonus,
            "xCS_pts": xCS_pts,
            "xG_pts": xG_pts,
            "xA_pts": xA_pts,
            "xSaves": xSaves,
            "xPenSaves": xPenSaves,
            "xDefCon": xDefCon,
            "xConc_penalty": xConc_penalty,
            "xCard_penalty": xCard_penalty,
            "math_pts": math_pts,
        }

    def predict(self, player, fixture, history, dgw_idx=0, current_gw=1):
        comps = self._predict_uncalibrated(player, fixture, history, dgw_idx, current_gw)
        return round(self.calibrator.apply(comps["math_pts"], comps["element_type"]), 2)


def _empty_components(player, element_type):
    return {
        "id": player.get("id"), "team": player.get("team"), "element_type": element_type,
        "p_play": 0.0, "p_60": 0.0, "xmin": 0.0, "play_frac": 0.0, "cond_frac": 0.0, "xApp": 0.0,
        "xg": 0.0, "xa": 0.0,
        "xg_cond": 0.0, "xa_cond": 0.0, "p_cs": 0.0, "opp_xg": 0.0, "player_opp_xg": 0.0,
        "saves90": 0.0, "yc90": 0.0, "rc90": 0.0, "og90": 0.0,
        "pen_miss90": 0.0, "pen_save90": 0.0, "recoveries90": 0.0, "tackles90": 0.0,
        "cbi90": 0.0, "xa90": 0.0, "defcon_mu": 0.0, "defcon_dispersion": 1.85,
        "cbit": 0.0, "cbirt": 0.0, "xBonus": 0.0, "xCS_pts": 0.0, "xG_pts": 0.0,
        "xA_pts": 0.0, "xSaves": 0.0, "xPenSaves": 0.0, "xDefCon": 0.0,
        "xConc_penalty": 0.0, "xCard_penalty": 0.0, "math_pts": 0.0,
    }


# --------------------------------------------------------------------- matrix


def _prepare(horizon_gws, bootstrap, fixtures, all_history, season,
             half_life, prior_weight, calibrate, prior_ratings,
             calibration_method="linear", lineup_overrides=None):
    if bootstrap is None:
        bootstrap = get_bootstrap_static()
    if fixtures is None:
        fixtures = get_fixtures()

    players = bootstrap.get("elements", [])
    teams = bootstrap.get("teams", [])
    current_gw = horizon_gws[0] if horizon_gws else 1

    if all_history is None:
        ids = [p["id"] for p in players]
        logger.info("Fetching history for %d players...", len(ids))
        summaries = get_all_element_summaries(ids)
        all_history = {pid: summaries.get(pid, {}).get("history", []) for pid in ids}
        # history_past rides along on the player dicts for prior-season priors.
        for p in players:
            p.setdefault("history_past", summaries.get(p["id"], {}).get("history_past", []))

    # `season` is the starting year, so 2025 is the 2025-26 season. A live run
    # leaves it None, which means "current rules".
    ensemble = EnsembleForecaster(half_life=half_life, prior_weight=prior_weight,
                                  calibration_method=calibration_method,
                                  defcon_enabled=defcon_active(season))
    if not ensemble.defcon_enabled:
        logger.info("Defensive contribution points did not exist in %s-%s; "
                    "scoring them off.", season, (season or 0) + 1)
    past_fixtures = [f for f in fixtures if f.get("finished")]
    results_df = pd.DataFrame([
        {"team_h": f["team_h"], "team_a": f["team_a"],
         "team_h_score": f.get("team_h_score"), "team_a_score": f.get("team_a_score")}
        for f in past_fixtures
        if f.get("team_h_score") is not None and f.get("team_a_score") is not None
    ]) if past_fixtures else None

    ensemble.fit(teams, past_fixtures, current_gw, all_history, players,
                 season=season, prior_ratings=prior_ratings, results_df=results_df,
                 all_fixtures=fixtures)

    if calibrate:
        ensemble.fit_calibration(players, fixtures, all_history, current_gw)

    # Applied after calibration on purpose: a lineups feed exists only for the
    # gameweek in front of you, so letting it colour the retrospective forecasts
    # the calibrator fits on would be the same class of leak the honest refit
    # above exists to remove.
    if lineup_overrides:
        ensemble.set_lineup_overrides(lineup_overrides)
        logger.info("Applied %d predicted-lineup overrides.", len(lineup_overrides))

    fixture_map: Dict[int, List[Dict[str, Any]]] = {}
    for fix in fixtures:
        if fix.get("event") in horizon_gws:
            fixture_map.setdefault(fix["event"], []).append(fix)

    return bootstrap, fixtures, all_history, players, ensemble, fixture_map, current_gw


def _forecast_gameweek(ensemble, players_by_team, fixtures_for_gw, all_history,
                       current_gw, rng=None):
    """
    Forecast one gameweek fixture-by-fixture.

    Doing it per fixture rather than per player is what makes rank-based bonus
    possible: every player in the match is needed at once. It also removes the
    duplicate work in the old implementation, which computed each player-fixture
    twice - once through `predict` and again for `p_play`.
    """
    results: Dict[int, Dict[str, Any]] = {}
    seen_count: Dict[int, int] = {}
    # One generator for the whole gameweek. Letting each fixture build its own
    # from a fixed seed made every match draw from an identical random stream,
    # so two similar players in unrelated matches came out correlated - the one
    # artefact a correlated simulation must not have.
    if rng is None:
        rng = np.random.default_rng(ensemble.sim.seed)

    # One batched minutes call for the whole gameweek before any forecasting.
    jobs, counts = [], {}
    for fixture in fixtures_for_gw:
        for team_id in (fixture.get("team_h"), fixture.get("team_a")):
            for p in players_by_team.get(team_id, []):
                idx = counts.get(p["id"], 0)
                counts[p["id"]] = idx + 1
                jobs.append((p, fixture, all_history.get(p["id"], []), idx))
    ensemble.prime_minutes(jobs)

    for fixture in fixtures_for_gw:
        squad = []
        for team_id in (fixture.get("team_h"), fixture.get("team_a")):
            for p in players_by_team.get(team_id, []):
                idx = seen_count.get(p["id"], 0)
                comps = ensemble._predict_uncalibrated(
                    p, fixture, all_history.get(p["id"], []),
                    dgw_idx=idx, current_gw=current_gw,
                )
                seen_count[p["id"]] = idx + 1
                if comps["p_play"] > 0:
                    squad.append(comps)

        if not squad:
            continue

        lam, mu = ensemble.dc.market.get_match_lambdas(fixture["team_h"], fixture["team_a"])
        team_lambdas = {fixture["team_h"]: mu, fixture["team_a"]: lam}

        sim = ensemble.sim.simulate(squad, team_lambdas, rng=rng)
        bonus_map = sim["bonus"]
        draws = sim["points"]

        for comps in squad:
            pid = comps["id"]
            comps["xBonus"] = bonus_map.get(pid, 0.0)
            comps["math_pts"] += comps["xBonus"]
            entry = results.setdefault(pid, {
                "raw": 0.0, "p_play": 0.0, "element_type": comps["element_type"],
                "draws": None, "components": [], "n_fixtures": 0,
            })
            entry["raw"] += comps["math_pts"]
            entry["n_fixtures"] += 1
            entry["p_play"] = max(entry["p_play"], comps["p_play"])
            entry["components"].append(comps)
            d = draws.get(pid)
            if d is not None:
                entry["draws"] = d if entry["draws"] is None else entry["draws"] + d

    return results


def build_calibration_samples(forecaster, players, fixtures, all_history,
                              target_gw: int, window_gws: int = 5,
                              honest: bool = True):
    """Predictions vs outcomes for the gameweeks preceding `target_gw`."""
    samples: List[Dict[str, Any]] = []
    gws = [g for g in range(max(1, target_gw - window_gws), target_gw)]
    if not gws:
        return samples

    fixtures_by_gw: Dict[int, List] = {}
    for f in fixtures:
        if f.get("event") in gws:
            fixtures_by_gw.setdefault(f["event"], []).append(f)

    active = {pid for pid, hist in all_history.items()
              if any(_f(h.get("minutes"), 0.0) > 0 for h in hist
                     if h.get("round", 0) >= gws[0] - 3)}
    players_by_team: Dict[int, List] = {}
    for p in players:
        if p["id"] in active:
            players_by_team.setdefault(p["team"], []).append(p)

    actuals: Dict[int, Dict[int, float]] = {}
    for pid, hist in all_history.items():
        for h in hist:
            r = h.get("round")
            if r in gws:
                actuals.setdefault(r, {})
                actuals[r][pid] = actuals[r].get(pid, 0.0) + _f(h.get("total_points"), 0.0)

    for gw in gws:
        gw_actual = actuals.get(gw)
        if not gw_actual:
            continue
        # Strict time cut: only history from before this gameweek.
        past_history = {
            pid: [h for h in hist if h.get("round", 0) < gw]
            for pid, hist in all_history.items()
        }
        forecaster.clear_minutes_cache()
        if honest:
            forecaster.refit_as_of(gw, players, past_history)
        results = _forecast_gameweek(
            forecaster, players_by_team, fixtures_by_gw.get(gw, []), past_history, gw
        )
        for pid, r in results.items():
            if pid in gw_actual:
                samples.append({
                    "element_type": r["element_type"],
                    "pred": r["raw"],
                    "actual": gw_actual[pid],
                    "gw": gw,
                    "n_fixtures": r.get("n_fixtures", 1),
                })
    if honest:
        # Put the forecaster back where production found it: fitted on
        # everything available up to the gameweek actually being forecast.
        forecaster.refit_as_of(target_gw, players, all_history)
    forecaster.clear_minutes_cache()
    return samples


def generate_xp_matrix(horizon_gws: List[int], bootstrap=None, fixtures=None,
                       all_history=None, season: int = None, half_life: float = 5.0,
                       prior_weight: float = 1.0, calibrate: bool = False,
                       prior_ratings=None, calibration_method: str = "linear",
                       lineup_overrides=None) -> Dict[int, Dict[Any, float]]:
    """Expected points per player per gameweek, calibrated."""
    (bootstrap, fixtures, all_history, players, ensemble, fixture_map,
     current_gw) = _prepare(horizon_gws, bootstrap, fixtures, all_history, season,
                            half_life, prior_weight, calibrate, prior_ratings,
                            calibration_method, lineup_overrides)

    players_by_team: Dict[int, List] = {}
    for p in players:
        players_by_team.setdefault(p["team"], []).append(p)

    xp_matrix: Dict[int, Dict[Any, float]] = {p["id"]: {} for p in players}

    rng = np.random.default_rng(ensemble.sim.seed)
    for gw in horizon_gws:
        gw_results = _forecast_gameweek(
            ensemble, players_by_team, fixture_map.get(gw, []), all_history,
            current_gw, rng=rng,
        )
        for p in players:
            pid = p["id"]
            r = gw_results.get(pid)
            if not r:
                xp_matrix[pid][gw] = 0.0
                xp_matrix[pid][f"{gw}_p_play"] = 0.0
            else:
                cal = ensemble.calibrator.apply(
                    r["raw"], r["element_type"], n_fixtures=r.get("n_fixtures", 1))
                xp_matrix[pid][gw] = round(cal, 2)
                xp_matrix[pid][f"{gw}_p_play"] = r["p_play"]
    return xp_matrix


def generate_merv_matrix(horizon_gws: List[int], bootstrap=None, fixtures=None,
                         all_history=None, season: int = None, risk_aversion: float = 0.0,
                         half_life: float = 5.0, prior_weight: float = 1.0,
                         calibrate: bool = False, prior_ratings=None,
                         calibration_method: str = "linear",
                         lineup_overrides=None, draws_out=None
                         ) -> Dict[int, Dict[Any, float]]:
    """
    Marginal Expected Rank Value: expected points adjusted for how a pick moves
    your variance relative to the field.

    Variance now comes from the correlated match simulation rather than an
    independent per-player Monte Carlo, so team stacking is priced correctly.
    With risk_aversion == 0 this is exactly `generate_xp_matrix`.

    Pass a dict as `draws_out` to receive the raw correlated point draws, keyed
    by (player_id, gameweek). `match_sim.squad_score_draws` turns those into a
    squad-level score distribution - which is the only way to state a range for
    a plan rather than a single number, because team-mates' scores are not
    independent.
    """
    from ownership_model import build_eo_matrix
    from monte_carlo import calculate_merv

    (bootstrap, fixtures, all_history, players, ensemble, fixture_map,
     current_gw) = _prepare(horizon_gws, bootstrap, fixtures, all_history, season,
                            half_life, prior_weight, calibrate, prior_ratings,
                            calibration_method, lineup_overrides)

    eo_matrix = build_eo_matrix(players)
    players_by_team: Dict[int, List] = {}
    for p in players:
        players_by_team.setdefault(p["team"], []).append(p)

    merv_matrix: Dict[int, Dict[Any, float]] = {p["id"]: {} for p in players}

    rng = np.random.default_rng(ensemble.sim.seed)
    for gw in horizon_gws:
        gw_results = _forecast_gameweek(
            ensemble, players_by_team, fixture_map.get(gw, []), all_history,
            current_gw, rng=rng,
        )
        for p in players:
            pid = p["id"]
            r = gw_results.get(pid)
            if not r:
                merv_matrix[pid][gw] = 0.0
                merv_matrix[pid][f"{gw}_p_play"] = 0.0
                continue
            xp = ensemble.calibrator.apply(
                r["raw"], r["element_type"], n_fixtures=r.get("n_fixtures", 1))
            if draws_out is not None and r.get("draws") is not None:
                draws_out[(pid, gw)] = r["draws"]
            if risk_aversion:
                draws = r["draws"]
                var = float(np.var(draws)) if draws is not None and len(draws) else 0.0
                value = calculate_merv(xp, var, eo_matrix.get(pid, 0.0), risk_aversion)
            else:
                value = xp
            merv_matrix[pid][gw] = round(value, 2)
            merv_matrix[pid][f"{gw}_p_play"] = r["p_play"]
    return merv_matrix


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mat = generate_xp_matrix([2])
    print(list(mat.items())[:5])
