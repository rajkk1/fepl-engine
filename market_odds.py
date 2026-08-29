"""
Market-implied team strength.

Derives per-team attack/defence rates from bookmaker 1X2 and over/under 2.5
prices, which are a far better fixture signal than FPL's own FDR.

Changes over the original implementation:
  * team names resolve through `team_mapping` (one-to-one, alias-seeded)
  * ratings are time-decayed and split home/away, with early-season shrinkage
    toward the league mean instead of an unweighted season average
  * home advantage is fitted from the data rather than a hardcoded 1.10/0.90
    applied on top of ratings that already contain home matches
  * missing/thin odds no longer degrade silently to a flat 1.4/1.4 for every
    club - a fallback chain runs and `status` records which source was used
  * multiple bookmakers are tried, with the consensus columns as backstop
"""
import logging
import math
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import root_scalar

from team_mapping import build_team_mapping, describe_mapping

logger = logging.getLogger(__name__)

_GLOBAL_ODDS_CACHE: Dict[str, Any] = {}

LEAGUE_MEAN_GOALS = 1.40
# Half-life in matches for the exponential time decay on team ratings.
RATING_HALF_LIFE = 10.0
# Pseudo-matches of league-average prior. Keeps GW1-4 ratings sane instead of
# letting two fixtures define a team.
RATING_PRIOR_STRENGTH = 4.0

# Bookmaker column groups, in preference order. Each entry is
# (home, draw, away, over2.5, under2.5).
_BOOK_COLUMNS = [
    ("B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"),
    ("PSH", "PSD", "PSA", "P>2.5", "P<2.5"),
    ("BWH", "BWD", "BWA", "BW>2.5", "BW<2.5"),
    ("AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5"),
    ("BbAvH", "BbAvD", "BbAvA", "BbAv>2.5", "BbAv<2.5"),
]


def _devig(*probs: float) -> List[float]:
    total = sum(probs)
    if total <= 0:
        return [0.0] * len(probs)
    return [p / total for p in probs]


class MarketOddsModel:
    def __init__(self):
        self.odds_df: Optional[pd.DataFrame] = None
        self.team_ratings: Dict[int, Dict[str, float]] = {}
        self.season_str: Optional[str] = None
        self.FPL_TO_FD: Dict[int, str] = {}
        self.FD_TO_FPL: Dict[str, int] = {}
        self.home_advantage: float = 1.10
        # Which source the current ratings came from, and how trustworthy.
        self.status: Dict[str, Any] = {"source": "uninitialised", "n_matches": 0}
        self._league_mean: float = LEAGUE_MEAN_GOALS

    # ------------------------------------------------------------------ odds

    @staticmethod
    def _season_str_for_now() -> str:
        import datetime

        now = datetime.datetime.now()
        y1 = now.year if now.month >= 7 else now.year - 1
        return f"{str(y1)[2:]}{str(y1 + 1)[2:]}"

    def fetch_odds(self, season_str: Optional[str] = None) -> bool:
        """Load the season's odds file. Returns True if usable odds were found."""
        season_str = season_str or self._season_str_for_now()
        self.season_str = season_str

        if season_str in _GLOBAL_ODDS_CACHE:
            self.odds_df = _GLOBAL_ODDS_CACHE[season_str]
            return self.odds_df is not None and len(self.odds_df) > 0

        url = f"https://www.football-data.co.uk/mmz4281/{season_str}/E0.csv"
        try:
            df = pd.read_csv(url)
        except Exception as e:
            logger.warning("Could not fetch market odds for %s: %s", season_str, e)
            _GLOBAL_ODDS_CACHE[season_str] = None
            self.odds_df = None
            return False

        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")

        # Take the first bookmaker group that is actually present, then fill
        # gaps row-by-row from the later groups. A single book missing a price
        # should not discard the match.
        frames = []
        for cols in _BOOK_COLUMNS:
            if all(c in df.columns for c in cols):
                sub = df[["Date", "HomeTeam", "AwayTeam", *cols]].copy()
                sub.columns = ["Date", "HomeTeam", "AwayTeam", "H", "D", "A", "O", "U"]
                frames.append(sub)
        if not frames:
            logger.warning("Odds file for %s has no recognised bookmaker columns", season_str)
            _GLOBAL_ODDS_CACHE[season_str] = None
            self.odds_df = None
            return False

        merged = frames[0]
        for extra in frames[1:]:
            merged = merged.combine_first(extra)
        merged = merged.dropna(subset=["Date", "HomeTeam", "AwayTeam", "H", "D", "A", "O", "U"])

        self.odds_df = merged.sort_values("Date").reset_index(drop=True)
        _GLOBAL_ODDS_CACHE[season_str] = self.odds_df
        logger.info("Loaded %d priced matches for season %s", len(self.odds_df), season_str)
        return len(self.odds_df) > 0

    # -------------------------------------------------------- goal expectation

    def implied_total_goals(self, p_over: float) -> float:
        """Invert P(total > 2.5) to a Poisson mean for the match total."""
        p_over = min(max(p_over, 1e-4), 1 - 1e-4)

        def obj(mu):
            return (1.0 - poisson.cdf(2, mu)) - p_over

        try:
            return root_scalar(obj, bracket=[0.1, 8.0]).root
        except ValueError:
            return 2.5

    @staticmethod
    def _match_probs(mu_h: float, mu_a: float, rho: float = -0.13):
        """Dixon-Coles adjusted home/away win probabilities."""
        max_goals = 10
        i = np.arange(max_goals)
        prob = np.outer(poisson.pmf(i, mu_h), poisson.pmf(i, mu_a))
        prob[0, 0] *= max(0.0, 1 - mu_h * mu_a * rho)
        prob[0, 1] *= max(0.0, 1 + mu_h * rho)
        prob[1, 0] *= max(0.0, 1 + mu_a * rho)
        prob[1, 1] *= max(0.0, 1 - rho)
        return float(np.sum(np.tril(prob, -1))), float(np.sum(np.triu(prob, 1)))

    def split_goals(self, mu_total: float, p_home: float, p_away: float):
        """Split a match total into home/away means matching the win-odds ratio."""

        def obj(f):
            ph, pa = self._match_probs(f * mu_total, (1 - f) * mu_total)
            return (ph / max(1e-6, pa)) - (p_home / max(1e-6, p_away))

        try:
            f = root_scalar(obj, bracket=[0.1, 0.9]).root
        except ValueError:
            f = p_home / max(1e-6, p_home + p_away)
        return f * mu_total, (1 - f) * mu_total

    # ------------------------------------------------------------- rating fit

    def fit_team_ratings(
        self,
        fpl_teams: Optional[List[Dict[str, Any]]] = None,
        current_gw_date=None,
        prior_ratings: Optional[Dict[int, Dict[str, float]]] = None,
        results_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Fit attack/defence ratings, falling back through progressively weaker
        sources. Always leaves `self.status` describing what was used.
        """
        self.team_ratings = {}

        if fpl_teams and self.odds_df is not None and len(self.odds_df) > 0:
            fd_names = list(self.odds_df["HomeTeam"].dropna().unique())
            if not self.FPL_TO_FD:
                self.FPL_TO_FD, self.FD_TO_FPL, _ = build_team_mapping(fpl_teams, fd_names)
                logger.debug("Team mapping:\n%s", describe_mapping(fpl_teams, self.FPL_TO_FD))

        used = self._fit_from_odds(current_gw_date) if self.odds_df is not None else 0

        if used == 0:
            # Fallback 1: last season's final ratings, regressed toward the mean.
            if prior_ratings:
                self._seed_from_prior(prior_ratings, fpl_teams)
                self.status = {"source": "prior_season", "n_matches": 0}
                logger.warning(
                    "No usable odds for %s - seeding team ratings from prior season.",
                    self.season_str,
                )
            # Fallback 2: a Poisson fit on actual results so far.
            elif results_df is not None and len(results_df) > 0:
                self._fit_from_results(results_df, fpl_teams)
                self.status = {"source": "results_poisson", "n_matches": len(results_df)}
                logger.warning(
                    "No usable odds for %s - fitting team ratings from results.",
                    self.season_str,
                )
            else:
                self._flat_ratings(fpl_teams)
                self.status = {"source": "flat_default", "n_matches": 0, "degraded": True}
                logger.error(
                    "No odds, no prior ratings and no results for %s. Team ratings are "
                    "FLAT - every fixture will look identical. Forecasts from this run "
                    "carry no fixture signal.",
                    self.season_str,
                )
        else:
            self.status = {"source": "market_odds", "n_matches": used}

        self._league_mean = float(
            np.mean([r["att_home"] + r["att_away"] for r in self.team_ratings.values()]) / 2.0
        ) if self.team_ratings else LEAGUE_MEAN_GOALS
        if not np.isfinite(self._league_mean) or self._league_mean <= 0:
            self._league_mean = LEAGUE_MEAN_GOALS

        self.status["n_teams_rated"] = len(self.team_ratings)
        return self.status

    def _fit_from_odds(self, current_gw_date) -> int:
        df = self.odds_df
        if df is None or len(df) == 0 or not self.FD_TO_FPL:
            return 0
        if current_gw_date is not None:
            df = df[df["Date"] <= current_gw_date]
        if len(df) == 0:
            return 0

        latest = df["Date"].max()
        acc: Dict[int, Dict[str, list]] = {}
        used = 0
        ha_home, ha_away = [], []

        for row in df.itertuples(index=False):
            h_id = self.FD_TO_FPL.get(row.HomeTeam)
            a_id = self.FD_TO_FPL.get(row.AwayTeam)
            if not h_id or not a_id:
                continue
            try:
                p_h, p_d, p_a = _devig(1.0 / row.H, 1.0 / row.D, 1.0 / row.A)
                p_over, _ = _devig(1.0 / row.O, 1.0 / row.U)
            except (ZeroDivisionError, TypeError):
                continue
            if not all(np.isfinite([p_h, p_d, p_a, p_over])):
                continue

            mu_total = self.implied_total_goals(p_over)
            mu_h, mu_a = self.split_goals(mu_total, p_h, p_a)

            # Exponential decay by match age, in matches-equivalent days.
            age_days = (latest - row.Date).days if pd.notna(row.Date) else 0
            w = 0.5 ** ((age_days / 7.0) / RATING_HALF_LIFE)

            for tid, key_s, key_c, scored, conceded in (
                (h_id, "att_home", "def_home", mu_h, mu_a),
                (a_id, "att_away", "def_away", mu_a, mu_h),
            ):
                d = acc.setdefault(tid, {k: [] for k in
                                         ("att_home", "def_home", "att_away", "def_away")})
                d[key_s].append((scored, w))
                d[key_c].append((conceded, w))

            ha_home.append((mu_h, w))
            ha_away.append((mu_a, w))
            used += 1

        if used == 0:
            return 0

        # Fitted home advantage: the league-wide home/away goal ratio.
        def _wmean(pairs, default):
            if not pairs:
                return default
            num = sum(v * w for v, w in pairs)
            den = sum(w for _, w in pairs)
            return num / den if den > 0 else default

        mean_home = _wmean(ha_home, LEAGUE_MEAN_GOALS)
        mean_away = _wmean(ha_away, LEAGUE_MEAN_GOALS)
        league_mean = (mean_home + mean_away) / 2.0
        self.home_advantage = (
            mean_home / league_mean if league_mean > 0 else 1.10
        )

        # Shrink each team toward the league mean by effective sample size.
        for tid in self.FPL_TO_FD:
            d = acc.get(tid)
            if not d:
                continue
            rating = {}
            for key, default in (
                ("att_home", mean_home), ("def_home", mean_away),
                ("att_away", mean_away), ("def_away", mean_home),
            ):
                pairs = d[key]
                n_eff = sum(w for _, w in pairs)
                raw = _wmean(pairs, default)
                k = RATING_PRIOR_STRENGTH
                rating[key] = (raw * n_eff + default * k) / (n_eff + k)
            # Convenience aggregates used by the rest of the engine.
            rating["scored"] = (rating["att_home"] + rating["att_away"]) / 2.0
            rating["conceded"] = (rating["def_home"] + rating["def_away"]) / 2.0
            rating["n_eff"] = sum(w for _, w in d["att_home"]) + sum(w for _, w in d["att_away"])
            self.team_ratings[tid] = rating

        return used

    def _seed_from_prior(self, prior: Dict[int, Dict[str, float]], fpl_teams):
        """Carry last season's ratings forward, regressed 50% to the mean."""
        for tid, r in prior.items():
            self.team_ratings[tid] = {
                k: 0.5 * float(r.get(k, LEAGUE_MEAN_GOALS)) + 0.5 * LEAGUE_MEAN_GOALS
                for k in ("att_home", "def_home", "att_away", "def_away")
            }
            self.team_ratings[tid]["scored"] = (
                self.team_ratings[tid]["att_home"] + self.team_ratings[tid]["att_away"]) / 2.0
            self.team_ratings[tid]["conceded"] = (
                self.team_ratings[tid]["def_home"] + self.team_ratings[tid]["def_away"]) / 2.0
            self.team_ratings[tid]["n_eff"] = 0.0
        self._fill_missing(fpl_teams)

    def _fit_from_results(self, results_df: pd.DataFrame, fpl_teams):
        """
        Time-decayed attack/defence from realised scorelines. Weaker than the
        market but far better than assuming every team is average.
        """
        acc: Dict[int, Dict[str, list]] = {}
        for row in results_df.itertuples(index=False):
            h, a = getattr(row, "team_h", None), getattr(row, "team_a", None)
            hs, as_ = getattr(row, "team_h_score", None), getattr(row, "team_a_score", None)
            if h is None or a is None or hs is None or as_ is None:
                continue
            if not (np.isfinite(hs) and np.isfinite(as_)):
                continue
            for tid, ks, kc, s, c in (
                (h, "att_home", "def_home", hs, as_),
                (a, "att_away", "def_away", as_, hs),
            ):
                d = acc.setdefault(int(tid), {k: [] for k in
                                              ("att_home", "def_home", "att_away", "def_away")})
                d[ks].append(float(s))
                d[kc].append(float(c))

        for tid, d in acc.items():
            rating = {}
            for key in ("att_home", "def_home", "att_away", "def_away"):
                vals = d[key]
                n = len(vals)
                raw = float(np.mean(vals)) if n else LEAGUE_MEAN_GOALS
                k = RATING_PRIOR_STRENGTH
                rating[key] = (raw * n + LEAGUE_MEAN_GOALS * k) / (n + k)
            rating["scored"] = (rating["att_home"] + rating["att_away"]) / 2.0
            rating["conceded"] = (rating["def_home"] + rating["def_away"]) / 2.0
            rating["n_eff"] = float(len(d["att_home"]) + len(d["att_away"]))
            self.team_ratings[tid] = rating
        self._fill_missing(fpl_teams)

    def _flat_ratings(self, fpl_teams):
        self.team_ratings = {}
        self._fill_missing(fpl_teams)

    def _fill_missing(self, fpl_teams):
        for t in (fpl_teams or []):
            self.team_ratings.setdefault(t["id"], {
                "att_home": LEAGUE_MEAN_GOALS, "def_home": LEAGUE_MEAN_GOALS,
                "att_away": LEAGUE_MEAN_GOALS, "def_away": LEAGUE_MEAN_GOALS,
                "scored": LEAGUE_MEAN_GOALS, "conceded": LEAGUE_MEAN_GOALS, "n_eff": 0.0,
            })

    # ------------------------------------------------------------ prediction

    def is_degraded(self) -> bool:
        """True when ratings carry no real fixture signal."""
        return bool(self.status.get("degraded"))

    def get_match_lambdas(self, home_id: int, away_id: int):
        """
        Expected goals for (home, away).

        Home advantage lives in the home/away-split ratings themselves, so it is
        not re-applied here - the original implementation multiplied by a fixed
        1.10/0.90 on top of ratings that already averaged over home fixtures.
        """
        L = self._league_mean if self._league_mean > 0 else LEAGUE_MEAN_GOALS
        h = self.team_ratings.get(home_id)
        a = self.team_ratings.get(away_id)
        if not h or not a:
            return LEAGUE_MEAN_GOALS * self.home_advantage, LEAGUE_MEAN_GOALS / self.home_advantage

        mu_home = h["att_home"] * a["def_away"] / L
        mu_away = a["att_away"] * h["def_home"] / L
        return max(0.05, mu_home), max(0.05, mu_away)

    def team_attack_baseline(self, team_id: int) -> float:
        """The team's own average attacking output, used to normalise player rates."""
        r = self.team_ratings.get(team_id)
        if not r:
            return LEAGUE_MEAN_GOALS
        base = r.get("scored", LEAGUE_MEAN_GOALS)
        return base if base > 0 else LEAGUE_MEAN_GOALS

    def export_ratings(self) -> Dict[int, Dict[str, float]]:
        """Snapshot suitable for seeding a later season via `prior_ratings`."""
        return {tid: dict(r) for tid, r in self.team_ratings.items()}
