"""
Market-implied team strength.

Derives per-team attack/defence rates from bookmaker 1X2 and over/under 2.5
prices, which are a far better fixture signal than FPL's own FDR.

Changes over the original implementation:
  * team names resolve through `team_mapping` (one-to-one, alias-seeded)
  * ratings are a time-decayed, opponent-adjusted log-linear fit with ridge
    shrinkage toward the league mean, instead of each club's average over its
    own fixtures - which rated a club by its schedule as much as its strength
  * one attack and one defence per club plus a fitted league-wide home
    advantage, applied exactly once (see `get_match_lambdas`)
  * missing/thin odds no longer degrade silently to a flat 1.4/1.4 for every
    club - a fallback chain runs and `status` records which source was used
  * multiple bookmakers are tried, with the consensus columns as backstop
"""
import logging
import os
import random
import time
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
# letting two fixtures define a team. In the log-linear fit this is a ridge
# penalty on every team parameter, scaled to add this many matches' worth of
# league-average evidence (see `fit_log_linear_ratings`).
RATING_PRIOR_STRENGTH = 4.0
# One attack and one defence per team plus a league-wide home advantage (False),
# or separate home and away attack and defence for every team (True).
#
# Measured, not assumed. Fitting on every matchweek's history and pricing the
# next matchweek, across all five cached seasons (1,850 matches), the pooled
# rating misses the market's own implied goals by 0.145 log-RMSE against 0.192
# split - and by 0.191 against 0.245 in the first ~100 matches of a season. It
# wins at every prior strength tried (1 to 16). Splitting halves the evidence
# behind every parameter, and the market's home edge is close enough to
# uniform across clubs that the extra freedom buys nothing back. Kept as a flag
# so the decision stays re-measurable.
RATING_VENUE_SPLIT = False

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


def fit_log_linear_ratings(
    home: List[Any], away: List[Any], mu_h, mu_a, weights,
    prior_strength: Optional[float] = None, venue_split: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Opponent-adjusted team ratings from per-match expected goals.

    A weighted ridge regression of the market's implied goals on

        log mu_home = c + h + attack[home] - defence[away]
        log mu_away = c     + attack[away] - defence[home]

    so a team's attack is what it scores against an *average* defence, not
    against whichever defences it happened to meet. With `venue_split` each team
    gets separate home and away attack and defence terms instead, and the home
    edge lives in two intercepts.

    The ridge is `RATING_PRIOR_STRENGTH` in the role it had in the averaging fit:
    k pseudo-matches of league-average evidence on every team term. For a team
    whose opponents were average this is exactly the old shrinkage, on log
    goals: its term is (sum of w * log-goals above the league) / (n_eff + k).
    The intercepts are not penalised, so teams are pulled toward the league and
    the league toward nothing.

    Least squares on log goals rather than a Poisson likelihood, because the
    targets are bookmaker means, not counts. A Poisson fit weights each match by
    its expected goals, so a side that met strong defences carries less
    information and is shrunk harder - two identical teams with different
    schedules then come apart through the prior even though the fit itself is
    opponent-adjusted.

    Every observation has exactly one attacker and one defender, so at the
    optimum each block of team terms sums to zero; the fit is solved in that
    subspace, which also keeps it well-posed with no ridge at all. The "average
    opponent" is therefore the unweighted average of the teams in the fit.

    Returns {"teams": {id: {att_home, att_away, def_home, def_away, n_eff}},
             "home_base", "away_base"}: each rating is expected goals against an
    average opponent at that venue, and the bases are the average team's.
    """
    if prior_strength is None:
        prior_strength = RATING_PRIOR_STRENGTH
    if venue_split is None:
        venue_split = RATING_VENUE_SPLIT

    teams = sorted(set(home) | set(away))
    idx = {t: i for i, t in enumerate(teams)}
    T, n = len(teams), len(home)
    hi = np.array([idx[t] for t in home], dtype=int)
    ai = np.array([idx[t] for t in away], dtype=int)
    rows = np.arange(n)

    # Columns: two unpenalised intercepts, then the penalised team blocks.
    #   pooled: [c, h | attack, defence]
    #   split:  [c_home, c_away | att_home, att_away, def_home, def_away]
    blocks = 4 if venue_split else 2
    p = 2 + blocks * T
    Xh = np.zeros((n, p))
    Xa = np.zeros((n, p))
    start = [2 + k * T for k in range(blocks)]
    if venue_split:
        att_h, att_a, def_h, def_a = start
        Xh[:, 0] = 1.0
        Xh[rows, att_h + hi] = 1.0
        Xh[rows, def_a + ai] = -1.0      # the visitors, defending away
        Xa[:, 1] = 1.0
        Xa[rows, att_a + ai] = 1.0
        Xa[rows, def_h + hi] = -1.0      # the hosts, defending at home
    else:
        att, dfn = start
        Xh[:, 0] = Xh[:, 1] = 1.0
        Xh[rows, att + hi] = 1.0
        Xh[rows, dfn + ai] = -1.0
        Xa[:, 0] = 1.0
        Xa[rows, att + ai] = 1.0
        Xa[rows, dfn + hi] = -1.0
    X = np.vstack([Xh, Xa])
    y = np.log(np.concatenate([np.asarray(mu_h, float), np.asarray(mu_a, float)]))
    w = np.concatenate([np.asarray(weights, float)] * 2)

    # Each team block lives in the sum-to-zero subspace: B maps the reduced
    # coefficients to the full ones. Z has orthonormal columns, so the ridge on
    # the full team terms is the same ridge on the reduced ones.
    Z = np.linalg.qr((np.eye(T) - 1.0 / T)[:, :T - 1])[0] if T > 1 else np.zeros((1, 0))
    B = np.zeros((p, 2 + blocks * Z.shape[1]))
    B[0, 0] = B[1, 1] = 1.0
    for k, s in enumerate(start):
        B[s:s + T, 2 + k * Z.shape[1]:2 + (k + 1) * Z.shape[1]] = Z
    Xr = X @ B
    pen = np.r_[0.0, 0.0, np.full(B.shape[1] - 2, float(prior_strength))]
    theta = B @ np.linalg.solve((Xr.T * w) @ Xr + np.diag(pen), Xr.T @ (w * y))

    n_eff = (np.bincount(hi, weights=w[:n], minlength=T)
             + np.bincount(ai, weights=w[:n], minlength=T))
    out: Dict[Any, Dict[str, float]] = {}
    if venue_split:
        c_h, c_a = theta[0], theta[1]
        home_base, away_base = math.exp(c_h), math.exp(c_a)
        for t, i in idx.items():
            out[t] = {
                "att_home": math.exp(c_h + theta[att_h + i]),
                "att_away": math.exp(c_a + theta[att_a + i]),
                # Conceded at home is the visitors' scoring: the away intercept.
                "def_home": math.exp(c_a - theta[def_h + i]),
                "def_away": math.exp(c_h - theta[def_a + i]),
                "n_eff": float(n_eff[i]),
            }
    else:
        c, h = theta[0], theta[1]
        home_base, away_base = math.exp(c + h), math.exp(c)
        for t, i in idx.items():
            a_t, d_t = theta[att + i], theta[dfn + i]
            out[t] = {
                "att_home": math.exp(c + h + a_t),
                "att_away": math.exp(c + a_t),
                "def_home": math.exp(c - d_t),
                "def_away": math.exp(c + h - d_t),
                "n_eff": float(n_eff[i]),
            }
    return {"teams": out, "home_base": home_base, "away_base": away_base}


# football-data.co.uk answers a transient outage with a 503 and a `retry-after`
# header. Probed three times four seconds apart during a real outage it returned
# 341, 70 and 77 - so it is not a per-request estimate of recovery, which would
# count down in step with elapsed time.
#
# It is very likely *deliberate jitter* rather than noise: the standard
# load-shedding pattern randomises Retry-After precisely so that thousands of
# clients hitting the same 503 do not all return at the same instant and
# re-create the overload. Read that way the header is meaningful as a back-off
# signal and meaningless as an ETA, which is the opposite of how it reads.
#
# Two consequences. We do not treat it as an ETA - a single daily client gains
# nothing from an arbitrary 70-to-341-second wait, and it would make the job's
# duration unpredictable. But we do add jitter of our own, because deterministic
# backoff across many clients is exactly what the server's jitter exists to
# prevent, and being a good citizen of a free data source costs nothing.
#
# The half that actually caused the bug is neither: a failure must not be
# *cached*. The previous code stored None against the season, so one 503 left
# the whole process fixture-blind - every later gameweek read that None, fell
# back to flat team ratings, and the resulting forecast was published anyway.
ODDS_MAX_ATTEMPTS = 4
ODDS_BASE_DELAY = 4.0          # 4s, 8s, 16s plus jitter - bounded


# A last-good copy on disk, because the ratings barely care how fresh the odds
# are. Team ratings decay with a 10-match half-life, so a file from yesterday is
# missing at most one round and is worth far more than the fallbacks below it:
# `results_poisson` uses real scorelines but knows nothing about *upcoming*
# fixtures, which is the whole reason for using the market in the first place.
#
# This does not rescue an outage that begins before the first successful fetch -
# there is nothing to fall back to - but it turns every later one into a
# non-event. football-data.co.uk was 503 for more than a day in September 2026,
# which cost two experiments and would have degraded every scheduled run.
ODDS_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "odds")
# Beyond this the copy is missing too many rounds to be worth preferring over a
# fit on actual results - UNLESS it is a complete season, which cannot go stale.
ODDS_CACHE_MAX_AGE_DAYS = 45.0
MATCHES_IN_A_FULL_SEASON = 380


def _cache_path(season_str: str) -> str:
    return os.path.join(ODDS_CACHE_DIR, f"{season_str}.csv")


def _save_cached_odds(season_str: str, df) -> None:
    """Keep the last good copy. Never fatal: a cache miss beats a crash."""
    try:
        os.makedirs(ODDS_CACHE_DIR, exist_ok=True)
        tmp = _cache_path(season_str) + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, _cache_path(season_str))   # atomic; no torn file
    except Exception as e:
        logger.debug("Could not cache odds for %s: %s", season_str, e)


def _data_age_days(df) -> float:
    """
    Days since the newest match in the file.

    This is the honest measure of how stale the market information is, and the
    file's mtime is not. An mtime says when the bytes were written, which is a
    different thing in two ways that both matter here: a copy written today
    from a mirror whose data ends six weeks ago is six weeks stale, and a fresh
    git checkout resets every mtime to now - so once `data/odds` is committed,
    an mtime check would declare a two-year-old file perfectly fresh.
    """
    d = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce").dropna()
    if d.empty:
        return float("inf")
    return max(0.0, (pd.Timestamp.now() - d.max()).total_seconds() / 86400.0)


def _load_cached_odds(season_str: str):
    """(dataframe, age_in_days), or (None, None)."""
    path = _cache_path(season_str)
    try:
        if not os.path.exists(path):
            return None, None
        df = pd.read_csv(path)
        age = _data_age_days(df)
        # A finished season is complete market information and never goes off;
        # only a part-played season can be missing rounds worth caring about.
        if age > ODDS_CACHE_MAX_AGE_DAYS and len(df) < MATCHES_IN_A_FULL_SEASON:
            logger.warning(
                "Cached odds for %s stop %.0f days ago and cover only %d "
                "matches (limit %.0f days); ignoring them in favour of the "
                "fallback chain.",
                season_str, age, len(df), ODDS_CACHE_MAX_AGE_DAYS)
            return None, None
        return df, age
    except Exception as e:
        logger.debug("Could not read cached odds for %s: %s", season_str, e)
        return None, None


def _read_odds_csv(url: str, season_str: str):
    """The odds CSV, or None if it could not be fetched. Never caches failure."""
    for attempt in range(ODDS_MAX_ATTEMPTS):
        try:
            return pd.read_csv(url)
        except Exception as e:
            # 404 means the season is genuinely not published; retrying cannot
            # help. Anything else - 5xx, timeouts, resets - is transient.
            if getattr(e, "code", None) == 404:
                logger.warning("No odds file published for %s (404).", season_str)
                return None
            if attempt == ODDS_MAX_ATTEMPTS - 1:
                # Only that the PRIMARY is unavailable. This function knows
                # nothing about the cache, the mirror or the rating fallbacks,
                # so it must not predict the outcome - it used to announce
                # "team ratings will be FLAT" and was then immediately followed
                # by a successful load from the cache, which makes an outage
                # look far worse than it is when read back in a CI log.
                # `fetch_odds` says what actually happened.
                logger.warning(
                    "Primary odds feed unavailable for %s after %d attempts "
                    "(%s); trying the fallbacks.",
                    season_str, ODDS_MAX_ATTEMPTS, e)
                return None
            # Jittered so repeated clients do not synchronise on the retry.
            wait = ODDS_BASE_DELAY * (2 ** attempt) * (1.0 + random.random())
            logger.warning(
                "Market odds for %s unavailable (%s); retrying in %.0fs [%d/%d]",
                season_str, e, wait, attempt + 1, ODDS_MAX_ATTEMPTS - 1)
            time.sleep(wait)
    return None


# ------------------------------------------------------------------ the mirror
#
# football-data.co.uk is one free static host, and in September 2026 it returned
# 503 for every season for more than a day - homepage included, any user agent,
# with `x-ws-origin: available`, so the CDN was shedding rather than the origin
# being down. Retries cannot help with that and the disk cache cannot help at
# all if the outage starts before the first successful fetch, which is exactly
# what happened.
#
# xgabora/Club-Football-Match-Data is an MIT-licensed redistribution of the same
# football-data.co.uk match data, updated periodically, carrying Bet365 1X2 and
# over/under 2.5 for every EPL match. Validated against outcomes across five
# seasons before being wired in: de-vigged home-win prices track actual home
# wins to within 1-5 points, implied total goals to within 0.2, and the
# price-bucket calibration is monotone in every season - so no swap and no
# column misalignment.
#
# It is one 40MB+ file covering every league and season, which is why it sits
# AFTER the disk cache in the chain and is memoised for the process: a run asks
# for two seasons (current, plus the prior one for priors) and must not pay for
# the download twice.
ODDS_MIRROR_URL = (
    "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data/"
    "main/data/Matches.csv"
)
# Their column names -> football-data.co.uk's, so everything downstream - the
# bookmaker-group selection, the disk cache format, the team-name mapping - is
# untouched by where the numbers came from.
_MIRROR_COLUMNS = {
    "OddHome": "B365H", "OddDraw": "B365D", "OddAway": "B365A",
    "Over25": "B365>2.5", "Under25": "B365<2.5",
    "FTHome": "FTHG", "FTAway": "FTAG",
}
_MIRROR_FRAME: Dict[str, Any] = {}


def _season_start_year(dates):
    """
    The season a match belongs to, by its date.

    August rather than July: no EPL season has ever started before August, and
    the covid-delayed 2019-20 ran to 26 July 2020. A July cutoff files those
    final matches under 2020-21 and hands back a 446-match season.
    """
    return dates.dt.year.where(dates.dt.month >= 8, dates.dt.year - 1)


def _fetch_mirror_odds(season_str: str):
    """One season in football-data.co.uk's own format, or None."""
    try:
        if "df" not in _MIRROR_FRAME:
            logger.warning(
                "Falling back to the odds mirror (%s). One large file, fetched "
                "once for this run.", ODDS_MIRROR_URL)
            _MIRROR_FRAME["df"] = pd.read_csv(ODDS_MIRROR_URL, low_memory=False)
        raw = _MIRROR_FRAME["df"]

        want = int(f"20{season_str[:2]}")
        e0 = raw[raw["Division"] == "E0"].copy()
        dates = pd.to_datetime(e0["MatchDate"], errors="coerce")
        sub = e0[_season_start_year(dates) == want].copy()
        if sub.empty:
            logger.warning("Odds mirror has no E0 matches for %s.", season_str)
            return None

        sub["Date"] = pd.to_datetime(sub["MatchDate"]).dt.strftime("%d/%m/%Y")
        out = sub.rename(columns=_MIRROR_COLUMNS)
        keep = ["Date", "HomeTeam", "AwayTeam"] + [
            c for c in _MIRROR_COLUMNS.values() if c in out.columns]
        out = out[keep].sort_values(
            "Date", key=lambda x: pd.to_datetime(x, format="%d/%m/%Y"))

        if len(out) > MATCHES_IN_A_FULL_SEASON + 20:
            # A season boundary that has gone wrong, rather than data worth
            # using. Better to fall through than to fit ratings on two seasons.
            logger.error("Odds mirror returned %d matches for %s; refusing it.",
                         len(out), season_str)
            return None
        logger.warning("Odds mirror supplied %d matches for %s.",
                       len(out), season_str)
        return out.reset_index(drop=True)
    except Exception as e:
        logger.error("Odds mirror unavailable for %s (%s).", season_str, e)
        return None


class MarketOddsModel:
    def __init__(self):
        self.odds_df: Optional[pd.DataFrame] = None
        self.team_ratings: Dict[int, Dict[str, float]] = {}
        self.season_str: Optional[str] = None
        self.FPL_TO_FD: Dict[int, str] = {}
        self.FD_TO_FPL: Dict[str, int] = {}
        # Informational: the average team's home goals over its all-venue mean.
        self.home_advantage: float = 1.0
        # Which source the current ratings came from, and how trustworthy.
        self.status: Dict[str, Any] = {"source": "uninitialised", "n_matches": 0}
        # The average team's home and away goals; see `_set_venue_bases`.
        self._home_base: float = LEAGUE_MEAN_GOALS
        self._away_base: float = LEAGUE_MEAN_GOALS
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
        df = _read_odds_csv(url, season_str)
        if df is None:
            # Before giving up, the last good copy. Slightly stale market
            # information still prices fixtures; the fallbacks below cannot.
            df, age = _load_cached_odds(season_str)
            if df is not None:
                logger.warning(
                    "Odds feed unavailable for %s; using the cached copy, whose "
                    "newest match is %.1f days old. Ratings will miss anything "
                    "since.", season_str, age)
        if df is None:
            # Then the mirror. Last, because it is a single large file and the
            # two cheaper sources are usually enough - but it is the only one
            # that can help when the primary is down and nothing was ever
            # cached, which is the case that actually bit.
            df = _fetch_mirror_odds(season_str)
        if df is None:
            # Every source is exhausted, so this is the one place that can
            # honestly say what the consequence is.
            logger.error(
                "No market odds for %s from the feed, the cache or the mirror. "
                "Team ratings will fall back to results, or to FLAT if there "
                "are none - the forecast will carry little or no fixture "
                "signal and the plan will be marked degraded.", season_str)
            # NOT cached. A failed fetch is not the same fact as "this season
            # has no odds", and conflating them meant one 503 left the whole
            # process fixture-blind: every later gameweek read the cached None
            # and fell back to flat team ratings.
            self.odds_df = None
            return False

        # Cache the file as fetched. Everything below rewrites `Date` into
        # datetimes, and a datetime round-trips through CSV as ISO, which the
        # `%d/%m/%Y` parse above turns into NaT - so caching the mutated frame
        # produced a file that reloaded as zero matches.
        as_fetched = df.copy()

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
            # This one is genuine and worth caching: the file arrived and does
            # not carry prices, which no amount of retrying will change.
            logger.warning("Odds file for %s has no recognised bookmaker columns", season_str)
            _GLOBAL_ODDS_CACHE[season_str] = None
            self.odds_df = None
            return False

        merged = frames[0]
        for extra in frames[1:]:
            merged = merged.combine_first(extra)
        merged = merged.dropna(subset=["Date", "HomeTeam", "AwayTeam", "H", "D", "A", "O", "U"])

        merged = merged.sort_values("Date").reset_index(drop=True)
        self.odds_df = self._augment_with_lambdas(merged)
        _GLOBAL_ODDS_CACHE[season_str] = self.odds_df
        _save_cached_odds(season_str, as_fetched)
        logger.info("Loaded %d priced matches for season %s", len(self.odds_df), season_str)
        return len(self.odds_df) > 0

    def _augment_with_lambdas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Precompute each match's (mu_home, mu_away) once, when the file is loaded.

        Inverting the over/under price and splitting the total each cost a
        `root_scalar` solve, and `split_goals` builds a 10x10 Dixon-Coles grid
        per iteration. Those depend only on the row's prices, never on the
        as-of date, but the ratings fit used to redo all of it on every call -
        once per gameweek in a backtest, and now several times per gameweek for
        the leak-free calibration refit. Doing it once per season instead is
        what makes an honest calibration affordable.
        """
        mu_h, mu_a = [], []
        for row in df.itertuples(index=False):
            try:
                p_h, p_d, p_a = _devig(1.0 / row.H, 1.0 / row.D, 1.0 / row.A)
                p_over, _ = _devig(1.0 / row.O, 1.0 / row.U)
                if not all(np.isfinite([p_h, p_d, p_a, p_over])):
                    raise ValueError
                total = self.implied_total_goals(p_over)
                h, a = self.split_goals(total, p_h, p_a)
            except (ZeroDivisionError, TypeError, ValueError):
                h = a = np.nan
            mu_h.append(h)
            mu_a.append(a)
        out = df.copy()
        out["mu_h"] = mu_h
        out["mu_a"] = mu_a
        return out

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

        self._set_venue_bases()
        self.status["n_teams_rated"] = len(self.team_ratings)
        return self.status

    def _set_venue_bases(self):
        """
        The average team's home and away goals, which `get_match_lambdas`
        divides out.

        It has to be one base per venue. A home attack rating and an away
        defence rating both already carry the home edge, so dividing their
        product by a single all-venue mean applied that edge twice: fitted to a
        league where every club is identical and scores 1.6 at home and 1.2 away,
        the old normaliser predicted 1.83 and 1.03.

        Geometric means, so that for the log-linear fit - whose team terms sum
        to zero - these are exactly its intercepts; for the fallbacks they are
        the same quantity estimated from whatever ratings they produced.
        """
        def gmean(key):
            vals = [r.get(key) for r in self.team_ratings.values()]
            vals = [v for v in vals if v is not None and np.isfinite(v) and v > 0]
            return float(np.exp(np.mean(np.log(vals)))) if vals else LEAGUE_MEAN_GOALS

        home = math.sqrt(gmean("att_home") * gmean("def_away"))
        away = math.sqrt(gmean("att_away") * gmean("def_home"))
        if not (np.isfinite(home) and home > 0):
            home = LEAGUE_MEAN_GOALS
        if not (np.isfinite(away) and away > 0):
            away = LEAGUE_MEAN_GOALS
        self._home_base, self._away_base = home, away
        self._league_mean = (home + away) / 2.0
        self.home_advantage = home / self._league_mean

    def _fit_from_odds(self, current_gw_date) -> int:
        df = self.odds_df
        if df is None or len(df) == 0 or not self.FD_TO_FPL:
            return 0
        if current_gw_date is not None:
            df = df[df["Date"] <= current_gw_date]
        if len(df) == 0:
            return 0

        latest = df["Date"].max()
        home_ids, away_ids, goals_h, goals_a, weights = [], [], [], [], []

        for row in df.itertuples(index=False):
            h_id = self.FD_TO_FPL.get(row.HomeTeam)
            a_id = self.FD_TO_FPL.get(row.AwayTeam)
            if not h_id or not a_id:
                continue
            # Precomputed once per season by `_augment_with_lambdas`; fall back
            # to solving in place for a frame that predates that column.
            mu_h = getattr(row, "mu_h", None)
            mu_a = getattr(row, "mu_a", None)
            if mu_h is None or mu_a is None or not np.isfinite([mu_h, mu_a]).all():
                try:
                    p_h, p_d, p_a = _devig(1.0 / row.H, 1.0 / row.D, 1.0 / row.A)
                    p_over, _ = _devig(1.0 / row.O, 1.0 / row.U)
                except (ZeroDivisionError, TypeError):
                    continue
                if not all(np.isfinite([p_h, p_d, p_a, p_over])):
                    continue
                mu_h, mu_a = self.split_goals(self.implied_total_goals(p_over), p_h, p_a)
            if not (mu_h > 0 and mu_a > 0):
                continue    # the fit is on log goals

            # Exponential decay by match age, in matches-equivalent days.
            age_days = (latest - row.Date).days if pd.notna(row.Date) else 0
            w = 0.5 ** ((age_days / 7.0) / RATING_HALF_LIFE)

            home_ids.append(h_id)
            away_ids.append(a_id)
            goals_h.append(float(mu_h))
            goals_a.append(float(mu_a))
            weights.append(w)

        if not home_ids:
            return 0

        # Opponent-adjusted. This used to average each team's implied goals
        # over its own fixtures and then combine those averages as if they were
        # opponent-neutral, which they are not: a side that opened against the
        # strong clubs came out with too little attack and too much conceded,
        # and every opponent it met was skewed the other way.
        fit = fit_log_linear_ratings(home_ids, away_ids, goals_h, goals_a, weights)
        for tid, rating in fit["teams"].items():
            # Convenience aggregates used by the rest of the engine.
            rating["scored"] = (rating["att_home"] + rating["att_away"]) / 2.0
            rating["conceded"] = (rating["def_home"] + rating["def_away"]) / 2.0
            self.team_ratings[tid] = rating

        return len(home_ids)

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

        Each rating is expected goals against an average opponent at that venue,
        so a home attack times an away defence counts the home edge twice; it is
        divided out once, by the average team's goals at that venue. The
        original implementation multiplied by a fixed 1.10/0.90 on top of
        ratings that already contained it, and its replacement divided both
        venues by one all-venue mean, which squared it.
        """
        L_home = self._home_base if self._home_base > 0 else LEAGUE_MEAN_GOALS
        L_away = self._away_base if self._away_base > 0 else LEAGUE_MEAN_GOALS
        h = self.team_ratings.get(home_id)
        a = self.team_ratings.get(away_id)
        if not h or not a:
            return L_home, L_away

        mu_home = h["att_home"] * a["def_away"] / L_home
        mu_away = a["att_away"] * h["def_home"] / L_away
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
