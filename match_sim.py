"""
Correlated match simulation.

Two things in the engine need the same machinery, so they share it here:

  1. **Bonus points.** Bonus is a rank statistic - 3/2/1 to the top three BPS
     scorers *in a match*. The previous model mapped one player's E[BPS] through
     a line, `max(0, (E[BPS] - 8) * 0.15)`, which cannot represent a
     top-3-of-22 outcome. Here we draw every involved player's BPS, rank them,
     award 3/2/1, and average over draws.

  2. **Rank-aware risk (MERV).** Variance was computed per player and then summed
     by the ILP, which treats three Arsenal players as three independent bets.
     Clean sheets and goals inside a team are strongly correlated, and team
     stacking is precisely where rank-awareness earns its keep. Simulating the
     match jointly gives correlated per-player point draws.

The BPS table below follows the official scoring. The previous E[BPS] omitted
every negative term and several positive ones.
"""
import logging
import math
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

POS_GKP, POS_DEF, POS_MID, POS_FWD = 1, 2, 3, 4
# Positions the engine actually models. FPL has shipped others (managers were
# element_type 5 in 2024-25), and an unmodelled position must be skipped
# rather than crash a whole gameweek's forecast.
MODELLED_POSITIONS = frozenset({POS_GKP, POS_DEF, POS_MID, POS_FWD})

POINTS_GOAL = {POS_GKP: 10, POS_DEF: 6, POS_MID: 5, POS_FWD: 4}
POINTS_CLEAN_SHEET = {POS_GKP: 4, POS_DEF: 4, POS_MID: 1, POS_FWD: 0}
POINTS_ASSIST = 3

# --- BPS table (official values) -------------------------------------------
BPS_PLAYING_60 = 6
BPS_PLAYING_UNDER_60 = 3
BPS_GOAL = {POS_GKP: 12, POS_DEF: 12, POS_MID: 18, POS_FWD: 24}
BPS_ASSIST = 9
BPS_CLEAN_SHEET = {POS_GKP: 12, POS_DEF: 12, POS_MID: 0, POS_FWD: 0}
BPS_SAVE = 2                    # per save
BPS_PENALTY_SAVE = 15
BPS_YELLOW = -3
BPS_RED = -9
BPS_OWN_GOAL = -6
BPS_PENALTY_MISS = -6
BPS_GOALS_CONCEDED_PER_2 = -4   # GK/DEF only
BPS_ERROR_LEADING_TO_GOAL = -3
BPS_DEFCON = 3                  # hitting the defensive-contribution threshold
# Volume terms, approximated from per-90 rates.
BPS_PER_RECOVERY = 1 / 3.0
BPS_PER_TACKLE_WON = 2.0
BPS_PER_KEY_PASS = 1.0
BPS_PER_BIG_CHANCE_CREATED = 3.0

DEFCON_THRESHOLD = {POS_DEF: 10, POS_MID: 12, POS_FWD: 12}


def _nb_params(mu: float, dispersion: float) -> Tuple[float, float]:
    """Negative-binomial (n, p) for a given mean and variance/mean ratio."""
    mu = max(1e-6, mu)
    var = max(mu * 1.0001, dispersion * mu)
    p = mu / var
    n = (mu * mu) / max(1e-9, var - mu)
    return n, p


class MatchSimulator:
    """Simulates one fixture jointly across all supplied players."""

    def __init__(self, n_sims: int = 2000, seed: int = 0):
        self.n_sims = n_sims
        self.seed = seed

    def simulate(
        self,
        players: List[Dict[str, Any]],
        team_lambdas: Dict[int, float],
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """
        `players` is a list of component dicts (as produced by
        EnsembleForecaster._predict_uncalibrated) each carrying at least:
            id, team, element_type, p_play, p_60, xg_cond, xa_cond,
            saves90, yc90, defcon_mu, defcon_dispersion, xmin
        `team_lambdas` maps team id -> expected goals conceded *by the opponent*
        (i.e. the goals this team's opponent is expected to score), used to draw
        a shared clean-sheet / goals-conceded outcome per team.

        Returns per-player expected bonus and the full simulated points matrix.
        """
        n = self.n_sims
        rng = rng or np.random.default_rng(self.seed)
        if not players:
            return {"bonus": {}, "points": {}, "n_sims": n}

        teams = sorted({p["team"] for p in players})
        # One shared goals-conceded draw per team per sim: this is what couples
        # team-mates' clean sheets together.
        conceded = {
            t: rng.poisson(lam=max(0.01, team_lambdas.get(t, 1.4)), size=n)
            for t in teams
        }
        clean_sheet = {t: (conceded[t] == 0) for t in teams}

        bps = np.zeros((len(players), n), dtype=float)
        pts = np.zeros((len(players), n), dtype=float)

        for i, p in enumerate(players):
            et = int(p.get("element_type", POS_MID))
            team = p["team"]
            p_play = float(p.get("p_play", 0.0))
            p_60 = float(p.get("p_60", 0.0))

            if p_play <= 0 or et not in MODELLED_POSITIONS:
                continue

            played = rng.random(n) < p_play
            # P(60+ | played); guard against p_60 > p_play from rounding.
            cond60 = min(1.0, p_60 / max(1e-9, p_play))
            played_60 = played & (rng.random(n) < cond60)

            goals = rng.poisson(lam=max(0.0, float(p.get("xg_cond", 0.0))), size=n) * played
            assists = rng.poisson(lam=max(0.0, float(p.get("xa_cond", 0.0))), size=n) * played

            cs = clean_sheet[team] & played_60
            team_conceded = conceded[team] * played_60

            yc = rng.random(n) < min(0.9, float(p.get("yc90", 0.0)) * (float(p.get("xmin", 0.0)) / 90.0))
            yc = yc & played

            saves = np.zeros(n, dtype=int)
            if et == POS_GKP:
                saves = rng.poisson(lam=max(0.0, float(p.get("saves90", 0.0))
                                            * (float(p.get("xmin", 0.0)) / 90.0)), size=n) * played

            # Defensive contributions: negative binomial on the per-match count.
            defcon_hit = np.zeros(n, dtype=bool)
            mu_dc = float(p.get("defcon_mu", 0.0))
            if et in DEFCON_THRESHOLD and mu_dc > 0:
                nb_n, nb_p = _nb_params(mu_dc, float(p.get("defcon_dispersion", 1.85)))
                counts = rng.negative_binomial(max(1e-3, nb_n), min(0.999, max(1e-6, nb_p)), size=n)
                defcon_hit = (counts >= DEFCON_THRESHOLD[et]) & played

            # ---- FPL points ----
            player_pts = played * 1.0 + played_60 * 1.0
            player_pts = player_pts + goals * POINTS_GOAL.get(et, 4) + assists * POINTS_ASSIST
            player_pts = player_pts + cs * POINTS_CLEAN_SHEET.get(et, 0)
            player_pts = player_pts + defcon_hit * 2.0
            player_pts = player_pts - yc * 1.0
            if et == POS_GKP:
                player_pts = player_pts + (saves // 3) * 1.0
            if et in (POS_GKP, POS_DEF):
                player_pts = player_pts - (team_conceded // 2) * 1.0

            # ---- BPS ----
            b = np.zeros(n, dtype=float)
            b += played_60 * BPS_PLAYING_60
            b += (played & ~played_60) * BPS_PLAYING_UNDER_60
            b += goals * BPS_GOAL.get(et, 18)
            b += assists * BPS_ASSIST
            b += cs * BPS_CLEAN_SHEET.get(et, 0)
            b += saves * BPS_SAVE
            b += yc * BPS_YELLOW
            b += defcon_hit * BPS_DEFCON
            if et in (POS_GKP, POS_DEF):
                b += (team_conceded // 2) * BPS_GOALS_CONCEDED_PER_2
            # Volume terms scaled by minutes actually played.
            min_frac = (played_60 * 1.0 + (played & ~played_60) * 0.4)
            b += min_frac * float(p.get("recoveries90", 0.0)) * BPS_PER_RECOVERY
            b += min_frac * float(p.get("tackles90", 0.0)) * BPS_PER_TACKLE_WON
            b += min_frac * float(p.get("key_passes90", 0.0)) * BPS_PER_KEY_PASS

            bps[i] = b
            pts[i] = player_pts

        # ---- bonus by rank within the match ----
        bonus = np.zeros_like(bps)
        # Only players who appeared are eligible.
        eligible = pts != 0
        masked = np.where(bps > 0, bps, -np.inf)
        if len(players) >= 3:
            order = np.argsort(-masked, axis=0)
            for rank, award in ((0, 3.0), (1, 2.0), (2, 1.0)):
                idx = order[rank]
                sims = np.arange(bps.shape[1])
                valid = np.isfinite(masked[idx, sims])
                bonus[idx[valid], sims[valid]] = award

        pts_with_bonus = pts + bonus

        return {
            "bonus": {p["id"]: float(bonus[i].mean()) for i, p in enumerate(players)},
            "points": {p["id"]: pts_with_bonus[i] for i, p in enumerate(players)},
            "mean_points": {p["id"]: float(pts_with_bonus[i].mean()) for i, p in enumerate(players)},
            "variance": {p["id"]: float(pts_with_bonus[i].var()) for i, p in enumerate(players)},
            "n_sims": self.n_sims,
        }


def squad_score_draws(
    point_draws: Dict[int, np.ndarray],
    squad_ids: List[int],
    captain_id: Optional[int] = None,
) -> np.ndarray:
    """
    Sum correlated per-player draws into a squad score distribution. Draws for
    players in the same fixture come from the same simulation, so team stacking
    correctly shows up as increased variance.
    """
    if not squad_ids:
        return np.zeros(0)
    lengths = {len(v) for k, v in point_draws.items() if k in squad_ids}
    if not lengths:
        return np.zeros(0)
    n = min(lengths)
    total = np.zeros(n)
    for pid in squad_ids:
        arr = point_draws.get(pid)
        if arr is None:
            continue
        total += arr[:n]
        if captain_id is not None and pid == captain_id:
            total += arr[:n]
    return total
