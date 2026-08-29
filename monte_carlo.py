"""
Rank-aware valuation.

The per-player Monte Carlo that used to live here has moved into `match_sim`,
where draws are generated jointly across a fixture. Summing independent
per-player variances treated three team-mates as three independent bets, which
is exactly backwards: clean sheets and goals inside a team are correlated, and
team stacking is the case rank-awareness exists to price.

What remains here is the valuation function itself.
"""
import logging
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def simulate_player_variance(comps: dict, n_sims: int = 2000) -> float:
    """
    Variance of a single player's gameweek score.

    Retained for callers that have components but no match context. Prefer the
    correlated draws from `match_sim.MatchSimulator.simulate`, which this
    delegates to so the two paths cannot drift apart in their scoring rules.
    """
    from match_sim import MatchSimulator

    if comps.get("p_play", 0.0) <= 0:
        return 0.0

    seed = int(abs(hash((comps.get("id", 42), round(comps.get("math_pts", 0.0), 3)))) % (2 ** 31))
    sim = MatchSimulator(n_sims=n_sims, seed=seed)
    player = dict(comps)
    player.setdefault("team", 0)
    team_lambdas = {player["team"]: comps.get("opp_xg", 1.4)}
    out = sim.simulate([player], team_lambdas)
    return float(out["variance"].get(player.get("id"), 0.0))


def calculate_merv(xp: float, variance: float, eo: float,
                   risk_aversion: float = 0.05) -> float:
    """
    Marginal Expected Rank Value.

    Owning a player changes your variance *relative to the field* by
    (1 - 2 * EO) * Var:

      * EO < 0.5  -> a differential. Owning them adds variance versus the field.
      * EO > 0.5  -> template. Owning them *removes* variance, because the field
                     already carries it and not owning them is the risky choice.

    The previous implementation clamped EO at 0.5, which made the second branch
    unreachable: it only ever penalised differentials and never rewarded template
    cover, biasing squads toward the field in the way that guarantees an average
    rank. EO is now bounded only to a sane [0, 2] range (it can exceed 1 because
    it includes captaincy).
    """
    eo_bounded = float(np.clip(eo, 0.0, 2.0))
    delta_variance = (1.0 - 2.0 * eo_bounded) * variance
    return float(xp - risk_aversion * delta_variance)
