import numpy as np
from typing import Dict, Any

def simulate_player_variance(comps: Dict[str, Any], n_sims=10000) -> float:
    """
    Simulates a player's gameweek 10,000 times using numpy to find their variance.
    """
    if comps.get("math_pts", 0) == 0:
        return 0.0
        
    p_play = comps["p_play"]
    p_60 = comps["p_60"]
    xg = comps["xg"]
    xa = comps["xa"]
    p_cs = comps["p_cs"]
    element_type = comps["element_type"]
    
    # 1. Simulate Appearance
    played = np.random.rand(n_sims) < p_play
    played_60 = played & (np.random.rand(n_sims) < (p_60 / (p_play + 1e-6)))
    
    # 2. Simulate Goals and Assists (Poisson)
    goals = np.random.poisson(lam=xg, size=n_sims) * played
    assists = np.random.poisson(lam=xa, size=n_sims) * played
    
    # 3. Simulate Clean Sheets
    cs = np.random.rand(n_sims) < p_cs
    cs = cs & played_60
    
    # 4. Points calculation
    pts = played * 1.0 + played_60 * 1.0
    
    if element_type == 1:
        pts += goals * 6 + assists * 3 + cs * 4
    elif element_type == 2:
        pts += goals * 6 + assists * 3 + cs * 4
    elif element_type == 3:
        pts += goals * 5 + assists * 3 + cs * 1
    elif element_type == 4:
        pts += goals * 4 + assists * 3
        
    static_extras = comps.get("xSaves", 0) + comps.get("xDefCon", 0) + comps.get("xBonus", 0) - comps.get("xConc_penalty", 0)
    pts = pts + (static_extras * played)
    
    return float(np.var(pts))

def calculate_merv(xp: float, variance: float, eo: float, risk_aversion: float = 0.05) -> float:
    """
    Calculates Marginal Expected Rank Value (MERV).
    Uses Mean-Variance optimization for Rank-Awareness (Probability of a Green Arrow).
    
    If risk_aversion > 0: We penalize variance.
    Delta Variance of picking a player vs field = (1 - 2 * EO) * Var.
    If EO > 0.5, picking them REDUCES variance (defensive shield).
    If EO < 0.5, picking them INCREASES variance (differential).
    """
    delta_variance = (1.0 - 2.0 * eo) * variance
    merv = xp - (risk_aversion * delta_variance)
    return float(merv)

