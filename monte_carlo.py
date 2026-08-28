import numpy as np
from typing import Dict, Any

import numpy as np

def simulate_player_variance(comps: dict, n_sims=10000) -> float:
    """
    Simulates a player's gameweek 10,000 times using numpy to find their variance.
    """
    if comps.get("math_pts", 0) == 0:
        return 0.0
        
    np.random.seed(comps.get("id", 42) + int(comps.get("math_pts", 0)*100))
        
    p_play = comps.get("p_play", 0.0)
    p_60 = comps.get("p_60", 0.0)
    
    if p_play <= 0:
        return 0.0
        
    p_cs = comps.get("p_cs", 0.0)
    element_type = comps.get("element_type", 3)
    
    # Un-discount rates for the conditional simulation
    xg_cond = comps.get("xg", 0.0) / p_play
    xa_cond = comps.get("xa", 0.0) / p_play
    
    # 1. Simulate Appearance
    played = np.random.rand(n_sims) < p_play
    played_60 = played & (np.random.rand(n_sims) < (p_60 / (p_play + 1e-6)))
    
    # 2. Simulate Goals and Assists (Poisson)
    goals = np.random.poisson(lam=xg_cond, size=n_sims) * played
    assists = np.random.poisson(lam=xa_cond, size=n_sims) * played
    
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
    # Bound the EO term so it doesn't cross 1.0 and flip the sign to reward variance
    eo_bounded = min(0.5, eo)
    delta_variance = (1.0 - 2.0 * eo_bounded) * variance
    merv = xp - (risk_aversion * delta_variance)
    return float(merv)

