import math
from typing import Dict, Any, List

def project_effective_ownership(player: Dict[str, Any], top_100k_shrinkage: float = 1.3) -> float:
    """
    Projects the Effective Ownership (EO) of a player.
    EO = Active Ownership + Active Captaincy.
    """
    raw_ownership = float(player.get("selected_by_percent", 0.0)) / 100.0
    
    # Dead teams pull down average ownership of "meta" players.
    active_ownership = min(0.95, raw_ownership * top_100k_shrinkage)
    
    # Estimate Captaincy probability based on active ownership.
    p_captain = active_ownership * (active_ownership ** 1.5)
    p_captain = min(0.85, p_captain)
    
    effective_ownership = active_ownership + p_captain
    return effective_ownership

def build_eo_matrix(players: List[Dict[str, Any]]) -> Dict[int, float]:
    """
    Builds a dictionary mapping player IDs to their projected Effective Ownership.
    """
    eo_matrix = {}
    for p in players:
        eo_matrix[p["id"]] = project_effective_ownership(p)
    return eo_matrix

