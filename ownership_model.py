"""
Effective ownership projection.

EO = active ownership + active captaincy. It drives the rank-aware term in
`calculate_merv`, where values above 0.5 mean owning a player *reduces* your
variance versus the field.
"""
import math
from typing import Dict, Any, List

# Managers with abandoned teams drag reported ownership below what the active
# field actually holds; the meta players are more owned among live managers.
ACTIVE_SHRINKAGE = 1.30


def project_effective_ownership(player: Dict[str, Any],
                                top_100k_shrinkage: float = ACTIVE_SHRINKAGE) -> float:
    """Projected effective ownership, in [0, 2)."""
    try:
        raw = float(player.get("selected_by_percent", 0.0) or 0.0) / 100.0
    except (TypeError, ValueError):
        raw = 0.0

    active_ownership = min(0.95, raw * top_100k_shrinkage)

    # Captaincy concentrates far more sharply than ownership: a 40%-owned
    # premium may be captained by most of the people who own them, while a
    # 40%-owned defender is captained by almost nobody. The convex exponent
    # reproduces that concentration.
    p_captain = min(0.85, active_ownership ** 2.5)

    return active_ownership + p_captain


def build_eo_matrix(players: List[Dict[str, Any]]) -> Dict[int, float]:
    """Map player id -> projected effective ownership."""
    return {p["id"]: project_effective_ownership(p) for p in players}
