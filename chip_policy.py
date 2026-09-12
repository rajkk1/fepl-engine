"""
When to play a chip.

This lived inside `weekly_manager` and was therefore measured by nothing: the
season replay drives the same optimiser but never passed `active_chip`, so a
wildcard, free hit, bench boost and triple captain were never played in any of
the three seasons the README reports. That is how chip rules keyed to a loop
index rather than a gameweek survived in the optimiser.

It sits here so the weekly job and the replay run the *same* policy, and a
change to the policy shows up in the replay's points rather than only in the
weekly output nobody scores.

The shape of the policy: a chip is worth playing when the marginal expected
points it buys, over the whole horizon, beat a threshold that falls as the
season runs out. Early on the bar is high because a better week is probably
still coming; by GW35 there is nothing left to wait for. The wildcard is scaled
against the expiry of its own half rather than the end of the season, because
the first-half set is gone after GW19 whether or not it was used.
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

SEASON_LAST_GW = 38
# The first-half chip set expires at the GW19 deadline; a second set unlocks
# from GW20. `weekly_manager.CHIP_HALF_SPLIT_GW` is the same boundary.
FIRST_HALF_LAST_GW = 19

# Marginal horizon xP a chip must buy before it is worth spending, at the start
# of a season. Scaled down by how much season is left; see the module docstring.
CHIP_BASE_THRESHOLD = {"tc": 10.0, "bb": 12.0, "fh": 15.0, "wc": 20.0}


def chip_thresholds(current_gw: int) -> Dict[str, float]:
    """The bar each chip must clear this gameweek, in horizon xP."""
    gws_until_season_end = max(1, SEASON_LAST_GW - current_gw)
    if current_gw <= FIRST_HALF_LAST_GW:
        gws_until_wc_expiry = max(1, FIRST_HALF_LAST_GW - current_gw)
    else:
        gws_until_wc_expiry = max(1, SEASON_LAST_GW - current_gw)

    season_left = gws_until_season_end / SEASON_LAST_GW
    wc_left = gws_until_wc_expiry / FIRST_HALF_LAST_GW
    return {
        "tc": CHIP_BASE_THRESHOLD["tc"] * season_left,
        "bb": CHIP_BASE_THRESHOLD["bb"] * season_left,
        "fh": CHIP_BASE_THRESHOLD["fh"] * season_left,
        "wc": CHIP_BASE_THRESHOLD["wc"] * wc_left,
    }


def choose_chip(solve: Callable[[Optional[str], Optional[int]], Dict[str, Any]],
                horizon_gws: Sequence[int],
                current_gw: int,
                available_chips: Sequence[str]) -> Dict[str, Any]:
    """
    Search chip x gameweek across the horizon.

    `solve(chip, gw)` must return an optimiser result; it is called once with
    (None, None) for the baseline and once per candidate. Returns a dict with
    `chip` ("" for none), `gw`, `gain` and the winning `res`, so the caller can
    use the plan that was actually scored rather than re-solving it.

    Searching gameweeks as well as chips is what makes "hold it for the double"
    reachable at all. It is only meaningful because the optimiser prices a chip
    in the gameweek it is assigned to - when it did not, every candidate past
    the first was scored against rules the game does not have.
    """
    base_res = solve(None, None)
    best: Dict[str, Any] = {
        "chip": "", "gw": horizon_gws[0], "gain": 0.0, "res": base_res}
    base_xp = base_res.get("total_xp", 0.0)

    thresholds = chip_thresholds(current_gw)
    for chip in [c for c in thresholds if c in available_chips]:
        for gw in horizon_gws:
            try:
                res = solve(chip, gw)
            except Exception as e:
                logger.warning("Chip %s @ GW%s failed to solve: %s", chip, gw, e)
                continue
            gain = res.get("total_xp", 0.0) - base_xp
            if gain > thresholds[chip] and gain > best["gain"]:
                best = {"chip": chip, "gw": gw, "gain": gain, "res": res}
    return best


def season_half(gw: int) -> int:
    return 1 if int(gw) <= FIRST_HALF_LAST_GW else 2


def available_chips(used: Dict[str, int], current_gw: int) -> List[str]:
    """
    Chips still in hand this half of the season.

    `used` maps chip code to the gameweek it was played in. A chip played in
    the other half has been replaced by that half's new set.
    """
    spent = {c for c, gw in (used or {}).items()
             if season_half(gw) == season_half(current_gw)}
    return [c for c in CHIP_BASE_THRESHOLD if c not in spent]
