"""
Robust FPL <-> Football-Data team name resolution.

The previous approach called `process.extractOne` independently per FPL team,
which is not injective: several FPL clubs could win the same Football-Data name
and the last writer clobbered the earlier ones. On 2025-26 that mapped Spurs onto
Sunderland's ratings and left Brighton with no rating at all.

This module resolves the mapping as a single global assignment problem so each
Football-Data name is claimed by at most one FPL club, seeded by an explicit
alias table for the cases fuzzy matching reliably gets wrong.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from thefuzz import fuzz

logger = logging.getLogger(__name__)

# Canonical aliases. Keys are normalised FPL names, values are the exact
# Football-Data (football-data.co.uk E0.csv) name. These are the pairs where
# token-based fuzzy scoring picks the wrong club or scores too low to trust.
EXPLICIT_ALIASES = {
    "spurs": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "man utd": "Man United",
    "man united": "Man United",
    "manchester united": "Man United",
    "man city": "Man City",
    "manchester city": "Man City",
    "nott'm forest": "Nott'm Forest",
    "nottingham forest": "Nott'm Forest",
    "newcastle united": "Newcastle",
    "wolverhampton wanderers": "Wolves",
    "brighton and hove albion": "Brighton",
    "brighton & hove albion": "Brighton",
    "sheffield united": "Sheffield United",
    "sheffield utd": "Sheffield United",
    "west bromwich albion": "West Brom",
    "leeds united": "Leeds",
    "luton town": "Luton",
    "ipswich town": "Ipswich",
    "leicester city": "Leicester",
    "norwich city": "Norwich",
    "cardiff city": "Cardiff",
    "swansea city": "Swansea",
    "stoke city": "Stoke",
    "hull city": "Hull",
    "coventry city": "Coventry",
    "birmingham city": "Birmingham",
    "west ham united": "West Ham",
}

# Below this fuzzy score we refuse to guess rather than inventing a mapping.
MIN_MATCH_SCORE = 60


def _norm(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _score(fpl_name: str, fd_name: str) -> int:
    """
    Blended fuzzy score. token_set handles 'Man Utd' vs 'Man United' (82).
    partial_ratio is deliberately excluded: it scores unrelated clubs as high as
    57 by sliding a substring, which is close enough to the accept threshold to
    manufacture false matches.
    """
    a, b = _norm(fpl_name), _norm(fd_name)
    if a == b:
        return 100
    return max(fuzz.token_set_ratio(a, b), fuzz.token_sort_ratio(a, b))


def build_team_mapping(
    fpl_teams: List[Dict[str, Any]],
    fd_names: List[str],
    strict: bool = False,
) -> Tuple[Dict[int, str], Dict[str, int], List[str]]:
    """
    Resolve FPL team ids to Football-Data names, one-to-one.

    Returns (fpl_to_fd, fd_to_fpl, problems). `problems` is a list of
    human-readable strings describing clubs that could not be mapped; it is
    empty on a clean resolution. With strict=True an unresolved club raises.
    """
    fd_names = list(dict.fromkeys(fd_names))  # de-dupe, keep order
    if not fpl_teams or not fd_names:
        return {}, {}, ["no teams or no football-data names supplied"]

    fpl_to_fd: Dict[int, str] = {}
    fd_to_fpl: Dict[str, int] = {}
    fd_available = set(fd_names)

    # 1. Explicit aliases win outright.
    unresolved = []
    for t in fpl_teams:
        alias = EXPLICIT_ALIASES.get(_norm(t["name"]))
        if alias and alias in fd_available:
            fpl_to_fd[t["id"]] = alias
            fd_to_fpl[alias] = t["id"]
            fd_available.discard(alias)
        else:
            unresolved.append(t)

    # 2. Everything else via a global one-to-one assignment. Using the Hungarian
    #    algorithm here (rather than per-team argmax) is what makes the result
    #    injective: no two clubs can be handed the same Football-Data name.
    problems: List[str] = []
    if unresolved:
        remaining = [n for n in fd_names if n in fd_available]
        if not remaining:
            problems += [f"{t['name']}: no unclaimed football-data name left" for t in unresolved]
        else:
            cost = np.array(
                [[100 - _score(t["name"], fd) for fd in remaining] for t in unresolved],
                dtype=float,
            )
            rows, cols = linear_sum_assignment(cost)
            assigned = set()
            for r, c in zip(rows, cols):
                team, fd = unresolved[r], remaining[c]
                score = 100 - cost[r, c]
                if score >= MIN_MATCH_SCORE:
                    fpl_to_fd[team["id"]] = fd
                    fd_to_fpl[fd] = team["id"]
                    assigned.add(r)
                else:
                    problems.append(
                        f"{team['name']}: best available match '{fd}' scored "
                        f"{score:.0f} < {MIN_MATCH_SCORE}"
                    )
            for i, team in enumerate(unresolved):
                if i not in assigned and not any(team["name"] in p for p in problems):
                    problems.append(f"{team['name']}: not assigned")

    # 3. Validate injectivity. This should be impossible by construction; the
    #    assertion exists because the previous bug was exactly this and silent.
    if len(set(fpl_to_fd.values())) != len(fpl_to_fd):
        raise RuntimeError(
            "team mapping is not one-to-one - this is a bug in build_team_mapping"
        )

    # 4. An unmapped FPL club is only a *problem* if there was an unclaimed
    #    Football-Data name it should have taken. When every FD club has been
    #    claimed, the leftovers are simply clubs not in that league season
    #    (promoted sides, or a bootstrap from a later season) - expected, not a fault.
    #    A leftover FD name only counts against us if it plausibly belongs to one
    #    of the unmapped clubs; otherwise it is a club absent from this bootstrap.
    unmapped_names = [p.split(":")[0] for p in problems]
    leftover_fd = [
        n for n in fd_names
        if n not in fd_to_fpl
        and any(_score(u, n) >= MIN_MATCH_SCORE for u in unmapped_names)
    ]
    if problems and not leftover_fd:
        logger.info(
            "%d FPL club(s) absent from this season's odds file (expected for "
            "promoted/relegated sides): %s",
            len(problems),
            ", ".join(p.split(":")[0] for p in problems),
        )
        problems = []
    elif problems:
        msg = (
            "Team name resolution incomplete:\n  "
            + "\n  ".join(problems)
            + "\n  unclaimed football-data names: "
            + ", ".join(leftover_fd)
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)

    return fpl_to_fd, fd_to_fpl, problems


def describe_mapping(fpl_teams: List[Dict[str, Any]], fpl_to_fd: Dict[int, str]) -> str:
    """Render the resolved mapping for logs and diagnostics."""
    by_id = {t["id"]: t["name"] for t in fpl_teams}
    lines = [f"  {by_id.get(i, i):<24} -> {fd}" for i, fd in sorted(fpl_to_fd.items())]
    missing = [n for i, n in sorted(by_id.items()) if i not in fpl_to_fd]
    if missing:
        lines.append("  UNMAPPED: " + ", ".join(missing))
    return "\n".join(lines)
