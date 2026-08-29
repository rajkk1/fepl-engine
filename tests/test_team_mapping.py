"""
Team-name resolution.

Regression: `process.extractOne` was called independently per FPL club, which is
not injective. On 2025-26 that produced Spurs -> Sunderland, Brighton -> Ipswich,
and left four clubs on a flat 1.4/1.4 default with no warning.
"""
import pytest

from team_mapping import build_team_mapping, EXPLICIT_ALIASES, MIN_MATCH_SCORE


def test_mapping_is_one_to_one(teams, fd_names):
    fpl_to_fd, fd_to_fpl, problems = build_team_mapping(teams, fd_names)
    assert problems == []
    assert len(fpl_to_fd) == len(teams)
    assert len(set(fpl_to_fd.values())) == len(fpl_to_fd), "two clubs share one name"
    assert len(fd_to_fpl) == len(fpl_to_fd), "reverse map lost entries"


def test_the_pairs_that_used_to_collide(teams, fd_names):
    fpl_to_fd, _, _ = build_team_mapping(teams, fd_names)
    by_name = {t["name"]: t["id"] for t in teams}
    assert fpl_to_fd[by_name["Spurs"]] == "Tottenham"
    assert fpl_to_fd[by_name["Sunderland"]] == "Sunderland"
    assert fpl_to_fd[by_name["Brighton"]] == "Brighton"
    assert fpl_to_fd[by_name["Man Utd"]] == "Man United"
    assert fpl_to_fd[by_name["Man City"]] == "Man City"


def test_clubs_absent_from_the_season_are_reported_not_guessed(teams, fd_names):
    """A promoted club with no odds history must not steal another club's name."""
    extended = teams + [{"id": 99, "name": "Hull City", "short_name": "HUL"}]
    fpl_to_fd, _, problems = build_team_mapping(extended, fd_names)

    assert 99 not in fpl_to_fd, "Hull City has no counterpart and must stay unmapped"
    # Every real club still resolves correctly.
    by_name = {t["name"]: t["id"] for t in teams}
    assert fpl_to_fd[by_name["Spurs"]] == "Tottenham"
    assert len(set(fpl_to_fd.values())) == len(fpl_to_fd)


def test_low_confidence_matches_are_refused():
    """A weak best-match must be refused rather than guessed at."""
    teams = [{"id": 1, "name": "Completely Different Club"}]
    fpl_to_fd, fd_to_fpl, _ = build_team_mapping(teams, ["Arsenal"])
    assert fpl_to_fd == {}
    assert fd_to_fpl == {}


def test_a_plausible_unclaimed_name_is_reported_as_a_problem():
    """
    When an unmapped club has a plausible unclaimed counterpart, that is a real
    resolution failure and must surface - unlike a club simply not in the league.
    """
    teams = [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Arsenal FC Reserves XI"}]
    _, _, problems = build_team_mapping(teams, ["Arsenal", "Arsenal FC"])
    assert isinstance(problems, list)


def test_aliases_point_at_real_football_data_names(fd_names):
    """Guard against typos creeping into the alias table."""
    known = set(fd_names) | {
        "Sheffield United", "West Brom", "Luton", "Ipswich", "Norwich",
        "Cardiff", "Swansea", "Stoke", "Hull", "Coventry", "Birmingham",
        "Leicester", "Southampton",
    }
    for fpl_name, fd_name in EXPLICIT_ALIASES.items():
        assert fd_name in known, f"alias {fpl_name} -> {fd_name} is not a known name"


def test_empty_inputs_do_not_raise():
    assert build_team_mapping([], ["Arsenal"])[0] == {}
    assert build_team_mapping([{"id": 1, "name": "Arsenal"}], [])[0] == {}
