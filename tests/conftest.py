"""Shared offline fixtures.

Everything here is synthetic so the unit suite runs without network access.
The one test that touches the live FPL API is marked `network` and skipped by
default (`-m network` to opt in).
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "network: hits a live external API")


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m"):
        return
    skip = pytest.mark.skip(reason="needs network; run with -m network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def teams():
    names = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
        "Leeds", "Liverpool", "Man City", "Man Utd", "Newcastle",
        "Nott'm Forest", "Spurs", "Sunderland", "West Ham", "Wolves",
    ]
    return [{"id": i + 1, "name": n, "short_name": n[:3].upper()}
            for i, n in enumerate(names)]


@pytest.fixture
def fd_names():
    """Football-Data spellings for the same 20 clubs."""
    return [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
        "Leeds", "Liverpool", "Man City", "Man United", "Newcastle",
        "Nott'm Forest", "Tottenham", "Sunderland", "West Ham", "Wolves",
    ]


@pytest.fixture
def player():
    return {
        "id": 1, "web_name": "Test", "element_type": 3, "team": 1,
        "now_cost": 75, "status": "a", "chance_of_playing_next_round": None,
        "news": "", "penalties_order": None, "selected_by_percent": "10.0",
    }


@pytest.fixture
def history():
    """Nine gameweeks of a nailed-on starter."""
    return [
        {
            "round": gw, "minutes": 90, "total_points": 5, "value": 75,
            "was_home": gw % 2 == 0, "opponent_team": (gw % 19) + 2,
            "starts": 1, "expected_goals": 0.30, "expected_assists": 0.20,
            "saves": 0, "yellow_cards": 0,
            "clearances_blocks_interceptions": 2, "tackles": 2, "recoveries": 6,
        }
        for gw in range(1, 10)
    ]


@pytest.fixture
def fixture():
    return {"event": 10, "team_h": 1, "team_a": 2, "finished": False,
            "kickoff_time": "2025-10-25T14:00:00Z"}


@pytest.fixture
def gw_frame():
    """A minimal merged-gameweek frame for harness tests."""
    import pandas as pd
    rows = []
    for gw in range(1, 8):
        for pid in range(1, 21):
            rows.append({"GW": gw, "element": pid,
                         "total_points": (pid % 7) + gw % 3,
                         "minutes": 90 if pid % 4 else 0,
                         "selected": 1000 * (21 - pid),
                         "value": 50 + pid, "xP": 2.0})
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _no_odds_backoff(monkeypatch):
    """
    Never sleep in the test suite.

    `test_smoke` reaches the real odds feed through `generate_xp_matrix`, which
    is a hidden network dependency the suite has always had - invisible while
    the feed was up. When football-data.co.uk went 503 the new retry turned each
    call into four attempts with exponential backoff and the suite went from 18
    seconds to 12 minutes. The retry is right for a daily job and wrong for a
    test, so the delay is zeroed rather than the behaviour changed: the same
    code paths run, just without waiting.
    """
    import market_odds

    monkeypatch.setattr(market_odds, "ODDS_BASE_DELAY", 0.0)


@pytest.fixture(autouse=True)
def _no_mirror_download(monkeypatch):
    """
    Never fetch the odds mirror in the suite.

    It is a single 40MB+ file, and the same hidden path through
    `generate_xp_matrix` that reaches the primary feed also reaches the mirror
    behind it. Without this the unit suite downloads it whenever
    football-data.co.uk is unreachable.

    Done by seeding the process-level frame rather than by patching
    `_fetch_mirror_odds`: the real function then still runs, so tests exercise
    the actual translation and season-slicing code, and a test that wants
    mirror content only has to put a frame here.
    """
    import pandas as pd

    import market_odds

    monkeypatch.setitem(
        market_odds._MIRROR_FRAME, "df",
        pd.DataFrame(columns=["Division", "MatchDate", "HomeTeam", "AwayTeam"]))
