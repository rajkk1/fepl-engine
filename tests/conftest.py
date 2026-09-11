"""Shared offline fixtures.

Everything here is synthetic so the unit suite runs without network access.
The one test that touches the live FPL API is marked `network` and skipped by
default (`-m network` to opt in).
"""
import os

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


@pytest.fixture(scope="session")
def _odds_cache_copy(tmp_path_factory):
    """A scratch copy of the committed `data/odds` floor, seeded once."""
    import shutil

    import market_odds

    dest = tmp_path_factory.mktemp("odds_cache")
    src = market_odds.ODDS_CACHE_DIR
    if os.path.isdir(src):
        shutil.copytree(src, dest, dirs_exist_ok=True)
    return str(dest)


@pytest.fixture(autouse=True)
def _odds_cache_is_scratch(monkeypatch, _odds_cache_copy):
    """
    The suite must never write to the committed `data/odds` floor.

    `test_smoke` reaches the live feed through `generate_xp_matrix` (see
    `_fast_odds_retry`), and a *successful* fetch writes the last good copy
    back to disk -- which is the repo's own committed floor. Running the tests
    therefore rewrote two tracked files, in football-data's full ~120-column
    form rather than the trimmed ten the repo commits: 24KB became 201KB, and
    it showed up as an unexplained 762-line diff next to whatever change was
    actually being made.

    The copy is seeded from the committed files so reads still see a realistic
    floor; only writes are diverted. `test_odds_resilience` points the same
    attribute at its own tmp_path and still overrides this, because an autouse
    fixture is applied before the test's own.
    """
    import market_odds

    monkeypatch.setattr(market_odds, "ODDS_CACHE_DIR", _odds_cache_copy)


@pytest.fixture(autouse=True)
def _no_live_odds_fetch(monkeypatch):
    """
    The unit suite must not reach football-data.co.uk.

    The README says the tests are offline by default and the daily workflow
    runs them with the comment "the daily job must not fail because an external
    API is briefly unavailable" -- but `test_smoke` reached the live feed
    through `generate_xp_matrix`, so an outage of the feed the engine is built
    to survive could fail the test step and stop the plan being published. That
    is the one failure mode the fallback chain exists to prevent.

    Blocked at `pd.read_csv` rather than at `_read_odds_csv`, for two reasons:
    the retry, the disk cache, the mirror and the rating fallbacks all still
    run for real, which is the behaviour worth exercising; and every test in
    `test_odds_resilience` patches this same attribute, so its own stub simply
    replaces this one.
    """
    import pandas as pd

    import market_odds

    real_read_csv = pd.read_csv

    def offline_read_csv(path_or_url, *args, **kwargs):
        if isinstance(path_or_url, str) and path_or_url.startswith(("http://", "https://")):
            raise OSError(f"network disabled in tests: {path_or_url}")
        return real_read_csv(path_or_url, *args, **kwargs)

    monkeypatch.setattr(market_odds.pd, "read_csv", offline_read_csv)


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
