"""
Behaviour when the odds feed is down.

football-data.co.uk went 503 mid-session. `fetch_odds` cached the failure as
`None` against the season, so every later gameweek in the process read it as
"this season has no odds", fell back to flat team ratings, and produced a
forecast with no fixture signal - which was then published as if sound.
"""
import urllib.error

import pandas as pd
import pytest

import market_odds as M


class _Boom(urllib.error.HTTPError):
    def __init__(self, code):
        self.code = code
        self.headers = {"retry-after": "177"}

    def __str__(self):
        return f"HTTP Error {self.code}"


@pytest.fixture(autouse=True)
def _clear_cache(tmp_path, monkeypatch):
    """
    Both caches, per test. The on-disk one must be isolated too: without this a
    test that fetches successfully writes into the repo's real `data/odds`, and
    every later test - and every later *run* - silently reads it. That is how
    the retry-count assertion below started failing.
    """
    M._GLOBAL_ODDS_CACHE.clear()
    monkeypatch.setattr(M, "ODDS_CACHE_DIR", str(tmp_path / "odds"))
    yield
    M._GLOBAL_ODDS_CACHE.clear()


def test_a_transient_failure_is_not_cached(monkeypatch):
    """The bug: one 503 must not become a permanent fact for the process."""
    monkeypatch.setattr(M.time, "sleep", lambda *_: None)
    monkeypatch.setattr(M.pd, "read_csv", lambda *a, **k: (_ for _ in ()).throw(_Boom(503)))

    m = M.MarketOddsModel()
    assert m.fetch_odds(season_str="2526") is False
    assert "2526" not in M._GLOBAL_ODDS_CACHE, "a failed fetch is not an answer"


def test_a_later_attempt_can_still_succeed(monkeypatch):
    """
    The consequence of not caching: once the feed recovers, the next gameweek in
    the same process gets real ratings instead of inheriting flat ones.
    """
    monkeypatch.setattr(M.time, "sleep", lambda *_: None)
    calls = {"n": 0}
    good = pd.DataFrame({
        "Date": ["12/08/2025"], "HomeTeam": ["Arsenal"], "AwayTeam": ["Chelsea"],
        "B365H": [2.0], "B365D": [3.4], "B365A": [3.6],
        "B365>2.5": [1.9], "B365<2.5": [1.9],
    })

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] <= M.ODDS_MAX_ATTEMPTS:      # exhaust the first call
            raise _Boom(503)
        return good.copy()

    monkeypatch.setattr(M.pd, "read_csv", flaky)
    assert M.MarketOddsModel().fetch_odds(season_str="2526") is False
    assert M.MarketOddsModel().fetch_odds(season_str="2526") is True


def test_a_transient_failure_is_retried(monkeypatch):
    monkeypatch.setattr(M.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def always_503(path, *a, **k):
        # Only the network attempts count. The cache fallback reads a local
        # path through the same `read_csv`, and folding that in would make this
        # assertion about plumbing rather than about retries.
        if str(path).startswith("http"):
            calls["n"] += 1
        raise _Boom(503)

    monkeypatch.setattr(M.pd, "read_csv", always_503)
    M.MarketOddsModel().fetch_odds(season_str="2526")
    assert calls["n"] == M.ODDS_MAX_ATTEMPTS


def test_a_404_is_not_retried(monkeypatch):
    """A season that is not published will not become published; retrying is
    pointless and would delay a scheduled job for nothing."""
    monkeypatch.setattr(M.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def missing(*a, **k):
        calls["n"] += 1
        raise _Boom(404)

    monkeypatch.setattr(M.pd, "read_csv", missing)
    M.MarketOddsModel().fetch_odds(season_str="9999")
    assert calls["n"] == 1


def test_backoff_is_bounded_and_jittered(monkeypatch):
    """
    The server's own `retry-after` is deliberate jitter, not an ETA - probed
    during a real outage it gave 341, 70 and 77 four seconds apart. We ignore it
    as an estimate but jitter our own waits, so repeated clients do not
    synchronise. The total must stay small enough for a daily cron.
    """
    waits = []
    monkeypatch.setattr(M.time, "sleep", lambda s: waits.append(s))
    monkeypatch.setattr(M.pd, "read_csv", lambda *a, **k: (_ for _ in ()).throw(_Boom(503)))

    M.MarketOddsModel().fetch_odds(season_str="2526")
    assert len(waits) == M.ODDS_MAX_ATTEMPTS - 1
    assert waits == sorted(waits), "backoff should grow"
    assert sum(waits) < 120, "a daily job must not stall"
    # Not the server's numbers.
    assert 177 not in waits and 341 not in waits


def test_a_file_without_prices_is_cached(monkeypatch):
    """This failure IS definitive - the file arrived and carries no odds
    columns, which no amount of retrying changes."""
    monkeypatch.setattr(M.time, "sleep", lambda *_: None)
    monkeypatch.setattr(M.pd, "read_csv", lambda *a, **k: pd.DataFrame(
        {"Date": ["12/08/2025"], "HomeTeam": ["A"], "AwayTeam": ["B"]}))
    assert M.MarketOddsModel().fetch_odds(season_str="2526") is False
    assert "2526" in M._GLOBAL_ODDS_CACHE


# ------------------------------------------- a degraded plan is not published


def test_a_degraded_fit_is_reported_to_the_caller():
    """
    `fit` has always returned a `degraded` flag and nothing read it, so a
    fixture-blind forecast was published as if sound. It must now reach the
    caller.
    """
    import inspect

    import xp_model

    src = inspect.getsource(xp_model.generate_merv_matrix)
    assert "status_out" in src
    assert "fit_status" in src


def test_the_exported_plan_carries_the_flag():
    """The publishing workflow reads this to decide whether to overwrite the
    last good plan, so the field has to be in the JSON."""
    import inspect

    import weekly_manager

    src = inspect.getsource(weekly_manager.main)
    assert '"degraded": degraded' in src
    assert "status_out=fit_status" in src


def test_the_workflow_will_not_publish_unconditionally():
    """
    The deploy and the alert must both be conditional. Which condition is
    `publish_gate`'s business and is tested there; this only pins that neither
    step publishes blind, since actions-gh-pages replaces the whole directory.
    """
    import pathlib

    wf = pathlib.Path(".github/workflows/weekly_optimizer.yml").read_text()
    for step in ("Send Discord notification", "Deploy JSON to GitHub Pages"):
        tail = wf[wf.index(step):]
        assert "if:" in tail[:tail.index("- name:") if "- name:" in tail else len(tail)], \
            f"{step} is unconditional"


# ------------------------------------------------ the last-good copy on disk


@pytest.fixture
def cache_dir(tmp_path):
    """Where `_clear_cache` has already pointed the cache."""
    return tmp_path / "odds"


def _priced():
    return pd.DataFrame({
        "Date": ["12/08/2025"], "HomeTeam": ["Arsenal"], "AwayTeam": ["Chelsea"],
        "B365H": [2.0], "B365D": [3.4], "B365A": [3.6],
        "B365>2.5": [1.9], "B365<2.5": [1.9],
    })


def test_a_successful_fetch_is_cached_to_disk(monkeypatch, cache_dir):
    monkeypatch.setattr(M.pd, "read_csv", lambda *a, **k: _priced())
    assert M.MarketOddsModel().fetch_odds(season_str="2526") is True
    assert (cache_dir / "2526.csv").exists()


def test_an_outage_falls_back_to_the_cached_copy(monkeypatch, cache_dir):
    """
    The point of the whole thing: a 503 should cost the most recent round of
    matches, not the entire fixture model.

    The cache is written here by a real `fetch_odds`, not by hand. Writing it by
    hand is what let the first version of this pass over a broken cache: the
    save happened after `Date` had been rewritten to datetimes, a datetime
    round-trips through CSV as ISO, and the `%d/%m/%Y` reload turned every row
    into NaT - so the file reloaded as zero matches and the fallback returned
    False. Only a genuine write-then-read catches that.
    """
    monkeypatch.setattr(M.time, "sleep", lambda *_: None)
    monkeypatch.setattr(M.pd, "read_csv", _reading_from_disk_only(cache_dir, live=True))
    assert M.MarketOddsModel().fetch_odds(season_str="2526") is True
    assert (cache_dir / "2526.csv").exists()

    # Now the feed goes down, in a fresh process-level cache.
    M._GLOBAL_ODDS_CACHE.clear()
    monkeypatch.setattr(M.pd, "read_csv", _reading_from_disk_only(cache_dir))

    m = M.MarketOddsModel()
    assert m.fetch_odds(season_str="2526") is True, "the cached copy was unusable"
    assert m.odds_df is not None and len(m.odds_df) == 1
    # And it is priced, not just non-empty - the per-match lambdas are what the
    # forecast actually consumes.
    assert m.odds_df["mu_h"].notna().all()


def _reading_from_disk_only(cache_dir, live=False):
    """read_csv that serves local paths for real, and either serves or 503s the
    network depending on `live`."""
    real = pd.read_csv

    def fake(path, *a, **k):
        if str(path).startswith("http"):
            if live:
                return _priced()
            raise _Boom(503)
        return real(path, *a, **k)

    return fake


def test_a_cached_copy_beyond_the_age_limit_is_ignored(monkeypatch, cache_dir):
    """Too many missing rounds and a fit on real results is the better bet."""
    import os

    M._save_cached_odds("2526", _priced())
    path = M._cache_path("2526")
    old = M.time.time() - (M.ODDS_CACHE_MAX_AGE_DAYS + 5) * 86400
    os.utime(path, (old, old))
    assert M._load_cached_odds("2526") == (None, None)


def test_the_cache_reports_its_age(cache_dir):
    import os

    M._save_cached_odds("2526", _priced())
    path = M._cache_path("2526")
    two_days = M.time.time() - 2 * 86400
    os.utime(path, (two_days, two_days))
    df, age = M._load_cached_odds("2526")
    assert df is not None
    assert 1.9 < age < 2.1


def test_caching_never_raises(monkeypatch):
    """A cache is a convenience; it must not be able to fail a run."""
    monkeypatch.setattr(M, "ODDS_CACHE_DIR", "/proc/nonexistent/nope")
    M._save_cached_odds("2526", _priced())           # must not raise
    assert M._load_cached_odds("2526") == (None, None)


def test_an_unreadable_cache_file_is_survivable(cache_dir):
    import os

    os.makedirs(cache_dir, exist_ok=True)
    with open(M._cache_path("2526"), "w") as f:
        f.write("\x00\x00 not a csv \x00")
    df, _ = M._load_cached_odds("2526")
    assert df is None or len(df) >= 0          # either way, no exception
