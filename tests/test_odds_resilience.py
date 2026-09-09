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
        # Only attempts on the PRIMARY count. The cache fallback reads a local
        # path and the mirror reads a different host, both through this same
        # `read_csv`; folding either in would make this assertion about
        # plumbing rather than about retries.
        if "football-data.co.uk" in str(path):
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

    def missing(path, *a, **k):
        if "football-data.co.uk" in str(path):
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


def _priced(days_ago=3):
    """One priced match, dated recently by default.

    The date matters now: cache staleness is measured from the newest match in
    the file, so a hardcoded past date would make every cache fixture stale and
    the fallback tests would pass for the wrong reason.
    """
    when = (pd.Timestamp.now() - pd.Timedelta(days=days_ago)).strftime("%d/%m/%Y")
    return pd.DataFrame({
        "Date": [when], "HomeTeam": ["Arsenal"], "AwayTeam": ["Chelsea"],
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
    M._save_cached_odds("2526", _priced(days_ago=M.ODDS_CACHE_MAX_AGE_DAYS + 5))
    assert M._load_cached_odds("2526") == (None, None)


def test_the_cache_reports_its_age(cache_dir):
    """The caller logs this to say how much the ratings are missing, so it has
    to be the age of the market information, not of the file."""
    M._save_cached_odds("2526", _priced(days_ago=2))
    df, age = M._load_cached_odds("2526")
    assert df is not None
    assert 1.9 < age < 2.6


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


# ------------------------------------------------------------------ the mirror


def _mirror_rows():
    """Two E0 matches in the mirror's own schema, one per season."""
    return pd.DataFrame({
        "Division": ["E0", "E0", "D1"],
        "MatchDate": ["2025-08-16", "2024-08-17", "2025-08-16"],
        "HomeTeam": ["Arsenal", "Chelsea", "Bayern Munich"],
        "AwayTeam": ["Chelsea", "Arsenal", "Leverkusen"],
        "FTHome": [1, 2, 3], "FTAway": [0, 2, 1],
        "OddHome": [2.0, 2.5, 1.5], "OddDraw": [3.4, 3.3, 4.0],
        "OddAway": [3.6, 2.8, 6.0],
        "Over25": [1.9, 1.8, 1.6], "Under25": [1.9, 2.0, 2.3],
    })


@pytest.fixture
def mirror(monkeypatch):
    """Mirror content served from memory - the real code path, no download."""
    monkeypatch.setitem(M._MIRROR_FRAME, "df", _mirror_rows())


def test_the_mirror_translates_into_football_datas_own_schema(mirror):
    """Downstream code - bookmaker-group selection, the cache format, the team
    mapping - must not be able to tell where the numbers came from."""
    out = M._fetch_mirror_odds("2526")
    assert list(out.columns[:3]) == ["Date", "HomeTeam", "AwayTeam"]
    for col in ("B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"):
        assert col in out.columns
    assert out["Date"].iloc[0] == "16/08/2025", "dates must be dd/mm/yyyy"
    assert len(out) == 1, "only E0, only the requested season"


def test_the_mirror_rescues_an_outage_with_no_cache_at_all(monkeypatch, mirror,
                                                           cache_dir):
    """
    The case the disk cache cannot cover, and the reason the mirror exists:
    the feed is down and nothing was ever cached, so there is no last-good copy
    to fall back to. This is what actually happened in September 2026.
    """
    monkeypatch.setattr(M.pd, "read_csv",
                        lambda *a, **k: (_ for _ in ()).throw(_Boom(503)))
    m = M.MarketOddsModel()
    assert m.fetch_odds(season_str="2526") is True
    assert m.odds_df is not None and m.odds_df["mu_h"].notna().all()


def test_the_mirror_is_fetched_once_per_process(monkeypatch):
    """It is one 40MB+ file and a run asks for two seasons (current, plus the
    prior one for priors). Downloading it twice is not acceptable."""
    M._MIRROR_FRAME.clear()
    calls = {"n": 0}

    def counted(*a, **k):
        calls["n"] += 1
        return _mirror_rows()

    monkeypatch.setattr(M.pd, "read_csv", counted)
    M._fetch_mirror_odds("2526")
    M._fetch_mirror_odds("2425")
    M._MIRROR_FRAME.clear()
    assert calls["n"] == 1


def test_a_season_boundary_gone_wrong_is_refused(monkeypatch, mirror):
    """Two seasons in one frame would fit team ratings across a summer of
    transfers. Better to fall through than to use that."""
    big = pd.concat([_mirror_rows().iloc[[0]]] * 500, ignore_index=True)
    M._MIRROR_FRAME["df"] = big
    assert M._fetch_mirror_odds("2526") is None


def test_the_season_boundary_is_august_not_july():
    """
    The covid-delayed 2019-20 season ran to 26 July 2020. A July cutoff files
    those matches under 2020-21 and yields a 446-match season - which is what
    the first version of this did.
    """
    d = pd.to_datetime(pd.Series(["2020-07-26", "2020-09-12", "2025-08-16"]))
    assert list(M._season_start_year(d)) == [2019, 2020, 2025]


def test_a_missing_season_in_the_mirror_is_not_fatal(mirror):
    assert M._fetch_mirror_odds("9999") is None


# ------------------------------------------- staleness is a fact about the data


def test_cache_age_comes_from_the_data_not_the_file(cache_dir):
    """
    The whole reason this is not `os.path.getmtime`: `data/odds` is committed,
    and a fresh git checkout stamps every file with the checkout time. An mtime
    check would call a two-year-old odds file perfectly fresh. It also
    mis-reads a cache written from the mirror, whose data can be weeks behind
    the moment it was written.
    """
    import os

    old_match = pd.Timestamp.now() - pd.Timedelta(days=200)
    df = _priced()
    df["Date"] = [old_match.strftime("%d/%m/%Y")]
    M._save_cached_odds("2526", df)
    os.utime(M._cache_path("2526"), None)          # touch: mtime is now

    age = M._data_age_days(pd.read_csv(M._cache_path("2526")))
    assert 199 < age < 201, "age must track the match date, not the mtime"


def test_a_complete_season_never_goes_stale(cache_dir):
    """A finished season is complete market information. It is years old by
    construction and must still be usable for prior-season priors."""
    rows = pd.concat([_priced()] * M.MATCHES_IN_A_FULL_SEASON, ignore_index=True)
    rows["Date"] = (pd.Timestamp.now() - pd.Timedelta(days=800)).strftime("%d/%m/%Y")
    M._save_cached_odds("2223", rows)
    df, age = M._load_cached_odds("2223")
    assert df is not None and age > M.ODDS_CACHE_MAX_AGE_DAYS


def test_a_stale_part_played_season_is_still_rejected(cache_dir):
    """The case the age limit is actually for: a part-season file that stopped
    updating months ago is missing rounds that matter."""
    rows = pd.concat([_priced()] * 20, ignore_index=True)
    rows["Date"] = (pd.Timestamp.now() - pd.Timedelta(days=200)).strftime("%d/%m/%Y")
    M._save_cached_odds("2627", rows)
    assert M._load_cached_odds("2627") == (None, None)


# ------------------------------------------------------ the committed cold floor


def test_the_committed_floor_is_present_and_usable():
    """
    `data/odds` is in the repository so a cold CI run has something to fall
    back to when the feed is down. If these files stop being committed, or stop
    parsing, the failure is silent - the run just goes fixture-blind.
    """
    import glob
    import pathlib

    root = pathlib.Path(M.__file__).parent
    files = sorted(glob.glob(str(root / "data" / "odds" / "*.csv")))
    assert files, "no committed odds floor"
    for path in files:
        df = pd.read_csv(path)
        assert {"Date", "HomeTeam", "AwayTeam", "B365H"} <= set(df.columns), path
        assert len(df) > 0, path
        d = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        assert d.notna().all(), f"{path} has dates the loader cannot parse"
