"""
Refresh the committed odds floor in `data/odds/`.

Normally nothing needs to run this: a successful fetch writes the cache on its
own, and CI carries it between runs with actions/cache. This exists for the two
cases where that is not enough - seeding a season that has never been fetched,
and refreshing the copy that is committed to the repository as a cold-start
floor.

    uv run python refresh_odds_cache.py                  # current + prior season
    uv run python refresh_odds_cache.py 2223 2324 2425   # named seasons

Prefers football-data.co.uk and falls back to the mirror, which is the same
order the engine uses. Writes nothing for a season it could not fetch, so a bad
run cannot replace a good file with an empty one.
"""
import argparse
import logging
import sys

import market_odds as M

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def seasons_now():
    """This season and the previous one - what a live run actually reads."""
    cur = M.MarketOddsModel._season_str_for_now()
    y = int(f"20{cur[:2]}")
    return [f"{str(y - 1)[2:]}{str(y)[2:]}", cur]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("seasons", nargs="*", metavar="SEASON",
                    help="football-data season tags, e.g. 2425. "
                         "Default: current and prior.")
    args = ap.parse_args(argv)
    wanted = args.seasons or seasons_now()

    failed = []
    for season in wanted:
        df = M._read_odds_csv(
            f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv", season)
        source = "football-data.co.uk"
        if df is None:
            df = M._fetch_mirror_odds(season)
            source = "mirror"
        if df is None or df.empty:
            logger.error("%s: no odds from either source; leaving any existing "
                         "file alone.", season)
            failed.append(season)
            continue
        M._save_cached_odds(season, df)
        age = M._data_age_days(df)
        print(f"  {season}: {len(df):3d} matches from {source}, newest "
              f"{age:.0f} days old -> {M._cache_path(season)}")

    if failed:
        print(f"\ncould not refresh: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
