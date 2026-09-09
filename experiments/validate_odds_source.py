"""
Check that an odds file is what it claims to be, before trusting a forecast to it.

Written when the mirror was added. A second source for the market prices is a
second chance to get home and away the wrong way round, or to line the columns
up one bookmaker off, and neither of those announces itself - the forecast just
quietly prices every fixture backwards. Three checks, each of which fails loudly
on a different plausible corruption:

  1. Calibration of the de-vigged home-win price against actual home wins. A
     home/away swap or a column misalignment destroys this.
  2. Implied total goals from the over/under price against actual total goals.
     Catches an over/under swap and any scaling error.
  3. Favourite-wins rate by price bucket, which must be monotone. Catches a row
     shuffle - odds attached to the wrong match.

    uv run python experiments/validate_odds_source.py
    uv run python experiments/validate_odds_source.py --dir some/other/odds

Deviations of ~0.05 at n≈70-100 per bucket are sampling noise, not a fault.
"""
import argparse
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOL = 0.10


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=None,
                    help="directory of football-data-format CSVs "
                         "(default: the engine's odds cache)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)

    import market_odds
    from market_odds import MarketOddsModel, _devig

    directory = args.dir or market_odds.ODDS_CACHE_DIR
    model = MarketOddsModel()
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not files:
        print(f"no CSVs in {directory}", file=sys.stderr)
        return 1

    worst = 0.0
    for path in files:
        tag = os.path.basename(path)[:-4]
        df = pd.read_csv(path)
        need = ["B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"]
        if "FTHG" not in df.columns or not set(need) <= set(df.columns):
            print(f"\n{tag}: no results or no prices; nothing to check against")
            continue
        df = df.dropna(subset=need + ["FTHG", "FTAG"])
        if len(df) < 50:
            print(f"\n{tag}: only {len(df)} priced matches, too few to check")
            continue

        p = np.array([_devig(1 / h, 1 / d, 1 / a)
                      for h, d, a in zip(df.B365H, df.B365D, df.B365A)])
        p_home, p_away = p[:, 0], p[:, 2]
        home_win = (df.FTHG > df.FTAG).to_numpy()
        away_win = (df.FTHG < df.FTAG).to_numpy()

        p_over = np.array([_devig(1 / o, 1 / u)[0]
                           for o, u in zip(df["B365>2.5"], df["B365<2.5"])])
        implied = np.array([model.implied_total_goals(x) for x in p_over])
        total = (df.FTHG + df.FTAG).to_numpy()

        print(f"\n{tag}  ({len(df)} priced matches)")
        print(f"  P(home win)  market {p_home.mean():.3f}  actual {home_win.mean():.3f}")
        print(f"  P(away win)  market {p_away.mean():.3f}  actual {away_win.mean():.3f}")
        print(f"  total goals  market {implied.mean():.2f}  actual {total.mean():.2f}"
              f"   |  P(over 2.5) market {p_over.mean():.3f}  "
              f"actual {(total > 2.5).mean():.3f}")

        bins = pd.cut(p_home, [0, .25, .4, .55, .7, 1.0])
        rates = (pd.DataFrame({"p": p_home, "won": home_win})
                 .groupby(bins, observed=True)
                 .agg(n=("won", "size"), market=("p", "mean"),
                      actual=("won", "mean")))
        for b, r in rates.iterrows():
            gap = abs(r.market - r.actual)
            worst = max(worst, gap)
            print(f"    {str(b):14s} n={int(r.n):3d}  market {r.market:.3f}  "
                  f"actual {r.actual:.3f}"
                  + ("" if gap < TOL else f"   <-- off by {gap:.3f}"))
        if not rates.actual.is_monotonic_increasing:
            print("    NOT MONOTONE by price bucket - suspect a row shuffle")

    print(f"\nworst bucket deviation across all files: {worst:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
