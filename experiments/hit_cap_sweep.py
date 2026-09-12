"""
Replay whole seasons at different transfer budgets, to price a hit empirically.

The optimiser charges a transfer beyond the free one at `optimizer.HIT_COST`
(4.0, the rules' price), discounted alongside the points it buys over a
five-gameweek horizon decayed at `HORIZON_DECAY`. The gain multiplier is
therefore ~3.78, so a hit pays for itself at a forecast edge of ~1.06 xP per
gameweek - a low bar, and well inside the forecast's own error. Whether that
makes the engine over-trade is a measurable question, not an arguable one.

The design that makes it measurable:

  - the forecast is computed ONCE per (season, gameweek, horizon) and shared
    across every arm, so the only difference between arms is how freely the
    optimiser may spend. Without this, each arm re-runs an identical forecast
    and the comparison also carries Monte-Carlo noise;
  - every arm therefore builds an identical GW1 squad (the squad build is not
    charged as transfers), which makes each gameweek a matched pair;
  - `--caps` sweeps the hard limit on hits per gameweek, `--costs` sweeps the
    price instead. The cap answers "should it be allowed to?"; the cost answers
    "what should it believe a hit is worth?";
  - chips are OFF unless `--chips` is passed. The replay plays them by default
    now, but this sweep must not inherit that: a wildcard or free hit suspends
    the transfer budget the sweep exists to vary, so chip weeks are exactly the
    weeks that carry no signal about the price of a hit. The published table was
    measured without them, and re-running with them on would not be comparable.

    uv run python experiments/hit_cap_sweep.py
    uv run python experiments/hit_cap_sweep.py --costs 4 6 8 --caps 3
    uv run python experiments/hit_cap_sweep.py --seasons 2024-25 --out /tmp/x.json

Then read it with `hit_cap_report.py`. The sweep is slow - the forecast is the
expensive part, roughly 200s a season, and each additional arm is another 38 ILP
solves - so it writes after every season and can be pointed at a subset.
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Full-xG seasons only. Earlier ones carry a caveat that biases this particular
# question: a weaker forecast means more noise-chasing, which makes hits look
# worse, which is the direction the hypothesis already points.
DEFAULT_SEASONS = ["2023-24", "2024-25", "2025-26"]
DEFAULT_CAPS = [0, 1, 2, 3]
# Held fixed while `--costs` sweeps the price. High enough that the price is
# what binds, rather than the cap.
DEFAULT_COST_SWEEP_CAP = 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--caps", nargs="+", type=int, default=None,
                    help="max hits per gameweek per arm")
    ap.add_argument("--costs", nargs="+", type=float, default=None,
                    help="sweep the hit price instead of the cap; "
                         "uses the first --caps value as a fixed limit")
    ap.add_argument("--chips", action="store_true",
                    help="Play chips during the replay. OFF by default, and "
                         "deliberately: the published hit-cap table was "
                         "measured without them, a chip week suspends the very "
                         "transfer budget this sweep varies, and searching "
                         "chip x gameweek multiplies the solves per arm by ~21.")
    ap.add_argument("--out", default="hit_cap_results.json")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR,
                        format="%(levelname)s %(message)s")

    import market_odds
    from simulator import fetch_data, run_season_simulation

    # The feed's own retry is right for a daily job and pointless here: an
    # outage just means the disk cache or the mirror answers instead.
    market_odds.ODDS_BASE_DELAY = 0.0

    sweep_costs = args.costs is not None
    arms = args.costs if sweep_costs else (args.caps or DEFAULT_CAPS)
    label = "cost" if sweep_costs else "cap"
    # The cap to hold fixed while the *price* is swept. It has to be permissive
    # or the sweep measures nothing: `--caps` used to default to [0, 1, 2, 3],
    # so a bare `--costs 4 6 8` pinned the cap at its first element - zero - and
    # every arm replayed a season in which no hit was affordable at any price.
    # The arms came out identical and the sweep looked like strong evidence that
    # the price does not matter.
    fixed_cap = (args.caps[0] if args.caps else DEFAULT_COST_SWEEP_CAP) \
        if sweep_costs else None
    if sweep_costs and fixed_cap == 0:
        ap.error("--costs with --caps 0 forbids every hit, so the price cannot "
                 "change anything; pass a permissive --caps (e.g. 3)")

    results, t0 = {}, time.time()
    for season in args.seasons:
        print(f"\n===== {season} =====", flush=True)
        data = fetch_data(season)
        xp_cache = {}                    # shared: identical forecasts per arm
        for arm in arms:
            t = time.time()
            r = run_season_simulation(
                season, xp_source="engine", data=data, verbose=args.verbose,
                max_hits_per_gw=fixed_cap if sweep_costs else int(arm),
                hit_cost=float(arm) if sweep_costs else 4.0,
                xp_cache=xp_cache, play_chips=args.chips)
            results[f"{season}|{arm}"] = {
                "season": season, label: arm,
                "total": r["total_points"], "hits": r["total_hits"],
                "transfers": r["total_transfers"], "history": r["history"],
            }
            print(f"  {label} {arm}: {r['total_points']:7.1f} pts   "
                  f"{r['total_hits']:3d} hits ({-4 * r['total_hits']:5d})   "
                  f"{r['total_transfers']:3d} transfers   "
                  f"[{time.time() - t:.0f}s]", flush=True)
        with open(args.out, "w") as f:            # after every season
            json.dump({"label": label, "runs": results}, f)

    print(f"\ntotal {time.time() - t0:.0f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
