"""
Parameter search against the backtest.

The Gamma-Poisson half-life and prior strength were previously one hardcoded
setting applied to every statistic, which cannot be right: minutes-like counts
stabilise within a few matches while expected goals take most of a season. This
script grid-searches them per statistic using the same walk-forward harness the
CI gate uses, so a tuning claim is backed by a measurement.

    uv run python tune.py --stat xg --season 2025-26 --from-gw 8 --to-gw 16
    uv run python tune.py --all --quick

Results are written as JSON so a chosen setting can be pasted into
`GammaPoissonFilter.STAT_SETTINGS`.
"""
import argparse
import itertools
import json
import logging
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HALF_LIVES = [3.0, 5.0, 8.0, 12.0, 20.0]
PRIOR_WEIGHTS = [0.5, 1.0, 2.0, 4.0]
TUNABLE_STATS = ["xg", "xa", "cbit", "cbirt", "saves", "yc"]

# Lower is better for these; higher is better for the rest.
LOWER_IS_BETTER = {"mae", "rmse", "captain_regret"}


def evaluate(stat_settings, season, gws, data, metric="mae") -> Tuple[float, Dict[str, Any]]:
    """Run the harness with one candidate setting and return its score."""
    import xp_model
    from backtest import run_backtest

    original = dict(xp_model.GammaPoissonFilter.STAT_SETTINGS)
    xp_model.GammaPoissonFilter.STAT_SETTINGS = {**original, **stat_settings}
    try:
        summary = run_backtest(
            season_str=season, test_gws=gws,
            df_gw=data[0], df_players=data[1], df_teams=data[2], df_fixtures=data[3],
            include_leaky_baseline=False, verbose=False,
        )
    finally:
        xp_model.GammaPoissonFilter.STAT_SETTINGS = original

    fepl = summary.get("models", {}).get("fepl")
    if not fepl:
        return (float("inf"), {})
    score = fepl[metric]
    return (score if metric in LOWER_IS_BETTER else -score), fepl


def tune_stat(stat, season, gws, data, metric="mae", quick=False):
    half_lives = HALF_LIVES[::2] if quick else HALF_LIVES
    priors = PRIOR_WEIGHTS[::2] if quick else PRIOR_WEIGHTS

    results = []
    best = (float("inf"), None, None)
    for hl, pw in itertools.product(half_lives, priors):
        score, detail = evaluate({stat: (hl, pw)}, season, gws, data, metric)
        results.append({"stat": stat, "half_life": hl, "prior_weight": pw,
                        "score": score, "metrics": detail})
        marker = ""
        if score < best[0]:
            best = (score, hl, pw)
            marker = "  <- best so far"
        shown = score if metric in LOWER_IS_BETTER else -score
        print(f"  {stat:<6} half_life={hl:<5} prior_weight={pw:<4} {metric}={shown:.4f}{marker}",
              flush=True)

    print(f"\n  BEST for {stat}: half_life={best[1]}, prior_weight={best[2]} "
          f"({metric}={best[0] if metric in LOWER_IS_BETTER else -best[0]:.4f})\n")
    return {"stat": stat, "best": {"half_life": best[1], "prior_weight": best[2]},
            "grid": results}


def main():
    ap = argparse.ArgumentParser(description="Grid-search Gamma-Poisson shrinkage")
    ap.add_argument("--stat", choices=TUNABLE_STATS)
    ap.add_argument("--all", action="store_true", help="tune every statistic in turn")
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--from-gw", type=int, default=8)
    ap.add_argument("--to-gw", type=int, default=16)
    ap.add_argument("--metric", default="mae",
                    choices=["mae", "rmse", "spearman", "captain_regret", "precision_at_15"])
    ap.add_argument("--quick", action="store_true", help="coarser grid")
    ap.add_argument("--json-out", default="tuning_results.json")
    args = ap.parse_args()

    if not args.stat and not args.all:
        ap.error("pass --stat NAME or --all")

    from backtest import fetch_data

    data = fetch_data(args.season)
    gws = list(range(args.from_gw, args.to_gw + 1))
    stats = TUNABLE_STATS if args.all else [args.stat]

    print(f"Tuning on {args.season} GW{gws[0]}-{gws[-1]}, optimising {args.metric}\n")
    out = []
    for stat in stats:
        out.append(tune_stat(stat, args.season, gws, data, args.metric, args.quick))

    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2, default=float)

    print("Suggested STAT_SETTINGS entries:")
    for r in out:
        print(f'    "{r["stat"]}": ({r["best"]["half_life"]}, {r["best"]["prior_weight"]}),')
    print(f"\nFull grid written to {args.json_out}")


if __name__ == "__main__":
    main()
