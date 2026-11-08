"""
Score the archived pre-deadline forecasts against what actually happened.

    uv run python score_archive.py                    # every archived season
    uv run python score_archive.py --season 2026-27
    uv run python score_archive.py --fetch            # also fetch outcomes not yet archived

Three forecasts, each frozen before the deadline by `forecast_archive`:

    fepl      the engine's expected points
    ep_next   FPL's own projection - the only outside benchmark there is
    ppg       trailing points-per-game, the README's main baseline

Everything that makes the README's numbers what they are is reused rather than
re-implemented: the metrics and per-gameweek evaluation come from
`backtest.evaluate_gameweek`, the summary table from `backtest.summarise`, and
the intervals from `backtest.paired_ci`, paired by gameweek. The scored
population follows `backtest.build_eval_population` - the 250 most-owned
players plus anyone with 180+ minutes over the previous five gameweeks - with
one substitution: ownership is the pre-deadline `selected_by_percent` frozen in
the snapshot, where the backtest reads the previous gameweek's `selected`
count. The two rank players almost identically.

What this can say that the backtest cannot: nothing here was measured on the
season being scored, and `ep_next` is a real pre-deadline forecast, which the
leaky `xP` column in the historical archive is not.
"""
import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional

import backtest
import forecast_archive as fa

logger = logging.getLogger(__name__)

MODELS = ("fepl", "ep_next", "ppg")

# The pairs worth an interval: the engine against each benchmark, and FPL's own
# projection against the naive one, which says how hard a benchmark it is.
COMPARISONS = (("fepl", "ep_next"), ("fepl", "ppg"), ("ep_next", "ppg"))

CI_METRICS = ("rmse", "mae", "spearman", "captured_at_15", "ndcg_at_15",
              "precision_at_15", "captain_regret")

# Fewer than this and `backtest.paired_ci` declines to give an interval.
MIN_GAMEWEEKS_FOR_CI = 3


def eval_population(players: List[Dict[str, Any]], top_n: int = 250) -> set:
    """
    `backtest.build_eval_population`'s rule, on snapshot rows.

    Chosen only from what was known before the deadline and never from any
    forecast, so changing a model cannot change which players it is scored on.
    """
    owned = sorted((p for p in players if p.get("selected_by_percent") is not None),
                   key=lambda p: (-float(p["selected_by_percent"]), int(p["id"])))
    pop = {int(p["id"]) for p in owned[:top_n]}
    pop.update(int(p["id"]) for p in players
               if (p.get("mins_last5") or 0.0) >= 180)
    return pop


def score_gameweek(snapshot: Dict[str, Any], actual: Dict[str, Any],
                   top_n: int = 250) -> Dict[str, Any]:
    """One gameweek, in `backtest.evaluate_gameweek`'s shape, or {} if unscorable."""
    gw = int(snapshot["gameweek"])
    if int(actual.get("gameweek", gw)) != gw:
        raise ValueError(f"outcomes for GW{actual.get('gameweek')} paired with "
                         f"the GW{gw} forecast")

    players = fa.rows(snapshot)
    outcomes = {int(r["id"]): r for r in fa.rows(actual)}
    points = {pid: float(r["total_points"]) for pid, r in outcomes.items()}
    played = {pid: (r.get("minutes") or 0) > 0 for pid, r in outcomes.items()}

    population = sorted(eval_population(players, top_n) & set(points))
    by_id = {int(p["id"]): p for p in players}
    positions = {pid: by_id[pid].get("element_type") or 3 for pid in population}
    costs = {pid: by_id[pid].get("now_cost") or 50 for pid in population}

    preds: Dict[str, Dict[int, float]] = {
        "fepl": {pid: float(by_id[pid].get("xp") or 0.0) for pid in population},
    }
    # Only when FPL's "next" gameweek was this one - otherwise the column is
    # null throughout and scoring it would score a forecast of zero.
    if snapshot.get("ep_next_gw") == gw:
        preds["ep_next"] = {pid: float(by_id[pid].get("ep_next") or 0.0)
                            for pid in population}
    # Missing history scores as zero, as in `backtest.run_backtest`.
    preds["ppg"] = {pid: float(by_id[pid].get("ppg") or 0.0) for pid in population}

    p_play = {pid: float(by_id[pid].get("p_play") or 0.0) for pid in population}
    res = backtest.evaluate_gameweek(preds, points, positions, costs, population,
                                     p_play, played)
    if res:
        res["gw"] = gw
        res["season"] = snapshot.get("season")
        res["commit"] = snapshot.get("commit")
        res["forecast_at"] = snapshot.get("forecast_at")
        res["degraded"] = bool(snapshot.get("degraded"))
    return res


def load_scorable(archive_dir: str, seasons: List[str]) -> List[Dict[str, Any]]:
    """(snapshot, outcome) pairs for every archived gameweek that has both."""
    out = []
    for season in seasons:
        for gw in fa.archived_gameweeks(archive_dir, season):
            snap = fa.load(fa.snapshot_path(archive_dir, season, gw))
            actual = fa.load(fa.actual_path(archive_dir, season, gw))
            if snap is None:
                continue
            if actual is None:
                logger.info("%s GW%d: no outcomes archived yet; skipped.", season, gw)
                continue
            out.append({"season": season, "gw": gw, "snapshot": snap, "actual": actual})
    return out


def score(archive_dir: str, seasons: Optional[List[str]] = None,
          include_degraded: bool = False, top_n: int = 250) -> List[Dict[str, Any]]:
    """Per-gameweek results for every scorable archived gameweek, in order."""
    seasons = seasons or fa.archived_seasons(archive_dir)
    per_gw = []
    for item in load_scorable(archive_dir, seasons):
        if item["snapshot"].get("degraded") and not include_degraded:
            # A fixture-blind forecast says nothing about the engine; it is
            # archived so the record is complete, not so it can be scored.
            logger.warning("%s GW%d: the archived forecast is degraded; skipped "
                           "(--include-degraded to score it).",
                           item["season"], item["gw"])
            continue
        res = score_gameweek(item["snapshot"], item["actual"], top_n=top_n)
        if res:
            per_gw.append(res)
    return per_gw


def comparisons(per_gw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Paired-bootstrap intervals for each comparison, as `backtest.paired_ci`."""
    out = []
    for metric in CI_METRICS:
        for model, base in COMPARISONS:
            ci = backtest.paired_ci(per_gw, model, base, metric)
            if ci:
                out.append({"metric": metric, "model": model, "baseline": base, **ci})
    return out


def _print_report(per_gw: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    summary = backtest.summarise(per_gw, label, verbose=True)

    commits = sorted({str(g.get("commit"))[:7] for g in per_gw if g.get("commit")})
    if len(commits) > 1:
        print(f" Forecasts came from {len(commits)} commits ({', '.join(commits)}): "
              "the engine changed during the sample.")

    print()
    print(f" {'GW':<10}{'n':>5}" + "".join(f"{m + ' RMSE':>15}" for m in MODELS))
    for g in per_gw:
        cells = "".join(
            f"{g['models'][m]['rmse']:>15.3f}" if m in g["models"] else f"{'-':>15}"
            for m in MODELS)
        print(f" {str(g['season']) + ' ' + str(g['gw']):<10}{g['n']:>5}{cells}")

    rows = comparisons(per_gw)
    print()
    if not rows:
        print(f" Paired intervals need at least {MIN_GAMEWEEKS_FOR_CI} scored "
              f"gameweeks; {len(per_gw)} so far.")
    else:
        print(" Paired by gameweek, model minus baseline (95% bootstrap CI). "
              "Lower is better for rmse, mae and captain_regret.")
        print(f"   {'metric':<17}{'model':<9}{'baseline':<10}{'diff':>9}{'95% CI':>22}")
        for r in rows:
            span = f"[{r['lo']:+.3f}, {r['hi']:+.3f}]"
            mark = "significant" if r["significant"] else "noise"
            print(f"   {r['metric']:<17}{r['model']:<9}{r['baseline']:<10}"
                  f"{r['diff']:>+9.3f}{span:>22}  {mark}")
    print("=" * 86)
    return {"summary": summary, "comparisons": rows}


def fetch_missing_outcomes(archive_dir: str) -> List[int]:
    """Save outcomes for the live season's finished gameweeks. Needs network."""
    import fpl_api

    bootstrap = fpl_api.get_bootstrap_static()
    season = fa.season_label(bootstrap)
    done = fa.record_actuals(archive_dir, season, bootstrap, fpl_api.get_event_live)
    if done:
        print(f"Archived outcomes for {season} GW{', GW'.join(map(str, done))}.")
    return done


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive-dir", default=fa.DEFAULT_ARCHIVE_DIR)
    ap.add_argument("--season", action="append", default=None,
                    help="Season to score, e.g. 2026-27 (repeatable). "
                         "Default: every season in the archive.")
    ap.add_argument("--fetch", action="store_true",
                    help="First fetch outcomes for finished gameweeks of the "
                         "live season that the archive does not have yet.")
    ap.add_argument("--include-degraded", action="store_true",
                    help="Also score gameweeks whose archived forecast was "
                         "fixture-blind.")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)

    if args.fetch:
        fetch_missing_outcomes(args.archive_dir)

    seasons = args.season or fa.archived_seasons(args.archive_dir)
    per_gw = score(args.archive_dir, seasons, include_degraded=args.include_degraded)
    if not per_gw:
        print(f"Nothing to score in {args.archive_dir}: no archived gameweek has "
              "both a pre-deadline forecast and its outcomes yet.")
        return 0

    report = _print_report(per_gw, "+".join(seasons) + " archive")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"per_gameweek": per_gw, **report}, f, indent=2, default=float)
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
