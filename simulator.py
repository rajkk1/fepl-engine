"""
Full-season replay of the engine's own decisions.

Everything in `backtest.py` measures the *forecast*. This measures the thing the
forecast is for: points on the board after following the engine's transfers,
captaincy and bench order for a whole season.

The two can come apart. A forecast can gain on mean absolute error while the
squad it produces scores no more, because the squad only cares about a few
hundred ranking decisions and MAE is dominated by the ~300 players it never
considers. So this reports the engine against forecasts it must beat to be worth
running at all, driving the *same* optimiser with each:

    engine   the expected-points model
    ppg      trailing points-per-game
    roll3    mean of the last three gameweeks

Holding the optimiser fixed is the point. Any difference is attributable to the
forecast, which is what makes the comparison worth anything.

Corrections against the previous version:
  * it returned nothing, so it could not be tested or compared against anything
  * a bare season total is uninterpretable without a comparator
  * autosubs ignored formation validity, substituting any bench player for any
    starter - so a keeper could come on for a forward, and a team could finish
    with no defenders
  * the vice-captain got the armband whenever the captain blanked, including
    when the captain played and simply scored nothing
"""
import argparse
import itertools
import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from backtest import build_baselines, fetch_data, build_mock_api
from optimizer import solve_fpl_optimization
import chip_policy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

POS_GKP, POS_DEF, POS_MID, POS_FWD = 1, 2, 3, 4
# Exactly one keeper, and 3-5 / 2-5 / 1-3 outfield. An autosub that breaks this
# is not a substitution FPL would have made.
FORMATION_LIMITS = {POS_GKP: (1, 1), POS_DEF: (3, 5), POS_MID: (2, 5), POS_FWD: (1, 3)}

XP_SOURCES = ("engine", "ppg", "roll3")


def _formation_ok(positions: List[int]) -> bool:
    if len(positions) != 11:
        return False
    for pos, (lo, hi) in FORMATION_LIMITS.items():
        if not lo <= positions.count(pos) <= hi:
            return False
    return True


def apply_autosubs(starters: List[int], bench: List[int], played: Dict[int, bool],
                   position: Dict[int, int]) -> List[int]:
    """
    The eleven that actually scored, after FPL's automatic substitutions.

    Bench order matters and so does formation: a bench player only comes on if
    the resulting eleven is still legal. The keeper is a separate exchange -
    only the reserve keeper can replace the starting keeper, and never an
    outfielder.
    """
    final = [p for p in starters if played.get(p, False)]
    missing = [p for p in starters if not played.get(p, False)]
    if not missing:
        return final

    bench_gk = [p for p in bench if position.get(p) == POS_GKP]
    bench_out = [p for p in bench if position.get(p) != POS_GKP]

    # Keeper first, and only for a keeper.
    if any(position.get(p) == POS_GKP for p in missing):
        for gk in bench_gk:
            if played.get(gk, False):
                final.append(gk)
                break

    outfield_missing = sum(1 for p in missing if position.get(p) != POS_GKP)
    usable = [p for p in bench_out if played.get(p, False)]
    if outfield_missing <= 0 or not usable:
        return final

    # Bench order is a preference, not a licence to break the formation, and
    # taking each eligible substitute greedily gets it wrong: with both forwards
    # absent, using the bench defender *and* the bench midfielder leaves no
    # forward at all, when declining the midfielder would have let the bench
    # forward on and kept the side legal. FPL skips a substitution that breaks
    # the formation, so the choice needs lookahead.
    #
    # The bench holds at most four, so enumerate: prefer making as many
    # substitutions as possible, and among equal-sized sets prefer the one
    # earliest in bench order - which is the order `combinations` yields.
    for size in range(min(outfield_missing, len(usable)), 0, -1):
        for combo in itertools.combinations(range(len(usable)), size):
            trial = final + [usable[i] for i in combo]
            counts = [position.get(p, POS_MID) for p in trial]
            if len(trial) == 11:
                if _formation_ok(counts):
                    return trial
            elif all(counts.count(pos) <= hi
                     for pos, (_, hi) in FORMATION_LIMITS.items()):
                # Cannot reach eleven; the maxima are all that can be enforced.
                return trial
    return final


def score_gameweek(starters: List[int], bench: List[int], captain_id: Optional[int],
                   vice_id: Optional[int], points: Dict[int, float],
                   played: Dict[int, bool], position: Dict[int, int],
                   triple: bool = False, bench_boost: bool = False) -> Dict[str, Any]:
    """
    Points the eleven actually returned, autosubs and armband included.

    `triple` is the triple captain chip. `bench_boost` scores all fifteen, and
    with the whole squad already counted there is nothing for an autosub to do
    - which is also why a bench boost cannot be scored by weighting the bench.
    """
    if bench_boost:
        final = list(starters) + list(bench)
    else:
        final = apply_autosubs(starters, bench, played, position)
    total = sum(points.get(p, 0.0) for p in final)

    # The armband only moves when the captain does not appear at all. A captain
    # who played and scored nothing still wears it.
    leader = captain_id
    if captain_id is not None and not played.get(captain_id, False):
        leader = vice_id if vice_id is not None and played.get(vice_id, False) else None
    extra = 0.0
    if leader is not None and leader in final:
        extra = points.get(leader, 0.0) * (2.0 if triple else 1.0)

    # Under a bench boost the whole squad started, so nobody came on for
    # anybody. Deriving this from `final` alone would report all four bench
    # players as substitutes every week the chip is played.
    autosubs = [] if bench_boost else [p for p in final if p not in starters]
    return {"points": total + extra, "eleven": final,
            "leader": leader, "autosubs": autosubs}


def _baseline_matrix(df_gw, horizon_gws: List[int], source: str,
                     player_ids: List[int]) -> Dict[int, Dict[Any, float]]:
    """
    A baseline forecast shaped like the engine's, so the optimiser cannot tell
    the difference. Each gameweek in the horizon gets the same point-in-time
    estimate: a trailing average has no opinion about fixtures.
    """
    base = build_baselines(df_gw, horizon_gws[0]).get(source, {})
    matrix: Dict[int, Dict[Any, float]] = {}
    for pid in player_ids:
        val = float(base.get(pid, 0.0))
        row: Dict[Any, float] = {}
        for gw in horizon_gws:
            row[gw] = val
            # The optimiser reads p_play to avoid fielding non-starters. A
            # trailing average knows nothing about availability, so it has to
            # say so rather than borrow the engine's opinion.
            row[f"{gw}_p_play"] = 1.0 if val > 0 else 0.0
        matrix[pid] = row
    return matrix


def run_season_simulation(season_str: str = "2024-25", horizon: int = 5,
                          xp_source: str = "engine", from_gw: int = 1,
                          to_gw: Optional[int] = None, data=None,
                          verbose: bool = True, max_hits_per_gw: int = 2,
                          xp_cache: Optional[Dict[Any, Any]] = None,
                          hit_cost: float = 4.0,
                          play_chips: bool = True) -> Dict[str, Any]:
    """
    Replay a season, returning the result rather than only logging it.

    `play_chips` runs the shipped chip policy (`chip_policy.choose_chip`), with
    both halves' sets, exactly as the weekly job would. It defaults on because
    a season in which no chip is ever played is not a season anyone plays --
    but it is what this harness used to measure, and the chip machinery was
    consequently covered by nothing. Pass False to reproduce the older numbers.
    """
    if xp_source not in XP_SOURCES:
        raise ValueError(f"xp_source must be one of {XP_SOURCES}")
    df_gw, df_players, df_teams, df_fixtures = data or fetch_data(season_str)
    season_int = int(season_str.split("-")[0])
    max_gw = int(df_gw["GW"].max())
    last_gw = min(to_gw or max_gw, max_gw)

    bank, free_transfers = 100.0, 0
    # chip code -> the gameweek it was played in. Both halves get a set.
    used_chips: Dict[str, int] = {}
    chips_played: List[Dict[str, Any]] = []
    # State to restore the week after a free hit: the squad is borrowed, not
    # bought, so the bank and the purchase prices come back with it.
    free_hit_restore: Optional[Dict[str, Any]] = None
    squad_ids: Optional[List[int]] = None
    prev_squad: List[int] = []
    buy_prices: Dict[int, float] = {}
    total_points = total_hits = total_transfers = 0
    history: List[Dict[str, Any]] = []
    start = time.time()

    for gw in range(from_gw, last_gw + 1):
        bootstrap, fixtures, all_history = build_mock_api(
            df_gw, df_players, df_teams, df_fixtures, current_gw=gw)
        horizon_gws = [g for g in range(gw, gw + horizon) if g <= max_gw]
        elements = bootstrap["elements"]
        now_cost = {p["id"]: float(p["now_cost"]) for p in elements}
        position = {p["id"]: int(p["element_type"]) for p in elements}

        if xp_source == "engine":
            # The forecast depends only on (season, gameweek, horizon) - never on
            # the squad - so it can be shared across variants that differ only in
            # how the optimiser is allowed to spend. Without this, comparing hit
            # thresholds re-runs an identical forecast once per threshold.
            key = (season_str, gw, tuple(horizon_gws))
            if xp_cache is not None and key in xp_cache:
                xp_matrix = xp_cache[key]
            else:
                from xp_model import generate_merv_matrix
                xp_matrix = generate_merv_matrix(
                    horizon_gws, bootstrap=bootstrap, fixtures=fixtures,
                    all_history=all_history, season=season_int)
                if xp_cache is not None:
                    xp_cache[key] = xp_matrix
        else:
            xp_matrix = _baseline_matrix(
                df_gw, horizon_gws, xp_source, [p["id"] for p in elements])

        # FPL sells at buy price plus half the rise, rounded down.
        sell_prices = {}
        for pid in (squad_ids or []):
            bought = buy_prices.get(pid, now_cost.get(pid, 50.0))
            cur = now_cost.get(pid, bought)
            sell_prices[pid] = (bought + (cur - bought) // 2) / 10.0 if cur > bought \
                else cur / 10.0

        def _solve(chip=None, chip_gw=None):
            return solve_fpl_optimization(
                bootstrap, xp_matrix, horizon_gws, initial_squad_ids=squad_ids,
                initial_bank=bank, initial_sell_prices=sell_prices,
                initial_ft=free_transfers, max_hits_per_gw=max_hits_per_gw,
                hit_cost=hit_cost, active_chip=chip, active_chip_gw=chip_gw)

        chip_now = None
        try:
            # A chip is searched across the horizon and played only if the
            # policy picks *this* gameweek for it; picked for a later one, the
            # plan still accounts for it and the search runs again next week
            # with a better forecast. That is what the weekly job does.
            if play_chips and squad_ids is not None:
                best = chip_policy.choose_chip(
                    _solve, horizon_gws, gw,
                    chip_policy.available_chips(used_chips, gw))
                res = best["res"]
                if best["chip"] and best["gw"] == gw:
                    chip_now = best["chip"]
            else:
                res = _solve()
        except Exception as e:
            logger.error("GW%d: solver failed (%s); holding the squad.", gw, e)
            res = None
            chip_now = None

        if chip_now:
            used_chips[chip_now] = gw
            chips_played.append({"chip": chip_now, "gw": gw})
            if chip_now == "fh":
                # The free hit squad is borrowed for one week. Snapshot what is
                # handed back: the squad, the bank, and the purchase prices,
                # since none of the week's buying actually happened.
                free_hit_restore = {
                    "squad_ids": list(squad_ids or []),
                    "bank": bank,
                    "buy_prices": dict(buy_prices),
                }

        if res and gw in res.get("gameweeks", {}):
            plan = res["gameweeks"][gw]
            starters = [p["id"] for p in plan["starters"]]
            bench = [p["id"] for p in plan["bench"]]
            captain_id = plan.get("captain_id")
            vice_id = next((p["id"] for p in plan["starters"]
                            if p.get("is_vice_captain")), None)
            hits = int(plan.get("hits", 0))
            transfers_in = [p["id"] for p in plan.get("transfers_in", [])]
            bank = float(plan.get("bank", bank))
            for pid in transfers_in:
                buy_prices[pid] = now_cost.get(pid, 50.0)
            squad_ids = starters + bench
            buy_prices = {k: v for k, v in buy_prices.items() if k in squad_ids}
            for pid in squad_ids:
                buy_prices.setdefault(pid, now_cost.get(pid, 50.0))
        else:
            if squad_ids is None:
                logger.error("GW%d: no squad and no solution; stopping.", gw)
                break
            starters, bench = squad_ids[:11], squad_ids[11:]
            captain_id = vice_id = starters[0] if starters else None
            hits, transfers_in = 0, []

        target = df_gw[df_gw["GW"] == gw]
        pts = target.groupby("element")["total_points"].sum().to_dict()
        mins = target.groupby("element")["minutes"].sum().to_dict()
        played = {int(k): v > 0 for k, v in mins.items()}

        scored = score_gameweek(starters, bench, captain_id, vice_id,
                                pts, played, position,
                                triple=(chip_now == "tc"),
                                bench_boost=(chip_now == "bb"))
        net = scored["points"] - 4 * hits
        total_points += net
        total_hits += hits
        total_transfers += len(transfers_in)

        history.append({
            "gw": gw, "points": scored["points"], "hits": hits, "net": net,
            "chip": chip_now,
            "cumulative": total_points, "transfers": len(transfers_in),
            "bank": bank, "captain": captain_id, "leader": scored["leader"],
            "autosubs": len(scored["autosubs"]),
            "squad_value": sum(now_cost.get(p, 0.0) for p in squad_ids) / 10.0,
            # Who moved, and what was held. Needed to ask how long a player the
            # optimiser paid a hit for is actually kept: the hit is priced
            # against a five-gameweek horizon, which is only the right price if
            # he is still there in five gameweeks.
            "in": list(transfers_in),
            "out": sorted(set(prev_squad) - set(squad_ids)),
            "squad": list(squad_ids),
        })
        # A free hit squad is borrowed for the week it plays and then handed
        # back, so the next gameweek resumes from the state saved above rather
        # than from what the chip bought.
        if chip_now == "fh" and free_hit_restore is not None:
            squad_ids = free_hit_restore["squad_ids"]
            bank = free_hit_restore["bank"]
            buy_prices = free_hit_restore["buy_prices"]
            free_hit_restore = None

        prev_squad = list(squad_ids)
        if verbose:
            logger.info("GW%-2d  %-6s  net %3d  running %4d  (hits %d, subs %d)%s",
                        gw, xp_source, net, total_points, hits,
                        len(scored["autosubs"]),
                        f"  [{chip_now.upper()}]" if chip_now else "")

        if gw == 1:
            free_transfers = 1
        elif chip_now in ("wc", "fh"):
            # The chip paid for the week's transfers, and saved free transfers
            # survive it rather than being spent by it.
            free_transfers = min(5, free_transfers + 1)
        else:
            free_transfers = max(1, min(5, free_transfers + 1 - len(transfers_in)))

    gws = max(1, len(history))
    return {
        "season": season_str, "xp_source": xp_source, "gameweeks": len(history),
        "total_points": float(total_points),
        "points_per_gw": float(total_points) / gws,
        "total_hits": total_hits, "total_transfers": total_transfers,
        "chips_played": chips_played,
        "elapsed_s": round(time.time() - start, 1), "history": history,
    }


def compare_sources(season_str: str = "2024-25", horizon: int = 5,
                    sources=XP_SOURCES, from_gw: int = 1,
                    to_gw: Optional[int] = None,
                    play_chips: bool = True) -> Dict[str, Any]:
    """Drive the same optimiser with each forecast and report the difference."""
    data = fetch_data(season_str)
    out = {}
    for src in sources:
        out[src] = run_season_simulation(
            season_str, horizon=horizon, xp_source=src,
            from_gw=from_gw, to_gw=to_gw, data=data, play_chips=play_chips)
    _print_comparison(out)
    return out


def _print_comparison(results: Dict[str, Any]):
    print()
    print("=" * 74)
    first = next(iter(results.values()))
    print(f" SEASON REPLAY  ({first['season']}, {first['gameweeks']} gameweeks, "
          f"same optimiser throughout)")
    print("=" * 74)
    print(f" {'forecast':<12}{'points':>9}{'per GW':>9}{'hits':>7}"
          f"{'transfers':>11}{'vs engine':>11}")
    print(" " + "-" * 70)
    eng = results.get("engine", {}).get("total_points")
    for src, r in sorted(results.items(), key=lambda kv: -kv[1]["total_points"]):
        delta = "" if eng is None or src == "engine" else f"{r['total_points'] - eng:+.0f}"
        print(f" {src:<12}{r['total_points']:>9.0f}{r['points_per_gw']:>9.2f}"
              f"{r['total_hits']:>7d}{r['total_transfers']:>11d}{delta:>11}")

    # Which chips were played and when. A season with none is a season this
    # harness is not measuring the chip machinery in, which is worth seeing
    # rather than inferring.
    for src, r in sorted(results.items()):
        played = r.get("chips_played") or []
        detail = ", ".join(f"{c['chip'].upper()}@GW{c['gw']}" for c in played) \
            if played else "none played"
        print(f"   chips  {src:<10}{detail}")

    # Per-gameweek paired test: a season total is one sample, and one sample
    # cannot tell a better forecast from a luckier one.
    #
    # The mean is the right statistic here, which is worth stating because it is
    # not obvious and the alternative was tried. Rank-based tests are the usual
    # answer to a noisy difference, but the paired difference is not heavy-tailed
    # (excess kurtosis -0.41 over 76 gameweeks) and the engine wins *bigger*
    # rather than *more often* - only 51.3% of gameweeks. A rank or sign test
    # discards exactly the signal:
    #
    #     t-test on the mean    p = 0.066
    #     Wilcoxon signed-rank  p = 0.158
    #     sign test / win rate  p = 0.909
    #
    # So the bootstrap mean stays the verdict. Win rate and a trimmed mean are
    # reported beside it because they describe *how* the edge arrives, which is
    # the thing a season total hides.
    if eng is not None and len(results) > 1:
        print()
        print(" engine minus baseline, paired by gameweek:")
        print(f"   {'baseline':<12}{'mean':>8}{'95% CI':>18}{'trimmed':>9}"
              f"{'win rate':>10}{'':>4}")
        e = np.array([h["net"] for h in results["engine"]["history"]], dtype=float)
        rng = np.random.default_rng(0)
        for src, r in results.items():
            if src == "engine":
                continue
            b = np.array([h["net"] for h in r["history"]], dtype=float)
            n = min(len(e), len(b))
            d = e[:n] - b[:n]
            boot = rng.choice(d, size=(20000, n), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            mark = "significant" if (lo > 0) == (hi > 0) else "noise"
            k = max(1, int(0.1 * n))
            trimmed = float(np.sort(d)[k:n - k].mean()) if n > 2 * k else float(d.mean())
            print(f"   {src:<12}{d.mean():>+8.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>18}"
                  f"{trimmed:>+9.2f}{(d > 0).mean():>10.3f}  {mark}")
    print("=" * 74)


def compare_seasons(seasons: List[str], horizon: int = 5, sources=XP_SOURCES,
                    from_gw: int = 1, to_gw: Optional[int] = None,
                    play_chips: bool = True) -> Dict[str, Any]:
    """
    Replay several seasons and pool the paired per-gameweek differences.

    The pooled figure is the one worth quoting - a single season is ~38 paired
    samples of a very noisy difference - but it used to be computed outside the
    repo, so the numbers in the README could not be re-derived from anything
    here. They can now.
    """
    per_season = {s: compare_sources(s, horizon=horizon, sources=sources,
                                     from_gw=from_gw, to_gw=to_gw,
                                     play_chips=play_chips)
                  for s in seasons}
    if len(seasons) > 1:
        _print_pooled(per_season)
    return per_season


def _print_pooled(per_season: Dict[str, Dict[str, Any]]):
    sources = sorted({s for r in per_season.values() for s in r})
    nets = {src: np.concatenate([
        np.array([h["net"] for h in per_season[season][src]["history"]], dtype=float)
        for season in per_season if src in per_season[season]])
        for src in sources}
    n_gw = len(next(iter(nets.values())))

    print()
    print("=" * 74)
    print(f" POOLED  ({', '.join(sorted(per_season))} - {n_gw} gameweeks, "
          f"paired by gameweek)")
    print("=" * 74)
    print(f" {'forecast':<12}{'points':>9}{'per GW':>9}{'hits':>7}{'transfers':>11}")
    print(" " + "-" * 70)
    for src in sorted(sources, key=lambda s: -float(nets[s].sum())):
        hits = sum(per_season[se][src]["total_hits"] for se in per_season)
        trans = sum(per_season[se][src]["total_transfers"] for se in per_season)
        print(f" {src:<12}{nets[src].sum():>9.0f}{nets[src].mean():>9.2f}"
              f"{hits:>7d}{trans:>11d}")

    if "engine" not in nets:
        print("=" * 74)
        return

    e = nets["engine"]
    rng = np.random.default_rng(0)
    print()
    print(" engine minus baseline, paired by gameweek:")
    print(f"   {'baseline':<12}{'mean':>8}{'95% CI':>18}{'total':>8}"
          f"{'win rate':>10}{'':>4}")
    for src in sorted(sources):
        if src == "engine":
            continue
        b = nets[src]
        n = min(len(e), len(b))
        d = e[:n] - b[:n]
        boot = rng.choice(d, size=(20000, n), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        mark = "significant" if (lo > 0) == (hi > 0) else "noise"
        print(f"   {src:<12}{d.mean():>+8.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>18}"
              f"{d.sum():>+8.0f}{(d > 0).mean():>10.3f}  {mark}")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(description="Full-season replay of the engine")
    ap.add_argument("--season", default="2024-25",
                    help="A single season; --seasons pools several")
    ap.add_argument("--seasons", nargs="+", default=None,
                    help="Replay these seasons and pool the paired differences. "
                         "The pooled figure is the one worth quoting: one season "
                         "is ~38 paired samples of a very noisy difference.")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--from-gw", type=int, default=1)
    ap.add_argument("--to-gw", type=int, default=None)
    ap.add_argument("--sources", nargs="+", default=list(XP_SOURCES),
                    choices=list(XP_SOURCES),
                    help="Forecasts to drive the optimiser with")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--no-chips", action="store_true",
                    help="Replay without ever playing a chip (the old behaviour)")
    args = ap.parse_args()

    if args.seasons:
        results = compare_seasons(args.seasons, horizon=args.horizon,
                                  sources=args.sources, from_gw=args.from_gw,
                                  to_gw=args.to_gw, play_chips=not args.no_chips)
    else:
        results = compare_sources(args.season, horizon=args.horizon,
                                  sources=args.sources, from_gw=args.from_gw,
                                  to_gw=args.to_gw, play_chips=not args.no_chips)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
