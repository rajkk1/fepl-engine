"""
Walk-forward backtest.

The previous harness compared the engine against the `xP` column in Vaastav's
merged gameweek data, treating it as "FPL's own expected points". That column is
scraped after the gameweek and FPL zeroes it for players who did not feature -
64-75% of non-participants had xP exactly 0.00, giving it an AUC of ~0.95 at
predicting participation. No pre-deadline forecast knows that, so it was not a
valid comparator and the model looked far worse against it than it deserved.

Baselines here are strictly computable before the deadline. The leaking column
is still reported, clearly labelled, so the contamination stays visible - but it
never gates anything.

Other corrections:
  * the evaluation population was chosen partly by the model's own predictions,
    so changing the model changed which players got scored
  * availability was stubbed to "everyone is 100% fit", so the rotation half of
    the minutes model was never exercised
  * Poisson deviance was the only headline number, and it detonated whenever a
    baseline value was zero
"""
import argparse
import json
import logging
import math
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VAASTAV = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# Columns whose absence means "this dataset does not report it", not "zero".
OPTIONAL_STAT_COLUMNS = [
    "clearances_blocks_interceptions", "recoveries", "tackles",
    "expected_goals", "expected_assists", "expected_goals_conceded",
    "saves", "yellow_cards", "red_cards", "starts", "bps", "defensive_contribution",
    "own_goals", "penalties_missed", "penalties_saved",
]


_DATA_CACHE: Dict[str, Any] = {}


def _read_csv_retrying(url: str, attempts: int = 4, base_delay: float = 3.0):
    """
    Read a remote CSV, retrying on transient failures.

    A multi-season backtest makes hundreds of requests and the host rate-limits,
    which arrives as an HTTP 400 rather than a 429. Failing the whole run on one
    refused request throws away an hour of work for something a short wait fixes.
    """
    for attempt in range(attempts):
        try:
            return pd.read_csv(url, low_memory=False)
        except Exception as e:
            if attempt == attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("Fetch failed (%s); retrying in %.0fs [%d/%d]: %s",
                           type(e).__name__, delay, attempt + 1, attempts - 1, url)
            time.sleep(delay)


def fetch_data(season_str: str = "2024-25"):
    """Season data, cached in-process: callers re-request the same seasons."""
    if season_str in _DATA_CACHE:
        return _DATA_CACHE[season_str]
    base = f"{VAASTAV}/{season_str}"
    logger.info("Downloading historical data for %s...", season_str)
    out = (
        _read_csv_retrying(f"{base}/gws/merged_gw.csv"),
        _read_csv_retrying(f"{base}/players_raw.csv"),
        _read_csv_retrying(f"{base}/teams.csv"),
        _read_csv_retrying(f"{base}/fixtures.csv"),
    )
    _DATA_CACHE[season_str] = out
    return out


# --------------------------------------------------------------- mock the API


def build_mock_api(df_gw, df_players, df_teams, df_fixtures, current_gw: int,
                   prior_season_gw=None):
    """
    Reconstruct the FPL API as it would have looked before `current_gw`.

    `prior_season_gw` is last season's merged gameweek frame. When supplied it
    populates `history_past`, which is what the live API returns alongside the
    current season and which the engine uses for prior-season priors. Without
    it the backtest cannot exercise that path at all.
    """
    teams = df_teams.to_dict(orient="records")

    fixtures = []
    for f in df_fixtures.to_dict(orient="records"):
        gw = f.get("event")
        if pd.isna(gw):
            continue
        finished = gw < current_gw
        fixtures.append({
            "event": int(gw),
            "team_h": int(f.get("team_h")),
            "team_a": int(f.get("team_a")),
            "team_h_score": int(f["team_h_score"]) if finished and not pd.isna(f.get("team_h_score")) else None,
            "team_a_score": int(f["team_a_score"]) if finished and not pd.isna(f.get("team_a_score")) else None,
            "finished": finished,
            "kickoff_time": f.get("kickoff_time"),
            "team_h_difficulty": int(f.get("team_h_difficulty", 3)),
            "team_a_difficulty": int(f.get("team_a_difficulty", 3)),
        })

    df_past = df_gw[df_gw["GW"] < current_gw]
    present = set(df_gw.columns)

    stats: Dict[int, Dict[str, float]] = {}
    for pid, group in df_past.groupby("element"):
        mins = group["minutes"].sum()
        xg = pd.to_numeric(group.get("expected_goals", 0), errors="coerce").sum()
        xa = pd.to_numeric(group.get("expected_assists", 0), errors="coerce").sum()
        pts = group["total_points"].sum()
        games = len(group)
        stats[int(pid)] = {
            "xg90": (xg / mins * 90) if mins > 0 else 0.0,
            "xa90": (xa / mins * 90) if mins > 0 else 0.0,
            "ppg": (pts / games) if games else 0.0,
            "val": float(group["value"].iloc[-1]) if not group.empty else 50.0,
            "minutes": float(mins),
            # Consecutive gameweeks with no minutes, ending at current_gw - 1.
            "absent_streak": _absent_streak(group),
        }

    df_prev = df_gw[df_gw["GW"] == current_gw - 1]
    own_pct: Dict[int, float] = {}
    if not df_prev.empty and "selected" in df_prev.columns:
        sel = df_prev.groupby("element")["selected"].max()
        managers = sel.sum() / 15.0
        if managers > 0:
            own_pct = (100.0 * sel / managers).to_dict()

    prior_aggregates = _prior_season_aggregates(prior_season_gw)

    elements = []
    for p in df_players.to_dict(orient="records"):
        pid = int(p["id"])
        st = stats.get(pid, {"xg90": 0.0, "xa90": 0.0, "ppg": 0.0,
                             "val": float(p.get("now_cost", 50)), "minutes": 0.0,
                             "absent_streak": 0})

        # Availability is NOT stubbed to "fit". We cannot recover injury news
        # retrospectively, so we say "unknown" (which is what the live API says
        # for an unflagged player) and let the minutes model's own absence
        # features do the work. Asserting 100% fitness for everyone silently
        # removed the rotation half of the problem.
        elements.append({
            "id": pid,
            "web_name": str(p.get("web_name", f"Player_{pid}")),
            "element_type": int(p.get("element_type", 3)),
            "team": int(p.get("team", 1)),
            "now_cost": st["val"],
            "status": "a",
            "chance_of_playing_next_round": None,
            "news": "",
            "minutes": st["minutes"],
            "expected_goals_per_90": st["xg90"],
            "expected_assists_per_90": st["xa90"],
            "points_per_game": st["ppg"],
            "selected_by_percent": own_pct.get(pid, 0.0),
            "penalties_order": _order(p.get("penalties_order")),
            # Set-piece duty: published by FPL, and the strongest freely
            # available assist signal a player's own xA cannot yet show.
            "corners_and_indirect_freekicks_order":
                _order(p.get("corners_and_indirect_freekicks_order")),
            "direct_freekicks_order": _order(p.get("direct_freekicks_order")),
            "history_past": prior_aggregates.get(pid, []),
        })

    bootstrap = {"teams": teams, "elements": elements}

    # Built column-wise rather than with iterrows(): this runs once per target
    # gameweek in a walk-forward backtest and row-wise pandas access dominated
    # the whole harness.
    base_cols = ["element", "GW", "minutes", "total_points", "value",
                 "was_home", "opponent_team"]
    if "element_type" in present:
        base_cols.append("element_type")
    opt_cols = [c for c in OPTIONAL_STAT_COLUMNS if c in present]

    slim = df_past[base_cols + [c for c in opt_cols if c not in base_cols]].copy()
    records = slim.to_dict(orient="records")

    all_history: Dict[int, List[Dict[str, Any]]] = {}
    for row in records:
        h = {
            "round": int(row["GW"]),
            "minutes": int(row["minutes"]),
            "total_points": int(row["total_points"]),
            "value": float(row.get("value", 50) or 50),
            "was_home": bool(row.get("was_home", False)),
            "opponent_team": int(row.get("opponent_team", 1) or 1),
            "element_type": (int(row["element_type"])
                             if "element_type" in row and not pd.isna(row["element_type"])
                             else None),
        }
        # Only emit optional stats the dataset actually carries. Emitting a zero
        # for an absent column made the Gamma-Poisson posterior treat it as an
        # observed zero, collapsing the rate instead of holding the prior.
        # 2024-25 has no CBIT/tackles/recoveries columns at all.
        for col in opt_cols:
            v = row.get(col)
            if v is not None and not pd.isna(v):
                h[col] = float(v)
        all_history.setdefault(int(row["element"]), []).append(h)
    for rows in all_history.values():
        rows.sort(key=lambda r: r["round"])

    for e in elements:
        all_history.setdefault(e["id"], [])
        if e["element_type"] is not None:
            for h in all_history[e["id"]]:
                if h.get("element_type") is None:
                    h["element_type"] = e["element_type"]

    return bootstrap, fixtures, all_history


def _prior_season_aggregates(prior_gw) -> Dict[int, List[Dict[str, Any]]]:
    """Season totals per player, shaped like the API's `history_past` entries."""
    if prior_gw is None or len(prior_gw) == 0:
        return {}
    cols = ["minutes", "total_points", "expected_goals", "expected_assists",
            "saves", "yellow_cards", "starts", "bonus"]
    have = [c for c in cols if c in prior_gw.columns]
    agg = prior_gw.groupby("element")[have].sum()
    out: Dict[int, List[Dict[str, Any]]] = {}
    for pid, row in agg.iterrows():
        entry = {c: float(row[c]) for c in have}
        entry["season_name"] = "prior"
        out[int(pid)] = [entry]
    return out


def _order(value):
    """FPL set-piece / penalty order: a small int, or None when not a taker."""
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _absent_streak(group) -> int:
    mins = list(group.sort_values("GW")["minutes"])
    streak = 0
    for m in reversed(mins):
        if m == 0:
            streak += 1
        else:
            break
    return streak


# ------------------------------------------------------------------ baselines


def build_baselines(df_gw, target_gw: int) -> Dict[str, Dict[int, float]]:
    """
    Point-in-time baselines. Everything here uses only gameweeks strictly before
    `target_gw`, so each is something you could genuinely have computed at the
    deadline.
    """
    past = df_gw[df_gw["GW"] < target_gw]
    out: Dict[str, Dict[int, float]] = {"ppg": {}, "roll3": {}, "roll3_mins": {}}
    if past.empty:
        return out

    for pid, g in past.groupby("element"):
        g = g.sort_values("GW")
        pts = g["total_points"].astype(float)
        mins = g["minutes"].astype(float)
        out["ppg"][int(pid)] = float(pts.mean())
        out["roll3"][int(pid)] = float(pts.tail(3).mean())
        # Points per game, scaled by how reliably they have been playing - a
        # trivially better baseline that the engine ought to clear comfortably.
        played_rate = float((mins.tail(5) > 0).mean()) if len(mins) else 0.0
        out["roll3_mins"][int(pid)] = float(pts.tail(3).mean()) * played_rate

    return out


def build_eval_population(df_gw, target_gw: int, top_n: int = 250) -> set:
    """
    Which players get scored.

    Chosen only from information independent of the model: ownership before the
    gameweek and recent minutes. The previous version unioned in the top 30 per
    position *by predicted xP*, so improving the model changed the population and
    even made the fixed baseline's score drift.
    """
    prev = df_gw[df_gw["GW"] == target_gw - 1]
    past = df_gw[(df_gw["GW"] < target_gw) & (df_gw["GW"] >= max(1, target_gw - 5))]

    pop = set()
    if not prev.empty and "selected" in prev.columns:
        top = prev.groupby("element")["selected"].max().sort_values(ascending=False)
        pop.update(int(i) for i in top.head(top_n).index)
    if not past.empty:
        # Anyone with meaningful recent minutes is a live FPL option.
        mins = past.groupby("element")["minutes"].sum()
        pop.update(int(i) for i in mins[mins >= 180].index)
    return pop


# -------------------------------------------------------------------- metrics


def _spearman_by_position(actual, pred, positions) -> float:
    import scipy.stats as st

    rhos = []
    for pos in (1, 2, 3, 4):
        idx = [i for i, p in enumerate(positions) if p == pos]
        if len(idx) < 3:
            continue
        a = [actual[i] for i in idx]
        p = [pred[i] for i in idx]
        if len(set(p)) < 2 or len(set(a)) < 2:
            continue
        r, _ = st.spearmanr(a, p)
        if not math.isnan(r):
            rhos.append(r)
    return float(np.mean(rhos)) if rhos else 0.0


def _precision_at_k(actual, pred, ids, k=15) -> float:
    order_a = [i for _, i in sorted(zip(actual, ids), key=lambda t: -t[0])]
    order_p = [i for _, i in sorted(zip(pred, ids), key=lambda t: -t[0])]
    return len(set(order_a[:k]) & set(order_p[:k]))


def _captain_regret(actual, pred, ids) -> float:
    """Points given up by captaining the model's top pick instead of the best."""
    if not ids:
        return 0.0
    best_actual = max(actual)
    pick = max(range(len(ids)), key=lambda i: pred[i])
    return float(best_actual - actual[pick])


def evaluate_gameweek(preds: Dict[str, Dict[int, float]], actual: Dict[int, float],
                      positions: Dict[int, int], costs: Dict[int, float],
                      population: List[int], p_play: Dict[int, float],
                      played: Dict[int, bool]) -> Dict[str, Any]:
    ids = [pid for pid in population if pid in actual]
    if not ids:
        return {}

    a = [float(actual[pid]) for pid in ids]
    pos = [positions.get(pid, 3) for pid in ids]
    out: Dict[str, Any] = {"n": len(ids), "models": {}}

    for name, pred_map in preds.items():
        p = [float(pred_map.get(pid, 0.0)) for pid in ids]
        err = np.array(a) - np.array(p)
        entry = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(-err)),
            "spearman": _spearman_by_position(a, p, pos),
            "precision_at_15": _precision_at_k(a, p, ids, 15),
            "captain_regret": _captain_regret(a, p, ids),
        }
        # Per-position and per-price-band error, so a model that is fine on
        # midfielders and broken on defenders cannot hide in the aggregate.
        entry["by_position"] = {}
        for pos_id in (1, 2, 3, 4):
            idx = [i for i, q in enumerate(pos) if q == pos_id]
            if idx:
                e = np.array([a[i] - p[i] for i in idx])
                entry["by_position"][pos_id] = {
                    "n": len(idx), "mae": float(np.mean(np.abs(e))),
                    "bias": float(np.mean(-e)),
                }
        entry["by_price"] = {}
        bands = [(0, 5.0), (5.0, 7.5), (7.5, 10.0), (10.0, 99.0)]
        for lo, hi in bands:
            idx = [i for i, pid in enumerate(ids) if lo <= costs.get(pid, 0) / 10.0 < hi]
            if idx:
                e = np.array([a[i] - p[i] for i in idx])
                entry["by_price"][f"{lo}-{hi}"] = {
                    "n": len(idx), "mae": float(np.mean(np.abs(e))),
                    "bias": float(np.mean(-e)),
                }
        out["models"][name] = entry

    # Minutes-model quality, which is the single largest driver of point error.
    if p_play:
        y = np.array([1.0 if played.get(pid) else 0.0 for pid in ids])
        q = np.clip(np.array([p_play.get(pid, 0.0) for pid in ids]), 1e-6, 1 - 1e-6)
        out["p_play_logloss"] = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
        if 0 < y.sum() < len(y):
            from sklearn.metrics import roc_auc_score
            out["p_play_auc"] = float(roc_auc_score(y, q))
    return out


# ------------------------------------------------------------------- backtest


def run_backtest(season_str: str = "2025-26", test_gws: Optional[List[int]] = None,
                 df_gw=None, df_players=None, df_teams=None, df_fixtures=None,
                 prior_season_gw=None,
                 calibrate: bool = False, risk_aversion: float = 0.0,
                 lineup_overrides=None,
                 calibration_method: str = "linear",
                 include_leaky_baseline: bool = True,
                 verbose: bool = True) -> Dict[str, Any]:
    """Walk forward through `test_gws`, scoring the engine against clean baselines."""
    from xp_model import generate_merv_matrix

    if df_gw is None:
        df_gw, df_players, df_teams, df_fixtures = fetch_data(season_str)

    max_gw = int(df_gw["GW"].max())
    if test_gws is None:
        # Everything with enough history to forecast from, through the end.
        test_gws = list(range(5, max_gw + 1))
    test_gws = [g for g in test_gws if 2 <= g <= max_gw]

    season_int = int(season_str.split("-")[0])
    per_gw: List[Dict[str, Any]] = []

    if verbose:
        logger.info("Backtesting %s over GW%d-%d", season_str, test_gws[0], test_gws[-1])

    for gw in test_gws:
        bootstrap, fixtures, all_history = build_mock_api(
            df_gw, df_players, df_teams, df_fixtures, current_gw=gw,
            prior_season_gw=prior_season_gw,
        )
        matrix = generate_merv_matrix(
            [gw], bootstrap=bootstrap, fixtures=fixtures, all_history=all_history,
            season=season_int, risk_aversion=risk_aversion, calibrate=calibrate,
            calibration_method=calibration_method, lineup_overrides=lineup_overrides,
        )

        target = df_gw[df_gw["GW"] == gw]
        actual = target.groupby("element")["total_points"].sum().to_dict()
        mins = target.groupby("element")["minutes"].sum().to_dict()
        played = {int(k): v > 0 for k, v in mins.items()}

        baselines = build_baselines(df_gw, gw)
        population = sorted(build_eval_population(df_gw, gw) & set(actual.keys()))

        pdict = {p["id"]: p for p in bootstrap["elements"]}
        positions = {pid: pdict.get(pid, {}).get("element_type", 3) for pid in population}
        costs = {pid: pdict.get(pid, {}).get("now_cost", 50) for pid in population}

        preds = {
            "fepl": {pid: matrix.get(pid, {}).get(gw, 0.0) for pid in population},
            "ppg": {pid: baselines["ppg"].get(pid, 0.0) for pid in population},
            "roll3": {pid: baselines["roll3"].get(pid, 0.0) for pid in population},
            "roll3_mins": {pid: baselines["roll3_mins"].get(pid, 0.0) for pid in population},
        }
        if include_leaky_baseline and "xP" in target.columns:
            leaky = target.groupby("element")["xP"].sum().to_dict()
            # Guard properly: any missing or zero value is treated as absent, not
            # as a confident zero. The old guard only skipped when *every* value
            # was zero, so partial zeros blew the deviance up to 40+.
            if sum(1 for v in leaky.values() if v and v > 0) > 0.5 * len(leaky):
                preds["fpl_xp_LEAKS"] = {pid: float(leaky.get(pid, 0.0)) for pid in population}

        p_play = {pid: matrix.get(pid, {}).get(f"{gw}_p_play", 0.0) for pid in population}

        res = evaluate_gameweek(preds, actual, positions, costs, population, p_play, played)
        if res:
            res["gw"] = gw
            per_gw.append(res)
            if verbose:
                f = res["models"]["fepl"]
                p = res["models"]["ppg"]
                logger.info(
                    "GW%-2d n=%-4d | FEPL mae %.3f rho %.3f p@15 %2d capreg %4.1f "
                    "| PPG mae %.3f rho %.3f p@15 %2d | p_play auc %.3f",
                    gw, res["n"], f["mae"], f["spearman"], f["precision_at_15"],
                    f["captain_regret"], p["mae"], p["spearman"], p["precision_at_15"],
                    res.get("p_play_auc", float("nan")),
                )

    return summarise(per_gw, season_str, verbose=verbose)


def summarise(per_gw: List[Dict[str, Any]], season_str: str,
              verbose: bool = True) -> Dict[str, Any]:
    if not per_gw:
        return {"season": season_str, "gameweeks": 0, "models": {}}

    names = list(per_gw[0]["models"].keys())
    summary: Dict[str, Any] = {
        "season": season_str,
        "gameweeks": len(per_gw),
        # Per-gameweek detail, so a difference between two models can be tested
        # for significance rather than eyeballed. Nine gameweeks is a very small
        # sample for a statistic as noisy as captain regret.
        "per_gameweek": per_gw,
        "p_play_auc": float(np.mean([g["p_play_auc"] for g in per_gw if "p_play_auc" in g]))
        if any("p_play_auc" in g for g in per_gw) else None,
        "p_play_logloss": float(np.mean([g["p_play_logloss"] for g in per_gw
                                         if "p_play_logloss" in g]))
        if any("p_play_logloss" in g for g in per_gw) else None,
        "models": {},
    }

    for name in names:
        rows = [g["models"][name] for g in per_gw if name in g["models"]]
        if not rows:
            continue
        summary["models"][name] = {
            "mae": float(np.mean([r["mae"] for r in rows])),
            "rmse": float(np.mean([r["rmse"] for r in rows])),
            "bias": float(np.mean([r["bias"] for r in rows])),
            "spearman": float(np.mean([r["spearman"] for r in rows])),
            "precision_at_15": float(np.mean([r["precision_at_15"] for r in rows])),
            "captain_regret": float(np.mean([r["captain_regret"] for r in rows])),
            "by_position": _merge_breakdown(rows, "by_position"),
            "by_price": _merge_breakdown(rows, "by_price"),
        }

    if verbose:
        _print_summary(summary)
    return summary


def _merge_breakdown(rows, key):
    out: Dict[Any, Dict[str, float]] = {}
    keys = {k for r in rows for k in r.get(key, {})}
    for k in keys:
        vals = [r[key][k] for r in rows if k in r.get(key, {})]
        if vals:
            out[k] = {
                "n": int(np.sum([v["n"] for v in vals])),
                "mae": float(np.mean([v["mae"] for v in vals])),
                "bias": float(np.mean([v["bias"] for v in vals])),
            }
    return out


POS_NAMES = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def _print_summary(s: Dict[str, Any]):
    print()
    print("=" * 86)
    print(f" BACKTEST SUMMARY  ({s['season']}, {s['gameweeks']} gameweeks)")
    print("=" * 86)
    if s.get("p_play_auc") is not None:
        print(f" minutes model: p_play AUC {s['p_play_auc']:.3f} | "
              f"log-loss {s['p_play_logloss']:.3f}")
    print()
    print(f" {'model':<16}{'MAE':>8}{'RMSE':>8}{'bias':>8}{'rho':>8}"
          f"{'P@15':>8}{'cap.regret':>12}")
    print(" " + "-" * 82)
    for name, m in sorted(s["models"].items(), key=lambda kv: kv[1]["mae"]):
        flag = "  <- LEAKS" if "LEAK" in name else ""
        print(f" {name:<16}{m['mae']:>8.3f}{m['rmse']:>8.3f}{m['bias']:>+8.3f}"
              f"{m['spearman']:>8.3f}{m['precision_at_15']:>8.1f}"
              f"{m['captain_regret']:>12.2f}{flag}")

    fepl = s["models"].get("fepl")
    if fepl:
        print()
        print(" FEPL error by position:")
        for pos, v in sorted(fepl["by_position"].items()):
            print(f"   {POS_NAMES.get(pos, pos):<5} n={v['n']:<6} mae {v['mae']:.3f}  "
                  f"bias {v['bias']:+.3f}")
        print(" FEPL error by price band (£m):")
        for band, v in sorted(fepl["by_price"].items(), key=lambda kv: float(kv[0].split("-")[0])):
            print(f"   {band:<10} n={v['n']:<6} mae {v['mae']:.3f}  bias {v['bias']:+.3f}")
    print("=" * 86)


# ----------------------------------------------------------------- CI gating


CLEAN_BASELINES = ("ppg", "roll3", "roll3_mins")

# True when a lower value is better.
LOWER_IS_BETTER = {
    "mae": True, "rmse": True, "captain_regret": True,
    "spearman": False, "precision_at_15": False,
}

# What the gate checks by default.
#
# MAE alone is the wrong test and passing it means less than it looks. Averaged
# over ~300 players it is dominated by correctly predicting low scores and
# non-starters, so a model can clear every baseline on MAE while being no better
# than trailing points-per-game at the two decisions FPL is actually won on:
# which fifteen players to own, and who to captain. Worse, MAE actively rewards
# shrinking the forecast toward the mean, which is precisely what blunts the top
# of the ranking.
#
# So the gate covers error *and* rank quality. `spearman` is within-position rank
# correlation: it is decision-relevant *and* it is stable enough to gate on.
#
# What gets gated is decided by measurement, not by how much the metric matters.
# Paired per-gameweek bootstrap over a full 2025-26 season (34 gameweeks), FEPL
# minus the trailing points-per-game baseline, 95% CI:
#
#     mae               -0.183  [-0.210, -0.157]   significant
#     spearman          +0.130  [+0.106, +0.155]   significant
#     precision_at_15   +0.059  [-0.382, +0.500]   NOT significant
#     captain_regret    +0.147  [-1.235, +1.353]   NOT significant
#
# precision@15 and captain regret are the metrics that matter most per point,
# and they are also too noisy to gate on with one season of data - the CI on
# each spans zero comfortably. Gating on them would fail builds at random and
# teach us to chase noise. They are reported as advisory instead, and gating
# them needs several seasons of evaluation, not a better threshold.
DEFAULT_GATE_METRICS = ("mae", "spearman")

# Reported on every gate run but not enforced: see the CIs above.
ADVISORY_GATE_METRICS = ("precision_at_15", "captain_regret")


def _compare(fepl: Dict[str, Any], baseline: Dict[str, Any], metric: str) -> bool:
    if LOWER_IS_BETTER[metric]:
        return fepl[metric] < baseline[metric]
    return fepl[metric] > baseline[metric]


def check_gate(summary: Dict[str, Any], metrics=DEFAULT_GATE_METRICS) -> Tuple[bool, str]:
    """
    The engine must beat every clean point-in-time baseline on every gated
    metric. The leaking FPL column is deliberately excluded - it is not a fair
    target.
    """
    if isinstance(metrics, str):
        metrics = (metrics,)
    models = summary.get("models", {})
    fepl = models.get("fepl")
    if not fepl:
        return False, "no FEPL results produced"

    failures, passes = [], []
    for metric in metrics:
        if metric not in LOWER_IS_BETTER:
            failures.append(f"unknown gate metric {metric!r}")
            continue
        beaten = []
        for name in CLEAN_BASELINES:
            b = models.get(name)
            if not b or metric not in b:
                continue
            if not _compare(fepl, b, metric):
                beaten.append(f"{name} {b[metric]:.3f}")
        if beaten:
            failures.append(
                f"{metric} (FEPL {fepl[metric]:.3f} vs " + ", ".join(beaten) + ")")
        else:
            passes.append(f"{metric} {fepl[metric]:.3f}")

    if failures:
        return False, "FEPL failed to beat baselines on: " + "; ".join(failures)
    return True, "FEPL beats all clean baselines on " + ", ".join(passes)


def advisory_report(summary: Dict[str, Any]) -> List[str]:
    """Non-gating checks, printed so a regression is visible without failing CI."""
    models = summary.get("models", {})
    fepl = models.get("fepl")
    if not fepl:
        return []
    out = []
    for metric in ADVISORY_GATE_METRICS:
        if metric not in fepl:
            continue
        worse = [f"{n} {models[n][metric]:.3f}" for n in CLEAN_BASELINES
                 if n in models and metric in models[n]
                 and not _compare(fepl, models[n], metric)]
        if worse:
            out.append(f"[WARN] {metric}: FEPL {fepl[metric]:.3f} is no better than "
                       + ", ".join(worse))
        else:
            out.append(f"[ok]   {metric}: FEPL {fepl[metric]:.3f} beats every baseline")
    return out


# What each season's data can and cannot support. Pooling seasons that measure
# different models is worse than not pooling at all, so the differences are
# named here rather than discovered later in a confusing average.
#
#   2020-21, 2021-22  no expected_goals / expected_assists at all. The whole
#                     attacking model falls back to positional priors, so these
#                     measure something that is not this engine.
#   2022-23..2024-25  no CBIT / tackles / recoveries columns, and FPL awarded no
#                     defensive contribution points either. The scoring gate in
#                     `xp_model` turns DefCon off for them, so they are
#                     comparable on everything except DefCon.
#   2025-26           full coverage.
FIRST_SEASON_WITH_XG = 2022


def season_caveats(season_str: str) -> List[str]:
    """Human-readable warnings about what a season's data cannot measure."""
    from match_sim import defcon_active

    start = int(season_str.split("-")[0])
    out = []
    if start < FIRST_SEASON_WITH_XG:
        out.append(
            "no expected_goals/expected_assists in this season's data - the "
            "attacking model runs on positional priors alone. Results are NOT "
            "comparable with later seasons and should not be pooled."
        )
    if not defcon_active(start):
        out.append(
            "FPL awarded no defensive contribution points this season; DefCon "
            "scoring is switched off so the model is scored under the rules "
            "actually in force."
        )
    return out


def _load_prior_season(season_str: str, choice: str):
    """`history_past` for one season: its own predecessor unless told otherwise."""
    if choice == "none":
        return None
    if choice == "auto":
        start = int(season_str.split("-")[0]) - 1
        choice = f"{start}-{str(start + 1)[2:]}"
    try:
        df = fetch_data(choice)[0]
        logger.info("Loaded %s as history_past for %s.", choice, season_str)
        return df
    except Exception as e:
        logger.warning(
            "Could not load %s as history_past for %s (%s). That season is "
            "measured WITHOUT the prior-season priors production has, so its "
            "results understate the model.", choice, season_str, e,
        )
        return None


def pool_gameweeks(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every scored gameweek across every season, as one sample."""
    return [g for s in summaries for g in s.get("per_gameweek", [])]


def paired_ci(gws: List[Dict[str, Any]], model: str, baseline: str, metric: str,
              n_boot: int = 20000, seed: int = 0) -> Optional[Dict[str, Any]]:
    """
    Bootstrap CI for (model - baseline) on one metric, paired by gameweek.

    Pairing matters: gameweeks differ enormously in how predictable they are, and
    comparing unpaired means drowns the difference in that shared variation.
    """
    rows = [(g["models"][model][metric], g["models"][baseline][metric])
            for g in gws if model in g["models"] and baseline in g["models"]]
    if len(rows) < 3:
        return None
    d = np.array([a - b for a, b in rows], dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": len(d), "diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "significant": bool((lo > 0) == (hi > 0))}


def _print_pooled(summaries: List[Dict[str, Any]]):
    """
    The whole reason to run more than one season: enough gameweeks to say which
    differences are real. precision@15 and captain regret both have confidence
    intervals spanning zero on a single season.
    """
    gws = pool_gameweeks(summaries)
    if not gws:
        return
    seasons = ", ".join(s["season"] for s in summaries)
    print()
    print("=" * 86)
    print(f" POOLED ACROSS SEASONS  ({seasons}; {len(gws)} gameweeks)")
    print("=" * 86)
    names = sorted({k for g in gws for k in g["models"]})
    agg = {}
    for name in names:
        rows = [g["models"][name] for g in gws if name in g["models"]]
        agg[name] = {m: float(np.mean([r[m] for r in rows]))
                     for m in ("mae", "rmse", "bias", "spearman",
                               "precision_at_15", "captain_regret")}
    print(f" {'model':<16}{'MAE':>8}{'RMSE':>8}{'bias':>8}{'rho':>8}"
          f"{'P@15':>8}{'cap.regret':>12}")
    print(" " + "-" * 82)
    for name, m in sorted(agg.items(), key=lambda kv: kv[1]["mae"]):
        flag = "  <- LEAKS" if "LEAK" in name else ""
        print(f" {name:<16}{m['mae']:>8.3f}{m['rmse']:>8.3f}{m['bias']:>+8.3f}"
              f"{m['spearman']:>8.3f}{m['precision_at_15']:>8.1f}"
              f"{m['captain_regret']:>12.2f}{flag}")

    print()
    print(" FEPL vs each clean baseline, paired by gameweek (95% CI of difference):")
    print(f"   {'metric':<17}{'baseline':<12}{'diff':>9}{'95% CI':>22}")
    for metric in ("mae", "spearman", "precision_at_15", "captain_regret"):
        for base in CLEAN_BASELINES:
            ci = paired_ci(gws, "fepl", base, metric)
            if not ci:
                continue
            span = f"[{ci['lo']:+.3f}, {ci['hi']:+.3f}]"
            mark = "significant" if ci["significant"] else "noise"
            print(f"   {metric:<17}{base:<12}{ci['diff']:>+9.3f}{span:>22}  {mark}")
    print("=" * 86)


def main():
    from xp_model import load_lineup_overrides

    ap = argparse.ArgumentParser(description="FEPL walk-forward backtest")
    ap.add_argument("--seasons", nargs="+", default=["2025-26"],
                    help="Seasons to test, e.g. 2023-24 2024-25 2025-26")
    ap.add_argument("--from-gw", type=int, default=5)
    ap.add_argument("--to-gw", type=int, default=None)
    # Calibration is OFF by default: measured on 2025-26 GW8-16 it made MAE
    # slightly worse (1.917 vs 1.906) while costing ~40% of runtime. See
    # calibration.py. The flag is kept so the decision stays re-measurable.
    ap.add_argument("--calibrate", action="store_true",
                    help="Enable per-position recalibration (off by default)")
    ap.add_argument("--calibration-method", default="linear",
                    choices=["linear", "isotonic", "none"])
    ap.add_argument("--risk-aversion", type=float, default=0.0)
    ap.add_argument("--prior-season", default="auto",
                    help="Season to load as history_past: 'auto' (previous season), "
                         "an explicit season like 2024-25, or 'none'")
    ap.add_argument("--gate", action="store_true",
                    help="Exit non-zero unless FEPL beats every clean baseline")
    ap.add_argument("--gate-metric", nargs="+", default=list(DEFAULT_GATE_METRICS),
                    choices=sorted(LOWER_IS_BETTER),
                    help="Metrics the gate enforces (default: error plus the "
                         "decision metrics, not MAE alone)")
    ap.add_argument("--lineups", default="",
                    help="Path to a predicted-lineups JSON feed (see "
                         "xp_model.load_lineup_overrides). Normally absent in a "
                         "backtest: no such feed exists retrospectively.")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    # Production always has `history_past` from the API, so a backtest without it
    # measures a weaker model than the one that actually runs. Load the previous
    # season by default and say plainly when we could not.
    all_summaries = []
    for season in args.seasons:
        # Each season needs *its own* predecessor. Resolving the prior once from
        # seasons[0] and reusing it gave 2025-26 the 2022-23 priors on any
        # multi-season run, which is worse than having none.
        prior_gw = _load_prior_season(season, args.prior_season)
        for warning in season_caveats(season):
            logger.warning("%s: %s", season, warning)
        gws = list(range(args.from_gw, (args.to_gw or 38) + 1))
        s = run_backtest(
            season_str=season, test_gws=gws,
            prior_season_gw=prior_gw,
            calibrate=args.calibrate, risk_aversion=args.risk_aversion,
            calibration_method=args.calibration_method,
            lineup_overrides=load_lineup_overrides(args.lineups or None),
        )
        all_summaries.append(s)

    if len(all_summaries) > 1:
        _print_pooled(all_summaries)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_summaries, f, indent=2, default=float)
        print(f"\nWrote {args.json_out}")

    if args.gate:
        failed = False
        for s in all_summaries:
            ok, msg = check_gate(s, args.gate_metric)
            print(f"[{'PASS' if ok else 'FAIL'}] {s['season']}: {msg}")
            for line in advisory_report(s):
                print(f"        {line}")
            failed = failed or not ok
        if len(all_summaries) > 1:
            # A season is a small sample and one bad one should not sink the
            # build on its own, but the pooled result is the real verdict.
            pooled = summarise(pool_gameweeks(all_summaries), "pooled", verbose=False)
            ok, msg = check_gate(pooled, args.gate_metric)
            print(f"[{'PASS' if ok else 'FAIL'}] pooled: {msg}")
            for line in advisory_report(pooled):
                print(f"        {line}")
            failed = failed or not ok
        raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
