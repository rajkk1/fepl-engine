import pandas as pd
import numpy as np
import logging
import urllib.request
import json
import os
import math
from typing import Dict, Any, List
from xp_model import generate_xp_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def fetch_data(season_str="2023-24"):
    base_url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season_str}"
    logger.info(f"Downloading historical Vaastav data for {season_str} (this will take ~10 seconds)...")
    df_gw = pd.read_csv(f"{base_url}/gws/merged_gw.csv", low_memory=False)
    df_players = pd.read_csv(f"{base_url}/players_raw.csv", low_memory=False)
    df_teams = pd.read_csv(f"{base_url}/teams.csv", low_memory=False)
    df_fixtures = pd.read_csv(f"{base_url}/fixtures.csv", low_memory=False)
    return df_gw, df_players, df_teams, df_fixtures

def build_mock_api(df_gw, df_players, df_teams, df_fixtures, current_gw: int):
    # 1. Mock Teams
    teams = df_teams.to_dict(orient="records")
    
    # 2. Mock Fixtures
    fixtures = []
    for f in df_fixtures.to_dict(orient="records"):
        gw = f.get("event")
        if pd.isna(gw): continue
        is_finished = (gw < current_gw)
        
        fixture = {
            "event": int(gw),
            "team_h": int(f.get("team_h")),
            "team_a": int(f.get("team_a")),
            "team_h_score": int(f.get("team_h_score")) if is_finished and not pd.isna(f.get("team_h_score")) else None,
            "team_a_score": int(f.get("team_a_score")) if is_finished and not pd.isna(f.get("team_a_score")) else None,
            "finished": is_finished,
            "team_h_difficulty": int(f.get("team_h_difficulty", 3)),
            "team_a_difficulty": int(f.get("team_a_difficulty", 3))
        }
        fixtures.append(fixture)

    # 3. Mock Bootstrap Players (Strictly Point-In-Time)
    # V-01 Fix: Calculate per-90 stats from df_past to prevent leakage
    df_past = df_gw[df_gw['GW'] < current_gw]
    
    player_stats = {}
    for pid, group in df_past.groupby('element'):
        total_mins = group['minutes'].sum()
        total_xg = pd.to_numeric(group.get('expected_goals', 0), errors='coerce').sum()
        total_xa = pd.to_numeric(group.get('expected_assists', 0), errors='coerce').sum()
        total_pts = group['total_points'].sum()
        games = len(group)
        last_val = group['value'].iloc[-1] if not group.empty else 50
        
        xg90 = (total_xg / total_mins * 90) if total_mins > 0 else 0.0
        xa90 = (total_xa / total_mins * 90) if total_mins > 0 else 0.0
        ppg = (total_pts / games) if games > 0 else 0.0
        
        player_stats[int(pid)] = {
            "xg90": xg90,
            "xa90": xa90,
            "ppg": ppg,
            "val": float(last_val)
        }

    # Extract exact ownership percentage from the previous gameweek
    df_prev = df_gw[df_gw['GW'] == current_gw - 1]
    own_pct = {}
    if not df_prev.empty:
        sel = df_prev.groupby('element')['selected'].max()
        total_managers = sel.sum() / 15.0  # exact: 15 picks per manager
        if total_managers > 0:
            own_pct = (100.0 * sel / total_managers).to_dict()

    elements = []
    for p in df_players.to_dict(orient="records"):
        pid = int(p.get("id"))
        stats = player_stats.get(pid, {"xg90": 0.0, "xa90": 0.0, "ppg": 0.0, "val": float(p.get("now_cost", 50))})
        
        element = {
            "id": pid,
            "web_name": str(p.get("web_name", f"Player_{pid}")),
            "element_type": int(p.get("element_type", 3)),
            "team": int(p.get("team", 1)),
            "now_cost": stats["val"],
            "status": "a", 
            "chance_of_playing_next_round": 100,
            "form": 0.0,
            "minutes": 0,
            "expected_goals_per_90": stats["xg90"],
            "expected_assists_per_90": stats["xa90"],
            "points_per_game": stats["ppg"],
            "selected_by_percent": own_pct.get(pid, 0.0),
            "penalties_order": int(p.get("penalties_order")) if not pd.isna(p.get("penalties_order")) else None
        }
        elements.append(element)
        
    bootstrap = {
        "teams": teams,
        "elements": elements
    }
    
    # 4. Mock All History (Only up to current_gw - 1)
    df_past = df_gw[df_gw['GW'] < current_gw]
    all_history = {}
    
    for pid, group in df_past.groupby('element'):
        history = []
        for _, row in group.iterrows():
            h = {
                "round": int(row['GW']),
                "minutes": int(row['minutes']),
                "total_points": int(row['total_points']),
                "expected_goals": float(row.get('expected_goals', 0) or 0),
                "expected_assists": float(row.get('expected_assists', 0) or 0),
                "expected_goals_conceded": float(row.get('expected_goals_conceded', 0) or 0),
                "expected_goal_involvements": float(row.get('expected_goal_involvements', 0) or 0),
                "bps": float(row.get('bps', 0) or 0),
                "value": float(row.get('value', 50)),
                "transfers_balance": float(row.get('transfers_balance', 0) or 0),
                "was_home": bool(row.get('was_home', False)),
                "opponent_team": int(row.get('opponent_team', 1)),
                "starts": int(row.get('starts', 0)),
                "clearances_blocks_interceptions": int(row.get('clearances_blocks_interceptions', 0) or 0),
                "recoveries": int(row.get('recoveries', 0) or 0),
                "tackles": int(row.get('tackles', 0) or 0),
                "saves": int(row.get('saves', 0) or 0),
                "yellow_cards": int(row.get('yellow_cards', 0) or 0),
                "red_cards": int(row.get('red_cards', 0) or 0),
                "fixture_difficulty": 3
            }
            # Assert mock has fields
            assert "saves" in h and "yellow_cards" in h
            history.append(h)
        all_history[int(pid)] = history

    for e in elements:
        all_history.setdefault(e["id"], [])

    return bootstrap, fixtures, all_history

def run_backtest(weights: tuple = None, df_gw=None, df_players=None, df_teams=None, df_fixtures=None, season_str="2025-26") -> float:
    if df_gw is None:
        logger.info(f"Initializing Backtester for {season_str}...")
        try:
            df_gw, df_players, df_teams, df_fixtures = fetch_data(season_str)
        except Exception as e:
            logger.error(f"Failed to fetch Vaastav data: {e}")
            return 999.0

    TEST_GWS = range(5, 11)
    
    total_error = 0.0
    total_baseline_error = 0.0
    total_predictions = 0

    all_spearman = []
    all_baseline_spearman = []
    
    # Calibration and precision metrics
    all_actuals = []
    all_preds = []
    all_baseline_preds = []
    
    captain_hits = 0
    baseline_captain_hits = 0
    top15_hits = 0
    baseline_top15_hits = 0

    if not weights:
        logger.info(f"Starting Walk-Forward Time Machine from GW {TEST_GWS.start} to {TEST_GWS.stop - 1}")
    
    for target_gw in TEST_GWS:
        if not weights:
            logger.info(f"\n--- Backtesting Gameweek {target_gw} ---")
        
        bootstrap, fixtures, all_history = build_mock_api(df_gw, df_players, df_teams, df_fixtures, current_gw=target_gw)
        from xp_model import generate_merv_matrix
        season_int = int(season_str.split("-")[0])
        xp_matrix = generate_merv_matrix([target_gw], bootstrap=bootstrap, fixtures=fixtures, all_history=all_history, weights=weights, season=season_int)
        
        df_target = df_gw[df_gw['GW'] == target_gw]
        
        actuals = df_target.groupby('element')['total_points'].sum().to_dict()
        
        # guard the baseline
        valid = df_target.groupby('element')['xP'].sum()
        if (valid == 0).all():
            logger.warning(f"GW{target_gw}: xP column empty, skipping baseline")
            baseline_xp_map = None
        else:
            baseline_xp_map = df_target.groupby('element')['xP'].sum().to_dict()
        
        # Population Restriction
        df_prev = df_gw[df_gw['GW'] == target_gw - 1]
        top_selected = set(df_prev.groupby('element')['selected'].max().sort_values(ascending=False).head(150).index.tolist())
        
        player_dict = {p["id"]: p for p in bootstrap.get("elements", [])}
        player_ids = list(player_dict.keys())
        pruned_ids = set()
        
        for pos in [1, 2, 3, 4]:
            pos_players = [pid for pid in player_ids if player_dict.get(pid, {}).get("element_type") == pos]
            pos_players.sort(key=lambda pid: xp_matrix.get(pid, {}).get(target_gw, 0.0), reverse=True)
            pruned_ids.update(pos_players[:30])
            pos_players.sort(key=lambda pid: player_dict.get(pid, {}).get("now_cost", 1000))
            pruned_ids.update(pos_players[:10])
            
        pruned_ids.update(top_selected)
        valid_pids = [pid for pid in actuals.keys() if pid in pruned_ids]
        
        gw_dev = 0.0
        gw_baseline_dev = 0.0
        gw_count = 0
        
        # Positional Spearman tracking
        pos_actuals = {1: [], 2: [], 3: [], 4: []}
        pos_preds = {1: [], 2: [], 3: [], 4: []}
        pos_baseline = {1: [], 2: [], 3: [], 4: []}
        
        gw_ranked_actual = []
        gw_ranked_pred = []
        gw_ranked_baseline = []
        
        for pid in valid_pids:
            points = actuals[pid]
            predicted_xp = xp_matrix.get(pid, {}).get(target_gw, 0.0)
            
            if baseline_xp_map is None:
                baseline_xp = 0.0
            else:
                baseline_xp = baseline_xp_map.get(pid, 2.0)
            
            dev = max(0.0, 2 * (points * math.log(points / max(1e-4, predicted_xp)) - (points - predicted_xp)) if points > 0 else 2 * predicted_xp)
            b_dev = max(0.0, 2 * (points * math.log(points / max(1e-4, baseline_xp)) - (points - baseline_xp)) if points > 0 else 2 * baseline_xp)
            
            gw_dev += dev
            gw_baseline_dev += b_dev
            gw_count += 1
            
            pos = player_dict.get(pid, {}).get("element_type", 3)
            pos_actuals[pos].append(points)
            pos_preds[pos].append(predicted_xp)
            pos_baseline[pos].append(baseline_xp)
            
            gw_ranked_actual.append((pid, points))
            gw_ranked_pred.append((pid, predicted_xp))
            gw_ranked_baseline.append((pid, baseline_xp))
            
            all_actuals.append(points)
            all_preds.append(predicted_xp)
            all_baseline_preds.append(baseline_xp)
                
        # GW Metrics
        import scipy.stats as stats
        gw_spearman = 0.0
        gw_b_spearman = 0.0
        pos_count_s = 0
        pos_count_bs = 0
        for pos in [1, 2, 3, 4]:
            if len(pos_actuals[pos]) > 2:
                s, _ = stats.spearmanr(pos_actuals[pos], pos_preds[pos])
                bs, _ = stats.spearmanr(pos_actuals[pos], pos_baseline[pos])
                if not math.isnan(s):
                    gw_spearman += s
                    pos_count_s += 1
                if not math.isnan(bs) and baseline_xp_map is not None:
                    gw_b_spearman += bs
                    pos_count_bs += 1
        
        if pos_count_s > 0:
            all_spearman.append(gw_spearman / pos_count_s)
        if pos_count_bs > 0:
            all_baseline_spearman.append(gw_b_spearman / pos_count_bs)
            
        # Top-k Metrics
        gw_ranked_actual.sort(key=lambda x: x[1], reverse=True)
        gw_ranked_pred.sort(key=lambda x: x[1], reverse=True)
        gw_ranked_baseline.sort(key=lambda x: x[1], reverse=True)
        
        top1_actual = gw_ranked_actual[0][0] if gw_ranked_actual else None
        top15_actuals = set(p[0] for p in gw_ranked_actual[:15])
        
        top5_pred = set(p[0] for p in gw_ranked_pred[:5])
        top15_pred = set(p[0] for p in gw_ranked_pred[:15])
        
        top5_baseline = set(p[0] for p in gw_ranked_baseline[:5])
        top15_baseline = set(p[0] for p in gw_ranked_baseline[:15])
        
        if top1_actual in top5_pred: captain_hits += 1
        if top1_actual in top5_baseline: baseline_captain_hits += 1
        
        top15_hits += len(top15_actuals & top15_pred)
        baseline_top15_hits += len(top15_actuals & top15_baseline)
            
        if gw_count > 0:
            dev = gw_dev / gw_count
            b_dev = gw_baseline_dev / gw_count
            if not weights:
                logger.info(f"GW {target_gw} Deviance: {dev:.2f} (Baseline: {b_dev:.2f})")
            total_error += gw_dev
            total_baseline_error += gw_baseline_dev
            total_predictions += gw_count

    if total_predictions > 0:
        final_dev = total_error / total_predictions
        final_b_dev = total_baseline_error / total_predictions
        
        final_spearman = sum(all_spearman) / len(all_spearman) if all_spearman else 0.0
        final_b_spearman = sum(all_baseline_spearman) / len(all_baseline_spearman) if all_baseline_spearman else 0.0
        
        # Calibration (actual = m * pred + c)
        import numpy as np
        m, c = np.polyfit(all_preds, all_actuals, 1) if len(all_preds) > 1 else (0, 0)
        bm, bc = np.polyfit(all_baseline_preds, all_actuals, 1) if len(all_baseline_preds) > 1 else (0, 0)
        
        if not weights:
            logger.info(f"\n==========================================")
            logger.info(f"FINAL BACKTEST DEVIANCE: {final_dev:.2f} (Baseline: {final_b_dev:.2f})")
            logger.info(f"SPEARMAN RHO (Positional): {final_spearman:.3f} (Baseline: {final_b_spearman:.3f})")
            logger.info(f"CALIBRATION: actual = {m:.2f} * pred + {c:.2f} (Baseline: {bm:.2f} * pred + {bc:.2f})")
            logger.info(f"PRECISION@15: {top15_hits / len(TEST_GWS):.1f}/15 (Baseline: {baseline_top15_hits / len(TEST_GWS):.1f}/15)")
            logger.info(f"CAPTAIN STRIKE (Top 1 in Top 5): {captain_hits}/{len(TEST_GWS)} (Baseline: {baseline_captain_hits}/{len(TEST_GWS)})")
            logger.info(f"==========================================")
        
        # For grid search, we want to maximize Spearman or minimize Deviance
        # Let's return Deviance for the grid search optimization
        return final_dev
    return 999.0

if __name__ == "__main__":
    run_backtest()
