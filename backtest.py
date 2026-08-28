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

# Vaastav GitHub URLs for 2023-24 Season
BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24"
MERGED_GW_URL = f"{BASE_URL}/gws/merged_gw.csv"
PLAYERS_RAW_URL = f"{BASE_URL}/players_raw.csv"
TEAMS_URL = f"{BASE_URL}/teams.csv"
FIXTURES_URL = f"{BASE_URL}/fixtures.csv"

def fetch_data():
    logger.info("Downloading historical Vaastav data (this will take ~10 seconds)...")
    df_gw = pd.read_csv(MERGED_GW_URL, low_memory=False)
    df_players = pd.read_csv(PLAYERS_RAW_URL, low_memory=False)
    df_teams = pd.read_csv(TEAMS_URL, low_memory=False)
    df_fixtures = pd.read_csv(FIXTURES_URL, low_memory=False)
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

    elements = []
    for p in df_players.to_dict(orient="records"):
        pid = int(p.get("id"))
        stats = player_stats.get(pid, {"xg90": 0.0, "xa90": 0.0, "ppg": 0.0, "val": float(p.get("now_cost", 50))})
        
        element = {
            "id": pid,
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
                "fixture_difficulty": 3
            }
            history.append(h)
        all_history[int(pid)] = history

    for e in elements:
        all_history.setdefault(e["id"], [])

    return bootstrap, fixtures, all_history

def run_backtest(weights: tuple = None, df_gw=None, df_players=None, df_teams=None, df_fixtures=None) -> float:
    if df_gw is None:
        logger.info("Initializing Backtester...")
        try:
            df_gw, df_players, df_teams, df_fixtures = fetch_data()
        except Exception as e:
            logger.error(f"Failed to fetch Vaastav data: {e}")
            return 999.0

    TEST_GWS = range(15, 21)
    
    total_error = 0.0
    total_baseline_error = 0.0
    total_predictions = 0

    if not weights:
        logger.info(f"Starting Walk-Forward Time Machine from GW {TEST_GWS.start} to {TEST_GWS.stop - 1}")
    
    for target_gw in TEST_GWS:
        if not weights:
            logger.info(f"\n--- Backtesting Gameweek {target_gw} ---")
        
        bootstrap, fixtures, all_history = build_mock_api(df_gw, df_players, df_teams, df_fixtures, current_gw=target_gw)
        xp_matrix = generate_xp_matrix([target_gw], bootstrap=bootstrap, fixtures=fixtures, all_history=all_history, weights=weights, season=2023)
        
        df_target = df_gw[df_gw['GW'] == target_gw]
        
        # E-03: Use groupby to properly sum double gameweeks instead of overwriting the dict
        actuals = df_target.groupby('element')['total_points'].sum().to_dict()
        minutes_played = df_target.groupby('element')['minutes'].sum().to_dict()
        
        gw_dev = 0.0
        gw_baseline_dev = 0.0
        gw_count = 0
        
        for pid, points in actuals.items():
            history = all_history.get(pid, [])
            baseline_xp = sum([float(h.get("total_points", 0)) for h in history[-5:]]) / min(5, max(1, len(history))) if history else 2.0
            
            predicted_xp = xp_matrix.get(pid, {}).get(target_gw, 0.0)
            
            # Use Poisson Deviance instead of Absolute Error
            dev = max(0.0, 2 * (points * math.log(points / max(1e-4, predicted_xp)) - (points - predicted_xp)) if points > 0 else 2 * predicted_xp)
            b_dev = max(0.0, 2 * (points * math.log(points / max(1e-4, baseline_xp)) - (points - baseline_xp)) if points > 0 else 2 * baseline_xp)
            
            gw_dev += dev
            gw_baseline_dev += b_dev
            gw_count += 1
                
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
        if not weights:
            logger.info(f"\n==========================================")
            logger.info(f"FINAL BACKTEST DEVIANCE: {final_dev:.2f} (Baseline: {final_b_dev:.2f})")
            logger.info(f"==========================================")
        return final_dev
    return 999.0

if __name__ == "__main__":
    run_backtest()
