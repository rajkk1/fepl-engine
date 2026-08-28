import pandas as pd
import logging
import argparse
import time
from backtest import fetch_data, build_mock_api
from xp_model import generate_xp_matrix
from optimizer import solve_fpl_optimization

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def run_season_simulation(season_str="2024-25", horizon=5):
    logger.info(f"==========================================")
    logger.info(f"   STARTING FPL SEASON SIMULATOR ({season_str})")
    logger.info(f"==========================================")
    
    try:
        df_gw, df_players, df_teams, df_fixtures = fetch_data(season_str)
    except Exception as e:
        logger.error(f"Failed to fetch data for {season_str}: {e}")
        return
        
    season_int = int(season_str.split('-')[0])
    
    # Track Manager State
    bank = 100.0
    free_transfers = 0
    squad_ids = None
    sell_prices = {}
    
    total_points = 0
    total_hits = 0
    total_transfers = 0
    
    max_gw = df_gw['GW'].max()
    
    start_time = time.time()
    
    # Store points history
    history = []
    
    for current_gw in range(1, max_gw + 1):
        logger.info(f"\n[GW {current_gw}] Simulating Manager Decisions...")
        
        # 1. Build Point-in-Time Mock API
        bootstrap, fixtures, all_history = build_mock_api(df_gw, df_players, df_teams, df_fixtures, current_gw)
        
        # 2. Generate XP Matrix for Horizon (current_gw to current_gw + horizon - 1)
        horizon_gws = [gw for gw in range(current_gw, current_gw + horizon) if gw <= max_gw]
        
        xp_matrix = generate_xp_matrix(
            horizon_gws, 
            bootstrap=bootstrap, 
            fixtures=fixtures, 
            all_history=all_history, 
            season=season_int
        )
        
        # Compute actual sell prices for the optimizer
        actual_sell_prices = {}
        if squad_ids:
            for pid, buy_price in sell_prices.items():
                now_cost = 50
                for p_dict in bootstrap["elements"]:
                    if p_dict["id"] == pid:
                        now_cost = p_dict["now_cost"]
                        break
                if now_cost > buy_price:
                    actual_sell_prices[pid] = buy_price + (now_cost - buy_price) // 2
                else:
                    actual_sell_prices[pid] = now_cost
        
        # 3. Optimize Transfers
        # Initial bank is provided. If GW1, squad_ids is None (Wildcard)
        results = solve_fpl_optimization(
            bootstrap,
            xp_matrix,
            horizon_gws,
            initial_squad_ids=squad_ids,
            initial_bank=bank,
            initial_sell_prices=actual_sell_prices if squad_ids else {},
            initial_ft=free_transfers
        )
        
        if "gameweeks" not in results or current_gw not in results["gameweeks"]:
            logger.error(f"Solver failed to find a valid solution for GW {current_gw}!")
            break
            
        gw_plan = results["gameweeks"][current_gw]
        
        starters = [p["id"] for p in gw_plan["starters"]]
        bench = [p["id"] for p in gw_plan["bench"]]
        squad_ids = starters + bench
        
        captain_id = gw_plan["captain_id"]
        hits = gw_plan["hits"]
        
        vice_id = None
        for p in gw_plan["starters"]:
            if p.get("is_vice_captain"):
                vice_id = p["id"]
        if vice_id is None and len(gw_plan["starters"]) > 1:
            starter_objs = sorted(gw_plan["starters"], key=lambda p: p["xp"], reverse=True)
            vice_id = starter_objs[1]["id"] if starter_objs[0]["id"] == captain_id else starter_objs[0]["id"]
        
        # 4. Update Manager State for NEXT Gameweek
        # Actual hit costs are applied this gameweek
        total_hits += hits
        total_transfers += len(gw_plan["transfers_in"])
        
        # FT logic
        if current_gw == 1:
            free_transfers = 1
        else:
            free_transfers = max(1, min(5, free_transfers + 1 - len(gw_plan["transfers_in"])))
            
        # Update sell prices based on 50% profit rule
        # sell_cost = buy_price + (now_cost - buy_price) // 2
        for pid in squad_ids:
            now_cost = 50
            for p_dict in bootstrap["elements"]:
                if p_dict["id"] == pid:
                    now_cost = p_dict["now_cost"]
                    break
                    
            if pid not in sell_prices:
                # If newly bought (or GW1), buy price is now_cost
                sell_prices[pid] = now_cost
                
        # Remove sold players from sell_prices
        for pid in list(sell_prices.keys()):
            if pid not in squad_ids:
                del sell_prices[pid]
                
        # Construct the actual sell prices dictionary for the optimizer
        # (This is now done before the solver, we just update the bank here)
        bank = gw_plan["bank"]
        
        # 5. Evaluate ACTUAL Points
        df_target = df_gw[df_gw['GW'] == current_gw]
        actual_pts_map = df_target.groupby('element')['total_points'].sum().to_dict()
        actual_mins_map = df_target.groupby('element')['minutes'].sum().to_dict()
        
        gw_actual_points = 0
        
        # Autosub Logic
        played_starters = [pid for pid in starters if actual_mins_map.get(pid, 0) > 0]
        unplayed_starters = [pid for pid in starters if actual_mins_map.get(pid, 0) == 0]
        
        # Add points for played starters
        for pid in played_starters:
            pts = actual_pts_map.get(pid, 0)
            if pid == captain_id:
                pts *= 2
            # If captain didn't play and this is VC, double points
            if pid == vice_id and actual_mins_map.get(captain_id, 0) == 0:
                pts *= 2
            gw_actual_points += pts
            
        # Sub in bench players for unplayed starters (simplistic: just take highest scoring available bench players)
        available_bench = [pid for pid in bench if actual_mins_map.get(pid, 0) > 0]
        # Sort available bench by actual points scored to simulate optimal autosubs (or just by order)
        # We will just take up to len(unplayed_starters) from available_bench in order
        subs_used = 0
        for pid in available_bench:
            if subs_used < len(unplayed_starters):
                gw_actual_points += actual_pts_map.get(pid, 0)
                subs_used += 1
                
        # Subtract hits
        gw_net_points = gw_actual_points - (hits * 4)
        total_points += gw_net_points
        
        logger.info(f"   Captain: {captain_id} | Hits: {hits} | Net Points: {gw_net_points}")
        logger.info(f"   Season Total: {total_points}")
        
        # Safe extraction for squad value
        val = 0.0
        for pid in squad_ids:
            # pid is 1-indexed in bootstrap, so usually index is pid-1 but let's be safe
            for p_dict in bootstrap["elements"]:
                if p_dict["id"] == pid:
                    val += p_dict["now_cost"] / 10.0
                    break
                    
        history.append({
            "gw": current_gw,
            "gw_points": gw_actual_points,
            "hits": hits,
            "net_points": gw_net_points,
            "total_points": total_points,
            "bank": bank,
            "squad_value": val,
            "transfers": len(gw_plan["transfers_in"])
        })
        
    elapsed = time.time() - start_time
    logger.info(f"\n==========================================")
    logger.info(f"SIMULATION COMPLETE in {elapsed:.1f}s")
    logger.info(f"==========================================")
    logger.info(f"🏆 FINAL SCORE: {total_points} points")
    logger.info(f"🔄 Total Transfers: {total_transfers}")
    logger.info(f"💥 Total Hits Taken: {total_hits}")
    logger.info(f"==========================================")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=str, default="2024-25")
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()
    
    run_season_simulation(args.season, args.horizon)
