"""
FPL Weekly Manager CLI
Automates the analysis of your team and prints your weekly transfers, captaincy, and chip recommendations.
"""
import sys
import logging
import os
import argparse
import json
from dotenv import load_dotenv

import fpl_api
from xp_model import generate_xp_matrix
from optimizer import solve_fpl_optimization

# Configure UTF-8 encoding for terminal output
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def get_manager_team_state(team_id: int, current_gw: int):
    """Attempt to fetch manager's latest team state, bank balance, and selling prices."""
    try:
        # Public fallback first
        picks_data = fpl_api.get_manager_picks(team_id, current_gw - 1 if current_gw > 1 else 1)
        squad_ids = [p["element"] for p in picks_data.get("picks", [])]
        bank = picks_data.get("entry_history", {}).get("bank", 0) / 10.0
        
        sell_prices = {}
        # O-05: Track actual selling prices if FPL_COOKIE is available
        cookie = os.getenv("FPL_COOKIE")
        if cookie:
            try:
                my_team_data = fpl_api.get_my_team(team_id, cookie)
                for p in my_team_data.get("picks", []):
                    if "selling_price" in p:
                        sell_prices[p["element"]] = p["selling_price"] / 10.0
            except Exception as auth_err:
                logging.warning(f"Failed to fetch authenticated my-team endpoint: {auth_err}")
                
        ft = 100 if current_gw == 1 else 1
        return squad_ids, bank, ft, sell_prices
    except Exception:
        # Pre-season or no team found
        return None, 100.0, 100, {}

def print_separator(char="=", length=70):
    print(char * length)

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="FEPL Weekly Action Plan CLI")
    parser.add_argument("--team", type=int, default=4309239, help="Your FPL Team ID (can also be set via FPL_TEAM_ID env var)")
    parser.add_argument("--horizon", type=int, default=5, help="Planning horizon in gameweeks (default: 5)")
    parser.add_argument("--chip", type=str, default="", help="Chip to activate: wc, fh, tc, bb")
    parser.add_argument("--ft", type=int, default=None, help="Number of free transfers currently available. Defaults to 1.")
    parser.add_argument("--export-json", type=str, default="", help="Path to export the weekly plan as JSON (e.g., plan.json)")
    args = parser.parse_args()

    team_id = os.getenv("FPL_TEAM_ID") or args.team
    if not team_id:
        raise ValueError("Missing required FPL Team ID. Provide --team or set FPL_TEAM_ID in environment.")
    team_id = int(team_id)

    print_separator("=")
    print(f" 🎯 FEPL WEEKLY MANAGER (Team ID: {team_id})")
    print_separator("=")
    print()

    # Fetch foundational data
    try:
        bootstrap = fpl_api.get_bootstrap_static()
        fixtures = fpl_api.get_fixtures()
        elements = bootstrap.get("elements", [])
        player_dict = {p["id"]: p for p in elements}
        teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
    except Exception as e:
        print(f"Error fetching FPL API data: {e}")
        return

    # Determine current Gameweek
    current_gw = fpl_api.get_current_gameweek(bootstrap)
    print(f"🗓️  Current Target Gameweek: GW{current_gw}")
    
    # Get Manager State
    squad_ids, bank, default_ft, sell_prices = get_manager_team_state(team_id, current_gw)
    ft = args.ft if args.ft is not None else default_ft
    
    if squad_ids:
        print(f"💰 Current Bank: £{bank:.1f}m | Free Transfers: {ft if ft < 100 else 'Unlimited (Wildcard/Pre-season)'}")
    else:
        print(f"⚠️ No active squad found (Assuming Pre-season/Wildcard state)")
        ft = 100

    print()
    print(f"⏳ Generating Marginal Expected Rank Value (MERV) for GW{current_gw} to GW{current_gw + args.horizon - 1}...")
    horizon_gws = list(range(current_gw, current_gw + args.horizon))
    from xp_model import generate_merv_matrix
    xp_matrix = generate_merv_matrix(horizon_gws, bootstrap=bootstrap, fixtures=fixtures)

    active_chip = args.chip if args.chip else ""
    
    # #9. Automated Chip Comparison
    if not active_chip:
        print("🤖 Evaluating all possible Chip Strategies...")
        best_gain = 0.0
        best_chip = ""
        
        # 1. Get baseline (No Chip)
        base_res = solve_fpl_optimization(
            bootstrap=bootstrap, xp_matrix=xp_matrix, horizon_gws=horizon_gws, 
            initial_squad_ids=squad_ids, initial_bank=bank, initial_ft=ft, 
            initial_sell_prices=sell_prices,
            max_hits_per_gw=2, active_chip=None
        )
        base_xp = base_res.get("total_xp", 0.0)
        
        # 2. Test Chips using dynamic Value-Of-Waiting thresholds
        # FPL 26/27 rules: Two wildcards (GW19 expiry for WC1).
        gws_until_wc1_expiry = max(1, 19 - current_gw) if current_gw <= 19 else max(1, 38 - current_gw)
        gws_until_season_end = max(1, 38 - current_gw)
        
        chip_thresholds = {
            "tc": 10.0 * (gws_until_season_end / 38.0), 
            "bb": 12.0 * (gws_until_season_end / 38.0), 
            "fh": 15.0 * (gws_until_season_end / 38.0), 
            "wc": 20.0 * (gws_until_wc1_expiry / 19.0)
        }
        
        chip_results = {"": base_res}
        
        for c, threshold in chip_thresholds.items():
            c_res = solve_fpl_optimization(
                bootstrap=bootstrap, xp_matrix=xp_matrix, horizon_gws=horizon_gws, 
                initial_squad_ids=squad_ids, initial_bank=bank, initial_ft=ft, 
                initial_sell_prices=sell_prices,
                max_hits_per_gw=2, active_chip=c
            )
            c_xp = c_res.get("total_xp", 0.0)
            gain = c_xp - base_xp
            
            if gain > threshold and gain > best_gain:
                best_gain = gain
                best_chip = c
            
            chip_results[c] = c_res
            
        active_chip = best_chip
        res = chip_results[active_chip]
        if active_chip:
            print(f"🔥 Automatic Chip Activation: {active_chip.upper()} (Marginal Gain: +{best_gain:.2f} xP)\n")
        else:
            print(f"🧠 Solving ILP Optimization Model (Active Chip: None)...\n")
    else:
        print(f"🧠 Solving ILP Optimization Model (Active Chip: {active_chip})...\n")
        res = solve_fpl_optimization(
            bootstrap=bootstrap, xp_matrix=xp_matrix, horizon_gws=horizon_gws, 
            initial_squad_ids=squad_ids, initial_bank=bank, initial_ft=ft, 
            initial_sell_prices=sell_prices,
            max_hits_per_gw=2, active_chip=active_chip
        )

    if res.get("status") != "Optimal":
        print("⚠️ Warning: ILP Solver did not find a strictly optimal solution.")

    gw1_data = res.get("gameweeks", {}).get(current_gw, {})
    starters = gw1_data.get("starters", [])
    bench = gw1_data.get("bench", [])
    t_in = gw1_data.get("transfers_in", [])
    t_out = gw1_data.get("transfers_out", [])
    hits = gw1_data.get("hits", 0)

    # Enrich player info
    for p in starters + bench:
        p["now_cost"] = player_dict[p["id"]]["now_cost"] / 10.0
        p["status"] = player_dict[p["id"]]["status"]

    # Identify Captain and Vice-Captain
    captain = next((p for p in starters if p.get("is_captain")), starters[0] if starters else None)
    vice_captain = next((p for p in starters if p.get("is_vice_captain")), starters[1] if len(starters) > 1 else captain)

    print_separator("=")
    print(f" 🎯 WEEKLY ACTION PLAN FOR GW{current_gw}")
    print_separator("=")
    
    # Transfers
    print("\n🔄 RECOMMENDED TRANSFERS:")
    if not t_in:
        print("  ✓ Roll Free Transfer (No transfers recommended)")
    else:
        for i in range(len(t_in)):
            p_in = t_in[i]
            p_out = t_out[i] if i < len(t_out) else {"web_name": "Unknown", "cost": 0.0}
            print(f"  [IN]  {p_in['web_name']:<15} (£{p_in['cost']:.1f}m)")
            print(f"  [OUT] {p_out['web_name']:<15} (£{p_out['cost']:.1f}m)")
            print("  -")
    
    if hits > 0:
        print(f"  ⚠️ Taking a point hit penalty: -{hits * 4} pts")

    # Captaincy
    print("\n👑 CAPTAINCY:")
    if captain:
        mult = 3 if active_chip == "tc" else 2
        print(f"  (C)  {captain['web_name']:<15} -> {captain['xp']:.2f} xP ({captain['xp'] * mult:.2f} pts expected)")
    if vice_captain:
        print(f"  (VC) {vice_captain['web_name']:<15} -> {vice_captain['xp']:.2f} xP backup")

    # Chips
    print("\n🃏 CHIP STRATEGY:")
    if active_chip == "wc":
        print("  [✓] Play Wildcard (Unlimited Free Transfers active)")
    elif active_chip == "fh":
        print("  [✓] Play Free Hit (1-GW Squad active)")
    elif active_chip == "tc":
        print(f"  [✓] Play Triple Captain on {captain['web_name'] if captain else 'Captain'}")
    elif active_chip == "bb":
        print("  [✓] Play Bench Boost (All 15 players score points)")
    else:
        print("  [✓] Save Chips for Double Gameweeks")

    # Starting XI
    print("\n⚽ STARTING XI:")
    for idx, p in enumerate(starters, 1):
        cap_str = " (C)" if p.get("is_captain") else (" (VC)" if p.get("is_vice_captain") else "")
        inj_str = " 🏥" if p.get("status") not in ["a", None] else ""
        t_code = teams.get(p.get("team"), "___")
        print(f" {idx:2d}. [{p['web_name']:<16}] ({t_code}) £{p['now_cost']:.1f}m | {p['xp']:.2f} xP{cap_str}{inj_str}")

    # Bench Order - O-03: Sort bench by xP (priority 1-3) but keep GK first
    bench_gkps = [p for p in bench if p.get("element_type") == 1]
    bench_outfield = [p for p in bench if p.get("element_type") != 1]
    bench_outfield.sort(key=lambda item: item.get("xp", 0.0), reverse=True)
    sorted_bench = bench_gkps + bench_outfield

    print("\n🪑 BENCH (Order is strictly priority 1 to 3 after GK):")
    for idx, p in enumerate(sorted_bench, 1):
        pos_str = "GK" if idx == 1 else f"B{idx-1}"
        inj_str = " 🏥" if p.get("status") not in ["a", None] else ""
        t_code = teams.get(p.get("team"), "___")
        print(f" {pos_str}. [{p['web_name']:<16}] ({t_code}) £{p['now_cost']:.1f}m | {p['xp']:.2f} xP{inj_str}")

    print("\n" + "-" * 70)
    print(f" 📊 Expected GW{current_gw} Score: {res.get('gameweeks', {}).get(current_gw, {}).get('gw_xp', sum(p['xp'] for p in starters)):.2f} pts")
    print(f" 💰 Remaining Bank: £{gw1_data.get('bank', 0.0):.1f}m")
    print("-" * 70)

    # Export to JSON if flag is set
    if args.export_json:
        export_data = {
            "gameweek": current_gw,
            "team_id": team_id,
            "transfers_in": t_in,
            "transfers_out": t_out,
            "hits": hits,
            "active_chip": active_chip,
            "captain": captain,
            "vice_captain": vice_captain,
            "starters": starters,
            "bench": sorted_bench,
            "expected_score": res.get("gameweeks", {}).get(current_gw, {}).get("gw_xp", sum(p["xp"] for p in starters)),
            "remaining_bank": gw1_data.get("bank", 0.0)
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.export_json)), exist_ok=True)
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=4)
        print(f"\n✅ JSON output saved to {args.export_json}")

if __name__ == "__main__":
    main()
