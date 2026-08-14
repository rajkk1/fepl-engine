"""
FPL Season Pre-Season Draft Optimizer Script
"""
import sys
import logging
import fpl_api
from xp_model import generate_xp_matrix
from optimizer import solve_fpl_optimization

# Configure UTF-8 encoding for terminal output
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    print("=" * 70)
    print(" FPL SEASON PRE-SEASON DRAFT OPTIMIZER -- GW1 INITIAL SQUAD ")
    print("=" * 70)
    print()

    bootstrap = fpl_api.get_bootstrap_static()
    fixtures = fpl_api.get_fixtures()
    teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
    
    # 1. Generate Expected Points Matrix across Horizon GW1 to GW5
    print("[1/2] Generating Ensemble xP Projections across GW1 to GW5...")
    horizon_gws = [1, 2, 3, 4, 5]
    xp_matrix = generate_xp_matrix(horizon_gws, bootstrap=bootstrap, fixtures=fixtures)

    # 2. Solve Integer Linear Program for Pre-Season £100.0m Budget
    print("[2/2] Solving PuLP Integer Linear Program for £100.0m Pre-Season Squad...\n")
    res = solve_fpl_optimization(
        bootstrap=bootstrap,
        xp_matrix=xp_matrix,
        horizon_gws=[1, 2, 3]
    )
    
    gw1_data = res.get("gameweeks", {}).get(1, {})
    starters = gw1_data.get("starters", [])
    bench = gw1_data.get("bench", [])
    total_xp = res.get("total_xp", 0.0)
    total_cost = res.get("total_cost", 0.0)

    print("=" * 70)
    print(f" OPTIMAL GW1 STARTING SQUAD (Total Projected Horizon xP: {total_xp:.2f} pts)")
    print("=" * 70)
    print()

    print("--- STARTING XI ---")
    for idx, p in enumerate(starters, 1):
        cap_str = " (C)" if p.get("is_captain") else ""
        t_code = teams.get(p.get("team"), "___")
        p_xp = p.get("xp", 0.0)
        p_cost = p.get("cost", 0.0)
        name = p.get("web_name", "Unknown")
        print(f" {idx:2d}. [{name:<16}] ({t_code}) £{p_cost:.1f}m | {p_xp:.2f} xP{cap_str}")

    print("\n--- BENCH ---")
    for idx, p in enumerate(bench, 12):
        t_code = teams.get(p.get("team"), "___")
        p_xp = p.get("xp", 0.0)
        p_cost = p.get("cost", 0.0)
        name = p.get("web_name", "Unknown")
        print(f" {idx:2d}. [{name:<16}] ({t_code}) £{p_cost:.1f}m | {p_xp:.2f} xP")

    print("\n" + "-" * 70)
    print(f" Total Squad Cost: £{total_cost:.1f}m | Remaining Bank: £{100.0 - total_cost:.1f}m")
    print("-" * 70)
    print()

    print("=" * 70)
    print(" 📋 MANUAL FPL TRANSFERS ENTRY CHEAT SHEET")
    print("=" * 70)
    for p in starters + bench:
        w_name = p.get("web_name", "")
        cost = p.get("cost", 0.0)
        t_code = teams.get(p.get("team"), "")
        role = "STARTING XI" if p in starters else "BENCH"
        print(f" - [{role:<11}] {w_name:<16} ({t_code}) £{cost:.1f}m")
    print("=" * 70)

if __name__ == "__main__":
    main()
