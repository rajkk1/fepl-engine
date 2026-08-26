import pulp
import logging
from typing import Dict, Any, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

POS_GKP = 1
POS_DEF = 2
POS_MID = 3
POS_FWD = 4

def solve_fpl_optimization(
    bootstrap: Dict[str, Any],
    xp_matrix: Dict[int, Dict[int, float]],
    horizon_gws: List[int],
    initial_squad_ids: Optional[List[int]] = None,
    initial_bank: float = 0.0,
    initial_ft: int = 1,
    locked_player_ids: Optional[List[int]] = None,
    banned_player_ids: Optional[List[int]] = None,
    max_hits_per_gw: int = 2,
    bench_weight: float = 0.10,
    active_chip: Optional[str] = None
) -> Dict[str, Any]:
    """
    Solve multi-period FPL squad selection, transfer optimization, and chip strategies using PuLP ILP solver.
    Supported chips: 'wc' (Wildcard), 'fh' (Free Hit), 'tc' (Triple Captain), 'bb' (Bench Boost).
    """
    elements = bootstrap.get("elements", [])
    teams = bootstrap.get("teams", [])
    
    player_dict = {p["id"]: p for p in elements}
    player_ids = list(player_dict.keys())
    
    locked_set = set(locked_player_ids or [])
    banned_set = set(banned_player_ids or [])
    
    # Prices (FPL API prices are stored multiplied by 10 e.g. 100 = £10.0m)
    now_cost = {pid: player_dict[pid]["now_cost"] / 10.0 for pid in player_ids}
    element_type = {pid: player_dict[pid]["element_type"] for pid in player_ids}
    team_id = {pid: player_dict[pid]["team"] for pid in player_ids}

    # Group players by position & club
    gkps = [pid for pid in player_ids if element_type[pid] == POS_GKP]
    defs = [pid for pid in player_ids if element_type[pid] == POS_DEF]
    mids = [pid for pid in player_ids if element_type[pid] == POS_MID]
    fwds = [pid for pid in player_ids if element_type[pid] == POS_FWD]
    
    club_players = {}
    for t in teams:
        club_id = t["id"]
        club_players[club_id] = [pid for pid in player_ids if team_id[pid] == club_id]

    prob = pulp.LpProblem("FPL_Multi_GW_Optimizer", pulp.LpMaximize)

    # Decision Variables across gameweeks
    # s[i, t]: player in squad
    # x[i, t]: player in starting XI
    # c[i, t]: captain
    # tin[i, t]: transferred IN
    # tout[i, t]: transferred OUT
    # hits[t]: hit points penalty at GW t
    s = {}
    x = {}
    c = {}
    tin = {}
    tout = {}
    hits = {}
    ft_carried = {}
    bank = {}

    gws = sorted(horizon_gws)
    
    for t in gws:
        hits[t] = pulp.LpVariable(f"hits_{t}", lowBound=0, cat=pulp.LpInteger)
        ft_carried[t] = pulp.LpVariable(f"ft_carried_{t}", lowBound=0, upBound=4, cat=pulp.LpInteger)
        bank[t] = pulp.LpVariable(f"bank_{t}", lowBound=0, cat=pulp.LpContinuous)
        
        for pid in player_ids:
            s[pid, t] = pulp.LpVariable(f"s_{pid}_{t}", cat=pulp.LpBinary)
            x[pid, t] = pulp.LpVariable(f"x_{pid}_{t}", cat=pulp.LpBinary)
            c[pid, t] = pulp.LpVariable(f"c_{pid}_{t}", cat=pulp.LpBinary)
            tin[pid, t] = pulp.LpVariable(f"tin_{pid}_{t}", cat=pulp.LpBinary)
            tout[pid, t] = pulp.LpVariable(f"tout_{pid}_{t}", cat=pulp.LpBinary)

    # Objective Function incorporating Triple Captain & Bench Boost chips
    obj_terms = []

    for idx, t in enumerate(gws):
        # Chips only apply to the current target gameweek (the first week in the horizon)
        is_chip_active_now = (active_chip is not None and idx == 0)
        
        tc_mult = 2.0 if (is_chip_active_now and active_chip == "tc") else 1.0  # Extra 2x for Triple Captain (total 3x)
        b_weight = 1.0 if (is_chip_active_now and active_chip == "bb") else bench_weight  # Full 1.0 weight for Bench Boost

        for pid in player_ids:
            xp_val = xp_matrix.get(pid, {}).get(t, 0.0)
            # Starter xP + Captain bonus (1x or 2x for TC) + Bench weight (0.10 or 1.0 for BB)
            obj_terms.append(x[pid, t] * xp_val)
            obj_terms.append(c[pid, t] * (xp_val * tc_mult))
            obj_terms.append((s[pid, t] - x[pid, t]) * (xp_val * b_weight))
        
        # Subtract hit penalties (-4 points per hit, 0 if Wildcard/Free Hit chip active)
        if is_chip_active_now and active_chip in ["wc", "fh"]:
            pass
        else:
            obj_terms.append(-4.0 * hits[t])

    prob += pulp.lpSum(obj_terms), "Total_Expected_Points"

    # Initial state (if initial squad provided, e.g. from team import or GW1 wildcard)
    is_gw1_wildcard = (initial_squad_ids is None or len(initial_squad_ids) != 15)
    
    if is_gw1_wildcard:
        # Single wildcard setup for first GW in horizon
        first_gw = gws[0]
        # Budget cap £100.0m
        prob += pulp.lpSum([s[pid, first_gw] * now_cost[pid] for pid in player_ids]) <= (100.0 + initial_bank), f"Initial_Budget_GW{first_gw}"
        prob += ft_carried[first_gw] == 0, f"No_FT_Carry_Wildcard"
    else:
        # Pre-existing squad transition
        first_gw = gws[0]
        initial_set = set(initial_squad_ids)
        for pid in player_ids:
            in_initial = 1 if pid in initial_set else 0
            prob += s[pid, first_gw] == in_initial + tin[pid, first_gw] - tout[pid, first_gw], f"Init_Trans_{pid}_{first_gw}"
        
        # Initial Bank equation
        prob += bank[first_gw] == initial_bank + pulp.lpSum([tout[pid, first_gw] * now_cost[pid] for pid in player_ids]) - pulp.lpSum([tin[pid, first_gw] * now_cost[pid] for pid in player_ids]), f"Bank_{first_gw}"
        
        # FT Rollover math for first GW
        num_transfers_first = pulp.lpSum([tin[pid, first_gw] for pid in player_ids])
        prob += num_transfers_first + ft_carried[first_gw] <= initial_ft + hits[first_gw], f"FT_Math_{first_gw}"

    # Constraints per Gameweek
    for idx, t in enumerate(gws):
        # 1. Squad Composition = 15 total
        prob += pulp.lpSum([s[pid, t] for pid in player_ids]) == 15, f"Squad_Size_{t}"
        prob += pulp.lpSum([s[pid, t] for pid in gkps]) == 2, f"GKP_Squad_{t}"
        prob += pulp.lpSum([s[pid, t] for pid in defs]) == 5, f"DEF_Squad_{t}"
        prob += pulp.lpSum([s[pid, t] for pid in mids]) == 5, f"MID_Squad_{t}"
        prob += pulp.lpSum([s[pid, t] for pid in fwds]) == 3, f"FWD_Squad_{t}"

        # 2. Max 3 players per club
        for club_id, p_list in club_players.items():
            prob += pulp.lpSum([s[pid, t] for pid in p_list]) <= 3, f"Club_Cap_{club_id}_{t}"

        # 3. Starting XI = 11 total
        prob += pulp.lpSum([x[pid, t] for pid in player_ids]) == 11, f"Starter_Size_{t}"
        prob += pulp.lpSum([x[pid, t] for pid in gkps]) == 1, f"GKP_Start_{t}"
        prob += pulp.lpSum([x[pid, t] for pid in defs]) >= 3, f"DEF_Start_Min_{t}"
        prob += pulp.lpSum([x[pid, t] for pid in defs]) <= 5, f"DEF_Start_Max_{t}"
        prob += pulp.lpSum([x[pid, t] for pid in mids]) >= 2, f"MID_Start_Min_{t}"
        prob += pulp.lpSum([x[pid, t] for pid in mids]) <= 5, f"MID_Start_Max_{t}"
        prob += pulp.lpSum([x[pid, t] for pid in fwds]) >= 1, f"FWD_Start_Min_{t}"
        prob += pulp.lpSum([x[pid, t] for pid in fwds]) <= 3, f"FWD_Start_Max_{t}"

        # Starters must be in squad
        for pid in player_ids:
            prob += x[pid, t] <= s[pid, t], f"Starter_In_Squad_{pid}_{t}"

        # 4. Captain
        prob += pulp.lpSum([c[pid, t] for pid in player_ids]) == 1, f"Captain_Count_{t}"
        for pid in player_ids:
            prob += c[pid, t] <= x[pid, t], f"Captain_Is_Starter_{pid}_{t}"

        # 5. User Locks and Bans
        for pid in locked_set:
            if pid in player_dict:
                prob += s[pid, t] == 1, f"Lock_{pid}_{t}"
        for pid in banned_set:
            if pid in player_dict:
                prob += s[pid, t] == 0, f"Ban_{pid}_{t}"

        # 6. Squad Transitions for subsequent GWs
        if idx > 0:
            prev_t = gws[idx - 1]
            for pid in player_ids:
                prob += s[pid, t] == s[pid, prev_t] + tin[pid, t] - tout[pid, t], f"Trans_{pid}_{t}"
            
            # Bank balance continuity
            prob += bank[t] == bank[prev_t] + pulp.lpSum([tout[pid, t] * now_cost[pid] for pid in player_ids]) - pulp.lpSum([tin[pid, t] * now_cost[pid] for pid in player_ids]), f"Bank_Cont_{t}"
            
            # FT Rollover math for subsequent GWs
            num_transfers = pulp.lpSum([tin[pid, t] for pid in player_ids])
            prob += num_transfers + ft_carried[t] <= 1 + ft_carried[prev_t] + hits[t], f"FT_Math_{t}"

        # Max hits limit per GW
        prob += hits[t] <= max_hits_per_gw, f"Max_Hits_{t}"

    # Solve the model using default PuLP solver (PULP_CBC_CMD)
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    status_str = pulp.LpStatus[status]

    if status != pulp.LpStatusOptimal:
        logger.warning(f"Solver completed with status: {status_str}")

    # Extract Optimization Solution Output
    results = {
        "status": status_str,
        "total_xp": round(pulp.value(prob.objective), 2),
        "gameweeks": {}
    }

    for t in gws:
        starters = []
        bench = []
        captain_id = None
        
        for pid in player_ids:
            s_val = pulp.value(s[pid, t]) or 0
            x_val = pulp.value(x[pid, t]) or 0
            c_val = pulp.value(c[pid, t]) or 0

            if s_val > 0.5:
                p_info = {
                    "id": pid,
                    "web_name": player_dict[pid]["web_name"],
                    "element_type": element_type[pid],
                    "team": team_id[pid],
                    "cost": now_cost[pid],
                    "xp": xp_matrix.get(pid, {}).get(t, 0.0)
                }
                
                if c_val > 0.5:
                    captain_id = pid
                    p_info["is_captain"] = True
                else:
                    p_info["is_captain"] = False

                if x_val > 0.5:
                    starters.append(p_info)
                else:
                    bench.append(p_info)

        # Assign Vice-Captain to the highest xP player who is not the Captain
        sorted_starters = sorted(starters, key=lambda item: item.get("xp", 0.0), reverse=True)
        if sorted_starters:
            top_vc = None
            for p in sorted_starters:
                if not p.get("is_captain"):
                    top_vc = p["id"]
                    break
            
            for p in starters:
                p["is_vice_captain"] = (p["id"] == top_vc)

        transfers_in = [
            {"id": pid, "web_name": player_dict[pid]["web_name"], "cost": now_cost[pid]}
            for pid in player_ids if (pulp.value(tin[pid, t]) or 0) > 0.5
        ]
        transfers_out = [
            {"id": pid, "web_name": player_dict[pid]["web_name"], "cost": now_cost[pid]}
            for pid in player_ids if (pulp.value(tout[pid, t]) or 0) > 0.5
        ]

        results["gameweeks"][t] = {
            "starters": starters,
            "bench": bench,
            "captain_id": captain_id,
            "transfers_in": transfers_in,
            "transfers_out": transfers_out,
            "hits": int(pulp.value(hits[t]) or 0),
            "bank": round(float(pulp.value(bank[t]) or 0.0), 2)
        }

    return results

if __name__ == "__main__":
    from xp_model import generate_xp_matrix
    from fpl_api import get_bootstrap_static, get_fixtures
    
    print("Testing PuLP ILP Optimization Engine...")
    bs = get_bootstrap_static()
    fx = get_fixtures()
    gws = [1, 2]
    xp_mat = generate_xp_matrix(gws, bs, fx)
    
    res = solve_fpl_optimization(bs, xp_mat, gws)
    print(f"Optimal Result Status: {res['status']}")
    print(f"Total Projected Horizon xP: {res['total_xp']} pts")
    for gw, gw_data in res["gameweeks"].items():
        print(f"\nGW{gw} Starters count: {len(gw_data['starters'])}, Bench count: {len(gw_data['bench'])}")
        print(f"Transfers IN: {[p['web_name'] for p in gw_data['transfers_in']]}")
