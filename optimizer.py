import pulp
import logging
from typing import Dict, Any, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

POS_GKP = 1
POS_DEF = 2
POS_MID = 3
POS_FWD = 4

# Expected points several gameweeks out are not worth the same as this week's.
# Injuries, rotation, form, price changes and fixture reschedules all accumulate,
# and beyond that the plan will simply be re-solved next week with better
# information -- so a distant gameweek is a hint about direction, not a
# commitment. Weighting every gameweek equally made the ILP trade a real point
# now for a speculative point in GW+4 at par, which over-commits the squad to
# fixtures that have not been priced yet.
HORIZON_DECAY = 0.86

# Bench value is not one number. An outfield sub scores when a starter in the
# same position blanks, which is common; the backup keeper only ever plays if
# the first-choice keeper does not, which is rare and already priced by owning
# him at all. A flat weight over-valued the bench keeper and under-valued the
# first outfield sub.
BENCH_WEIGHT_BY_POSITION = {POS_GKP: 0.02, POS_DEF: 0.12, POS_MID: 0.12, POS_FWD: 0.10}

def solve_fpl_optimization(
    bootstrap: Dict[str, Any],
    xp_matrix: Dict[int, Dict[int, float]],
    horizon_gws: List[int],
    initial_squad_ids: Optional[List[int]] = None,
    initial_bank: float = 0.0,
    initial_ft: int = 1,
    initial_sell_prices: Optional[Dict[int, float]] = None,
    locked_player_ids: Optional[List[int]] = None,
    banned_player_ids: Optional[List[int]] = None,
    max_hits_per_gw: int = 2,
    bench_weight: Optional[float] = None,
    active_chip: Optional[str] = None,
    active_chip_gw: Optional[int] = None,
    horizon_decay: float = HORIZON_DECAY,
) -> Dict[str, Any]:
    """
    Solve multi-period FPL squad selection, transfer optimization, and chip strategies using PuLP ILP solver.
    Supported chips: 'wc' (Wildcard), 'fh' (Free Hit), 'tc' (Triple Captain), 'bb' (Bench Boost).

    `horizon_decay` discounts each gameweek past the first by that factor, so a
    forecast four weeks out is worth roughly half a forecast for this week. Pass
    1.0 to weight the whole horizon equally (the previous behaviour).
    `bench_weight` overrides the per-position bench values with a single number.
    """
    elements = bootstrap.get("elements", [])
    teams = bootstrap.get("teams", [])
    
    player_dict = {p["id"]: p for p in elements}
    player_ids = list(player_dict.keys())
    
    locked_set = set(locked_player_ids or [])
    banned_set = set(banned_player_ids or [])
    
    # Prune players to top 30 per position by total XP over the horizon + top 10 cheapest to dramatically speed up solver
    if len(horizon_gws) > 0:
        total_xp_map = {pid: sum(xp_matrix.get(pid, {}).get(gw, 0.0) for gw in horizon_gws) for pid in player_ids}
        must_keep = set(initial_squad_ids or []) | locked_set
        pruned_ids = set()
        for pos in [POS_GKP, POS_DEF, POS_MID, POS_FWD]:
            pos_players = [pid for pid in player_ids if player_dict[pid]["element_type"] == pos]
            # Top by XP
            pos_players.sort(key=lambda pid: total_xp_map.get(pid, 0.0), reverse=True)
            pruned_ids.update(pos_players[:30])
            # Cheapest fodder
            pos_players.sort(key=lambda pid: player_dict[pid]["now_cost"])
            pruned_ids.update(pos_players[:10])
        pruned_ids.update(must_keep)
        player_ids = [pid for pid in player_ids if pid in pruned_ids]
    
    # Prices (FPL API prices are stored multiplied by 10 e.g. 100 = £10.0m)
    now_cost = {pid: player_dict[pid]["now_cost"] / 10.0 for pid in player_ids}
    
    # O-05: Selling price differs from now_cost if player was bought cheaper and rose
    sell_cost = {pid: now_cost[pid] for pid in player_ids}
    if initial_sell_prices:
        for pid, sp in initial_sell_prices.items():
            if pid in sell_cost:
                sell_cost[pid] = sp

    # Team mapping for constraints
    player_team = {pid: player_dict[pid]["team"] for pid in player_ids}
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
    s = {}
    x = {}
    c = {}
    vc = {}
    tin = {}
    tout = {}
    hits = {}
    ft_carried = {}
    bank = {}

    gws = sorted(horizon_gws)
    
    for t in gws:
        hits[t] = pulp.LpVariable(f"hits_{t}", lowBound=0, cat=pulp.LpInteger)
        # O-04: Free transfers bank up to 5
        ft_carried[t] = pulp.LpVariable(f"ft_carried_{t}", lowBound=0, upBound=5, cat=pulp.LpInteger)
        bank[t] = pulp.LpVariable(f"bank_{t}", lowBound=0, cat=pulp.LpContinuous)
        
        for pid in player_ids:
            s[pid, t] = pulp.LpVariable(f"s_{pid}_{t}", cat=pulp.LpBinary)
            x[pid, t] = pulp.LpVariable(f"x_{pid}_{t}", cat=pulp.LpBinary)
            c[pid, t] = pulp.LpVariable(f"c_{pid}_{t}", cat=pulp.LpBinary)
            vc[pid, t] = pulp.LpVariable(f"vc_{pid}_{t}", cat=pulp.LpBinary)
            tin[pid, t] = pulp.LpVariable(f"tin_{pid}_{t}", cat=pulp.LpBinary)
            tout[pid, t] = pulp.LpVariable(f"tout_{pid}_{t}", cat=pulp.LpBinary)

    # Objective Function incorporating Triple Captain & Bench Boost chips
    obj_terms = []

    for idx, t in enumerate(gws):
        chip_target_gw = active_chip_gw if active_chip_gw is not None else gws[0]
        is_chip_active_now = (active_chip is not None and t == chip_target_gw)
        tc_mult = 2.0 if (is_chip_active_now and active_chip == "tc") else 1.0
        # Confidence in a gameweek's forecast decays with how far away it is.
        decay = horizon_decay ** idx

        # The vice-captain scores only when the *captain* blanks. Who the captain
        # is, is itself a decision variable, so referencing their p_play directly
        # would make the objective bilinear. Instead we estimate P(captain blanks)
        # once per gameweek from the players actually in contention for the
        # armband - far better than the flat 0.05 this replaced, and still linear.
        contenders = sorted(
            player_ids, key=lambda q: xp_matrix.get(q, {}).get(t, 0.0), reverse=True
        )[:10]
        contender_p_play = [
            xp_matrix.get(q, {}).get(f"{t}_p_play", 1.0) for q in contenders
        ] or [1.0]
        p_cap_blank = max(0.0, min(0.5, 1.0 - max(contender_p_play)))

        for pid in player_ids:
            xp_val = xp_matrix.get(pid, {}).get(t, 0.0)
            
            # P(plays) from the engine, used to price the vice-captain fallback.
            p_play = xp_matrix.get(pid, {}).get(f"{t}_p_play", 1.0)
            if player_dict[pid].get("status") in ["i", "s", "u", "n"]:
                p_play = 0.0
            
            if is_chip_active_now and active_chip == "bb":
                b_weight = 1.0
            elif bench_weight is not None:
                b_weight = bench_weight
            else:
                b_weight = BENCH_WEIGHT_BY_POSITION.get(
                    player_dict[pid].get("element_type"), 0.10)

            starter_pts = x[pid, t] * xp_val
            bench_pts = (s[pid, t] - x[pid, t]) * xp_val * b_weight

            captain_pts = c[pid, t] * xp_val * tc_mult
            vc_pts = vc[pid, t] * xp_val * p_cap_blank * tc_mult

            obj_terms.append(decay * (starter_pts + bench_pts + captain_pts + vc_pts))

            # Terminal Squad Value (Add small incentive to hold squad value at end of horizon)
            if idx == len(gws) - 1:
                obj_terms.append(s[pid, t] * (now_cost[pid] * 0.01))

        # Subtract hit penalties (-4 points per hit, 0 if Wildcard/Free Hit chip
        # active). Discounted alongside the points they buy: a hit taken in a
        # later gameweek is as speculative as the gain it is chasing, and
        # charging it at full price while discounting the reward would make the
        # solver structurally refuse every future transfer.
        if is_chip_active_now and active_chip in ["wc", "fh"]:
            pass
        else:
            obj_terms.append(-4.0 * decay * hits[t])

    # Add terminal value for remaining free transfers at the end of the horizon (+1.5 expected points per FT)
    if len(gws) > 0:
        obj_terms.append(1.5 * ft_carried[gws[-1]])

    prob += pulp.lpSum(obj_terms), "Total_Expected_Points"

    # Initial state (if initial squad provided, e.g. from team import or GW1 wildcard)
    is_gw1_wildcard = (initial_squad_ids is None or len(initial_squad_ids) != 15)
    
    if is_gw1_wildcard:
        # Single wildcard setup for first GW in horizon
        first_gw = gws[0]
        # Budget cap £100.0m (fallback bank is 100.0, so use it directly)
        budget = initial_bank if initial_bank >= 90.0 else 100.0
        prob += bank[first_gw] == budget - pulp.lpSum([s[pid, first_gw] * now_cost[pid] for pid in player_ids]), f"Bank_{first_gw}"
        prob += ft_carried[first_gw] == 0, f"No_FT_Carry_Wildcard"
    else:
        # Pre-existing squad transition
        first_gw = gws[0]
        initial_set = set(initial_squad_ids)
        for pid in player_ids:
            in_initial = 1 if pid in initial_set else 0
            prob += s[pid, first_gw] == in_initial + tin[pid, first_gw] - tout[pid, first_gw], f"Init_Trans_{pid}_{first_gw}"
        
        # Initial Bank equation (O-05: Use sell_cost for transfers out)
        prob += bank[first_gw] == initial_bank + pulp.lpSum([tout[pid, first_gw] * sell_cost[pid] for pid in player_ids]) - pulp.lpSum([tin[pid, first_gw] * now_cost[pid] for pid in player_ids]), f"Bank_{first_gw}"
        
        # FT Rollover math for first GW
        if active_chip in ["wc", "fh"] and first_gw == (active_chip_gw if active_chip_gw is not None else gws[0]):
            # Free Hit and Wildcard allow unlimited transfers
            prob += ft_carried[first_gw] == 0, f"No_FT_Carry_Chip"
        else:
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

        # 4. Captain and Vice Captain
        prob += pulp.lpSum([c[pid, t] for pid in player_ids]) == 1, f"Captain_Count_{t}"
        prob += pulp.lpSum([vc[pid, t] for pid in player_ids]) == 1, f"ViceCaptain_Count_{t}"
        for pid in player_ids:
            prob += c[pid, t] <= x[pid, t], f"Captain_Is_Starter_{pid}_{t}"
            prob += vc[pid, t] <= x[pid, t], f"ViceCaptain_Is_Starter_{pid}_{t}"
            prob += c[pid, t] + vc[pid, t] <= 1, f"Captain_Not_VC_{pid}_{t}"

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
            
            if active_chip == "fh" and idx == 1:
                # FREE HIT REVERT: In the week after a Free Hit, the squad reverts to the original team
                for pid in player_ids:
                    in_initial = 1 if pid in set(initial_squad_ids or []) else 0
                    prob += s[pid, t] == in_initial + tin[pid, t] - tout[pid, t], f"Trans_{pid}_{t}"
                
                # Bank balance also reverts to initial (O-05)
                prob += bank[t] == initial_bank + pulp.lpSum([tout[pid, t] * sell_cost[pid] for pid in player_ids]) - pulp.lpSum([tin[pid, t] * now_cost[pid] for pid in player_ids]), f"Bank_Cont_{t}"
                
                # FT math operates normally, assuming 1 FT carried over from the FH week
                num_transfers = pulp.lpSum([tin[pid, t] for pid in player_ids])
                prob += num_transfers + ft_carried[t] <= initial_ft + hits[t], f"FT_Math_{t}"
            else:
                for pid in player_ids:
                    prob += s[pid, t] == s[pid, prev_t] + tin[pid, t] - tout[pid, t], f"Trans_{pid}_{t}"
                
                # Bank balance continuity (O-05)
                prob += bank[t] == bank[prev_t] + pulp.lpSum([tout[pid, t] * sell_cost[pid] for pid in player_ids]) - pulp.lpSum([tin[pid, t] * now_cost[pid] for pid in player_ids]), f"Bank_Cont_{t}"
                
                # FT Rollover math for subsequent GWs
                num_transfers = pulp.lpSum([tin[pid, t] for pid in player_ids])
                prob += num_transfers + ft_carried[t] <= 1 + ft_carried[prev_t] + hits[t], f"FT_Math_{t}"

        # Max hits limit per GW
        if idx == 0 and active_chip in ["wc", "fh"]:
            pass # No hit limit on Wildcard or Free Hit
        else:
            prob += hits[t] <= max_hits_per_gw, f"Max_Hits_{t}"

    # Solve the model using default PuLP solver (PULP_CBC_CMD)
    # Removing threads to prevent Windows CBC deadlocks, but keeping timeLimit at 300s to ensure true optimality.
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=300, gapRel=0.0)
    status = prob.solve(solver)
    status_str = pulp.LpStatus[status]

    if status != pulp.LpStatusOptimal:
        logger.error(f"Solver failed to find optimal solution: {status_str}")
        raise ValueError(f"Solver failed: {status_str}")

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
                vc_val = pulp.value(vc[pid, t]) or 0
                
                p_play = xp_matrix.get(pid, {}).get(f"{t}_p_play")
                if p_play is None:
                    chance_raw = player_dict[pid].get("chance_of_playing_next_round")
                    p_play = 1.0 if chance_raw is None else chance_raw / 100.0
                if player_dict[pid].get("status") in ["i", "s", "u", "n"]:
                    p_play = 0.0

                p_info = {
                    "id": pid,
                    "web_name": player_dict[pid]["web_name"],
                    "element_type": element_type[pid],
                    "team": team_id[pid],
                    "cost": now_cost[pid],
                    "xp": xp_matrix.get(pid, {}).get(t, 0.0),
                    "p_play": p_play,
                    "is_captain": bool(c_val > 0.5),
                    "is_vice_captain": bool(vc_val > 0.5)
                }
                
                if c_val > 0.5:
                    captain_id = pid

                if x_val > 0.5:
                    starters.append(p_info)
                else:
                    bench.append(p_info)

        transfers_in = [
            {"id": pid, "web_name": player_dict[pid]["web_name"], "cost": now_cost[pid]}
            for pid in player_ids if (pulp.value(tin[pid, t]) or 0) > 0.5
        ]
        transfers_out = [
            {"id": pid, "web_name": player_dict[pid]["web_name"], "cost": now_cost[pid]}
            for pid in player_ids if (pulp.value(tout[pid, t]) or 0) > 0.5
        ]

        # Sort bench by expected autosub value: p_play * xp
        # GKs are always subbed for GKs, so keep them separate or just sort everyone and let FPL formation rules apply
        # We will sort all bench players by p_play * xp descending
        bench.sort(key=lambda p: p["p_play"] * p["xp"], reverse=True)
        
        # Compute true expected points for the gameweek
        chip_gw = active_chip_gw if active_chip_gw is not None else gws[0]
        is_chip_active_now = (active_chip is not None and t == chip_gw)
        tc_mult = 2.0 if (is_chip_active_now and active_chip == "tc") else 1.0
        
        gw_xp = 0.0
        for p in starters:
            gw_xp += p["xp"]
            if p["is_captain"]:
                # Captain already accounted for 1x in starter, add the extra mult, discount by their p_play
                gw_xp += p["xp"] * tc_mult
            if p["is_vice_captain"]:
                cap_p_play = next((q["p_play"] for q in starters if q["is_captain"]), 1.0)
                gw_xp += p["xp"] * tc_mult * max(0.0, min(0.5, 1.0 - cap_p_play))
                
        # Add heuristic bench contribution to gw_xp if BB is active or using autosub weight
        bb_active = is_chip_active_now and active_chip == "bb"
        for p in bench:
            if bb_active:
                b_weight = 1.0
            elif bench_weight is not None:
                b_weight = bench_weight
            else:
                b_weight = BENCH_WEIGHT_BY_POSITION.get(p.get("element_type"), 0.10)
            gw_xp += p["xp"] * b_weight

        results["gameweeks"][t] = {
            "starters": starters,
            "bench": bench,
            "captain_id": captain_id,
            "gw_xp": round(gw_xp, 2),
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
