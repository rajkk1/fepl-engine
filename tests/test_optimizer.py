import pytest
from optimizer import solve_fpl_optimization

@pytest.fixture
def base_bootstrap():
    return {
        "elements": [
            {
                "id": i, 
                "web_name": f"P{i}", 
                # 1-2=GK, 3-7=DEF, 8-12=MID, 13-15=FWD, 16-17=FWD, 18-19=MID, 20-30=DEF
                "element_type": 1 if i<=2 else 2 if (i<=7 or i>=20) else 3 if (i<=12 or i>=18) else 4, 
                "team": (i%15)+1, 
                "now_cost": 50
            }
            for i in range(1, 31)
        ],
        "teams": [{"id": i} for i in range(1, 16)]
    }

def test_wildcard_allows_unlimited_transfers(base_bootstrap):
    initial_squad_ids = list(range(1, 16))
    
    xp_matrix = {}
    for pid in range(1, 31):
        if pid <= 15:
            xp_matrix[pid] = {1: 2.0}
        else:
            xp_matrix[pid] = {1: 20.0}

    res_base = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp_matrix, horizon_gws=[1],
        initial_squad_ids=initial_squad_ids, initial_bank=10.0, initial_ft=1,
        max_hits_per_gw=2, active_chip=None
    )
    transfers_made_base = len(res_base["gameweeks"][1]["transfers_in"])
    assert transfers_made_base <= 3

    res_wc = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp_matrix, horizon_gws=[1],
        initial_squad_ids=initial_squad_ids, initial_bank=10.0, initial_ft=1,
        max_hits_per_gw=2, active_chip="wc"
    )
    transfers_made_wc = len(res_wc["gameweeks"][1]["transfers_in"])
    assert transfers_made_wc > 3

def test_free_hit_revert(base_bootstrap):
    initial_squad_ids = list(range(1, 16))
    
    xp_matrix = {}
    for pid in range(1, 31):
        # High score in GW1 for replacements, low in GW2
        # High score in GW2 for initial squad
        if pid <= 15:
            xp_matrix[pid] = {1: 1.0, 2: 20.0}
        else:
            xp_matrix[pid] = {1: 20.0, 2: 1.0}

    res_fh = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp_matrix, horizon_gws=[1, 2],
        initial_squad_ids=initial_squad_ids, initial_bank=10.0, initial_ft=1,
        max_hits_per_gw=2, active_chip="fh"
    )
    
    gw1_squad = set(p["id"] for p in res_fh["gameweeks"][1]["starters"] + res_fh["gameweeks"][1]["bench"])
    assert len(gw1_squad.intersection(initial_squad_ids)) < 15
    
    gw2_squad = set(p["id"] for p in res_fh["gameweeks"][2]["starters"] + res_fh["gameweeks"][2]["bench"])
    assert len(gw2_squad.intersection(initial_squad_ids)) >= 13  # Allowing for up to 2 hits

def test_budget_constraints(base_bootstrap):
    initial_squad_ids = list(range(1, 16))
    
    xp_matrix = {pid: {1: 2.0} for pid in range(1, 16)}
    # Add a premium player who costs too much
    base_bootstrap["elements"].append({
        "id": 99, "web_name": "Premium", "element_type": 4, "team": 1, "now_cost": 150
    })
    xp_matrix[99] = {1: 100.0}

    res = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp_matrix, horizon_gws=[1],
        initial_squad_ids=initial_squad_ids, initial_bank=0.0, initial_ft=1,
        max_hits_per_gw=2, active_chip=None
    )
    
    # Premium player should NOT be in the squad because of budget constraints
    gw1_squad = set(p["id"] for p in res["gameweeks"][1]["starters"] + res["gameweeks"][1]["bench"])
    assert 99 not in gw1_squad

def test_bench_boost_active(base_bootstrap):
    initial_squad_ids = list(range(1, 16))
    
    xp_matrix = {pid: {1: 10.0} for pid in range(1, 16)}

    # Without BB, only the starting 11 generate points
    res_base = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp_matrix, horizon_gws=[1],
        initial_squad_ids=initial_squad_ids, initial_bank=10.0, initial_ft=1,
        max_hits_per_gw=0, active_chip=None
    )
    
    # With BB, all 15 generate points
    res_bb = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp_matrix, horizon_gws=[1],
        initial_squad_ids=initial_squad_ids, initial_bank=10.0, initial_ft=1,
        max_hits_per_gw=0, active_chip="bb"
    )
    
    assert res_bb["gameweeks"][1]["gw_xp"] > res_base["gameweeks"][1]["gw_xp"]
