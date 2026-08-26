import pytest
from optimizer import solve_fpl_optimization

def test_wildcard_allows_unlimited_transfers():
    bootstrap = {
        "elements": [
            {
                "id": i, 
                "web_name": f"P{i}", 
                # 1-2=GK, 3-7=DEF, 8-12=MID, 13-15=FWD, 16-17=FWD, 18-19=MID
                "element_type": 1 if i<=2 else 2 if i<=7 else 3 if (i<=12 or i>=18) else 4, 
                "team": (i%7)+1, 
                "now_cost": 50
            }
            for i in range(1, 20)
        ],
        "teams": [{"id": i} for i in range(1, 8)]
    }

    initial_squad_ids = list(range(1, 16))
    
    xp_matrix = {}
    for pid in range(1, 20):
        if pid <= 15:
            xp_matrix[pid] = {1: 2.0}
        else:
            xp_matrix[pid] = {1: 20.0}

    res_base = solve_fpl_optimization(
        bootstrap=bootstrap, xp_matrix=xp_matrix, horizon_gws=[1],
        initial_squad_ids=initial_squad_ids, initial_bank=10.0, initial_ft=1,
        max_hits_per_gw=2, active_chip=None
    )
    transfers_made_base = len(res_base["gameweeks"][1]["transfers_in"])
    assert transfers_made_base <= 3

    res_wc = solve_fpl_optimization(
        bootstrap=bootstrap, xp_matrix=xp_matrix, horizon_gws=[1],
        initial_squad_ids=initial_squad_ids, initial_bank=10.0, initial_ft=1,
        max_hits_per_gw=2, active_chip="wc"
    )
    transfers_made_wc = len(res_wc["gameweeks"][1]["transfers_in"])
    assert transfers_made_wc == 4
