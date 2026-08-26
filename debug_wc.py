import pytest
from optimizer import solve_fpl_optimization

bootstrap = {
    "elements": [
        {"id": i, "web_name": f"P{i}", "element_type": 1 if i<=2 else 2 if i<=7 else 3 if i<=12 else 4, "team": (i%7)+1, "now_cost": 50}
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

res_wc = solve_fpl_optimization(
    bootstrap=bootstrap, xp_matrix=xp_matrix, horizon_gws=[1],
    initial_squad_ids=initial_squad_ids, initial_bank=10.0, initial_ft=1,
    max_hits_per_gw=2, active_chip="wc"
)
print([t['id'] for t in res_wc['gameweeks'][1]['transfers_in']])
print([t['id'] for t in res_wc['gameweeks'][1]['transfers_out']])
print(res_wc['gameweeks'][1]['starters'])
