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

def test_bench_weight_parameter_is_honoured(base_bootstrap):
    """
    Regression: `bench_weight` was a documented parameter but 0.05 was hardcoded
    in both the objective and the reporting path, so it did nothing.
    """
    initial = list(range(1, 16))
    xp_matrix = {pid: {1: 5.0} for pid in range(1, 31)}

    low = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp_matrix, horizon_gws=[1],
        initial_squad_ids=initial, initial_bank=10.0, initial_ft=1,
        max_hits_per_gw=0, bench_weight=0.0)
    high = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp_matrix, horizon_gws=[1],
        initial_squad_ids=initial, initial_bank=10.0, initial_ft=1,
        max_hits_per_gw=0, bench_weight=0.5)

    assert high["gameweeks"][1]["gw_xp"] > low["gameweeks"][1]["gw_xp"]


def test_vice_captain_priced_by_modelled_blank_probability(base_bootstrap):
    """
    Regression: the VC term used a flat P_CAP_BLANK = 0.05 rather than the
    captain's own blank probability, which the matrix already carries.
    """
    initial = list(range(1, 16))
    reliable = {pid: {1: 5.0, "1_p_play": 1.0} for pid in range(1, 31)}
    flaky = {pid: {1: 5.0, "1_p_play": 0.5} for pid in range(1, 31)}

    r_rel = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=reliable, horizon_gws=[1],
        initial_squad_ids=initial, initial_bank=10.0, initial_ft=1, max_hits_per_gw=0)
    r_fla = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=flaky, horizon_gws=[1],
        initial_squad_ids=initial, initial_bank=10.0, initial_ft=1, max_hits_per_gw=0)

    assert r_fla["gameweeks"][1]["gw_xp"] > r_rel["gameweeks"][1]["gw_xp"]


def test_vice_captain_value_tracks_the_captain_not_itself(base_bootstrap):
    """
    The VC scores when the *captain* blanks. Pricing it off the VC's own blank
    probability would reward picking an unreliable vice-captain, which is
    backwards. The captain is a decision variable, so the objective estimates
    P(captain blanks) from the players in contention for the armband.
    """
    initial = list(range(1, 16))
    # One clearly best captain who is a certain starter; everyone else is flaky.
    matrix = {pid: {1: 3.0, "1_p_play": 0.5} for pid in range(1, 31)}
    matrix[8] = {1: 12.0, "1_p_play": 1.0}

    res = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=matrix, horizon_gws=[1],
        initial_squad_ids=initial, initial_bank=10.0, initial_ft=1, max_hits_per_gw=0)

    starters = res["gameweeks"][1]["starters"]
    vc = next(p for p in starters if p["is_vice_captain"])
    others = [p for p in starters if not p["is_captain"] and not p["is_vice_captain"]]
    # With a reliable captain the VC is nearly worthless, so it should go to a
    # high-xP starter rather than being used to chase an unreliable player.
    assert vc["xp"] >= max([p["xp"] for p in others], default=0.0)


def test_chip_can_target_a_later_gameweek(base_bootstrap):
    """
    Regression: weekly_manager never passed active_chip_gw, so a chip could only
    ever be evaluated in the first gameweek of the horizon.
    """
    initial = list(range(1, 16))
    xp_matrix = {pid: {1: 1.0, 2: 8.0} for pid in range(1, 31)}
    res = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp_matrix, horizon_gws=[1, 2],
        initial_squad_ids=initial, initial_bank=10.0, initial_ft=1,
        max_hits_per_gw=0, active_chip="bb", active_chip_gw=2)
    assert res["status"] == "Optimal"
    assert res["gameweeks"][2]["gw_xp"] > res["gameweeks"][1]["gw_xp"] * 2


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


def test_future_gameweeks_are_discounted(base_bootstrap):
    """
    A point in GW+4 is not worth a point now: injuries, rotation and price
    changes accumulate, and the plan is re-solved next week anyway. Weighting the
    horizon equally traded a real point now for a speculative one at par.
    """
    xp_matrix = {pid: {1: 1.0, 2: 1.0, 3: 1.0} for pid in range(1, 31)}
    res_flat = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp_matrix, horizon_gws=[1, 2, 3],
        initial_squad_ids=list(range(1, 16)), initial_bank=10.0, initial_ft=1,
        horizon_decay=1.0,
    )
    res_decayed = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp_matrix, horizon_gws=[1, 2, 3],
        initial_squad_ids=list(range(1, 16)), initial_bank=10.0, initial_ft=1,
        horizon_decay=0.86,
    )
    assert res_decayed["total_xp"] < res_flat["total_xp"]


def test_a_distant_gain_no_longer_outweighs_a_nearer_one(base_bootstrap):
    """With decay, the solver should prefer points it can bank sooner."""
    near = {pid: {1: 2.0, 2: 2.0} for pid in range(1, 31)}
    near[20] = {1: 9.0, 2: 2.0}     # big gain in GW1
    far = {pid: {1: 2.0, 2: 2.0} for pid in range(1, 31)}
    far[20] = {1: 2.0, 2: 9.0}      # same gain, a week later

    def total(matrix):
        return solve_fpl_optimization(
            bootstrap=base_bootstrap, xp_matrix=matrix, horizon_gws=[1, 2],
            initial_squad_ids=list(range(1, 16)), initial_bank=10.0, initial_ft=1,
            horizon_decay=0.86,
        )["total_xp"]

    assert total(near) > total(far)


def test_bench_keeper_is_worth_less_than_a_bench_outfielder(base_bootstrap):
    """
    An outfield sub scores when a starter in his position blanks, which happens
    often. The backup keeper plays only if the first-choice keeper does not.
    A flat bench weight priced those the same.
    """
    from optimizer import BENCH_WEIGHT_BY_POSITION, POS_GKP, POS_DEF, POS_MID, POS_FWD
    assert BENCH_WEIGHT_BY_POSITION[POS_GKP] < BENCH_WEIGHT_BY_POSITION[POS_DEF]
    assert BENCH_WEIGHT_BY_POSITION[POS_GKP] < BENCH_WEIGHT_BY_POSITION[POS_MID]
    assert BENCH_WEIGHT_BY_POSITION[POS_GKP] < BENCH_WEIGHT_BY_POSITION[POS_FWD]
