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


# --- Chips played later in the horizon -------------------------------------
#
# Every structural chip rule used to key off the loop index rather than the
# gameweek the chip was actually planned for, so all of it was only ever
# exercised at the first gameweek of the horizon. `weekly_manager` searches
# chip x gameweek across the whole horizon, which means every "hold it for
# GW+N" evaluation was priced against a game nobody plays. The existing
# later-gameweek test covers bench boost, which touches the objective only.

@pytest.fixture
def one_big_week():
    """
    Squad is fine in GW1, 2 and 4; only the players outside it score in GW3.
    The pool outside the initial squad is 11 DEF, 2 MID, 2 FWD and no keeper,
    so squad structure allows at most 9 of them in at once.
    """
    xp = {}
    for pid in range(1, 31):
        if pid <= 15:
            xp[pid] = {1: 5.0, 2: 5.0, 3: 1.0, 4: 5.0}
        else:
            xp[pid] = {1: 0.1, 2: 0.1, 3: 20.0, 4: 0.1}
    return xp


def _squad(res, gw):
    return set(p["id"] for p in res["gameweeks"][gw]["starters"] + res["gameweeks"][gw]["bench"])


def test_a_wildcard_later_in_the_horizon_is_still_unlimited(base_bootstrap, one_big_week):
    initial = list(range(1, 16))
    res = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=one_big_week, horizon_gws=[1, 2, 3, 4],
        initial_squad_ids=initial, initial_bank=10.0, initial_ft=1,
        max_hits_per_gw=2, active_chip="wc", active_chip_gw=3, horizon_decay=1.0)

    assert res["status"] == "Optimal"
    # One free transfer plus two hits would have capped this at three.
    assert len(res["gameweeks"][3]["transfers_in"]) > 3
    assert res["gameweeks"][3]["hits"] == 0
    # A wildcard is permanent: the new squad is still there the week after.
    assert len(_squad(res, 4) & set(initial)) < 15


def test_a_free_hit_later_in_the_horizon_is_handed_back(base_bootstrap, one_big_week):
    initial = list(range(1, 16))
    res = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=one_big_week, horizon_gws=[1, 2, 3, 4],
        initial_squad_ids=initial, initial_bank=10.0, initial_ft=1,
        max_hits_per_gw=2, active_chip="fh", active_chip_gw=3, horizon_decay=1.0)

    assert res["status"] == "Optimal"
    assert len(res["gameweeks"][3]["transfers_in"]) > 3
    assert res["gameweeks"][3]["hits"] == 0
    # The chip week is a different squad, and GW4 gets the old one back for
    # free. Previously GW4 inherited the free-hit squad and paid hits to undo it.
    assert len(_squad(res, 3) & set(initial)) < 15
    assert _squad(res, 4) == set(initial)
    assert res["gameweeks"][4]["hits"] == 0
    assert len(res["gameweeks"][4]["transfers_in"]) == 0


def test_a_free_hit_reverts_to_the_squad_before_it_not_to_the_original(base_bootstrap):
    """
    A free hit in GW3 hands back whatever was held in GW2, which by then may
    already differ from the squad the horizon started with. Reverting to
    `initial_squad_ids` would silently undo the transfer made in GW2.
    """
    initial = list(range(1, 16))
    xp = {}
    for pid in range(1, 31):
        if pid <= 15:
            xp[pid] = {1: 5.0, 2: 5.0, 3: 1.0, 4: 5.0}
        else:
            xp[pid] = {1: 0.1, 2: 0.1, 3: 20.0, 4: 0.1}
    # Player 20 (a defender) is worth owning from GW2 onward, so the solver
    # should buy him in GW2 with its free transfer and still hold him in GW4.
    xp[20] = {1: 0.1, 2: 40.0, 3: 20.0, 4: 40.0}

    res = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp, horizon_gws=[1, 2, 3, 4],
        initial_squad_ids=initial, initial_bank=10.0, initial_ft=1,
        max_hits_per_gw=2, active_chip="fh", active_chip_gw=3, horizon_decay=1.0)

    assert res["status"] == "Optimal"
    assert 20 in _squad(res, 2)
    assert 20 in _squad(res, 4), "the free hit gave back the pre-horizon squad, not the GW2 one"


def test_a_hit_cannot_be_bought_in_a_week_with_no_transfers(base_bootstrap, one_big_week):
    """
    `hits` appeared on the right of the free-transfer constraint, so the solver
    could pay -4 in a quiet week purely to inflate the transfer bank and fund a
    rebuild later. That is not a move FPL offers, and it surfaced in the plan as
    a recommended hit against a gameweek with no transfer in it.
    """
    initial = list(range(1, 16))
    res = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=one_big_week, horizon_gws=[1, 2, 3, 4],
        initial_squad_ids=initial, initial_bank=10.0, initial_ft=1,
        max_hits_per_gw=2, horizon_decay=1.0)

    assert res["status"] == "Optimal"
    for gw, plan in res["gameweeks"].items():
        n_in = len(plan["transfers_in"])
        assert plan["hits"] <= max(0, n_in - 1), (
            f"GW{gw} charged {plan['hits']} hit(s) for {n_in} transfer(s)")


def test_free_transfers_bank_to_five_and_no_further(base_bootstrap):
    """
    FPL caps the bank at five. The cap sat on the carried-out variable alone,
    so a manager holding five could spend `1 + 5` in a single week.
    """
    initial = list(range(1, 16))
    xp = {}
    for pid in range(1, 31):
        xp[pid] = {gw: (3.0 if pid <= 15 else 0.0) for gw in range(1, 7)}
        xp[pid][7] = 1.0 if pid <= 15 else 30.0

    res = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp, horizon_gws=list(range(1, 8)),
        initial_squad_ids=initial, initial_bank=10.0, initial_ft=1,
        max_hits_per_gw=0, horizon_decay=1.0)

    assert res["status"] == "Optimal"
    for gw, plan in res["gameweeks"].items():
        assert plan["hits"] == 0
        assert len(plan["transfers_in"]) <= 5, (
            f"GW{gw} made {len(plan['transfers_in'])} free transfers; FPL allows at most 5")
    assert len(res["gameweeks"][7]["transfers_in"]) == 5


def test_a_chip_keeps_the_free_transfers_it_did_not_use(base_bootstrap):
    """
    Saved free transfers survive a wildcard rather than being spent by it, so
    the week after one starts from the bank plus the week's own transfer. The
    bank used to be zeroed on the chip week.
    """
    initial = list(range(1, 16))
    # Three defenders are worth owning in GW2 and ruinous in GW1, so the
    # wildcard cannot simply buy them in its own week and hold them: they have
    # to be bought in GW2, out of the bank the chip week left behind.
    xp = {pid: {1: 3.0, 2: 3.0} for pid in range(1, 16)}
    for pid in range(16, 31):
        xp[pid] = {1: 0.0, 2: 0.0}
    for pid in (20, 21, 22):
        xp[pid] = {1: -100.0, 2: 30.0}

    res = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp, horizon_gws=[1, 2],
        initial_squad_ids=initial, initial_bank=10.0, initial_ft=3,
        max_hits_per_gw=0, active_chip="wc", active_chip_gw=1, horizon_decay=1.0)

    assert res["status"] == "Optimal"
    # 3 banked, unspent by the chip, + 1 earned = 4 available in GW2 with no hits.
    assert len(res["gameweeks"][2]["transfers_in"]) == 3
    assert res["gameweeks"][2]["hits"] == 0


def test_a_transfer_that_gains_nothing_is_not_recommended(base_bootstrap):
    """
    A free transfer is free, so a swap worth zero points scored exactly the
    same as holding and the solver was free to pick either. It picked one, and
    the plan then told the manager to churn a player for an identically rated
    replacement. Ties belong to doing nothing.
    """
    initial = list(range(1, 16))
    xp = {pid: {gw: 3.0 for gw in (1, 2, 3)} for pid in range(1, 31)}

    res = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp, horizon_gws=[1, 2, 3],
        initial_squad_ids=initial, initial_bank=10.0, initial_ft=9)

    assert res["status"] == "Optimal"
    for gw, plan in res["gameweeks"].items():
        assert plan["transfers_in"] == [], f"GW{gw} recommended a transfer worth nothing"


def test_the_tiebreak_cannot_outvote_a_real_difference(base_bootstrap):
    """
    The nudge toward holding must sit far below the smallest difference the
    forecast can even express -- xP is rounded to two decimals -- so that it
    settles exact ties and nothing else.

    Note what the transfer has to clear to be worth making, which is not the
    tiebreak: a banked free transfer already carries +1.5 of terminal value, so
    a swap worth a hundredth of a point is correctly declined with or without
    this term. The test therefore uses a gain that clears that bar.
    """
    from optimizer import TRANSFER_TIEBREAK

    # Even a whole squad's worth of it stays under one expressible unit of xP.
    assert 15 * TRANSFER_TIEBREAK < 0.01

    initial = list(range(1, 16))
    xp = {pid: {1: 3.0} for pid in range(1, 31)}
    xp[20] = {1: 6.0}

    res = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp, horizon_gws=[1],
        initial_squad_ids=initial, initial_bank=10.0, initial_ft=1)

    assert [p["id"] for p in res["gameweeks"][1]["transfers_in"]] == [20]


def test_a_player_is_never_transferred_in_and_out_in_the_same_week(base_bootstrap):
    """
    `s == s_prev + tin - tout` is satisfied just as well by tin = tout = 1 for a
    player who is simply held, so every held player carried a pair of symmetric
    assignments for the solver to rule out. Cutting them changes no optimum --
    a self-transfer buys nothing -- and made the chip search roughly four times
    faster, which matters because a timed-out solve is silently dropped from
    the chip comparison rather than merely being slow.
    """
    import pulp
    from optimizer import solve_fpl_optimization as solve

    initial = list(range(1, 16))
    xp = {pid: {1: 3.0, 2: 3.0} for pid in range(1, 31)}

    # Inspect the model rather than the plan: the constraint is about which
    # solutions exist, and the reported plan cannot show a self-transfer.
    captured = {}
    real_solve = pulp.LpProblem.solve

    def spy(self, *a, **kw):
        captured["names"] = set(self.constraints)
        return real_solve(self, *a, **kw)

    pulp.LpProblem.solve = spy
    try:
        solve(bootstrap=base_bootstrap, xp_matrix=xp, horizon_gws=[1, 2],
              initial_squad_ids=initial, initial_bank=10.0, initial_ft=1)
    finally:
        pulp.LpProblem.solve = real_solve

    cuts = [n for n in captured["names"] if n.startswith("No_Self_Transfer_")]
    assert cuts, "the symmetry cut is missing; the chip search will be ~4x slower"


def test_the_symmetry_cut_does_not_change_the_answer(base_bootstrap):
    """It removes only dominated solutions, so the plan must be unchanged."""
    initial = list(range(1, 16))
    xp = {pid: {1: 2.0, 2: 2.0, 3: 2.0} for pid in range(1, 31)}
    for pid in (20, 21, 22):
        xp[pid] = {1: 9.0, 2: 9.0, 3: 9.0}

    res = solve_fpl_optimization(
        bootstrap=base_bootstrap, xp_matrix=xp, horizon_gws=[1, 2, 3],
        initial_squad_ids=initial, initial_bank=10.0, initial_ft=1)

    assert res["status"] == "Optimal"
    # The three better players are still bought, one per free transfer.
    owned = set()
    for gw in (1, 2, 3):
        owned |= {p["id"] for p in res["gameweeks"][gw]["transfers_in"]}
    assert {20, 21, 22} <= owned


# --- bench cover is worth more behind an unreliable eleven ------------------
#
# A substitute only scores if a starter fails to appear, so the value of the
# bench depends on the eleven in front of it. The weight used to be a constant,
# which priced cover identically for a nailed XI and a set of coin flips.

def _bench_fixture(p_play_by_id=None, default_p_play=1.0):
    els = []
    pid = 1
    for t in range(1, 21):
        for et in (1, 2, 2, 3, 3, 4):
            els.append({"id": pid, "web_name": f"P{pid}", "element_type": et,
                        "team": t, "now_cost": 50})
            pid += 1
    player_dict = {p["id"]: p for p in els}
    xp = {p["id"]: {1: 5.0,
                    "1_p_play": (p_play_by_id or {}).get(p["id"], default_p_play)}
          for p in els}
    return xp, player_dict, list(player_dict)


def test_bench_cover_is_worth_more_when_the_eleven_are_unreliable():
    from optimizer import bench_weights_for_gameweek, POS_MID

    weights = []
    for pp in (0.99, 0.95, 0.90, 0.85):
        xp, pd_, ids = _bench_fixture(default_p_play=pp)
        weights.append(bench_weights_for_gameweek(xp, pd_, ids, 1)[POS_MID])

    assert weights == sorted(weights), f"not monotone in rotation risk: {weights}"
    assert weights[0] < weights[-1]


def test_an_availability_blind_forecast_keeps_the_shipped_constants():
    """
    `ppg` and `roll3` assert p_play = 1.0 for everyone, because a trailing
    average has no opinion about who starts. Deriving a bench weight from that
    would hand them a worthless bench and then charge them for the autosubs they
    actually need - the replay shows `ppg` taking 1.079 a gameweek against the
    engine's 0.447 - which would flatter the engine for a reason that has
    nothing to do with forecast quality.
    """
    from optimizer import bench_weights_for_gameweek, BENCH_WEIGHT_BY_POSITION

    xp, pd_, ids = _bench_fixture(default_p_play=1.0)
    assert bench_weights_for_gameweek(xp, pd_, ids, 1) == BENCH_WEIGHT_BY_POSITION


def test_a_forecast_with_no_p_play_at_all_keeps_the_shipped_constants():
    from optimizer import bench_weights_for_gameweek, BENCH_WEIGHT_BY_POSITION

    _, pd_, ids = _bench_fixture()
    bare = {pid: {1: 5.0} for pid in ids}
    assert bench_weights_for_gameweek(bare, pd_, ids, 1) == BENCH_WEIGHT_BY_POSITION


def test_the_bench_keeper_is_priced_off_the_first_choice_keeper_alone():
    """
    The backup keeper only ever replaces the keeper, so ten shaky outfielders
    must not make him valuable.
    """
    from optimizer import bench_weights_for_gameweek, POS_GKP, POS_MID

    xp, pd_, ids = _bench_fixture(default_p_play=0.8)
    keepers = [p for p in ids if pd_[p]["element_type"] == POS_GKP]
    for k in keepers:
        xp[k]["1_p_play"] = 1.0          # every keeper is nailed

    w = bench_weights_for_gameweek(xp, pd_, ids, 1)
    assert w[POS_GKP] == 0.0
    assert w[POS_MID] > 0.0


def test_the_weight_never_exceeds_face_value():
    """Only three outfield substitutes can come on, however many starters fail."""
    from optimizer import bench_weights_for_gameweek

    xp, pd_, ids = _bench_fixture(default_p_play=0.0)
    for w in bench_weights_for_gameweek(xp, pd_, ids, 1).values():
        assert 0.0 <= w <= 1.0


def test_a_realistic_nailed_squad_lands_near_the_hand_picked_constant():
    """
    The shipped 0.12 was a guess, and replaying three seasons says it was a good
    one: the engine takes 0.447 autosubs a gameweek across four bench players.
    A formula that disagreed with it at the engine's own operating point would
    be the formula to doubt.
    """
    from optimizer import bench_weights_for_gameweek, BENCH_WEIGHT_BY_POSITION, POS_MID

    xp, pd_, ids = _bench_fixture(default_p_play=0.96)
    w = bench_weights_for_gameweek(xp, pd_, ids, 1)[POS_MID]
    assert abs(w - BENCH_WEIGHT_BY_POSITION[POS_MID]) < 0.05, w


def test_a_risky_eleven_buys_a_better_bench(base_bootstrap):
    """The whole point: the squad built should actually change."""
    initial = list(range(1, 16))
    nailed = {pid: {1: 5.0, "1_p_play": 1.0} for pid in range(1, 31)}
    risky = {pid: {1: 5.0, "1_p_play": 0.7} for pid in range(1, 31)}

    a = solve_fpl_optimization(bootstrap=base_bootstrap, xp_matrix=nailed,
                               horizon_gws=[1], initial_squad_ids=initial,
                               initial_bank=10.0, max_hits_per_gw=0)
    b = solve_fpl_optimization(bootstrap=base_bootstrap, xp_matrix=risky,
                               horizon_gws=[1], initial_squad_ids=initial,
                               initial_bank=10.0, max_hits_per_gw=0)
    # Everyone is priced the same here, so the only thing that can move the
    # reported gameweek xP is how the bench is valued.
    assert b["gameweeks"][1]["gw_xp"] > a["gameweeks"][1]["gw_xp"]


def test_the_explicit_bench_weight_override_still_wins():
    """`--bench-weight` is a deliberate instruction and must not be second-guessed."""
    from optimizer import bench_weights_for_gameweek
    import optimizer as o
    import inspect

    src = inspect.getsource(o.solve_fpl_optimization)
    # The override is checked before the per-gameweek weights are consulted.
    assert src.index("elif bench_weight is not None") < src.index("gw_bench_weights.get")


def test_a_binary_p_play_is_not_an_availability_model():
    """
    `_baseline_matrix` says p_play 1.0 for any player it rates and 0.0 for
    everyone else - 302 of 780 players in 2025-26. That is "certainly" and "no
    idea", not a rotation model, so a bench weight must not be derived from it.

    Checking only whether the blank probability was non-zero let exactly this
    through whenever a zero-rated player reached the likely eleven, which moved
    the baselines by ~0.3% in a replay they should have been insulated from.
    """
    from optimizer import bench_weights_for_gameweek, BENCH_WEIGHT_BY_POSITION

    xp, pd_, ids = _bench_fixture(default_p_play=1.0)
    # A third of the pool is rated zero, exactly as the baselines do it.
    for pid in ids[::3]:
        xp[pid]["1_p_play"] = 0.0
        xp[pid][1] = 0.0

    assert bench_weights_for_gameweek(xp, pd_, ids, 1) == BENCH_WEIGHT_BY_POSITION


def test_one_graded_probability_is_enough_to_use_the_model():
    """A forecast that grades anyone is modelling availability."""
    from optimizer import bench_weights_for_gameweek, BENCH_WEIGHT_BY_POSITION

    xp, pd_, ids = _bench_fixture(default_p_play=1.0)
    xp[ids[0]]["1_p_play"] = 0.5

    assert bench_weights_for_gameweek(xp, pd_, ids, 1) != BENCH_WEIGHT_BY_POSITION
