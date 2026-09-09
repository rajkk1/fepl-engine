"""
The price of a transfer beyond the free one.

The rules say a hit costs 4 points. The number that belongs in the ILP objective
is not automatically the rule's number, because the objective weighs a *certain*
-4 against an *estimated* gain, and an estimate chosen for being the largest in
the pool is not unbiased. So `hit_cost` is a modelling parameter, and this file
pins the mechanics of it.

What the measurement actually found (three full-xG seasons, 114 gameweeks,
forecast computed once per gameweek and shared across arms so the only
difference is spending freedom):

    hit cap      0       1       2       3
    pooled    6362    6202    6383    6293
    hits         0      46      64      69

Every difference sits inside the noise and the three seasons disagree on the
sign in every comparison, so the shipped default of 2 hits and 4.0 points is
kept. The optimum is flat; this is not where the remaining points are.

Two hypotheses going in were both wrong, and are recorded here so they are not
re-run from scratch:

  - "the engine does not hold a player long enough to earn back the hit" - it
    holds a transferred-in player 7.75 gameweeks on average (median 6), which
    is LONGER than the five it prices against, so the 3.78 multiplier is
    slightly conservative rather than optimistic;
  - "the optimiser's curse means the estimated edge is illusory" - over a fixed
    five-gameweek window a paid transfer really does out-gain the player it
    replaced, by +7.45 points.

The reason those two facts do not add up to a season-level gain is that
"points of the player in, minus points of the player out" is not a measurement
of a transfer's value. It conditions on the outgoing player being the squad's
worst-rated asset, so it is positive by construction under ANY transfer policy,
including a pointless one. Replaying the season against a different cap is the
only comparison that holds the alternative use of the slot and the budget fixed
- and there the extra transfers buy a different squad of about equal quality
(the arms share only ~8 of 15 players by mid-season).
"""
import optimizer
import simulator


def test_hit_cost_is_a_parameter_defaulting_to_the_rules():
    import inspect

    sig = inspect.signature(optimizer.solve_fpl_optimization)
    assert sig.parameters["hit_cost"].default == optimizer.HIT_COST
    assert optimizer.HIT_COST == 4.0, "the shipped default matches the rules"


def test_the_objective_uses_the_parameter_not_a_literal():
    """A hardcoded -4.0 would make the sweep silently measure nothing - every
    arm would produce identical plans and the experiment would 'prove' the
    price is irrelevant."""
    import inspect

    src = inspect.getsource(optimizer.solve_fpl_optimization)
    assert "-hit_cost * decay * hits[t]" in src
    assert "-4.0 * decay" not in src


def test_the_replay_threads_the_price_through():
    import inspect

    sig = inspect.signature(simulator.run_season_simulation)
    assert "hit_cost" in sig.parameters
    src = inspect.getsource(simulator.run_season_simulation)
    assert "hit_cost=hit_cost" in src


def test_the_hit_is_discounted_alongside_the_points_it_buys():
    """
    Charging the hit at full price while discounting the gain would make the
    solver structurally refuse every transfer after the first gameweek, for
    reasons that have nothing to do with whether the transfer is good.
    """
    import inspect

    src = inspect.getsource(optimizer.solve_fpl_optimization)
    assert "decay * hits[t]" in src


def test_the_breakeven_edge_is_what_the_decay_implies():
    """
    The gain multiplier is what actually sets the threshold, and it is easy to
    change `HORIZON_DECAY` without noticing what it does to transfer appetite.
    At 0.86 over five gameweeks a hit pays for itself at an edge of ~1.06 xP
    per gameweek, which is a low bar - so this is worth pinning.
    """
    mult = sum(optimizer.HORIZON_DECAY ** i for i in range(5))
    assert 3.7 < mult < 3.9
    breakeven = optimizer.HIT_COST / mult
    assert 1.0 < breakeven < 1.15


def test_the_replay_records_what_moved():
    """
    Holding duration and per-transfer outcomes cannot be measured without
    knowing who came in, who left, and what was held. The history carried
    only counts, which is why the question was unanswerable before.
    """
    import inspect

    src = inspect.getsource(simulator.run_season_simulation)
    for field in ('"in": list(transfers_in)', '"out": sorted(', '"squad": list(squad_ids)'):
        assert field in src, f"history no longer records {field}"
