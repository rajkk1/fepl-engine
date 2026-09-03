"""
Selling prices.

FPL pays the purchase price plus HALF of any rise, rounded down; a fall is taken
in full. Without a cookie the engine defaulted to the *current* price, which
over-states the budget by half of every rise - and produced a recommended
transfer that FPL refused for want of £0.1m.
"""
import pytest

import weekly_manager as W
from weekly_manager import fpl_selling_price as sell


def _els(*specs):
    """(id, now_cost_tenths, cost_change_start) -> bootstrap-style elements."""
    return [{"id": i, "web_name": f"P{i}", "now_cost": n, "cost_change_start": c}
            for i, n, c in specs]


# ------------------------------------------------------------- the FPL rule


@pytest.mark.parametrize("buy,now,expected", [
    (40, 41, 40),    # a 0.1 rise is worth nothing: profit halves and floors
    (46, 47, 46),
    (76, 77, 76),
    (75, 77, 76),    # a 0.2 rise pays 0.1
    (90, 96, 93),    # a 0.6 rise pays 0.3
    (90, 97, 93),    # 0.7 floors to 0.3, not 0.35
    (50, 50, 50),    # unchanged
    (60, 55, 55),    # a fall is taken in full
])
def test_fpl_selling_price_rule(buy, now, expected):
    assert sell(buy, now) == expected


def test_a_rise_is_never_worth_more_than_half():
    for buy in range(40, 130, 5):
        for rise in range(0, 20):
            got = sell(buy, buy + rise)
            assert got - buy <= rise / 2
            assert got >= buy


# --------------------------------------------------- reconstruction from API


def test_purchase_price_comes_from_the_transfer_history():
    els = _els((1, 90, 5))
    transfers = [{"element_in": 1, "element_in_cost": 85, "event": 3, "time": "t"}]
    assert W.purchase_prices([1], els, transfers) == {1: 85}


def test_an_unheld_player_falls_back_to_the_season_start_price():
    """Held since GW1, so the purchase price is now_cost minus the rise."""
    els = _els((1, 90, 5))
    assert W.purchase_prices([1], els, []) == {1: 85}


def test_the_most_recent_purchase_wins():
    """Sold and bought back: the latest price is the one FPL remembers."""
    els = _els((1, 100, 0))
    transfers = [
        {"element_in": 1, "element_in_cost": 80, "event": 2, "time": "a"},
        {"element_in": 1, "element_in_cost": 95, "event": 9, "time": "a"},
    ]
    assert W.purchase_prices([1], els, transfers) == {1: 95}


def test_the_reported_case_reproduces():
    """
    The four risers in the reported squad. The engine believed it could sell at
    current prices and was £0.4m over, so the recommended transfer was £0.1m
    short at the FPL end.
    """
    els = _els((1, 41, 1), (2, 47, 1), (3, 77, 1), (4, 77, 2))
    sp = W.estimate_sell_prices([1, 2, 3, 4], els, [])
    assert sp == {1: 4.0, 2: 4.6, 3: 7.6, 4: 7.6}
    at_now = sum(e["now_cost"] / 10.0 for e in els)
    assert round(at_now - sum(sp.values()), 1) == 0.4


def test_prices_are_returned_in_pounds_to_match_the_optimiser():
    els = _els((1, 90, 0))
    assert W.estimate_sell_prices([1], els, []) == {1: 9.0}


def test_reconstruction_survives_missing_and_malformed_data():
    els = _els((1, 90, 5))
    assert W.estimate_sell_prices([], els, []) == {}
    assert W.estimate_sell_prices([99], els, []) == {}          # unknown player
    assert W.purchase_prices([1], els, None) == {1: 85}
    assert W.purchase_prices([1], els, [{"event": 3}]) == {1: 85}       # no ids
    assert W.purchase_prices([1], els, [{"element_in": 1, "event": 3}]) == {1: 85}
    assert W.purchase_prices([1], _els((1, 90, None)), []) == {1: 90}


def test_a_faller_is_not_penalised_twice():
    """A player who dropped sells at the current price, not below it."""
    els = _els((1, 80, -10))     # bought 9.0, now 8.0
    assert W.estimate_sell_prices([1], els, []) == {1: 8.0}
