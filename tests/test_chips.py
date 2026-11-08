"""
Chip availability.

The engine used to assume every chip was still in hand whenever it could not
reach the authenticated my-team endpoint - which is the default, since that
endpoint needs a cookie. It then recommended a wildcard that had been played in
GW2. Chip state is public, so this is read rather than assumed.
"""
import pytest

import weekly_manager as W


def _hist(*played):
    return {"chips": [{"name": n, "time": "x", "event": e} for n, e in played]}


def test_a_played_wildcard_is_not_available():
    """The exact reported bug: wildcard played in GW2, still recommended."""
    available, used = W.chips_from_history(_hist(("wildcard", 2)), current_gw=4)
    assert "wc" not in available
    assert used == {"wc": 2}


def test_the_other_chips_survive_a_played_wildcard():
    available, _ = W.chips_from_history(_hist(("wildcard", 2)), current_gw=4)
    assert set(available) == {"fh", "bb", "tc"}


def test_a_second_set_unlocks_in_the_second_half():
    """
    FPL splits the season: the first-half chips expire at the GW19 deadline and
    a fresh set unlocks from GW20. A wildcard played in GW2 must not rule out
    the second-half one.
    """
    h = _hist(("wildcard", 2))
    assert "wc" not in W.chips_from_history(h, current_gw=19)[0]
    assert "wc" in W.chips_from_history(h, current_gw=20)[0]
    assert "wc" in W.chips_from_history(h, current_gw=34)[0]


def test_a_chip_played_in_the_second_half_is_spent_for_that_half():
    h = _hist(("bboost", 25))
    assert "bb" in W.chips_from_history(h, current_gw=10)[0]
    assert "bb" not in W.chips_from_history(h, current_gw=30)[0]


def test_every_api_chip_name_is_recognised():
    """A rename upstream must not silently read as 'still available'."""
    h = _hist(("wildcard", 3), ("freehit", 4), ("bboost", 5), ("3xc", 6))
    available, used = W.chips_from_history(h, current_gw=7)
    assert available == []
    assert set(used) == {"wc", "fh", "bb", "tc"}


def test_missing_or_malformed_history_is_survivable():
    for h in ({}, {"chips": None}, {"chips": []},
              {"chips": [{"name": "wildcard"}]},          # no event
              {"chips": [{"event": 3}]},                  # no name
              {"chips": [{"name": "mystery", "event": 3}]}):
        available, used = W.chips_from_history(h, current_gw=5)
        assert set(available) == set(W.ALL_CHIPS)
        assert used == {}


def test_unreadable_chip_history_assumes_none_rather_than_all(monkeypatch):
    """
    Fail safe. Recommending a chip the manager does not hold is a worse failure
    than missing one they do, so an unreachable endpoint yields an empty list -
    the opposite of the old optimistic default.
    """
    def boom(team_id, use_cache=True):
        raise OSError("offline")

    monkeypatch.setattr(W.fpl_api, "get_manager_history", boom)
    assert W.chip_state(1234, 5) == []


def test_chip_state_reads_the_public_endpoint(monkeypatch):
    monkeypatch.setattr(W.fpl_api, "get_manager_history",
                        lambda team_id, use_cache=True: _hist(("wildcard", 2)))
    assert set(W.chip_state(1234, 4)) == {"fh", "bb", "tc"}


def test_used_chips_override_parses_and_validates():
    assert W._parse_used_chips("wc,bb") == {"wc", "bb"}
    assert W._parse_used_chips(" WC  fh ") == {"wc", "fh"}
    assert W._parse_used_chips("") == set()
    with pytest.raises(ValueError):
        W._parse_used_chips("wc,wildcard")


# ------------------------------------------------- free transfers are a guess


def test_free_transfers_are_flagged_as_assumed_without_a_cookie(monkeypatch):
    """
    With no cookie and a history that cannot be followed (here it has no
    gameweeks at all), the count falls back to 1. Assuming 1 when the manager
    has 0 turns a planned free transfer into a -4 hit, so the number must be
    labelled rather than presented as fact.
    """
    monkeypatch.delenv("FPL_COOKIE", raising=False)
    monkeypatch.setattr(W.fpl_api, "get_manager_picks",
                        lambda t, e: {"picks": [{"element": 1}],
                                      "entry_history": {"bank": 3}})
    monkeypatch.setattr(W.fpl_api, "get_manager_history",
                        lambda t, use_cache=True: {"chips": []})
    monkeypatch.setattr(W.fpl_api, "get_manager_transfers",
                        lambda t, use_cache=True: [])
    monkeypatch.setattr(W.fpl_api, "get_bootstrap_static",
                        lambda: {"elements": [{"id": 1, "now_cost": 50,
                                               "cost_change_start": 0}]})
    *_, ft_known = W.get_manager_team_state(123, current_gw=5)
    assert ft_known is False


def test_gameweek_one_is_genuinely_unlimited_not_a_guess(monkeypatch):
    """Pre-season is a rule, so it is knowledge rather than an assumption."""
    monkeypatch.delenv("FPL_COOKIE", raising=False)
    monkeypatch.setattr(W.fpl_api, "get_manager_picks",
                        lambda t, e: {"picks": [], "entry_history": {"bank": 0}})
    monkeypatch.setattr(W.fpl_api, "get_manager_history",
                        lambda t, use_cache=True: {"chips": []})
    monkeypatch.setattr(W.fpl_api, "get_manager_transfers",
                        lambda t, use_cache=True: [])
    monkeypatch.setattr(W.fpl_api, "get_bootstrap_static",
                        lambda: {"elements": []})
    _, _, ft, _, _, ft_known = W.get_manager_team_state(123, current_gw=1)
    assert ft >= 100 and ft_known is True


def test_an_unreachable_team_does_not_claim_to_know_the_count(monkeypatch):
    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(W.fpl_api, "get_manager_picks", boom)
    monkeypatch.setattr(W.fpl_api, "get_manager_history", boom)
    *_, ft_known = W.get_manager_team_state(123, current_gw=5)
    assert ft_known is False


def test_a_rejected_cookie_is_an_error_not_a_note(monkeypatch, caplog):
    """
    An expired cookie silently returns the engine to guessed chips, selling
    prices and free transfers - the exact assumptions the authenticated path
    exists to replace. FPL sessions expire, so this must be loud.
    """
    import logging as _logging

    monkeypatch.setenv("FPL_COOKIE", "sentinel-not-a-real-cookie")

    def rejected(team_id, cookie):
        raise PermissionError("403")

    monkeypatch.setattr(W.fpl_api, "get_my_team", rejected)
    monkeypatch.setattr(W.fpl_api, "get_manager_picks",
                        lambda t, e: {"picks": [], "entry_history": {"bank": 0}})
    monkeypatch.setattr(W.fpl_api, "get_manager_history",
                        lambda t, use_cache=True: {"chips": []})
    monkeypatch.setattr(W.fpl_api, "get_manager_transfers",
                        lambda t, use_cache=True: [])
    monkeypatch.setattr(W.fpl_api, "get_bootstrap_static", lambda: {"elements": []})

    with caplog.at_level(_logging.INFO):
        *_, ft_known = W.get_manager_team_state(123, current_gw=5)

    assert ft_known is False, "a rejected cookie cannot leave the count 'known'"
    errors = [r for r in caplog.records if r.levelno >= _logging.ERROR]
    assert errors, "a rejected cookie must be logged at ERROR"
    assert "expired" in errors[0].getMessage().lower()
    # The credential itself must never reach the log.
    assert "sentinel-not-a-real-cookie" not in caplog.text


# ------------------------------------- free transfers, rebuilt from history
#
# The count used to be assumed to be 1 without a cookie. The public history
# carries every week's transfers and their cost, and the chip list tells a
# Wildcard week from one that banked, which is enough to follow it.


def _season(*weeks, chips=()):
    """weeks: (event, transfers made, points they cost)."""
    return {
        "current": [{"event": e, "event_transfers": n, "event_transfers_cost": c}
                    for e, n, c in weeks],
        "chips": [{"name": name, "event": e} for name, e in chips],
    }


def test_a_transfer_is_earned_each_week_and_banked():
    h = _season((1, 0, 0), (2, 0, 0), (3, 1, 0), (4, 0, 0))
    # Into GW2: 1. Banked to 2, spent one to 1+1=2, banked to 3.
    assert W.free_transfers_from_history(h, current_gw=2) == 1
    assert W.free_transfers_from_history(h, current_gw=3) == 2
    assert W.free_transfers_from_history(h, current_gw=4) == 2
    assert W.free_transfers_from_history(h, current_gw=5) == 3


def test_the_bank_stops_at_five():
    h = _season(*[(gw, 0, 0) for gw in range(1, 12)])
    assert W.free_transfers_from_history(h, current_gw=12) == W.MAX_BANKED_FT


def test_a_week_that_cost_points_spent_every_free_transfer():
    h = _season((1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 4, 4))
    assert W.free_transfers_from_history(h, current_gw=5) == 1


def test_a_wildcard_or_free_hit_keeps_the_bank_but_not_the_week():
    h = _season((1, 0, 0), (2, 0, 0), (3, 11, 0), (4, 15, 0),
                chips=[("wildcard", 3), ("freehit", 4)])
    # Into GW3: 2 (one banked). Each chip week uses its own transfer and keeps
    # the banked one, so GW4 and GW5 also start with 2.
    assert W.free_transfers_from_history(h, current_gw=4) == 2
    assert W.free_transfers_from_history(h, current_gw=5) == 2


def test_the_real_history_that_settled_the_chip_rule(caplog):
    """
    Taken from a live public history: a Wildcard in GW2 on one free transfer,
    then three transfers for -8 in GW3 - so GW3 had exactly one. The "+1 after
    a chip" rule said two, and the hit cost is what showed it.
    """
    h = _season((1, 0, 0), (2, 0, 0), (3, 3, 8), (4, 3, 8), (5, 2, 4),
                (6, 1, 0), (7, 1, 0), (8, 1, 0), (9, 1, 0), (10, 1, 0),
                chips=[("wildcard", 2), ("bboost", 6)])
    with caplog.at_level("WARNING"):
        assert W.free_transfers_from_history(h, current_gw=11) == 1
    assert "disagrees" not in caplog.text


def test_a_history_that_contradicts_the_rules_says_so(caplog):
    # Two free transfers' worth of moves for nothing, when the rules allow one:
    # some allowance the engine does not model was granted and spent.
    h = _season((1, 0, 0), (2, 2, 0))
    with caplog.at_level("WARNING"):
        W.free_transfers_from_history(h, current_gw=3)
    assert "GW2" in caplog.text and "disagrees" in caplog.text


def test_a_late_joiner_starts_from_the_week_they_picked_a_squad():
    h = _season((6, 0, 0), (7, 0, 0))
    assert W.free_transfers_from_history(h, current_gw=7) == 1
    assert W.free_transfers_from_history(h, current_gw=8) == 2


def test_a_gap_in_the_history_is_not_guessed_across():
    h = _season((1, 0, 0), (2, 0, 0), (4, 0, 0))
    assert W.free_transfers_from_history(h, current_gw=5) is None
    assert W.free_transfers_from_history({"chips": []}, current_gw=5) is None


def _public_state(monkeypatch, picks_by_gw, history):
    monkeypatch.delenv("FPL_COOKIE", raising=False)
    monkeypatch.setattr(W.fpl_api, "get_manager_picks",
                        lambda t, e: picks_by_gw[e])
    monkeypatch.setattr(W.fpl_api, "get_manager_history",
                        lambda t, use_cache=True: history)
    monkeypatch.setattr(W.fpl_api, "get_manager_transfers",
                        lambda t, use_cache=True: [])
    monkeypatch.setattr(W.fpl_api, "get_bootstrap_static",
                        lambda: {"elements": [
                            {"id": i, "now_cost": 50, "cost_change_start": 0}
                            for i in range(1, 40)]})


def test_without_a_cookie_the_count_is_rebuilt_not_assumed(monkeypatch):
    history = _season((1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0))
    _public_state(monkeypatch,
                  {4: {"picks": [{"element": i} for i in range(1, 16)],
                       "entry_history": {"bank": 5}}},
                  history)
    _, _, ft, _, _, ft_known = W.get_manager_team_state(123, current_gw=5)
    assert ft == 4 and ft_known is True


def test_after_a_free_hit_the_plan_starts_from_the_squad_handed_back(monkeypatch):
    """
    Last gameweek's picks after a Free Hit are the borrowed squad. The squad and
    bank that carry on are the ones from before the chip.
    """
    kept = [{"element": i} for i in range(1, 16)]
    borrowed = [{"element": i} for i in range(20, 35)]
    _public_state(monkeypatch, {
        5: {"picks": kept, "entry_history": {"bank": 12}, "active_chip": None},
        6: {"picks": borrowed, "entry_history": {"bank": 3},
            "active_chip": "freehit"},
    }, _season(*[(gw, 0, 0) for gw in range(1, 7)], chips=[("freehit", 6)]))

    squad, bank, *_ = W.get_manager_team_state(123, current_gw=7)
    assert squad == list(range(1, 16))
    assert bank == pytest.approx(1.2)
