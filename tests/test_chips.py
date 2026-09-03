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
    Only the authenticated endpoint knows the free-transfer count. Assuming 1
    when the manager has 0 turns a planned free transfer into a -4 hit, so the
    number must be labelled rather than presented as fact.
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
