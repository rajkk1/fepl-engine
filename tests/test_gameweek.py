"""
Which gameweek the engine plans for.

It published a plan on 2026-09-07 for GW3, whose deadline had passed on
2026-09-04, recommending three transfers and a -8 hit for a gameweek that could
no longer be played. The squad in it was also stale, because picks are read from
`current_gw - 1`.
"""
import datetime as dt

import pytest

from fpl_api import get_current_gameweek

UTC = dt.timezone.utc


def _ev(eid, deadline, **flags):
    base = {"id": eid, "deadline_time": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished": False, "is_current": False, "is_next": False}
    base.update(flags)
    return base


def _season(now, n=6, spacing_days=7):
    """Gameweeks around `now`: three behind, the rest ahead."""
    return [_ev(i + 1, now + dt.timedelta(days=spacing_days * (i - 2)))
            for i in range(n)]


def test_a_passed_deadline_is_never_planned_for():
    """The reported bug, reproduced by its flags."""
    now = dt.datetime(2026, 9, 7, 7, 0, tzinfo=UTC)
    events = [
        _ev(1, dt.datetime(2026, 8, 21, 17, 30, tzinfo=UTC), finished=True),
        _ev(2, dt.datetime(2026, 8, 28, 17, 30, tzinfo=UTC), finished=True),
        # matches played, results not yet confirmed: the old test's blind spot
        _ev(3, dt.datetime(2026, 9, 4, 17, 30, tzinfo=UTC),
            finished=False, is_current=True),
        _ev(4, dt.datetime(2026, 9, 12, 12, 30, tzinfo=UTC), is_next=True),
    ]
    assert get_current_gameweek({"events": events}) == 4


def test_the_flag_combination_that_caused_it_is_ignored():
    """`is_current and not finished` must not win over an open deadline."""
    now = dt.datetime.now(UTC)
    events = [
        _ev(1, now - dt.timedelta(days=3), is_current=True, finished=False),
        _ev(2, now + dt.timedelta(days=4)),
    ]
    assert get_current_gameweek({"events": events}) == 2


def test_iteration_order_cannot_pre_empt_the_answer():
    """
    The old loop returned on the first matching event, so a flag on an earlier
    gameweek beat the correct later one regardless of the deadlines.
    """
    now = dt.datetime.now(UTC)
    events = [
        _ev(9, now + dt.timedelta(days=9), is_next=True),
        _ev(3, now - dt.timedelta(days=1), is_current=True),
        _ev(4, now + dt.timedelta(hours=2)),      # soonest open deadline
    ]
    assert get_current_gameweek({"events": events}) == 4


def test_the_soonest_open_deadline_wins():
    now = dt.datetime.now(UTC)
    events = [_ev(7, now + dt.timedelta(days=30)),
              _ev(5, now + dt.timedelta(days=1)),
              _ev(6, now + dt.timedelta(days=10))]
    assert get_current_gameweek({"events": events}) == 5


def test_a_deadline_seconds_away_is_still_open():
    now = dt.datetime.now(UTC)
    events = [_ev(4, now + dt.timedelta(seconds=30)),
              _ev(5, now + dt.timedelta(days=7))]
    assert get_current_gameweek({"events": events}) == 4


def test_end_of_season_falls_back_rather_than_crashing():
    now = dt.datetime.now(UTC)
    events = [_ev(37, now - dt.timedelta(days=14), finished=True),
              _ev(38, now - dt.timedelta(days=7), finished=True, is_current=True)]
    assert get_current_gameweek({"events": events}) == 38


def test_unparseable_and_missing_deadlines_are_survivable():
    now = dt.datetime.now(UTC)
    for events in (
        [],
        [{"id": 4}],                                        # no deadline
        [{"deadline_time": "2099-01-01T00:00:00Z"}],        # no id
        [{"id": 4, "deadline_time": "not-a-date"}],
        [{"id": 4, "deadline_time": None}],
    ):
        assert isinstance(get_current_gameweek({"events": events}), int)


def test_a_naive_deadline_is_treated_as_utc():
    now = dt.datetime.now(UTC)
    events = [{"id": 4, "deadline_time":
               (now + dt.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S")}]
    assert get_current_gameweek({"events": events}) == 4


def test_missing_events_key_does_not_raise():
    assert get_current_gameweek({}) == 1
