"""
Market team ratings: opponent adjustment, the home edge, shrinkage.

Every test here is synthetic. The implied goals are generated from a known
log-linear league, so "the right answer" is not a matter of judgement.

The headline case is two clubs of identical strength, one that only ever meets
strong sides and one that only ever meets weak ones. Averaging each club's
implied goals over its own fixtures - which is what the ratings used to do -
rates the first a much worse attack than the second, and every forecast built
on those ratings inherits that.
"""
import math

import numpy as np
import pandas as pd
import pytest

import market_odds
from market_odds import MarketOddsModel, fit_log_linear_ratings

HARD, EASY = 1, 2
STRONG = [3, 4, 5, 6]
WEAK = [7, 8, 9, 10]
LEVEL = math.log(1.25)       # an average side's away goals
HOME_EDGE = math.log(1.20)


def _league(base: float = 0.0, gap: float = 0.4):
    """HARD and EASY share `base`; the strong and weak blocs sit `gap` either side."""
    att = {HARD: base, EASY: base}
    att.update({t: gap for t in STRONG})
    att.update({t: -gap for t in WEAK})
    return att, dict(att)          # same attack and defence, for simplicity


def _schedule(rounds: int = 8, opponents_meet: int = 0):
    """
    (round, home, away). HARD meets only STRONG and EASY only WEAK, on the same
    rounds and at the same venues, so the time-decay weights are identical. The
    other six play strong-strong, weak-weak and strong-weak, which connects the
    league - and not as two halves that only ever meet each other, which would
    leave one direction of attack-versus-defence undetermined.

    `opponents_meet` adds that many extra strong-weak double headers per round:
    more evidence about the opponents, so less of the prior leaks through them.
    """
    out = []
    for r in range(rounds):
        k, at_home = r % 4, (r // 4) % 2 == 0
        s, w = STRONG[k], WEAK[k]
        out.append((r, HARD, s) if at_home else (r, s, HARD))
        out.append((r, EASY, w) if at_home else (r, w, EASY))
        rest_s = [t for t in STRONG if t != s]
        rest_w = [t for t in WEAK if t != w]
        rest_s = rest_s[r % 3:] + rest_s[:r % 3]
        rest_w = rest_w[r % 3:] + rest_w[:r % 3]
        for h, a in ((rest_s[0], rest_s[1]), (rest_w[0], rest_w[1]),
                     (rest_s[2], rest_w[2])):
            out.append((r, h, a) if r % 2 else (r, a, h))
        for _ in range(opponents_meet):
            for t in STRONG:
                for u in WEAK:
                    out += [(r, t, u), (r, u, t)]
    return out


def _implied(fixtures, att, dfn):
    """The market's expected goals, exactly from the log-linear league."""
    mu_h = [math.exp(LEVEL + HOME_EDGE + att[h] - dfn[a]) for _, h, a in fixtures]
    mu_a = [math.exp(LEVEL + att[a] - dfn[h]) for _, h, a in fixtures]
    return mu_h, mu_a


def _odds_frame(fixtures, att, dfn) -> pd.DataFrame:
    mu_h, mu_a = _implied(fixtures, att, dfn)
    start = pd.Timestamp("2025-08-16")
    return pd.DataFrame({
        "Date": [start + pd.Timedelta(days=7 * r) for r, _, _ in fixtures],
        "HomeTeam": [f"T{h}" for _, h, _ in fixtures],
        "AwayTeam": [f"T{a}" for _, _, a in fixtures],
        "mu_h": mu_h, "mu_a": mu_a,
    })


def _fitted(frame: pd.DataFrame) -> MarketOddsModel:
    """A model fitted to `frame` through the production entry point."""
    m = MarketOddsModel()
    m.odds_df = frame
    names = sorted(set(frame["HomeTeam"]) | set(frame["AwayTeam"]))
    m.FD_TO_FPL = {n: int(n[1:]) for n in names}
    m.FPL_TO_FD = {v: k for k, v in m.FD_TO_FPL.items()}
    status = m.fit_team_ratings(fpl_teams=None)
    assert status["source"] == "market_odds"
    return m


def _opponent_blind(frame: pd.DataFrame, k: float):
    """
    The rule this replaced, kept here as the reference the test must beat: each
    club's decay-weighted average of its own implied goals, shrunk toward the
    league mean by k pseudo-matches, with no regard to whom it played.
    """
    latest = frame["Date"].max()
    w = 0.5 ** (((latest - frame["Date"]).dt.days / 7.0) / market_odds.RATING_HALF_LIFE)
    mean = float(np.average(np.r_[frame["mu_h"], frame["mu_a"]], weights=np.r_[w, w]))
    scored = {}
    for tid in (HARD, EASY):
        name = f"T{tid}"
        goals = np.r_[frame.loc[frame["HomeTeam"] == name, "mu_h"],
                      frame.loc[frame["AwayTeam"] == name, "mu_a"]]
        weights = np.r_[w[frame["HomeTeam"] == name], w[frame["AwayTeam"] == name]]
        scored[tid] = (float(np.sum(goals * weights)) + k * mean) / (float(weights.sum()) + k)
    return scored


# --------------------------------------------------------- opponent adjustment


@pytest.mark.parametrize("base", [0.0, 0.25])
def test_schedule_strength_does_not_leak_into_attack(monkeypatch, base):
    """
    Two clubs of identical strength, one meeting only strong opponents and one
    only weak ones, must come out with the same ratings.

    With the ridge off this is exact for any schedule. The shipped ridge still
    shrinks the *opponents* toward the league, and that leaks through them in
    proportion to how little is known about them, so here they are given enough
    matches to be well established - which is what a season provides.
    """
    att, dfn = _league(base)

    monkeypatch.setattr(market_odds, "RATING_PRIOR_STRENGTH", 0.0)
    m = _fitted(_odds_frame(_schedule(), att, dfn))
    for key in ("att_home", "att_away", "def_home", "def_away", "scored", "conceded"):
        assert m.team_ratings[HARD][key] == pytest.approx(m.team_ratings[EASY][key], rel=1e-9), key
    assert m.team_attack_baseline(HARD) == pytest.approx(m.team_attack_baseline(EASY), rel=1e-9)

    monkeypatch.setattr(market_odds, "RATING_PRIOR_STRENGTH", 4.0)
    m = _fitted(_odds_frame(_schedule(opponents_meet=10), att, dfn))
    for key in ("att_home", "att_away", "def_home", "def_away"):
        assert m.team_ratings[HARD][key] == pytest.approx(m.team_ratings[EASY][key], rel=0.005), key


def test_the_rule_it_replaced_fails_the_same_schedule(monkeypatch):
    """
    The schedule test has teeth: on the same data the opponent-blind average
    rates the club that met strong sides as an attack about a third worse than
    its identical twin, while the fit agrees with itself to well under 1%.
    """
    monkeypatch.setattr(market_odds, "RATING_PRIOR_STRENGTH", 4.0)
    att, dfn = _league()
    frame = _odds_frame(_schedule(opponents_meet=10), att, dfn)

    old = _opponent_blind(frame, k=4.0)
    old_gap = math.log(old[EASY] / old[HARD])
    assert old_gap > 0.3, "the averaging rule should mis-rate HARD badly here"

    m = _fitted(frame)
    new_gap = abs(math.log(m.team_attack_baseline(EASY) / m.team_attack_baseline(HARD)))
    assert new_gap < old_gap / 100


def test_every_club_is_rated_at_its_true_strength(monkeypatch):
    """
    Not merely HARD equal to EASY: every club, including the strong and weak
    sides whose own results were read against HARD and EASY, comes back at the
    strength the league was generated from. The bias ran both ways - a club met
    by a weak schedule also flattered everyone it played.

    Strength is relative to the league's average club, so the true terms here
    sum to zero; the fit makes the same choice.
    """
    monkeypatch.setattr(market_odds, "RATING_PRIOR_STRENGTH", 0.0)
    att, dfn = _league()
    m = _fitted(_odds_frame(_schedule(), att, dfn))
    for t, r in m.team_ratings.items():
        assert r["att_home"] == pytest.approx(math.exp(LEVEL + HOME_EDGE + att[t]), rel=1e-9)
        assert r["att_away"] == pytest.approx(math.exp(LEVEL + att[t]), rel=1e-9)
        assert r["def_home"] == pytest.approx(math.exp(LEVEL - dfn[t]), rel=1e-9)
        assert r["def_away"] == pytest.approx(math.exp(LEVEL + HOME_EDGE - dfn[t]), rel=1e-9)


def test_player_rates_do_not_inherit_the_schedule(monkeypatch, player):
    """
    End to end through the player-rate normalisation in xp_model: two strikers
    who take the same share of their (identical) teams' chances, one on each
    schedule, must be forecast the same for the same fixture.
    """
    from xp_model import GammaPoissonFilter, MarketOddsPredictor

    monkeypatch.setattr(market_odds, "RATING_PRIOR_STRENGTH", 0.0)
    att, dfn = _league()
    fixtures = _schedule()
    dc = MarketOddsPredictor()
    dc.market = _fitted(_odds_frame(fixtures, att, dfn))
    gpf = GammaPoissonFilter(stat_settings={"xg": (8.0, 1e-9)})

    def forecast(team):
        hist = []
        for r, h, a in fixtures:
            if team not in (h, a):
                continue
            home = h == team
            team_goals = math.exp(LEVEL + (HOME_EDGE if home else 0.0)
                                  + att[team] - dfn[a if home else h])
            hist.append({"round": r + 1, "minutes": 90, "was_home": home,
                         "opponent_team": a if home else h,
                         "expected_goals": 0.3 * team_goals})
        rates = gpf.predict_match({**player, "element_type": 4, "team": team},
                                  hist, dc, len(fixtures))
        ctx = dc.fixture_context(team, {"team_h": team, "team_a": STRONG[0]})
        return rates["xg90"] * ctx["att_mult"]

    assert forecast(HARD) == pytest.approx(forecast(EASY), rel=1e-6)
    # And it is the true expectation for that fixture.
    truth = 0.3 * math.exp(LEVEL + HOME_EDGE + att[HARD] - dfn[STRONG[0]])
    assert forecast(HARD) == pytest.approx(truth, rel=1e-6)


# ------------------------------------------------------------- the home edge


@pytest.mark.parametrize("venue_split", [False, True])
def test_fitted_home_edge_is_applied_once(monkeypatch, venue_split):
    """
    Regression: a home attack and an away defence each already carry the home
    edge, and dividing their product by one all-venue league mean squared it.
    In a league of identical clubs priced at 1.6 at home and 1.2 away, the old
    normaliser forecast 1.83 and 1.03.
    """
    monkeypatch.setattr(market_odds, "RATING_VENUE_SPLIT", venue_split)
    teams = list(range(1, 7))
    fixtures = [(r, h, a) for r in range(2) for h in teams for a in teams if h != a]
    frame = pd.DataFrame({
        "Date": [pd.Timestamp("2025-08-16") + pd.Timedelta(days=7 * r) for r, _, _ in fixtures],
        "HomeTeam": [f"T{h}" for _, h, _ in fixtures],
        "AwayTeam": [f"T{a}" for _, _, a in fixtures],
        "mu_h": 1.6, "mu_a": 1.2,
    })
    m = _fitted(frame)
    for h in teams:
        for a in teams:
            if h != a:
                mu_h, mu_a = m.get_match_lambdas(h, a)
                assert mu_h == pytest.approx(1.6, rel=1e-9)
                assert mu_a == pytest.approx(1.2, rel=1e-9)
    # An unrated club meets the league average at that venue.
    assert m.get_match_lambdas(99, 1) == pytest.approx((1.6, 1.2), rel=1e-9)


def test_fit_reproduces_the_market_including_unplayed_pairings():
    """
    The ratings exist to price fixtures the odds file has not reached yet. Fit
    on one round of matches between twenty clubs, then price every reverse
    fixture, none of which was in the fit.
    """
    rng = np.random.default_rng(7)
    att = dict(zip(range(1, 21), rng.normal(0, 0.25, 20)))
    dfn = dict(zip(range(1, 21), rng.normal(0, 0.25, 20)))
    teams = list(range(1, 21))
    # A single round robin: each pair meets once, at one venue.
    first_leg = [(0, h, a) for i, h in enumerate(teams) for a in teams[i + 1:]]
    first_leg = [(r, h, a) if (h + a) % 2 else (r, a, h) for r, h, a in first_leg]

    fit = fit_log_linear_ratings(
        [h for _, h, _ in first_leg], [a for _, _, a in first_leg],
        *_implied(first_leg, att, dfn), weights=[1.0] * len(first_leg),
        prior_strength=0.0, venue_split=False,
    )
    m = MarketOddsModel()
    m.team_ratings = fit["teams"]
    m._set_venue_bases()
    assert m._home_base == pytest.approx(fit["home_base"], rel=1e-12)
    assert m._away_base == pytest.approx(fit["away_base"], rel=1e-12)

    reverse = [(1, a, h) for _, h, a in first_leg]
    for (_, h, a), want_h, want_a in zip(reverse, *_implied(reverse, att, dfn)):
        got_h, got_a = m.get_match_lambdas(h, a)
        assert got_h == pytest.approx(want_h, rel=1e-9)
        assert got_a == pytest.approx(want_a, rel=1e-9)


# ----------------------------------------------------------------- shrinkage


def test_a_thin_start_is_shrunk_toward_the_league():
    """
    One round is not a season. The ridge keeps a club the market rated well in
    its opener between the league and that one reading, and pulls harder the
    stronger the prior - the job `RATING_PRIOR_STRENGTH` always had.
    """
    teams = list(range(1, 21))
    opener = [(0, teams[i], teams[i + 1]) for i in range(0, 20, 2)]
    att = {t: 0.0 for t in teams}
    att[1] = 0.5
    dfn = {t: 0.0 for t in teams}
    mu_h, mu_a = _implied(opener, att, dfn)
    args = ([h for _, h, _ in opener], [a for _, _, a in opener], mu_h, mu_a, [1.0] * 10)

    shown = mu_h[0]                              # team 1, at home
    ratings = [fit_log_linear_ratings(*args, prior_strength=k)["teams"][1]["att_home"]
               for k in (0.5, 4.0, 16.0)]
    league = fit_log_linear_ratings(*args, prior_strength=4.0)["home_base"]
    assert league < ratings[2] < ratings[1] < ratings[0] < shown


def test_unpenalised_intercepts_keep_the_league_level():
    """The ridge pulls clubs toward the league, never the league toward 1.4."""
    teams = list(range(1, 7))
    fixtures = [(0, h, a) for h in teams for a in teams if h != a]
    att = {t: 0.1 * (t - 3.5) for t in teams}
    dfn = {t: -0.05 * (t - 3.5) for t in teams}
    mu_h, mu_a = _implied(fixtures, att, dfn)
    fit = fit_log_linear_ratings([h for _, h, _ in fixtures], [a for _, _, a in fixtures],
                                 mu_h, mu_a, [1.0] * len(fixtures), prior_strength=50.0)
    assert fit["away_base"] == pytest.approx(math.exp(LEVEL), rel=1e-9)
    assert fit["home_base"] == pytest.approx(math.exp(LEVEL + HOME_EDGE), rel=1e-9)


# ----------------------------------------------------------------- interfaces


def test_exported_ratings_seed_the_next_season(monkeypatch):
    """export_ratings -> prior_ratings -> _seed_from_prior keeps working."""
    att, dfn = _league()
    m = _fitted(_odds_frame(_schedule(opponents_meet=2), att, dfn))
    exported = m.export_ratings()
    assert set(exported[STRONG[0]]) >= {"att_home", "att_away", "def_home", "def_away"}

    nxt = MarketOddsModel()
    nxt.odds_df = None
    status = nxt.fit_team_ratings(
        fpl_teams=[{"id": t} for t in range(1, 12)], prior_ratings=exported)
    assert status["source"] == "prior_season"
    strong_h, weak_a = nxt.get_match_lambdas(STRONG[0], WEAK[0])
    weak_h, strong_a = nxt.get_match_lambdas(WEAK[0], STRONG[0])
    assert strong_h > weak_h and strong_a > weak_a
    # A club the prior season never saw is seeded at the league default.
    assert nxt.team_ratings[11]["n_eff"] == 0.0
    assert nxt.team_attack_baseline(STRONG[0]) > nxt.team_attack_baseline(WEAK[0])
