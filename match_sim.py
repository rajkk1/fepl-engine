"""
Correlated match simulation.

Two things in the engine need the same machinery, so they share it here:

  1. **Bonus points.** Bonus is a rank statistic - 3/2/1 to the top three BPS
     scorers *in a match*. The previous model mapped one player's E[BPS] through
     a line, `max(0, (E[BPS] - 8) * 0.15)`, which cannot represent a
     top-3-of-22 outcome. Here we draw every involved player's BPS, rank them,
     award 3/2/1, and average over draws.

  2. **Rank-aware risk (MERV).** Variance was computed per player and then summed
     by the ILP, which treats three Arsenal players as three independent bets.
     Clean sheets and goals inside a team are strongly correlated, and team
     stacking is precisely where rank-awareness earns its keep. Simulating the
     match jointly gives correlated per-player point draws.

The BPS table below follows the official scoring. The previous E[BPS] omitted
every negative term and several positive ones.
"""
import logging
import math
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

POS_GKP, POS_DEF, POS_MID, POS_FWD = 1, 2, 3, 4
# Positions the engine actually models. FPL has shipped others (managers were
# element_type 5 in 2024-25), and an unmodelled position must be skipped
# rather than crash a whole gameweek's forecast.
MODELLED_POSITIONS = frozenset({POS_GKP, POS_DEF, POS_MID, POS_FWD})

POINTS_GOAL = {POS_GKP: 10, POS_DEF: 6, POS_MID: 5, POS_FWD: 4}
POINTS_CLEAN_SHEET = {POS_GKP: 4, POS_DEF: 4, POS_MID: 1, POS_FWD: 0}
POINTS_ASSIST = 3

# --- BPS table (official values) -------------------------------------------
BPS_PLAYING_60 = 6
BPS_PLAYING_UNDER_60 = 3
BPS_GOAL = {POS_GKP: 12, POS_DEF: 12, POS_MID: 18, POS_FWD: 24}
BPS_ASSIST = 9
BPS_CLEAN_SHEET = {POS_GKP: 12, POS_DEF: 12, POS_MID: 0, POS_FWD: 0}
BPS_SAVE = 2                    # per save
BPS_PENALTY_SAVE = 15
BPS_YELLOW = -3
BPS_RED = -9
BPS_OWN_GOAL = -6
BPS_PENALTY_MISS = -6
BPS_GOALS_CONCEDED_PER_2 = -4   # GK/DEF only
BPS_DEFCON = 3                  # hitting the defensive-contribution threshold
# Volume terms, approximated from per-90 rates.
BPS_PER_RECOVERY = 1 / 3.0
BPS_PER_TACKLE_WON = 2.0
BPS_PER_KEY_PASS = 1.0

# FPL match points for the rare events. These are individually small, but they
# are the whole of the BPS table's negative half, and leaving them out biased
# every simulated bonus race toward players who commit fouls and concede.
POINTS_YELLOW = -1
POINTS_RED = -3
POINTS_OWN_GOAL = -2
POINTS_PENALTY_MISS = -2
POINTS_PENALTY_SAVE = 5

DEFCON_THRESHOLD = {POS_DEF: 10, POS_MID: 12, POS_FWD: 12}

# FPL introduced defensive contribution points in 2025-26. Earlier seasons did
# not award them, and the Gamma-Poisson filter correctly treats the absent CBIT
# columns as *missing* rather than zero - so it falls back to the positional
# prior and happily predicts DefCon points for a season in which none existed.
# Backtesting before 2025-26 without this gate scores every defender and
# midfielder under rules FPL was not playing by.
DEFCON_FIRST_SEASON = 2025


def defcon_active(season: Optional[int]) -> bool:
    """Whether defensive-contribution points existed in a given season."""
    return season is None or int(season) >= DEFCON_FIRST_SEASON

# Mean minutes actually played inside each of the minutes model's appearance
# buckets, measured over 2025-26 (n = 11,498 appearances):
#
#     bucket   assumed   actual
#     1-59        30.0    22.23
#     60-89       75.0    74.83
#     90          90.0    90.00
#
# The 1-59 midpoint was assumed to be the arithmetic middle of the bucket, but
# substitute appearances are skewed short. Overstating every cameo by ~8 minutes
# accounted for more than half of a 74-minute-per-team-match over-allocation,
# and it inflated every rate-driven term for exactly the fringe players whose
# minutes are least certain.
BUCKET_MINUTES = (22.2, 74.8, 90.0)

# Representative length of a sub appearance, used to size a short appearance's
# exposure to goals conceded.
SHORT_APPEARANCE_MINUTES = BUCKET_MINUTES[0]

# P(a goal carries an FPL assist). FPL's assist rule is more generous than
# Opta's, and the ratio is stable: 0.896 (2023-24), 0.905 (2024-25), 0.934
# (2025-26) league-wide assists per goal.
ASSISTS_PER_GOAL = 0.90

# Variance/mean of a team's goals in a match. Left at Poisson, deliberately.
#
# Splitting a Poisson total multinomially returns *independent* Poissons, so in
# theory only an over-dispersed total can correlate team-mates, and the measured
# Pearson dispersion over 2280 market-priced team-matches is 1.09 - which looks
# like a mandate to raise this. It is not. Comparing the whole goal distribution
# against 2023-24..2025-26 shows the 1.09 is not a fat tail at all:
#
#     goals        0       1       2       3       4      >=6
#     actual  0.2320  0.3268  0.2513  0.1215  0.0478   0.0044
#     phi=1.00  0.2469  0.3285  0.2268  0.1215  0.0509   0.0075
#     phi=1.09  0.2570  0.3215  0.2219  0.1162  0.0535   0.0114
#
# Reality has *less* mass at 0 than Poisson and *less* at 6+, with the excess
# piled on 2. That is the Dixon-Coles low-score pattern, not a Gamma mixture, so
# a negative binomial moves 0 and 6+ the wrong way and makes the clean-sheet
# rate worse: P(concede nothing) errs +0.0163 at phi=1.00 and +0.0304 at 1.09.
# Buying +0.028 of team-mate correlation by degrading every defender's and
# keeper's clean sheet is a bad trade.
#
# The fix that would improve both at once is a Dixon-Coles adjustment to this
# marginal (shift mass 0 -> 2), which raises correlation *and* corrects P(0).
# Until that exists, this stays Poisson. Raise it only to experiment.
GOAL_DISPERSION = 1.0


def _nb_params(mu: float, dispersion: float) -> Tuple[float, float]:
    """Negative-binomial (n, p) for a given mean and variance/mean ratio."""
    mu = max(1e-6, mu)
    var = max(mu * 1.0001, dispersion * mu)
    p = mu / var
    n = (mu * mu) / max(1e-9, var - mu)
    return n, p


def _allocate(rng, totals: np.ndarray, weights: np.ndarray, denom) -> np.ndarray:
    """
    Hand out each of `totals` team events (per sim) to one of the `weights` rows.

    Row j claims an event with probability `weights[j] / denom`. The leftover
    probability belongs to nobody in the list - own goals, and any player the
    caller did not pass in - and keeping that residual bucket is precisely what
    preserves each player's marginal. With `denom` set to the team's expected
    goals and `totals` drawn Poisson with that same mean,

        E[events_j] = E[totals] * w_j / denom = w_j

    so a player's expected goals are exactly what they were when every player
    drew an independent Poisson. Only the *joint* distribution changes, which is
    the entire point: team-mates now share one team total.

    `denom` is raised to the realised weight sum when a team's listed xG
    over-accounts for its expected goals, which costs a little of that exactness
    in return for never handing out more goals than were scored.

    Implemented as a chain of conditional binomials - one vectorised call per
    player, rather than one multinomial draw per simulation.
    """
    k, n = weights.shape
    out = np.zeros((k, n), dtype=np.int64)
    denom = np.maximum(np.broadcast_to(np.asarray(denom, dtype=float), (n,)),
                       weights.sum(axis=0))
    denom = np.maximum(denom, 1e-9)
    remaining = np.asarray(totals, dtype=np.int64).copy()
    consumed = np.zeros(n, dtype=float)
    for j in range(k):
        share = weights[j] / denom
        tail = 1.0 - consumed
        cond = np.where(tail > 1e-12, share / np.maximum(tail, 1e-12), 0.0)
        np.clip(cond, 0.0, 1.0, out=cond)
        take = rng.binomial(remaining, cond)
        out[j] = take
        remaining -= take
        consumed += share
    return out


class MatchSimulator:
    """Simulates one fixture jointly across all supplied players."""

    def __init__(self, n_sims: int = 2000, seed: int = 0,
                 goal_dispersion: float = GOAL_DISPERSION):
        self.n_sims = n_sims
        self.seed = seed
        self.goal_dispersion = float(goal_dispersion)

    def _team_goals(self, rng, lam: float, n: int) -> np.ndarray:
        """
        One team's goals in one match, drawn `n` times.

        Poisson at dispersion 1.0. Above that, a negative binomial with the same
        mean: real team goals are over-dispersed (measured Pearson dispersion
        1.09 over 2280 team-matches priced by the market), and that
        over-dispersion is the *only* thing that can correlate team-mates'
        attacking returns. Splitting a Poisson total multinomially returns
        independent Poissons - the classic thinning result - so a shared total
        alone buys nothing. The variance of the total is what couples them.
        """
        if self.goal_dispersion <= 1.0:
            return rng.poisson(lam=lam, size=n)
        nb_n, nb_p = _nb_params(lam, self.goal_dispersion)
        return rng.negative_binomial(max(1e-3, nb_n), min(0.999, max(1e-6, nb_p)), size=n)

    def simulate(
        self,
        players: List[Dict[str, Any]],
        team_lambdas: Dict[int, float],
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """
        `players` is a list of component dicts (as produced by
        EnsembleForecaster._predict_uncalibrated) each carrying at least:
            id, team, element_type, p_play, p_60, xg_cond, xa_cond,
            saves90, yc90, defcon_mu, defcon_dispersion, play_frac
        `team_lambdas` maps team id -> expected goals conceded *by the opponent*
        (i.e. the goals this team's opponent is expected to score), used to draw
        a shared clean-sheet / goals-conceded outcome per team.

        Pass `rng` to share one generator across every fixture in a gameweek.
        Without it each match would redraw from an identically seeded stream,
        which correlates players in unrelated matches -- the exact artefact the
        correlated simulation exists to avoid.

        Rate-like inputs (`saves90`, `yc90`, `defcon_mu`, ...) are interpreted
        *conditional on the player appearing*, scaled by `play_frac` =
        E[minutes | played] / 90. Scaling by unconditional expected minutes
        instead conflates "half a chance of playing 90" with "certainly plays
        45", which understates every threshold and floor term for rotation
        risks.

        Returns per-player expected bonus and the full simulated points matrix.
        """
        n = self.n_sims
        rng = rng or np.random.default_rng(self.seed)
        if not players:
            return {"bonus": {}, "points": {}, "n_sims": n}

        # One shared goals-conceded draw per team per sim: this is what couples
        # team-mates' clean sheets together.
        sim_teams = sorted(set(team_lambdas) | {p["team"] for p in players})
        lam_conceded = {t: max(0.01, float(team_lambdas.get(t, 1.4))) for t in sim_teams}
        conceded = {t: self._team_goals(rng, lam_conceded[t], n) for t in sim_teams}
        # Per-player clean sheets are derived below from each player's own
        # on-pitch share of this shared draw, so there is no team-level
        # clean-sheet flag: a keeper subbed at half time in a 0-1 can still
        # keep one, and the team did not.

        # The goals a team *scores* are the goals its opponent concedes: the
        # very same draw, read from the other side of the fixture. Sharing it is
        # what couples team-mates' *attacking* returns. Before this, each
        # player's goals came from an independent Poisson, so three club-mates
        # correlated at -0.07 - statistically indistinguishable from opposition
        # players, and slightly negative because they compete for the same
        # bonus. A 4-0 win never produced the simultaneous hauls it produces in
        # reality, so the joint upper tail was much too thin. That tail is
        # exactly where captaincy and precision@15 are decided, and it is the
        # half of "team stacking prices correctly" that clean sheets alone could
        # never deliver.
        opponent = {}
        for t in sim_teams:
            others = [k for k in sim_teams if k != t]
            # A fixture has two sides. Anything else - a caller passing one
            # team, a malformed lambda map - keeps the old independent draws.
            opponent[t] = others[0] if len(others) == 1 else None

        n_players = len(players)
        active = [
            i for i, p in enumerate(players)
            if float(p.get("p_play", 0.0)) > 0
            and int(p.get("element_type", POS_MID)) in MODELLED_POSITIONS
        ]

        # Appearances are drawn up front, because allocating a team's goals
        # needs to know who was on the pitch in that simulation.
        played_m = np.zeros((n_players, n), dtype=bool)
        played60_m = np.zeros((n_players, n), dtype=bool)
        for i in active:
            p = players[i]
            p_play = float(p["p_play"])
            played_m[i] = rng.random(n) < p_play
            cond60 = min(1.0, float(p.get("p_60", 0.0)) / max(1e-9, p_play))
            played60_m[i] = played_m[i] & (rng.random(n) < cond60)

        goals_m = np.zeros((n_players, n), dtype=np.int64)
        assists_m = np.zeros((n_players, n), dtype=np.int64)
        for t in sim_teams:
            idx = [i for i in active if players[i]["team"] == t]
            if not idx:
                continue
            opp = opponent[t]
            xg = np.array([float(players[i].get("xg_cond", 0.0)) for i in idx])
            xa = np.array([float(players[i].get("xa_cond", 0.0)) for i in idx])
            if opp is None:
                for row, i in enumerate(idx):
                    goals_m[i] = rng.poisson(lam=max(0.0, xg[row]), size=n) * played_m[i]
                    assists_m[i] = rng.poisson(lam=max(0.0, xa[row]), size=n) * played_m[i]
                continue
            team_goals = conceded[opp]
            lam_scored = lam_conceded[opp]
            w_g = np.maximum(xg, 0.0)[:, None] * played_m[idx]
            goals_m[idx] = _allocate(rng, team_goals, w_g, lam_scored)
            # Not every goal is assisted, and the assister is drawn from the
            # same team total, so a big win lifts creators alongside scorers.
            assisted = rng.binomial(team_goals, ASSISTS_PER_GOAL)
            w_a = np.maximum(xa, 0.0)[:, None] * played_m[idx]
            assists_m[idx] = _allocate(rng, assisted, w_a, lam_scored * ASSISTS_PER_GOAL)

        bps = np.zeros((len(players), n), dtype=float)
        pts = np.zeros((len(players), n), dtype=float)
        appeared = np.zeros((len(players), n), dtype=bool)

        for i, p in enumerate(players):
            et = int(p.get("element_type", POS_MID))
            team = p["team"]
            p_play = float(p.get("p_play", 0.0))

            if p_play <= 0 or et not in MODELLED_POSITIONS:
                continue

            # Appearances and attacking returns were drawn above: goals and
            # assists come out of a shared team total, not a private Poisson.
            played = played_m[i]
            played_60 = played60_m[i]
            goals = goals_m[i]
            assists = assists_m[i]

            # E[minutes | played] / 90. Rates below are conditional on playing,
            # so the unconditional p_play is applied by the `played` mask alone.
            play_frac = float(p.get("play_frac", 0.0))
            if play_frac <= 0:
                play_frac = min(1.0, float(p.get("xmin", 0.0)) / max(1e-9, p_play * 90.0))
            play_frac = min(1.0, max(0.0, play_frac))

            # A player is only exposed to goals conceded while they are on the
            # pitch. Thinning the team's shared draw keeps the correlation
            # between team-mates that the shared draw exists to create, while
            # still charging a substitute only for the part of the match they
            # actually played. Using the full-match total made every partial
            # appearance look like a full one.
            #
            # The exposure has to match the appearance actually drawn, not the
            # average one: a player who went 60+ in this sim saw more of the
            # match than `play_frac` (which averages in the short cameos), and
            # using the average made the clean sheet too generous for exactly
            # the long shifts that qualify for one.
            long_frac = min(1.0, max(play_frac, float(p.get("cond_frac", play_frac))))
            short_frac = min(long_frac, SHORT_APPEARANCE_MINUTES / 90.0)
            exposure = np.where(played_60, long_frac, short_frac)
            team_conceded = rng.binomial(conceded[team], exposure) * played
            # The concession deduction applies whenever the player is on the
            # pitch, not only after 60 minutes: gating it on `played_60` gave
            # every sub a free pass on goals conceded.
            cs = (team_conceded == 0) & played_60

            yc = rng.random(n) < min(0.9, float(p.get("yc90", 0.0)) * play_frac)
            yc = yc & played
            rc = (rng.random(n) < min(0.5, float(p.get("rc90", 0.0)) * play_frac)) & played
            og = rng.poisson(lam=max(0.0, float(p.get("og90", 0.0)) * play_frac), size=n) * played
            pen_miss = rng.poisson(
                lam=max(0.0, float(p.get("pen_miss90", 0.0)) * play_frac), size=n) * played

            saves = np.zeros(n, dtype=int)
            pen_saves = np.zeros(n, dtype=int)
            if et == POS_GKP:
                saves = rng.poisson(lam=max(0.0, float(p.get("saves90", 0.0))
                                            * play_frac), size=n) * played
                pen_saves = rng.poisson(
                    lam=max(0.0, float(p.get("pen_save90", 0.0)) * play_frac), size=n) * played

            # Defensive contributions: negative binomial on the per-match count.
            defcon_hit = np.zeros(n, dtype=bool)
            mu_dc = float(p.get("defcon_mu", 0.0))
            if et in DEFCON_THRESHOLD and mu_dc > 0:
                nb_n, nb_p = _nb_params(mu_dc, float(p.get("defcon_dispersion", 1.85)))
                counts = rng.negative_binomial(max(1e-3, nb_n), min(0.999, max(1e-6, nb_p)), size=n)
                defcon_hit = (counts >= DEFCON_THRESHOLD[et]) & played

            # ---- FPL points ----
            player_pts = played * 1.0 + played_60 * 1.0
            player_pts = player_pts + goals * POINTS_GOAL.get(et, 4) + assists * POINTS_ASSIST
            player_pts = player_pts + cs * POINTS_CLEAN_SHEET.get(et, 0)
            player_pts = player_pts + defcon_hit * 2.0
            player_pts = player_pts + yc * POINTS_YELLOW + rc * POINTS_RED
            player_pts = player_pts + og * POINTS_OWN_GOAL + pen_miss * POINTS_PENALTY_MISS
            if et == POS_GKP:
                player_pts = player_pts + (saves // 3) * 1.0
                player_pts = player_pts + pen_saves * POINTS_PENALTY_SAVE
            if et in (POS_GKP, POS_DEF):
                player_pts = player_pts - (team_conceded // 2) * 1.0

            # ---- BPS ----
            b = np.zeros(n, dtype=float)
            b += played_60 * BPS_PLAYING_60
            b += (played & ~played_60) * BPS_PLAYING_UNDER_60
            b += goals * BPS_GOAL.get(et, 18)
            b += assists * BPS_ASSIST
            b += cs * BPS_CLEAN_SHEET.get(et, 0)
            b += saves * BPS_SAVE
            b += pen_saves * BPS_PENALTY_SAVE
            b += yc * BPS_YELLOW + rc * BPS_RED
            b += og * BPS_OWN_GOAL + pen_miss * BPS_PENALTY_MISS
            b += defcon_hit * BPS_DEFCON
            if et in (POS_GKP, POS_DEF):
                b += (team_conceded // 2) * BPS_GOALS_CONCEDED_PER_2
            # Volume terms scaled by minutes actually played.
            min_frac = (played_60 * 1.0 + (played & ~played_60) * 0.4)
            b += min_frac * float(p.get("recoveries90", 0.0)) * BPS_PER_RECOVERY
            b += min_frac * float(p.get("tackles90", 0.0)) * BPS_PER_TACKLE_WON
            b += min_frac * float(p.get("key_passes90", 0.0)) * BPS_PER_KEY_PASS

            bps[i] = b
            pts[i] = player_pts
            appeared[i] = played

        # ---- bonus by rank within the match ----
        #
        # Real BPS is an integer, so ties for a bonus place are common and FPL
        # *shares* the award rather than breaking the tie:
        #     tie for 1st  -> 3, 3, then 1 to the next player (2 is skipped)
        #     tie for 2nd  -> 3, 2, 2   (no 1 awarded)
        #     tie for 3rd  -> 3, 2, 1, 1
        #     3-way for 1st-> 3, 3, 3
        # Simulating BPS as a continuous quantity made exact ties vanish, so
        # every one of these was silently resolved by array position instead.
        # Rounding to integers restores the real tie frequency, and the award
        # below follows the published rules.
        bonus = np.zeros_like(bps)
        masked = np.where(appeared, np.rint(bps), -np.inf)

        v1 = masked.max(axis=0)
        c1 = np.where(np.isfinite(v1), (masked == v1).sum(axis=0), 0)

        second = np.where(masked < v1, masked, -np.inf)
        v2 = second.max(axis=0)
        c2 = np.where(np.isfinite(v2), (second == v2).sum(axis=0), 0)

        third = np.where(second < v2, second, -np.inf)
        v3 = third.max(axis=0)

        top1 = (masked == v1) & np.isfinite(v1)
        top2 = (second == v2) & np.isfinite(v2)
        top3 = (third == v3) & np.isfinite(v3)

        bonus += np.where(top1, 3.0, 0.0)
        # A unique leader leaves the 2-point place open; exactly two leaders
        # consume it, so the next distinct score drops straight to 1.
        bonus += np.where(top2 & (c1 == 1), 2.0, 0.0)
        bonus += np.where(top2 & (c1 == 2), 1.0, 0.0)
        # The 1-point place survives only if both places above it were unique.
        bonus += np.where(top3 & (c1 == 1) & (c2 == 1), 1.0, 0.0)

        pts_with_bonus = pts + bonus

        return {
            "bonus": {p["id"]: float(bonus[i].mean()) for i, p in enumerate(players)},
            "points": {p["id"]: pts_with_bonus[i] for i, p in enumerate(players)},
            "mean_points": {p["id"]: float(pts_with_bonus[i].mean()) for i, p in enumerate(players)},
            "variance": {p["id"]: float(pts_with_bonus[i].var()) for i, p in enumerate(players)},
            # Realised attacking means. Allocating a shared team total can only
            # be trusted if it leaves these equal to the xg/xa that went in, so
            # they are exposed for the tests and for diagnostics.
            "mean_goals": {p["id"]: float(goals_m[i].mean()) for i, p in enumerate(players)},
            "mean_assists": {p["id"]: float(assists_m[i].mean()) for i, p in enumerate(players)},
            # Mean BPS among simulations the player actually appeared in, which
            # is the quantity comparable to a realised `bps` in the data. Bonus
            # is a rank statistic over these, so a per-position error here maps
            # straight into a per-position bonus bias.
            "mean_bps": {
                p["id"]: float(bps[i][appeared[i]].mean()) if appeared[i].any() else 0.0
                for i, p in enumerate(players)
            },
            "n_sims": self.n_sims,
        }


def squad_score_draws(
    point_draws: Dict[int, np.ndarray],
    squad_ids: List[int],
    captain_id: Optional[int] = None,
) -> np.ndarray:
    """
    Sum correlated per-player draws into a squad score distribution. Draws for
    players in the same fixture come from the same simulation, so team stacking
    correctly shows up as increased variance.
    """
    if not squad_ids:
        return np.zeros(0)
    lengths = {len(v) for k, v in point_draws.items() if k in squad_ids}
    if not lengths:
        return np.zeros(0)
    n = min(lengths)
    total = np.zeros(n)
    for pid in squad_ids:
        arr = point_draws.get(pid)
        if arr is None:
            continue
        total += arr[:n]
        if captain_id is not None and pid == captain_id:
            total += arr[:n]
    return total
