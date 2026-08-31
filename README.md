# FEPL Engine

A Fantasy Premier League forecasting and optimisation engine. It projects
expected points per player per gameweek, then solves for the mathematically
optimal squad, transfers, captaincy and chip timing using integer linear
programming.

It runs entirely on GitHub Actions and publishes a static JSON plan — no server,
no hosting cost.

## How it works

**Forecast** (`xp_model.py`)

1. **Team strength** from bookmaker odds (`market_odds.py`). 1X2 and over/under
   2.5 prices are de-vigged and inverted into per-team attack/defence rates,
   time-decayed and shrunk toward the league mean early in a season. Team names
   resolve one-to-one against Football-Data (`team_mapping.py`).
2. **Minutes** from a gradient-boosted classifier over four buckets
   (0 / 1–59 / 60–89 / 90), combined with FPL's availability flags and injury
   news. Features include fixture congestion — days since the team's last match
   and matches played in the previous fortnight — because a five-game rolling
   mean cannot see that those five games came in eighteen days. An optional
   predicted-lineups feed overrides start probability directly; see below.

   The classifier scores players one at a time and has no idea a team fields
   eleven of them, so its output is reconciled against a real team-match. Two
   things were wrong. The 1–59 bucket was valued at its arithmetic midpoint of
   30 minutes when substitute appearances are skewed short and the measured mean
   is **22.2**, overstating every cameo by ~8 minutes. And what remained was
   cameo mass spread across fringe players who never got on: 1059 predicted
   minutes per team-match against a true **985**. A single tilt of the odds of
   appearing, solved per team, fixes the total. Working in odds makes it
   self-targeting — a nailed starter's odds are enormous and barely move, while
   a fringe player at even money absorbs the correction. Team minutes error went
   from +26.9 to −1.3 and the P(plays) calibration gap in the 0.6–0.8 band, the
   rotation risks that decide bench order, from −0.093 to −0.021.
3. **Rates** per 90 — goals, assists, defensive contributions, saves, cards,
   red cards and own goals — estimated by a time-decayed Gamma-Poisson filter
   that shrinks toward positional and prior-season priors, and conditions on the
   fixture.
4. **Penalties and set pieces**, as explicit terms rather than something the
   rate history is left to discover. FPL publishes penalty, corner/indirect and
   direct free-kick takers; that duty predicts returns before it shows up in a
   player's own xG/xA, which matters most for a new signing or someone who has
   just taken the job over. The share the history has already had a chance to
   observe is subtracted back out first, so an established taker is not counted
   twice.
5. **Bonus and risk** from a correlated match simulation (`match_sim.py`). Bonus
   is a rank statistic (3/2/1 to the top three BPS scorers in a match), so it is
   simulated rather than approximated — including FPL's tie rules, which *share*
   a place rather than breaking it (3+3+1, 3+2+2, 3+3+3), so a match can pay out
   more than six. The same simulation supplies correlated per-player point
   draws, which is what makes team stacking price correctly and is where the
   plan's score range comes from.

   Both halves of a match are drawn *jointly*. A team's goals conceded is one
   shared draw, which couples team-mates' clean sheets; the goals it **scores**
   are the very same draw read from the other side of the fixture, allocated
   among its players in proportion to their xG, with assists drawn from that
   same total. Allocating from a shared total leaves every player's expected
   goals exactly where they were — only the joint distribution changes, which
   is the point. Measured against 2023-24..2025-26, two same-club attackers who
   both play 60+ minutes correlate at **+0.098**; before this the engine had
   them at **−0.07**, indistinguishable from opposition players and negative
   only because club-mates compete for the same bonus. Stacking consequently
   looked like it *reduced* risk; it now correctly prices as riskier.

   The team total itself is **not Poisson**. Net of the spread in λ across
   matches, team goals given the market's expectation are *under*-dispersed
   (conditional dispersion 0.864) — fewer 0s, fewer blowouts, a bulge at 2 —
   so the distribution is reweighted cell by cell to the measured shape, with
   the base rate solved so the mean is exactly what the market said. Poisson
   over-states P(concede nothing) by **+0.016**, a clean sheet handed to every
   keeper and defender that reality does not award; the tilted version errs by
   +0.003. The analytic and simulated clean sheet read the same function, so
   they cannot disagree about a defender.
6. **Calibration** (`calibration.py`) — a per-position correction fitted on a
   rolling window of completed gameweeks. **Off by default**: it was built for
   an over-spread model that has since been fixed, and it no longer earns its
   keep (see *Is it any good?*). Enable with `--calibrate`.

**Optimise** (`optimizer.py`)

A multi-gameweek ILP over the £100.0m budget, 3-per-club cap, valid formations,
free-transfer banking, selling-price mechanics and all four chips (wildcard,
free hit, bench boost, triple captain), including chip timing across the horizon.

Future gameweeks are discounted at 0.86 per week (`--horizon-decay`). A forecast
four weeks out carries injuries, rotation, form and fixture reschedules that have
not happened yet, and the plan will be re-solved next week with better
information — so a distant gameweek is a hint about direction, not a commitment.
Weighting the whole horizon equally traded a real point now for a speculative
point in GW+4 at par. Hits are discounted alongside the points they buy, since
charging them at full price while discounting the reward would make the solver
structurally refuse every future transfer.

Bench value is per position, not one number: an outfield sub scores when a
starter in the same position blanks, which is common, while the backup keeper
only plays if the first-choice keeper does not.

**Rank-awareness** (`monte_carlo.py`, `ownership_model.py`)

Optional. With `--risk-aversion > 0` the objective switches from raw expected
points to Marginal Expected Rank Value, which prices a pick by how it moves your
variance *relative to the field*: differentials add variance, template players
remove it. Off by default.

## Is it any good?

`backtest.py` walks forward through a season, rebuilding the API as it looked
before each gameweek, and scores the engine against baselines that are strictly
computable before the deadline — trailing points-per-game and rolling means.

```bash
# fast check, one season
uv run python backtest.py --seasons 2025-26 --from-gw 8 --to-gw 16 --gate

# the real evaluation: three seasons pooled, gated on the pooled result
uv run python backtest.py --seasons 2023-24 2024-25 2025-26 \
  --from-gw 5 --to-gw 38 --gate
```

Seasons are not interchangeable, and the harness says so rather than quietly
averaging them. 2020-21 and 2021-22 carry no `expected_goals`/`expected_assists`
at all, so the attacking model would run on positional priors and measure a
different engine — they are excluded. FPL introduced **defensive contribution**
points in 2025-26, and because the Gamma-Poisson filter correctly treats an
absent column as *missing* rather than zero, it holds the positional prior and
will happily award DefCon points for a season that had no such rule. Scoring is
therefore gated by season. Without that gate, backtesting 2024-25 over-predicts
defenders by **+0.414** points per gameweek and midfielders by **+0.195**, purely
from a rule that did not exist.

**The gate is not MAE alone, and that matters.** Averaged over ~300 players, MAE
is dominated by correctly predicting low scores and players who did not start. A
model can clear every baseline on MAE while being no better than trailing
points-per-game at the two decisions FPL is actually won on — which fifteen
players to own, and who to captain. MAE also rewards shrinking the forecast
toward the mean, which is exactly what blunts the top of the ranking. So the gate
enforces error *and* within-position rank correlation, and reports precision@15
and captain regret alongside them — enforced only where they are stable enough
to enforce, which the table below settles.

Pooled over 2023-24 to 2025-26, GW5–38 — **102 gameweeks**. 2022-23 is excluded:
its xG/xA coverage is partial (see above), and including it both biased and
blurred every number below.

```
 model                MAE    RMSE    bias     rho    P@15  cap.regret
 fpl_xp_LEAKS       1.548   2.440  +0.176   0.739     5.7        6.15  <- LEAKS
 fepl               1.768   2.735  -0.139   0.585     2.7       10.62
 ppg                1.962   2.887  -0.000   0.460     2.4       10.76
 roll3_mins         1.969   3.072  -0.133   0.526     1.7       12.24
 roll3              2.017   3.075  +0.047   0.516     1.8       12.24
```

**Which of those differences are real.** Paired per-gameweek bootstrap, FEPL
minus each clean baseline, 95% CI over the 102 gameweeks:

| metric | vs `ppg` | vs `roll3` | vs `roll3_mins` |
|---|---|---|---|
| **RMSE** (gated) | **−0.154** [−0.177, −0.131] | **−0.346** [−0.371, −0.322] | **−0.345** [−0.368, −0.322] |
| MAE | **−0.166** [−0.181, −0.150] | **−0.223** [−0.240, −0.206] | **−0.177** [−0.193, −0.161] |
| rank correlation | **+0.127** [+0.112, +0.141] | **+0.068** [+0.059, +0.077] | **+0.058** [+0.050, +0.067] |
| precision@15 | +0.265 [+0.000, +0.520] | **+0.873** [+0.569, +1.176] | **+0.971** [+0.667, +1.284] |
| **points captured@15** | **+0.035** [+0.018, +0.052] | **+0.082** [+0.062, +0.102] | **+0.085** [+0.065, +0.105] |
| **NDCG@15** | **+0.034** [+0.016, +0.052] | **+0.083** [+0.060, +0.106] | **+0.087** [+0.065, +0.110] |
| captain regret | −0.755 [−1.843, +0.373] | **−1.941** [−3.275, −0.608] | **−1.922** [−3.265, −0.578] |

Bold is significant. The engine beats every clean baseline on error, on ranking,
and — measured properly — **at the top of the ranking too**.

**Why two metrics for the same thing.** precision@15 vs points-per-game reads
+0.222 [−0.037, +0.481]: not significant, and no more data exists. But that is
partly the metric's own fault, not only a sample-size limit. It is an integer
count out of 15, so it scores a missed 20-point haul the same as a missed
6-pointer and a player ranked 16th the same as one ranked 300th. Replacing the
count with the *share of available top-15 points actually captured* asks the same
question of the same data and answers it: **+0.029 [+0.012, +0.046]**,
significant. NDCG@15, which also weights by position, agrees at +0.026 [+0.006,
+0.045].

Captain regret is the remaining unresolved one, and for the same reason: it reads
a single player's realised score once a gameweek, which is the most volatile
draw in the game.

That tie is the single most useful thing this harness has established. It is not
a gap that one season could have revealed — over 2025-26 alone, precision@15 vs
`ppg` came out at +0.059 [−0.382, +0.500], which is uninformative in both
directions. It is also why the gate enforces MAE and rank correlation only, and
reports the rest as advisory: gating on a statistic whose CI spans zero fails
builds at random, and this repo has already watched that happen — a change whose
true effect on precision@15 was −0.118 [−0.324, +0.059] drifted the metric below
the baseline on noise alone.

**The error gate is RMSE, not MAE, and that is a correction.** FPL points are
heavily right-skewed: over 11,114 scored player-gameweeks the mean is 2.37 and
the median is 1. A forecast minimises MAE at the median and RMSE at the mean, so
on this distribution **MAE structurally rewards under-prediction**:

| constant forecast | MAE | RMSE |
|---|---|---|
| 1.00 | 2.015 | 3.339 |
| 2.00 | 2.071 | 3.069 |
| 2.37 (the mean) | 2.220 | **3.047** |
| 3.00 | 2.478 | 3.112 |

The optimiser sums expected points over a squad, so it needs the conditional
mean. Gating on MAE was rewarding the very bias this engine spent several
commits chasing — and the two metrics eventually disagreed outright: correcting
the expected-assists conversion cut the bias at **every** price band and improved
RMSE, while making MAE significantly worse. MAE is still reported, and is still
the steadier number, but steadiness is not the same as measuring the right thing.

**Expected assists are not FPL assists.** FPL credits winning a scored penalty, a
shot deflected into a scorer's path, and an error forced by the passer; xA
measures only the chance-creating pass. League-wide FPL awards **1.39×** what xA
implies (1.424 / 1.374 / 1.379 across the three clean seasons), while xG tracks
goals almost exactly (0.998 / 0.982 / 0.943). The engine fed xA straight through
as if it were assists. Because the error is multiplicative it was invisible on a
cheap player creating nothing and worth ~0.6 points a gameweek on a £10m creator,
so it surfaced as a price gradient rather than a broken component.

**What bias is left.** Overall −0.034, near enough unbiased. By price band, with
the clean three seasons:

| price band | bias | 95% CI | |
|---|---|---|---|
| under £5.0m | +0.046 | [+0.005, +0.086] | significant |
| £5.0–7.5m | −0.097 | [−0.146, −0.049] | significant |
| £7.5–10.0m | **−0.381** | [−0.570, −0.188] | significant |
| £10.0m+ | −0.048 | [−0.452, +0.350] | **not significant** |

By position: GKP +0.098 [+0.035, +0.159], MID −0.078 [−0.123, −0.032], FWD −0.131
[−0.208, −0.055] all significant; DEF +0.034 not.

The £7.5–10.0m band is now the largest measured defect. An earlier version of
this file called the £10m+ band the biggest problem at −0.515; on clean data it
is **−0.048 and indistinguishable from zero** — that reading was mostly 2022-23's
understated xG, and the band is small enough (~450 player-gameweeks) that its
interval spans ±0.4 either way, so "not significant" here means "cannot tell",
not "fixed".

**Does any of it put points on the board?** Everything above measures the
*forecast*. `simulator.py` measures what the forecast is for: it replays a season
following the engine's own transfers, captaincy and bench order, driving the
**same optimiser** with each forecast so any difference is attributable to the
forecast alone.

```
 season      engine     ppg   roll3   hits
 2023-24       2066    1937    1833     32
 2024-25       2186    2010    1776     16
 2025-26       2020    1964    1629     16
```

Pooled over all three clean seasons — **114 gameweeks**, paired by gameweek:

| baseline | mean | 95% CI | 3-season total | win rate | |
|---|---|---|---|---|---|
| `ppg` | **+3.17** | [+0.13, +6.21] | **+361** | 0.553 | significant |
| `roll3` | **+9.07** | [+5.28, +12.71] | **+1034** | 0.711 | significant |

Read the interval rather than the verdict on the first row. The lower bound is
**+0.13** and the t-test gives p = 0.045; at this effect size significance needs
~107 gameweeks and there are 114, so it clears the bar by a season's margin at
most. Two seasons (76 gameweeks) gave [−0.13, +6.33] and did not clear it. This
is weak evidence for a real end-to-end edge over points-per-game, not a settled
result.

The mean is the right statistic, which is worth recording because the obvious
alternative is wrong. Rank-based tests are the usual answer to a noisy paired
difference, but this one is not heavy-tailed (excess kurtosis −0.41) and the
engine wins *bigger* rather than *more often* — a 55% win rate against a +3.17
mean. A rank test therefore discards the signal: on the two-season sample the
t-test gave p = 0.066, Wilcoxon 0.158, and a sign test 0.909.

Note where much of the margin over `roll3` comes from: the engine takes 16 hits
in a recent season where `roll3` takes 71, and ~53 transfers against ~108. A
forecast that is stable week to week does not churn the squad, and that
discipline is much of the gap rather than better player selection.

The forecast corrections here do carry through. Replaying 2024-25 before and
after the prior and expected-assists fixes, on an unchanged optimiser, moves the
season from **2124 to 2186** (+62) — itself +1.63 pts/gw [−1.63, +4.95], so not
individually significant, but consistent in direction.

**How much room is actually left.** FPL points are extremely noisy, so a perfect
forecast still misses. Estimated from the model's own (well-calibrated)
predictive distribution, the irreducible MAE floor is ≈1.72 — a forecaster who
knew every player's true distribution would still score ~1.72. Against that
floor the baseline's total closable gap is ~0.33 and the engine has taken ~0.18
of it. The remaining headroom in MAE is about 0.15, not 0.5. (That floor is
measured on 2025-26, so it is the right yardstick for that season's 1.858 rather
than the four-season pooled 1.789 — seasons differ in how predictable they are.)

**Where that remaining headroom lives.** Replacing the minutes model with a
perfect oracle of who starts (GW8–16):

```
 variant              MAE     rho    P@15   cap.reg     AUC
 model              1.920   0.600    2.20      9.40   0.877
 oracle_start       1.668   0.752    2.20      8.20   0.960
```

Knowing the XI takes MAE below the modelled-minutes floor and lifts rank
correlation two thirds of the way to the leaking column's ceiling. It does
**nothing** for precision@15. Those are two different problems: MAE and ranking
are a minutes problem, and the top-15 is a *ceiling* problem — everyone in
contention for a haul was always going to start.

**On calibration.** `--calibrate` is off by default, and the reason is more
interesting than "it doesn't work". The fit is real: measured honestly, the
per-position slopes are `{GKP 0.78, DEF 0.77, MID 0.94, FWD 0.86}`, so the raw
forecast *is* over-spread. Applying that correction still costs more than it
returns — MAE 1.914 → 1.932 for +0.1 on precision@15 and nothing on captain
regret. A least-squares fit is dominated by the mass of low-scoring
players, so what it learns is mostly "shrink toward the mean", and shrinking
compresses exactly the top of the ranking where the armband is decided.

Getting those honest slopes required fixing a leak: the calibrator was scoring
retrospective gameweeks with a minutes model and team ratings already trained on
those same gameweeks, which flattered the model and pulled the fitted slope
toward the identity (it previously reported `{0.83, 1.06, 1.01, 1.03}` — i.e.
"nothing to correct"). `EnsembleForecaster.refit_as_of` now rolls the fitted
parameters back per gameweek, not just the history fed to them.

> **A note on FPL's `xP` column.** The backtest reports it, labelled `LEAKS`, but
> never gates on it. That column is scraped after the gameweek and FPL zeroes it
> for players who did not feature — 64–75% of non-participants have it at exactly
> 0.00, giving it ~0.95 AUC at predicting participation. It is not a forecast you
> could have had at the deadline, so treating it as a target is misleading.

## Setup

Fork the repo and set two repository secrets
(`Settings > Secrets and variables > Actions`):

- `FPL_TEAM_ID` — your FPL team ID, from the URL when viewing your points.
- `DISCORD_WEBHOOK_URL` — for deadline alerts.

Optionally set `FPL_COOKIE` to let the engine read your exact selling prices,
bank and remaining chips from the authenticated endpoint. Without it, it falls
back to the public endpoint and assumes all chips are available.

Enable GitHub Pages on the `gh-pages` branch to publish `weekly_plan.json`.

## Local use

```bash
uv sync
uv run python weekly_manager.py --horizon 5
```

Useful flags:

| Flag | Effect |
|---|---|
| `--horizon N` | Planning horizon in gameweeks (default 5) |
| `--chip wc\|fh\|bb\|tc` | Force a chip instead of searching for one |
| `--risk-aversion 0.05` | Enable rank-aware (MERV) valuation |
| `--lineups path.json` | Predicted-lineups feed (see below) |
| `--horizon-decay 0.86` | Per-gameweek discount on future xP; `1.0` weights the horizon equally |
| `--calibrate` | Enable per-position recalibration (off by default) |
| `--export-json path` | Write the plan as JSON |

## Predicted lineups

Minutes dominate FPL point error, and the minutes model tops out around 0.88 AUC
on its own. Knowing who actually starts is the single largest accuracy gain
available to this engine — and it is the one thing the public API cannot tell you
before a deadline.

No lineups feed ships here, and none is invented. What ships is the ingestion
path. Point `--lineups` (or `FPL_LINEUPS`) at a JSON file of start probabilities
keyed by FPL element id:

```json
{"123": 0.95, "456": 0.10}
```

`{"lineups": [{"id": 123, "p_start": 0.95}]}` and the bare list form both work
too. Values are clamped to `[0, 1]` and applied *before* availability, so a feed
cannot start a player FPL has flagged as injured. The overrides are also applied
after calibration fitting, so a feed that only exists for the upcoming gameweek
cannot leak into the retrospective forecasts the calibrator trains on.

Run the tests with `uv run pytest tests/ -q`. They are offline by default; add
`-m network` to include the live API check.

## Repository layout

| File | Role |
|---|---|
| `xp_model.py` | Expected points: rates, minutes, fixtures, set pieces, assembly |
| `market_odds.py` | Bookmaker odds to team attack/defence ratings |
| `team_mapping.py` | One-to-one FPL ↔ Football-Data club resolution |
| `match_sim.py` | Correlated match simulation: bonus and risk |
| `calibration.py` | Per-position recalibration of expected points (off by default) |
| `optimizer.py` | Multi-gameweek ILP for squad, transfers, chips |
| `monte_carlo.py`, `ownership_model.py` | Rank-aware valuation |
| `backtest.py` | Walk-forward evaluation and the CI accuracy gate |
| `simulator.py` | Full-season replay of the engine's own decisions |
| `weekly_manager.py` | CLI entry point |
| `notify.py` | Discord deadline alerts |
| `public/` | Published JSON plan |

`public/index.html` redirects to the raw `weekly_plan.json`; `public/app.html`
deep-links to a companion mobile app that is not part of this repository. There
is no styled web viewer here.

## Licence

MIT — see `LICENSE`.
