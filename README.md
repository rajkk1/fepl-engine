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
   looked like it *reduced* risk.
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

# the real evaluation: four seasons pooled, gated on the pooled result
uv run python backtest.py --seasons 2022-23 2023-24 2024-25 2025-26 \
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

Pooled over 2022-23 to 2025-26, GW5–38 — **135 gameweeks**:

```
 model                MAE    RMSE    bias     rho    P@15  cap.regret
 fpl_xp_LEAKS       1.548   2.440  +0.176   0.739     5.7        6.15  <- LEAKS
 fepl               1.777   2.736  -0.115   0.586     2.5       10.77
 ppg                1.962   2.887  -0.000   0.460     2.4       10.76
 roll3_mins         1.969   3.072  -0.133   0.526     1.7       12.24
 roll3              2.017   3.075  +0.047   0.516     1.8       12.24
```

**Which of those differences are real.** Paired per-gameweek bootstrap, FEPL
minus each clean baseline, 95% CI over the 135 gameweeks:

| metric | vs `ppg` | vs `roll3` | vs `roll3_mins` |
|---|---|---|---|
| MAE | **−0.185** [−0.199, −0.170] | **−0.240** [−0.257, −0.223] | **−0.192** [−0.209, −0.176] |
| rank correlation | **+0.126** [+0.112, +0.139] | **+0.070** [+0.062, +0.078] | **+0.059** [+0.052, +0.067] |
| precision@15 | +0.111 [−0.148, +0.370] | **+0.719** [+0.422, +1.022] | **+0.815** [+0.519, +1.119] |
| captain regret | +0.015 [−1.081, +1.119] | **−1.474** [−2.600, −0.348] | **−1.467** [−2.644, −0.274] |

Bold is significant. The honest summary: against the rolling-mean baselines the
engine wins on everything, including the two decision metrics. Against **trailing
points-per-game it wins decisively on error and ranking and ties at the top of
the table** — precision@15 and captain regret both still straddle zero after four
seasons. Points-per-game is a much better baseline than it looks: "who has been
scoring" already encodes form and role, which is most of what picks out the
elite.

That tie is the single most useful thing this harness has established. It is not
a gap that one season could have revealed — over 2025-26 alone, precision@15 vs
`ppg` came out at +0.059 [−0.382, +0.500], which is uninformative in both
directions. It is also why the gate enforces MAE and rank correlation only, and
reports the rest as advisory: gating on a statistic whose CI spans zero fails
builds at random, and this repo has already watched that happen — a change whose
true effect on precision@15 was −0.118 [−0.324, +0.059] drifted the metric below
the baseline on noise alone.

**Known bias.** The engine now under-predicts by 0.115 points per player per
gameweek, and unevenly: on 2025-26 keepers run +0.168 while midfielders run
−0.135. Most of that is the single per-team minutes tilt, which cannot correct
the fringe and nailed bands at once — it leaves fringe players slightly
under-allocated and nailed players slightly over. A uniform bias would matter
little to a solver that only ranks, but a *per-position* one distorts choices
between positions, so this is a real open defect rather than a rounding
detail.

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
