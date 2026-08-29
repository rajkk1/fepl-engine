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
   (0 / 1–59 / 60–89 / 90), combined with FPL's availability flags and injury news.
3. **Rates** per 90 — goals, assists, defensive contributions, saves, cards —
   estimated by a time-decayed Gamma-Poisson filter that shrinks toward
   positional and prior-season priors, and conditions on the fixture.
4. **Bonus and risk** from a correlated match simulation (`match_sim.py`). Bonus
   is a rank statistic (3/2/1 to the top three BPS scorers in a match), so it is
   simulated rather than approximated. The same simulation supplies correlated
   per-player point draws, which is what makes team stacking price correctly.
5. **Calibration** (`calibration.py`) — a per-position correction fitted on a
   rolling window of completed gameweeks, so the forecast's spread matches
   reality rather than over-rating its own favourites.

**Optimise** (`optimizer.py`)

A multi-gameweek ILP over the £100.0m budget, 3-per-club cap, valid formations,
free-transfer banking, selling-price mechanics and all four chips (wildcard,
free hit, bench boost, triple captain), including chip timing across the horizon.

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
uv run python backtest.py --seasons 2025-26 --from-gw 8 --to-gw 16
```

The CI gate requires the engine to beat every clean baseline on MAE. On 2025-26
GW6–19 it currently does, on both error and within-position rank correlation.

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
| `--export-json path` | Write the plan as JSON |

Run the tests with `uv run pytest tests/ -q`. They are offline by default; add
`-m network` to include the live API check.

## Repository layout

| File | Role |
|---|---|
| `xp_model.py` | Expected points: rates, minutes, fixtures, assembly |
| `market_odds.py` | Bookmaker odds to team attack/defence ratings |
| `team_mapping.py` | One-to-one FPL ↔ Football-Data club resolution |
| `match_sim.py` | Correlated match simulation: bonus and risk |
| `calibration.py` | Per-position recalibration of expected points |
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
