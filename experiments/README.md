# experiments/

One-off measurements that answered a specific question and are kept so the
answer can be re-checked rather than re-argued. **None of this runs in CI** —
these are slow, they hit the network, and they are read by a person.

The conclusions they produced are written up in the root README (*Should it take
a hit?*, *When the odds feed is down*) and pinned by `tests/test_hit_price.py`.
The scripts are here so that a future disagreement with those conclusions can be
settled by re-running them.

## Was the engine right to take a −8?

```bash
uv run python experiments/hit_cap_sweep.py           # ~20 min, writes JSON
uv run python experiments/hit_cap_report.py
```

`hit_cap_sweep.py` replays whole seasons at several transfer budgets. The
forecast is computed once per (season, gameweek) and **shared across arms**, so
the only difference between arms is how freely the optimiser may spend — and
every arm therefore builds an identical GW1 squad, which makes each gameweek a
matched pair. `--caps` sweeps the hard limit on hits; `--costs` sweeps the price
the objective believes instead.

`hit_cap_report.py` reads the run in four sections, ordered by how much they
should be trusted:

1. **Validity** — are the arms actually matched, and is the difference explained
   by something mechanical like team value bleeding out through the sell-price
   haircut? If this section fails, nothing below it means anything.
2. **Headline** — paired per-gameweek differences with intervals, plus season
   totals, because the season is the only truly independent unit and three of
   them is very little power. It also reports the autocorrelation of the paired
   weekly differences, so the claim that the iid interval is honest is checked
   rather than assumed.
3. **Per-transfer outcomes, and why they mislead.** Two measures that both look
   strongly positive and both mislead — kept in the report precisely because
   they are the numbers a person naturally reaches for when justifying a hit.
4. **Counterfactual** — what the extra transfers actually bought, by comparing
   the squads the arms end up holding.

The answer, for the record: at the shipped setting the hits are a wash. Every
comparison contains zero and the three seasons disagree on the sign of every
one. Both of the plausible explanations for over-trading turned out to be false,
and the reason the per-transfer numbers look profitable anyway is that "points
of the player in, minus points of the player out" is positive by construction
under any transfer policy at all.

## Is a new odds source what it claims to be?

```bash
uv run python experiments/validate_odds_source.py
```

Run this before trusting a forecast to a new or changed odds source. A second
source is a second chance to get home and away the wrong way round, or to line
the columns up one bookmaker off, and neither announces itself — the forecast
just quietly prices every fixture backwards.

Three checks, each failing loudly on a different corruption: home-win price
against actual home wins (catches a swap or misalignment), implied total goals
against actual (catches an over/under swap or a scaling error), and
favourite-wins rate by price bucket, which must be monotone (catches odds
attached to the wrong match). Deviations around 0.05 at n≈70–100 a bucket are
sampling noise, not a fault.
