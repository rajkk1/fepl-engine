"""
Read a `hit_cap_sweep.py` run, with the checks that stop it being over-read.

Four sections, in the order they should be trusted:

  1. VALIDITY. Do the arms actually start identical? Is the gap explained by
     something mechanical, like team value bleeding away through the sell-price
     haircut? If these fail, nothing below means anything.
  2. THE HEADLINE. Paired per-gameweek differences, with the season total
     alongside - because the season is the only truly independent unit and
     three of them is very little power.
  3. PER-TRANSFER OUTCOMES, and why they are misleading. Two measures that both
     look strongly positive and both mislead, for reasons worth keeping written
     down.
  4. THE COUNTERFACTUAL. What the extra transfers actually bought, by comparing
     the squads the arms end up holding.

    uv run python experiments/hit_cap_report.py
    uv run python experiments/hit_cap_report.py --results /tmp/x.json
"""
import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HORIZON = 5


def _pts(frame, pid, gw):
    """Points, with a missing row counting as zero.

    `or 0.0` does not do this: numpy NaN is truthy, so `nan or 0.0` is nan and
    a single absent gameweek poisons every aggregate downstream. That produced
    a table of NaNs the first time this ran.
    """
    if pid not in frame.columns or gw not in frame.index:
        return 0.0
    v = frame.at[gw, pid]
    return float(v) if pd.notna(v) else 0.0


def _boot(arr, rng, n=20000):
    arr = np.asarray(arr, dtype=float)
    return np.array([rng.choice(arr, len(arr), replace=True).mean()
                     for _ in range(n)])


def _block_boot(per_season, rng, block=6, n=20000):
    """
    Moving-block bootstrap, resampled within each season and pooled.

    A plain paired bootstrap over gameweeks assumes the weekly differences are
    independent. That is worth checking rather than assuming - the arms build
    the same GW1 squad and then drift apart, so a squad difference in GW20 is
    still there in GW21, and treating 38 weeks as 38 observations would
    understate the interval. Measured, the weekly *point* differences turn out
    not to persist (lag-1 ~0.05), so this lands within rounding of the iid
    interval. Kept because the check is the point.
    """
    out = np.empty(n)
    for i in range(n):
        tot = cnt = 0.0
        for d in per_season:
            k = len(d)
            for _ in range(max(1, int(np.ceil(k / block)))):
                st = rng.integers(0, max(1, k - block + 1))
                seg = d[st:st + block]
                tot += seg.sum()
                cnt += len(seg)
        out[i] = tot / cnt
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="hit_cap_results.json")
    ap.add_argument("--decay", type=float, default=None,
                    help="horizon decay, for the break-even arithmetic")
    ap.add_argument("--arm", default=None,
                    help="which arm sections 3 and 4 examine "
                         "(default: the engine's shipped setting, if present)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)

    import optimizer

    decay = args.decay if args.decay is not None else optimizer.HORIZON_DECAY

    with open(args.results) as f:
        blob = json.load(f)
    label, runs = blob["label"], blob["runs"]
    arms = sorted({v[label] for v in runs.values()})
    seasons = sorted({v["season"] for v in runs.values()})
    base = arms[0]
    rng = np.random.default_rng(0)

    def net_series(season, arm):
        return {h["gw"]: h["net"] for h in runs[f"{season}|{arm}"]["history"]}

    def paired(a, b):
        out = []
        for s in seasons:
            x, y = net_series(s, a), net_series(s, b)
            gws = sorted(set(x) & set(y))
            out.append(np.array([y[g] - x[g] for g in gws], dtype=float))
        return out

    mult = sum(decay ** i for i in range(HORIZON))
    print(f"gain multiplier over {HORIZON} gws at decay {decay}: {mult:.2f}")
    print(f"=> a hit breaks even at a forecast edge of "
          f"{optimizer.HIT_COST / mult:.3f} xP/gw")

    # ---------------------------------------------------------- 1. validity
    print("\n=== 1. VALIDITY ===")
    for s in seasons:
        squads = {a: tuple(sorted(next(
            h for h in runs[f"{s}|{a}"]["history"] if h["gw"] == 1)["squad"]))
            for a in arms}
        n = len(set(squads.values()))
        print(f"  {s}: GW1 squad {'identical' if n == 1 else 'DIFFERENT'} "
              f"across arms ({n} distinct of {len(arms)})"
              + ("" if n == 1 else "  <-- arms are not matched pairs"))
    print("\n  team value, to rule out the sell-price haircut as the mechanism")
    for s in seasons:
        for a in arms:
            h = runs[f"{s}|{a}"]["history"]
            print(f"    {s} {label} {a}: value {h[0]['squad_value']:.1f} -> "
                  f"{h[-1]['squad_value']:.1f} "
                  f"({h[-1]['squad_value'] - h[0]['squad_value']:+.1f})")

    # ---------------------------------------------------------- 2. headline
    print("\n=== 2. HEADLINE ===")
    print(f"  {'':12s}" + "".join(f"{label}={a:<10}" for a in arms))
    for s in seasons:
        print(f"  {s:12s}" + "".join(
            f"{runs[f'{s}|{a}']['total']:6.0f}({runs[f'{s}|{a}']['hits']:2d}h) "
            for a in arms))
    tot = {a: sum(runs[f"{s}|{a}"]["total"] for s in seasons) for a in arms}
    hit = {a: sum(runs[f"{s}|{a}"]["hits"] for s in seasons) for a in arms}
    print(f"  {'pooled':12s}" + "".join(f"{tot[a]:6.0f}({hit[a]:2d}h) "
                                        for a in arms))

    print("\n  autocorrelation of the paired weekly difference "
          "(is the iid interval honest?)")
    for a in arms[1:]:
        ps = paired(base, a)
        acs = []
        for lag in (1, 2, 3, 6):
            num = den = 0.0
            for d in ps:
                z = d - d.mean()
                num += (z[:-lag] * z[lag:]).sum()
                den += (z * z).sum()
            acs.append(num / den if den else 0.0)
        print(f"    {label}={a}: " + "  ".join(
            f"lag{l} {v:+.2f}" for l, v in zip((1, 2, 3, 6), acs)))

    print(f"\n  paired difference vs {label}={base}")
    for a in arms[1:]:
        ps = paired(base, a)
        d = np.concatenate(ps)
        nl, nh = np.percentile(_boot(d, rng), [2.5, 97.5])
        bl, bh = np.percentile(_block_boot(ps, rng), [2.5, 97.5])
        star = "  *" if (bl > 0) or (bh < 0) else ""
        print(f"    {label}={a}: {d.mean():+.3f} pts/gw  iid [{nl:+.3f}, "
              f"{nh:+.3f}]  block [{bl:+.3f}, {bh:+.3f}]  "
              f"season {d.mean() * 38:+.0f}{star}")

    print(f"\n  season totals vs {label}={base}  "
          f"(the only independent unit; n={len(seasons)})")
    for a in arms[1:]:
        t = [runs[f"{s}|{a}"]["total"] - runs[f"{s}|{base}"]["total"]
             for s in seasons]
        agree = "agree" if len({x > 0 for x in t}) == 1 else "DISAGREE"
        print(f"    {label}={a}: " + "  ".join(f"{x:+.0f}" for x in t)
              + f"   mean {np.mean(t):+.0f}   (seasons {agree})")

    # -------------------------------------------------- 3. per-transfer traps
    print("\n=== 3. PER-TRANSFER OUTCOMES (and why they mislead) ===")
    from simulator import fetch_data

    # Sections 3 and 4 describe one arm's behaviour, and the one worth
    # describing is what the engine actually ships - not the most permissive
    # arm in the sweep, which is what this first defaulted to and which made
    # the numbers disagree with the ones written up in the README.
    shipped = 2 if label == "cap" else optimizer.HIT_COST
    if args.arm is not None:
        arm = type(arms[0])(args.arm)
    elif shipped in arms:
        arm = shipped
    else:
        arm = arms[len(arms) // 2]
    print(f"\n  (sections 3 and 4 examine {label}={arm}"
          + ("" if arm == shipped else f", NOT the shipped {shipped}") + ")")
    holds, held_win, fixed_win = [], defaultdict(list), defaultdict(list)
    for s in seasons:
        key = f"{s}|{arm}"
        if key not in runs:
            continue
        hist = runs[key]["history"]
        df_gw, df_players = fetch_data(s)[0], fetch_data(s)[1]
        frame = (df_gw.groupby(["GW", "element"])["total_points"].sum()
                 .unstack(fill_value=np.nan))
        pos = dict(zip(df_players["id"], df_players["element_type"]))
        by_gw = {h["gw"]: h for h in hist}
        last = max(by_gw)

        for h in hist:
            if h["gw"] == 1 or not h["in"]:
                continue                 # GW1 builds the squad
            used = set()
            for rank, pid_in in enumerate(h["in"], start=1):
                kept = 0
                for g in range(h["gw"] + 1, last + 1):
                    if pid_in in by_gw.get(g, {}).get("squad", []):
                        kept += 1
                    else:
                        break
                holds.append(kept + 1)
                # An FPL transfer is always same-position, so pair on position
                # rather than list order.
                outs = [q for q in h["out"]
                        if pos.get(q) == pos.get(pid_in) and q not in used]
                if not outs:
                    continue
                pid_out = outs[0]
                used.add(pid_out)
                paid = rank > (len(h["in"]) - h["hits"])
                for name, span in (
                        ("held", range(h["gw"], min(h["gw"] + kept + 1, last + 1))),
                        ("fixed", range(h["gw"], min(h["gw"] + HORIZON, last + 1)))):
                    gain = (sum(_pts(frame, pid_in, g) for g in span)
                            - sum(_pts(frame, pid_out, g) for g in span))
                    (held_win if name == "held" else fixed_win)[paid].append(gain)

    hh = np.array(holds)
    print(f"  holding duration of a transferred-in player (n={len(hh)}): "
          f"mean {hh.mean():.2f} gws, median {np.median(hh):.0f}")
    print(f"    realised multiplier {np.mean([sum(decay ** i for i in range(k)) for k in hh]):.2f} "
          f"vs the {mult:.2f} the optimiser prices against")

    for title, data, why in (
        ("over the weeks actually held", held_win,
         "INFLATED: the optimiser picks the window in response to the outcome -\n"
         "     a player who hauls is kept so his good weeks all count, one who\n"
         "     blanks is sold so his bad weeks stop. That is optional stopping."),
        (f"over a fixed {HORIZON}-gameweek window", fixed_win,
         "STILL MISLEADING: it conditions on the outgoing player being the\n"
         "     squad's worst-rated asset, so it is positive by construction under\n"
         "     ANY transfer policy, including a worthless one."),
    ):
        print(f"\n  realised gain per transfer, {title}")
        for paid, name in ((False, "free transfers"), (True, "paid transfers")):
            arr = data[paid]
            if not arr:
                continue
            lo, hi = np.percentile(_boot(arr, rng, 10000), [2.5, 97.5])
            cost = optimizer.HIT_COST if paid else 0.0
            print(f"    {name:16s} n={len(arr):3d}  gross {np.mean(arr):+6.2f} "
                  f"[{lo:+.2f}, {hi:+.2f}]  net {np.mean(arr) - cost:+.2f}")
        print(f"     {why}")

    # ------------------------------------------------------ 4. counterfactual
    print("\n=== 4. COUNTERFACTUAL: what the extra transfers bought ===")
    print("  The only comparison that holds the alternative use of the roster")
    print("  slot and the budget fixed is the replay itself.")

    # Compare the examined arm against the most *restrictive* one - the arm that
    # spent least on transfers - since the question is what the extra ones
    # bought. Which end that is depends on the sweep: the tightest budget is the
    # LOWEST cap but the HIGHEST price. This used to take `arms[0]` either way,
    # so on a cost sweep it compared the shipped arm with itself and reported a
    # 15/15 overlap and "got there first in 0" - which reads like a finding
    # rather than a tautology.
    strict = arms[0] if label == "cap" else arms[-1]
    if strict == arm:
        strict = arms[1] if label == "cap" else arms[0]
    if strict == arm:
        print(f"  (only one arm in this sweep; nothing to compare {label}={arm} "
              "against)")
        return 0
    print(f"  ({label}={arm} against the tightest budget swept, {label}={strict})")

    for s in seasons:
        a = {h["gw"]: set(h["squad"]) for h in runs[f"{s}|{strict}"]["history"]}
        b = {h["gw"]: set(h["squad"]) for h in runs[f"{s}|{arm}"]["history"]}
        gws = sorted(set(a) & set(b))
        marks = [g for g in (1, 6, 12, 19, 26, 32, 38) if g in a and g in b]
        print(f"  {s}: overlap /15  " + "  ".join(
            f"gw{g}:{len(a[g] & b[g]):2d}" for g in marks)
            + f"   mean {np.mean([len(a[g] & b[g]) for g in gws]):.1f}")

        first_a, first_b = {}, {}
        for h in runs[f"{s}|{strict}"]["history"]:
            if h["gw"] > 1:
                for p in h["in"]:
                    first_a.setdefault(p, h["gw"])
        for h in runs[f"{s}|{arm}"]["history"]:
            if h["gw"] > 1:
                for p in h["in"]:
                    first_b.setdefault(p, h["gw"])
        shared = set(first_a) & set(first_b)
        ahead = [p for p in shared if first_b[p] < first_a[p]]
        print(f"       {label}={arm} bought {len(first_b)} distinct players, "
              f"{label}={strict} {len(first_a)}; {len(shared)} shared, of which "
              f"{label}={arm} got there first in {len(ahead)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
