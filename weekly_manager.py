"""
FPL Weekly Manager CLI
Automates the analysis of your team and prints your weekly transfers, captaincy, and chip recommendations.
"""
import sys
import logging
import os
import argparse
import json
from dotenv import load_dotenv

import fpl_api
from xp_model import generate_xp_matrix
from optimizer import solve_fpl_optimization

# Configure UTF-8 encoding for terminal output
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- selling prices -------------------------------------------------------
#
# FPL pays the purchase price plus HALF of any rise, rounded down; a fall is
# taken in full. So a player bought at 4.0 and now worth 4.1 sells for 4.0 - the
# 0.1 rise is worth nothing - while 7.5 rising to 7.7 sells for 7.6.
#
# The authenticated my-team endpoint states selling prices outright, but it
# needs a cookie. Without one the engine used to fall back to the *current*
# price, over-stating the budget by half of every rise, and recommending
# transfers that FPL then refuses for want of £0.1m. Both halves are public:
# the purchase price is in the transfer history, and for anyone still in the
# squad from the start it is `now_cost - cost_change_start`.


def fpl_selling_price(buy_tenths: int, now_tenths: int) -> int:
    """FPL's selling price, in tenths. Integer division floors the profit."""
    if now_tenths <= buy_tenths:
        return now_tenths
    return buy_tenths + (now_tenths - buy_tenths) // 2


def purchase_prices(squad_ids, elements, transfers):
    """
    What each squad member cost, in tenths.

    A player transferred in more than once takes the most recent price, so the
    history is walked in order. Anyone with no transfer-in has been held since
    the start, and their purchase price is the season-start price.
    """
    by_id = {e["id"]: e for e in elements or []}
    bought = {}
    for t in sorted(transfers or [],
                    key=lambda x: (x.get("event") or 0, x.get("time") or "")):
        pid, cost = t.get("element_in"), t.get("element_in_cost")
        if pid is not None and cost is not None:
            bought[int(pid)] = int(cost)
    out = {}
    for pid in squad_ids or []:
        e = by_id.get(pid)
        if not e:
            continue
        out[pid] = bought.get(pid, int(e["now_cost"]) - int(e.get("cost_change_start", 0) or 0))
    return out


def estimate_sell_prices(squad_ids, elements, transfers):
    """Selling prices in £m, reconstructed from public data."""
    by_id = {e["id"]: e for e in elements or []}
    out = {}
    for pid, buy in purchase_prices(squad_ids, elements, transfers).items():
        now = int(by_id[pid]["now_cost"])
        out[pid] = fpl_selling_price(buy, now) / 10.0
    return out


# --- chip availability ----------------------------------------------------
#
# FPL splits the season in half for chips: the first-half set expires at the
# GW19 deadline and a second set unlocks from GW20. So a wildcard played in GW2
# is gone for the first half but does not rule one out later in the season.
CHIP_HALF_SPLIT_GW = 20
ALL_CHIPS = ("wc", "fh", "bb", "tc")
# FPL's own names for them, as they appear in the API.
CHIP_CODES = {"wildcard": "wc", "freehit": "fh", "bboost": "bb", "3xc": "tc"}
CHIP_LABELS = {"wc": "Wildcard", "fh": "Free Hit", "bb": "Bench Boost",
               "tc": "Triple Captain"}


def _season_half(gw: int) -> int:
    return 1 if int(gw) < CHIP_HALF_SPLIT_GW else 2


def chips_from_history(history, current_gw: int):
    """
    Which chips are still in hand, read from the public chip history.

    Returns (available, used_this_half). A chip played in the *current* half is
    spent; one played in the other half has been replaced by the new set.
    """
    used = {}
    for c in (history or {}).get("chips") or []:
        code = CHIP_CODES.get(c.get("name"))
        event = c.get("event")
        if not code or not event:
            continue
        if _season_half(event) == _season_half(current_gw):
            used[code] = int(event)
    return [c for c in ALL_CHIPS if c not in used], used


def _parse_used_chips(raw: str):
    """`--used-chips wc,bb` -> {"wc", "bb"}, rejecting anything unrecognised."""
    out = set()
    for tok in (raw or "").replace(",", " ").split():
        t = tok.strip().lower()
        if t not in ALL_CHIPS:
            raise ValueError(
                f"unknown chip {tok!r}; expected any of {', '.join(ALL_CHIPS)}")
        out.add(t)
    return out


def get_manager_team_state(team_id: int, current_gw: int):
    """Attempt to fetch manager's latest team state, bank balance, and selling prices."""
    try:
        cookie = os.getenv("FPL_COOKIE")
        squad_ids = []
        bank = 0.0
        ft = 100 if current_gw == 1 else 1
        sell_prices = {}
        available_chips = ["wc", "fh", "bb", "tc"]
        
        if cookie:
            try:
                my_team_data = fpl_api.get_my_team(team_id, cookie)
                squad_ids = [p["element"] for p in my_team_data.get("picks", [])]
                if "transfers" in my_team_data:
                    ft = max(0, my_team_data["transfers"].get("limit", 1) - my_team_data["transfers"].get("made", 0))
                    if "bank" in my_team_data["transfers"]:
                        bank = my_team_data["transfers"]["bank"] / 10.0
                for p in my_team_data.get("picks", []):
                    if "selling_price" in p:
                        sell_prices[p["element"]] = p["selling_price"] / 10.0
                
                if "chips" in my_team_data:
                    available_chips = []
                    for c in my_team_data["chips"]:
                        if c.get("status_for_entry") in ["available", "active"]:
                            n = c.get("name")
                            if n == "wildcard": available_chips.append("wc")
                            if n == "freehit": available_chips.append("fh")
                            if n == "bboost": available_chips.append("bb")
                            if n == "3xc": available_chips.append("tc")
                
                return squad_ids, bank, ft, sell_prices, available_chips
            except Exception as auth_err:
                logging.warning(f"Failed to fetch authenticated my-team endpoint: {auth_err}")
                
        # Public fallback. Chip state comes from the public history endpoint
        # rather than being assumed: this path used to return the optimistic
        # default and recommend a wildcard that had already been played.
        picks_data = fpl_api.get_manager_picks(team_id, current_gw - 1 if current_gw > 1 else 1)
        squad_ids = [p["element"] for p in picks_data.get("picks", [])]
        bank = picks_data.get("entry_history", {}).get("bank", 0) / 10.0
        ft = 100 if current_gw == 1 else 1
        available_chips = chip_state(team_id, current_gw)
        # Selling prices reconstructed from public data rather than defaulted to
        # the current price, which over-stated the budget by half of every rise.
        try:
            elements = fpl_api.get_bootstrap_static().get("elements", [])
            sell_prices = estimate_sell_prices(
                squad_ids, elements, fpl_api.get_manager_transfers(team_id))
            over = sum(
                next(e["now_cost"] for e in elements if e["id"] == pid) / 10.0 - sp
                for pid, sp in sell_prices.items())
            if over > 0:
                logging.info(
                    "Selling prices reconstructed from public data: £%.1fm less "
                    "than current prices, because FPL pays only half of a rise.",
                    over)
        except Exception as e:
            logging.error(
                "Could not reconstruct selling prices (%s). Falling back to "
                "current prices, which OVER-STATES your budget - a recommended "
                "transfer may be rejected for want of a tenth. Set FPL_COOKIE "
                "for exact figures.", e)
            sell_prices = {}
        return squad_ids, bank, ft, sell_prices, available_chips
    except Exception:
        # Pre-season or no team found. Chips are still worth checking, and if
        # even that fails we assume NONE rather than all: recommending a chip
        # the manager does not hold is a worse failure than missing one.
        return None, 100.0, 100, {}, chip_state(team_id, current_gw)


def chip_state(team_id: int, current_gw: int):
    """Chips still in hand, or an empty list if it cannot be established."""
    try:
        available, used = chips_from_history(
            fpl_api.get_manager_history(team_id), current_gw)
        if used:
            spent = ", ".join(f"{CHIP_LABELS[c]} (GW{gw})"
                              for c, gw in sorted(used.items(), key=lambda kv: kv[1]))
            logging.info("Chips already played this half: %s", spent)
        return available
    except Exception as e:
        logging.error(
            "Could not read chip history (%s). Assuming NO chips are available "
            "so nothing is recommended that you may not hold; pass --chip to "
            "force one, or --used-chips to say what is spent.", e)
        return []

def print_separator(char="=", length=70):
    print(char * length)


def _score_percentiles(point_draws, gw, starters, captain):
    """
    10th / 50th / 90th percentile of the starting XI's score for `gw`.

    Returns None when no draws are available (which is the case whenever the
    forecast came from a path that did not request them).
    """
    if not point_draws or not starters:
        return None
    try:
        import numpy as np
        from match_sim import squad_score_draws

        by_pid = {pid: arr for (pid, g), arr in point_draws.items() if g == gw}
        ids = [p["id"] for p in starters if p["id"] in by_pid]
        if not ids:
            return None
        cap_id = captain["id"] if captain and captain["id"] in by_pid else None
        totals = squad_score_draws(by_pid, ids, captain_id=cap_id)
        if totals is None or len(totals) == 0:
            return None
        return tuple(float(v) for v in np.percentile(totals, [10, 50, 90]))
    except Exception as e:
        logging.debug("Could not compute a score range: %s", e)
        return None

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="FEPL Weekly Action Plan CLI")
    parser.add_argument("--team", type=int, default=4309239, help="Your FPL Team ID (can also be set via FPL_TEAM_ID env var)")
    parser.add_argument("--horizon", type=int, default=5, help="Planning horizon in gameweeks (default: 5)")
    parser.add_argument("--chip", type=str, default="", help="Chip to activate: wc, fh, tc, bb")
    parser.add_argument("--used-chips", type=str, default="",
                        help="Chips you have already played this half of the "
                             "season, e.g. 'wc,bb'. Normally read from your "
                             "public chip history; use this to correct it.")
    parser.add_argument("--ft", type=int, default=None, help="Number of free transfers currently available. Defaults to 1.")
    parser.add_argument("--export-json", type=str, default="", help="Path to export the weekly plan as JSON (e.g., plan.json)")
    parser.add_argument("--risk-aversion", type=float, default=0.0,
                        help="Rank-aware risk weight. 0 = pure expected points; "
                             "0.02-0.10 trades points for a better rank distribution.")
    parser.add_argument("--lineups", type=str, default="",
                        help="Path to a predicted-lineups JSON feed (or set "
                             "FPL_LINEUPS). Minutes dominate FPL point error, so "
                             "this is the single largest accuracy gain available.")
    parser.add_argument("--horizon-decay", type=float, default=None,
                        help="Per-gameweek discount on future expected points "
                             "(default 0.86). Pass 1.0 to weight the horizon "
                             "equally.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Enable per-position recalibration (off by default; "
                             "see calibration.py for why).")
    args = parser.parse_args()

    team_id = os.getenv("FPL_TEAM_ID") or args.team
    if not team_id:
        raise ValueError("Missing required FPL Team ID. Provide --team or set FPL_TEAM_ID in environment.")
    team_id = int(team_id)

    print_separator("=")
    print(f" 🎯 FEPL WEEKLY MANAGER (Team ID: {team_id})")
    print_separator("=")
    print()

    # Fetch foundational data
    try:
        bootstrap = fpl_api.get_bootstrap_static()
        fixtures = fpl_api.get_fixtures()
        elements = bootstrap.get("elements", [])
        player_dict = {p["id"]: p for p in elements}
        teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
    except Exception as e:
        print(f"Error fetching FPL API data: {e}")
        return

    # Determine current Gameweek
    current_gw = fpl_api.get_current_gameweek(bootstrap)
    print(f"🗓️  Current Target Gameweek: GW{current_gw}")
    
    # Get Manager State
    squad_ids, bank, default_ft, sell_prices, available_chips = get_manager_team_state(team_id, current_gw)
    ft = args.ft if args.ft is not None else default_ft

    # A manual override always wins over what the API implied.
    try:
        manually_used = _parse_used_chips(args.used_chips)
    except ValueError as e:
        raise SystemExit(f"--used-chips: {e}")
    if manually_used:
        available_chips = [c for c in available_chips if c not in manually_used]

    # Say plainly what the engine believes it holds. A wrong belief here is the
    # difference between a usable plan and one built on a chip you do not have,
    # and it used to be invisible.
    if available_chips:
        print(f"🃏 Chips available: "
              f"{', '.join(CHIP_LABELS[c] for c in available_chips)}")
    else:
        print("🃏 Chips available: none")
    spent = [c for c in ALL_CHIPS if c not in available_chips]
    if spent:
        print(f"   (already played or unavailable: "
              f"{', '.join(CHIP_LABELS[c] for c in spent)})")
    
    if squad_ids:
        print(f"💰 Current Bank: £{bank:.1f}m | Free Transfers: {ft if ft < 100 else 'Unlimited (Wildcard/Pre-season)'}")
    else:
        print(f"⚠️ No active squad found (Assuming Pre-season/Wildcard state)")
        ft = 100

    print()
    label = "Marginal Expected Rank Value (MERV)" if args.risk_aversion > 0 else "expected points (xP)"
    print(f"⏳ Generating {label} for GW{current_gw} to GW{current_gw + args.horizon - 1}...")
    horizon_gws = list(range(current_gw, current_gw + args.horizon))
    from xp_model import generate_merv_matrix, load_lineup_overrides

    lineups = load_lineup_overrides(args.lineups or None)
    if lineups:
        print(f"📋 Using {len(lineups)} predicted-lineup entries.")
    else:
        print("📋 No predicted-lineups feed configured "
              "(--lineups / FPL_LINEUPS); minutes come from the model alone.")

    point_draws = {}
    xp_matrix = generate_merv_matrix(
        horizon_gws, bootstrap=bootstrap, fixtures=fixtures,
        risk_aversion=args.risk_aversion, calibrate=args.calibrate,
        lineup_overrides=lineups, draws_out=point_draws,
    )

    active_chip = args.chip.strip().lower() if args.chip else ""
    if active_chip and active_chip not in ALL_CHIPS:
        raise SystemExit(f"--chip: unknown chip {args.chip!r}; "
                         f"expected any of {', '.join(ALL_CHIPS)}")
    if active_chip and active_chip not in available_chips:
        # Forcing is allowed - the manager may know better than the API - but it
        # must be a deliberate act rather than a silent assumption.
        logging.warning(
            "%s is being forced with --chip, but your chip history says it is "
            "not available. Proceeding as instructed.", CHIP_LABELS[active_chip])
    active_chip_gw = horizon_gws[0]

    def _solve(chip=None, chip_gw=None):
        return solve_fpl_optimization(
            bootstrap=bootstrap, xp_matrix=xp_matrix, horizon_gws=horizon_gws,
            initial_squad_ids=squad_ids, initial_bank=bank, initial_ft=ft,
            initial_sell_prices=sell_prices,
            max_hits_per_gw=2, active_chip=chip, active_chip_gw=chip_gw,
            **({} if args.horizon_decay is None
               else {"horizon_decay": args.horizon_decay}),
        )

    if not active_chip:
        # Search chip x gameweek, not just chip. The previous version only ever
        # evaluated a chip at the first gameweek of the horizon, so "hold it for
        # a double gameweek" was unreachable, and it crashed on the result:
        # `int(best_chip.split("_")[1])` raised IndexError for any bare chip code.
        print("🤖 Evaluating chip strategies across the horizon...")

        base_res = _solve(None, None)
        base_xp = base_res.get("total_xp", 0.0)

        # Value of waiting: the later in the season, the lower the bar for using
        # a chip now, because there are fewer remaining chances to beat it.
        gws_until_wc1_expiry = max(1, 19 - current_gw) if current_gw <= 19 else max(1, 38 - current_gw)
        gws_until_season_end = max(1, 38 - current_gw)
        chip_thresholds = {
            "tc": 10.0 * (gws_until_season_end / 38.0),
            "bb": 12.0 * (gws_until_season_end / 38.0),
            "fh": 15.0 * (gws_until_season_end / 38.0),
            "wc": 20.0 * (gws_until_wc1_expiry / 19.0),
        }

        # Only consider chips the manager still holds.
        candidates = [c for c in chip_thresholds if c in available_chips]
        skipped = [c for c in chip_thresholds if c not in available_chips]
        if skipped:
            print(f"   (already used / unavailable: {', '.join(sorted(skipped)).upper()})")

        best = {"chip": "", "gw": horizon_gws[0], "gain": 0.0, "res": base_res}
        for c in candidates:
            threshold = chip_thresholds[c]
            for gw in horizon_gws:
                try:
                    c_res = _solve(c, gw)
                except Exception as e:
                    logging.warning("Chip %s @ GW%s failed to solve: %s", c, gw, e)
                    continue
                gain = c_res.get("total_xp", 0.0) - base_xp
                if gain > threshold and gain > best["gain"]:
                    best = {"chip": c, "gw": gw, "gain": gain, "res": c_res}

        active_chip = best["chip"]
        active_chip_gw = best["gw"]
        res = best["res"]
        if active_chip:
            when = "this gameweek" if active_chip_gw == current_gw else f"GW{active_chip_gw}"
            print(f"🔥 Chip recommendation: {active_chip.upper()} in {when} "
                  f"(marginal gain: +{best['gain']:.2f} xP)\n")
        else:
            print("🧠 Solving ILP Optimization Model (Active Chip: None)...\n")
    else:
        print(f"🧠 Solving ILP Optimization Model (Active Chip: {active_chip})...\n")
        res = _solve(active_chip, active_chip_gw)

    if res.get("status") != "Optimal":
        print("⚠️ Warning: ILP Solver did not find a strictly optimal solution.")

    gw1_data = res.get("gameweeks", {}).get(current_gw, {})
    starters = gw1_data.get("starters", [])
    bench = gw1_data.get("bench", [])
    t_in = gw1_data.get("transfers_in", [])
    t_out = gw1_data.get("transfers_out", [])
    hits = gw1_data.get("hits", 0)

    # Enrich player info
    for p in starters + bench:
        p["now_cost"] = player_dict[p["id"]]["now_cost"] / 10.0
        p["status"] = player_dict[p["id"]]["status"]

    # Identify Captain and Vice-Captain
    captain = next((p for p in starters if p.get("is_captain")), starters[0] if starters else None)
    vice_captain = next((p for p in starters if p.get("is_vice_captain")), starters[1] if len(starters) > 1 else captain)

    print_separator("=")
    print(f" 🎯 WEEKLY ACTION PLAN FOR GW{current_gw}")
    print_separator("=")
    
    # Transfers
    print("\n🔄 RECOMMENDED TRANSFERS:")
    if not t_in:
        print("  ✓ Roll Free Transfer (No transfers recommended)")
    else:
        for i in range(len(t_in)):
            p_in = t_in[i]
            p_out = t_out[i] if i < len(t_out) else {"web_name": "Unknown", "cost": 0.0}
            print(f"  [IN]  {p_in['web_name']:<15} (£{p_in['cost']:.1f}m)")
            print(f"  [OUT] {p_out['web_name']:<15} (£{p_out['cost']:.1f}m)")
            print("  -")
    
    if hits > 0:
        print(f"  ⚠️ Taking a point hit penalty: -{hits * 4} pts")

    # Captaincy
    print("\n👑 CAPTAINCY:")
    if captain:
        mult = 3 if active_chip == "tc" else 2
        print(f"  (C)  {captain['web_name']:<15} -> {captain['xp']:.2f} xP ({captain['xp'] * mult:.2f} pts expected)")
    if vice_captain:
        print(f"  (VC) {vice_captain['web_name']:<15} -> {vice_captain['xp']:.2f} xP backup")

    # Chips
    print("\n🃏 CHIP STRATEGY:")
    if active_chip and active_chip_gw != current_gw:
        print(f"  [→] Hold {active_chip.upper()} for GW{active_chip_gw} (do not play it this week)")
    elif active_chip == "wc":
        print("  [✓] Play Wildcard (Unlimited Free Transfers active)")
    elif active_chip == "fh":
        print("  [✓] Play Free Hit (1-GW Squad active)")
    elif active_chip == "tc":
        print(f"  [✓] Play Triple Captain on {captain['web_name'] if captain else 'Captain'}")
    elif active_chip == "bb":
        print("  [✓] Play Bench Boost (All 15 players score points)")
    else:
        print("  [✓] Save Chips for Double Gameweeks")

    # Starting XI
    print("\n⚽ STARTING XI:")
    for idx, p in enumerate(starters, 1):
        cap_str = " (C)" if p.get("is_captain") else (" (VC)" if p.get("is_vice_captain") else "")
        inj_str = " 🏥" if p.get("status") not in ["a", None] else ""
        t_code = teams.get(p.get("team"), "___")
        print(f" {idx:2d}. [{p['web_name']:<16}] ({t_code}) £{p['now_cost']:.1f}m | {p['xp']:.2f} xP{cap_str}{inj_str}")

    # Bench Order - O-03: Sort bench by xP (priority 1-3) but keep GK first
    bench_gkps = [p for p in bench if p.get("element_type") == 1]
    bench_outfield = [p for p in bench if p.get("element_type") != 1]
    bench_outfield.sort(key=lambda item: item.get("xp", 0.0), reverse=True)
    sorted_bench = bench_gkps + bench_outfield

    print("\n🪑 BENCH (Order is strictly priority 1 to 3 after GK):")
    for idx, p in enumerate(sorted_bench, 1):
        pos_str = "GK" if idx == 1 else f"B{idx-1}"
        inj_str = " 🏥" if p.get("status") not in ["a", None] else ""
        t_code = teams.get(p.get("team"), "___")
        print(f" {pos_str}. [{p['web_name']:<16}] ({t_code}) £{p['now_cost']:.1f}m | {p['xp']:.2f} xP{inj_str}")

    print("\n" + "-" * 70)
    print(f" 📊 Expected GW{current_gw} Score: {res.get('gameweeks', {}).get(current_gw, {}).get('gw_xp', sum(p['xp'] for p in starters)):.2f} pts")

    # A single expected score hides how wide the outcome is, and team-mates'
    # scores are correlated, so the spread cannot be recovered by adding up
    # per-player variances. These draws come from the joint match simulation,
    # which is the only place that correlation is represented.
    score_range = _score_percentiles(point_draws, current_gw, starters, captain)
    if score_range:
        p10, p50, p90 = score_range
        print(f" 🎲 Likely range:      {p10:.0f} - {p90:.0f} pts "
              f"(median {p50:.0f}, 10th-90th percentile)")

    print(f" 💰 Remaining Bank: £{gw1_data.get('bank', 0.0):.1f}m")
    print("-" * 70)

    # Export to JSON if flag is set
    if args.export_json:
        export_data = {
            "gameweek": current_gw,
            "team_id": team_id,
            "transfers_in": t_in,
            "transfers_out": t_out,
            "hits": hits,
            "active_chip": active_chip,
            "active_chip_gw": active_chip_gw if active_chip else None,
            "captain": captain,
            "vice_captain": vice_captain,
            "starters": starters,
            "bench": sorted_bench,
            "expected_score": res.get("gameweeks", {}).get(current_gw, {}).get("gw_xp", sum(p["xp"] for p in starters)),
            "remaining_bank": gw1_data.get("bank", 0.0)
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.export_json)), exist_ok=True)
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=4)
        print(f"\n✅ JSON output saved to {args.export_json}")

if __name__ == "__main__":
    main()
