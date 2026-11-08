"""
A per-gameweek record of what was forecast before each deadline.

Every number the README quotes for the forecast comes from replaying past
seasons, and several of the engine's constants were measured on those same
seasons, so the edge they show may be flattered. The honest test is a season
none of it was tuned on, scored as it happens. That needs two things nothing
used to keep:

  * what the engine forecast for a gameweek *before its deadline*. The weekly
    job overwrites `public/weekly_plan.json`, and that only ever held the
    chosen squad anyway, not the forecast for every player;
  * FPL's own pre-deadline projection, `ep_next`, the only real outside
    benchmark. It lives in bootstrap-static and changes when the deadline
    passes, so it can only be captured before then.

Neither can be reconstructed afterwards. A week that is not archived is lost.

Layout, one directory per season:

    data/forecast_archive/2026-27/gw11_forecast.json   before the deadline
    data/forecast_archive/2026-27/gw11_actual.json     after FPL checks the data

The rules, all enforced here rather than trusted to the caller:

  * a snapshot is only written from a forecast completed before the deadline;
  * while the deadline is ahead the latest snapshot replaces the previous one -
    a later forecast has later news - and once it has passed the file is frozen,
    whatever runs next. The earlier of the deadline recorded in the file and the
    one FPL currently states is used, so a moved deadline cannot thaw it;
  * a degraded (fixture-blind) forecast never replaces a sound one, the same
    rule `publish_gate` applies to the published plan;
  * outcomes are written once, after FPL marks the gameweek `data_checked`, and
    never rewritten.

`score_archive.py` scores the snapshots against the outcomes.
"""
import datetime
import json
import logging
import os
import re
import subprocess
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1

DEFAULT_ARCHIVE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "forecast_archive")

# One row per player. A header plus positional rows keeps a gameweek around
# 80KB rather than twice that with repeated keys, and one row per line keeps a
# day-to-day diff readable. Readers go through `rows()`, which keys by this
# header, so a column added later does not break an older file.
SNAPSHOT_COLUMNS = (
    "id", "web_name", "team", "element_type", "now_cost", "status",
    "chance_of_playing_next_round", "selected_by_percent",
    # FPL's own projection for this gameweek; null when FPL's "next" event is
    # some other gameweek, so it is never scored against the wrong one.
    "ep_next",
    # The engine: expected points, and the minutes forecast behind them.
    "xp", "p_play", "p_60", "xmin", "n_fixtures",
    # Trailing points-per-game strictly before this gameweek, defined exactly as
    # `backtest.build_baselines` defines it, and the minutes `backtest`'s
    # evaluation population is chosen by. Frozen here because the history they
    # come from is only fully available from the live API.
    "ppg", "ppg_n", "mins_last5",
    # Expected points for every gameweek in `horizon_gws`, in order.
    "xp_horizon",
)

ACTUAL_COLUMNS = ("id", "total_points", "minutes", "starts")

_SNAPSHOT_RE = re.compile(r"^gw(\d+)_forecast\.json$")


# ------------------------------------------------------------------ helpers


def _utc(dt: datetime.datetime) -> datetime.datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=datetime.timezone.utc)


def parse_time(raw: Any) -> Optional[datetime.datetime]:
    """FPL's `2026-11-13T18:30:00Z`, or our own ISO stamps, as aware UTC."""
    if not raw:
        return None
    try:
        return _utc(datetime.datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None


def iso(dt: datetime.datetime) -> str:
    return _utc(dt).astimezone(datetime.timezone.utc).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")


def now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _float(value: Any, digits: Optional[int] = None) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return round(out, digits) if digits is not None else out


def season_label(bootstrap: Dict[str, Any],
                 now: Optional[datetime.datetime] = None) -> str:
    """
    `2026-27`, from the year of GW1's deadline.

    Gameweek numbers repeat every season, so the season has to be part of the
    key. Falls back to the calendar (a season starts in August) when the
    events carry no usable deadline.
    """
    events = sorted((e for e in bootstrap.get("events") or [] if e.get("id") is not None),
                    key=lambda e: int(e["id"]))
    first = parse_time(events[0].get("deadline_time")) if events else None
    if first is not None:
        year = first.year
    else:
        now = now or now_utc()
        year = now.year if now.month >= 7 else now.year - 1
    return f"{year}-{(year + 1) % 100:02d}"


def event_deadline(bootstrap: Dict[str, Any], gw: int) -> Optional[str]:
    for e in bootstrap.get("events") or []:
        if e.get("id") is not None and int(e["id"]) == int(gw):
            return e.get("deadline_time")
    return None


def fpl_next_event(bootstrap: Dict[str, Any]) -> Optional[int]:
    """The gameweek FPL's `ep_next` and `chance_of_playing_next_round` refer to."""
    for e in bootstrap.get("events") or []:
        if e.get("is_next") and e.get("id") is not None:
            return int(e["id"])
    return None


def commit_sha() -> Optional[str]:
    """The commit that made the forecast: Actions says, otherwise ask git."""
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=5, cwd=os.path.dirname(os.path.abspath(__file__)))
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def run_reference() -> Optional[Dict[str, str]]:
    """Which Actions run wrote it, when one did."""
    run_id = os.getenv("GITHUB_RUN_ID")
    if not run_id:
        return None
    server = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    repo = os.getenv("GITHUB_REPOSITORY", "")
    out = {"id": run_id}
    if repo:
        out["url"] = f"{server}/{repo}/actions/runs/{run_id}"
    return out


def trailing_form(history: Iterable[Dict[str, Any]], gw: int
                  ) -> Tuple[Optional[float], int, float]:
    """
    (points per game, rows it averages, minutes in the last five gameweeks),
    using only rounds strictly before `gw`.

    Points-per-game is the mean over history *rows*, so a double gameweek is
    two rows - the same definition as `backtest.build_baselines`, which reads
    the same per-fixture rows from vaastav's archive. Keeping it identical is
    what makes the live numbers comparable with the README's.
    """
    pts, recent = [], 0.0
    for h in history or []:
        rnd = h.get("round")
        if rnd is None or int(rnd) >= int(gw):
            continue
        pts.append(_float(h.get("total_points")) or 0.0)
        if int(rnd) >= int(gw) - 5:
            recent += _float(h.get("minutes")) or 0.0
    ppg = sum(pts) / len(pts) if pts else None
    return ppg, len(pts), recent


# ----------------------------------------------------------------- snapshot


def build_snapshot(bootstrap: Dict[str, Any], xp_matrix: Dict[int, Dict[Any, float]],
                   gw: int, horizon_gws: List[int], forecast_at: datetime.datetime,
                   minutes: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None,
                   history: Optional[Dict[int, List[Dict[str, Any]]]] = None,
                   degraded: bool = False, fixture_model: Optional[str] = None,
                   config: Optional[Dict[str, Any]] = None,
                   commit: Optional[str] = None,
                   run: Optional[Dict[str, str]] = None,
                   season: Optional[str] = None) -> Dict[str, Any]:
    """
    Everything forecast for `gw`, for every player, as one archivable document.

    `forecast_at` is when the forecast was complete: every input, FPL's own
    projection included, had been fetched by then. It is what the deadline
    rule is judged against, so pass the time *after* the forecast was built.
    """
    gw = int(gw)
    horizon_gws = [int(g) for g in horizon_gws]
    minutes = minutes or {}
    history = history or {}
    ep_next_gw = fpl_next_event(bootstrap)
    ep_valid = ep_next_gw == gw

    rows = []
    for e in sorted(bootstrap.get("elements") or [], key=lambda e: int(e["id"])):
        pid = int(e["id"])
        xp_row = xp_matrix.get(pid) or {}
        mins = minutes.get((pid, gw)) or {}
        ppg, ppg_n, recent = trailing_form(history.get(pid, []), gw)
        rows.append([
            pid,
            e.get("web_name"),
            e.get("team"),
            e.get("element_type"),
            e.get("now_cost"),
            e.get("status"),
            e.get("chance_of_playing_next_round"),
            _float(e.get("selected_by_percent"), 1),
            _float(e.get("ep_next"), 2) if ep_valid else None,
            _float(xp_row.get(gw, 0.0), 2) or 0.0,
            _float(xp_row.get(f"{gw}_p_play", mins.get("p_play", 0.0)), 4) or 0.0,
            _float(mins.get("p_60", 0.0), 4) or 0.0,
            _float(mins.get("xmin", 0.0), 1) or 0.0,
            int(mins.get("n_fixtures", 0) or 0),
            _float(ppg, 3),
            ppg_n,
            _float(recent, 1),
            [_float(xp_row.get(g, 0.0), 2) or 0.0 for g in horizon_gws],
        ])

    return {
        "format": FORMAT_VERSION,
        "kind": "forecast",
        "season": season or season_label(bootstrap),
        "gameweek": gw,
        "deadline_time": event_deadline(bootstrap, gw),
        "forecast_at": iso(forecast_at),
        "commit": commit,
        "run": run,
        "horizon_gws": horizon_gws,
        "degraded": bool(degraded),
        "fixture_model": fixture_model,
        "ep_next_gw": ep_next_gw,
        "config": config or {},
        "columns": list(SNAPSHOT_COLUMNS),
        "rows": rows,
    }


def snapshot_path(archive_dir: str, season: str, gw: int) -> str:
    return os.path.join(archive_dir, season, f"gw{int(gw):02d}_forecast.json")


def actual_path(archive_dir: str, season: str, gw: int) -> str:
    return os.path.join(archive_dir, season, f"gw{int(gw):02d}_actual.json")


def dumps(doc: Dict[str, Any]) -> str:
    """Pretty header, one compact row per line: small, and diffs by player."""
    head = {k: v for k, v in doc.items() if k != "rows"}
    text = json.dumps(head, indent=2, ensure_ascii=False)
    body = ",\n".join("    " + json.dumps(r, ensure_ascii=False, separators=(",", ":"))
                      for r in doc.get("rows") or [])
    rows = f'"rows": [\n{body}\n  ]' if body else '"rows": []'
    return text[:-2] + f",\n  {rows}\n}}\n"


def _write_atomic(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else None
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return None


def rows(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A snapshot or outcome file's rows as dicts, keyed by its own header."""
    cols = doc.get("columns") or []
    return [dict(zip(cols, r)) for r in doc.get("rows") or []]


def write_snapshot(snapshot: Dict[str, Any], archive_dir: str,
                   now: Optional[datetime.datetime] = None) -> Tuple[bool, str]:
    """
    Write `snapshot` unless a rule forbids it. Returns (written?, why).

    Never raises for a refusal: the caller is the daily job, and the plan must
    still go out.
    """
    now = _utc(now or now_utc())
    gw, season = int(snapshot["gameweek"]), snapshot["season"]
    deadline = parse_time(snapshot.get("deadline_time"))
    forecast_at = parse_time(snapshot.get("forecast_at"))

    if deadline is None:
        return False, f"GW{gw} has no deadline in the bootstrap, so nothing can say it is pre-deadline"
    if forecast_at is None or forecast_at >= deadline:
        return False, (f"the GW{gw} forecast was completed at "
                       f"{snapshot.get('forecast_at')}, not before the deadline "
                       f"{snapshot.get('deadline_time')}; it is not a pre-deadline forecast")

    path = snapshot_path(archive_dir, season, gw)
    existing = load(path) if os.path.exists(path) else None
    if existing is not None:
        # The earlier of the two deadlines: a deadline FPL has since moved
        # later must not reopen a file that was already frozen.
        recorded = parse_time(existing.get("deadline_time"))
        frozen_from = min(d for d in (deadline, recorded) if d is not None)
        if now >= frozen_from:
            return False, (f"GW{gw}'s deadline ({iso(frozen_from)}) has passed; "
                           f"the archived pre-deadline snapshot is final")
        if snapshot.get("degraded") and not existing.get("degraded"):
            return False, (f"this GW{gw} forecast is degraded and the archived one "
                           f"is sound; keeping the sound one")
        prior_at = parse_time(existing.get("forecast_at"))
        if prior_at is not None and prior_at > forecast_at:
            return False, (f"the archived GW{gw} snapshot ({iso(prior_at)}) is newer "
                           f"than this one")
    elif os.path.exists(path) and now >= deadline:
        # Present but unreadable, after the deadline: still not ours to touch.
        return False, f"GW{gw}'s deadline has passed and {path} exists"

    _write_atomic(path, dumps(snapshot))
    verb = "replaced" if existing is not None else "wrote"
    return True, f"{verb} {path}"


# ----------------------------------------------------------------- outcomes


def build_actual(live: Dict[str, Any], season: str, gw: int,
                 fetched_at: datetime.datetime) -> Dict[str, Any]:
    """Outcomes from `event/{gw}/live/`: points, minutes and starts per player."""
    out = []
    for el in live.get("elements") or []:
        if el.get("id") is None:
            continue
        stats = el.get("stats") or {}
        out.append([int(el["id"]), int(stats.get("total_points") or 0),
                    int(stats.get("minutes") or 0), int(stats.get("starts") or 0)])
    out.sort(key=lambda r: r[0])
    return {
        "format": FORMAT_VERSION,
        "kind": "actual",
        "season": season,
        "gameweek": int(gw),
        "fetched_at": iso(fetched_at),
        "source": f"fantasy.premierleague.com/api/event/{int(gw)}/live/",
        "columns": list(ACTUAL_COLUMNS),
        "rows": out,
    }


def archived_gameweeks(archive_dir: str, season: str) -> List[int]:
    try:
        names = os.listdir(os.path.join(archive_dir, season))
    except FileNotFoundError:
        return []
    return sorted(int(m.group(1)) for m in map(_SNAPSHOT_RE.match, names) if m)


def archived_seasons(archive_dir: str) -> List[str]:
    try:
        return sorted(d for d in os.listdir(archive_dir)
                      if re.fullmatch(r"\d{4}-\d{2}", d)
                      and os.path.isdir(os.path.join(archive_dir, d)))
    except FileNotFoundError:
        return []


def gameweek_checked(bootstrap: Dict[str, Any], gw: int) -> bool:
    """Played and with FPL's data check done, so bonus and points are final."""
    for e in bootstrap.get("events") or []:
        if e.get("id") is not None and int(e["id"]) == int(gw):
            return bool(e.get("finished")) and bool(e.get("data_checked"))
    return False


def record_actuals(archive_dir: str, season: str, bootstrap: Dict[str, Any],
                   fetch_live: Callable[[int], Dict[str, Any]],
                   now: Optional[datetime.datetime] = None) -> List[int]:
    """
    Save the outcome of every archived gameweek that FPL has finished checking
    and that has no outcome yet. Returns the gameweeks written.

    The live API forgets a season each summer, so outcomes are saved beside the
    forecasts rather than fetched at scoring time. One request per gameweek,
    once. `bootstrap` must be the live one for `season`.
    """
    if season_label(bootstrap) != season:
        return []
    done = []
    for gw in archived_gameweeks(archive_dir, season):
        path = actual_path(archive_dir, season, gw)
        if os.path.exists(path) or not gameweek_checked(bootstrap, gw):
            continue
        try:
            live = fetch_live(gw)
        except Exception as e:
            logger.warning("Could not fetch GW%d outcomes (%s); will retry next run.",
                           gw, type(e).__name__)
            continue
        doc = build_actual(live, season, gw, now or now_utc())
        if not doc["rows"]:
            logger.warning("GW%d outcomes came back empty; not saving them.", gw)
            continue
        _write_atomic(path, dumps(doc))
        done.append(gw)
    return done


# --------------------------------------------------------------- the daily job


def archive_forecast(archive_dir: str, *, bootstrap: Dict[str, Any],
                     xp_matrix: Dict[int, Dict[Any, float]], horizon_gws: List[int],
                     forecast_at: datetime.datetime,
                     minutes: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None,
                     history: Optional[Dict[int, List[Dict[str, Any]]]] = None,
                     degraded: bool = False, fixture_model: Optional[str] = None,
                     config: Optional[Dict[str, Any]] = None,
                     fetch_live: Optional[Callable[[int], Dict[str, Any]]] = None,
                     now: Optional[datetime.datetime] = None
                     ) -> Tuple[bool, str, List[int]]:
    """Snapshot the gameweek ahead, then save any outcomes that are now final."""
    season = season_label(bootstrap)
    snap = build_snapshot(
        bootstrap, xp_matrix, horizon_gws[0], horizon_gws, forecast_at,
        minutes=minutes, history=history, degraded=degraded,
        fixture_model=fixture_model, config=config, commit=commit_sha(),
        run=run_reference(), season=season)
    written, why = write_snapshot(snap, archive_dir, now=now)
    recorded = (record_actuals(archive_dir, season, bootstrap, fetch_live, now=now)
                if fetch_live else [])
    return written, why, recorded
