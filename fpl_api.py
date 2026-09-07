import requests
import datetime
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

BASE_URL = "https://fantasy.premierleague.com/api/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

import time

_cache: Dict[str, Any] = {}
CACHE_TTL = 3600 # 1 hour

def fetch_json(url: str, use_cache: bool = True, cookie: Optional[str] = None) -> Dict[str, Any]:
    """Fetch JSON data from FPL API with caching support."""
    if use_cache and url in _cache:
        cached = _cache[url]
        if time.time() - cached["timestamp"] < CACHE_TTL:
            return cached["data"]
    
    try:
        req_headers = HEADERS.copy()
        if cookie:
            req_headers["Cookie"] = cookie
            
        response = requests.get(url, headers=req_headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if use_cache:
            _cache[url] = {"data": data, "timestamp": time.time()}
        return data
    except Exception as e:
        logger.error(f"Error fetching FPL endpoint {url}: {e}")
        raise

def get_bootstrap_static(use_cache: bool = True) -> Dict[str, Any]:
    """Fetch main FPL bootstrap-static dataset (elements, teams, events, element_types)."""
    url = f"{BASE_URL}bootstrap-static/"
    return fetch_json(url, use_cache=use_cache)

def get_fixtures(event_id: Optional[int] = None, use_cache: bool = True) -> List[Dict[str, Any]]:
    """Fetch all fixtures or fixtures for a specific gameweek event."""
    url = f"{BASE_URL}fixtures/"
    if event_id is not None:
        url += f"?event={event_id}"
    return fetch_json(url, use_cache=use_cache)

def get_element_summary(player_id: int, use_cache: bool = True) -> Dict[str, Any]:
    """Fetch detailed stats and past fixtures for a specific player."""
    url = f"{BASE_URL}element-summary/{player_id}/"
    return fetch_json(url, use_cache=use_cache)

import asyncio
import httpx

async def _fetch_all_summaries_async(player_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    async with httpx.AsyncClient(headers=HEADERS, timeout=20.0) as client:
        sem = asyncio.Semaphore(50) # Prevent rate limiting
        
        async def fetch(pid):
            async with sem:
                url = f"{BASE_URL}element-summary/{pid}/"
                for attempt in range(5):
                    try:
                        resp = await client.get(url)
                        if resp.status_code == 429:
                            await asyncio.sleep((2 ** attempt) + 1)
                            continue
                        resp.raise_for_status()
                        return pid, resp.json()
                    except Exception as e:
                        if attempt < 4:
                            await asyncio.sleep((2 ** attempt) + 1)
                            continue
                        logger.error(f"Error async fetching summary for {pid}: {e}")
                        return pid, {}

        tasks = [fetch(pid) for pid in player_ids]
        results = await asyncio.gather(*tasks)
        return dict(results)

MIN_COVERAGE_WARN = 0.95
MIN_COVERAGE_FAIL = 0.70

# Coverage achieved by the most recent call, so callers can record how complete
# the run's inputs were rather than assuming they were perfect.
last_summary_coverage: float = 0.0


def get_all_element_summaries(player_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Fetch all element summaries concurrently with exponential backoff.

    Partial rate-limiting used to abort the entire run. A forecast built from 92%
    of players is far more useful than no forecast at all, so we now degrade and
    record coverage, failing only when the data is too thin to model.
    """
    global last_summary_coverage
    results = asyncio.run(_fetch_all_summaries_async(player_ids))

    total = max(1, len(player_ids))
    valid_count = sum(1 for data in results.values() if data)
    last_summary_coverage = valid_count / total

    if last_summary_coverage < MIN_COVERAGE_FAIL:
        raise RuntimeError(
            f"FPL API heavily rate-limited: only {valid_count}/{total} summaries "
            f"({last_summary_coverage:.1%}). Too sparse to forecast."
        )
    if last_summary_coverage < MIN_COVERAGE_WARN:
        logger.warning(
            "Incomplete player history: %d/%d summaries (%.1f%%). Players with "
            "missing history fall back to positional priors.",
            valid_count, total, 100 * last_summary_coverage,
        )
    return results

def get_manager_info(team_id: int) -> Dict[str, Any]:
    """Fetch manager overview, overall rank, and league memberships."""
    url = f"{BASE_URL}entry/{team_id}/"
    return fetch_json(url, use_cache=False)

def get_manager_picks(team_id: int, event_id: int) -> Dict[str, Any]:
    """Fetch squad picks for a specific manager and gameweek."""
    url = f"{BASE_URL}entry/{team_id}/event/{event_id}/picks/"
    return fetch_json(url, use_cache=False)

def get_manager_history(team_id: int, use_cache: bool = True) -> Dict[str, Any]:
    """
    A manager's gameweek history and, crucially, the chips they have played.

    This is a *public* endpoint. Chip state does not need the authenticated
    my-team call, which is what the engine used to rely on - and when that was
    unavailable it assumed every chip was still in hand.
    """
    url = f"{BASE_URL}entry/{team_id}/history/"
    return fetch_json(url, use_cache)


def get_manager_transfers(team_id: int, use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Every transfer a manager has made, with the price paid for each player in.

    Public, and the missing half of a selling price: FPL pays you the purchase
    price plus half the rise, so the purchase price has to be known. Without
    this the engine assumed it could sell at the current price and over-stated
    the budget by half of every price rise.
    """
    url = f"{BASE_URL}entry/{team_id}/transfers/"
    out = fetch_json(url, use_cache)
    return out if isinstance(out, list) else []


def get_my_team(team_id: int, cookie: str) -> Dict[str, Any]:
    """Fetch the authenticated team state to retrieve exact selling prices."""
    url = f"{BASE_URL}my-team/{team_id}/"
    return fetch_json(url, use_cache=False, cookie=cookie)

def get_current_gameweek(bootstrap: Optional[Dict[str, Any]] = None) -> int:
    """
    The earliest gameweek that can still be acted on.

    Decided by deadline, not by FPL's flags. Once a deadline passes you cannot
    transfer into that gameweek, whatever `is_current` says - and FPL leaves
    `is_current` set on the just-played gameweek until the next deadline, while
    flipping `finished` only once every match is finalised. The previous test
    was `is_current and not finished`, which is true for exactly the window
    between "matches played" and "results confirmed", so the engine planned
    transfers for a gameweek that had been unplayable for days. It published such
    a plan on 2026-09-07 for GW3, whose deadline had passed on 2026-09-04,
    complete with a -8 point hit.

    Iteration order made the flags unusable anyway: the check ran per event in
    order, so `is_current` on an earlier gameweek returned before `is_next` on
    the later one was ever reached.
    """
    if bootstrap is None:
        bootstrap = get_bootstrap_static()

    events = bootstrap.get("events") or []
    now = datetime.datetime.now(datetime.timezone.utc)

    upcoming = []
    for event in events:
        raw = event.get("deadline_time")
        eid = event.get("id")
        if not raw or eid is None:
            continue
        try:
            deadline = datetime.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=datetime.timezone.utc)
        if deadline > now:
            upcoming.append((deadline, int(eid)))
    if upcoming:
        return min(upcoming)[1]

    # No deadline ahead of us: the season is over, or no deadline could be
    # parsed. Fall back to the flags, then to the last gameweek that exists.
    for event in events:
        if event.get("is_next") and event.get("id") is not None:
            return int(event["id"])
    for event in events:
        if event.get("is_current") and event.get("id") is not None:
            return int(event["id"])
    ids = [int(e["id"]) for e in events if e.get("id") is not None]
    return max(ids) if ids else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing FPL API integration...")
    try:
        bs = get_bootstrap_static()
        players = bs.get("elements", [])
        teams = bs.get("teams", [])
        gw = get_current_gameweek(bs)
        print(f"Successfully fetched {len(players)} players, {len(teams)} teams. Next Gameweek: GW{gw}")
    except Exception as err:
        print(f"API Fetch failed: {err}")
