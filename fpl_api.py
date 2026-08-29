import requests
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

def get_my_team(team_id: int, cookie: str) -> Dict[str, Any]:
    """Fetch the authenticated team state to retrieve exact selling prices."""
    url = f"{BASE_URL}my-team/{team_id}/"
    return fetch_json(url, use_cache=False, cookie=cookie)

def get_current_gameweek(bootstrap: Optional[Dict[str, Any]] = None) -> int:
    """Determine current or next active gameweek ID."""
    if bootstrap is None:
        bootstrap = get_bootstrap_static()
    
    events = bootstrap.get("events", [])
    for event in events:
        if event.get("is_next"):
            return event.get("id")
        if event.get("is_current") and not event.get("finished"):
            return event.get("id")
    
    # Default fallback
    for event in events:
        if not event.get("finished"):
            return event.get("id")
    
    return 1

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
