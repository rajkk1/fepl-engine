import requests
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

BASE_URL = "https://fantasy.premierleague.com/api/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

_cache: Dict[str, Any] = {}

def fetch_json(url: str, use_cache: bool = True) -> Dict[str, Any]:
    """Fetch JSON data from FPL API with caching support."""
    if use_cache and url in _cache:
        return _cache[url]
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        if use_cache:
            _cache[url] = data
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

def get_manager_info(team_id: int) -> Dict[str, Any]:
    """Fetch manager overview, overall rank, and league memberships."""
    url = f"{BASE_URL}entry/{team_id}/"
    return fetch_json(url, use_cache=False)

def get_manager_picks(team_id: int, event_id: int) -> Dict[str, Any]:
    """Fetch squad picks for a specific manager and gameweek."""
    url = f"{BASE_URL}entry/{team_id}/event/{event_id}/picks/"
    return fetch_json(url, use_cache=False)

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
