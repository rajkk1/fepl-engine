import os
import logging
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://fantasy.premierleague.com/",
    "Origin": "https://fantasy.premierleague.com"
}


def create_fpl_session(email: Optional[str] = None, password: Optional[str] = None) -> Optional[requests.Session]:
    """Return configured requests Session for FPL API data retrieval."""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def submit_squad_picks(
    team_id: int,
    starters: List[Dict[str, Any]],
    bench: List[Dict[str, Any]],
    chip: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format squad picks and return clean squad summary.
    """
    total_players = len(starters) + len(bench)
    return {
        "success": True,
        "message": f"Optimal {total_players}-player squad generated for Team {team_id}. You can manually copy the picks into your FPL transfers page."
    }
