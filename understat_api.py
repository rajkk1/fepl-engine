import pandas as pd
import logging
import unicodedata
from thefuzz import process, fuzz
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def normalize_name(name: str) -> str:
    """Remove accents and lower case for better fuzzy matching."""
    if not name: return ""
    name = unicodedata.normalize('NFKD', str(name)).encode('ASCII', 'ignore').decode('utf-8')
    return name.lower().strip()

CUSTOM_OVERRIDES = {
    "degaard": "martin odegaard",
    "white": "ben white",
    "b.fernandes": "bruno fernandes",
    "rodrigo": "rodri",
    "bruno g.": "bruno guimaraes",
    "bernardo": "bernardo silva",
    "raya": "david raya",
    "cunha": "matheus cunha",
    "darwin": "darwin nunez",
    "andreas": "andreas pereira",
    "martinez": "emiliano martinez",
    "ederson m.": "ederson",
    "diogo j.": "diogo jota",
    "a.becker": "alisson",
    "l.paqueta": "lucas paqueta",
    "t.silva": "thiago silva",
    "de cordova-reid": "bobby reid",
    "j.palhinha": "joao palhinha",
    "ruben": "ruben dias",
    "g.jesus": "gabriel jesus",
    "udogie": "iyenoma destiny udogie",
    "vini souza": "vinicius souza",
    "n.semedo": "nelson semedo",
    "alvarez": "edson alvarez",
    "igor": "igor julio",
    "b.badiashile": "benoit badiashile mukinayi",
    "ansu fati": "ansu fati",
    "beyer": "louis beyer",
    "h.bueno": "hugo bueno",
    "e.royal": "emerson",
    "c.doucoure": "cheick oumar doucoure",
    "p.fornals": "pablo fornals",
    "y. chermiti": "youssef chermiti",
    "andersen": "mads juel andersen",
    "dalot": "diogo dalot",
    "muniz": "rodrigo muniz",
    "sarr": "pape sarr",
    "caicedo": "moises caicedo",
    "almiron": "miguel almiron",
    "roerslev": "mads roerslev",
    "cash": "matthew cash",
    "lerma": "jefferson lerma",
    "cucurella": "marc cucurella",
    "vinicius": "carlos vinicius"
}

_GLOBAL_UNDERSTAT_CACHE = {}

class UnderstatMatcher:
    def __init__(self):
        self.understat_players = []
        self.fpl_to_understat = {}
        self.understat_stats = {} # {fpl_id: {"shots": int, "xg": float}}

    def fetch_and_map(self, fpl_players: List[Dict[str, Any]], season=None):
        """Fetch understat data from Vaastav's repo and robustly map it to FPL players."""
        import datetime
        if season is None:
            now = datetime.datetime.now()
            season = now.year if now.month >= 7 else now.year - 1
            
        if season in _GLOBAL_UNDERSTAT_CACHE:
            self.understat_stats = _GLOBAL_UNDERSTAT_CACHE[season]
            return
            
        try:
            # Map season int to Vaastav format (e.g. 2023 -> '2023-24')
            season_str = f"{season}-{str(season+1)[2:]}"
            url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season_str}/understat/understat_player.csv"
            
            df = pd.read_csv(url)
            self.understat_players = df.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to fetch Vaastav understat data: {e}")
            self.understat_players = []
            return

        if not self.understat_players:
            return

        # Prepare understat name lookup
        us_names = {normalize_name(p['player_name']): p for p in self.understat_players}
        us_name_list = list(us_names.keys())
        
        for p in fpl_players:
            pid = p['id']
            fpl_web = normalize_name(p.get('web_name', ''))
            fpl_full = normalize_name(f"{p.get('first_name', '')} {p.get('second_name', '')}")
            
            match_name = None
            score = 0
            
            if fpl_full in us_names:
                match_name = fpl_full
                score = 100
            elif fpl_web in us_names:
                match_name = fpl_web
                score = 100
            elif fpl_web in CUSTOM_OVERRIDES and CUSTOM_OVERRIDES[fpl_web] in us_names:
                match_name = CUSTOM_OVERRIDES[fpl_web]
                score = 100
            else:
                best_match, best_score = process.extractOne(fpl_full, us_name_list, scorer=fuzz.token_sort_ratio)
                if best_score > 85:
                    match_name = best_match
                    score = best_score
                else:
                    alt_match, alt_score = process.extractOne(f"{fpl_web} {p.get('second_name', '')}", us_name_list, scorer=fuzz.token_sort_ratio)
                    if alt_score > 85:
                        match_name = alt_match
                        score = alt_score

            if match_name and score > 85:
                us_p = us_names[match_name]
                self.fpl_to_understat[pid] = us_p['id']
                self.understat_stats[pid] = {
                    "shots": int(us_p.get("shots", 0)),
                    "xg": float(us_p.get("xG", 0.0)),
                    "minutes": int(us_p.get("time", 0)),
                    "xa": float(us_p.get("xA", 0.0)),
                    "key_passes": int(us_p.get("key_passes", 0))
                }
                
        _GLOBAL_UNDERSTAT_CACHE[season] = self.understat_stats
                
    def get_player_stats(self, pid: int) -> Optional[Dict[str, float]]:
        return self.understat_stats.get(pid)
