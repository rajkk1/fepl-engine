import pandas as pd
import numpy as np
import math
from scipy.stats import poisson
from scipy.optimize import root_scalar
from thefuzz import process, fuzz

_GLOBAL_ODDS_CACHE = {}

_GLOBAL_FPL_TO_FD = {}
_GLOBAL_FD_TO_FPL = {}

class MarketOddsModel:
    def __init__(self):
        self.odds_df = None
        self.team_ratings = {}
        self.FPL_TO_FD = _GLOBAL_FPL_TO_FD
        self.FD_TO_FPL = _GLOBAL_FD_TO_FPL

    def fetch_odds(self, season_str=None):
        import datetime
        if season_str is None:
            now = datetime.datetime.now()
            y1 = now.year if now.month >= 7 else now.year - 1
            y2 = y1 + 1
            season_str = f"{str(y1)[2:]}{str(y2)[2:]}"
            
        if season_str in _GLOBAL_ODDS_CACHE:
            self.odds_df = _GLOBAL_ODDS_CACHE[season_str]
            return
            
        url = f"https://www.football-data.co.uk/mmz4281/{season_str}/E0.csv"
        try:
            df = pd.read_csv(url)
            self.odds_df = df[['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'B365>2.5', 'B365<2.5']].dropna()
            _GLOBAL_ODDS_CACHE[season_str] = self.odds_df
        except Exception as e:
            print(f"Warning: Failed to fetch market odds: {e}")
            
    def implied_total_goals(self, p_over):
        def obj(mu):
            return (1.0 - poisson.cdf(2, mu)) - p_over
        try:
            res = root_scalar(obj, bracket=[0.1, 8.0])
            return res.root
        except ValueError:
            return 2.5
            
    def split_goals(self, mu_total, p_home, p_away):
        def obj(f):
            mu_h = f * mu_total
            mu_a = (1 - f) * mu_total
            return (mu_h / (mu_a + 1e-6)) - (p_home / (p_away + 1e-6))
        try:
            res = root_scalar(obj, bracket=[0.01, 0.99])
            f = res.root
            return f * mu_total, (1 - f) * mu_total
        except ValueError:
            f = p_home / (p_home + p_away)
            return f * mu_total, (1 - f) * mu_total

    def fit_team_ratings(self, fpl_teams=None):
        if self.odds_df is None or len(self.odds_df) == 0:
            return

        fd_teams = list(self.odds_df['HomeTeam'].unique())
        
        if fpl_teams and not self.FPL_TO_FD:
            for t in fpl_teams:
                # Fuzzy match FPL team names to Football-Data names
                best_match, score = process.extractOne(t['name'], fd_teams, scorer=fuzz.token_sort_ratio)
                self.FPL_TO_FD[t['id']] = best_match
                self.FD_TO_FPL[best_match] = t['id']

        team_goals_scored = {k: [] for k in self.FPL_TO_FD.keys()}
        team_goals_conceded = {k: [] for k in self.FPL_TO_FD.keys()}

        for _, row in self.odds_df.iterrows():
            h_id = self.FD_TO_FPL.get(row['HomeTeam'])
            a_id = self.FD_TO_FPL.get(row['AwayTeam'])
            if not h_id or not a_id: continue

            prob_h = 1.0 / row['B365H']
            prob_d = 1.0 / row['B365D']
            prob_a = 1.0 / row['B365A']
            total_1x2 = prob_h + prob_d + prob_a
            p_h, p_d, p_a = prob_h/total_1x2, prob_d/total_1x2, prob_a/total_1x2

            prob_o = 1.0 / row['B365>2.5']
            prob_u = 1.0 / row['B365<2.5']
            total_ou = prob_o + prob_u
            p_over = prob_o / total_ou

            mu_total = self.implied_total_goals(p_over)
            mu_home, mu_away = self.split_goals(mu_total, p_h, p_a)

            team_goals_scored[h_id].append(mu_home)
            team_goals_conceded[h_id].append(mu_away)
            team_goals_scored[a_id].append(mu_away)
            team_goals_conceded[a_id].append(mu_home)

        for tid in self.FPL_TO_FD.keys():
            if len(team_goals_scored[tid]) > 0:
                self.team_ratings[tid] = {
                    "scored": np.mean(team_goals_scored[tid]),
                    "conceded": np.mean(team_goals_conceded[tid])
                }
            else:
                self.team_ratings[tid] = {"scored": 1.4, "conceded": 1.4}

    def get_match_lambdas(self, home_id: int, away_id: int):
        h_att = self.team_ratings.get(home_id, {}).get("scored", 1.4)
        a_def = self.team_ratings.get(away_id, {}).get("conceded", 1.4)
        a_att = self.team_ratings.get(away_id, {}).get("scored", 1.4)
        h_def = self.team_ratings.get(home_id, {}).get("conceded", 1.4)
        
        mu_home = math.sqrt(h_att * a_def) * 1.10
        mu_away = math.sqrt(a_att * h_def) * 0.90
        
        return mu_home, mu_away
