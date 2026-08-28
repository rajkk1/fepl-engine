import math
import logging
from typing import Dict, Any, List, Optional
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from fpl_api import get_bootstrap_static, get_fixtures, get_all_element_summaries

logger = logging.getLogger(__name__)

POS_GKP = 1
POS_DEF = 2
POS_MID = 3
POS_FWD = 4
POINTS_GOAL = {POS_GKP: 6, POS_DEF: 6, POS_MID: 5, POS_FWD: 4}
POINTS_CLEAN_SHEET = {POS_GKP: 4, POS_DEF: 4, POS_MID: 1, POS_FWD: 0}
POINTS_ASSIST = 3

from market_odds import MarketOddsModel

class MarketOddsPredictor:
    def __init__(self):
        self.market = MarketOddsModel()

    def fit_team_ratings(self, teams, current_gw_date=None, season_str=None):
        self.market.fetch_odds(season_str=season_str)
        self.market.fit_team_ratings(fpl_teams=teams, current_gw_date=current_gw_date)

    def predict_match(self, player, fixture):
        element_type = player.get("element_type", POS_MID)
        is_home = (fixture.get("team_h") == player.get("team"))
        
        home_id = fixture.get("team_h") if is_home else fixture.get("team_a")
        away_id = fixture.get("team_a") if is_home else fixture.get("team_h")

        lam, mu = self.market.get_match_lambdas(home_id, away_id)
        
        team_xg = lam if is_home else mu
        opp_xg = mu if is_home else lam
        
        # M-05: Divide by the player's own team's baseline xG (average of home/away attacking strength)
        team_att_baseline = self.market.team_ratings.get(player.get("team"), {}).get("scored", 1.4)
        team_baseline = team_att_baseline * 1.0 # The average multiplier is 1.0 since it blends 1.10 and 0.90
        
        p_cs = math.exp(-opp_xg)

        player_xg90 = float(player.get("expected_goals_per_90", 0.0) or 0)
        player_xa90 = float(player.get("expected_assists_per_90", 0.0) or 0)
        
        player_match_xg = player_xg90 * (team_xg / team_baseline)
        player_match_xa = player_xa90 * (team_xg / team_baseline)
            
        return {
            "xg": player_match_xg,
            "xa": player_match_xa,
            "p_cs": p_cs,
            "opp_xg": opp_xg
        }


class KalmanFormFilter:
    def __init__(self, process_variance=0.05, measurement_variance=0.3):
        self.q = process_variance
        self.r = measurement_variance

    def filter_series(self, observations, fallback_prior):
        if not observations: return fallback_prior
        
        estimate = fallback_prior
        error_cov = 1.0
        
        for obs, var_multiplier in observations:
            error_cov += self.q
            current_r = self.r * var_multiplier
            kalman_gain = error_cov / (error_cov + current_r)
            estimate = estimate + kalman_gain * (obs - estimate)
            error_cov = (1 - kalman_gain) * error_cov
            
        return estimate

    def predict_match(self, player, history, market_predictor):
        raw_xg90 = float(player.get("expected_goals_per_90", 0.0) or 0)
        raw_xa90 = float(player.get("expected_assists_per_90", 0.0) or 0)
        
        if not history:
            return {"xg": raw_xg90, "xa": raw_xa90}
            
        xg_obs = []
        xa_obs = []
        player_team = player.get("team")
        
        for h in history[-5:]:
            mins = h.get("minutes", 0)
            if mins >= 10:
                # 1. Measurement variance is inversely proportional to minutes played
                var_multiplier = 90.0 / mins
                
                # 2. Condition on fixture multiplier
                was_home = h.get("was_home")
                opp_id = h.get("opponent_team")
                home_id = player_team if was_home else opp_id
                away_id = opp_id if was_home else player_team
                
                lam, mu = market_predictor.market.get_match_lambdas(home_id, away_id)
                team_xg = lam if was_home else mu
                
                team_att_baseline = market_predictor.market.team_ratings.get(player_team, {}).get("scored", 1.4)
                team_baseline = team_att_baseline * 1.0
                fixture_multiplier = team_xg / team_baseline if team_baseline > 0 else 1.0
                
                obs_xg90 = (float(h.get("expected_goals", 0) or 0) / (mins / 90.0)) / fixture_multiplier
                obs_xa90 = (float(h.get("expected_assists", 0) or 0) / (mins / 90.0)) / fixture_multiplier
                
                xg_obs.append((obs_xg90, var_multiplier))
                xa_obs.append((obs_xa90, var_multiplier))
                
        filt_xg = self.filter_series(xg_obs, raw_xg90)
        filt_xa = self.filter_series(xa_obs, raw_xa90)
        
        return {"xg": filt_xg, "xa": filt_xa}




from sklearn.ensemble import HistGradientBoostingClassifier

class MinutesClassifier:
    def __init__(self):
        self.model = HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)
        self.is_trained = False
        
    def _get_class(self, mins):
        if mins == 0: return 0
        if mins < 60: return 1
        if mins < 90: return 2
        return 3

    def train(self, all_history):
        X, y = [], []
        for pid, history in all_history.items():
            starts_w, mins_w = [], []
            for h in history:
                avg_starts = sum(starts_w) / len(starts_w) if starts_w else 0.0
                avg_mins = sum(mins_w) / len(mins_w) if mins_w else 0.0
                
                cost = float(h.get("value", 50))
                # Treat chance as 100 if missing, which is standard
                chance = 100.0 # Historical chance isn't perfectly recorded in Vaastav, assume 100 for training
                
                X.append([avg_starts, avg_mins, cost, chance])
                y.append(self._get_class(h.get("minutes", 0)))
                
                starts_w.append(float(h.get("starts", 0)))
                mins_w.append(float(h.get("minutes", 0)))
                if len(starts_w) > 4: starts_w.pop(0)
                if len(mins_w) > 4: mins_w.pop(0)
                
        if len(X) > 50:
            self.model.fit(X, y)
            self.is_trained = True

    def predict_proba(self, player, history):
        if not self.is_trained:
            return [0.1, 0.1, 0.1, 0.7] # Default fallback
            
        status = player.get("status", "a")
        chance_raw = player.get("chance_of_playing_next_round")
        
        if status in ["i", "s", "u"] or chance_raw == 0:
            return [1.0, 0.0, 0.0, 0.0]
            
        chance = float(chance_raw) if chance_raw is not None else 100.0
        cost = float(player.get("now_cost", 50))
        
        starts_w = [float(h.get("starts", 0)) for h in history[-4:]] if history else [0.0]
        mins_w = [float(h.get("minutes", 0)) for h in history[-4:]] if history else [0.0]
        
        avg_starts = sum(starts_w) / len(starts_w) if starts_w else 0.0
        avg_mins = sum(mins_w) / len(mins_w) if mins_w else 0.0
        
        proba = self.model.predict_proba([[avg_starts, avg_mins, cost, chance]])[0]
        
        # Scale by actual FPL availability multiplier if flagged
        if chance < 100:
            mult = chance / 100.0
            p0 = 1.0 - mult + (proba[0] * mult)
            return [p0, proba[1]*mult, proba[2]*mult, proba[3]*mult]
            
        return proba.tolist()

class EnsembleForecaster:
    def __init__(self, weights: tuple = None):
        pv = 0.01
        mv = 0.50
        if weights is not None:
            pv, mv = weights
            
        self.dc = MarketOddsPredictor()
        self.kf = KalmanFormFilter(process_variance=pv, measurement_variance=mv)
        self.mc = MinutesClassifier()

    def fit(self, teams, past_fixtures, current_gw, all_history, fpl_players=None, season=None):
        import datetime
        current_gw_date = None
        if past_fixtures:
            current_gw_date = max([pd.to_datetime(f.get("kickoff_time")).tz_convert(None) for f in past_fixtures if f.get("kickoff_time")], default=None)
            
        season_str = None
        if season:
            season_str = f"{str(season)[2:]}{str(season + 1)[2:]}"
            
        self.dc.fit_team_ratings(teams, current_gw_date=current_gw_date, season_str=season_str)
        self.mc.train(all_history)
            
    def _predict_uncalibrated(self, player, fixture, history, dgw_idx=0):
        p_states = self.mc.predict_proba(player, history)
        p_0, p_1_59, p_60_89, p_90 = p_states
        
        # Base probabilities from the classifier
        p_play = p_1_59 + p_60_89 + p_90
        p_60 = p_60_89 + p_90
        xMin = (p_1_59 * 30.0) + (p_60_89 * 75.0) + (p_90 * 90.0)
        
        # Apply fatigue penalty for second matches in a double gameweek
        if dgw_idx > 0:
            xMin = max(0.0, xMin - 22.0)
            p_play *= (xMin / ((p_1_59 * 30.0) + (p_60_89 * 75.0) + (p_90 * 90.0) + 1e-6))
            p_60 = max(0.0, min(1.0, (xMin - 30) / 45.0))
        
        if xMin <= 0: return 0.0
        
        xApp = (p_60 * 2.0) + ((p_play - p_60) * 1.0)
        
        # 1. Base Rate Estimation (Kalman)
        kf_res = self.kf.predict_match(player, history, self.dc)
        
        # 2. Override FPL dictionary so downstream component uses the Kalman shrunk rates
        player_copy = dict(player)
        player_copy["expected_goals_per_90"] = kf_res["xg"]
        player_copy["expected_assists_per_90"] = kf_res["xa"]
        
        # 3. Fixture Multiplier (MarketOdds)
        dc_res = self.dc.predict_match(player_copy, fixture)
        
        xg = dc_res["xg"]
        xa = dc_res["xa"]
        p_cs = dc_res["p_cs"]
        opp_xg = dc_res["opp_xg"]
        
        # 4. Minutes Scaling
        min_frac = xMin / 90.0
        xg *= min_frac
        xa *= min_frac
        
        element_type = player.get("element_type", POS_MID)
        
        xG_pts = xg * POINTS_GOAL.get(element_type, 4)
        xA_pts = xa * POINTS_ASSIST
        
        cs_pts = POINTS_CLEAN_SHEET.get(element_type, 0)
        xCS_pts = p_60 * p_cs * cs_pts
        
        # Calculate expected goals conceded penalty E[floor(X/2)] using Poisson PMF
        # Fix: Scale lambda (opp_xg) by minutes fraction BEFORE computing PMF
        player_opp_xg = opp_xg * min_frac
        e_floor_x2 = sum(math.exp(-player_opp_xg) * (player_opp_xg**k) / math.factorial(k) * (k // 2) for k in range(10))
        xConc_penalty = e_floor_x2 if element_type in [POS_GKP, POS_DEF] else 0.0
        
        # Calculate expected saves points E[floor(saves/3)] assuming saves ~ Poisson(opp_xg * 2.5)
        expected_saves = player_opp_xg * 2.5
        e_floor_saves3 = sum(math.exp(-expected_saves) * (expected_saves**k) / math.factorial(k) * (k // 3) for k in range(15))
        xSaves = e_floor_saves3 if element_type == POS_GKP else 0.0
        # M-07: Scale xBonus down to approximate the 6-point per-fixture limit
        xBonus = ((xg * 1.5) + (xa * 1.0) + (p_cs * 0.2 * p_60)) * 0.4
        
        math_pts = xApp + xCS_pts + xG_pts + xA_pts + xSaves + xBonus - xConc_penalty
        
        return math_pts

    def predict(self, player, fixture, history, dgw_idx=0):
        raw_xp = self._predict_uncalibrated(player, fixture, history, dgw_idx)
        return round(raw_xp, 2)

def generate_xp_matrix(horizon_gws: List[int], bootstrap=None, fixtures=None, all_history=None, weights: tuple = None, season: int = None) -> Dict[int, Dict[int, float]]:
    if bootstrap is None: bootstrap = get_bootstrap_static()
    if fixtures is None: fixtures = get_fixtures()
    
    players = bootstrap.get("elements", [])
    teams = bootstrap.get("teams", [])
    
    current_gw = 1
    for f in fixtures:
        if f.get("finished"):
            current_gw = max(current_gw, f.get("event") or 1)
            
    player_ids = [p["id"] for p in players]
    
    if all_history is None:
        logger.info(f"Fetching historical data for {len(player_ids)} players concurrently...")
        summaries = get_all_element_summaries(player_ids)
        all_history = {pid: summaries.get(pid, {}).get("history", []) for pid in player_ids}
    
    ensemble = EnsembleForecaster(weights=weights)
    past_fixtures = [f for f in fixtures if f.get("finished")]
    ensemble.fit(teams, past_fixtures, current_gw, all_history, fpl_players=players, season=season)
    
    fixture_map = {}
    for fix in fixtures:
        if fix.get("event") in horizon_gws:
            fixture_map.setdefault(fix["event"], {})
            fixture_map[fix["event"]].setdefault(fix["team_h"], []).append(fix)
            fixture_map[fix["event"]].setdefault(fix["team_a"], []).append(fix)

    xp_matrix = {}
    for p in players:
        pid = p["id"]
        xp_matrix[pid] = {}
        history = all_history.get(pid, [])

        for gw in horizon_gws:
            gw_fixes = fixture_map.get(gw, {}).get(p["team"], [])
            if not gw_fixes:
                xp_matrix[pid][gw] = 0.0
            else:
                total_xp = 0.0
                for idx, f in enumerate(gw_fixes):
                    total_xp += ensemble.predict(p, f, history, dgw_idx=idx)
                xp_matrix[pid][gw] = round(total_xp, 2)
    return xp_matrix

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mat = generate_xp_matrix([2])
    print(list(mat.items())[:5])
