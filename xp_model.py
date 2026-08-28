import math
import logging
from typing import Dict, Any, List, Optional
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
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

    def fit_team_ratings(self, teams, past_fixtures, current_gw, season=None):
        # We ignore past_fixtures and just fetch live odds
        # Map season integer (e.g. 2023) to season string ("2324") if provided
        season_str = None
        if season:
            season_str = f"{str(season)[2:]}{str(season + 1)[2:]}"
        self.market.fetch_odds(season_str=season_str)
        self.market.fit_team_ratings(fpl_teams=teams)

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
        
        if player.get("penalties_order") == 1:
            player_match_xg += 0.11
            
        defcon_90 = float(player.get("defensive_contribution_per_90", 0.0) or 0.0)
        xDefCon = defcon_90 / 3.0
            
        return {
            "xg": player_match_xg,
            "xa": player_match_xa,
            "p_cs": p_cs,
            "opp_xg": opp_xg,
            "xDefCon": xDefCon
        }

from understat_api import UnderstatMatcher

class EmpiricalBayesRateModel:
    def __init__(self):
        self.understat = UnderstatMatcher()
        self.league_xg_per_shot = 0.10
        self.league_xa_per_kp = 0.11

    def fit(self, fpl_players, season=None):
        # We only run the event loop if we are not already inside one
        try:
            self.understat.fetch_and_map(fpl_players, season=season)
        except RuntimeError:
            pass # Avoid event loop errors in CI if run async
            
        total_shots = 0
        total_xg = 0.0
        total_kp = 0
        total_xa = 0.0
        
        for pid, stats in self.understat.understat_stats.items():
            total_shots += stats['shots']
            total_xg += stats['xg']
            total_kp += stats['key_passes']
            total_xa += stats['xa']
            
        if total_shots > 0:
            self.league_xg_per_shot = total_xg / total_shots
        if total_kp > 0:
            self.league_xa_per_kp = total_xa / total_kp

    def get_shrunk_rates(self, pid: int, raw_xg90: float, raw_xa90: float):
        stats = self.understat.get_player_stats(pid)
        if not stats or stats['minutes'] < 90:
            return raw_xg90, raw_xa90
            
        # Empirical Bayes Shrinkage for xG/Shot
        M_shots = 30.0 
        shots = stats['shots']
        raw_xg_per_shot = stats['xg'] / shots if shots > 0 else 0.0
        shrunk_xg_per_shot = (shots * raw_xg_per_shot + M_shots * self.league_xg_per_shot) / (shots + M_shots)
        shots_per_90 = shots / (stats['minutes'] / 90.0)
        shrunk_xg90 = shots_per_90 * shrunk_xg_per_shot
        
        # Empirical Bayes Shrinkage for xA/KeyPass
        M_kp = 20.0
        kp = stats['key_passes']
        raw_xa_per_kp = stats['xa'] / kp if kp > 0 else 0.0
        shrunk_xa_per_kp = (kp * raw_xa_per_kp + M_kp * self.league_xa_per_kp) / (kp + M_kp)
        kp_per_90 = kp / (stats['minutes'] / 90.0)
        shrunk_xa90 = kp_per_90 * shrunk_xa_per_kp
        
        return shrunk_xg90, shrunk_xa90

class KalmanFormFilter:
    def __init__(self, process_variance=0.05, measurement_variance=0.3):
        self.q = process_variance
        self.r = measurement_variance

    def filter_series(self, observations, fallback_prior):
        if not observations: return fallback_prior
        
        estimate = fallback_prior
        error_cov = 1.0
        
        for obs in observations:
            error_cov += self.q
            kalman_gain = error_cov / (error_cov + self.r)
            estimate = estimate + kalman_gain * (obs - estimate)
            error_cov = (1 - kalman_gain) * error_cov
            
        return estimate

    def predict_match(self, player, history):
        element_type = player.get("element_type", POS_MID)
        raw_xg90 = float(player.get("expected_goals_per_90", 0.0) or 0)
        raw_xa90 = float(player.get("expected_assists_per_90", 0.0) or 0)
        
        if not history:
            return {"xg": raw_xg90, "xa": raw_xa90}
            
        # Extract recent observations, converting strictly to per-90 rates
        xg_obs = []
        xa_obs = []
        for h in history[-5:]:
            mins = h.get("minutes", 0)
            if mins >= 30:
                xg_obs.append(float(h.get("expected_goals", 0) or 0) / (mins / 90.0))
                xa_obs.append(float(h.get("expected_assists", 0) or 0) / (mins / 90.0))
                
        filt_xg = self.filter_series(xg_obs, raw_xg90)
        filt_xa = self.filter_series(xa_obs, raw_xa90)
        
        return {"xg": filt_xg, "xa": filt_xa}


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

class TrueGradientBoostedTree:
    def __init__(self):
        # Use a Poisson objective to properly handle the right-skewed, zero-inflated target distribution
        self.model = HistGradientBoostingRegressor(loss='poisson', max_iter=100, max_depth=4, random_state=42)
        self.is_trained = False
        self.teams_map = {}

    def train(self, all_history, teams):
        self.teams_map = {t["id"]: t for t in teams}
        X = []
        y = []
        for pid, history in all_history.items():
            bps_w, xgi_w, xgc_w, starts_w = [], [], [], []
            for h in history:
                # M-03: Calculate window statistics strictly BEFORE adding current row
                avg_bps = sum(bps_w) / len(bps_w) if bps_w else 0.0
                avg_xgi = sum(xgi_w) / len(xgi_w) if xgi_w else 0.0
                avg_xgc = sum(xgc_w) / len(xgc_w) if xgc_w else 0.0
                avg_starts = sum(starts_w) / len(starts_w) if starts_w else 0.0
                
                was_home = 1.0 if h.get("was_home") else 0.0
                opp_id = h.get("opponent_team", 1)
                fdr = float(self.teams_map.get(opp_id, {}).get("strength") or 3.0)
                
                # Append target row features (dropped value and transfers_balance)
                X.append([avg_bps, was_home, fdr, avg_xgi, avg_xgc, avg_starts])
                y.append(max(0.0, float(h.get("total_points", 0))))
                
                # NOW append current row to the window for the next iteration
                bps_w.append(float(h.get("bps", 0) or 0))
                xgi_w.append(float(h.get("expected_goal_involvements", 0) or 0))
                xgc_w.append(float(h.get("expected_goals_conceded", 0) or 0))
                starts_w.append(float(h.get("starts", 0) or 0))
                
                if len(bps_w) > 4: bps_w.pop(0)
                if len(xgi_w) > 4: xgi_w.pop(0)
                if len(xgc_w) > 4: xgc_w.pop(0)
                if len(starts_w) > 4: starts_w.pop(0)
        
        if len(X) > 50:
            self.model.fit(X, y)
            self.is_trained = True

    def predict_match(self, player, fixture, history):
        if not self.is_trained:
            return 0.0
            
        bps_w = [float(h.get("bps", 0) or 0) for h in history[-4:]] if history else [0.0]
        xgi_w = [float(h.get("expected_goal_involvements", 0) or 0) for h in history[-4:]] if history else [0.0]
        xgc_w = [float(h.get("expected_goals_conceded", 0) or 0) for h in history[-4:]] if history else [0.0]
        starts_w = [float(h.get("starts", 0) or 0) for h in history[-4:]] if history else [0.0]
        
        avg_bps = sum(bps_w) / len(bps_w) if bps_w else 0.0
        avg_xgi = sum(xgi_w) / len(xgi_w) if xgi_w else 0.0
        avg_xgc = sum(xgc_w) / len(xgc_w) if xgc_w else 0.0
        avg_starts = sum(starts_w) / len(starts_w) if starts_w else 0.0
        
        was_home = 1.0 if fixture.get("team_h") == player.get("team") else 0.0
        opp_id = fixture.get("team_a") if was_home else fixture.get("team_h")
        fdr = float(self.teams_map.get(opp_id, {}).get("strength") or 3.0)
        
        pred = self.model.predict([[avg_bps, was_home, fdr, avg_xgi, avg_xgc, avg_starts]])[0]
        return round(max(0.0, pred), 2)

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
        if weights is None:
            # Re-enable the analytical and Bayesian paths (w_dc, w_kf, w_gt)
            weights = (0.60, 0.40, 0.00)
        self.w_dc, self.w_kf, self.w_gt = weights
        self.dc = MarketOddsPredictor()
        self.kf = KalmanFormFilter()
        self.gt = TrueGradientBoostedTree()
        self.mc = MinutesClassifier()
        self.eb = EmpiricalBayesRateModel()
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.is_calibrated = False

    def fit(self, teams, past_fixtures, current_gw, all_history, fpl_players=None, season=None):
        self.dc.fit_team_ratings(teams, past_fixtures, current_gw, season=season)
        self.gt.train(all_history, teams)
        self.mc.train(all_history)
        if fpl_players:
            self.eb.fit(fpl_players, season=season)
            
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
        
        # Apply Empirical Bayes Shrinkage to the player's underlying rates
        raw_xg90 = float(player.get("expected_goals_per_90", 0.0) or 0)
        raw_xa90 = float(player.get("expected_assists_per_90", 0.0) or 0)
        shrunk_xg, shrunk_xa = self.eb.get_shrunk_rates(player["id"], raw_xg90, raw_xa90)
        
        # Override the FPL dictionary so downstream components use the Bayesian shrunk rates
        player_copy = dict(player)
        player_copy["expected_goals_per_90"] = shrunk_xg
        player_copy["expected_assists_per_90"] = shrunk_xa
        
        # Base components
        dc_res = self.dc.predict_match(player_copy, fixture)
        kf_res = self.kf.predict_match(player_copy, history)
        
        # M-01/Bug Fix: Blend at the component level and normalize weights
        comp_sum = self.w_dc + self.w_kf
        if comp_sum > 0:
            xg = (self.w_dc * dc_res["xg"] + self.w_kf * kf_res["xg"]) / comp_sum
            xa = (self.w_dc * dc_res["xa"] + self.w_kf * kf_res["xa"]) / comp_sum
        else:
            xg, xa = 0.0, 0.0
        p_cs = dc_res["p_cs"]
        opp_xg = dc_res["opp_xg"]
        xDefCon = dc_res.get("xDefCon", 0.0)
        
        # Scale attacking returns by the probability of being on the pitch
        min_frac = xMin / 90.0
        xg *= min_frac
        xa *= min_frac
        xDefCon *= min_frac
        
        element_type = player.get("element_type", POS_MID)
        
        xG_pts = xg * POINTS_GOAL.get(element_type, 4)
        xA_pts = xa * POINTS_ASSIST
        
        cs_pts = POINTS_CLEAN_SHEET.get(element_type, 0)
        xCS_pts = p_60 * p_cs * cs_pts
        # Calculate true expected goals conceded penalty E[floor(X/2)] using Poisson PMF
        e_floor_x2 = sum(math.exp(-opp_xg) * (opp_xg**k) / math.factorial(k) * (k // 2) for k in range(10))
        xConc_penalty = e_floor_x2 * min_frac if element_type in [POS_GKP, POS_DEF] else 0.0
        
        # Calculate true expected saves points E[floor(saves/3)] assuming saves ~ Poisson(opp_xg * 2.5)
        expected_saves = opp_xg * 2.5
        e_floor_saves3 = sum(math.exp(-expected_saves) * (expected_saves**k) / math.factorial(k) * (k // 3) for k in range(15))
        xSaves = e_floor_saves3 * min_frac if element_type == POS_GKP else 0.0
        # M-07: Scale xBonus down to approximate the 6-point per-fixture limit
        xBonus = ((xg * 1.5) + (xa * 1.0) + (p_cs * 0.2 * p_60)) * 0.4
        
        math_pts = xApp + xCS_pts + xG_pts + xA_pts + xSaves + xBonus + xDefCon - xConc_penalty
        
        # GBT is a monolithic black box that predicts total points directly
        # It is trained on unconditional points (including 0-minute matches), so no min_frac scaling is needed
        gt_pts = self.gt.predict_match(player, fixture, history)
        
        ensemble_xp = (1.0 - self.w_gt) * math_pts + (self.w_gt * gt_pts)
        return ensemble_xp

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
