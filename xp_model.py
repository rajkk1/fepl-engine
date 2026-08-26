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

class DixonColesModel:
    def __init__(self, decay_rate=0.005):
        self.decay_rate = decay_rate
        self.alpha = {}
        self.beta = {}
        self.gamma = 1.18 # home adv
        self.avg_goals = 1.4

    def fit_team_ratings(self, teams, past_fixtures, current_gw):
        team_ids = [t['id'] for t in teams]
        num_teams = len(team_ids)
        
        match_data = []
        for f in past_fixtures:
            if f.get("finished") and f.get("team_h_score") is not None:
                gw_diff = max(0, current_gw - (f.get("event") or 1))
                weight = math.exp(-self.decay_rate * gw_diff * 7)
                match_data.append((f["team_h"], f["team_a"], f["team_h_score"], f["team_a_score"], weight))
                
        if not match_data:
            for tid in team_ids:
                self.alpha[tid] = 1.0
                self.beta[tid] = 1.0
            return

        def neg_log_likelihood(params):
            alphas = {tid: params[i] for i, tid in enumerate(team_ids)}
            betas = {tid: params[num_teams + i] for i, tid in enumerate(team_ids)}
            gamma = params[2 * num_teams]
            
            nll = 0.0
            for h, a, hg, ag, w in match_data:
                lam = alphas[h] * betas[a] * gamma
                mu = alphas[a] * betas[h]
                ll_h = hg * math.log(max(lam, 1e-5)) - lam
                ll_a = ag * math.log(max(mu, 1e-5)) - mu
                nll -= w * (ll_h + ll_a)
            
            nll += 100 * (sum(alphas.values())/num_teams - 1.0)**2
            nll += 100 * (sum(betas.values())/num_teams - 1.0)**2
            return nll

        init_params = [1.0] * (2 * num_teams) + [1.18]
        bounds = [(0.1, 5.0)] * (2 * num_teams) + [(0.5, 2.0)]
        
        res = minimize(neg_log_likelihood, init_params, bounds=bounds, method="L-BFGS-B")
        
        for i, tid in enumerate(team_ids):
            self.alpha[tid] = res.x[i]
            self.beta[tid] = res.x[num_teams + i]
        self.gamma = res.x[2 * num_teams]

    def predict_match(self, player, fixture, xMin):
        element_type = player.get("element_type", POS_MID)
        is_home = (fixture.get("team_h") == player.get("team"))
        
        home_id = fixture.get("team_h") if is_home else fixture.get("team_a")
        away_id = fixture.get("team_a") if is_home else fixture.get("team_h")

        lam = self.alpha.get(home_id, 1.0) * self.beta.get(away_id, 1.0) * self.gamma
        mu = self.alpha.get(away_id, 1.0) * self.beta.get(home_id, 1.0)
        
        team_xg = lam if is_home else mu
        opp_xg = mu if is_home else lam
        
        p_cs = math.exp(-opp_xg)

        min_frac = xMin / 90.0
        cs_pts = POINTS_CLEAN_SHEET.get(element_type, 0)
        xCS_pts = (p_cs * cs_pts * min_frac) if xMin >= 60 else 0.0
        xConc_penalty = (opp_xg / 2.0 * min_frac) if (element_type in [POS_GKP, POS_DEF] and xMin >= 60) else 0.0

        # 1. Fixture-Adjusted Attacking xP (Allocate team xG to player based on historical share)
        player_xg90 = float(player.get("expected_goals_per_90", 0.0) or 0)
        player_xa90 = float(player.get("expected_assists_per_90", 0.0) or 0)
        
        # Assuming average PL team scores ~1.5 goals per match
        player_match_xg = (player_xg90 / 1.5) * team_xg * min_frac
        player_match_xa = (player_xa90 / 1.5) * team_xg * min_frac
        
        # #6. Penalty Taker Flag: Statistically a team gets ~0.15 penalties per match. 
        # A penalty is ~0.76 xG. So 1st choice takers get an extra ~0.11 xG per match.
        if player.get("penalties_order") == 1:
            player_match_xg += (0.11 * min_frac)
        
        xG_pts = player_match_xg * POINTS_GOAL.get(element_type, 4)
        xA_pts = player_match_xa * POINTS_ASSIST
        
        # 2. Goalkeeper Saves Model (~0.7 save points per expected goal conceded)
        xSaves = (opp_xg * 0.7 * min_frac) if element_type == POS_GKP else 0.0
        
        # 3. Simple Heuristic Bonus Model (Bonus highly correlated with xG, xA, and CS)
        xBonus = (player_match_xg * 1.5) + (player_match_xa * 1.0) + (p_cs * 0.2 if xMin >= 60 else 0.0)

        total_pts = xCS_pts + xG_pts + xA_pts + xSaves + xBonus - xConc_penalty
        return round(max(0.0, total_pts), 2)


class KalmanFormFilter:
    def __init__(self, Q=0.05, R=0.25):
        self.Q = Q
        self.R = R

    def filter_series(self, obs, initial):
        x_hat = initial
        P = 1.0
        for y in obs:
            P_minus = P + self.Q
            K = P_minus / (P_minus + self.R)
            x_hat = x_hat + K * (y - x_hat)
            P = (1 - K) * P_minus
        return max(0.0, x_hat)

    def predict_match(self, player, history, xMin):
        element_type = player.get("element_type", POS_MID)
        
        # FIX: Convert per-match expected goals into per-90 rates so it matches the initial state units!
        xg_obs = [float(h.get("expected_goals", 0) or 0) * 90 / max(1, h.get("minutes", 0)) for h in history if h.get("minutes", 0) > 0]
        xa_obs = [float(h.get("expected_assists", 0) or 0) * 90 / max(1, h.get("minutes", 0)) for h in history if h.get("minutes", 0) > 0]
        
        raw_xg90 = float(player.get("expected_goals_per_90", 0.0) or 0)
        raw_xa90 = float(player.get("expected_assists_per_90", 0.0) or 0)

        filt_xg = self.filter_series(xg_obs, raw_xg90)
        filt_xa = self.filter_series(xa_obs, raw_xa90)
        
        min_frac = xMin / 90.0
        xG_pts = filt_xg * min_frac * POINTS_GOAL.get(element_type, 4)
        xA_pts = filt_xa * min_frac * POINTS_ASSIST

        return round(xG_pts + xA_pts, 2)


class TrueGradientBoostedTree:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        self.is_trained = False

    def train(self, all_history):
        X = []
        y = []
        for pid, history in all_history.items():
            bps_window = []
            for h in history:
                bps = float(h.get("bps", 0) or 0)
                bps_window.append(bps)
                if len(bps_window) > 4:
                    bps_window.pop(0)
                avg_bps = sum(bps_window) / len(bps_window)
                
                # FIX: Only train on matches where they started (minutes > 60) so the model predicts "Points if starting". 
                # This prevents double-shrinking when we multiply by xMin/90 later.
                if h.get("minutes", 0) > 60:
                    # FIX: Dropped FDR. BPS is now a 4-match rolling average.
                    # We will just train on transfers_balance, value, and avg_bps.
                    value = h.get("value", 50)
                    transfers_bal = h.get("transfers_balance", 0)
                    X.append([value, transfers_bal, avg_bps])
                    y.append(h.get("total_points", 0))
        
        if len(X) > 50:
            self.model.fit(X, y)
            self.is_trained = True

    def predict_match(self, player, fixture, history, xMin):
        if not self.is_trained:
            return 0.0
            
        value = float(player.get("now_cost", 50) or 50)
        transfers_bal = float(player.get("transfers_in_event", 0) or 0) - float(player.get("transfers_out_event", 0) or 0)
        
        recent_bps = [float(h.get("bps", 0) or 0) for h in history[-4:]] if history else [0.0]
        avg_bps = sum(recent_bps) / len(recent_bps) if recent_bps else 0.0
        
        # Predict points assuming 90 minutes played
        pred = self.model.predict([[value, transfers_bal, avg_bps]])[0]
        
        # Scale by actual expected minutes
        return round(max(0.0, pred * (xMin/90.0)), 2)

class EnsembleForecaster:
    def __init__(self, weights: tuple = None):
        if weights is None:
            # Optimal weights discovered via Grid Search (MAE: 2.33)
            weights = (0.40, 0.60, 0.00)
        self.w_dc, self.w_kf, self.w_gt = weights
        self.dc = DixonColesModel()
        self.kf = KalmanFormFilter()
        self.gt = TrueGradientBoostedTree()

    def fit(self, teams, past_fixtures, current_gw, all_history):
        self.dc.fit_team_ratings(teams, past_fixtures, current_gw)
        self.gt.train(all_history)
        
    def predict(self, player, fixture, history, xMin):
        if xMin <= 0: return 0.0
        xApp = 2.0 if xMin >= 60 else (1.0 if xMin > 0 else 0.0)
        
        xp_dc = self.dc.predict_match(player, fixture, xMin)
        xp_kf = self.kf.predict_match(player, history, xMin)
        xp_gt = self.gt.predict_match(player, fixture, history, xMin)
        
        ensemble_xp = xApp + (self.w_dc * xp_dc) + (self.w_kf * xp_kf) + (self.w_gt * xp_gt)
        return round(ensemble_xp, 2)

def calculate_expected_minutes(player: Dict[str, Any], current_gw: int = 1, history: List[Dict] = None) -> float:
    status = player.get("status", "a")
    chance = player.get("chance_of_playing_next_round")
    if status in ["i", "s", "u"] or chance == 0:
        return 0.0
        
    if chance is not None:
        avail_mult = float(chance) / 100.0
    else:
        avail_mult = 1.0

    form = float(player.get("form", 0.0) or 0.0)
    minutes = float(player.get("minutes", 0) or 0)
    cost = float(player.get("now_cost", 0) or 0)
    
    avg_mins = minutes / max(1, current_gw)
    
    if avg_mins >= 60 or form > 3.0 or cost >= 75:
        base_mins = 82.0
    elif avg_mins >= 30:
        base_mins = 60.0
    elif minutes > 0:
        base_mins = 35.0
    else:
        base_mins = 15.0
        
    # #5. Continuous xMin Model: Calculate EMA of recent minutes
    if history and len(history) > 0:
        alpha = 0.3
        ema = float(history[0].get("minutes", 0))
        for h in history[1:]:
            ema = (alpha * float(h.get("minutes", 0))) + ((1 - alpha) * ema)
            
        # Blend 70% EMA with 30% bucket baseline. 
        # This protects premium players returning from injury who would otherwise have an EMA of 0.
        base_mins = (ema * 0.7) + (base_mins * 0.3)
        
    return round(base_mins * avail_mult, 1)

def generate_xp_matrix(horizon_gws: List[int], bootstrap=None, fixtures=None, all_history=None, weights: tuple = None) -> Dict[int, Dict[int, float]]:
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
    ensemble.fit(teams, past_fixtures, current_gw, all_history)
    
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
        xMin = calculate_expected_minutes(p, current_gw, history)

        for gw in horizon_gws:
            gw_fixes = fixture_map.get(gw, {}).get(p["team"], [])
            if not gw_fixes or xMin <= 0:
                xp_matrix[pid][gw] = 0.0
            else:
                total_xp = 0.0
                for idx, f in enumerate(gw_fixes):
                    # #8. DGW Rotation Adjustment: reduce xMin by 22 mins for the second fixture
                    adjusted_xmin = xMin if idx == 0 else max(xMin - 22.0, 0.0) 
                    total_xp += ensemble.predict(p, f, history, adjusted_xmin)
                xp_matrix[pid][gw] = round(total_xp, 2)
    return xp_matrix

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mat = generate_xp_matrix([2])
    print(list(mat.items())[:5])
