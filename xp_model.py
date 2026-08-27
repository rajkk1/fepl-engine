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
                match_data.append((f["team_h"], f["team_a"], f["team_h_score"], f["team_a_score"], gw_diff))
                
        if not match_data:
            for tid in team_ids:
                self.alpha[tid] = 0.0
                self.beta[tid] = 0.0
            return

        def neg_log_likelihood(params):
            alphas = {tid: params[i] for i, tid in enumerate(team_ids)}
            betas = {tid: params[num_teams + i] for i, tid in enumerate(team_ids)}
            gamma = params[2 * num_teams]
            rho = params[2 * num_teams + 1]
            decay = 0.005 # Fixed to avoid degeneracy
            
            nll = 0.0
            for h, a, hg, ag, gw_diff in match_data:
                w = math.exp(-decay * gw_diff * 7)
                lam = math.exp(alphas[h] - betas[a] + gamma)
                mu = math.exp(alphas[a] - betas[h])
                ll_h = hg * math.log(max(lam, 1e-5)) - lam
                ll_a = ag * math.log(max(mu, 1e-5)) - mu
                
                # Item #12: Dixon-Coles Rho Correction for low-scoring draws
                tau = 1.0
                if hg == 0 and ag == 0: tau = 1.0 - lam * mu * rho
                elif hg == 0 and ag == 1: tau = 1.0 + lam * rho
                elif hg == 1 and ag == 0: tau = 1.0 + mu * rho
                elif hg == 1 and ag == 1: tau = 1.0 - rho
                
                nll -= w * (ll_h + ll_a + math.log(max(tau, 1e-10)))
            
            # M-06: Sum-to-zero identification and soft prior shrinkage
            nll += 100 * (sum(alphas.values()))**2
            nll += 100 * (sum(betas.values()))**2
            nll += 10 * sum([a**2 for a in alphas.values()])
            nll += 10 * sum([b**2 for b in betas.values()])
            return nll

        init_params = [0.0] * (2 * num_teams) + [0.2, 0.0]
        bounds = [(-2.0, 2.0)] * (2 * num_teams) + [(0.0, 1.0), (-0.2, 0.2)]
        
        res = minimize(neg_log_likelihood, init_params, bounds=bounds, method='L-BFGS-B')
        
        for i, tid in enumerate(team_ids):
            self.alpha[tid] = res.x[i]
            self.beta[tid] = res.x[num_teams + i]
        self.gamma = res.x[2 * num_teams]
        self.rho = res.x[2 * num_teams + 1]

    def predict_match(self, player, fixture):
        element_type = player.get("element_type", POS_MID)
        is_home = (fixture.get("team_h") == player.get("team"))
        
        home_id = fixture.get("team_h") if is_home else fixture.get("team_a")
        away_id = fixture.get("team_a") if is_home else fixture.get("team_h")

        lam = math.exp(self.alpha.get(home_id, 0.0) - self.beta.get(away_id, 0.0) + self.gamma)
        mu = math.exp(self.alpha.get(away_id, 0.0) - self.beta.get(home_id, 0.0))
        
        team_xg = lam if is_home else mu
        opp_xg = mu if is_home else lam
        
        p_cs = math.exp(-opp_xg)

        player_xg90 = float(player.get("expected_goals_per_90", 0.0) or 0)
        player_xa90 = float(player.get("expected_assists_per_90", 0.0) or 0)
        
        # M-05: Divide by the player's own team's baseline xG so strong teams aren't double-counted
        team_baseline = math.exp(self.alpha.get(player.get("team"), 0.0)) * 1.4
        
        player_match_xg = player_xg90 * (team_xg / team_baseline)
        player_match_xa = player_xa90 * (team_xg / team_baseline)
        
        if player.get("penalties_order") == 1:
            player_match_xg += 0.11
            
        # M-08: Add Defensive Contribution Points (DefCon) based on player's actual API stats
        # FPL awards 1 point per 3 defensive actions
        defcon_90 = float(player.get("defensive_contribution_per_90", 0.0) or 0.0)
        xDefCon = defcon_90 / 3.0
            
        return {
            "xg": player_match_xg,
            "xa": player_match_xa,
            "p_cs": p_cs,
            "opp_xg": opp_xg,
            "xDefCon": xDefCon
        }

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


class TrueGradientBoostedTree:
    def __init__(self):
        # Increased depth and estimators because we now have 8 features instead of 3
        self.model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
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
                y.append(float(h.get("total_points", 0)))
                
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

class EnsembleForecaster:
    def __init__(self, weights: tuple = None):
        if weights is None:
            # Optimal leak-free weights discovered via Grid Search (Spearman: 0.689)
            weights = (0.00, 0.00, 1.00)
        self.w_dc, self.w_kf, self.w_gt = weights
        self.dc = DixonColesModel()
        self.kf = KalmanFormFilter()
        self.gt = TrueGradientBoostedTree()

    def fit(self, teams, past_fixtures, current_gw, all_history):
        self.dc.fit_team_ratings(teams, past_fixtures, current_gw)
        self.gt.train(all_history, teams)
        
    def predict(self, player, fixture, history, xMin):
        if xMin <= 0: return 0.0
        
        # M-04: Point estimate minutes replaced with a distribution
        # Probability of playing at least 60 minutes
        p_60 = max(0.0, min(1.0, (xMin - 30) / 45.0))
        p_play = min(1.0, xMin / 60.0)
        
        xApp = (p_60 * 2.0) + ((p_play - p_60) * 1.0)
        
        # Base components
        dc_res = self.dc.predict_match(player, fixture)
        kf_res = self.kf.predict_match(player, history)
        
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
        xConc_penalty = (opp_xg / 2.0 * min_frac) if element_type in [POS_GKP, POS_DEF] else 0.0
        
        xSaves = (opp_xg * 0.7 * min_frac) if element_type == POS_GKP else 0.0
        # M-07: Scale xBonus down to approximate the 6-point per-fixture limit
        xBonus = ((xg * 1.5) + (xa * 1.0) + (p_cs * 0.2 * p_60)) * 0.4
        
        math_pts = xApp + xCS_pts + xG_pts + xA_pts + xSaves + xBonus + xDefCon - xConc_penalty
        
        # GBT is a monolithic black box that predicts total points directly
        gt_pts = self.gt.predict_match(player, fixture, history) * min_frac
        
        ensemble_xp = (1.0 - self.w_gt) * math_pts + (self.w_gt * gt_pts)
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
