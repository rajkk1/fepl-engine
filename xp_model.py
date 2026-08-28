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


class GammaPoissonFilter:
    def __init__(self, half_life=5.0, prior_weight=5.0):
        self.half_life = half_life
        self.prior_weight = prior_weight
        
        self.pos_priors = {
            POS_GKP: {"xg": 0.00, "xa": 0.01, "cbit": 0.5, "cbirt": 0.5},
            POS_DEF: {"xg": 0.05, "xa": 0.08, "cbit": 3.0, "cbirt": 3.0},
            POS_MID: {"xg": 0.15, "xa": 0.15, "cbit": 1.0, "cbirt": 2.0},
            POS_FWD: {"xg": 0.40, "xa": 0.15, "cbit": 0.5, "cbirt": 1.0}
        }

    def predict_match(self, player, history, market_predictor, current_gw):
        pos = player.get("element_type", POS_MID)
        prior = self.pos_priors.get(pos, self.pos_priors[POS_MID])
        
        a0_xg = prior["xg"] * self.prior_weight
        a0_xa = prior["xa"] * self.prior_weight
        a0_cbit = prior["cbit"] * self.prior_weight
        a0_cbirt = prior["cbirt"] * self.prior_weight
        b0 = self.prior_weight
        
        sum_xg, sum_xa, sum_cbit, sum_cbirt, sum_w = 0.0, 0.0, 0.0, 0.0, 0.0
        
        player_team = player.get("team")
        
        # Sort history to be safe
        history = sorted(history, key=lambda x: x.get("round", 0))
        
        for h in history:
            mins = h.get("minutes", 0)
            if mins > 0:
                gw = h.get("round", current_gw - 1)
                age = max(1, current_gw - gw)
                w_i = 0.5 ** (age / self.half_life)
                
                # Condition on fixture multiplier to extract raw form
                was_home = h.get("was_home")
                opp_id = h.get("opponent_team")
                home_id = player_team if was_home else opp_id
                away_id = opp_id if was_home else player_team
                
                lam, mu = market_predictor.market.get_match_lambdas(home_id, away_id)
                team_xg = lam if was_home else mu
                
                team_att_baseline = market_predictor.market.team_ratings.get(player_team, {}).get("scored", 1.4)
                team_baseline = team_att_baseline * 1.0 if team_att_baseline > 0 else 1.4
                fixture_multiplier = team_xg / team_baseline if team_baseline > 0 else 1.0
                
                # Raw accumulated metrics in the match
                obs_xg = float(h.get("expected_goals", 0) or 0) / fixture_multiplier
                obs_xa = float(h.get("expected_assists", 0) or 0) / fixture_multiplier
                
                # DefCon metrics
                cbi = int(h.get("clearances_blocks_interceptions", 0) or 0)
                tackles = int(h.get("tackles", 0) or 0)
                recoveries = int(h.get("recoveries", 0) or 0)
                cbit = cbi + tackles
                cbirt = cbit + recoveries
                
                sum_xg += w_i * obs_xg
                sum_xa += w_i * obs_xa
                sum_cbit += w_i * cbit
                sum_cbirt += w_i * cbirt
                sum_w += w_i * (mins / 90.0)
                
        xg90 = (a0_xg + sum_xg) / (b0 + sum_w)
        xa90 = (a0_xa + sum_xa) / (b0 + sum_w)
        cbit90 = (a0_cbit + sum_cbit) / (b0 + sum_w)
        cbirt90 = (a0_cbirt + sum_cbirt) / (b0 + sum_w)
        
        return {"xg": xg90, "xa": xa90, "cbit90": cbit90, "cbirt90": cbirt90}




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

    def predict_proba(self, player, history, fixture=None, current_gw=1):
        if not self.is_trained:
            return [0.1, 0.1, 0.1, 0.7] # Default fallback
            
        status = player.get("status", "a")
        chance_raw = player.get("chance_of_playing_next_round")
        
        if status in ["i", "s", "u", "n"] or chance_raw == 0:
            chance = 0.0
        else:
            chance = float(chance_raw) if chance_raw is not None else 100.0
            
        # Parse return date from news
        news = player.get("news", "")
        import re
        import datetime
        match = re.search(r"Expected back (\d{1,2} [a-zA-Z]{3})", news)
        if match and fixture and fixture.get("kickoff_time"):
            try:
                date_str = match.group(1)
                # Assume current year
                curr_year = datetime.datetime.now().year
                ret_date = datetime.datetime.strptime(f"{date_str} {curr_year}", "%d %b %Y")
                kickoff = pd.to_datetime(fixture["kickoff_time"]).tz_localize(None)
                
                # If return date is month 1-5 and kickoff is month 8-12, return date is next year
                if ret_date.month < 6 and kickoff.month > 7:
                    ret_date = ret_date.replace(year=curr_year + 1)
                    
                if ret_date.date() <= kickoff.date():
                    chance = 100.0
                else:
                    chance = 0.0
            except Exception:
                pass
        elif chance < 100.0 and fixture:
            # Linear decay towards 100% over the horizon if no exact date given
            gw = fixture.get("event", current_gw)
            offset = max(0, gw - current_gw)
            # Ramp by 15% per gameweek
            chance = min(100.0, chance + (15.0 * offset))
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
        self.dc = MarketOddsPredictor()
        self.gpf = GammaPoissonFilter(half_life=5.0, prior_weight=5.0)
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
            
    def _predict_uncalibrated(self, player, fixture, history, dgw_idx=0, current_gw=1):
        p_states = self.mc.predict_proba(player, history, fixture, current_gw)
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
        
        if xMin <= 0:
            return {
                "p_play": 0.0, "p_60": 0.0, "xApp": 0.0, "xg": 0.0, "xa": 0.0, "p_cs": 0.0,
                "cbit": 0.0, "cbirt": 0.0, "player_opp_xg": 0.0, "element_type": player.get("element_type", 3),
                "e_bps": 0.0, "xBonus": 0.0, "xCS_pts": 0.0, "xG_pts": 0.0, "xA_pts": 0.0,
                "xSaves": 0.0, "xDefCon": 0.0, "xConc_penalty": 0.0, "math_pts": 0.0
            }
        
        xApp = (p_60 * 2.0) + ((p_play - p_60) * 1.0)
        
        # 1. Base Rate Estimation (GammaPoisson)
        gw = fixture.get("event", 1)
        gpf_res = self.gpf.predict_match(player, history, self.dc, gw)
        
        # 2. Override FPL dictionary so downstream component uses shrunk rates
        player_copy = dict(player)
        player_copy["expected_goals_per_90"] = gpf_res["xg"]
        player_copy["expected_assists_per_90"] = gpf_res["xa"]
        
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
        
        cbit = gpf_res["cbit90"] * min_frac
        cbirt = gpf_res["cbirt90"] * min_frac
        
        element_type = player.get("element_type", POS_MID)
        
        xG_pts = xg * POINTS_GOAL.get(element_type, 4)
        xA_pts = xa * POINTS_ASSIST
        
        cs_pts = POINTS_CLEAN_SHEET.get(element_type, 0)
        xCS_pts = p_60 * p_cs * cs_pts
        
        # Calculate expected goals conceded penalty
        player_opp_xg = opp_xg * min_frac
        e_floor_x2 = sum(math.exp(-player_opp_xg) * (player_opp_xg**k) / math.factorial(k) * (k // 2) for k in range(10))
        xConc_penalty = e_floor_x2 if element_type in [POS_GKP, POS_DEF] else 0.0
        
        # Calculate expected saves points E[floor(saves/3)] assuming saves ~ Poisson(opp_xg * 2.5)
        expected_saves = player_opp_xg * 2.5
        e_floor_saves3 = sum(math.exp(-expected_saves) * (expected_saves**k) / math.factorial(k) * (k // 3) for k in range(15))
        xSaves = e_floor_saves3 if element_type == POS_GKP else 0.0
        
        # DefCon (Defensive Contributions) added in 25/26
        xDefCon = 0.0
        if element_type in [POS_GKP, POS_DEF]:
            # P(CBIT >= 10) * 2
            p_under_10 = sum(math.exp(-cbit) * (cbit**k) / math.factorial(k) for k in range(10))
            xDefCon = max(0.0, 1.0 - p_under_10) * 2.0
        elif element_type in [POS_MID, POS_FWD]:
            # P(CBIRT >= 12) * 2
            p_under_12 = sum(math.exp(-cbirt) * (cbirt**k) / math.factorial(k) for k in range(12))
            xDefCon = max(0.0, 1.0 - p_under_12) * 2.0
            
        # Calculate expected BPS (BPS-based bonus model)
        bps_mins = (p_1_59 * 3.0) + (p_60 * 6.0)
        bps_goals = xg * (24.0 if element_type == POS_FWD else (18.0 if element_type == POS_MID else 12.0))
        bps_assists = xa * 9.0
        bps_cs = (p_cs * p_60) * (12.0 if element_type in [POS_GKP, POS_DEF] else 0.0)
        bps_saves = (player_opp_xg * 2.5) * 2.0 if element_type == POS_GKP else 0.0
        bps_defcon = cbit * 0.5 + cbirt * 0.2
        
        e_bps = bps_mins + bps_goals + bps_assists + bps_cs + bps_saves + bps_defcon
        
        # Map E[BPS] to expected bonus points using a smooth baseline curve
        # Typically, a player needs ~25+ BPS to enter bonus contention.
        # Every BPS above 22 adds roughly 0.15 expected bonus points.
        xBonus = max(0.0, (e_bps - 22.0) * 0.15)
        math_pts = xApp + xCS_pts + xG_pts + xA_pts + xSaves + xDefCon + xBonus - xConc_penalty
        
        components = {
            "p_play": p_play,
            "p_60": p_60,
            "xApp": xApp,
            "xg": xg,
            "xa": xa,
            "p_cs": p_cs,
            "cbit": cbit,
            "cbirt": cbirt,
            "player_opp_xg": player_opp_xg,
            "element_type": element_type,
            "e_bps": e_bps,
            "xBonus": xBonus,
            "xCS_pts": xCS_pts,
            "xG_pts": xG_pts,
            "xA_pts": xA_pts,
            "xSaves": xSaves,
            "xDefCon": xDefCon,
            "xConc_penalty": xConc_penalty,
            "math_pts": math_pts
        }
        return components

    def predict(self, player, fixture, history, dgw_idx=0, current_gw=1):
        comps = self._predict_uncalibrated(player, fixture, history, dgw_idx, current_gw)
        return round(comps["math_pts"], 2)

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
    current_gw = horizon_gws[0] if horizon_gws else 1
    ensemble.fit(teams, past_fixtures, current_gw, all_history, players, season=season)
    
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
                    total_xp += ensemble.predict(p, f, history, dgw_idx=idx, current_gw=current_gw)
                xp_matrix[pid][gw] = round(total_xp, 2)
    return xp_matrix

def generate_merv_matrix(horizon_gws: List[int], bootstrap=None, fixtures=None, all_history=None, weights: tuple = None, season: int = None, risk_aversion: float = 0.0) -> Dict[int, Dict[int, float]]:
    """
    Generates a matrix of Marginal Expected Rank Value (MERV) instead of raw xP.
    Rank-Aware objective that penalizes variance for differentials and rewards variance reduction for highly owned players.
    """
    from ownership_model import build_eo_matrix
    from monte_carlo import simulate_player_variance, calculate_merv
    
    if bootstrap is None: bootstrap = get_bootstrap_static()
    if fixtures is None: fixtures = get_fixtures()
    
    players = bootstrap.get("elements", [])
    teams = bootstrap.get("teams", [])
    eo_matrix = build_eo_matrix(players)
    
    current_gw = horizon_gws[0] if horizon_gws else 1
    ensemble = EnsembleForecaster(weights=weights)
    past_fixtures = [f for f in fixtures if f.get("finished")]
    
    if all_history is None:
        player_ids = [p["id"] for p in players]
        summaries = get_all_element_summaries(player_ids)
        all_history = {pid: summaries.get(pid, {}).get("history", []) for pid in player_ids}
        
    ensemble.fit(teams, past_fixtures, current_gw, all_history, players, season=season)
    
    fixture_map = {}
    for fix in fixtures:
        if fix.get("event") in horizon_gws:
            fixture_map.setdefault(fix["event"], {})
            fixture_map[fix["event"]].setdefault(fix["team_h"], []).append(fix)
            fixture_map[fix["event"]].setdefault(fix["team_a"], []).append(fix)

    merv_matrix = {}
    for p in players:
        pid = p["id"]
        merv_matrix[pid] = {}
        history = all_history.get(pid, [])
        eo = eo_matrix.get(pid, 0.0)

        for gw in horizon_gws:
            gw_fixes = fixture_map.get(gw, {}).get(p["team"], [])
            if not gw_fixes:
                merv_matrix[pid][gw] = 0.0
            else:
                total_merv = 0.0
                for idx, f in enumerate(gw_fixes):
                    comps = ensemble._predict_uncalibrated(p, f, history, dgw_idx=idx, current_gw=current_gw)
                    xp = comps["math_pts"]
                    var = simulate_player_variance(comps)
                    total_merv += calculate_merv(xp, var, eo, risk_aversion)
                merv_matrix[pid][gw] = round(total_merv, 2)
    return merv_matrix

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mat = generate_xp_matrix([2])
    print(list(mat.items())[:5])
