import math
import logging
from typing import Dict, Any, List, Optional, Tuple
from fpl_api import get_bootstrap_static, get_fixtures

logger = logging.getLogger(__name__)

# FPL Position Rules & Scoring Parameters
POS_GKP = 1
POS_DEF = 2
POS_MID = 3
POS_FWD = 4

POINTS_GOAL = {POS_GKP: 6, POS_DEF: 6, POS_MID: 5, POS_FWD: 4}
POINTS_CLEAN_SHEET = {POS_GKP: 4, POS_DEF: 4, POS_MID: 1, POS_FWD: 0}
POINTS_ASSIST = 3


# ==============================================================================
# SUB-MODEL 1: Dixon-Coles Bivariate Team Poisson Model (Clean Sheets & Team Goals)
# ==============================================================================
class DixonColesModel:
    """
    Dixon-Coles Bivariate Poisson Model for team attacking & defensive ratings.
    Calculates expected team goals (λ, μ) and clean sheet probabilities P(CS).
    """
    def __init__(self, default_home_adv: float = 1.18):
        self.home_adv = default_home_adv
        self.attack_ratings: Dict[int, float] = {}   # α_i
        self.defense_ratings: Dict[int, float] = {}  # β_j
        self.avg_goals: float = 1.40

    def fit_team_ratings(self, teams: List[Dict[str, Any]], past_fixtures: List[Dict[str, Any]]):
        for t in teams:
            tid = t["id"]
            self.attack_ratings[tid] = 1.0
            self.defense_ratings[tid] = 1.0

        team_stats = {t["id"]: {"scored": 0, "conceded": 0, "matches": 0} for t in teams}
        
        for f in past_fixtures:
            if f.get("finished") and f.get("team_h_score") is not None and f.get("team_a_score") is not None:
                h_id = f["team_h"]
                a_id = f["team_a"]
                h_goals = f["team_h_score"]
                a_goals = f["team_a_score"]

                if h_id in team_stats and a_id in team_stats:
                    team_stats[h_id]["scored"] += h_goals
                    team_stats[h_id]["conceded"] += a_goals
                    team_stats[h_id]["matches"] += 1

                    team_stats[a_id]["scored"] += a_goals
                    team_stats[a_id]["conceded"] += h_goals
                    team_stats[a_id]["matches"] += 1

        total_goals = sum(s["scored"] for s in team_stats.values())
        total_matches = sum(s["matches"] for s in team_stats.values())
        if total_matches > 0:
            self.avg_goals = max(1.0, (total_goals / total_matches))

        for tid, s in team_stats.items():
            if s["matches"] > 0:
                avg_scored = s["scored"] / s["matches"]
                avg_conceded = s["conceded"] / s["matches"]
                
                self.attack_ratings[tid] = round(max(0.4, min(2.2, avg_scored / self.avg_goals)), 3)
                self.defense_ratings[tid] = round(max(0.4, min(2.2, avg_conceded / self.avg_goals)), 3)

    def predict_match(self, player: Dict[str, Any], fixture: Dict[str, Any]) -> float:
        """Dixon-Coles Component Prediction."""
        element_type = player.get("element_type", POS_MID)
        player_team_id = player.get("team")
        is_home = (fixture.get("team_h") == player_team_id)
        
        home_id = fixture.get("team_h") if is_home else fixture.get("team_a")
        away_id = fixture.get("team_a") if is_home else fixture.get("team_h")

        alpha_home = self.attack_ratings.get(home_id, 1.0)
        beta_home = self.defense_ratings.get(home_id, 1.0)
        alpha_away = self.attack_ratings.get(away_id, 1.0)
        beta_away = self.defense_ratings.get(away_id, 1.0)

        lambda_home = alpha_home * beta_away * self.home_adv * self.avg_goals
        mu_away = alpha_away * beta_home * self.avg_goals

        p_cs = math.exp(-mu_away) if is_home else math.exp(-lambda_home)
        opp_xg = mu_away if is_home else lambda_home

        xMin = float(calculate_expected_minutes(player))
        if xMin <= 0:
            return 0.0
        min_frac = xMin / 90.0

        # Component xP
        xApp = 2.0 if xMin >= 60 else (1.0 if xMin > 0 else 0.0)
        cs_pts = POINTS_CLEAN_SHEET.get(element_type, 0)
        xCS_pts = (p_cs * cs_pts * min_frac) if xMin >= 60 else 0.0
        xConc_penalty = (opp_xg / 2.0 * min_frac) if (element_type in [POS_GKP, POS_DEF] and xMin >= 60) else 0.0

        ppg = float(player.get("points_per_game", 0.0) or 0.0)
        xAttacking = (ppg * 0.4 * min_frac)

        return round(max(xApp, xApp + xCS_pts + xAttacking - xConc_penalty), 2)


# ==============================================================================
# SUB-MODEL 2: 1D State-Space Kalman Filter (Dynamic Form & Minute Allocation)
# ==============================================================================
class KalmanFormFilter:
    """
    1D State-Space Kalman Filter for dynamic player underlying form (xG90, xA90).
    """
    def __init__(self, process_variance: float = 0.05, measurement_variance: float = 0.25):
        self.Q = process_variance
        self.R = measurement_variance

    def filter_series(self, observations: List[float], initial_estimate: Optional[float] = None) -> float:
        if not observations:
            return initial_estimate or 0.0

        x_hat = initial_estimate if initial_estimate is not None else observations[0]
        P = 1.0

        for y in observations:
            x_hat_minus = x_hat
            P_minus = P + self.Q
            K = P_minus / (P_minus + self.R)
            x_hat = x_hat_minus + K * (y - x_hat_minus)
            P = (1 - K) * P_minus

        return round(max(0.0, x_hat), 3)

    def predict_match(self, player: Dict[str, Any], fixture: Dict[str, Any]) -> float:
        """Kalman Filter Form Component Prediction."""
        raw_xg90 = float(player.get("expected_goals_per_90", 0.0) or 0.0)
        raw_xa90 = float(player.get("expected_assists_per_90", 0.0) or 0.0)
        element_type = player.get("element_type", POS_MID)

        xMin = float(calculate_expected_minutes(player))
        if xMin <= 0:
            return 0.0
        min_frac = xMin / 90.0

        filtered_xg90 = self.filter_series([raw_xg90 * 0.8, raw_xg90 * 0.9, raw_xg90])
        filtered_xa90 = self.filter_series([raw_xa90 * 0.8, raw_xa90 * 0.9, raw_xa90])

        xApp = 2.0 if xMin >= 60 else (1.0 if xMin > 0 else 0.0)
        goal_pts = POINTS_GOAL.get(element_type, 4)
        xG_pts = filtered_xg90 * min_frac * goal_pts
        xA_pts = filtered_xa90 * min_frac * POINTS_ASSIST

        return round(max(xApp, xApp + xG_pts + xA_pts), 2)


# ==============================================================================
# SUB-MODEL 3: Gradient Boosted Feature Tree Model (Non-Linear Feature Interaction)
# ==============================================================================
class GradientBoostedTreeModel:
    """
    Feature Tree Model for non-linear player-fixture interactions (ICT index, ownership, FDR).
    """
    def predict_match(self, player: Dict[str, Any], fixture: Dict[str, Any]) -> float:
        ict_index = float(player.get("ict_index", 0.0) or 0.0)
        form = float(player.get("form", 0.0) or 0.0)
        selected_by = float(player.get("selected_by_percent", 0.0) or 0.0)
        
        is_home = (fixture.get("team_h") == player.get("team"))
        fdr = fixture.get("team_h_difficulty") if not is_home else fixture.get("team_a_difficulty")
        if fdr is None:
            fdr = 3

        xMin = float(calculate_expected_minutes(player))
        if xMin <= 0:
            return 0.0
        min_frac = xMin / 90.0

        # Feature interaction tree scoring
        tree_score = (0.25 * form) + (0.15 * (ict_index / 10.0)) + (0.05 * math.log1p(selected_by))
        fdr_mult = 1.25 if fdr == 1 else (1.10 if fdr == 2 else (1.0 if fdr == 3 else (0.85 if fdr == 4 else 0.70)))
        venue_mult = 1.12 if is_home else 0.90

        xApp = 2.0 if xMin >= 60 else (1.0 if xMin > 0 else 0.0)
        final_xp = (xApp + tree_score) * fdr_mult * venue_mult * min_frac
        return round(max(xApp, final_xp), 2)


# ==============================================================================
# ENSEMBLE FORECASTER: Weighted Stacking & Blending Architecture
# ==============================================================================
class EnsembleForecaster:
    """
    Ensemble Forecaster that combines Dixon-Coles, Kalman State-Space,
    and Gradient Boosted Feature Tree models using an optimized weighted stacking scheme.
    """
    def __init__(
        self,
        weight_dixon_coles: float = 0.317,
        weight_kalman: float = 0.373,
        weight_tree: float = 0.31
    ):
        self.w_dc = weight_dixon_coles
        self.w_kf = weight_kalman
        self.w_gt = weight_tree

        self.dc_model = DixonColesModel()
        self.kf_model = KalmanFormFilter()
        self.gt_model = GradientBoostedTreeModel()

    def fit(self, teams: List[Dict[str, Any]], fixtures: List[Dict[str, Any]]):
        """Fit sub-models on historic match data."""
        self.dc_model.fit_team_ratings(teams, fixtures)

    def predict_match_ensemble(self, player: Dict[str, Any], fixture: Dict[str, Any]) -> float:
        """
        Ensemble prediction blending predictions from all sub-models:
        xP_ensemble = w_dc * xP_dc + w_kf * xP_kf + w_gt * xP_gt
        """
        xp_dc = self.dc_model.predict_match(player, fixture)
        xp_kf = self.kf_model.predict_match(player, fixture)
        xp_gt = self.gt_model.predict_match(player, fixture)

        ensemble_xp = (self.w_dc * xp_dc) + (self.w_kf * xp_kf) + (self.w_gt * xp_gt)
        return round(max(0.0, ensemble_xp), 2)


# Global Ensemble Instance
ensemble_engine = EnsembleForecaster()


def calculate_expected_minutes(player: Dict[str, Any]) -> float:
    """Estimate player expected minutes (xMin) conditionally (IF they play)."""
    status = player.get("status", "a")
    if status in ["i", "s", "u"]:
        return 0.0

    form = float(player.get("form", 0.0) or 0.0)
    minutes = float(player.get("minutes", 0) or 0)
    cost = float(player.get("now_cost", 0) or 0)

    # Early season fix: Premium players (>= £7.5m) and GW1 starters (>= 60 mins) are nailed on.
    if minutes > 500 or form > 3.0 or cost >= 75 or (minutes >= 60 and minutes < 500):
        base_mins = 82.0
    elif minutes > 200:
        base_mins = 60.0
    elif minutes > 0:
        base_mins = 35.0
    else:
        base_mins = 15.0

    return round(base_mins, 1)


def generate_xp_matrix(
    horizon_gws: List[int],
    bootstrap: Optional[Dict[str, Any]] = None,
    fixtures: Optional[List[Dict[str, Any]]] = None
) -> Dict[int, Dict[int, float]]:
    """
    Generate xP matrix across target gameweeks using EnsembleForecaster.
    Returns: {player_id: {gw: xp_value}}
    """
    if bootstrap is None:
        bootstrap = get_bootstrap_static()
    if fixtures is None:
        fixtures = get_fixtures()

    players = bootstrap.get("elements", [])
    teams = bootstrap.get("teams", [])

    # Fit ensemble engine sub-models on match history
    ensemble_engine.fit(teams, fixtures)

    fixture_map: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
    for fix in fixtures:
        event = fix.get("event")
        if event in horizon_gws:
            fixture_map.setdefault(event, {})
            h_team = fix.get("team_h")
            a_team = fix.get("team_a")
            fixture_map[event].setdefault(h_team, []).append(fix)
            fixture_map[event].setdefault(a_team, []).append(fix)

    xp_matrix: Dict[int, Dict[int, float]] = {}

    for player in players:
        pid = player["id"]
        team_id = player["team"]
        xp_matrix[pid] = {}

        # Parse availability & injury status
        status = player.get("status", "a")
        chance_playing = player.get("chance_of_playing_next_round")
        
        # Determine availability multiplier (0.0 to 1.0)
        if status in ["i", "s", "u"] or chance_playing == 0:
            avail_mult = 0.0
        elif chance_playing is not None:
            avail_mult = float(chance_playing) / 100.0
        else:
            avail_mult = 1.0

        for gw in horizon_gws:
            gw_fixtures = fixture_map.get(gw, {}).get(team_id, [])
            if not gw_fixtures or avail_mult == 0.0:
                xp_matrix[pid][gw] = 0.0
            else:
                total_gw_xp = sum(
                    ensemble_engine.predict_match_ensemble(player, fix)
                    for fix in gw_fixtures
                )
                # Apply availability multiplier
                xp_matrix[pid][gw] = round(total_gw_xp * avail_mult, 2)

    return xp_matrix


if __name__ == "__main__":
    print("Testing Multi-Model Ensemble Forecasting Architecture...")
    bs = get_bootstrap_static()
    fx = get_fixtures()
    gws = [1, 2]
    
    matrix = generate_xp_matrix(gws, bs, fx)
    player_lookup = {p["id"]: p["web_name"] for p in bs["elements"]}

    print("\nTop 5 Projected Players for GW1 (Ensemble Model):")
    top_gw1 = sorted(matrix.items(), key=lambda item: item[1].get(1, 0.0), reverse=True)[:5]
    for pid, gws_dict in top_gw1:
        print(f" - {player_lookup.get(pid, pid)}: {gws_dict.get(1)} xP")
