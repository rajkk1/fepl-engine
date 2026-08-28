import math

with open("c:/Users/rajkk/FEPL/xp_model.py", "r", encoding="utf-8") as f:
    content = f.read()

new_logic = """        xG_pts = xg * POINTS_GOAL.get(element_type, 4)
        xA_pts = xa * POINTS_ASSIST
        
        # CS points: use player_opp_xg for p_cs to fix CS/lambda inconsistency
        player_opp_xg = opp_xg * min_frac
        p_cs_player = math.exp(-player_opp_xg)
        
        cs_pts = POINTS_CLEAN_SHEET.get(element_type, 0)
        xCS_pts = p_60 * p_cs_player * cs_pts
        
        # Calculate expected goals conceded penalty
        e_floor_x2 = sum(math.exp(-player_opp_xg) * (player_opp_xg**k) / math.factorial(k) * (k // 2) for k in range(10))
        xConc_penalty = e_floor_x2 if element_type in [POS_GKP, POS_DEF] else 0.0
        
        # Card Penalty
        xCard_penalty = p_play * 0.3
        
        # Calculate expected saves points using SoT
        sot90 = gpf_res.get("sot90", 0.0)
        expected_saves = max(0.0, (sot90 * min_frac) - player_opp_xg)
        xSaves = (expected_saves / 3.0) if element_type == POS_GKP else 0.0
        
        # DefCon (Defensive Contributions) added in 25/26
        xDefCon = 0.0
        from scipy.stats import nbinom
        if element_type == POS_DEF:
            # P(CBIT >= 10) * 2
            mu = cbit
            if mu > 0:
                v = 1.85 * mu
                p = mu / v
                n = (mu**2) / (v - mu)
                xDefCon = max(0.0, 1.0 - nbinom.cdf(9, n, p)) * 2.0
        elif element_type in [POS_MID, POS_FWD]:
            # P(CBIRT >= 12) * 2
            mu = cbirt
            if mu > 0:
                v = 1.85 * mu
                p = mu / v
                n = (mu**2) / (v - mu)
                xDefCon = max(0.0, 1.0 - nbinom.cdf(11, n, p)) * 2.0
            
        # Calculate expected BPS (BPS-based bonus model)
        bps_mins = (p_1_59 * 3.0) + (p_60 * 6.0)
        bps_goals = xg * (24.0 if element_type == POS_FWD else (18.0 if element_type == POS_MID else 12.0))
        bps_assists = xa * 9.0
        bps_cs = (p_cs_player * p_60) * (12.0 if element_type in [POS_GKP, POS_DEF] else 0.0)
        bps_saves = expected_saves * 2.0 if element_type == POS_GKP else 0.0
        bps_defcon = cbirt * 0.33
        
        e_bps = bps_mins + bps_goals + bps_assists + bps_cs + bps_saves + bps_defcon
        
        # Map E[BPS] to expected bonus points using an empirical percentile map
        xBonus = max(0.0, (e_bps - 8.0) * 0.15)
        math_pts = xApp + xCS_pts + xG_pts + xA_pts + xSaves + xDefCon + xBonus - xConc_penalty - xCard_penalty"""

start_str = "        xG_pts = xg * POINTS_GOAL.get(element_type, 4)"
end_str = "math_pts = xApp + xCS_pts + xG_pts + xA_pts + xSaves + xDefCon + xBonus - xConc_penalty"

start_idx = content.find(start_str)
end_idx = content.find(end_str) + len(end_str)

content = content[:start_idx] + new_logic + content[end_idx:]

content = content.replace('"p_cs": p_cs,', '"p_cs": p_cs_player,')
content = content.replace('POINTS_GOAL = {POS_GKP: 6', 'POINTS_GOAL = {POS_GKP: 10')

gpf_logic = """class GammaPoissonFilter:
    def __init__(self, half_life=5.0, prior_weight=1.0):
        self.half_life = half_life
        self.prior_weight = prior_weight
        
        self.pos_priors = {
            POS_GKP: {"xg": 0.00, "xa": 0.01, "cbit": 0.5, "cbirt": 0.5, "sot": 4.5},
            POS_DEF: {"xg": 0.05, "xa": 0.08, "cbit": 7.45, "cbirt": 7.45, "sot": 0.0},
            POS_MID: {"xg": 0.15, "xa": 0.15, "cbit": 1.0, "cbirt": 7.86, "sot": 0.0},
            POS_FWD: {"xg": 0.40, "xa": 0.15, "cbit": 0.5, "cbirt": 4.09, "sot": 0.0}
        }

    def predict_match(self, player, history, market_predictor, current_gw):
        pos = player.get("element_type", POS_MID)
        prior = self.pos_priors.get(pos, self.pos_priors[POS_MID])
        
        a0_xg = prior["xg"] * self.prior_weight
        a0_xa = prior["xa"] * self.prior_weight
        a0_cbit = prior["cbit"] * self.prior_weight
        a0_cbirt = prior["cbirt"] * self.prior_weight
        a0_sot = prior["sot"] * self.prior_weight
        b0 = self.prior_weight
        
        sum_xg, sum_xa, sum_cbit, sum_cbirt, sum_sot, sum_w = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        player_team = player.get("team")
        
        history = sorted(history, key=lambda x: x.get("round", 0))
        
        for h in history:
            mins = h.get("minutes", 0)
            if mins > 0:
                gw = h.get("round", current_gw - 1)
                age = max(1, current_gw - gw)
                w_i = 0.5 ** (age / self.half_life)
                
                was_home = h.get("was_home")
                opp_id = h.get("opponent_team")
                home_id = player_team if was_home else opp_id
                away_id = opp_id if was_home else player_team
                
                lam, mu = market_predictor.market.get_match_lambdas(home_id, away_id)
                team_xg = lam if was_home else mu
                
                team_att_baseline = market_predictor.market.team_ratings.get(player_team, {}).get("scored", 1.4)
                team_baseline = team_att_baseline * 1.0 if team_att_baseline > 0 else 1.4
                fixture_multiplier = team_xg / team_baseline if team_baseline > 0 else 1.0
                fixture_multiplier = max(0.6, min(1.6, fixture_multiplier))
                
                obs_xg = float(h.get("expected_goals", 0) or 0) / fixture_multiplier
                obs_xa = float(h.get("expected_assists", 0) or 0) / fixture_multiplier
                
                cbi = int(h.get("clearances_blocks_interceptions", 0) or 0)
                tackles = int(h.get("tackles", 0) or 0)
                recoveries = int(h.get("recoveries", 0) or 0)
                cbit = cbi + tackles
                cbirt = cbit + recoveries
                
                saves = int(h.get("saves", 0) or 0)
                gc = int(h.get("goals_conceded", 0) or 0)
                obs_sot = (saves + gc) / fixture_multiplier
                
                sum_xg += w_i * obs_xg
                sum_xa += w_i * obs_xa
                sum_cbit += w_i * cbit
                sum_cbirt += w_i * cbirt
                sum_sot += w_i * obs_sot
                sum_w += w_i * (mins / 90.0)
                
        xg90 = (a0_xg + sum_xg) / (b0 + sum_w)
        xa90 = (a0_xa + sum_xa) / (b0 + sum_w)
        cbit90 = (a0_cbit + sum_cbit) / (b0 + sum_w)
        cbirt90 = (a0_cbirt + sum_cbirt) / (b0 + sum_w)
        sot90 = (a0_sot + sum_sot) / (b0 + sum_w)
        
        return {
            "xg": xg90,
            "xa": xa90,
            "cbit90": cbit90,
            "cbirt90": cbirt90,
            "sot90": sot90
        }
"""
start_gpf = "class GammaPoissonFilter:"
end_gpf = "class HistGradientBoostingClassifier:"
end_gpf_idx = content.find(end_gpf)
start_gpf_idx = content.find(start_gpf)
if end_gpf_idx > -1 and start_gpf_idx > -1:
    content = content[:start_gpf_idx] + gpf_logic + "\n\nfrom sklearn.ensemble import HistGradientBoostingClassifier\n" + content[end_gpf_idx + len(end_gpf):]


with open("c:/Users/rajkk/FEPL/xp_model.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Patched xp_model.py perfectly")
