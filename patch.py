import math

with open("c:/Users/rajkk/FEPL/xp_model.py", "r") as f:
    content = f.read()

new_logic = """
        xG_pts = xg * POINTS_GOAL.get(element_type, 4)
        xA_pts = xa * POINTS_ASSIST
        
        # CS points: use player_opp_xg for p_cs to fix CS/\\u03bb inconsistency
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
        math_pts = xApp + xCS_pts + xG_pts + xA_pts + xSaves + xDefCon + xBonus - xConc_penalty - xCard_penalty
"""

start_str = "xG_pts = xg * POINTS_GOAL.get(element_type, 4)"
end_str = "math_pts = xApp + xCS_pts + xG_pts + xA_pts + xSaves + xDefCon + xBonus - xConc_penalty"

start_idx = content.find(start_str)
end_idx = content.find(end_str) + len(end_str)

new_content = content[:start_idx] + new_logic.strip() + content[end_idx:]

with open("c:/Users/rajkk/FEPL/xp_model.py", "w") as f:
    f.write(new_content)

print("Patched xp_model.py")
