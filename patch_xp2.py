import math
import re

with open("c:/Users/rajkk/FEPL/xp_model.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix MarketOddsPredictor (it got corrupted last run)
content = content.replace('"p_cs": p_cs_player,', '"p_cs": p_cs,')

# Fix components dict manually using regex to replace "p_cs": p_cs, -> "p_cs": p_cs_player, ONLY inside _predict_uncalibrated
# Actually, let's just do it directly.
new_components = """
        components = {
            "p_play": p_play,
            "p_60": p_60,
            "xApp": xApp,
            "xg": xg,
            "xa": xa,
            "p_cs": p_cs_player,
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
            "xCard_penalty": xCard_penalty,
            "math_pts": math_pts
        }
"""
start_comp = "        components = {"
end_comp = "        return components"
start_comp_idx = content.find(start_comp)
end_comp_idx = content.find(end_comp)
if start_comp_idx > -1 and end_comp_idx > -1:
    content = content[:start_comp_idx] + new_components.strip('\n') + "\n" + content[end_comp_idx:]

with open("c:/Users/rajkk/FEPL/xp_model.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed xp_model.py correctly")
