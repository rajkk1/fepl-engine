import os

with open("c:/Users/rajkk/FEPL/weekly_manager.py", "r", encoding="utf-8") as f:
    content = f.read()

new_logic = """
def get_manager_team_state(team_id: int, current_gw: int):
    \"\"\"Attempt to fetch manager's latest team state, bank balance, and selling prices.\"\"\"
    try:
        cookie = os.getenv("FPL_COOKIE")
        squad_ids = []
        bank = 0.0
        ft = 100 if current_gw == 1 else 1
        sell_prices = {}
        
        if cookie:
            try:
                my_team_data = fpl_api.get_my_team(team_id, cookie)
                squad_ids = [p["element"] for p in my_team_data.get("picks", [])]
                if "transfers" in my_team_data:
                    ft = max(0, my_team_data["transfers"].get("limit", 1) - my_team_data["transfers"].get("made", 0))
                    if "bank" in my_team_data["transfers"]:
                        bank = my_team_data["transfers"]["bank"] / 10.0
                for p in my_team_data.get("picks", []):
                    if "selling_price" in p:
                        sell_prices[p["element"]] = p["selling_price"] / 10.0
                return squad_ids, bank, ft, sell_prices
            except Exception as auth_err:
                pass
                
        # Public fallback
        picks_data = fpl_api.get_manager_picks(team_id, current_gw - 1 if current_gw > 1 else 1)
        squad_ids = [p["element"] for p in picks_data.get("picks", [])]
        bank = picks_data.get("entry_history", {}).get("bank", 0) / 10.0
        ft = 100 if current_gw == 1 else 1
        return squad_ids, bank, ft, sell_prices
    except Exception:
        # Pre-season or no team found
        return None, 100.0, 100, {}
"""

start_str = "def get_manager_team_state(team_id: int, current_gw: int):"
end_str = "        return None, 100.0, 100, {}"

start_idx = content.find(start_str)
end_idx = content.find(end_str) + len(end_str)
content = content[:start_idx] + new_logic.strip('\n') + content[end_idx:]

with open("c:/Users/rajkk/FEPL/weekly_manager.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Patched weekly_manager.py")
