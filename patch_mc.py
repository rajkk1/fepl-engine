import math

with open("c:/Users/rajkk/FEPL/monte_carlo.py", "r") as f:
    content = f.read()

# Fix 1: Bound MERV EO term
new_logic1 = """
    # Bound the EO term so it doesn't cross 1.0 and flip the sign to reward variance
    eo_bounded = min(0.5, eo)
    delta_variance = (1.0 - 2.0 * eo_bounded) * variance
"""
content = content.replace(
    "    delta_variance = (1.0 - 2.0 * eo) * variance",
    new_logic1.strip('\n')
)

# Fix 2: Seed the random generator and fix double-discounting of minutes
new_logic2 = """
import numpy as np

def simulate_player_variance(comps: dict, n_sims=10000) -> float:
    \"\"\"
    Simulates a player's gameweek 10,000 times using numpy to find their variance.
    \"\"\"
    if comps.get("math_pts", 0) == 0:
        return 0.0
        
    np.random.seed(comps.get("id", 42) + int(comps.get("math_pts", 0)*100))
        
    p_play = comps.get("p_play", 0.0)
    p_60 = comps.get("p_60", 0.0)
    
    if p_play <= 0:
        return 0.0
        
    # Un-discount rates for the conditional simulation
    xg_cond = comps.get("xg", 0.0) / p_play
    xa_cond = comps.get("xa", 0.0) / p_play
    
    # 1. Simulate Appearance
    played = np.random.rand(n_sims) < p_play
    played_60 = played & (np.random.rand(n_sims) < (p_60 / (p_play + 1e-6)))
    
    # 2. Simulate Goals and Assists (Poisson)
    goals = np.random.poisson(lam=xg_cond, size=n_sims) * played
    assists = np.random.poisson(lam=xa_cond, size=n_sims) * played
"""

start_str = "def simulate_player_variance(comps: Dict[str, Any], n_sims=10000) -> float:\n"
end_str = "    # 3. Simulate Clean Sheets\n"

# I will just replace the function body
content = content.replace(
    """def simulate_player_variance(comps: Dict[str, Any], n_sims=10000) -> float:
    \"\"\"
    Simulates a player's gameweek 10,000 times using numpy to find their variance.
    \"\"\"
    if comps.get("math_pts", 0) == 0:
        return 0.0
        
    p_play = comps["p_play"]
    p_60 = comps["p_60"]
    xg = comps["xg"]
    xa = comps["xa"]
    p_cs = comps["p_cs"]
    element_type = comps["element_type"]
    
    # 1. Simulate Appearance
    played = np.random.rand(n_sims) < p_play
    played_60 = played & (np.random.rand(n_sims) < (p_60 / (p_play + 1e-6)))
    
    # 2. Simulate Goals and Assists (Poisson)
    goals = np.random.poisson(lam=xg, size=n_sims) * played
    assists = np.random.poisson(lam=xa, size=n_sims) * played""",
    new_logic2.strip('\n')
)

with open("c:/Users/rajkk/FEPL/monte_carlo.py", "w") as f:
    f.write(content)

print("Patched monte_carlo.py")
