import math

with open("c:/Users/rajkk/FEPL/backtest.py", "r") as f:
    content = f.read()

# Fix 1: Guard baseline xP
new_logic1 = """
        actuals = df_target.groupby('element')['total_points'].sum().to_dict()
        
        # guard the baseline
        valid = df_target.groupby('element')['xP'].sum()
        if (valid == 0).all():
            logger.warning(f"GW{target_gw}: xP column empty, skipping baseline")
            baseline_xp_map = None
        else:
            baseline_xp_map = df_target.groupby('element')['xP'].sum().to_dict()
"""
content = content.replace(
    "        actuals = df_target.groupby('element')['total_points'].sum().to_dict()\n        baseline_xp_map = df_target.groupby('element')['xP'].sum().to_dict()",
    new_logic1.strip('\n')
)

# Fix 2: Handle baseline_xp_map = None
new_logic2 = """
        for pid in valid_pids:
            points = actuals[pid]
            predicted_xp = xp_matrix.get(pid, {}).get(target_gw, 0.0)
            
            if baseline_xp_map is None:
                baseline_xp = 0.0
            else:
                baseline_xp = baseline_xp_map.get(pid, 2.0)
"""
content = content.replace(
    """        for pid in valid_pids:
            points = actuals[pid]
            baseline_xp = baseline_xp_map.get(pid, 2.0)
            predicted_xp = xp_matrix.get(pid, {}).get(target_gw, 0.0)""",
    new_logic2.strip('\n')
)

# Fix 3: Handle nans in baseline dev and spearman
new_logic3 = """
        import scipy.stats as stats
        gw_spearman = 0.0
        gw_b_spearman = 0.0
        pos_count_s = 0
        pos_count_bs = 0
        for pos in [1, 2, 3, 4]:
            if len(pos_actuals[pos]) > 2:
                s, _ = stats.spearmanr(pos_actuals[pos], pos_preds[pos])
                bs, _ = stats.spearmanr(pos_actuals[pos], pos_baseline[pos])
                if not math.isnan(s):
                    gw_spearman += s
                    pos_count_s += 1
                if not math.isnan(bs) and baseline_xp_map is not None:
                    gw_b_spearman += bs
                    pos_count_bs += 1
        
        if pos_count_s > 0:
            all_spearman.append(gw_spearman / pos_count_s)
        if pos_count_bs > 0:
            all_baseline_spearman.append(gw_b_spearman / pos_count_bs)
"""
start_str = "        import scipy.stats as stats\n        gw_spearman = 0.0"
end_str = "            all_baseline_spearman.append(gw_b_spearman / pos_count)"
start_idx = content.find(start_str)
end_idx = content.find(end_str) + len(end_str)
content = content[:start_idx] + new_logic3.strip('\n') + content[end_idx:]

with open("c:/Users/rajkk/FEPL/backtest.py", "w") as f:
    f.write(content)

print("Patched backtest.py")
