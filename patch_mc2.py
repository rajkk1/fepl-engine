with open("c:/Users/rajkk/FEPL/monte_carlo.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(
    'p_cs = comps.get("p_cs", 0.0)',
    'p_cs = comps.get("p_cs", 0.0)\n    element_type = comps.get("element_type", 3)'
)

with open("c:/Users/rajkk/FEPL/monte_carlo.py", "w", encoding="utf-8") as f:
    f.write(content)
