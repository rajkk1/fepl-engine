# Fantasy Premier League (FPL) Team Optimizer & Transfer Planner

An interactive web application and mathematical Integer Linear Programming (ILP) optimizer for Fantasy Premier League.

## Features
- **Manager Squad Import**: Fetch squad, ITB, free transfers, and chips via FPL Team ID.
- **Multi-Gameweek ILP Solver**: Solves transfer strategy across 1 to 8 gameweeks using PuLP.
- **Interactive Pitch Dashboard**: Starting XI + Bench visualizer with projected points ($xP$).
- **Custom Player Constraints**: Lock or ban players, configure hit limits, and chip strategies.

## Setup & Running
```bash
uv venv
uv pip install -e .
uv run python server.py
```
