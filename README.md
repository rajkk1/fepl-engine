# FEPL Engine (Zero-Server FPL Optimizer)

This repository contains the backend engine for a fully autonomous Fantasy Premier League optimization system. It uses Integer Linear Programming (ILP) with `pulp` to calculate mathematically optimal transfers, starting XIs, and captaincy choices for a specific FPL team.

## Architecture

This project uses a **Zero-Server Architecture** to eliminate cloud hosting costs:

1. **Daily Automation (GitHub Actions)**: Every day at 12:00 PM UTC, GitHub Actions spins up a runner, installs the Python dependencies, and runs `weekly_manager.py`.
2. **ILP Math (PuLP)**: The script queries the official FPL API for player data, injury status, and expected points (xP). It formulates a complex ILP math problem enforcing constraints like a £100.0m budget cap, maximum 3 players per team, and valid formations.
3. **Static JSON Snapshot**: The optimal team strategy is exported as `weekly_plan.json` and automatically deployed to a `gh-pages` branch, making it publicly available at a static GitHub Pages URL.
4. **Push Notifications**: 36 hours before an FPL Gameweek deadline, GitHub Actions runs `notify.py` to ping your private Discord server with a beautiful alert!
5. **React Native Android App**: The companion Android app is a simple, stateless UI viewer that fetches the static JSON file and renders it beautifully without needing to compute the heavy math locally.

## Setup Instructions

If you fork this repository to use for yourself, you need to configure two GitHub Repository Secrets (`Settings > Secrets and variables > Actions`):

- `FPL_TEAM_ID`: Your official FPL Team ID (found in the URL when viewing your team points).
- `DISCORD_WEBHOOK_URL`: Your private Discord Webhook URL for deadline alerts.

Make sure to enable GitHub Pages to serve from the `gh-pages` branch to make the JSON public!

## Local Development

If you want to run the math locally on your own PC:

```bash
uv sync
uv run python weekly_manager.py
```
