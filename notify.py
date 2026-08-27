import os
import json
import requests
import datetime
from fpl_api import get_bootstrap_static

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def check_deadline_and_notify():
    if not DISCORD_WEBHOOK_URL:
        print("Error: DISCORD_WEBHOOK_URL not found in environment variables.")
        return

    # 1. Get FPL events data
    try:
        bootstrap = get_bootstrap_static()
        events = bootstrap.get("events", [])
    except Exception as e:
        print(f"Failed to fetch FPL API: {e}")
        return

    # 2. Find the next Gameweek
    next_event = None
    for event in events:
        if event.get("is_next"):
            next_event = event
            break

    if not next_event:
        print("No upcoming gameweek found.")
        return

    gw_name = next_event.get("name", "Unknown Gameweek")
    deadline_str = next_event.get("deadline_time", "")
    
    if not deadline_str:
        print("No deadline time found for the next gameweek.")
        return

    # 3. Calculate time until deadline
    # E-03: Use fromisoformat to safely handle fractional seconds/offsets instead of strptime
    clean_deadline = deadline_str.replace("Z", "+00:00")
    deadline_dt = datetime.datetime.fromisoformat(clean_deadline)
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    time_diff = deadline_dt - now_dt
    hours_until = time_diff.total_seconds() / 3600

    print(f"Current UTC time: {now_dt}")
    print(f"Next Gameweek: {gw_name}, Deadline: {deadline_dt}")
    print(f"Hours until deadline: {hours_until:.2f}")

    # E-03: Track a sent-alert marker so we don't fire on two consecutive days for double gameweeks
    sent_marker_file = "public/sent_alerts.json"
    sent_alerts = {}
    if os.path.exists(sent_marker_file):
        try:
            with open(sent_marker_file, "r") as f:
                sent_alerts = json.load(f)
        except Exception:
            pass
            
    if sent_alerts.get(gw_name):
        print(f"Alert for {gw_name} was already sent previously. Skipping.")
        return

    # 4. Trigger alert if within 36 hours (but not already passed)
    if 0 < hours_until <= 36:
        print("Deadline is within 36 hours! Sending Discord notification...")
        send_discord_alert(gw_name, hours_until)
        
        # Save the marker
        sent_alerts[gw_name] = True
        os.makedirs("public", exist_ok=True)
        with open(sent_marker_file, "w") as f:
            json.dump(sent_alerts, f)
    else:
        print("Deadline is not within the 36-hour alert window. No notification sent.")

def send_discord_alert(gw_name, hours_until):
    """Sends a rich embedded message to a Discord Webhook."""
    
    # Read the latest JSON to tell them who to captain!
    captain = "Unknown"
    try:
        with open("public/weekly_plan.json", "r") as f:
            plan = json.load(f)
            captain_obj = plan.get("captain")
            if captain_obj:
                captain = captain_obj.get("web_name", "Unknown")
    except Exception:
        pass

    embed = {
        "title": f"🚨 FPL Deadline Alert: {gw_name}",
        "description": f"The deadline for **{gw_name}** is in exactly **{int(hours_until)} hours**!\n\nOpen your FEPL Assistant app to review your mathematically optimal transfers.",
        "color": 16711680, # Red
        "fields": [
            {
                "name": "Recommended Captain",
                "value": f"👑 {captain}",
                "inline": True
            },
            {
                "name": "App Link",
                "value": "[Open FPL App](https://rajkk1.github.io/fepl-engine/app.html)",
                "inline": True
            }
        ],
        "footer": {
            "text": "FEPL Zero-Server Engine"
        }
    }
    
    payload = {
        "username": "FPL Assistant",
        "avatar_url": "https://fantasy.premierleague.com/static/libs/ext/img/icon.png",
        "embeds": [embed]
    }

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        if response.status_code in [200, 204]:
            print("Discord notification sent successfully!")
        else:
            print(f"Failed to send Discord notification: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Exception while sending Discord notification: {e}")

if __name__ == "__main__":
    check_deadline_and_notify()
