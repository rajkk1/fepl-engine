import os
import time
import requests
import logging
import fpl_api

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def send_expo_push_notification(token: str, gameweek: int, deadline_str: str):
    """Send a push notification via Expo's Push API."""
    url = "https://exp.host/--/api/v2/push/send"
    headers = {
        "Accept": "application/json",
        "Accept-encoding": "gzip, deflate",
        "Content-Type": "application/json",
    }
    payload = {
        "to": token,
        "title": "🚨 FPL Deadline Approaching!",
        "body": f"The Gameweek {gameweek} deadline is coming up! Open your FEPL Manager app to view your optimal Weekly Action Plan.",
        "sound": "default",
        "badge": 1,
        "data": {"gameweek": gameweek, "deadline": deadline_str}
    }
    
    logging.info(f"Sending Push Notification to {token}...")
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        logging.info("Push notification sent successfully.")
    else:
        logging.error(f"Failed to send notification: {response.text}")

def main():
    expo_token = os.getenv("EXPO_TOKEN")
    if not expo_token:
        logging.warning("No EXPO_TOKEN found in environment variables. Skipping notifications.")
        return

    try:
        bootstrap = fpl_api.get_bootstrap_static()
        events = bootstrap.get("events", [])
        
        # Find the next upcoming event
        next_event = next((e for e in events if e.get("is_next")), None)
        
        if not next_event:
            logging.info("No upcoming FPL gameweek found.")
            return
            
        deadline_epoch = next_event.get("deadline_time_epoch", 0)
        gameweek = next_event.get("id")
        deadline_str = next_event.get("deadline_time")
        
        current_time = time.time()
        hours_until_deadline = (deadline_epoch - current_time) / 3600.0
        
        logging.info(f"Next Gameweek: GW{gameweek}")
        logging.info(f"Hours until deadline: {hours_until_deadline:.1f}h")
        
        # If the deadline is within the next 36 hours, send a notification
        if 0 < hours_until_deadline <= 36.0:
            logging.info("Deadline is within 36 hours. Triggering push notification!")
            send_expo_push_notification(expo_token, gameweek, deadline_str)
        else:
            logging.info("Deadline is not within the notification window (36 hours).")
            
    except Exception as e:
        logging.error(f"Error checking deadlines: {e}")

if __name__ == "__main__":
    main()
