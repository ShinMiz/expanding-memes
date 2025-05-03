import json
from pathlib import Path
import datetime

LOG_DIR = Path("simulation_logs")
LOG_DIR.mkdir(exist_ok=True)

def save_json(obj, filename):
    with open(LOG_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def log_timestamped_event(event_name: str, data: dict):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{event_name}_{timestamp}.json"
    save_json(data, filename)
