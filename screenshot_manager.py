
import os
import cv2
import time
from datetime import datetime
from config import KNOWN_INTERVAL, UNKNOWN_INTERVAL, SCREENSHOTS_ENABLED


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
KNOWN_DIR = os.path.join(SCREENSHOT_DIR, "Known")
UNKNOWN_DIR = os.path.join(SCREENSHOT_DIR, "Unknown")

# Interval per categorie
KNOWN_INTERVAL = KNOWN_INTERVAL
UNKNOWN_INTERVAL = UNKNOWN_INTERVAL

# Aparte timers
laatste_known_screenshot = 0
laatste_unknown_screenshot = 0


def save_screenshot(frame, json_data):

    global laatste_known_screenshot
    global laatste_unknown_screenshot

    if not SCREENSHOTS_ENABLED:
        return

    if not json_data.get("face_detected", False):
        return

    person = json_data.get("person", "Unknown")
    recognized = json_data.get("recognized", False)

    huidige_tijd = time.time()

    # Known
    if recognized:

        if huidige_tijd - laatste_known_screenshot < KNOWN_INTERVAL:
            return

        target_dir = KNOWN_DIR
        laatste_known_screenshot = huidige_tijd

    # Unknown
    else:

        if huidige_tijd - laatste_unknown_screenshot < UNKNOWN_INTERVAL:
            return

        target_dir = UNKNOWN_DIR
        laatste_unknown_screenshot = huidige_tijd

    os.makedirs(target_dir, exist_ok=True)

    safe_person = "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in str(person)
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S_%f"
    )

    filename = f"{timestamp}_{safe_person}.jpg"

    filepath = os.path.join(target_dir, filename)

    success = cv2.imwrite(filepath, frame)

    if success:
        print(f"[SCREENSHOT] Opgeslagen: {filepath}")
    else:
        print("[SCREENSHOT] Opslaan mislukt!")
