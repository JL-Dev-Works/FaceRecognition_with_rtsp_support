from dotenv import load_dotenv
import os

KNOWN_INTERVAL = 2
UNKNOWN_INTERVAL = 2


## from main.py

# Kies: "webcam", "video" of "rtsp"
SOURCE_MODE = "video"

ANALYSE_INTERVAL = 2
RECOGNITION_THRESHOLD = 80

## from screenshot_manager.py

# Interval voordat screenshot manager nog een screenshot mag maken per categorie.
KNOWN_INTERVAL = 3
UNKNOWN_INTERVAL = 3

# Kies hier of er ook screenshots worden gemaakt.
SCREENSHOTS_ENABLED = False