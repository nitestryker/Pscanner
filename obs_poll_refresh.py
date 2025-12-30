import os
import time
from pathlib import Path

from obsws_python import ReqClient

# ================== CONFIG ==================
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = "eenJ9JPJpFMSXZEf"  # "" if none

BROWSER_SOURCE_NAME = "Browser"
HTML_FILE = r"C:\Users\nites\OneDrive\Desktop\Radio Test\obs_text\full_transcript_log.html" 

POLL_INTERVAL = 1.0  # seconds
# ============================================

def main():
    p = Path(HTML_FILE).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"HTML_FILE not found: {p}")

    client = ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD)
    print("Connected to OBS")
    print(f"Polling: {p} every {POLL_INTERVAL}s")

    last_mtime = p.stat().st_mtime

    while True:
        time.sleep(POLL_INTERVAL)
        mtime = p.stat().st_mtime
        if mtime != last_mtime:
            last_mtime = mtime
            client.press_input_properties_button(
                inputName=BROWSER_SOURCE_NAME,
                propertyName="refreshnocache"
            )
            print(f"Refreshed (change detected): {time.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
