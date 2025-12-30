import time
from obsws_python import ReqClient

# ================== CONFIG ==================
OBS_HOST = "localhost"
OBS_PORT = 4455            # OBS 28+ default
OBS_PASSWORD = "eenJ9JPJpFMSXZEf"  # leave "" if none

BROWSER_SOURCE_NAME = "Browser" # exact name in OBS
REFRESH_INTERVAL = 1       # seconds
# ============================================

def main():
    client = ReqClient(
        host=OBS_HOST,
        port=OBS_PORT,
        password=OBS_PASSWORD
    )

    print("Connected to OBS")
    print(f"Refreshing '{BROWSER_SOURCE_NAME}' every {REFRESH_INTERVAL}s")

    while True:
        client.press_input_properties_button(
            inputName=BROWSER_SOURCE_NAME,
            propertyName="refreshnocache"
        )
        time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    main()
