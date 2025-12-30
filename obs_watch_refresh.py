import os
import time
from pathlib import Path

from obsws_python import ReqClient
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# ================== CONFIG ==================
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = "eenJ9JPJpFMSXZEf"  # "" if none

BROWSER_SOURCE_NAME = "Browser"  # exact name in OBS
HTML_FILE = r"C:\Users\nites\OneDrive\Desktop\Radio Test\obs_text\full_transcript_log.html"  # file to watch
DEBOUNCE_MS = 500  # combine rapid saves into one refresh
# ============================================


class DebouncedRefreshHandler(FileSystemEventHandler):
    def __init__(self, target_file: Path, refresh_fn, debounce_ms: int):
        self.target_file = target_file.resolve()
        self.refresh_fn = refresh_fn
        self.debounce_s = debounce_ms / 1000.0
        self._last_refresh = 0.0

    def on_modified(self, event):
        if event.is_directory:
            return
        changed = Path(event.src_path).resolve()
        if changed != self.target_file:
            return

        now = time.time()
        if (now - self._last_refresh) >= self.debounce_s:
            self._last_refresh = now
            self.refresh_fn()

    # Some editors write via temp file + rename
    def on_moved(self, event):
        if event.is_directory:
            return
        dest = Path(getattr(event, "dest_path", "")).resolve()
        if dest == self.target_file:
            self.on_modified(event)


def main():
    html_path = Path(HTML_FILE).expanduser()
    if not html_path.exists():
        raise FileNotFoundError(f"HTML_FILE not found: {html_path}")

    client = ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD)
    print("Connected to OBS")
    print(f"Watching: {html_path}")
    print(f"Refreshing source: {BROWSER_SOURCE_NAME}")

    def refresh_browser_source():
        try:
            client.press_input_properties_button(
                inputName=BROWSER_SOURCE_NAME,
                propertyName="refreshnocache",
            )
            print(f"Refreshed (change detected): {time.strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"Refresh failed: {e}")

    handler = DebouncedRefreshHandler(
        target_file=html_path,
        refresh_fn=refresh_browser_source,
        debounce_ms=DEBOUNCE_MS,
    )

    observer = Observer()
    observer.schedule(handler, str(html_path.parent), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        print("Stopped")


if __name__ == "__main__":
    main()
