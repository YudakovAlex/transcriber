import sys
import threading
import time


class Spinner:
    def __init__(self, message):
        self._message = message
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_event.set()
        self._thread.join(timeout=1)
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    def _spin(self):
        frames = "|/-\\"
        idx = 0
        while not self._stop_event.is_set():
            sys.stdout.write(f"\r{frames[idx % len(frames)]} {self._message}")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.2)
