"""
video_stream.py
----------------
Step 1: Get a video feed into your code.

Wraps OpenCV's VideoCapture so the rest of the pipeline doesn't care whether
the source is a webcam, a saved video file, or a live RTSP camera stream.

For a real CCTV camera, `source` will usually look like:
    rtsp://username:password@192.168.1.64:554/Streaming/Channels/101
(Every camera brand has its own RTSP path — check the camera's manual or
its web admin page for the exact URL.)
"""

import time
import cv2


class VideoStream:
    """
    Usage:
        stream = VideoStream(source="rtsp://...")   # or 0 for webcam, or "video.mp4"
        for frame in stream.frames():
            ...
        stream.release()
    """

    def __init__(self, source, reconnect_delay_sec: float = 2.0, max_reconnect_attempts: int = 5):
        """
        source: int (webcam index, e.g. 0), str path to a video file, or an RTSP/HTTP stream URL.
        reconnect_delay_sec: how long to wait before retrying a dropped connection.
        max_reconnect_attempts: how many times to retry before giving up (only matters for live streams).
        """
        self.source = source
        self.is_live_stream = isinstance(source, str) and source.lower().startswith(("rtsp://", "http://", "https://"))
        self.reconnect_delay_sec = reconnect_delay_sec
        self.max_reconnect_attempts = max_reconnect_attempts
        self.cap = None
        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ConnectionError(f"Could not open video source: {self.source}")

    def get_fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Some RTSP cameras report 0 or nonsense FPS — fall back to a sane default.
        return fps if fps and fps > 0 else 25.0

    def _try_reconnect(self) -> bool:
        """Attempt to re-open a dropped live stream. Returns True if successful."""
        print(f"[VideoStream] Connection lost to {self.source}, attempting to reconnect...")
        self.cap.release()
        for attempt in range(1, self.max_reconnect_attempts + 1):
            time.sleep(self.reconnect_delay_sec)
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                print(f"[VideoStream] Reconnected after {attempt} attempt(s).")
                return True
            print(f"[VideoStream] Reconnect attempt {attempt}/{self.max_reconnect_attempts} failed.")
        return False

    def frames(self):
        """
        Generator that yields video frames one at a time (as numpy arrays / BGR images).
        For a video FILE, it stops naturally at the end.
        For a LIVE stream (webcam/RTSP), it keeps going and tries to reconnect on failure.
        """
        while True:
            ok, frame = self.cap.read()

            if ok:
                yield frame
                continue

            # Read failed.
            if self.is_live_stream:
                if self._try_reconnect():
                    continue
                else:
                    print("[VideoStream] Giving up after repeated reconnect failures.")
                    break
            else:
                # It's a finite video file — this just means we reached the end.
                break

    def release(self):
        if self.cap is not None:
            self.cap.release()


if __name__ == "__main__":
    # Quick self-test: generates a tiny synthetic video and reads it back,
    # so you can confirm this file works before wiring in a real camera.
    import numpy as np
    import tempfile
    import os

    tmp_path = os.path.join(tempfile.gettempdir(), "ibvap_selftest.mp4")
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(20):
        frame = np.full((240, 320, 3), (i * 10) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    stream = VideoStream(tmp_path)
    count = sum(1 for _ in stream.frames())
    stream.release()
    os.remove(tmp_path)

    print(f"Self-test OK — read {count} frames from a synthetic test video (expected 20).")
