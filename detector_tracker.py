"""
detector_tracker.py
--------------------
Step 2: Detect people and vehicles in each frame.
Step 3: Track the same object across frames (persistent IDs).

We use YOLOv8 (via the `ultralytics` package) for detection, and its
built-in ByteTrack integration for tracking — so steps 2 and 3 happen
together in a single `model.track(...)` call. This is the standard,
well-tested combo for this kind of pipeline; no need to build a tracker
from scratch.

NOTE: The first time you run this, `ultralytics` will automatically
download the model weights file (yolov8n.pt, ~6MB) — you need an internet
connection for that one-time download. After that it's fully offline.
"""

from dataclasses import dataclass
from typing import List

import cv2
from ultralytics import YOLO

# COCO class IDs we care about for border surveillance.
# (YOLOv8's default model is trained on the COCO dataset, which already
# includes "person" and several vehicle classes — no custom training needed
# to get started.)
RELEVANT_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# One consistent color per class, just for on-screen visualization.
CLASS_COLORS = {
    "person": (0, 255, 0),       # green
    "bicycle": (255, 200, 0),
    "car": (0, 165, 255),
    "motorcycle": (0, 165, 255),
    "bus": (0, 0, 255),
    "truck": (0, 0, 255),
}


@dataclass
class Detection:
    track_id: int          # persistent ID — same object keeps the same ID across frames
    class_name: str        # "person", "car", "truck", etc.
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class PersonVehicleDetectorTracker:
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.4):
        """
        model_path: which YOLOv8 weights to use.
            "yolov8n.pt" = nano, fastest, good for edge devices / real-time demo.
            "yolov8s.pt" / "yolov8m.pt" = more accurate but slower — use these
            if you have a decent GPU and want better accuracy over speed.
        confidence_threshold: minimum detection confidence (0-1) to keep a box.
            Raise this if you're seeing too many false detections.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def process(self, frame) -> List[Detection]:
        """
        Runs detection + tracking on a single frame.
        Returns a list of Detection objects — empty list if nothing relevant was found.
        """
        results = self.model.track(
            frame,
            persist=True,            # remember tracks between calls -> stable IDs over time
            tracker="bytetrack.yaml",  # ByteTrack: fast, reliable, no extra model to load
            classes=list(RELEVANT_CLASSES.keys()),  # ignore irrelevant COCO classes (dog, chair, etc.)
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections: List[Detection] = []
        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            # `id` is None on frames where tracking hasn't locked onto anything yet
            # (e.g. the very first frame) — just return no detections for that frame.
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
            x1, y1, x2, y2 = box.astype(int)
            detections.append(
                Detection(
                    track_id=int(track_id),
                    class_name=RELEVANT_CLASSES.get(int(class_id), "unknown"),
                    confidence=float(conf),
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                )
            )
        return detections


def draw_detections(frame, detections: List[Detection]):
    """
    Draws bounding boxes + track ID + class label on the frame, in place.
    Also returns the frame for convenience (e.g. `frame = draw_detections(frame, dets)`).
    """
    for d in detections:
        color = CLASS_COLORS.get(d.class_name, (255, 255, 255))
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)

        label = f"#{d.track_id} {d.class_name} {d.confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (d.x1, d.y1 - text_h - 6), (d.x1 + text_w + 4, d.y1), color, -1)
        cv2.putText(frame, label, (d.x1 + 2, d.y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame
