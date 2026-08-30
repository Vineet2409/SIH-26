# IBVAP Prototype — Steps 1-3

This is a working prototype covering:
- **Step 1** (`video_stream.py`) — reads a video feed from a webcam, video file, or RTSP camera
- **Step 2** (`detector_tracker.py`) — detects people and vehicles in each frame using YOLOv8
- **Step 3** (`detector_tracker.py`) — tracks each detected object across frames with a persistent ID (via ByteTrack)

`main.py` wires all three together into something you can actually run and see working.

---

## Setup

You'll need Python 3.9+ and an internet connection (only for the one-time setup below — after that, video file/webcam use works fully offline).

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

The first time you run `main.py`, `ultralytics` will automatically download the YOLOv8 model weights (`yolov8n.pt`, ~6MB) — this needs internet once, then it's cached locally.

---

## Running it

**With your webcam** (easiest way to test):
```bash
python main.py --source 0 --show
```

**On a video file** (e.g. sample CCTV footage you download for testing):
```bash
python main.py --source path/to/video.mp4 --show --output result.mp4
```

**On a real IP camera via RTSP:**
```bash
python main.py --source "rtsp://username:password@192.168.1.64:554/Streaming/Channels/101" --show
```
(The exact RTSP URL format depends on the camera brand — check its manual or admin web page. Hikvision, Dahua, and CP Plus are common Indian CCTV brands and their RTSP formats are easy to find online.)

**Useful flags:**
| Flag | What it does |
|---|---|
| `--show` | Opens a live window with boxes + track IDs drawn on the video |
| `--output result.mp4` | Saves the annotated video to a file |
| `--conf 0.5` | Raise this if you're getting too many false detections (default 0.4) |
| `--model yolov8s.pt` | Use a more accurate (but slower) model if you have a decent GPU |

---

## What you should see

- A window (if `--show` is used) with green boxes around people and orange/red boxes around vehicles, each labeled like `#3 person 0.87` — the `#3` is the persistent track ID.
- Console output like:
  ```
  [EVENT] New person detected — track ID #1 (confidence 0.91) at frame 12
  [EVENT] New car detected — track ID #2 (confidence 0.85) at frame 18
  ```
  This is a first, minimal version of the "event logging" requirement from the problem
  statement — every new object that appears gets logged once. This is exactly the hook
  point for Step 4 (virtual fence rules) later: instead of just printing to console,
  you'll check *where* each track is and raise a real alert if it crosses a line.

---

## If you don't have a GPU

`yolov8n.pt` (the default) is the smallest, fastest YOLOv8 model and runs reasonably well
on just a CPU for demo purposes — you don't need a GPU to get this working for your
hackathon demo. A GPU (or a Jetson edge box later) matters more when you want higher
frame rates on multiple camera streams simultaneously.

---

## Next steps (Steps 4-8, not in this prototype yet)

- **Step 4:** let a user draw a line/zone on the frame, check if a tracked object's
  `center` (already available on each `Detection`) crosses it — this hooks directly
  into the `[EVENT]` logging block in `main.py`.
- **Step 5:** ANPR — crop the vehicle boxes already being detected here and run OCR on them.
- **Step 6:** replace the console `print()` alerts with a proper web dashboard.
- **Step 7-8:** tune thresholds, then package onto a small edge device (e.g. Jetson).
