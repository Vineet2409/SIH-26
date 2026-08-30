"""
main.py
-------
Ties together Steps 1-3:
  1. video_stream.py    -> reads frames from a camera/file
  2. detector_tracker.py -> detects people/vehicles and tracks them across frames

Run examples:
  Webcam:      python main.py --source 0 --show
  Video file:  python main.py --source sample.mp4 --show --output result.mp4
  RTSP camera: python main.py --source "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101" --show
"""

import argparse
import time

import cv2

from video_stream import VideoStream
from detector_tracker import PersonVehicleDetectorTracker, draw_detections


def parse_args():
    parser = argparse.ArgumentParser(description="IBVAP prototype — Steps 1-3")
    parser.add_argument("--source", required=True,
                         help="0 for webcam, a path to a video file, or an RTSP/HTTP camera URL")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights file")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold (0-1)")
    parser.add_argument("--show", action="store_true", help="Display the annotated video in a window")
    parser.add_argument("--output", default=None, help="Optional path to save the annotated video, e.g. out.mp4")
    return parser.parse_args()


def main():
    args = parse_args()

    # argparse gives us strings — convert "0" to an int for webcam sources.
    source = int(args.source) if args.source.isdigit() else args.source

    print(f"[IBVAP] Opening video source: {source}")
    stream = VideoStream(source)

    print(f"[IBVAP] Loading model: {args.model} (first run downloads weights if not cached)")
    detector_tracker = PersonVehicleDetectorTracker(model_path=args.model, confidence_threshold=args.conf)

    writer = None
    seen_track_ids = set()  # so we can log each object once, the moment it first appears

    frame_count = 0
    start_time = time.time()

    try:
        for frame in stream.frames():
            frame_count += 1
            detections = detector_tracker.process(frame)

            # --- Simple event logging: print a line the first time a new track ID shows up.
            # This is the seed of the "real-time alert generation and event logging"
            # requirement from the problem statement — right now it just logs to the
            # console, but this is exactly where Step 4 (virtual fence rules) and
            # Step 6 (the alert dashboard) will hook in later.
            for d in detections:
                if d.track_id not in seen_track_ids:
                    seen_track_ids.add(d.track_id)
                    print(f"[EVENT] New {d.class_name} detected — track ID #{d.track_id} "
                          f"(confidence {d.confidence:.2f}) at frame {frame_count}")

            annotated = draw_detections(frame, detections)

            if args.output:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.output, fourcc, stream.get_fps(), (w, h))
                writer.write(annotated)

            if args.show:
                cv2.imshow("IBVAP - Steps 1-3 (press q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[IBVAP] Quit requested by user.")
                    break

    except KeyboardInterrupt:
        print("[IBVAP] Interrupted by user (Ctrl+C).")

    finally:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"[IBVAP] Processed {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS). "
              f"Unique objects tracked: {len(seen_track_ids)}")

        stream.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
