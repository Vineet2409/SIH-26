# main.py

# Ties together Steps 1-4:
# 1. video_stream.py -> reads frames
# 2. detector_tracker.py -> detects and tracks objects
# 3. Event logging
# 4. Virtual fence alerts

import argparse
import time
import cv2

from video_stream import VideoStream
from detector_tracker import PersonVehicleDetectorTracker, draw_detections

def parse_args():
    parser = argparse.ArgumentParser(
        description="IBVAP prototype — Steps 1-4"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="0 for webcam, a path to a video file, or an RTSP/HTTP camera URL"
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 weights file"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Detection confidence threshold (0-1)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated video in a window"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the annotated video, e.g. out.mp4"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Convert "0" to an integer for webcam sources.
    source = int(args.source) if args.source.isdigit() else args.source

    print(f"[IBVAP] Opening video source: {source}")

    stream = VideoStream(source)

    print(
        f"[IBVAP] Loading model: {args.model} "
        f"(first run downloads weights if not cached)"
    )

    detector_tracker = PersonVehicleDetectorTracker(
        model_path=args.model,
        confidence_threshold=args.conf
    )

    writer = None

    # Keep track of objects that have already appeared.
    seen_track_ids = set()

    # ---------------------------------------------------------
    # VIRTUAL FENCE SETTINGS
    # ---------------------------------------------------------

    # Y-coordinate of the horizontal virtual fence.
    # Increase this number to move the fence DOWN.
    # Decrease this number to move the fence UP.
    fence_y = 300

    # Stores the previous Y-position of every tracked object.
    previous_positions = {}

    # Prevents the same object from generating repeated alerts.
    alerted_tracks = set()

    frame_count = 0
    start_time = time.time()

    try:
        for frame in stream.frames():

            frame_count += 1

            # -------------------------------------------------
            # YOLO DETECTION + TRACKING
            # -------------------------------------------------

            detections = detector_tracker.process(frame)

            # -------------------------------------------------
            # NEW OBJECT EVENT LOGGING
            # -------------------------------------------------

            for d in detections:

                if d.track_id not in seen_track_ids:

                    seen_track_ids.add(d.track_id)

                    print(
                        f"[EVENT] New {d.class_name} detected — "
                        f"track ID #{d.track_id} "
                        f"(confidence {d.confidence:.2f}) "
                        f"at frame {frame_count}"
                    )

            # -------------------------------------------------
            # VIRTUAL FENCE DETECTION
            # -------------------------------------------------

            for d in detections:

                # Get the center of the detected object.
                center_x, center_y = d.center

                # If we have seen this object before,
                # compare its previous position with its current position.
                if d.track_id in previous_positions:

                    previous_y = previous_positions[d.track_id]

                    # Object crossed the fence from TOP to BOTTOM.
                    crossed_down = (
                        previous_y < fence_y <= center_y
                    )

                    # Object crossed the fence from BOTTOM to TOP.
                    crossed_up = (
                        previous_y > fence_y >= center_y
                    )

                    # Generate an alert only once per tracked object.
                    if (
                        (crossed_down or crossed_up)
                        and d.track_id not in alerted_tracks
                    ):

                        alerted_tracks.add(d.track_id)

                        print(
                            f"[ALERT] {d.class_name.capitalize()} "
                            f"#{d.track_id} crossed the virtual fence!"
                        )

                # Save the current position for the next frame.
                previous_positions[d.track_id] = center_y

            # -------------------------------------------------
            # DRAW DETECTIONS
            # -------------------------------------------------

            annotated = draw_detections(frame, detections)

            # -------------------------------------------------
            # DRAW VIRTUAL FENCE
            # -------------------------------------------------

            cv2.line(
                annotated,
                (0, fence_y),
                (annotated.shape[1], fence_y),
                (0, 0, 255),
                3
            )

            cv2.putText(
                annotated,
                "VIRTUAL FENCE",
                (20, fence_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            # -------------------------------------------------
            # SAVE OUTPUT VIDEO
            # -------------------------------------------------

            if args.output:

                if writer is None:

                    h, w = annotated.shape[:2]

                    fourcc = cv2.VideoWriter_fourcc(
                        *"mp4v"
                    )

                    writer = cv2.VideoWriter(
                        args.output,
                        fourcc,
                        stream.get_fps(),
                        (w, h)
                    )

                writer.write(annotated)

            # -------------------------------------------------
            # SHOW CAMERA WINDOW
            # -------------------------------------------------

            if args.show:

                cv2.imshow(
                    "IBVAP - Steps 1-4 (press q to quit)",
                    annotated
                )

                # Press Q to stop.
                if cv2.waitKey(1) & 0xFF == ord("q"):

                    print("[IBVAP] Quit requested by user.")

                    break

    except KeyboardInterrupt:

        print("[IBVAP] Interrupted by user (Ctrl+C).")

    finally:

        elapsed = time.time() - start_time

        fps = (
            frame_count / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"[IBVAP] Processed {frame_count} frames "
            f"in {elapsed:.1f}s "
            f"({fps:.1f} FPS). "
            f"Unique objects tracked: {len(seen_track_ids)}"
        )

        # Release camera/video source.
        stream.release()

        # Release output video if one was created.
        if writer is not None:
            writer.release()

        # Close OpenCV windows.
        if args.show:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
