import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
import platform
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DEEPFACE_HOME", str(PROJECT_ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from deepface import DeepFace  # noqa: E402


EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 0),
    "fear": (128, 0, 128),
    "happy": (0, 200, 255),
    "sad": (255, 0, 0),
    "surprise": (0, 255, 255),
    "neutral": (220, 220, 220),
}


def open_video_capture(source):
    if isinstance(source, int) and platform.system() == "Darwin":
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open camera/video source. On macOS, allow Camera access for the app "
            "running this script in System Settings > Privacy & Security > Camera. "
            "If the wrong camera is selected, try --source 1 or --source 2."
        )
    return cap


def read_initial_frame(cap, warmup_frames=5):
    frame = None
    ok = False
    for _ in range(warmup_frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.05)

    if not ok or frame is None:
        raise RuntimeError(
            "Camera opened but did not return frames. Check Camera permission, close other "
            "apps using the webcam, or try another --source index."
        )
    return frame


def is_nearly_black(frame, threshold=3.0):
    return float(frame.mean()) < threshold


def build_fast_detector():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Cannot load OpenCV face cascade: {cascade_path}")
    return detector


def expand_box(box, frame_shape, ratio=0.18):
    x, y, w, h = box
    frame_h, frame_w = frame_shape[:2]
    pad_x = int(w * ratio)
    pad_y = int(h * ratio)
    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_y, 0)
    x2 = min(x + w + pad_x, frame_w)
    y2 = min(y + h + pad_y, frame_h)
    return x1, y1, x2 - x1, y2 - y1


def detect_fast(detector, frame):
    frame_h, frame_w = frame.shape[:2]
    target_w = 360
    scale = min(1.0, target_w / float(frame_w))
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    min_size = max(48, int(min(gray.shape[:2]) * 0.12))
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    if scale < 1:
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

    return expand_box((x, y, w, h), (frame_h, frame_w))


def average_emotions(history):
    totals = {}
    for emotions in history:
        for emotion, score in emotions.items():
            totals[emotion] = totals.get(emotion, 0.0) + float(score)
    return {emotion: score / len(history) for emotion, score in totals.items()}


def analyze_face_crop(frame, box, history):
    x, y, w, h = box
    face = frame[y : y + h, x : x + w]
    analyzed = DeepFace.analyze(
        img_path=face,
        actions=("emotion",),
        detector_backend="skip",
        enforce_detection=False,
        silent=True,
    )
    result = analyzed[0] if isinstance(analyzed, list) else analyzed
    history.append(dict(result.get("emotion") or {}))

    emotions = average_emotions(history)
    dominant = max(emotions, key=emotions.get)
    return {
        "region": {"x": x, "y": y, "w": w, "h": h},
        "emotion": emotions,
        "dominant_emotion": dominant,
    }


def is_valid_face_region(frame, result):
    region = result.get("region") or {}
    x = int(region.get("x", 0))
    y = int(region.get("y", 0))
    w = int(region.get("w", 0))
    h = int(region.get("h", 0))
    frame_h, frame_w = frame.shape[:2]

    if w <= 0 or h <= 0:
        return False

    area_ratio = (w * h) / float(frame_w * frame_h)
    full_width = w >= frame_w * 0.92 and x <= frame_w * 0.04
    full_height = h >= frame_h * 0.92 and y <= frame_h * 0.04

    return area_ratio < 0.75 and not (full_width and full_height)


def analyze_frame(frame, args, fast_detector, emotion_history):
    if args.mode == "fast":
        box = detect_fast(fast_detector, frame)
        if box is None:
            raise ValueError("No face detected")
        return [analyze_face_crop(frame, box, emotion_history)]

    analyzed = DeepFace.analyze(
        img_path=frame,
        actions=("emotion",),
        detector_backend=args.detector,
        enforce_detection=not args.allow_fallback,
        silent=True,
    )
    results = analyzed if isinstance(analyzed, list) else [analyzed]
    results = [result for result in results if args.allow_fallback or is_valid_face_region(frame, result)]
    if not results:
        return []

    emotion_history.append(dict(results[0].get("emotion") or {}))
    emotions = average_emotions(emotion_history)
    dominant = max(emotions, key=emotions.get)
    results[0]["emotion"] = emotions
    results[0]["dominant_emotion"] = dominant
    return results


def put_status(frame, text, color=(255, 255, 255)):
    cv2.putText(frame, text, (10, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_emotions(frame, result):
    region = result.get("region") or {}
    x = int(region.get("x", 0))
    y = int(region.get("y", 0))
    w = int(region.get("w", 0))
    h = int(region.get("h", 0))

    dominant = result.get("dominant_emotion", "unknown")
    score = float((result.get("emotion") or {}).get(dominant, 0.0))
    color = EMOTION_COLORS.get(dominant, (255, 255, 255))

    if w > 0 and h > 0:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label = f"{dominant}: {score:.1f}%"
    label_y = max(y - 10, 30)
    cv2.putText(frame, label, (max(x, 10), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    emotions = sorted((result.get("emotion") or {}).items(), key=lambda item: item[1], reverse=True)
    panel_x = max(x, 10)
    panel_y = y + h + 28 if h > 0 else 34
    for index, (emotion, value) in enumerate(emotions[:4]):
        text = f"{emotion:<8} {value:5.1f}%"
        cv2.putText(
            frame,
            text,
            (panel_x, panel_y + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            EMOTION_COLORS.get(emotion, (255, 255, 255)),
            2,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime facial emotion recognition from webcam.")
    parser.add_argument("--source", default=0, help="Camera index or video path. Default: 0")
    parser.add_argument(
        "--mode",
        choices=("fast", "deep"),
        default="deep",
        help="fast uses OpenCV Haar detection. deep uses DeepFace detector.",
    )
    parser.add_argument("--detector", default="mtcnn", help="DeepFace detector backend for --mode deep.")
    parser.add_argument("--every", type=int, default=12, help="Analyze every N frames.")
    parser.add_argument("--smooth", type=int, default=8, help="Average the last N predictions.")
    parser.add_argument("--allow-fallback", action="store_true", help="Allow full-frame fallback for debugging.")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = open_video_capture(source)
    first_frame = read_initial_frame(cap)
    if is_nearly_black(first_frame):
        print(
            "Warning: camera is returning almost-black frames. Check lens cover, room light, "
            "camera privacy settings, or try another --source index."
        )

    fast_detector = build_fast_detector() if args.mode == "fast" else None
    emotion_history = deque(maxlen=max(args.smooth, 1))

    print("Opening camera. Press q in the video window to quit.")
    last_results = []
    last_error = ""
    pending = None
    frame_index = 0
    started = time.time()

    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if pending is not None and pending.done():
                try:
                    last_results = pending.result()
                    last_error = "" if last_results else "No face detected"
                except Exception as exc:
                    last_results = []
                    last_error = str(exc)
                    emotion_history.clear()
                pending = None

            if pending is None and (frame_index % max(args.every, 1) == 0 or not last_results):
                pending = executor.submit(
                    analyze_frame,
                    frame.copy(),
                    args,
                    fast_detector,
                    emotion_history,
                )

            for result in last_results:
                draw_emotions(frame, result)

            elapsed = max(time.time() - started, 0.001)
            fps = (frame_index + 1) / elapsed
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if last_error:
                put_status(frame, "No face detected - improve front light or try --mode fast", (0, 0, 255))
            elif not last_results:
                put_status(frame, "Analyzing...", (0, 200, 255))

            cv2.imshow("Facial Emotion Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
