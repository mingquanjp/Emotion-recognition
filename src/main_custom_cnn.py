import argparse
import csv
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from custom_cnn_inference import (
    DEFAULT_LABELS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPORTS_DIR,
    load_emotion_model,
    load_labels,
    predict_face,
)
from face_detectors import SUPPORTED_DETECTORS, create_face_detector, crop_box


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


def average_probabilities(history, labels):
    probabilities = np.zeros(len(labels), dtype=np.float32)
    for item in history:
        probabilities += np.array([item[label] for label in labels], dtype=np.float32)
    probabilities /= max(len(history), 1)
    return probabilities


def prediction_from_probabilities(probabilities, labels, top_k):
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_predictions = [
        {
            "label_id": int(index),
            "label": labels[int(index)],
            "probability": float(probabilities[index]),
            "percentage": float(probabilities[index] * 100.0),
        }
        for index in top_indices
    ]
    best = top_predictions[0]
    return {
        "predicted_label_id": best["label_id"],
        "predicted_label": best["label"],
        "confidence": best["probability"],
        "confidence_percentage": best["percentage"],
        "top_predictions": top_predictions,
        "probabilities": {
            labels[index]: float(probabilities[index])
            for index in range(len(labels))
        },
    }


def analyze_frame(frame, model, labels, detector, args, probability_history):
    detection = detector.detect_largest(frame, padding=args.padding)
    if not detection.face_detected:
        return detection, None, 0.0

    face_crop = crop_box(frame, detection)
    started = time.perf_counter()
    prediction = predict_face(model, face_crop, labels, top_k=args.top_k)
    predict_time_ms = (time.perf_counter() - started) * 1000.0

    probability_history.append(prediction["probabilities"])
    smoothed = average_probabilities(probability_history, labels)
    smoothed_prediction = prediction_from_probabilities(smoothed, labels, args.top_k)
    return detection, smoothed_prediction, predict_time_ms


def draw_prediction(frame, detection, prediction):
    if not detection or not detection.face_detected or not prediction:
        return

    dominant = prediction["predicted_label"]
    color = EMOTION_COLORS.get(dominant, (255, 255, 255))
    x, y, w, h = detection.x, detection.y, detection.w, detection.h

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label = f"{dominant}: {prediction['confidence_percentage']:.1f}%"
    label_y = max(y - 10, 30)
    cv2.putText(frame, label, (max(x, 10), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    detector_label = detection.detector
    if detection.confidence is not None:
        detector_label = f"{detector_label}: {detection.confidence:.2f}"
    cv2.putText(
        frame,
        detector_label,
        (max(x, 10), min(y + h + 24, frame.shape[0] - 72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )

    panel_x = max(x, 10)
    panel_y = min(y + h + 50, frame.shape[0] - 48)
    for index, item in enumerate(prediction["top_predictions"]):
        text = f"{item['label']:<8} {item['percentage']:5.1f}%"
        cv2.putText(
            frame,
            text,
            (panel_x, panel_y + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            EMOTION_COLORS.get(item["label"], (255, 255, 255)),
            2,
        )


def put_status(frame, text, color=(255, 255, 255)):
    cv2.putText(frame, text, (10, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def build_log_path(args):
    if args.log_path:
        return Path(args.log_path)
    return DEFAULT_REPORTS_DIR / "webcam_logs" / f"webcam_{args.detector}_log.csv"


def open_log_writer(args):
    if not args.save_log:
        return None, None

    log_path = build_log_path(args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", newline="", encoding="utf-8")
    fieldnames = [
        "frame_index",
        "timestamp",
        "detector",
        "face_detected",
        "detector_confidence",
        "detect_time_ms",
        "predict_time_ms",
        "predicted_emotion",
        "emotion_confidence",
        "emotion_confidence_percentage",
        "fps",
    ]
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()
    print(f"Saving webcam log to: {log_path}")
    return log_file, writer


def write_log_row(writer, frame_index, detector_name, detection, prediction, predict_time_ms, fps):
    if writer is None:
        return
    writer.writerow(
        {
            "frame_index": frame_index,
            "timestamp": time.time(),
            "detector": detector_name,
            "face_detected": bool(detection and detection.face_detected),
            "detector_confidence": "" if not detection or detection.confidence is None else detection.confidence,
            "detect_time_ms": "" if not detection else detection.detect_time_ms,
            "predict_time_ms": predict_time_ms,
            "predicted_emotion": "" if not prediction else prediction["predicted_label"],
            "emotion_confidence": "" if not prediction else prediction["confidence"],
            "emotion_confidence_percentage": "" if not prediction else prediction["confidence_percentage"],
            "fps": fps,
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime webcam emotion recognition with the custom CNN model.")
    parser.add_argument("--source", default=0, help="Camera index or video path. Default: 0")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to the trained custom CNN .keras model.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH), help="Path to labels.json.")
    parser.add_argument(
        "--detector",
        choices=SUPPORTED_DETECTORS,
        default="mtcnn",
        help="Face detector before custom CNN inference. Default: mtcnn.",
    )
    parser.add_argument("--every", type=int, default=8, help="Run detection and prediction every N frames.")
    parser.add_argument("--smooth", type=int, default=8, help="Average the last N emotion probability vectors.")
    parser.add_argument("--padding", type=float, default=0.18, help="Padding ratio added around detected face boxes.")
    parser.add_argument("--min-face-size", type=int, default=60, help="Minimum face size in pixels.")
    parser.add_argument("--max-stale", type=int, default=24, help="Clear old predictions after this many stale frames.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to draw.")
    parser.add_argument("--save-log", action="store_true", help="Save per-analysis webcam metrics to CSV.")
    parser.add_argument("--log-path", help="Optional path for --save-log CSV output.")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    labels = load_labels(args.labels)
    model = load_emotion_model(args.model)
    detector = create_face_detector(args.detector, min_face_size=args.min_face_size)

    cap = open_video_capture(source)
    first_frame = read_initial_frame(cap)
    if is_nearly_black(first_frame):
        print(
            "Warning: camera is returning almost-black frames. Check lens cover, room light, "
            "camera privacy settings, or try another --source index."
        )

    log_file, log_writer = open_log_writer(args)
    probability_history = deque(maxlen=max(args.smooth, 1))
    last_detection = None
    last_prediction = None
    last_error = ""
    last_analysis_frame = -max(args.max_stale, 1)
    frame_index = 0
    started = time.time()
    pending = None
    pending_frame_index = None

    print("Opening camera with custom CNN. Press q in the video window to quit.")
    print(f"Detector: {args.detector}, every={args.every}, smooth={args.smooth}, padding={args.padding}")

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                elapsed = max(time.time() - started, 0.001)
                fps = (frame_index + 1) / elapsed

                if pending is not None and pending.done():
                    try:
                        detection, prediction, predict_time_ms = pending.result()
                        if prediction is None:
                            last_error = "No face detected"
                            last_detection = None
                            last_prediction = None
                            probability_history.clear()
                        else:
                            last_error = ""
                            last_detection = detection
                            last_prediction = prediction
                            last_analysis_frame = pending_frame_index
                    except Exception as exc:
                        detection = None
                        predict_time_ms = 0.0
                        last_error = str(exc)
                        last_detection = None
                        last_prediction = None
                        probability_history.clear()

                    write_log_row(
                        log_writer,
                        pending_frame_index,
                        args.detector,
                        detection,
                        last_prediction,
                        predict_time_ms,
                        fps,
                    )
                    pending = None
                    pending_frame_index = None

                should_analyze = pending is None and frame_index % max(args.every, 1) == 0
                if should_analyze:
                    pending = executor.submit(
                        analyze_frame,
                        frame.copy(),
                        model,
                        labels,
                        detector,
                        args,
                        probability_history,
                    )
                    pending_frame_index = frame_index

                if frame_index - last_analysis_frame > max(args.max_stale, 1):
                    last_detection = None
                    last_prediction = None
                    probability_history.clear()

                draw_prediction(frame, last_detection, last_prediction)

                cv2.putText(
                    frame,
                    f"FPS {fps:.1f}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                if pending is not None and last_prediction is None:
                    put_status(frame, "Analyzing...", (0, 200, 255))
                elif last_error:
                    put_status(frame, last_error, (0, 0, 255))

                cv2.imshow("Custom CNN Emotion Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_index += 1
    finally:
        cap.release()
        if log_file is not None:
            log_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
