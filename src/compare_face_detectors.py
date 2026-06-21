import argparse
import csv
import json
from pathlib import Path

from custom_cnn_inference import (
    DEFAULT_LABELS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPORTS_DIR,
    draw_prediction_panel,
    load_emotion_model,
    load_labels,
    predict_face,
    read_image,
    write_image,
)
from face_detectors import SUPPORTED_DETECTORS, create_face_detector, crop_box, draw_detection


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Haar Cascade and MTCNN face detectors before custom CNN emotion inference."
    )
    parser.add_argument("--image-dir", default="image", help="Folder containing raw face images.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to .keras model.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH), help="Path to labels.json.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_REPORTS_DIR / "detector_comparison"),
        help="Folder to save CSV, JSON and annotated images.",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=SUPPORTED_DETECTORS,
        default=list(SUPPORTED_DETECTORS),
        help="Detectors to compare. Default: haar mtcnn.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top emotion predictions to save.")
    parser.add_argument("--padding", type=float, default=0.18, help="Padding ratio added around face boxes.")
    parser.add_argument("--min-face-size", type=int, default=60, help="Minimum face size in pixels.")
    parser.add_argument("--recursive", action="store_true", help="Search image-dir recursively.")
    return parser.parse_args()


def find_images(image_dir, recursive=False):
    image_dir = Path(image_dir)
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in image_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def top_prediction_columns(prediction, top_k):
    columns = {}
    top_predictions = prediction.get("top_predictions", []) if prediction else []
    for index in range(top_k):
        rank = index + 1
        item = top_predictions[index] if index < len(top_predictions) else {}
        columns[f"top_{rank}_label"] = item.get("label", "")
        columns[f"top_{rank}_percentage"] = item.get("percentage", "")
    return columns


def build_row(image_path, detector_name, detection, prediction, output_image_path, error, top_k):
    row = {
        "image": str(image_path),
        "detector": detector_name,
        "face_detected": detection.face_detected if detection else False,
        "detector_confidence": "" if not detection or detection.confidence is None else detection.confidence,
        "detect_time_ms": "" if not detection else detection.detect_time_ms,
        "face_x": "" if not detection or not detection.face_detected else detection.x,
        "face_y": "" if not detection or not detection.face_detected else detection.y,
        "face_w": "" if not detection or not detection.face_detected else detection.w,
        "face_h": "" if not detection or not detection.face_detected else detection.h,
        "predicted_emotion": "" if not prediction else prediction["predicted_label"],
        "emotion_confidence": "" if not prediction else prediction["confidence"],
        "emotion_confidence_percentage": "" if not prediction else prediction["confidence_percentage"],
        "output_image": str(output_image_path),
        "error": error,
    }
    row.update(top_prediction_columns(prediction, top_k))
    return row


def write_csv(path, rows, top_k):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "detector",
        "face_detected",
        "detector_confidence",
        "detect_time_ms",
        "face_x",
        "face_y",
        "face_w",
        "face_h",
        "predicted_emotion",
        "emotion_confidence",
        "emotion_confidence_percentage",
    ]
    for index in range(1, top_k + 1):
        fieldnames.extend([f"top_{index}_label", f"top_{index}_percentage"])
    fieldnames.extend(["output_image", "error"])

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = {}
    for detector_name in sorted({row["detector"] for row in rows}):
        detector_rows = [row for row in rows if row["detector"] == detector_name]
        detected_rows = [row for row in detector_rows if row["face_detected"]]
        detect_times = [float(row["detect_time_ms"]) for row in detector_rows if row["detect_time_ms"] != ""]
        summary[detector_name] = {
            "total_images": len(detector_rows),
            "detected_images": len(detected_rows),
            "no_face_images": len(detector_rows) - len(detected_rows),
            "face_detection_rate": len(detected_rows) / len(detector_rows) if detector_rows else 0.0,
            "average_detect_time_ms": sum(detect_times) / len(detect_times) if detect_times else None,
        }
    return summary


def main():
    args = parse_args()
    image_paths = find_images(args.image_dir, args.recursive)
    if not image_paths:
        raise RuntimeError(f"No images found in: {args.image_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.labels)
    model = load_emotion_model(args.model)
    detectors = {name: create_face_detector(name, min_face_size=args.min_face_size) for name in args.detectors}

    rows = []
    json_results = []

    for image_path in image_paths:
        image = read_image(image_path)
        for detector_name, detector in detectors.items():
            output_image_path = output_dir / f"{detector_name}_{image_path.stem}.png"
            detection = None
            prediction = None
            error = ""

            try:
                detection = detector.detect_largest(image, padding=args.padding)
                annotated = draw_detection(image, detection)

                if detection.face_detected:
                    face_crop = crop_box(image, detection)
                    prediction = predict_face(model, face_crop, labels, top_k=args.top_k)
                    annotated = draw_prediction_panel(annotated, prediction)
                else:
                    error = "No face detected"

                write_image(output_image_path, annotated)
            except Exception as exc:
                error = str(exc)
                write_image(output_image_path, image)

            row = build_row(image_path, detector_name, detection, prediction, output_image_path, error, args.top_k)
            rows.append(row)
            json_results.append(
                {
                    "image": str(image_path),
                    "detector": detector_name,
                    "detection": None if detection is None else detection.to_dict(),
                    "prediction": prediction,
                    "output_image": str(output_image_path),
                    "error": error,
                }
            )

    csv_path = output_dir / "detector_comparison.csv"
    json_path = output_dir / "detector_comparison.json"
    summary_path = output_dir / "detector_comparison_summary.json"

    write_csv(csv_path, rows, args.top_k)
    json_path.write_text(json.dumps(json_results, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")

    print(f"Compared {len(args.detectors)} detectors on {len(image_paths)} images.")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved JSON to: {json_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved annotated images to: {output_dir}")


if __name__ == "__main__":
    main()
