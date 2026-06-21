import argparse
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
    write_json,
)
from face_detectors import SUPPORTED_DETECTORS, create_face_detector, crop_box, draw_detection


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect the largest face in a raw image and run custom CNN emotion inference."
    )
    parser.add_argument("--image", required=True, help="Path to a raw image.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to .keras model.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH), help="Path to labels.json.")
    parser.add_argument(
        "--detector",
        choices=SUPPORTED_DETECTORS,
        default="haar",
        help="Face detector used before custom CNN inference. Default: haar.",
    )
    parser.add_argument(
        "--output-image",
        default=str(DEFAULT_REPORTS_DIR / "face_inference_result.png"),
        help="Path to save annotated result image.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_REPORTS_DIR / "face_inference_result.json"),
        help="Path to save inference result JSON.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to print.")
    parser.add_argument(
        "--padding",
        type=float,
        default=0.18,
        help="Padding ratio added around the detected face box.",
    )
    parser.add_argument("--min-face-size", type=int, default=60, help="Minimum face size in pixels.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)

    labels = load_labels(args.labels)
    model = load_emotion_model(args.model)
    image = read_image(image_path)
    detector = create_face_detector(args.detector, min_face_size=args.min_face_size)
    detection = detector.detect_largest(image, padding=args.padding)
    if not detection.face_detected:
        raise RuntimeError(
            "No face detected. Try another detector, a clearer frontal image, better lighting, "
            "or infer_image_custom_cnn.py with a manually cropped face."
        )

    face_crop = crop_box(image, detection)
    prediction = predict_face(model, face_crop, labels, top_k=args.top_k)

    annotated = draw_detection(image, detection)
    annotated = draw_prediction_panel(annotated, prediction)

    result = {
        "mode": "raw_image_with_face_detector",
        "image_path": str(image_path),
        "model_path": str(Path(args.model)),
        "detection": detection.to_dict(),
        "face_box": {"x": detection.x, "y": detection.y, "w": detection.w, "h": detection.h},
        **prediction,
    }

    write_image(args.output_image, annotated)
    write_json(args.output_json, result)

    print(f"Image: {image_path}")
    print(f"Detector: {args.detector}")
    print(f"Face box: x={detection.x}, y={detection.y}, w={detection.w}, h={detection.h}")
    print(f"Detection time: {detection.detect_time_ms:.2f} ms")
    print(f"Prediction: {prediction['predicted_label']} ({prediction['confidence_percentage']:.2f}%)")
    print("Top predictions:")
    for item in prediction["top_predictions"]:
        print(f"- {item['label']}: {item['percentage']:.2f}%")
    print(f"Saved annotated image to: {args.output_image}")
    print(f"Saved JSON result to: {args.output_json}")


if __name__ == "__main__":
    main()
