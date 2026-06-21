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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run custom CNN emotion inference on a manually cropped face image."
    )
    parser.add_argument("--image", required=True, help="Path to a cropped face image.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to .keras model.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH), help="Path to labels.json.")
    parser.add_argument(
        "--output-image",
        default=str(DEFAULT_REPORTS_DIR / "manual_inference_result.png"),
        help="Path to save annotated result image.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_REPORTS_DIR / "manual_inference_result.json"),
        help="Path to save inference result JSON.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to print.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)

    labels = load_labels(args.labels)
    model = load_emotion_model(args.model)
    image = read_image(image_path)
    prediction = predict_face(model, image, labels, top_k=args.top_k)

    annotated = draw_prediction_panel(image, prediction)
    result = {
        "mode": "manual_crop",
        "image_path": str(image_path),
        "model_path": str(Path(args.model)),
        **prediction,
    }

    write_image(args.output_image, annotated)
    write_json(args.output_json, result)

    print(f"Image: {image_path}")
    print(f"Prediction: {prediction['predicted_label']} ({prediction['confidence_percentage']:.2f}%)")
    print("Top predictions:")
    for item in prediction["top_predictions"]:
        print(f"- {item['label']}: {item['percentage']:.2f}%")
    print(f"Saved annotated image to: {args.output_image}")
    print(f"Saved JSON result to: {args.output_json}")


if __name__ == "__main__":
    main()
