import argparse
import platform
from pathlib import Path

import cv2


def open_capture(index):
    if platform.system() == "Darwin":
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(index)


def probe_camera(index, output_dir):
    cap = open_capture(index)
    if not cap.isOpened():
        return {
            "index": index,
            "opened": False,
            "message": "not available",
        }

    frame = None
    ok = False
    for _ in range(20):
        ok, frame = cap.read()
        if ok and frame is not None:
            break

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not ok or frame is None:
        return {
            "index": index,
            "opened": True,
            "message": "opened but returned no frame",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / f"camera_{index}.jpg"
    cv2.imwrite(str(preview_path), frame)

    return {
        "index": index,
        "opened": True,
        "width": width,
        "height": height,
        "mean_brightness": round(float(frame.mean()), 2),
        "preview": str(preview_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Probe available OpenCV camera indexes.")
    parser.add_argument("--max-index", type=int, default=5, help="Highest camera index to test.")
    parser.add_argument(
        "--output-dir",
        default="reports/camera_previews",
        help="Directory for preview images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    print("Testing OpenCV camera indexes...")
    for index in range(args.max_index + 1):
        result = probe_camera(index, output_dir)
        if not result["opened"]:
            print(f"- source {index}: {result['message']}")
            continue

        if "preview" not in result:
            print(f"- source {index}: {result['message']}")
            continue

        print(
            f"- source {index}: {result['width']}x{result['height']}, "
            f"mean={result['mean_brightness']}, preview={result['preview']}"
        )

    print("Open the preview images and run main.py with the matching --source index.")


if __name__ == "__main__":
    main()
