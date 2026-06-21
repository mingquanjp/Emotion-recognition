import contextlib
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2


SUPPORTED_DETECTORS = ("haar", "mtcnn")


@dataclass
class FaceDetection:
    detector: str
    face_detected: bool
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    confidence: Optional[float] = None
    detect_time_ms: float = 0.0

    def to_dict(self):
        return {
            "detector": self.detector,
            "face_detected": self.face_detected,
            "x": int(self.x),
            "y": int(self.y),
            "w": int(self.w),
            "h": int(self.h),
            "confidence": None if self.confidence is None else float(self.confidence),
            "detect_time_ms": float(self.detect_time_ms),
        }


def expand_box(box, image_shape, ratio):
    x, y, w, h = [int(value) for value in box]
    image_h, image_w = image_shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = max(0, min(w, image_w - x))
    h = max(0, min(h, image_h - y))
    pad_x = int(w * ratio)
    pad_y = int(h * ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_w, x + w + pad_x)
    y2 = min(image_h, y + h + pad_y)
    return x1, y1, x2 - x1, y2 - y1


def crop_box(image_bgr, detection):
    return image_bgr[detection.y : detection.y + detection.h, detection.x : detection.x + detection.w]


class HaarFaceDetector:
    name = "haar"

    def __init__(self, min_face_size=60):
        self.min_face_size = min_face_size
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise RuntimeError(f"Cannot load OpenCV Haar cascade: {cascade_path}")

    def detect_largest(self, image_bgr, padding=0.18):
        started = time.perf_counter()
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_size = max(self.min_face_size, int(min(gray.shape[:2]) * 0.12))
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(min_size, min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        detect_time_ms = (time.perf_counter() - started) * 1000.0
        if len(faces) == 0:
            return FaceDetection(self.name, False, detect_time_ms=detect_time_ms)

        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = expand_box((x, y, w, h), image_bgr.shape, padding)
        return FaceDetection(self.name, True, x, y, w, h, None, detect_time_ms)


class MTCNNFaceDetector:
    name = "mtcnn"

    def __init__(self, min_face_size=60):
        from mtcnn import MTCNN

        self.detector = MTCNN(min_face_size=min_face_size)

    def detect_largest(self, image_bgr, padding=0.18):
        started = time.perf_counter()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with contextlib.redirect_stdout(io.StringIO()):
            faces = self.detector.detect_faces(image_rgb)
        detect_time_ms = (time.perf_counter() - started) * 1000.0
        if not faces:
            return FaceDetection(self.name, False, detect_time_ms=detect_time_ms)

        largest = max(faces, key=lambda face: max(0, face["box"][2]) * max(0, face["box"][3]))
        x, y, w, h = expand_box(largest["box"], image_bgr.shape, padding)
        return FaceDetection(
            self.name,
            True,
            x,
            y,
            w,
            h,
            float(largest.get("confidence", 0.0)),
            detect_time_ms,
        )


def create_face_detector(detector_name, min_face_size=60):
    detector_name = detector_name.lower()
    if detector_name == "haar":
        return HaarFaceDetector(min_face_size=min_face_size)
    if detector_name == "mtcnn":
        return MTCNNFaceDetector(min_face_size=min_face_size)
    raise ValueError(f"Unsupported detector '{detector_name}'. Choose one of: {', '.join(SUPPORTED_DETECTORS)}")


def draw_detection(image_bgr, detection, color=(0, 255, 255)):
    output = image_bgr.copy()
    if detection.face_detected:
        cv2.rectangle(
            output,
            (detection.x, detection.y),
            (detection.x + detection.w, detection.y + detection.h),
            color,
            2,
        )
        label = f"{detection.detector}"
        if detection.confidence is not None:
            label = f"{label}: {detection.confidence:.2f}"
        cv2.putText(
            output,
            label,
            (max(detection.x, 10), max(detection.y - 10, 26)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    else:
        cv2.putText(
            output,
            f"{detection.detector}: no face detected",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    return output
