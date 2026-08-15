"""Face detection and encoding.

Detection: YuNet (OpenCV DNN) when the ONNX model is present, else dlib HOG.
Encoding: dlib ResNet-50 embedding via the `face_recognition` package
(bundles dlib_face_recognition_resnet_model_v1.dat, so no extra download).

If dlib/face_recognition failed to import, `engine_status()` reports it and
the app tells you to pivot to mediapipe rather than silently degrading.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import MODELS_DIR

ENGINE_DLIB = "dlib/face_recognition"
ENGINE_MEDIAPIPE = "mediapipe (fallback)"
ENGINE_NONE = "unavailable"

try:
    import face_recognition
    import dlib

    _HAS_DLIB = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_DLIB = False

YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"


def engine_status() -> str:
    if _HAS_DLIB:
        return ENGINE_DLIB
    return ENGINE_NONE


def engine_available() -> bool:
    return _HAS_DLIB


def _load_yunet() -> cv2.FaceDetectorYN | None:
    if not Path(YUNET_PATH).exists():
        return None
    try:
        net = cv2.FaceDetectorYN_create(
            str(YUNET_PATH), "", (320, 320), score_threshold=0.6, nms_threshold=0.4
        )
        return net
    except Exception:
        return None


class FaceEngine:
    """Face detection + embedding wrapper. Boxes are (top, right, bottom, left)."""

    def __init__(self) -> None:
        if not _HAS_DLIB:
            raise RuntimeError(
                "dlib/face_recognition is not importable on this machine. "
                "You need to pivot to the mediapipe fallback engine."
            )
        self._hog = dlib.get_frontal_face_detector()
        self._yunet = _load_yunet()
        self.detector_name = "yunet" if self._yunet is not None else "hog"

    def detect(self, rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return list of (top, right, bottom, left) boxes."""
        if self._yunet is not None:
            return self._detect_yunet(rgb)
        return self._detect_hog(rgb)

    def _detect_hog(self, rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=1, model="hog")
        return [tuple(map(int, loc)) for loc in locs]

    def _detect_yunet(self, rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = rgb.shape[:2]
        self._yunet.setInputSize((w, h))
        faces = self._yunet.detect(rgb)[1]
        boxes = []
        if faces is not None:
            for f in faces:
                x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                top = max(0, y)
                bottom = min(h, y + fh)
                left = max(0, x)
                right = min(w, x + fw)
                if right - left > 0 and bottom - top > 0:
                    boxes.append((top, right, bottom, left))
        return boxes

    def encode(
        self, rgb: np.ndarray, boxes: list[tuple[int, int, int, int]] | None = None
    ) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
        """Return [(encoding, box), ...]."""
        if boxes is None:
            boxes = self.detect(rgb)
        if not boxes:
            return []
        encs = face_recognition.face_encodings(rgb, known_face_locations=boxes)
        return [(np.asarray(e), b) for e, b in zip(encs, boxes)]

    def best_distance(
        self, encoding: np.ndarray, known: list[list[float]]
    ) -> tuple[float, int] | None:
        """Return (min_distance, index) against enrolled encodings, else None."""
        if not known:
            return None
        dists = face_recognition.face_distance(known, encoding)
        idx = int(np.argmin(dists))
        return float(dists[idx]), idx


def encode_face(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    """Convenience: encode a single detected face."""
    encs = face_recognition.face_encodings(rgb, known_face_locations=[box])
    return np.asarray(encs[0]) if encs else None
