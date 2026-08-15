"""Phase 1 smoke test: engine, detection, encoding, matching, and the
liveness stack against a single static face image (dev-time artifact only).
"""

import sys
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

from facesudo.recognition import FaceEngine
from facesudo.runtime import build_predictor
from facesudo import preprocessing
from facesudo.liveness import run_liveness
from facesudo.config import Config

IMG = "/tmp/fstest_face.jpg"


def main() -> int:
    cfg = Config()
    engine = FaceEngine()
    predictor = build_predictor()
    print(f"engine: dlib  | detector: {engine.detector_name} | predictor parts: 68-model active")

    bgr = cv2.imread(IMG)
    if bgr is None:
        print("could not read test image")
        return 1
    print(f"image: {bgr.shape[1]}x{bgr.shape[0]}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # --- detection ---
    boxes = engine.detect(rgb)
    print(f"detect -> {len(boxes)} face(s)")
    if not boxes:
        print("FAIL: no face detected on a clean frontal photo")
        return 1

    # --- encoding + self-match ---
    encs = engine.encode(rgb, boxes)
    print(f"encode -> {len(encs)} embedding(s), dim {encs[0][0].shape[0]}")
    best = engine.best_distance(encs[0][0], [encs[0][0].tolist()])
    print(f"self-distance: {best[0]:.4f} (expect ~0)")
    if best[0] > 0.05:
        print("FAIL: self-distance unexpectedly high")
        return 1

    # --- low-light pipeline on a dimmed copy ---
    dim = np.clip(bgr.astype(np.float32) * 0.25, 0, 255).astype(np.uint8)
    enhanced = preprocessing.enhance([dim, dim, dim, dim], True, 1.0)
    dim_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    boxes_dim = engine.detect(dim_rgb)
    print(f"dim-light (25%) detect after CLAHE+avg -> {len(boxes_dim)} face(s)")
    if boxes_dim:
        enc_dim = engine.encode(dim_rgb, boxes_dim)
        if enc_dim:
            d = engine.best_distance(enc_dim[0][0], [encs[0][0].tolist()])
            print(f"dim-vs-bright distance: {d[0]:.4f} (same person should be < 0.55)")
    else:
        print("note: dim image had no face detected even after enhancement (expected in hard cases)")

    # --- liveness stack against the static image (repeated frames) ---
    frames = [dim, dim, dim, dim, dim]
    liveness_cfg = {k: True for k in ("blink", "head_turn", "texture", "micro_motion", "parallax")}
    ok, results = run_liveness(liveness_cfg, frames, engine, predictor,
                               prompt_dir="left", lowlight=True, strength=1.0)
    print("liveness results (static image, no motion):")
    for r in results:
        print(f"  {r.layer:12s} {'PASS' if r.passed else 'FAIL'}  {r.detail}")
    print(f"all_liveness_pass={ok}  (expected False: a frozen frame has no blink/motion)")

    print("\nSMOKE TEST DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
