"""Offline comparison of candidate blink signals on saved probe frames.
Checks which signal separates visually-confirmed closed-eye frames from
open-eye frames. Run: .venv/bin/python tools/offline_blink_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import dlib
import numpy as np

from facesudo.liveness import is_68_point
from facesudo.runtime import build_engine, build_predictor

OUT = "/tmp/blinkprobe"
N_FRAMES = 40

CLOSED = {0, 1, 16, 29, 39}
OPEN = {12}


def eye_boxes(shape, gray, pad=0):
    """Returns per-eye crop boxes from 68-pt landmarks."""
    if not is_68_point(shape):
        return []
    boxes = []
    for idxs in (range(36, 42), range(42, 48)):
        pts = [(shape.part(i).x, shape.part(i).y) for i in idxs]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = max(0, min(xs) - pad), min(gray.shape[1], max(xs) + pad)
        y0, y1 = max(0, min(ys) - pad), min(gray.shape[0], max(ys) + pad)
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    return boxes


def cand_signals(shape, gray, clahe_gray):
    vals = {}
    for box in eye_boxes(shape, gray, pad=2):
        x0, y0, x1, y1 = box
        raw = gray[y0:y1, x0:x1]
        cla = clahe_gray[y0:y1, x0:x1]
        if raw.size == 0:
            continue
        vals.setdefault("var_raw", []).append(float(raw.var()))
        vals.setdefault("var_clahe", []).append(float(cla.var()))
        edges = cv2.Canny(raw, 30, 90)
        vals.setdefault("edges_raw", []).append(int(edges.sum() / 255.0))
        edges_c = cv2.Canny(cla, 30, 90)
        vals.setdefault("edges_clahe", []).append(int(edges_c.sum() / 255.0))
        vals.setdefault("range_clahe", []).append(float(cla.max() - cla.min()))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in vals.items()}


def detect_small(frame):
    sm = cv2.resize(frame, (320, int(frame.shape[0] * 320 / frame.shape[1])))
    rgb = cv2.cvtColor(sm, cv2.COLOR_BGR2RGB)
    boxes = eng.detect(rgb)
    if not boxes:
        return None
    sx = frame.shape[1] / sm.shape[1]
    sy = frame.shape[0] / sm.shape[0]
    t, r, b, l = boxes[0]
    return (int(l * sx), int(t * sy), int(r * sx), int(b * sy))


def ear(shape, gray):
    if not is_68_point(shape):
        return None
    vals = []
    for idxs in (range(36, 42), range(42, 48)):
        pts = np.array([(shape.part(i).x, shape.part(i).y) for i in idxs], dtype=float)
        v = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3])
        vals.append(v / (2.0 * h))
    return float(np.mean(vals))


eng = build_engine()
predictor = build_predictor()

print("frame label ear_raw var_raw edges_raw var_clahe edges_clahe range_clahe")
for n in range(N_FRAMES):
    f = cv2.imread(os.path.join(OUT, f"f{n:04d}.png"))
    if f is None:
        continue
    box = detect_small(f)
    if box is None:
        print(n, "no face")
        continue
    l, t, r, b = box
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    try:
        sh = predictor(rgb, dlib.rectangle(l, t, r, b))
    except Exception:
        print(n, "lm fail")
        continue
    cla = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    s = cand_signals(sh, gray, cla)
    label = "C" if n in CLOSED else ("O" if n in OPEN else ".")
    print(f"{n} {label} {ear(sh, gray) and round(ear(sh, gray), 3)} "
          f"{round(s['var_raw'],1)} {round(s['edges_raw'],1)} "
          f"{round(s['var_clahe'],1)} {round(s['edges_clahe'],1)} "
          f"{round(s['range_clahe'],1)}")
