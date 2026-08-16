import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import dlib

from facesudo.liveness import is_68_point
from facesudo.runtime import build_engine, build_predictor

OUT = "/tmp/blinkprobe"


def detect_small(frame, eng):
    sm = cv2.resize(frame, (320, int(frame.shape[0] * 320 / frame.shape[1])))
    rgb = cv2.cvtColor(sm, cv2.COLOR_BGR2RGB)
    boxes = eng.detect(rgb)
    if not boxes:
        return None
    sx = frame.shape[1] / sm.shape[1]
    sy = frame.shape[0] / sm.shape[0]
    t, r, b, l = boxes[0]
    return (int(l * sx), int(t * sy), int(r * sx), int(b * sy))


eng = build_engine()
predictor = build_predictor()

for n in (12, 29, 0, 39):
    f = cv2.imread(os.path.join(OUT, f"f{n:04d}.png"))
    box = detect_small(f, eng)
    if box is None:
        continue
    l, t, r, b = box
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    sh = predictor(rgb, dlib.rectangle(l, t, r, b))
    print(f"--- f{n:04d} box=({l},{t},{r},{b}) face_w={r-l} is68={is_68_point(sh)}")
    if is_68_point(sh):
        eye = [(sh.part(i).x, sh.part(i).y) for i in range(36, 48)]
        xs = [p[0] for p in eye]
        ys = [p[1] for p in eye]
        print(f"    eyes: x[{min(xs)},{max(xs)}] y[{min(ys)},{max(ys)}] (frame 640x480)")
        print(f"    left eye pts: {eye[:6]}")
        print(f"    right eye pts: {eye[6:]}")
        for i in range(68):
            cv2.circle(f, (sh.part(i).x, sh.part(i).y), 2, (0, 255, 0), -1)
        cv2.rectangle(f, (l, t), (r, b), (0, 0, 255), 2)
        # eye crops
        for idx, idxs in enumerate((range(36, 42), range(42, 48))):
            pts = [(sh.part(i).x, sh.part(i).y) for i in idxs]
            xs2 = [p[0] for p in pts]
            ys2 = [p[1] for p in pts]
            x0, x1 = max(0, min(xs2) - 20), min(640, max(xs2) + 20)
            y0, y1 = max(0, min(ys2) - 15), min(480, max(ys2) + 15)
            crop = cv2.resize(f[y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"/tmp/fs_eye_{n}_{idx}.png", crop)
    cv2.imwrite(f"/tmp/fs_landmarks_{n}.png", f)
    print("    saved /tmp/fs_landmarks_%d.png + eye crops" % n)
