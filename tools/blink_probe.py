"""Timed blink capture. Eyes OPEN the whole time except 3 deliberate blinks
with ~1s pauses (user says "one", "two", "three" out loud as they blink).
Saves every frame + raw signals so blink windows can be compared offline.
Run: .venv/bin/python tools/blink_probe.py
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import dlib
import numpy as np

from facesudo.camera import Camera
from facesudo.liveness import is_68_point, openness_score
from facesudo.preprocessing import enhance_quick
from facesudo.runtime import build_engine, build_predictor

OUT = "/tmp/blinkprobe"


def eye_darkness(gray, shape):
    if not is_68_point(shape):
        return None
    pts = [(shape.part(i).x, shape.part(i).y) for i in range(36, 48)]
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
    vals = gray[mask > 0]
    return float(vals.mean()) if len(vals) else None


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))

    cam = Camera(index=0)
    cam.open()
    for _ in range(4):
        cam.read()
    eng = build_engine()
    predictor = build_predictor()

    print("Keep your EYES OPEN for the first 4 seconds.")
    print("Then blink 3 times, slowly, pausing ~1s between each (say 'one','two','three').")
    print("Then keep eyes open until it finishes (~12s total).")
    t0 = time.time()
    n = 0
    with open(os.path.join(OUT, "signals.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t", "ear320", "ear_full", "dark320", "var_eye"])
        while time.time() - t0 < 12.0:
            f = cam.read()
            if f is None:
                continue
            t = time.time() - t0
            sm = enhance_quick(f, True, 1.0)
            h, wd = sm.shape[:2]
            if wd > 320:
                sm = cv2.resize(sm, (320, int(h * 320 / wd)))
            rgb = cv2.cvtColor(sm, cv2.COLOR_BGR2RGB)
            boxes = eng.detect(rgb)
            if not boxes:
                w.writerow([n, round(t, 2), "", "", "", ""])
                n += 1
                cv2.imwrite(os.path.join(OUT, f"f{n:04d}.png"), f)
                time.sleep(0.05)
                continue
            top, right, bottom, left = boxes[0]
            try:
                shape = predictor(rgb, dlib.rectangle(left, top, right, bottom))
            except Exception:
                w.writerow([n, round(t, 2), "", "", "", ""])
                n += 1
                cv2.imwrite(os.path.join(OUT, f"f{n:04d}.png"), f)
                time.sleep(0.05)
                continue
            g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
            ear_small = openness_score(shape, g)
            dark = eye_darkness(g, shape)

            sx = f.shape[1] / sm.shape[1]
            sy = f.shape[0] / sm.shape[0]
            rgb_full = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            gf = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            try:
                shape_full = predictor(
                    rgb_full,
                    dlib.rectangle(int(left * sx), int(top * sy),
                                   int(right * sx), int(bottom * sy)))
                ear_full = openness_score(shape_full, gf)
                # eye-region variance on CLAHE-enhanced full-res face crop
                m = 30
                x0, y0 = max(0, int(left * sx) - m), max(0, int(top * sy) - m)
                x1, y1 = min(f.shape[1], int(right * sx) + m), min(f.shape[0], int(bottom * sy) + m)
                crop = f[y0:y1, x0:x1]
                cg = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
                cg_full = cv2.cvtColor(cg, cv2.COLOR_GRAY2RGB)
                sh_c = predictor(cg_full, dlib.rectangle(0, 0, cg.shape[1], cg.shape[0]))
                vals = []
                for idxs in (range(36, 42), range(42, 48)):
                    pts = np.array([(sh_c.part(i).x, sh_c.part(i).y) for i in idxs], dtype=float)
                    mx, my = pts.min(axis=0), pts.max(axis=0)
                    ex0, ey0 = max(0, int(mx[0]) - 2), max(0, int(my[0]) - 2)
                    ex1, ey1 = min(cg.shape[1], int(mx[1]) + 2), min(cg.shape[0], int(my[1]) + 2)
                    eye = cg[ey0:ey1, ex0:ex1]
                    if eye.size:
                        vals.append(float(eye.var()))
                var_eye = float(np.mean(vals)) if vals else ""
            except Exception:
                ear_full, var_eye = None, ""
            w.writerow([n, round(t, 2),
                        round(ear_small, 3) if ear_small is not None else "",
                        round(ear_full, 3) if ear_full is not None else "",
                        round(dark, 1) if dark is not None else "",
                        round(var_eye, 1) if var_eye != "" else ""])
            cv2.imwrite(os.path.join(OUT, f"f{n:04d}.png"), f)
            n += 1
            time.sleep(0.05)
    cam.release()
    print(f"done, {n} frames saved to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
