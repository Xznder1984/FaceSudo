"""Layered spoof/liveness detection.

Layers (each independently toggleable in the GUI):

  blink         - eye-openness tracked over frames; requires an open->closed->open cycle
  head_turn     - randomized left/right prompt; requires a yaw swing in response
  texture       - skin vs printed-photo vs screen texture/periodicity analysis
  micro_motion  - internal vs rigid optical-flow signature inside the face box
  parallax      - background-vs-face displacement cue during head motion (best-effort)

Policy: an enabled layer must return PASS. N/A (e.g. the cue couldn't be
computed) counts as PASS so a flaky heuristic never locks the user out --
the whole matcher always falls back to the password prompt anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import dlib
import numpy as np

from .config import MODELS_DIR
from .preprocessing import enhance_single

_68POINT_PATH = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"

# 5-point model indices (bundled with face_recognition)
P_LEFT_EYE = 0
P_RIGHT_EYE = 1
P_NOSE = 2
P_MOUTH_LEFT = 3
P_MOUTH_RIGHT = 4

_EYE68_LEFT = list(range(36, 42))
_EYE68_RIGHT = list(range(42, 48))


@dataclass
class LayerResult:
    layer: str
    passed: bool
    detail: str = ""


@dataclass
class LivenessContext:
    frames_enhanced: list[np.ndarray] = field(default_factory=list)  # BGR
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)  # per frame
    prompt_dir: str = ""  # "left" | "right" | ""
    landmarks: list = field(default_factory=list)  # per frame shape or None


def _bbox_center(box) -> tuple[float, float]:
    top, right, bottom, left = box
    return ((left + right) / 2.0, (top + bottom) / 2.0)


def _dist(p, q) -> float:
    return float(np.hypot(p.x - q.x, p.y - q.y))


def build_landmark_sets(frames_rgb, boxes, predictor):
    """Run the shape predictor per frame where a box exists."""
    shapes = []
    for rgb, box in zip(frames_rgb, boxes):
        if box is None:
            shapes.append(None)
            continue
        top, right, bottom, left = box
        rect = dlib.rectangle(left, top, right, bottom)
        try:
            shapes.append(predictor(rgb, rect))
        except Exception:
            shapes.append(None)
    return shapes


# ---------------------------------------------------------------------------
# Blink detection
# ---------------------------------------------------------------------------

def _eye_openness_5pt(gray: np.ndarray, shape, box) -> float | None:
    le, re = shape.part(P_LEFT_EYE), shape.part(P_RIGHT_EYE)
    length = _dist(le, re)
    if length < 4:
        return None
    mx, my = (le.x + re.x) / 2.0, (le.y + re.y) / 2.0
    half = int(0.55 * length)
    h, w = gray.shape
    x0, y0 = max(0, int(mx - half)), max(0, int(my - half))
    x1, y1 = min(w, int(mx + half)), min(h, int(my + half))
    patch = gray[y0:y1, x0:x1]
    if patch.size < 16:
        return None
    mean = float(patch.mean())
    if mean < 1.0:
        return None
    darkness = float(np.mean(patch < (mean * 0.72)))
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)
    gvar_n = float((np.abs(gx) + np.abs(gy)).mean()) / (mean + 1.0)
    return 0.6 * min(1.0, darkness * 4.0) + 0.4 * min(1.0, gvar_n / 3.0)


def _ear_68(shape) -> float:
    def ear(pts):
        p = [shape.part(i) for i in pts]
        a = _dist(p[1], p[5])
        b = _dist(p[2], p[4])
        c = _dist(p[0], p[3])
        return (a + b) / (2.0 * c + 1e-6)

    return (ear(_EYE68_LEFT) + ear(_EYE68_RIGHT)) / 2.0


def check_blink(ctx: LivenessContext, predictor, use_68: bool) -> LayerResult:
    scores = []
    for i, (frame, shape) in enumerate(zip(ctx.frames_enhanced, ctx.landmarks)):
        if shape is None:
            continue
        if use_68:
            scores.append(_ear_68(shape))
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            s = _eye_openness_5pt(gray, shape, ctx.boxes[i])
            if s is not None:
                scores.append(s)
    if len(scores) < 4:
        return LayerResult("blink", False, f"insufficient frames tracked ({len(scores)})")

    if use_68:
        lo, hi = 0.22, 0.26
        low, high = lo, hi
    else:
        low, high = 0.30, 0.48

    blinks = 0
    state = "init"
    for s in scores:
        if state == "init":
            if s >= high:
                state = "open"
        elif state == "open":
            if s <= low:
                state = "closed"
        elif state == "closed":
            if s >= high:
                blinks += 1
                state = "open"
    ok = blinks >= 1
    detail = f"blinks={blinks} min={min(scores):.2f} max={max(scores):.2f}"
    return LayerResult("blink", ok, detail)


# ---------------------------------------------------------------------------
# Head turn detection
# ---------------------------------------------------------------------------

def _yaw(shape) -> float:
    nose, le, re = shape.part(P_NOSE), shape.part(P_LEFT_EYE), shape.part(P_RIGHT_EYE)
    dl, dr = _dist(nose, le), _dist(nose, re)
    return (dl - dr) / (dl + dr + 1e-6)


def check_head_turn(ctx: LivenessContext) -> LayerResult:
    yaws = []
    for shape in ctx.landmarks:
        if shape is not None:
            yaws.append(_yaw(shape))
    if len(yaws) < 6:
        return LayerResult("head_turn", False, f"insufficient frames ({len(yaws)})")

    baseline = float(np.mean(yaws[:3]))
    peak = max(abs(y - baseline) for y in yaws)
    swing = float(np.mean(yaws[-3:]) - baseline)

    # A rigidly held photo / frozen frame keeps yaw ~ baseline -> fails here.
    moved = peak > 0.10
    if not moved:
        return LayerResult("head_turn", False, f"no yaw response (peak {peak:.3f})")

    if ctx.prompt_dir in ("left", "right"):
        expected = -1.0 if ctx.prompt_dir == "left" else 1.0
        matches_dir = swing * expected > 0.06
        if matches_dir:
            detail = f"swing={swing:+.3f} peak={peak:.3f} dir={'matched'}"
        else:
            detail = f"swing={swing:+.3f} peak={peak:.3f} dir=opposite(accepted)"
        return LayerResult("head_turn", True, detail)
    return LayerResult("head_turn", moved, f"swing={swing:+.3f} peak={peak:.3f}")


# ---------------------------------------------------------------------------
# Texture analysis (skin vs photo vs screen)
# ---------------------------------------------------------------------------

def _face_patch_gray(frame_bgr, box) -> np.ndarray | None:
    top, right, bottom, left = box
    h, w = frame_bgr.shape[:2]
    top, right, bottom, left = max(0, top), min(w, right), min(h, bottom), max(0, left)
    if right - left < 24 or bottom - top < 24:
        return None
    return cv2.cvtColor(frame_bgr[top:bottom, left:right], cv2.COLOR_BGR2GRAY)


def _fft_scores(g: np.ndarray):
    g = cv2.resize(g, (160, 160))
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log1p(np.abs(f))
    h, w = g.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    low_e = float(mag[r2 <= 8 ** 2].sum())
    band_e = float(mag[(r2 > 8 ** 2) & (r2 <= 28 ** 2)].sum())
    high_e = float(mag[r2 > 28 ** 2].sum())
    ratio = band_e / (low_e + 1e-6)
    # periodicity: strength of strongest off-center spectral peak vs local mean
    outer = mag[(r2 > 6 ** 2)]
    peak = float(outer.max()) if outer.size else 0.0
    periodicity = peak / (float(outer.mean()) + 1e-6)
    lap_var = float(cv2.Laplacian(g, cv2.CV_64F).var())
    return {"ratio": ratio, "periodicity": periodicity, "lap_var": lap_var, "mean": float(mag.mean())}


def check_texture(ctx: LivenessContext) -> LayerResult:
    # use the largest face of the middle frame
    mid = len(ctx.frames_enhanced) // 2
    box = ctx.boxes[mid] if mid < len(ctx.boxes) else None
    if box is None:
        return LayerResult("texture", False, "no face to analyze")
    patch = _face_patch_gray(ctx.frames_enhanced[mid], box)
    if patch is None:
        return LayerResult("texture", False, "face too small")
    s = _fft_scores(patch)

    smooth = s["lap_var"] < 18.0      # printed photo: low micro-texture
    screeny = s["periodicity"] > 3.2  # LCD/OLED raster: strong periodic peak

    if smooth:
        return LayerResult("texture", False,
                           f"too smooth (lap_var={s['lap_var']:.0f} periodic={s['periodicity']:.2f})")
    if screeny:
        return LayerResult("texture", False,
                           f"screen-like periodic texture (periodic={s['periodicity']:.2f})")
    return LayerResult("texture", True,
                       f"lap_var={s['lap_var']:.0f} periodic={s['periodicity']:.2f} band={s['ratio']:.1f}")


# ---------------------------------------------------------------------------
# Micro-motion analysis
# ---------------------------------------------------------------------------

def check_micro_motion(ctx: LivenessContext) -> LayerResult:
    """Distinguishes a rigidly-held print (uniform flow) or frozen frame
    from a live face (eyes/lips move independently of the head)."""
    if len(ctx.frames_enhanced) < 3:
        return LayerResult("micro_motion", False, "too few frames")

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in ctx.frames_enhanced]
    internal_stds, global_means = [], []
    any_motion = 0.0
    for i in range(1, len(grays)):
        box = ctx.boxes[i - 1] or ctx.boxes[i]
        if box is None:
            continue
        flow = cv2.calcOpticalFlowFarneback(
            grays[i - 1], grays[i], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = cv2.magnitude(flow[..., 0], flow[..., 1])
        top, right, bottom, left = box
        h, w = mag.shape
        top, right, bottom, left = max(0, top), min(w, right), min(h, bottom), max(0, left)
        if right - left < 10 or bottom - top < 10:
            continue
        internal = mag[top:bottom, left:right]
        mask = np.ones(mag.shape, bool)
        mask[top:bottom, left:right] = False
        background = mag[mask]
        int_mean = float(internal.mean())
        int_std = float(internal.std())
        glob_mean = float(background.mean()) if background.size else int_mean
        internal_stds.append(int_std / (glob_mean + 1e-3))
        global_means.append(glob_mean)
        any_motion = max(any_motion, int_mean, glob_mean)

    if not internal_stds:
        return LayerResult("micro_motion", False, "no flow computed")

    mean_ratio = float(np.mean(internal_stds))
    # 1) completely frozen frame -> reject (nothing alive)
    if any_motion < 0.05:
        return LayerResult("micro_motion", False, f"no motion detected ({any_motion:.3f})")
    # 2) rigid uniform motion (photo) -> internal flow mirrors global -> low variance ratio
    if mean_ratio < 0.35 and any_motion > 0.15:
        return LayerResult("micro_motion", False, f"rigid/uniform motion (ratio={mean_ratio:.2f})")
    return LayerResult("micro_motion", True, f"flow ratio={mean_ratio:.2f} motion={any_motion:.2f}")


# ---------------------------------------------------------------------------
# Parallax / depth cue (best-effort)
# ---------------------------------------------------------------------------

def check_parallax(ctx: LivenessContext, prompt_active: bool) -> LayerResult:
    """During a head turn, a real face shifts relative to the background
    (parallax); a rigidly moved print drags the background along with it."""
    if not prompt_active or len(ctx.frames_enhanced) < 6:
        return LayerResult("parallax", True, "n/a (no motion window)")

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in ctx.frames_enhanced]
    face_pts = []
    bg_pts = []
    bg_prev = None
    prev_gray = None

    for i, gray in enumerate(grays):
        box = ctx.boxes[i]
        if box is None:
            continue
        top, right, bottom, left = box
        h, w = gray.shape
        top, right, bottom, left = max(0, top), min(w, right), min(h, bottom), max(0, left)
        face_pts.append(_bbox_center((top, right, bottom, left)))
        if prev_gray is not None and i > 0:
            mask = np.ones(gray.shape, np.uint8) * 255
            mask[top:bottom, left:right] = 0
            feats = cv2.goodFeaturesToTrack(prev_gray, maxCorners=12, qualityLevel=0.01,
                                            minDistance=12, mask=mask)
            if feats is not None and bg_prev is not None:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, bg_prev.reshape(-1, 1, 2), None)
                if nxt is not None and status is not None:
                    good = (status.flatten() == 1)
                    if good.any():
                        d = (nxt[good] - bg_prev[good]).reshape(-1, 2)
                        bg_pts.append(float(np.linalg.norm(d.mean(axis=0))))
            bg_prev = feats
        prev_gray = gray

    if len(face_pts) < 3:
        return LayerResult("parallax", True, "n/a (face not tracked)")
    if not bg_pts:
        return LayerResult("parallax", True, "n/a (no background features)")

    face_disp = float(np.linalg.norm(np.asarray(face_pts[-1]) - np.asarray(face_pts[0])))
    bg_disp = float(np.mean(bg_pts))
    if face_disp < 3.0:
        return LayerResult("parallax", True, "n/a (no face displacement)")

    # Rigid print: background moves along with the print -> similar magnitudes.
    ratio = bg_disp / (face_disp + 1e-6)
    rigid = ratio > 0.85 and bg_disp > 4.0
    if rigid:
        return LayerResult("parallax", False,
                           f"rigid motion bg={bg_disp:.1f} face={face_disp:.1f} ratio={ratio:.2f}")
    return LayerResult("parallax", True,
                       f"parallax bg={bg_disp:.1f} face={face_disp:.1f} ratio={ratio:.2f}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_liveness(
    layers_cfg: dict,
    frames_raw: list[np.ndarray],
    engine,
    predictor,
    prompt_dir: str = "",
    lowlight: bool = True,
    strength: float = 1.0,
) -> tuple[bool, list[LayerResult]]:
    """Run all enabled liveness layers. Returns (all_pass, results)."""
    enhanced = [enhance_single(f, lowlight, strength) for f in frames_raw]
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in enhanced]

    boxes: list[tuple | None] = []
    for rgb in rgb_frames:
        det = engine.detect(rgb)
        if det:
            # largest face
            det.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            boxes.append(det[0])
        else:
            boxes.append(None)

    use_68 = _68POINT_PATH.exists()
    ctx = LivenessContext(frames_enhanced=enhanced, boxes=boxes, prompt_dir=prompt_dir)
    ctx.landmarks = build_landmark_sets(rgb_frames, boxes, predictor)

    results: list[LayerResult] = []

    if layers_cfg.get("blink", True):
        results.append(check_blink(ctx, predictor, use_68))
    if layers_cfg.get("head_turn", True):
        results.append(check_head_turn(ctx))
    if layers_cfg.get("texture", True):
        results.append(check_texture(ctx))
    if layers_cfg.get("micro_motion", True):
        results.append(check_micro_motion(ctx))
    if layers_cfg.get("parallax", True):
        results.append(check_parallax(ctx, prompt_dir in ("left", "right")))

    all_pass = all(r.passed for r in results)
    return all_pass, results
