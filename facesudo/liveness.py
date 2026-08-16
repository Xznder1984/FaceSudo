"""Layered spoof/liveness detection.

Layers (each independently toggleable in the GUI):

  blink         - eye-openness tracked over frames; requires an open->closed->open cycle
  head_turn     - randomized left/right prompt; requires a yaw swing in response
  texture       - skin vs printed-photo vs screen texture/periodicity analysis
  micro_motion  - internal vs rigid optical-flow signature inside the face box
  parallax      - background-vs-face displacement cue during head motion (best-effort)

Design notes:
  - Detection / landmarks / optical flow run on frames downscaled to
    LIVENESS_WIDTH so the pipeline stays fast even on low-power Intel CPUs
    (e.g. the MacBook Air 2018 i5-8210Y). Texture uses the full-res patch.
  - Blink uses an adaptive baseline (median EAR/openness of the session)
    instead of fixed thresholds, because fixed thresholds fail across
    camera models, lighting, and face size.
  - Policy: an enabled layer must return PASS. N/A (cue couldn't be
    computed) counts as PASS so a flaky heuristic never locks the user out --
    the whole matcher always falls back to the password prompt anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import dlib
import numpy as np

from .preprocessing import enhance_quick

LIVENESS_WIDTH = 320

# 5-point model indices (bundled with face_recognition)
P_LEFT_EYE = 0
P_RIGHT_EYE = 1
P_NOSE = 2
P_MOUTH_LEFT = 3
P_MOUTH_RIGHT = 4

# 68-point model indices (bundled with face_recognition as
# shape_predictor_68_face_landmarks.dat, returned by pose_predictor_model_location)
E68_LEFT_EYE = 36   # left eye outer corner
E68_RIGHT_EYE = 45  # right eye outer corner
E68_NOSE_TIP = 30
_EYE68_LEFT = list(range(36, 42))
_EYE68_RIGHT = list(range(42, 48))


def is_68_point(shape) -> bool:
    try:
        return shape.num_parts == 68
    except Exception:
        return False


@dataclass
class LayerResult:
    layer: str
    passed: bool
    detail: str = ""


@dataclass
class LivenessContext:
    frames_small: list[np.ndarray] = field(default_factory=list)  # BGR downscaled
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)  # small coords
    prompt_dir: str = ""  # "left" | "right" | ""
    landmarks: list = field(default_factory=list)  # per frame shape or None
    full_res_frames: list[np.ndarray] = field(default_factory=list)  # for texture
    scale: float = 1.0


def _bbox_center(box) -> tuple[float, float]:
    top, right, bottom, left = box
    return ((left + right) / 2.0, (top + bottom) / 2.0)


def _dist(p, q) -> float:
    return float(np.hypot(p.x - q.x, p.y - q.y))


def _to_width(bgr: np.ndarray, width: int = LIVENESS_WIDTH) -> tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    if w <= width:
        return bgr, 1.0
    scale = width / w
    nh = int(round(h * scale))
    return cv2.resize(bgr, (width, nh), interpolation=cv2.INTER_AREA), scale


def _scale_box(box, scale: float) -> tuple[int, int, int, int]:
    top, right, bottom, left = box
    return (int(top / scale), int(right / scale), int(bottom / scale), int(left / scale))


def _largest(boxes):
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def _prep_frames(frames_raw: list[np.ndarray], lowlight: bool, strength: float):
    """Downscale + enhance each frame to LIVENESS_WIDTH once. Returns
    (smalls_bgr, scales). Detection/landmarks/flow run on these smalls.

    Uses the CLAHE-only pass: fastNLM denoise costs ~1.3 s/frame on a
    low-power Intel CPU and multi-frame liveness analysis doesn't need it.
    The recognition embedding still uses the full pipeline on the averaged
    frame (see preprocessing.enhance), keeping enrollment/match consistent.
    """
    smalls, scales = [], []
    for f in frames_raw:
        sm, sc = _to_width(f, LIVENESS_WIDTH)
        smalls.append(enhance_quick(sm, lowlight, strength))
        scales.append(sc)
    return smalls, scales


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
# Blink detection (adaptive)
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


def openness_score(shape, gray) -> float | None:
    """Single openness value: EAR for 68-point, darkness ratio for 5-point."""
    if is_68_point(shape):
        return _ear_68(shape)
    return _eye_openness_5pt(gray, shape, None)


def _detect_blinks(records: list[tuple[bool, float | None]], ratio: float,
                   margin: float) -> tuple[int, float]:
    """Adaptive open->closed->open cycle count.

    `records` is a per-frame list of (present, score) pairs, where `present`
    is whether landmarks were found. On a dim webcam the landmark detector
    routinely drops frames *during* a blink (closed eyes), so a blink is
    counted either by:
      - an EAR/openness dip below a relative baseline and recovery, or
      - a transient detection gap (face lost for <=3 frames, then recovered),
        which is exactly what a blink looks like to the detector.

    The closed threshold is relative to the running-median baseline so it
    still bites when the baseline is low (dim light, low-res landmarks):
    `closed = max(baseline - margin, baseline * ratio)`.
    """
    if not records:
        return 0, 0.0
    if not any(p for p, _ in records):
        return 0, 0.0
    valid = [s for p, s in records if p and s is not None]
    baseline = float(np.median(valid)) if valid else 0.0
    closed = max(baseline - margin, baseline * ratio)
    reopen = closed + (baseline - closed) * 0.5

    blinks = 0
    state = "init"
    i = 0
    n = len(records)
    while i < n:
        present, s = records[i]
        if not present:
            j = i
            while j < n and not records[j][0]:
                j += 1
            gap_len = j - i
            if state in ("open", "closed") and gap_len <= 3 and j < n and records[j][0]:
                blinks += 1
                state = "open"
            elif gap_len > 3:
                state = "init"  # face left the frame; reset
            i = j
            continue
        if s is None:
            i += 1
            continue
        if state == "init":
            if s >= closed:
                state = "open"
        elif state == "open":
            if s < closed:
                state = "closed"
        elif state == "closed":
            if s >= reopen:
                blinks += 1
                state = "open"
        i += 1
    return blinks, baseline


def check_blink(ctx: LivenessContext) -> LayerResult:
    records: list[tuple[bool, float | None]] = []
    uses_68 = False
    for i, (small, shape) in enumerate(zip(ctx.frames_small, ctx.landmarks)):
        if shape is None:
            records.append((False, None))
            continue
        if is_68_point(shape):
            uses_68 = True
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        s = openness_score(shape, gray)
        records.append((True, s))
    if len(records) < 4:
        return LayerResult("blink", False, f"insufficient frames tracked ({len(records)})")

    margin = 0.08 if uses_68 else 0.15
    ratio = 0.65 if uses_68 else 0.75
    blinks, baseline = _detect_blinks(records, ratio, margin)
    ok = blinks >= 1
    present = sum(1 for p, _ in records if p)
    valid = [s for p, s in records if p and s is not None]
    spread = (max(valid) - min(valid)) if len(valid) >= 2 else 0.0
    detail = (f"blinks={blinks} baseline={baseline:.2f} spread={spread:.2f} "
              f"tracked={present}/{len(records)}")
    return LayerResult("blink", ok, detail)


# ---------------------------------------------------------------------------
# Head turn detection
# ---------------------------------------------------------------------------

def _yaw_of_shape(shape) -> float:
    """View-relative yaw proxy: positive nose-to-left-eye / right-eye ratio
    means the head is turned to the subject's right. Works for both the
    5-point and 68-point landmark models."""
    if is_68_point(shape):
        nose, le, re = (shape.part(E68_NOSE_TIP), shape.part(E68_LEFT_EYE),
                        shape.part(E68_RIGHT_EYE))
    else:
        nose, le, re = (shape.part(P_NOSE), shape.part(P_LEFT_EYE),
                        shape.part(P_RIGHT_EYE))
    dl, dr = _dist(nose, le), _dist(nose, re)
    return (dl - dr) / (dl + dr + 1e-6)


def frame_yaw(rgb, box, predictor) -> float | None:
    shapes = build_landmark_sets([rgb], [box], predictor)
    if shapes[0] is None:
        return None
    return _yaw_of_shape(shapes[0])


def quick_yaw_swing(frames_raw: list[np.ndarray], engine, predictor,
                    lowlight: bool, strength: float) -> float:
    """Baseline-to-end yaw swing over frames (for live feedback)."""
    smalls, _ = _prep_frames(frames_raw, lowlight, strength)
    rgb_smalls = [cv2.cvtColor(s, cv2.COLOR_BGR2RGB) for s in smalls]
    boxes = [_largest(engine.detect(r)) for r in rgb_smalls]
    shapes = build_landmark_sets(rgb_smalls, boxes, predictor)
    yaws = [_yaw_of_shape(sh) for sh in shapes if sh is not None]
    if len(yaws) < 4:
        return 0.0
    baseline = float(np.mean(yaws[:2]))
    return float(np.mean(yaws[-2:]) - baseline)


def quick_blink_count(frames_raw: list[np.ndarray], engine, predictor,
                      lowlight: bool, strength: float) -> int:
    """Adaptive blink count over frames (for live feedback)."""
    smalls, _ = _prep_frames(frames_raw, lowlight, strength)
    rgb_smalls = [cv2.cvtColor(s, cv2.COLOR_BGR2RGB) for s in smalls]
    boxes = [_largest(engine.detect(r)) for r in rgb_smalls]
    shapes = build_landmark_sets(rgb_smalls, boxes, predictor)
    records: list[tuple[bool, float | None]] = []
    uses_68 = False
    for sm, shape in zip(smalls, shapes):
        if shape is None:
            records.append((False, None))
            continue
        if is_68_point(shape):
            uses_68 = True
        s = openness_score(shape, cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY))
        records.append((True, s))
    margin = 0.08 if uses_68 else 0.15
    ratio = 0.65 if uses_68 else 0.75
    blinks, _ = _detect_blinks(records, ratio, margin)
    return blinks


def check_head_turn(ctx: LivenessContext) -> LayerResult:
    yaws = []
    for shape in ctx.landmarks:
        if shape is not None:
            yaws.append(_yaw_of_shape(shape))
    if len(yaws) < 6:
        return LayerResult("head_turn", False, f"insufficient frames ({len(yaws)})")

    baseline = float(np.mean(yaws[:3]))
    peak = max(abs(y - baseline) for y in yaws)
    swing = float(np.mean(yaws[-3:]) - baseline)

    moved = peak > 0.055
    if not moved:
        return LayerResult("head_turn", False, f"no yaw response (peak {peak:.3f})")

    if ctx.prompt_dir in ("left", "right"):
        expected = -1.0 if ctx.prompt_dir == "left" else 1.0
        matches_dir = swing * expected > 0.05
        dir_note = "matched" if matches_dir else "opposite(accepted)"
        return LayerResult("head_turn", True, f"swing={swing:+.3f} peak={peak:.3f} dir={dir_note}")
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
    ratio = band_e / (low_e + 1e-6)
    outer = mag[(r2 > 6 ** 2)]
    peak = float(outer.max()) if outer.size else 0.0
    periodicity = peak / (float(outer.mean()) + 1e-6)
    lap_var = float(cv2.Laplacian(g, cv2.CV_64F).var())
    return {"ratio": ratio, "periodicity": periodicity, "lap_var": lap_var}


def check_texture(ctx: LivenessContext) -> LayerResult:
    """Skin vs print vs screen.

    On a small / dim / low-res webcam patch the "too smooth" cue is
    unreliable, so it is only trusted for a reasonably large face patch.
    A strong periodic peak (LCD/OLED raster) is the clearest screen cue.
    """
    mid = len(ctx.frames_small) // 2
    box_small = ctx.boxes[mid] if mid < len(ctx.boxes) else None
    if box_small is None:
        return LayerResult("texture", False, "no face to analyze")

    # Texture runs on the full-res patch for maximum detail. The orchestrator
    # stores just the enhanced middle frame here.
    if ctx.full_res_frames:
        box = _scale_box(box_small, ctx.scale)
        patch = _face_patch_gray(ctx.full_res_frames[0], box)
    else:
        patch = _face_patch_gray(ctx.frames_small[mid], box_small)
    if patch is None:
        return LayerResult("texture", True, "n/a (face too small)")

    pw = patch.shape[1]
    s = _fft_scores(patch)
    screeny = s["periodicity"] > 3.5
    smooth_large = s["lap_var"] < 10.0 and pw >= 90

    if screeny:
        return LayerResult("texture", False,
                           f"screen-like periodic texture (periodic={s['periodicity']:.2f})")
    if smooth_large:
        return LayerResult("texture", False,
                           f"too smooth (lap_var={s['lap_var']:.0f} w={pw})")
    return LayerResult("texture", True,
                       f"lap_var={s['lap_var']:.0f} periodic={s['periodicity']:.2f} w={pw}")


# ---------------------------------------------------------------------------
# Micro-motion analysis
# ---------------------------------------------------------------------------

def check_micro_motion(ctx: LivenessContext) -> LayerResult:
    if len(ctx.frames_small) < 3:
        return LayerResult("micro_motion", False, "too few frames")

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in ctx.frames_small]
    internal_stds = []
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
        any_motion = max(any_motion, int_mean, glob_mean)

    if not internal_stds:
        return LayerResult("micro_motion", False, "no flow computed")

    # Flow magnitudes are measured on the downscaled frames; scale the
    # thresholds so they behave like the original full-resolution values.
    scale = ctx.scale or 1.0
    no_motion = 0.05 * scale
    rigid_motion = 0.15 * scale
    mean_ratio = float(np.mean(internal_stds))
    if any_motion < no_motion:
        return LayerResult("micro_motion", False, f"no motion detected ({any_motion:.3f})")
    if mean_ratio < 0.35 and any_motion > rigid_motion:
        return LayerResult("micro_motion", False, f"rigid/uniform motion (ratio={mean_ratio:.2f})")
    return LayerResult("micro_motion", True, f"flow ratio={mean_ratio:.2f} motion={any_motion:.2f}")


# ---------------------------------------------------------------------------
# Parallax / depth cue (best-effort)
# ---------------------------------------------------------------------------

def check_parallax(ctx: LivenessContext, prompt_active: bool) -> LayerResult:
    if not prompt_active or len(ctx.frames_small) < 6:
        return LayerResult("parallax", True, "n/a (no motion window)")

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in ctx.frames_small]
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
    # Displacements are measured on the downscaled frames; scale thresholds
    # to behave like the original full-resolution values.
    scale = ctx.scale or 1.0
    if face_disp < 3.0 * scale:
        return LayerResult("parallax", True, "n/a (no face displacement)")

    ratio = bg_disp / (face_disp + 1e-6)
    rigid = ratio > 0.85 and bg_disp > 4.0 * scale
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
    smalls, scales = _prep_frames(frames_raw, lowlight, strength)
    rgb_smalls = [cv2.cvtColor(s, cv2.COLOR_BGR2RGB) for s in smalls]

    boxes: list[tuple | None] = [_largest(engine.detect(r)) for r in rgb_smalls]

    mid = len(frames_raw) // 2
    full_mid = enhance_quick(frames_raw[mid], lowlight, strength) if frames_raw else None

    ctx = LivenessContext(frames_small=smalls, boxes=boxes, prompt_dir=prompt_dir,
                          full_res_frames=[full_mid] if full_mid is not None else [],
                          scale=scales[mid] if scales else 1.0)
    ctx.landmarks = build_landmark_sets(rgb_smalls, boxes, predictor)

    results: list[LayerResult] = []
    if layers_cfg.get("blink", True):
        results.append(check_blink(ctx))
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
