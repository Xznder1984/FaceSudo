"""Match orchestrator: camera capture -> liveness stack -> encoding -> decision.

Every failure path here results in "not matched", which the sudo wrapper
turns into a plain password prompt. Nothing in this module ever denies
anything outright -- it just declines to auto-fill.

The heavy lifting is in `run_match_with_session`, which drives a `MatchSession`
abstraction so the terminal CLI and the PyQt6 guided window share one code
path. The session receives live preview frames, step prompts, and real-time
per-phase feedback (blink detected, head-turn swing) as they happen.
"""

from __future__ import annotations

import random
import sys
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import preprocessing
from .camera import Camera, CameraError
from .log import MatchLog
from .liveness import (
    LIVENESS_WIDTH,
    LayerResult,
    _yaw_of_shape,
    openness_score,
    run_liveness,
)


@dataclass
class MatchResult:
    ok: bool = False
    reason: str = "not attempted"
    distance: float | None = None
    best_index: int | None = None
    elapsed: float = 0.0
    liveness: list[LayerResult] = field(default_factory=list)
    engine: str = ""
    detector: str = ""

    def summary(self) -> str:
        parts = [f"elapsed={self.elapsed:.1f}s"]
        if self.distance is not None:
            parts.append(f"distance={self.distance:.3f}")
        for lr in self.liveness:
            parts.append(f"{lr.layer}={('pass' if lr.passed else 'fail')}")
        return " ".join(parts)


class MatchSession:
    """Sink for live match feedback. The terminal CliSession prints; the GUI
    session emits Qt signals. All methods may be called from the capture
    thread."""

    def prompt(self, message: str) -> None:
        pass

    def feedback(self, message: str) -> None:
        pass

    def preview(self, frame_bgr, box=None, yaw=None) -> None:
        pass

    def cancelled(self) -> bool:
        return False


class CliSession(MatchSession):
    def __init__(self, prompt_ui=None) -> None:
        self.ui = prompt_ui

    def prompt(self, message: str) -> None:
        if self.ui is not None:
            self.ui.show(message)
        else:
            print(f"FaceSudo: {message}", file=sys.stderr, flush=True)

    def feedback(self, message: str) -> None:
        print(f"  -> {message}", file=sys.stderr, flush=True)


class CancelFlag:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()


def _largest_box(engine, rgb) -> tuple | None:
    boxes = engine.detect(rgb)
    if not boxes:
        return None
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes[0]


def _small_bgr(frame, lowlight: bool, strength: float):
    """Enhance + downscale one frame to LIVENESS_WIDTH for fast analysis.
    Uses the CLAHE-only pass: the full denoise step is too slow to run on
    every live frame on a low-power CPU (the authoritative liveness check
    still uses the full pipeline)."""
    sm = preprocessing.enhance_quick(frame, lowlight, strength)
    h, w = sm.shape[:2]
    if w > LIVENESS_WIDTH:
        sm = cv2.resize(sm, (LIVENESS_WIDTH, int(h * LIVENESS_WIDTH / w)),
                        interpolation=cv2.INTER_AREA)
    return sm


def _annotate(frame_bgr, engine, predictor, lowlight: float, strength: float):
    """Downscaled detect + yaw + landmark shape for live overlay/feedback."""
    sm = _small_bgr(frame_bgr, lowlight, strength)
    rgb = cv2.cvtColor(sm, cv2.COLOR_BGR2RGB)
    box = _largest_box(engine, rgb)
    shape = _shape_of(rgb, box, predictor) if box is not None else None
    yaw = _yaw_of_shape(shape) if shape is not None else None
    return sm, box, yaw, shape


def _shape_of(rgb, box, predictor):
    import dlib

    top, right, bottom, left = box
    try:
        return predictor(rgb, dlib.rectangle(left, top, right, bottom))
    except Exception:
        return None


def _full_shape(frame_bgr, sm, box, predictor):
    """68-pt shape on the full-res frame, mapped from a small-frame box.

    Blink analysis needs the full-res landmarks: on a dim webcam the
    downscaled predictor returns a near-fixed eye template that never moves
    during a blink."""
    import dlib

    if box is None:
        return None
    top, right, bottom, left = box
    sx = frame_bgr.shape[1] / sm.shape[1]
    sy = frame_bgr.shape[0] / sm.shape[0]
    try:
        return predictor(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                         dlib.rectangle(int(left * sx), int(top * sy),
                                        int(right * sx), int(bottom * sy)))
    except Exception:
        return None


class _LiveBlinkCounter:
    """Streaming blink counter: EAR-dip around a running median baseline OR a
    transient landmark-detection gap (the landmark detector drops frames
    during a blink on dim webcams)."""

    def __init__(self, ratio: float, margin: float) -> None:
        self.ratio = ratio
        self.margin = margin
        self.records: list[tuple[bool, float | None]] = []
        self.state = "init"
        self.count = 0
        self._gap_start: int | None = None

    def feed(self, present: bool, score) -> int | None:
        self.records.append((present, score))
        if not present:
            if self._gap_start is None:
                self._gap_start = len(self.records) - 1
            elif len(self.records) - 1 - self._gap_start > 3:
                self.state = "init"  # face left; reset
            return None

        if self._gap_start is not None:
            gap = len(self.records) - 1 - self._gap_start
            self._gap_start = None
            if gap <= 3 and self.state in ("open", "closed"):
                self.count += 1
                self.state = "open"
                return self.count
        if score is None:
            return None
        valid = [s for p, s in self.records if p and s is not None]
        if len(valid) < 3:
            return None
        hi = sorted(valid)
        baseline = float(np.median(hi[len(hi) // 2:]))
        closed = max(baseline - self.margin, baseline * self.ratio)
        reopen = closed + (baseline - closed) * 0.5
        s = score
        if self.state == "init":
            if s >= closed:
                self.state = "open"
        elif self.state == "open":
            if s < closed:
                self.state = "closed"
        elif self.state == "closed":
            if s >= reopen:
                self.count += 1
                self.state = "open"
                return self.count
        return None

    @property
    def total(self) -> int:
        return self.count


def _capture_burst(cam, n: int, delay: float = 0.03,
                   deadline: float | None = None, on_frame=None, stop=None,
                   stop_min: int = 0) -> list:
    frames = []
    for _ in range(n):
        if deadline is not None and time.monotonic() > deadline:
            break
        frame = cam.read()
        if frame is None:
            break
        frames.append(frame)
        if on_frame is not None:
            on_frame(frame)
        if stop is not None and stop() and len(frames) >= stop_min:
            break
        if delay > 0:
            time.sleep(delay)
    return frames


def _phase_deadline(n: int, delay: float) -> float:
    """Per-phase capture window. Deliberately NOT capped by the overall
    timeout: the overall timeout only bounds the face hunt, while each phase
    always gets its capture time + grace so a slow CPU can't starve frames."""
    return time.monotonic() + n * delay + 2.0


def run_match_with_session(
    config,
    store,
    engine,
    predictor,
    session: MatchSession | None = None,
    timeout: float | None = None,
) -> MatchResult:
    session = session or CliSession()
    started = time.monotonic()
    deadline = started + (timeout if timeout is not None else config.timeout)

    if not store.count():
        return MatchResult(reason="no enrolled faces")

    cam = Camera(index=config.camera_index)
    try:
        cam.open()
    except CameraError as e:
        return MatchResult(reason=f"camera error: {e}")

    lowlight = config.lowlight
    strength = config.lowlight_strength
    result = MatchResult(engine="dlib", detector=engine.detector_name)

    try:
        # --- Face hunt: live preview until a face appears ---
        session.prompt("camera on - SIT UP and face the camera (full face in frame)")
        face_box = None
        while time.monotonic() < deadline:
            if session.cancelled():
                result.reason = "cancelled by user"
                return result
            burst = _capture_burst(cam, 5, delay=0.03, deadline=deadline)
            if not burst:
                result.reason = "no frames from camera"
                return result
            for frame in burst:
                sm, box, _, _ = _annotate(frame, engine, predictor, lowlight, strength)
                session.preview(sm, box)
                if box is not None:
                    face_box = box
                    break
            if face_box is not None:
                break
        if face_box is None:
            result.reason = "timeout waiting for a face"
            return result
        session.feedback("face found")

        # --- Phase A: blink / texture / micro-motion window (live feedback) ---
        session.prompt("Now blink TWICE slowly: open... close... open... close")
        counter = None

        def _on_a(frame):
            nonlocal counter
            sm, box, _, _ = _annotate(frame, engine, predictor, lowlight, strength)
            session.preview(sm, box)
            if counter is None:
                counter = _LiveBlinkCounter(0.8, 0.08)
            shape = _full_shape(frame, sm, box, predictor)
            present = shape is not None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = openness_score(shape, gray) if present else None
            new = counter.feed(present, score)
            if new:
                session.feedback(f"blink {new} detected")

        def _stop_a():
            return counter is not None and counter.total >= 2

        frames_a = _capture_burst(cam, 40, delay=0.02,
                                  deadline=time.monotonic() + 7.0,
                                  on_frame=_on_a, stop=_stop_a, stop_min=6)
        if session.cancelled():
            result.reason = "cancelled by user"
            return result
        if len(frames_a) < 3:
            result.reason = "camera dropped too many frames"
            return result
        session.feedback(f"blinks detected: {counter.total if counter else 0}")

        # --- Phase B: randomized head-turn prompt (live yaw feedback) ---
        turn_dir = random.choice(["left", "right"])
        expected = -1.0 if turn_dir == "left" else 1.0
        session.prompt(f"Now turn your head to the {turn_dir} and hold it")
        yaws: list[float] = []
        swing_seen = False

        def _on_b(frame):
            nonlocal swing_seen
            sm, box, yaw, _ = _annotate(frame, engine, predictor, lowlight, strength)
            session.preview(sm, box, yaw)
            if yaw is not None:
                yaws.append(yaw)
                if len(yaws) >= 4 and not swing_seen:
                    base = float(np.mean(yaws[:2]))
                    cur = float(np.mean(yaws[-2:]))
                    swing = cur - base
                    if abs(swing) > 0.10:
                        swing_seen = True
                        note = "good" if swing * expected > 0.0 else "opposite (fine)"
                        session.feedback(f"{note} - hold the turn")

        def _stop_b():
            return len(yaws) >= 4 and abs(float(np.mean(yaws[-2:]) - np.mean(yaws[:2]))) > 0.10

        frames_b = _capture_burst(cam, 26, delay=0.06,
                                  deadline=_phase_deadline(26, 0.06),
                                  on_frame=_on_b, stop=_stop_b, stop_min=6)
        if session.cancelled():
            result.reason = "cancelled by user"
            return result
        if len(frames_b) < 3:
            result.reason = "camera dropped too many frames"
            return result
        swing = float(np.mean(yaws[-2:]) - np.mean(yaws[:2])) if len(yaws) >= 4 else 0.0
        session.feedback(f"head turn {turn_dir}: swing {swing:+.2f}")

        # --- Recognition on the multi-frame average. Only the blink-window
        # frames are used: the user faces the camera throughout phase A,
        # whereas phase B frames include the (intentionally) turned face. ---
        frames = frames_a
        if len(frames) < 6:
            result.reason = "camera dropped too many frames"
            return result

        # --- Full liveness stack. Blink/texture/micro_motion run on the
        # blink window (frames_a); head_turn/parallax on the turn window
        # (frames_b). Splitting keeps the head-turn frames from skewing the
        # adaptive blink baseline. Each call passes explicit on/off for every
        # layer so run_liveness never enables a layer by default.
        ALL_LAYERS = ("blink", "head_turn", "texture", "micro_motion", "parallax")
        layer_cfg = {k: config.liveness_enabled(k) for k in ALL_LAYERS}

        def _subset(active: set[str]) -> dict:
            return {k: (layer_cfg[k] if k in active else False) for k in ALL_LAYERS}

        ok_a, res_a = run_liveness(
            _subset({"blink", "texture", "micro_motion"}),
            frames_a, engine, predictor, prompt_dir="",
            lowlight=lowlight, strength=strength)
        ok_b, res_b = run_liveness(
            _subset({"head_turn", "parallax"}),
            frames_b, engine, predictor, prompt_dir=turn_dir,
            lowlight=lowlight, strength=strength)
        liveness_results = res_a + res_b
        liveness_ok = ok_a and ok_b
        result.liveness = liveness_results
        for lr in liveness_results:
            mark = "PASS" if lr.passed else "FAIL"
            session.feedback(f"{lr.layer}: {mark} ({lr.detail})")
        if not liveness_ok:
            failed = [lr.layer for lr in liveness_results if not lr.passed]
            result.reason = "liveness check failed: " + ", ".join(failed)
            return result

        # --- Recognition on the multi-frame average ---
        session.prompt("matching...")
        avg = preprocessing.enhance(frames, lowlight, strength)
        rgb = cv2.cvtColor(avg, cv2.COLOR_BGR2RGB)
        box = _largest_box(engine, rgb)
        if box is None:
            result.reason = "no face found on averaged frame"
            return result

        encs = engine.encode(rgb, [box])
        if not encs:
            result.reason = "encoding failed"
            return result
        encoding, _ = encs[0]

        known = store.encodings()
        best = engine.best_distance(encoding, known)
        if best is None:
            result.reason = "no enrolled encodings"
            return result
        result.distance, result.best_index = best

        if result.distance <= config.threshold:
            result.ok = True
            result.reason = f"matched (distance {result.distance:.3f})"
        else:
            result.reason = f"no match (distance {result.distance:.3f} > {config.threshold:.2f})"
        return result
    finally:
        cam.release()
        result.elapsed = time.monotonic() - started
        MatchLog().append(result.ok, result.reason, result.summary())


def run_match(
    config,
    store,
    engine,
    predictor,
    prompt=None,
    timeout: float | None = None,
) -> MatchResult:
    """Terminal-compatible wrapper kept for the CLI verify path."""
    return run_match_with_session(config, store, engine, predictor,
                                  CliSession(prompt), timeout)
