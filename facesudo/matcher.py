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
    is_68_point,
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
            print(f"FaceSudo: {message}", flush=True)

    def feedback(self, message: str) -> None:
        print(f"  -> {message}", flush=True)


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


class _LiveBlinkCounter:
    """Streaming open->closed->open counter around a running median baseline."""

    def __init__(self, margin: float) -> None:
        self.margin = margin
        self.scores: list[float] = []
        self.state = "init"
        self.count = 0

    def feed(self, score) -> int | None:
        if score is None:
            return None
        self.scores.append(float(score))
        if len(self.scores) < 3:
            return None
        baseline = float(np.median(self.scores))
        closed = baseline - self.margin
        reopen = baseline - self.margin * 0.4
        s = self.scores[-1]
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
                   deadline: float | None = None, on_frame=None) -> list:
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
        session.prompt("camera on - looking for your face...")
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
                _, box, _, _ = _annotate(frame, engine, predictor, lowlight, strength)
                session.preview(frame, box)
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
        session.prompt("Now blink naturally a few times")
        counter = None

        def _on_a(frame):
            nonlocal counter
            sm, box, _, shape = _annotate(frame, engine, predictor, lowlight, strength)
            session.preview(frame, box)
            if shape is None:
                return
            if counter is None:
                counter = _LiveBlinkCounter(0.08 if is_68_point(shape) else 0.15)
            gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
            new = counter.feed(openness_score(shape, gray))
            if new:
                session.feedback(f"blink {new} detected")

        frames_a = _capture_burst(cam, 8, delay=0.03,
                                  deadline=_phase_deadline(8, 0.03),
                                  on_frame=_on_a)
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
            session.preview(frame, box, yaw)
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

        frames_b = _capture_burst(cam, 8, delay=0.04,
                                  deadline=_phase_deadline(8, 0.04),
                                  on_frame=_on_b)
        if session.cancelled():
            result.reason = "cancelled by user"
            return result
        if len(frames_b) < 3:
            result.reason = "camera dropped too many frames"
            return result
        swing = float(np.mean(yaws[-2:]) - np.mean(yaws[:2])) if len(yaws) >= 4 else 0.0
        session.feedback(f"head turn {turn_dir}: swing {swing:+.2f}")

        frames = frames_a + frames_b
        if len(frames) < 6:
            result.reason = "camera dropped too many frames"
            return result

        # --- Full liveness stack over all captured frames ---
        liveness_ok, liveness_results = run_liveness(
            {
                k: config.liveness_enabled(k)
                for k in ("blink", "head_turn", "texture", "micro_motion", "parallax")
            },
            frames,
            engine,
            predictor,
            prompt_dir=turn_dir,
            lowlight=lowlight,
            strength=strength,
        )
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
