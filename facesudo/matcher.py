"""Match orchestrator: camera capture -> liveness stack -> encoding -> decision.

Every failure path here results in "not matched", which the sudo wrapper
turns into a plain password prompt. Nothing in this module ever denies
anything outright -- it just declines to auto-fill.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import cv2

from . import preprocessing
from .camera import Camera, CameraError
from .log import MatchLog
from .liveness import LayerResult, run_liveness

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


class PromptUI:
    """Sink for user-facing instructions. Swap for a GUI callback in the GUI."""

    def show(self, message: str) -> None:
        print(message, flush=True)


def _largest_box(engine, rgb) -> tuple | None:
    boxes = engine.detect(rgb)
    if not boxes:
        return None
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes[0]


def _capture_burst(cam, n: int, delay: float = 0.03, deadline: float | None = None) -> list:
    frames = []
    for _ in range(n):
        if deadline is not None and time.monotonic() > deadline:
            break
        frame = cam.read()
        if frame is None:
            break
        frames.append(frame)
        if delay > 0:
            time.sleep(delay)
    return frames


def run_match(
    config,
    store,
    engine,
    predictor,
    prompt: PromptUI | None = None,
    timeout: float | None = None,
) -> MatchResult:
    prompt = prompt or PromptUI()
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
        # --- Face-hunt: keep grabbing until a face appears or we run out of time
        while time.monotonic() < deadline:
            burst = _capture_burst(cam, 6, delay=0.04, deadline=deadline)
            if not burst:
                result.reason = "no frames from camera"
                return result
            sample_rgb = cv2.cvtColor(
                preprocessing.enhance_single(burst[len(burst) // 2], lowlight, strength),
                cv2.COLOR_BGR2RGB,
            )
            if _largest_box(engine, sample_rgb) is not None:
                face_found = True
                break
            prompt.show("FaceSudo: camera on - looking for your face...")
            face_found = False
        else:
            result.reason = "timeout waiting for a face"
            return result

        if not face_found:
            result.reason = "timeout waiting for a face"
            return result

        # --- Phase A: blink / texture / micro-motion window
        prompt.show("FaceSudo: blink naturally")
        frames_a = _capture_burst(cam, 10, delay=0.05, deadline=deadline)

        # --- Phase B: randomized head-turn prompt
        turn_dir = random.choice(["left", "right"])
        prompt.show(f"FaceSudo: turn your head to the {turn_dir}")
        frames_b = _capture_burst(cam, 10, delay=0.07, deadline=deadline)

        frames = frames_a + frames_b
        if len(frames) < 6:
            result.reason = "camera dropped too many frames"
            return result

        # --- Liveness stack over all captured frames
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
        if not liveness_ok:
            failed = [lr.layer for lr in liveness_results if not lr.passed]
            result.reason = "liveness check failed: " + ", ".join(failed)
            return result

        # --- Recognition on the multi-frame average
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
