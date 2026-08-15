"""Camera capture with auto-exposure handling and IR-capability probing.

On macOS the built-in iSight/FaceTime camera is visible-light only (no IR).
An external IR webcam can be configured with config `ir_camera: true`; that
is an optional upgrade, not a requirement.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time

import cv2
import numpy as np

WARMUP_FRAMES = 8


class CameraError(RuntimeError):
    pass


@contextlib.contextmanager
def _quiet_cv():
    """OpenCV prints 'camera failed to initialize' noise to stderr when
    probing non-existent devices; silence it for those calls."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def probe_cameras(max_index: int = 4) -> list[dict]:
    """Return metadata about available capture devices."""
    results = []
    for i in range(max_index):
        with _quiet_cv():
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ok, frame = cap.read()
                results.append(
                    {
                        "index": i,
                        "ok": ok,
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if ok else 0,
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if ok else 0,
                    }
                )
            cap.release()
    return results


def detect_ir_capability(index: int = 0) -> bool:
    """Best-effort IR probe.

    macOS AVFoundation does not expose an IR flag, so this returns False for
    the built-in camera and documents the limitation. If a user selects an
    external IR webcam via config, they can set `ir_camera: true` manually.
    """
    try:
        from subprocess import run

        r = run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True)
        low = (r.stdout or "").lower()
        return "ir" in low and "infrared" in low
    except Exception:
        return False


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.cap: cv2.VideoCapture | None = None

    def open(self, timeout: float = 8.0) -> None:
        """Open the camera. Bounded by `timeout` so a busy camera (another app
        holding the webcam, macOS allows one consumer at a time) never hangs
        the sudo path -- it raises CameraError instead, which is fail-open."""
        box: dict = {}

        def _do_open() -> None:
            with _quiet_cv():
                cap = cv2.VideoCapture(self.index)
            box["cap"] = cap

        t = threading.Thread(target=_do_open, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive() or "cap" not in box:
            raise CameraError(
                f"camera index {self.index} did not open within {timeout}s "
                "(is another app using the webcam?)"
            )
        cap = box["cap"]
        if not cap.isOpened():
            raise CameraError(f"Could not open camera index {self.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self._tune_exposure(cap)
        self.cap = cap
        for _ in range(WARMUP_FRAMES):
            cap.read()  # let auto-exposure settle

    @staticmethod
    def _tune_exposure(cap) -> None:
        """Enable auto-exposure where supported; fall back to manual nudges."""
        try:
            # On AVFoundation: 0.25 auto, 0.75 manual (values vary by build).
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        except cv2.error:
            pass
        try:
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            cap.set(cv2.CAP_PROP_GAIN, 0)  # 0 = auto gain where supported
        except cv2.error:
            pass

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def grab_frames(self, n: int = 8, delay: float = 0.03) -> list[np.ndarray]:
        """Collect n frames spaced by `delay` seconds. First frame is dropped
        as part of the settle."""
        frames: list[np.ndarray] = []
        first = self.read()
        if first is None:
            return frames
        for _ in range(n):
            frame = self.read()
            if frame is None:
                break
            frames.append(frame)
            if delay > 0:
                time.sleep(delay)
        return frames

    def measure_brightness(self, frame: np.ndarray | None = None) -> float:
        if frame is None:
            frame = self.read()
            if frame is None:
                return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
