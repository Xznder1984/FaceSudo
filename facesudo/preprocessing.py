"""Low-light / mixed-light preprocessing pipeline.

Order of operations:
  1. Multi-frame averaging to suppress photon noise from camera gain.
  2. Bilateral / fastNLM denoising on the averaged frame.
  3. CLAHE on the L* channel (LAB) for adaptive contrast equalization --
     handles half-lit rooms far better than plain histogram equalization.

Everything is applied identically at enrollment and at match time so the
embedding space stays consistent.
"""

from __future__ import annotations

import cv2
import numpy as np


def average_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Mean of frames as float32, clipped back to uint8."""
    if not frames:
        raise ValueError("no frames to average")
    stack = np.stack(frames, axis=0).astype(np.float32)
    return np.clip(np.mean(stack, axis=0), 0, 255).astype(np.uint8)


def denoise(frame: np.ndarray, h: float = 8.0) -> np.ndarray:
    """Edge-preserving denoise. Uses fastNLM on luminance; keeps chroma."
    Small h keeps skin texture while killing gain noise.
    """
    if h <= 0.01:
        return frame
    try:
        return cv2.fastNlMeansDenoisingColored(frame, None, h, h, 7, 21)
    except cv2.error:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L = cv2.fastNlMeansDenoising(L, None, h, 7, 21)
        return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)


def clahe_color(frame: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE on the L* channel of LAB. Handles uneven/mixed lighting."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)


def enhance_single(frame: np.ndarray, enabled: bool, strength: float = 1.0) -> np.ndarray:
    """Denoise + CLAHE on a single frame (no averaging)."""
    if not enabled or strength <= 0.0:
        return frame
    strength = max(0.1, min(2.0, strength))
    clip = 1.2 + 0.8 * strength
    h = 5.0 + 3.0 * strength
    return clahe_color(denoise(frame, h=h), clip_limit=clip, tile=8)


def enhance(frames: list[np.ndarray], enabled: bool, strength: float = 1.0) -> np.ndarray:
    """Full low-light pipeline. `strength` scales CLAHE clip and denoise h.

    Returns the processed BGR frame ready for detection + encoding.
    """
    avg = average_frames(frames)
    if not enabled or strength <= 0.0:
        return avg

    strength = max(0.1, min(2.0, strength))
    clip = 1.2 + 0.8 * strength          # 2.0 at strength 1.0
    h = 5.0 + 3.0 * strength             # 8.0 at strength 1.0
    denoised = denoise(avg, h=h)
    return clahe_color(denoised, clip_limit=clip, tile=8)
