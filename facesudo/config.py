"""Configuration storage for FaceSudo.

Lives at ~/.facesudo/config.json. Contains no secrets -- the sudo password
and the SQLite encryption key live in the macOS Keychain only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".facesudo"
CONFIG_PATH = CONFIG_DIR / "config.json"
DB_PATH = CONFIG_DIR / "facesudo.db"
LOG_PATH = CONFIG_DIR / "facesudo.db"  # match log shares the same SQLite file
MODELS_DIR = CONFIG_DIR / "models"

DEFAULTS = {
    "enabled": True,
    "threshold": 0.55,          # face_distance threshold; lower = stricter
    "timeout": 10,              # seconds allowed for a match attempt
    "lowlight": True,           # run the low-light enhancement pipeline
    "lowlight_strength": 1.0,   # 0.0 - 2.0 multiplier for CLAHE/denoise params
    "camera_index": 0,
    "ir_camera": False,         # set True if using an external IR webcam
    "liveness": {
        "blink": True,
        "head_turn": True,
        "texture": True,
        "micro_motion": True,
        "parallax": True,
    },
}

# Set of nested keys so we can whitelist user edits.
_LIVENESS_KEYS = set(DEFAULTS["liveness"].keys())


class Config:
    def __init__(self) -> None:
        self._data: dict = {}
        self.load()

    @property
    def enabled(self) -> bool:
        return bool(self._data.get("enabled", True))

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._data["enabled"] = bool(value)
        self.save()

    @property
    def threshold(self) -> float:
        return float(self._data.get("threshold", DEFAULTS["threshold"]))

    @property
    def timeout(self) -> int:
        return int(self._data.get("timeout", DEFAULTS["timeout"]))

    @property
    def lowlight(self) -> bool:
        return bool(self._data.get("lowlight", DEFAULTS["lowlight"]))

    @property
    def lowlight_strength(self) -> float:
        return float(self._data.get("lowlight_strength", DEFAULTS["lowlight_strength"]))

    @property
    def camera_index(self) -> int:
        return int(self._data.get("camera_index", DEFAULTS["camera_index"]))

    @property
    def ir_camera(self) -> bool:
        return bool(self._data.get("ir_camera", DEFAULTS["ir_camera"]))

    def liveness_enabled(self, layer: str) -> bool:
        layers = self._data.get("liveness", {})
        return bool(layers.get(layer, DEFAULTS["liveness"].get(layer, True)))

    def load(self) -> None:
        self._data = json.loads(json.dumps(DEFAULTS))
        try:
            raw = CONFIG_PATH.read_text()
            loaded = json.loads(raw)
            for k, v in loaded.items():
                if k == "liveness" and isinstance(v, dict):
                    merged = dict(DEFAULTS["liveness"])
                    merged.update({lk: lv for lk, lv in v.items() if lk in _LIVENESS_KEYS})
                    self._data["liveness"] = merged
                elif k in DEFAULTS:
                    self._data[k] = v
        except FileNotFoundError:
            self.save()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        os.replace(tmp, CONFIG_PATH)

    def update(self, **kwargs) -> None:
        """Whitelisted key updates. Nested liveness keys accepted as liveness_blink=..."""
        for k, v in kwargs.items():
            if k == "enabled":
                self._data["enabled"] = bool(v)
            elif k == "threshold":
                self._data["threshold"] = float(v)
            elif k == "timeout":
                self._data["timeout"] = int(v)
            elif k == "lowlight":
                self._data["lowlight"] = bool(v)
            elif k == "lowlight_strength":
                self._data["lowlight_strength"] = max(0.0, min(2.0, float(v)))
            elif k == "camera_index":
                self._data["camera_index"] = int(v)
            elif k == "ir_camera":
                self._data["ir_camera"] = bool(v)
            elif k.startswith("liveness_"):
                layer = k[len("liveness_"):]
                if layer in _LIVENESS_KEYS:
                    self._data["liveness"][layer] = bool(v)
            else:
                continue  # ignore unknown keys
        self.save()
