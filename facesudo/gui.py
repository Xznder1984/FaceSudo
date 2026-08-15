"""FaceSudo GUI (PyQt6).

Two windows:
  - SettingsWindow: enable/disable, confidence, timeout, liveness toggles,
    low-light controls, Keychain-backed password field, re-enrollment, log view.
  - EnrollmentWindow: live preview with guided angle prompts.

Framework choice: PyQt6 over customtkinter because Qt renders with the native
macOS Aqua widgets (system buttons, text fields, focus ring, HiDPI) instead of
custom-drawn canvas widgets, which gives a far more native look on macOS.
"""

from __future__ import annotations

import sys

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import keychain
from .config import Config
from .log import MatchLog
from .storage import EncodingStore

LIVENESS_LABELS = [
    ("blink", "Blink detection"),
    ("head_turn", "Head-turn prompt"),
    ("texture", "Texture analysis (photo/screen)"),
    ("micro_motion", "Micro-motion analysis"),
    ("parallax", "Parallax depth cue"),
]


def _bgr_to_pixmap(frame) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def _engine_brief() -> str:
    from .recognition import engine_status

    return engine_status()


# ---------------------------------------------------------------------------
# Log window
# ---------------------------------------------------------------------------

class LogWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FaceSudo - match log")
        self.resize(720, 400)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["time (UTC)", "outcome", "reason"])
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.table)
        btn = QPushButton("Refresh")
        btn.clicked.connect(self.refresh)
        layout.addWidget(btn)
        self.refresh()

    def refresh(self) -> None:
        rows = MatchLog().recent(200)
        self.table.setRowCount(len(rows))
        for i, r in enumerate(reversed(rows)):
            self.table.setItem(i, 0, QTableWidgetItem(r["ts"]))
            self.table.setItem(i, 1, QTableWidgetItem("OK" if r["success"] else "FAIL"))
            self.table.setItem(i, 2, QTableWidgetItem(r["reason"]))


# ---------------------------------------------------------------------------
# Settings window
# ---------------------------------------------------------------------------

class SettingsWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = Config()
        self.setWindowTitle("FaceSudo settings")
        self.resize(480, 620)

        central = QWidget()
        self.setCentralWidget(central)
        form = QFormLayout(central)

        self.enabled_cb = QCheckBox("Enable face-sudo in sudo calls")
        self.enabled_cb.setChecked(self.cfg.enabled)
        form.addRow("", self.enabled_cb)

        # confidence slider: distance 0.90..0.10, displayed as 10..90
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(int(round((1.0 - self.cfg.threshold) * 100)))
        self.conf_label = QLabel(self._conf_text(self.conf_slider.value()))
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(self._conf_text(v)))
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.conf_slider, 1)
        hl.addWidget(self.conf_label)
        form.addRow("Confidence", row)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(3, 60)
        self.timeout_spin.setValue(self.cfg.timeout)
        self.timeout_spin.setSuffix(" s")
        form.addRow("Timeout", self.timeout_spin)

        self.lowlight_cb = QCheckBox("Low-light preprocessing (CLAHE + denoise)")
        self.lowlight_cb.setChecked(self.cfg.lowlight)
        form.addRow("", self.lowlight_cb)

        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 200)
        self.strength_slider.setValue(int(self.cfg.lowlight_strength * 100))
        self.strength_label = QLabel(f"{self.cfg.lowlight_strength:.1f}")
        self.strength_slider.valueChanged.connect(
            lambda v: self.strength_label.setText(f"{v / 100:.1f}"))
        srow = QWidget()
        shl = QHBoxLayout(srow)
        shl.setContentsMargins(0, 0, 0, 0)
        shl.addWidget(self.strength_slider, 1)
        shl.addWidget(self.strength_label)
        form.addRow("Enhancement strength", srow)

        self.liveness_cbs = {}
        form.addRow("Spoof-detection layers", QLabel("(uncheck any that are too sensitive on your webcam)"))
        for key, label in LIVENESS_LABELS:
            cb = QCheckBox(label)
            cb.setChecked(self.cfg.liveness_enabled(key))
            self.liveness_cbs[key] = cb
            form.addRow("", cb)

        # password
        self.pw_field = QLineEdit()
        self.pw_field.setEchoMode(QLineEdit.EchoMode.Password)
        self.pw_field.setPlaceholderText("enter sudo password")
        self.pw_save = QPushButton("Save to Keychain")
        self.pw_save.clicked.connect(self._save_password)
        prow = QWidget()
        phl = QHBoxLayout(prow)
        phl.setContentsMargins(0, 0, 0, 0)
        phl.addWidget(self.pw_field, 1)
        phl.addWidget(self.pw_save)
        form.addRow("Sudo password", prow)

        self.pw_clear = QPushButton("Remove password from Keychain")
        self.pw_clear.clicked.connect(self._clear_password)
        form.addRow("", self.pw_clear)

        # actions
        self.enroll_btn = QPushButton("Enroll / Re-enroll face (guided)")
        self.enroll_btn.clicked.connect(self._enroll)
        form.addRow("", self.enroll_btn)

        self.log_btn = QPushButton("View match log")
        self.log_btn.clicked.connect(self._show_log)
        form.addRow("", self.log_btn)

        save_btn = QPushButton("Save settings")
        save_btn.clicked.connect(self._save)
        form.addRow("", save_btn)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        form.addRow(self.status_label)
        self._refresh_status()

    @staticmethod
    def _conf_text(v: int) -> str:
        return f"distance <= {1.0 - v / 100:.2f}"

    def _refresh_status(self) -> None:
        store = EncodingStore()
        pw = "stored" if keychain.has_sudo_password() else "NOT stored"
        self.status_label.setText(
            f"engine: {_engine_brief()}   |   enrolled: {store.count()} samples   |   "
            f"Keychain password: {pw}"
        )

    def _save_password(self) -> None:
        pw = self.pw_field.text()
        if not pw:
            self.status_label.setText("No password entered.")
            return
        keychain.set_sudo_password(pw)
        self.pw_field.clear()  # wiped immediately after the Keychain write
        self.status_label.setText("Sudo password saved to macOS Keychain.")
        self._refresh_status()

    def _clear_password(self) -> None:
        keychain.clear_sudo_password()
        self.status_label.setText("Sudo password removed from Keychain.")
        self._refresh_status()

    def _enroll(self) -> None:
        self.enroll_win = EnrollmentWindow(self.cfg)
        self.enroll_win.finished.connect(self._refresh_status)
        self.enroll_win.show()

    def _show_log(self) -> None:
        self.log_win = LogWindow()
        self.log_win.show()

    def _save(self) -> None:
        cfg = self.cfg
        cfg.update(
            enabled=self.enabled_cb.isChecked(),
            threshold=1.0 - self.conf_slider.value() / 100.0,
            timeout=self.timeout_spin.value(),
            lowlight=self.lowlight_cb.isChecked(),
            lowlight_strength=self.strength_slider.value() / 100.0,
            liveness_blink=self.liveness_cbs["blink"].isChecked(),
            liveness_head_turn=self.liveness_cbs["head_turn"].isChecked(),
            liveness_texture=self.liveness_cbs["texture"].isChecked(),
            liveness_micro_motion=self.liveness_cbs["micro_motion"].isChecked(),
            liveness_parallax=self.liveness_cbs["parallax"].isChecked(),
        )
        self.status_label.setText("Settings saved.")
        self._refresh_status()


# ---------------------------------------------------------------------------
# Enrollment window
# ---------------------------------------------------------------------------

class EnrollmentWindow(QDialog):
    ANGLES = [
        ("center", "face the camera directly"),
        ("left", "turn your head slightly to your LEFT"),
        ("right", "turn your head slightly to your RIGHT"),
        ("center", "face the camera directly"),
        ("up", "tilt your chin slightly UP"),
        ("center", "face the camera directly"),
        ("left", "turn your head slightly to your LEFT"),
        ("right", "turn your head slightly to your RIGHT"),
    ]

    def __init__(self, cfg: Config, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("FaceSudo enrollment")
        self.resize(680, 560)

        from .camera import Camera, CameraError

        self.camera = Camera(index=cfg.camera_index)
        try:
            self.camera.open()
        except CameraError as e:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel(f"Could not open camera: {e}"))
            self.camera = None
            return

        from .runtime import build_engine, build_predictor

        self.engine = build_engine()
        self.predictor = build_predictor()
        self.store = EncodingStore()

        layout = QVBoxLayout(self)
        self.preview = QLabel("starting camera...")
        self.preview.setMinimumSize(480, 360)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview, 1)

        self.instruction = QLabel("")
        self.instruction.setWordWrap(True)
        self.instruction.setStyleSheet("font-size: 15px; font-weight: bold;")
        layout.addWidget(self.instruction)

        self.progress = QLabel("")
        layout.addWidget(self.progress)

        self.capture_btn = QPushButton("Capture now")
        self.capture_btn.clicked.connect(self._manual_capture)
        layout.addWidget(self.capture_btn)

        self.close_btn = QPushButton("Finish / Cancel")
        self.close_btn.clicked.connect(self.close)
        layout.addWidget(self.close_btn)

        self.angle_idx = 0
        self.buffer = []
        self.captured = 0
        self.want = min(8, 8)
        self._update_instruction()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(60)

    def _update_instruction(self) -> None:
        if self.angle_idx < len(self.ANGLES):
            _, hint = self.ANGLES[self.angle_idx]
            self.instruction.setText(f"Step {self.angle_idx + 1}/{self.want}: {hint}")
        else:
            self.instruction.setText("All samples captured - you can close this window.")
        self.progress.setText(f"Captured {self.captured}/{self.want} samples")

    def _yaw_of(self, rgb, box) -> float | None:
        import dlib

        top, right, bottom, left = box
        rect = dlib.rectangle(left, top, right, bottom)
        try:
            shape = self.predictor(rgb, rect)
        except Exception:
            return None
        from .liveness import _yaw_of_shape

        return _yaw_of_shape(shape)

    def _tick(self) -> None:
        if self.camera is None:
            return
        frame = self.camera.read()
        if frame is None:
            return
        from . import preprocessing

        enhanced = preprocessing.enhance_single(
            frame, self.cfg.lowlight, self.cfg.lowlight_strength)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        box = None
        yaw = None
        try:
            boxes = self.engine.detect(rgb)
            if boxes:
                box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                yaw = self._yaw_of(rgb, box)
        except Exception:
            pass

        display = enhanced.copy()
        if box is not None:
            top, right, bottom, left = box
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
            if yaw is not None:
                cv2.putText(display, f"yaw {yaw:+.2f}", (left, top - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        self.preview.setPixmap(_bgr_to_pixmap(display).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

        if self.angle_idx >= len(self.ANGLES):
            return
        name, _ = self.ANGLES[self.angle_idx]
        if box is None:
            return
        satisfied = name == "up" or yaw is None or (
            name == "center" and abs(yaw) < 0.10) or (
            name == "left" and yaw < -0.08) or (
            name == "right" and yaw > 0.08)
        if satisfied:
            self.buffer.append(frame)
        else:
            self.buffer = []
        if len(self.buffer) >= 5:
            self._store_buffer()

    def _store_buffer(self) -> None:
        if not self.buffer:
            return
        from . import preprocessing

        avg = preprocessing.enhance(self.buffer, self.cfg.lowlight, self.cfg.lowlight_strength)
        rgb = cv2.cvtColor(avg, cv2.COLOR_BGR2RGB)
        encs = self.engine.encode(rgb)
        self.buffer = []
        if not encs:
            return
        enc = encs[0][0]
        self.store.add_sample("default", enc.tolist())
        self.captured += 1
        self.angle_idx += 1
        self._update_instruction()
        if self.angle_idx >= len(self.ANGLES):
            self.instruction.setText("Enrollment complete.")
            self.close_btn.setText("Close")

    def _manual_capture(self) -> None:
        frames = self.camera.grab_frames(5, delay=0.04)
        self.buffer = frames
        self._store_buffer()

    def closeEvent(self, event) -> None:
        if getattr(self, "timer", None) is not None:
            self.timer.stop()
        if getattr(self, "camera", None) is not None:
            self.camera.release()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    win = SettingsWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
