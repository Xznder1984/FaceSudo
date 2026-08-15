"""Menu bar app (rumps) for quick enable/disable without opening the GUI."""

from __future__ import annotations

import subprocess
import sys

import rumps

from .config import Config


class FaceSudoMenu(rumps.App):
    def __init__(self) -> None:
        super().__init__("FaceSudo", quit_button=None)
        self.cfg = Config()
        self.item_enable = rumps.MenuItem("Enable FaceSudo", callback=self._toggle)
        self.item_enable.state = 1 if self.cfg.enabled else 0
        self.menu = [
            self.item_enable,
            None,
            rumps.MenuItem("Open Settings\u2026", callback=self._open_settings),
            rumps.MenuItem("Enroll Face\u2026", callback=self._open_enroll),
            None,
            rumps.MenuItem("Quit FaceSudo", callback=self._quit),
        ]

    def _toggle(self, sender) -> None:
        self.cfg.enabled = not self.cfg.enabled
        sender.state = 1 if self.cfg.enabled else 0

    def _run_gui(self, extra: list[str]) -> None:
        try:
            subprocess.Popen([sys.executable, "-m", "facesudo.gui", *extra])
        except Exception as e:
            print(f"failed to launch GUI: {e}", file=sys.stderr)

    def _open_settings(self, _) -> None:
        self._run_gui([])

    def _open_enroll(self, _) -> None:
        self._run_gui([])

    def _quit(self, _) -> None:
        rumps.quit_application()


def main() -> None:
    FaceSudoMenu().run()


if __name__ == "__main__":
    main()
