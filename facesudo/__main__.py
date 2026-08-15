"""Entry point: `python -m facesudo ...` dispatches to CLI/GUI/menu."""

from __future__ import annotations

import sys


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "gui":
        from . import gui

        return gui.main()
    if argv and argv[0] == "menu":
        from . import menu

        menu.main()
        return 0
    if argv and argv[0] == "enroll-gui":
        from .gui import EnrollmentWindow
        from PyQt6.QtWidgets import QApplication
        from .config import Config

        app = QApplication(sys.argv)
        w = EnrollmentWindow(Config())
        w.show()
        return app.exec()
    from . import cli

    return cli.main()


if __name__ == "__main__":
    sys.exit(main())
