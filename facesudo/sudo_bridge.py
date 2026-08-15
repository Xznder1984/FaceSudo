"""PTY bridge that runs `sudo` and feeds the Keychain password to its prompt
only after a verified face match. The password never appears on argv, in logs,
or in any config file -- it is passed directly to the sudo child's terminal.

If sudo doesn't prompt (credential timestamp still valid) the password is
never written. If the password is wrong, the user can type the correct one
into the prompt and the bridge keeps working (fail open).
"""

from __future__ import annotations

import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios
import tty

SUDO_BIN = "/usr/bin/sudo"
PROMPT_TOKEN = "FaceSudo-Prompt:"


def _sync_winsize(master_fd: int) -> None:
    try:
        fd = sys.stdin.fileno()
        if os.isatty(fd):
            size = fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\x00" * 8)
            fcntl.ioctl(master_fd, termios.TIOCSWINSZ, size)
    except OSError:
        pass


def bridge_sudo(args: list[str], password: str) -> int:
    """Run `sudo args` in a pty, answering the password prompt once, then
    bridge the pty to the real terminal until sudo exits. Returns sudo's
    exit code."""
    pid, master = pty.fork()
    if pid == 0:
        os.execv(SUDO_BIN, [SUDO_BIN, "-p", PROMPT_TOKEN, *args])
        os._exit(127)

    _sync_winsize(master)

    saved = None
    try:
        fd_in = sys.stdin.fileno()
        if os.isatty(fd_in):
            saved = termios.tcgetattr(fd_in)
            tty.setcbreak(fd_in)
    except Exception:
        saved = None

    def _on_winch(_sig, _frame):
        _sync_winsize(master)

    old_winch = signal.signal(signal.SIGWINCH, _on_winch)

    answered = False
    status = 1
    try:
        buf = b""
        while True:
            try:
                r, _, _ = select.select([master, fd_in], [], [], 0.1)
            except (OSError, ValueError):
                break

            if master in r:
                try:
                    data = os.read(master, 4096)
                except OSError:
                    data = b""
                if not data:
                    break
                buf += data
                if not answered and PROMPT_TOKEN in buf.decode(errors="replace"):
                    answered = True
                    os.write(master, password.encode("utf-8") + b"\n")
                    buf = b""
                elif buf:
                    try:
                        os.write(sys.stdout.fileno(), buf)
                    except OSError:
                        pass
                    buf = b""

            if fd_in in r:
                try:
                    data = os.read(fd_in, 4096)
                except OSError:
                    data = b""
                if not data:
                    break
                try:
                    os.write(master, data)
                except OSError:
                    pass

        _, status = os.waitpid(pid, 0)
    finally:
        signal.signal(signal.SIGWINCH, old_winch)
        if saved is not None:
            try:
                termios.tcsetattr(fd_in, termios.TCSADRAIN, saved)
            except Exception:
                pass

    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return 1


def exec_sudo_plain(args: list[str]) -> int:
    """Replace this process with `sudo args` so the normal password prompt
    appears on the terminal. Used for every non-match / disabled path."""
    os.execv(SUDO_BIN, [SUDO_BIN, *args])
    return 127  # unreachable unless execv fails
