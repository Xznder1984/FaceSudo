"""macOS Keychain access for FaceSudo via the `keyring` package.

Two secret slots, both stored in the macOS login Keychain (auditable in
Keychain Access under the "com.facesudo" service):

  - sudo password  (username "sudo-password")
  - SQLite encryption key (username "encryption-key", a Fernet key)

The sudo password is written here from the GUI and read ONLY inside the
verified-match code path (see sudo_bridge.py).
"""

from __future__ import annotations

import keyring

SERVICE = "com.facesudo"

_SUDO_USER = "sudo-password"
_KEY_USER = "encryption-key"


def _keyring() -> keyring.backends.Backend:
    # keyring.resolve returns the active backend (macOS Keychain on macOS).
    return keyring


def set_sudo_password(password: str) -> None:
    keyring.set_password(SERVICE, _SUDO_USER, password)


def get_sudo_password() -> str | None:
    value = keyring.get_password(SERVICE, _SUDO_USER)
    if not value:
        return None
    return value


def clear_sudo_password() -> None:
    try:
        keyring.delete_password(SERVICE, _SUDO_USER)
    except keyring.errors.PasswordDeleteError:
        pass


def has_sudo_password() -> bool:
    return get_sudo_password() is not None


def get_encryption_key() -> bytes | None:
    value = keyring.get_password(SERVICE, _KEY_USER)
    if not value:
        return None
    return value.encode("utf-8")


def set_encryption_key(key_bytes: bytes) -> None:
    keyring.set_password(SERVICE, _KEY_USER, key_bytes.decode("utf-8"))
