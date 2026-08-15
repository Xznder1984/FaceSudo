"""FaceSudo command line interface.

Commands:
  enroll            interactive face enrollment (CLI)
  verify            one-shot face match test (exit 0 on match)
  auth              used by the .zshrc wrapper: match then bridge sudo
  status            show configuration and enrollment state
  set-password      write the sudo password to Keychain (masked)
  clear-password    remove the sudo password from Keychain
  config            get/set config values
  list              list enrolled samples
  clear-enrollment  remove all enrolled samples
  log               show recent match attempts
  test-camera       camera / low-light diagnostics
  reset             wipe everything (encodings, log, optional Keychain key)
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import cv2

from . import keychain
from .config import Config
from .log import MatchLog
from .matcher import run_match
from .storage import EncodingStore
from .sudo_bridge import SUDO_BIN, bridge_sudo, exec_sudo_plain
from .recognition import engine_status


def _build():
    from .runtime import build_engine, build_predictor

    engine = build_engine()
    predictor = build_predictor()
    return engine, predictor


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

_ANGLE_PATHS = [
    ("center", "face the camera directly"),
    ("left", "turn your head slightly to your LEFT"),
    ("right", "turn your head slightly to your RIGHT"),
    ("center", "face the camera directly"),
    ("up", "tilt your chin slightly UP"),
    ("center", "face the camera directly"),
    ("left", "turn your head slightly to your LEFT"),
    ("right", "turn your head slightly to your RIGHT"),
]


def cmd_enroll(cfg: Config, args) -> int:
    from .camera import Camera, CameraError

    count = min(max(args.count, 4), 12)
    label = args.label
    store = EncodingStore()
    if args.clear_first:
        store.clear(label)
    engine, predictor = _build()
    import numpy as np
    import time

    cam = Camera(index=cfg.camera_index)
    try:
        cam.open()
    except CameraError as e:
        print(f"camera error: {e}", file=sys.stderr)
        return 1

    print(f"FaceSudo enrollment: {count} samples for label '{label}'")
    print("Bring a well-lit area; dim or mixed light is fine but faces the camera.\n")
    captured = 0
    previous = store.encodings()
    try:
        for angle, hint in _ANGLE_PATHS[:count]:
            print(f"\n[{captured + 1}/{count}] {hint}")
            for remaining in (3, 2, 1):
                print(f"  ...{remaining}", flush=True)
                time.sleep(0.9)
            frames = cam.grab_frames(6, delay=0.04)
            if len(frames) < 3:
                print("  capture failed, retry", file=sys.stderr)
                continue
            from . import preprocessing

            avg = preprocessing.enhance(frames, cfg.lowlight, cfg.lowlight_strength)
            rgb = cv2.cvtColor(avg, cv2.COLOR_BGR2RGB)
            encs = engine.encode(rgb)
            if not encs:
                print("  no face detected - retry")
                continue
            enc = encs[0][0]
            if previous:
                dists = cv2.norm if False else None
                import face_recognition

                d = float(min(face_recognition.face_distance(previous, enc)))
                if d > 0.70:
                    print(f"  warning: this sample is far from prior samples (d={d:.2f})")
            store.add_sample(label, enc.tolist())
            previous.append(enc.tolist())
            captured += 1
            print(f"  captured sample {captured} OK")
    finally:
        cam.release()

    print(f"\nEnrollment complete: {captured}/{count} samples stored.")
    return 0 if captured >= 4 else 1


# ---------------------------------------------------------------------------
# Verify / auth
# ---------------------------------------------------------------------------

def cmd_verify(cfg: Config, args) -> int:
    if engine_status().startswith("unavailable"):
        print("recognition engine unavailable (dlib did not import)", file=sys.stderr)
        return 2

    if args.gui:
        try:
            from .gui import run_match_dialog

            result = run_match_dialog(cfg)
            if result is not None:
                print(f"{'PASS' if result.ok else 'FAIL'}: {result.reason}")
                print(result.summary())
                return 0 if result.ok else 1
        except Exception as e:
            print(f"GUI match unavailable ({e}); falling back to terminal.", file=sys.stderr)

    engine, predictor = _build()
    store = EncodingStore()
    print(f"FaceSudo verify (engine={engine_status()}, detector={engine.detector_name})")
    result = run_match(cfg, store, engine, predictor, timeout=args.timeout)
    print(f"{'PASS' if result.ok else 'FAIL'}: {result.reason}")
    print(result.summary())
    return 0 if result.ok else 1


def cmd_auth(cfg: Config, args) -> int:
    """The .zshrc wrapper entry. Always ends by either bridging sudo with the
    Keychain password (matched) or exec'ing plain sudo (fallback)."""
    rest = args.sudo_args
    if rest and rest[0] == "--":
        rest = rest[1:]  # tolerate `facesudo-auth -- <args>`

    if not cfg.enabled:
        return exec_sudo_plain(rest)
    if not engine_status().startswith("dlib"):
        print("[FaceSudo] recognition engine unavailable; using password.", file=sys.stderr)
        return exec_sudo_plain(rest)
    if not keychain.has_sudo_password():
        print("[FaceSudo] no sudo password in Keychain; using password prompt.", file=sys.stderr)
        return exec_sudo_plain(rest)

    # Skip the camera when sudo's credential timestamp is already valid.
    check = subprocess.run([SUDO_BIN, "-n", "true"], capture_output=True)
    if check.returncode == 0:
        return exec_sudo_plain(rest)

    result = None
    if cfg.match_gui:
        try:
            from .gui import run_match_dialog

            result = run_match_dialog(cfg)
        except Exception as e:
            print(f"[FaceSudo] guided window unavailable ({e}); using terminal flow.",
                  file=sys.stderr)
        if result is None:
            print("[FaceSudo] match cancelled; password prompt.", file=sys.stderr)
            return exec_sudo_plain(rest)
    else:
        engine, predictor = _build()
        store = EncodingStore()
        result = run_match(cfg, store, engine, predictor)

    if result.ok:
        password = keychain.get_sudo_password()
        if not password:
            print("[FaceSudo] match OK but Keychain is empty; falling back.", file=sys.stderr)
            return exec_sudo_plain(rest)
        print(f"[FaceSudo] matched ({result.distance:.3f}); running sudo.", file=sys.stderr)
        return bridge_sudo(rest, password)
    print(f"[FaceSudo] not matched ({result.reason}); password prompt.", file=sys.stderr)
    return exec_sudo_plain(rest)


# ---------------------------------------------------------------------------
# Status / config
# ---------------------------------------------------------------------------

def cmd_status(cfg: Config, args) -> int:
    store = EncodingStore()
    print(f"engine:        {engine_status()}")
    print(f"enabled:       {cfg.enabled}")
    print(f"threshold:     {cfg.threshold:.2f} (distance)")
    print(f"timeout:       {cfg.timeout}s")
    print(f"lowlight:      {cfg.lowlight} (strength {cfg.lowlight_strength})")
    print(f"camera:        index {cfg.camera_index}{' (IR)' if cfg.ir_camera else ''}")
    print("liveness:")
    for k in ("blink", "head_turn", "texture", "micro_motion", "parallax"):
        print(f"  {k:14s} {'on' if cfg.liveness_enabled(k) else 'off'}")
    print(f"enrolled:      {store.count()} samples")
    print(f"sudo password: {'stored in Keychain' if keychain.has_sudo_password() else 'NOT stored'}")
    return 0


def cmd_config(cfg: Config, args) -> int:
    keys = ("enabled", "threshold", "timeout", "lowlight", "lowlight_strength",
            "camera_index", "ir_camera", "match_gui",
            "liveness_blink", "liveness_head_turn", "liveness_texture",
            "liveness_micro_motion", "liveness_parallax")
    if not args.set:
        for k in keys:
            if k.startswith("liveness_"):
                print(f"{k}={1 if cfg.liveness_enabled(k[9:]) else 0}")
            elif k == "enabled":
                print(f"enabled={int(cfg.enabled)}")
            elif k == "lowlight":
                print(f"lowlight={int(cfg.lowlight)}")
            elif k == "ir_camera":
                print(f"ir_camera={int(cfg.ir_camera)}")
            else:
                print(f"{k}={getattr(cfg, k)}")
        return 0
    updates = {}
    for part in args.set:
        if "=" not in part:
            print(f"bad pair: {part}", file=sys.stderr)
            return 1
        k, v = part.split("=", 1)
        if k not in keys:
            print(f"unknown key: {k}", file=sys.stderr)
            return 1
        if k in ("enabled", "ir_camera", "lowlight", "match_gui") or k.startswith("liveness_"):
            updates[k] = v in ("1", "true", "True", "yes", "on")
        elif k in ("threshold",):
            updates[k] = float(v)
        elif k in ("timeout", "camera_index"):
            updates[k] = int(v)
        elif k in ("lowlight_strength",):
            updates[k] = float(v)
    cfg.update(**updates)
    print("config updated")
    return 0


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def cmd_set_password(cfg: Config, args) -> int:
    import getpass

    pw = getpass.getpass("sudo password: ")
    if not pw:
        print("empty password; aborting", file=sys.stderr)
        return 1
    confirm = getpass.getpass("confirm: ")
    if pw != confirm:
        print("passwords do not match", file=sys.stderr)
        return 1
    keychain.set_sudo_password(pw)
    print("stored in macOS Keychain (service com.facesudo).")
    return 0


def cmd_clear_password(cfg: Config, args) -> int:
    keychain.clear_sudo_password()
    print("sudo password removed from Keychain.")
    return 0


# ---------------------------------------------------------------------------
# Enrollment admin
# ---------------------------------------------------------------------------

def cmd_list(cfg: Config, args) -> int:
    store = EncodingStore()
    for row in store.all():
        ok = "enc" if row["encoding"] else "BAD"
        print(f"{row['created_at']}  {row['label']:20s} {ok} {row['id'][:8]}")
    print(f"total: {store.count()} samples")
    return 0


def cmd_clear_enrollment(cfg: Config, args) -> int:
    store = EncodingStore()
    n = store.clear()
    print(f"removed {n} samples.")
    return 0


def cmd_log(cfg: Config, args) -> int:
    log = MatchLog()
    rows = log.recent(args.limit)
    for r in reversed(rows):
        mark = "OK " if r["success"] else "FAIL"
        print(f"{r['ts']}  {mark}  {r['reason']}")
    if not rows:
        print("no match attempts logged yet.")
    return 0


def cmd_test_camera(cfg: Config, args) -> int:
    from .camera import Camera, detect_ir_capability, probe_cameras

    print("devices:")
    for dev in probe_cameras():
        print(f"  index {dev['index']}: {dev['width']}x{dev['height']} {'OK' if dev['ok'] else 'no frame'}")
    print(f"IR capability (built-in): {'yes' if detect_ir_capability() else 'no - visible-light only'}")
    print("note: an external IR webcam is an optional upgrade that improves")
    print("      dim-light reliability. Set ir_camera=true if you add one.")
    cam = Camera(index=cfg.camera_index)
    try:
        cam.open()
    except Exception as e:
        print(f"camera {cfg.camera_index} failed: {e}", file=sys.stderr)
        return 1
    try:
        frames = cam.grab_frames(6)
        if not frames:
            print("no frames captured", file=sys.stderr)
            return 1
        from . import preprocessing

        print(f"raw mean brightness:  {cam.measure_brightness(frames[-1]):.1f}")
        print(f"avg (preprocess) brightness: {cam.measure_brightness(preprocessing.enhance(frames, True, 1.0)):.1f}")
        rgb = cv2.cvtColor(preprocessing.enhance(frames, cfg.lowlight, cfg.lowlight_strength),
                           cv2.COLOR_BGR2RGB)
        from .runtime import build_engine

        eng = build_engine()
        boxes = eng.detect(rgb)
        print(f"face detected: {len(boxes)} (detector={eng.detector_name})")
    finally:
        cam.release()
    return 0


def cmd_reset(cfg: Config, args) -> int:
    store = EncodingStore()
    n = store.clear()
    MatchLog().clear()
    print(f"removed {n} samples and cleared log.")
    if args.also_key:
        keychain.clear_sudo_password()
        print("Keychain entries removed (password + encryption key re-created on next use).")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="facesudo", description="Face-recognition sudo authentication")
    sub = p.add_subparsers(dest="cmd")

    e = sub.add_parser("enroll", help="interactive CLI enrollment")
    e.add_argument("--label", default="default")
    e.add_argument("--count", type=int, default=8)
    e.add_argument("--clear-first", action="store_true")
    e.set_defaults(fn=cmd_enroll)

    v = sub.add_parser("verify", help="one-shot match test")
    v.add_argument("--timeout", type=int, default=None)
    v.add_argument("--gui", action="store_true", help="use the guided camera window")
    v.set_defaults(fn=cmd_verify)

    a = sub.add_parser("auth", help="zsh wrapper entry (not for manual use)")
    a.add_argument("sudo_args", nargs=argparse.REMAINDER)
    a.set_defaults(fn=cmd_auth)

    sub.add_parser("status").set_defaults(fn=cmd_status)
    c = sub.add_parser("config", help="show or set config keys (key=value ...)")
    c.add_argument("set", nargs="*")
    c.set_defaults(fn=cmd_config)
    sub.add_parser("set-password").set_defaults(fn=cmd_set_password)
    sub.add_parser("clear-password").set_defaults(fn=cmd_clear_password)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    sub.add_parser("clear-enrollment").set_defaults(fn=cmd_clear_enrollment)
    lg = sub.add_parser("log")
    lg.add_argument("--limit", type=int, default=200)
    lg.set_defaults(fn=cmd_log)
    sub.add_parser("test-camera").set_defaults(fn=cmd_test_camera)
    r = sub.add_parser("reset")
    r.add_argument("--also-key", action="store_true")
    r.set_defaults(fn=cmd_reset)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = Config()
    if not args.cmd:
        parser.print_help()
        return 0
    return args.fn(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
