# FaceSudo

Authenticate `sudo` on an **Intel Mac** with **face recognition** instead of
typing your password. Everything runs locally — camera, matching, storage,
and logging. No cloud calls.

When you type `sudo <command>` in your terminal, FaceSudo activates your
camera, runs a layered anti-spoofing / liveness check against your enrolled
face, and — on a confident match — retrieves your sudo password from the
**macOS Keychain at that moment** and feeds it to the sudo prompt. You never
type the password.

On any failure — no face found, liveness check failed, low confidence,
timeout (10 s default), camera error — it **always falls back to the normal
password prompt**. It never locks you out, and it never fails closed.

## What this does and does not restrict

Read this. It matters.

- **What it does:** for a `sudo` that *you* invoke from your own interactive
  terminal (zsh), it *offers* the Keychain password automatically after a
  live-face match. It's a convenience layer over normal sudo, nothing more.
- **What it does NOT do:** it cannot and does not distinguish "a human typed
  this command in a terminal" from "another process invoked sudo". Any local
  process running as you can call sudo. FaceSudo is **not** a security
  boundary — if an attacker already has a shell on your machine, this doesn't
  stop them. It exists to save you keystrokes, not to be a sandbox.
- **It does not replace PAM.** macOS PAM is fragile and user-supplied PAM
  modules can brick password auth. This project deliberately uses a `.zshrc`
  wrapper instead and always fails open.
- **Photo/video spoofing** is mitigated but not eliminated. The liveness
  stack (below) makes a static photo or a screen-replay harder to use, but a
  determined attacker with good equipment and physical access can likely
  defeat consumer webcam liveness. Treat this as a convenience, not as
  two-factor authentication. Physical possession of your unlocked Mac is the
  real access.
- **First-party code path:** if your password is already cached by sudo
  (timestamp valid), FaceSudo skips the camera entirely.

## Password handling (auditable)

- The sudo password is stored **only** in the **macOS Keychain** via
  `keyring`, under service `com.facesudo` (username `sudo-password`). You can
  inspect or delete it any time in *Keychain Access*.
- It is **never** written to the SQLite database, any config file, or any
  other on-disk file.
- The GUI password field is masked, and is **cleared immediately** after the
  Keychain write succeeds.
- In-memory lifetime is limited to the single moment it's needed: it is read
  from the Keychain only inside the **matched-and-verified** code path
  (`facesudo auth` → `sudo_bridge.bridge_sudo`), never at app startup and
  never on idle.
- The password is passed to the sudo child process over a pty and never
  appears on `argv`, in logs, or in the shell history.

Face encodings live in `~/.facesudo/facesudo.db` (SQLite), **encrypted at
rest** with Fernet. The encryption key is also in the macOS Keychain
(username `encryption-key`) — it never touches disk.

## Architecture

```
facesudo/
  camera.py         capture, auto-exposure, IR-capability probe
  preprocessing.py  low-light pipeline: multi-frame averaging -> denoise -> CLAHE
  recognition.py    detection (YuNet DNN, falls back to dlib HOG) + dlib embeddings
  liveness.py       layered spoof detection
  matcher.py        orchestrates capture -> liveness -> recognition -> decision
  storage.py        encrypted SQLite encoding store (Fernet, key in Keychain)
  keychain.py       Keychain access (password + encryption key)
  log.py            local match-attempt log (timestamps + outcome only)
  sudo_bridge.py    pty bridge that answers the sudo prompt with the Keychain password
  cli.py            CLI: enroll / verify / auth / status / config / test-camera ...
  gui.py            PyQt6 settings + guided enrollment + guided match window
  menu.py           rumps menu-bar app (quick enable/disable)
  install.sh        venv + models + launchers + .zshrc wrapper
```

### Low-light / mixed-light handling

- **CLAHE** (adaptive histogram equalization) on the L\* channel — handles
  half-lit rooms far better than plain equalization.
- **Multi-frame averaging** to suppress gain noise in dim light.
- **Edge-preserving denoise** (fastNLM) before CLAHE.
- **Auto-exposure** via camera properties where the webcam supports it
  (AVFoundation).
- The built-in camera is **visible-light only**. FaceSudo probes for IR and
  reports it; if you often work in the dark, a cheap **external IR webcam**
  is an optional upgrade that meaningfully improves dim-light reliability
  (set `ir_camera=true` in config). It is not required.

### Spoof-detection layers (each independently toggleable)

| Layer | What it does |
|---|---|
| `blink` | tracks eye-openness over frames; requires an open→closed→open cycle |
| `head_turn` | randomized "turn left / turn right" prompt; requires a yaw swing |
| `texture` | skin vs printed-photo vs screen: micro-texture + FFT periodicity |
| `micro_motion` | internal vs rigid optical-flow signature inside the face box |
| `parallax` | background-vs-face displacement cue during head motion (best-effort) |

An enabled layer must **pass**. A layer that can't compute its cue (e.g. no
background features) counts as *N/A* → passes, so a flaky heuristic never
locks you out. If one layer is too sensitive for your webcam, disable it in
the GUI.

### Guided camera flow

When `match_gui` is enabled (default), `sudo` and `facesudo verify` open a
live camera window that walks you through each step in real time:

1. **Looking for your face** — live preview with a green box once detected.
2. **"Now blink naturally"** — counts each blink as it happens.
3. **"Turn your head to the left/right"** — live `yaw` readout on your face
   and "good, hold the turn" once the swing is seen.
4. **Matching** — per-layer PASS/FAIL shown, then the result.

The same flow runs in the terminal when `match_gui` is off or Qt is
unavailable; the sudo path falls back to the normal password prompt if the
window is cancelled, and `facesudo verify --gui` falls back to terminal
matching if Qt can't start. Liveness analysis always runs on a downscaled
320 px pipeline so the whole attempt stays fast even on a low-power Intel
CPU; texture analysis still uses the full-resolution frame.

## Requirements

- Intel Mac (x86_64), macOS 12+ (developed on macOS 14)
- Python 3.12 via Homebrew (`brew install python@3.12`)
- Xcode Command Line Tools (`xcode-select --install`)
- `cmake` (for the dlib C++ compile)

## Install

```bash
git clone https://github.com/Xznder1984/FaceSudo.git
cd FaceSudo
./install.sh
source ~/.zshrc          # or open a new terminal
```

`install.sh`:
1. creates a venv and installs dependencies (dlib compiles from source — this
   takes several minutes; if it fails you'll be told clearly, see below);
2. downloads the YuNet DNN detector (~230 KB);
3. writes launchers to `~/.facesudo/bin`;
4. appends the `sudo()` wrapper to `~/.zshrc`.

For stronger blink detection you can also fetch the 68-point landmark model:
`./install.sh --download-68` (~96 MB).

### Engine verification (Phase 1 gate)

This project requires **dlib / face_recognition** to compile cleanly on Intel
Mac. After install, check it with:

```bash
facesudo status
```

It reports `engine: dlib/face_recognition`. If instead it says
`engine: unavailable`, the dlib compile failed and FaceSudo **does not
silently degrade** — you must pivot to the mediapipe fallback engine (see
Troubleshooting). No face code runs against a broken engine.

## First-run setup

```bash
facesudo set-password     # stores your sudo password in the Keychain (masked, never on disk)
facesudo enroll           # guided CLI enrollment: 8 samples across angles/lighting
```

Or use the GUI for everything:

```bash
facesudo-settings
```

In the GUI you can also: enable/disable, adjust confidence and timeout,
toggle individual liveness layers, tune low-light strength, re-enroll, and
view the local match log.

Verify a match end-to-end:

```bash
facesudo verify           # PASS/FAIL with per-layer liveness detail
facesudo log              # recent attempts
facesudo test-camera      # camera + low-light diagnostics
```

## Try it

```bash
sudo whoami
```

Your camera will light up, you'll be asked to blink and turn your head
slightly, and on a match your password is fed to sudo automatically.
Otherwise you get the normal password prompt.

## Config

`~/.facesudo/config.json` (no secrets in it):

| key | default | meaning |
|---|---|---|
| `enabled` | `true` | master switch |
| `threshold` | `0.55` | max face_distance for a match (lower = stricter) |
| `timeout` | `10` | seconds allowed for a match attempt |
| `lowlight` | `true` | run CLAHE + denoise pipeline |
| `lowlight_strength` | `1.0` | 0–2 enhancement strength |
| `camera_index` | `0` | capture device |
| `ir_camera` | `false` | set `true` if using an external IR webcam |
| `match_gui` | `true` | use the guided camera window for sudo / verify |
| `liveness_*` | `true` | per-layer spoof-detection switches |

CLI: `facesudo config threshold=0.5 liveness_head_turn=0`

## Privacy

All processing is local. Match attempts are logged locally (timestamp,
outcome, layer results) with **no images and no recognized payloads**. The
match log never leaves your machine.

## Troubleshooting

- **dlib fails to compile.** Confirm Xcode CLT (`xcode-select --install`) and
  cmake, then `pip install --no-cache-dir dlib` and check the error. If it
  still won't build on your Intel Mac, the mediapipe fallback engine is the
  pivot: the recognition module reports `engine: unavailable` and the app
  tells you so instead of silently degrading. (A mediapipe-based embedding
  backend can be added behind the same `FaceEngine` interface.)
- **No face found in dim light.** Improve lighting or run
  `facesudo config lowlight_strength=1.5`. Consider an external IR webcam.
- **Liveness layer too sensitive.** Uncheck that layer in the GUI
  (`facesudo config liveness_texture=0`, etc.).
- **`sudo` still asks for the password.** Check `facesudo status` (enabled?),
  that Keychain has the password, that your face is enrolled, and try
  `facesudo verify`.
- **Wrong password stored.** `facesudo set-password` again (it writes
  directly to the Keychain on submit).

## Roadmap / phase status

- [x] Phase 1 — core recognition engine, low-light pipeline, layered liveness
- [x] Phase 2 — enrollment (CLI + GUI), encrypted SQLite store
- [x] Phase 3 — `.zshrc` sudo wrapper, Keychain-fed bridge, always fail open
- [x] Phase 4 — GUI settings panel + rumps menu-bar app
- [ ] Automated test suite (camera-in-the-loop testing is manual)

## License

MIT (or as you prefer — see LICENSE).
