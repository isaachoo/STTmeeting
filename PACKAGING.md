# Packaging for Windows

How to turn this repo into `MeetingCopilot-Setup.exe` — a normal Windows install with
no Python, no command line, and no editing `.env` in Notepad.

**Target experience**

```
Download MeetingCopilot-Setup.exe
  → Next, Next, Install
  → App opens, paste two API keys, Save
  → Start meeting
```

**Approach**: PyInstaller in one-folder mode to bundle Python and the dependencies,
then Inno Setup to wrap that folder in a proper installer. Both are free.

**Effort**: roughly 1 day of work in total. Phase 1 is the real work; phases 2–4 are
mostly configuration.

---

## The four things that must change in the code first

Packaging is not just running PyInstaller. Four assumptions in the current code break
the moment it becomes an installed application, and all four are in Phase 1.

| Assumption today | Why it breaks | Fix |
|---|---|---|
| `config.BASE_DIR` is the repo folder, and `data/` lives inside it | `C:\Program Files\` is not user-writable, and an upgrade replaces the folder — losing every past meeting | Split read-only resources from writable data; data goes to `%LOCALAPPDATA%` |
| Keys come from `.env` next to the code | Non-technical users will not edit a dotfile in Program Files | Settings page in the app, saved to `%LOCALAPPDATA%` |
| Flask finds `templates/` and `static/` relative to `app.py` | Inside a bundle those paths are somewhere else | Resolve via `sys._MEIPASS` when frozen |
| The user opens a browser and types the address | Nobody wants to remember `127.0.0.1:5000` | Open the browser automatically on launch |

---

## Phase 1 — Make the app installable-friendly

### 1.1 Add a path helper

New file `paths.py`:

```python
"""Where things live, whether running from source or from an installed build."""

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    """True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def resource_dir() -> Path:
    """Read-only files that ship with the app: templates, static."""
    if is_frozen():
        # onedir puts data next to the exe; onefile extracts to _MEIPASS.
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def data_dir() -> Path:
    """Writable per-user data: database, settings, saved audio.

    Deliberately NOT inside the install folder -- Program Files is not writable
    and an upgrade would wipe it.
    """
    if is_frozen():
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "MeetingCopilot"
    else:
        base = Path(__file__).resolve().parent / "data"
    base.mkdir(parents=True, exist_ok=True)
    return base
```

### 1.2 Point `config.py` at those

Replace the `BASE_DIR` / `DATA_DIR` block:

```python
from paths import data_dir, is_frozen, resource_dir

BASE_DIR = resource_dir()
DATA_DIR = data_dir()
AUDIO_DIR = DATA_DIR / "audio"
DB_PATH = DATA_DIR / "meetings.sqlite3"
SETTINGS_PATH = DATA_DIR / "settings.json"

# Installed builds read settings.json; running from source keeps using .env.
load_dotenv(BASE_DIR / ".env")
```

Then have config read `settings.json` **after** `.env`, so the in-app settings win
for an installed build. Keep every existing environment variable working — running
from source must not change.

### 1.3 Tell Flask where its files are

In `app.py`:

```python
from paths import resource_dir

app = Flask(
    __name__,
    template_folder=str(resource_dir() / "templates"),
    static_folder=str(resource_dir() / "static"),
)
```

### 1.4 Add the Settings page

This is the single biggest usability win — it removes Notepad from the flow entirely.

- `GET /settings` — a form: Deepgram key, OpenRouter key, optional Tavily key,
  language, advisor model, attendee mode.
- `POST /settings` — validate, write `settings.json` (mask keys when displaying:
  show only the last 4 characters of a saved key).
- If required keys are missing on startup, redirect `/` to `/settings` with a short
  explanation instead of showing the red banner.
- After saving, show "Saved — restart the app for changes to take effect", or
  reload the config in place if that turns out to be simple. Restarting is honest
  and simpler; do that first.

**Do not** put the keys in the page's HTML once saved. Send back a masked value.

### 1.5 Launch behaviour

In `main()`:

```python
def main() -> None:
    db.init()
    port = first_free_port(config.PORT)      # 5000, else 5001, 5002...
    if already_running(port):                # another instance owns it
        webbrowser.open(f"http://127.0.0.1:{port}")
        return
    threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="127.0.0.1", port=port, threaded=True)
```

Three separate improvements there, all worth having:

- **Auto-open the browser** one second after start.
- **Fall back to another port** if 5000 is taken by something else.
- **Single instance**: a second launch focuses the existing app instead of failing
  with a confusing "port in use" traceback.

Keep the console window for now — when something goes wrong, the user can read the
error and send it to you. A windowless build with a tray icon can come later.

### 1.6 Verify from source

Nothing above should change how the app behaves when run from the repo.

```
.venv\Scripts\python -m unittest discover -s tests
.venv\Scripts\python app.py
```

All tests pass, the app still works, `data\` still lands in the repo folder.

---

## Phase 2 — Build the executable

### 2.1 Install the build tool

```
.venv\Scripts\pip install pyinstaller
```

### 2.2 Make an icon

A 256×256 `.ico` at `build_assets\icon.ico`. Any online PNG-to-ICO converter will do.

### 2.3 Write the spec file

`MeetingCopilot.spec`:

```python
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
        ('.env.example', '.'),
    ],
    hiddenimports=[
        'flask_sock',
        'simple_websocket',
        'wsproto',
        'websocket',          # websocket-client, for Deepgram
        'dotenv',
        'engineio.async_drivers.threading',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'numpy', 'PIL', 'pytest'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='MeetingCopilot',
    debug=False,
    console=True,                 # keep visible until it is proven stable
    icon='build_assets/icon.ico',
    version='build_assets/version_info.txt',
)

coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,       # UPX compression triggers antivirus
    name='MeetingCopilot',
)
```

**One-folder, not one-file, on purpose.** One-file is tidier but it unpacks to a temp
directory on every launch (slower start) and is far more likely to be flagged by
antivirus. One-folder is the safer choice for a tool you want colleagues to run.

### 2.4 Build

```
.venv\Scripts\pyinstaller MeetingCopilot.spec --clean --noconfirm
```

Output: `dist\MeetingCopilot\MeetingCopilot.exe` plus an `_internal` folder.
Expect 40–60 MB.

### 2.5 Test the build on this machine

```
dist\MeetingCopilot\MeetingCopilot.exe
```

- [ ] Browser opens by itself
- [ ] Settings page appears (no keys yet), accepts keys, saves
- [ ] `%LOCALAPPDATA%\MeetingCopilot\settings.json` exists
- [ ] A meeting starts, transcript flows, microphone permission prompt appears
- [ ] `%LOCALAPPDATA%\MeetingCopilot\meetings.sqlite3` grows
- [ ] Export downloads work
- [ ] Closing the console window stops the app cleanly

**Most likely failure**: a missing module at startup — the console will name it. Add
it to `hiddenimports` and rebuild. That loop is normal; expect two or three rounds.

---

## Phase 3 — Wrap it in an installer

### 3.1 Install Inno Setup

From <https://jrsoftware.org/isdl.php>. Free.

### 3.2 Write the installer script

`installer.iss`:

```ini
[Setup]
AppName=Meeting Copilot
AppVersion=1.0.0
AppPublisher=Isaac Ho
DefaultDirName={autopf}\MeetingCopilot
DefaultGroupName=Meeting Copilot
OutputBaseFilename=MeetingCopilot-Setup
OutputDir=installer_output
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupIconFile=build_assets\icon.ico
; Per-user install: no admin prompt, and no Program Files permission problems
PrivilegesRequired=lowest
DisableProgramGroupPage=yes

[Files]
Source: "dist\MeetingCopilot\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Meeting Copilot"; Filename: "{app}\MeetingCopilot.exe"
Name: "{userdesktop}\Meeting Copilot"; Filename: "{app}\MeetingCopilot.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Run]
Filename: "{app}\MeetingCopilot.exe"; Description: "Start Meeting Copilot"; Flags: nowait postinstall skipifsilent

; Deliberately no [UninstallDelete] for %LOCALAPPDATA%\MeetingCopilot.
; Uninstalling must not throw away the user's meeting history.
```

`PrivilegesRequired=lowest` installs per-user into `%LOCALAPPDATA%\Programs`, which
avoids the UAC prompt entirely and sidesteps Program Files permissions. For a
personal tool that is the right trade.

### 3.3 Build the installer

Open `installer.iss` in Inno Setup and press **Build → Compile**, or:

```
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
```

Output: `installer_output\MeetingCopilot-Setup.exe`, around 40–60 MB.

---

## Phase 4 — Test like a new user

Do this on **a different machine, or a fresh Windows VM with no Python installed**.
Testing on the development machine proves nothing — it already has everything.

- [ ] Setup runs without an admin prompt
- [ ] Start Menu and desktop shortcuts work
- [ ] App launches, browser opens
- [ ] Settings page accepts keys and persists across a restart
- [ ] Microphone works — this is the one to check carefully, since a packaged app
      still serves over `http://127.0.0.1`, which browsers treat as a secure origin
- [ ] A full meeting runs: transcript, copilot, attendee, notes, exports
- [ ] Reinstalling over the top keeps past meetings
- [ ] Uninstall removes the program and keeps `%LOCALAPPDATA%\MeetingCopilot`

---

## Phase 5 — Optional polish, later

Only worth doing once the above works.

- **No console window**: set `console=False` and add a system tray icon (`pystray`)
  with Open / Quit. Do not do this before the app is stable — you lose the error
  messages.
- **A real app window**: `pywebview` opens an Edge WebView2 window instead of a
  browser tab. Test microphone permissions in WebView2 before committing to it.
- **Auto-update**: check GitHub Releases for a newer version on startup and link to
  it. A full updater is not worth building.
- **Portable ZIP**: the `dist\MeetingCopilot` folder zipped, for colleagues who
  cannot install software. Works as-is.

---

## The awkward part: SmartScreen

An unsigned executable triggers **"Windows protected your PC"** on first run. The user
must click **More info → Run anyway**. This is not a bug and cannot be worked around
in code.

Options:

| Option | Cost | Verdict |
|---|---|---|
| Live with it | free | Fine for you and a handful of colleagues — just warn them in advance |
| OV code signing certificate | ~US$100–200/year | Removes the warning only after building reputation |
| EV code signing certificate | ~US$300–400/year | Immediate SmartScreen trust, requires a hardware token |

For an internal tool, live with it and put a line in your instructions: *"Windows will
warn you the first time. Click More info, then Run anyway."*

**Antivirus false positives** are a related risk. One-folder mode and no UPX (both set
above) make them much less likely. If your corporate antivirus still objects, that is a
conversation with IT, not something to fix in the build.

---

## Files this adds to the repo

```
paths.py                      # where resources and data live
build_assets/icon.ico         # app icon
build_assets/version_info.txt # exe version metadata
MeetingCopilot.spec           # PyInstaller build definition
installer.iss                 # Inno Setup installer definition
build.bat                     # one command: build exe + installer
```

`build.bat`:

```bat
@echo off
cd /d "%~dp0"
echo [1/2] Building the executable...
.venv\Scripts\pyinstaller MeetingCopilot.spec --clean --noconfirm || goto :failed
echo [2/2] Building the installer...
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss || goto :failed
echo.
echo Done: installer_output\MeetingCopilot-Setup.exe
pause
exit /b 0
:failed
echo.
echo Build failed - see the messages above.
pause
exit /b 1
```

Add to `.gitignore`:

```
build/
dist/
installer_output/
*.spec.bak
```

---

## Honest summary

**What this fixes**: no Python install, no command line, no editing `.env`, a normal
Start Menu app, and a clean way to hand the tool to a colleague.

**What it does not fix**: it still needs an internet connection and API keys, it still
costs the same per meeting, and Cantonese accuracy is unchanged. Packaging is a
distribution improvement, not a capability one.

**When it is worth doing**: when you want to give this to someone else, or install it
on a second machine. If it is only ever you on one laptop, `run.bat` already does the
job and this is a day spent on nothing.
