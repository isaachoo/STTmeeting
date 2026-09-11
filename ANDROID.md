# Building the Android app

This turns the meeting copilot into a standalone Android app: the same Python
server that runs on your PC runs *inside* the app, and the same web page is
shown in a full-screen view. Nothing needs the PC once the app is on the phone.
Deepgram and OpenRouter are reached over the phone's own internet.

Written for someone who has never built an Android app. It is a one-time setup
of about an hour (mostly downloads), then a two-minute rebuild whenever the
project changes.

> **Read this first.** The Android project under `android/` was written and
> checked carefully, and the Python side of it was run and tested — but it has
> **not** been compiled or run on a phone by the author, because the machine it
> was written on cannot reach Google's Android SDK servers. The first build is
> where any mistake will show up. If Android Studio reports an error, copy the
> **first** red line from the *Build* window and send it back; it is almost
> always a one-line fix.

---

## Part 1 — Install the tools on your Windows PC (once)

### 1.1 Python 3.12

The build packages Python into the app and needs the **same version** on your
PC to do it. Check what you have — in PowerShell:

```powershell
py -3.12 --version
```

If it prints `Python 3.12.x`, skip to 1.2. If not, install it:

1. Go to <https://www.python.org/downloads/windows/>.
2. Under **Python 3.12**, download the *Windows installer (64-bit)*.
3. Run it. Tick **"Add python.exe to PATH"**. Click *Install Now*.
4. Open a *new* PowerShell window and run `py -3.12 --version` again.

Having 3.12 alongside another Python is fine; `run.bat` on the PC is unaffected.

### 1.2 Android Studio

1. Download from <https://developer.android.com/studio> (about 1.2 GB).
2. Run the installer with all defaults.
3. Start Android Studio. A **Setup Wizard** appears: choose **Standard**, accept
   the licences (you will have to scroll and click each one), and let it
   download the SDK (another 1–2 GB). This is the slow part; go and get a coffee.
4. When it shows the *Welcome to Android Studio* screen, you are done.

Android Studio includes its own Java, so you do **not** install Java separately.

### 1.3 Get the project

If you already have the project on the PC from the desktop instructions, just
make sure it is up to date:

```powershell
cd C:\Users\isaac.ho\STTmeeting
git pull origin claude/cantonese-meeting-copilot-u5khdk
```

Otherwise clone it (outside OneDrive — see the README for why):

```powershell
cd C:\Users\isaac.ho
git clone https://github.com/isaachoo/STTmeeting.git
cd STTmeeting
git checkout claude/cantonese-meeting-copilot-u5khdk
```

---

## Part 2 — Open the project and let it set itself up

1. In Android Studio's welcome screen, click **Open**.
2. Browse to the project and select the **`android`** folder inside it —
   `C:\Users\isaac.ho\STTmeeting\android`. **Not** the `STTmeeting` folder itself.
3. If asked *"Trust this project?"* — **Trust Project**.
4. Android Studio now runs a **Gradle sync**: a progress bar at the bottom for
   several minutes. It downloads the build tools, Gradle 8.9, the Chaquopy
   plugin, and Python 3.12 for Android. Wait until the bar disappears.

Things that may pop up during the sync — and what to do:

| Pop-up / message | What to do |
|---|---|
| **"Install missing SDK package(s)"** (Android SDK Platform 35, Build-Tools) | Click the link, accept the licence, **Install**. Sync restarts on its own. |
| **"Android Gradle Plugin can be upgraded"** / *Upgrade Assistant* | **Decline / Don't remind me.** The project pins versions that are known to work together; upgrading past AGP 9.3 breaks the Python plugin. |
| **"Gradle JDK … recommended"** | Ignore. The bundled JDK 17 is correct. |
| Sync ends with **BUILD SUCCESSFUL** in the *Build* window | You are ready for Part 3. |

If the sync fails, see **Troubleshooting** at the bottom — the three common
causes are there.

---

## Part 3 — Build the APK

1. In the top menu: **Build ▸ Build App Bundle(s) / APK(s) ▸ Build APK(s)**.
2. The first build takes 3–10 minutes: it packages Python, installs the app's
   pip dependencies for Android, and copies in the copilot's code and web files.
   Later builds take under a minute.
3. When it finishes, a small notification appears bottom-right: **"APK(s)
   generated successfully"** with a **locate** link. Click it.

   If you miss it, the file is always at:
   `C:\Users\isaac.ho\STTmeeting\android\app\build\outputs\apk\debug\app-debug.apk`

That file is your app. It is a *debug* build, which is exactly right for
installing on your own phone (see Part 6 if you ever need a signed *release*
build).

---

## Part 4 — Put it on the phone

Either method works. USB is more reliable; file transfer needs no cable.

### 4.1 By USB (Android Studio installs it for you)

1. On the phone: **Settings ▸ About phone**, tap **Build number** seven times
   until it says *You are now a developer*.
2. **Settings ▸ System ▸ Developer options**: turn on **USB debugging**.
3. Connect the phone by USB. On the phone, tap **Allow** on *"Allow USB
   debugging?"* (tick *Always allow from this computer*).
4. In Android Studio the phone's name appears in the device dropdown at the
   top. Click the green **▶ Run** button. The app installs and opens.

### 4.2 By copying the file

1. Copy `app-debug.apk` to the phone — by USB cable as a file, or send it to
   yourself (WhatsApp, email, Google Drive, OneDrive).
2. On the phone, open the file (from *Files*, or from the message/Drive).
3. Android asks to allow installs from that app ("Install unknown apps") —
   allow it, once. Then **Install**.
4. Google Play Protect may say *"Unknown app"* — tap **More details ▸ Install
   anyway**. This is normal for any app not from the Play Store.

---

## Part 5 — First run on the phone

1. Open **Meeting Copilot**. The screen shows *Starting the copilot…* for
   10–30 seconds on the very first launch (Python is being unpacked). After
   that it is a few seconds.
2. It asks for the **Microphone** — *Allow*. And for **Notifications** — *Allow*
   (this is what lets it keep recording with the screen off).
3. The familiar page appears. The banner says which keys are missing. Tap
   **Enter API keys**, paste your Deepgram and OpenRouter keys, **Save keys**.
   They are stored on the phone only, in the app's private storage.
4. A permanent notification **"Meeting Copilot is running"** appears while the
   app is alive. That is deliberate — it is what stops Android killing the
   recording. Its **Quit** button stops everything.

### Using it in a meeting

- **Plug the phone in.** Streaming audio for two hours is fine on battery, but
  the screen stays on while the app is open (on purpose — see below), and that
  is what drains it.
- **Put it in the middle of the table**, screen up. Phone microphones are good;
  a phone in the middle beats a laptop at the end.
- **Leave the app open.** The screen is kept on while the copilot is in front.
  Switching to another app or locking the phone *should* keep recording thanks
  to the service, but Android's WebView is allowed to pause microphone capture
  for a backgrounded app and some phone makers are more aggressive than others.
  Test it once on your phone before relying on it: start a meeting, lock the
  phone for a minute, unlock, and check the transcript kept going.
- **Downloads** (Markdown/JSON exports, reports) go to the phone's **Downloads**
  folder; a message confirms the filename. They show up in *Files* and in any
  "attach a file" picker.
- **The review workspace** works exactly as on the PC — *Review meeting →* after
  a meeting stops, or *Session ▸ Past meetings ▸ Review*.
- Meetings recorded on the phone live on the phone; ones recorded on the PC live
  on the PC. There is no sync between them. Export as Markdown to move one.

### Updating the app later

```powershell
cd C:\Users\isaac.ho\STTmeeting
git pull origin claude/cantonese-meeting-copilot-u5khdk
```

Then in Android Studio: **Build ▸ Build APK(s)**, and install the new file over
the old one (Part 4 — your meetings and keys are kept). If Android refuses with
*"App not installed"*, it is because the old one was installed from a different
PC or keystore; uninstall it first (this deletes its meetings — export them).

---

## Part 6 — A signed release build (optional)

Only needed if you want to hand the app to colleagues or later publish it. For
your own phone, the debug build is fine.

1. **Build ▸ Generate Signed App Bundle / APK ▸ APK ▸ Next**.
2. **Create new…** under *Key store path*. Save it somewhere safe *outside* the
   project (e.g. `C:\Users\isaac.ho\meetingcopilot.jks`), choose passwords,
   fill in a name. **Keep this file and its passwords** — every future update
   must be signed with the same key or phones will refuse to install it.
3. **Next**, choose *release*, **Create**.
4. The APK is at `android\app\build\outputs\apk\release\app-release.apk`.

---

## What is where, if you are curious

```
android/
  build.gradle.kts            versions of the build tools (pinned on purpose)
  app/build.gradle.kts        copies the copilot's code + web files into the build,
                              packages Python 3.12 and requirements.txt via Chaquopy
  app/src/python/android_entry.py   the hand-over: sets the data/web paths, runs app.main()
  app/src/main/java/.../
    MainActivity.kt           the WebView, microphone permission, downloads
    CopilotService.kt         foreground service that runs the Python server
    WebAssets.kt              unpacks templates/ and static/ so Flask can read them
    Downloads.kt              saves exports to the phone's Downloads folder
```

The copilot itself is not duplicated: the Python packages, `templates/` and
`static/` are copied from the repository at build time, so the phone always
runs exactly the code the PC runs. Two small changes were made to the shared
code for this: `config.py` reads `MEETING_DATA_DIR` and `app.py` reads
`MEETING_TEMPLATE_DIR` / `MEETING_STATIC_DIR`, all optional.

What the phone build leaves out: the **local (sherpa-onnx) transcriber**. It
needs native libraries built for Android, which is a separate project. Deepgram,
Speechmatics and Qwen3-ASR via OpenRouter work (the last is pure Python and
needs only the OpenRouter key); the transcriber dropdown will still list
*Local*, and choosing it will fail with a clear message.

---

## Troubleshooting

**Gradle sync fails immediately with a network / proxy error**
Your office network is blocking the downloads. Try on a home network or a phone
hotspot for the first sync; afterwards everything is cached and it builds offline.

**`Chaquopy: buildPython … not found` / `py -3.12` errors**
Python 3.12 is not installed, or was installed without *Add to PATH*. Do 1.1
again, then **File ▸ Sync Project with Gradle Files**.

**`Failed to install the following SDK components`**
Click the licence link in the error, accept, and sync again. If it persists:
**Tools ▸ SDK Manager ▸ SDK Platforms**, tick *Android 15 (API 35)*, **Apply**.

**`Minimum supported Gradle version is …` / `The project is using an incompatible version of the Android Gradle plugin`**
Android Studio changed a version on you. In `android/build.gradle.kts` the
Android Gradle Plugin must be **8.x** and in
`android/gradle/wrapper/gradle-wrapper.properties` Gradle must be **8.9**. Put
them back and sync.

**Sync is fine, but the build fails inside a task with `Python` or `pip` in its name**
Read the first error line: pip could not find a package for Android. Send that
line back — it means one dependency needs pinning to a different version.

**The app opens but stays on "Starting the copilot…"**
The server inside the app did not start. Connect the phone by USB and run this
in PowerShell from the `android` folder to see Python's own error:

```powershell
.\gradlew.bat --quiet
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" logcat -s python.stderr python.stdout CopilotService
```

(`adb.exe` is installed with Android Studio at that path.) Send back the
lines starting from the first `Traceback`.

**The microphone does not work in the app**
Settings ▸ Apps ▸ Meeting Copilot ▸ Permissions ▸ Microphone ▸ **Allow**. Then
force-stop and reopen the app.

**"Start" is refused with `Missing: DEEPGRAM_API_KEY`**
Keys entered on the PC do not travel to the phone. Enter them again in the app
(Session ▸ API keys). They are kept across updates.

**Recording stops when the phone is locked**
Your phone's battery optimiser is killing it. Settings ▸ Apps ▸ Meeting Copilot
▸ Battery ▸ **Unrestricted** (wording varies: *Don't optimise*, *No restrictions*).
On Samsung also check *Settings ▸ Battery ▸ Background usage limits* and make
sure the app is not in *Sleeping apps*. Or simply keep the app open — it keeps
the screen on for exactly this reason.
