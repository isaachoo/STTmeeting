"""Where the Android app hands over to the copilot's own code.

Called once from the foreground service, on a background thread, with the
directories Android has given the app. It sets the three environment variables
the copilot reads at import time and then runs the same `main()` a PC runs.

Nothing here knows about Android beyond the paths. If the server runs on a PC
it runs here; the only difference is where the files live.
"""

import logging
import os
from pathlib import Path


def start(data_dir: str, template_dir: str, static_dir: str, port: int = 5000) -> None:
    # Must happen before `config` is imported, because config reads these once.
    os.environ["MEETING_DATA_DIR"] = data_dir
    os.environ["MEETING_TEMPLATE_DIR"] = template_dir
    os.environ["MEETING_STATIC_DIR"] = static_dir
    os.environ["HOST"] = "127.0.0.1"  # never reachable from another device
    os.environ["PORT"] = str(port)
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Chaquopy routes stdout/stderr to logcat (tags python.stdout / python.stderr),
    # which is where `adb logcat` will show anything that goes wrong.
    logging.getLogger().setLevel(logging.INFO)

    import app as app_module

    app_module.main()
