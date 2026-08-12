"""API keys entered in the app, saved outside the code.

Keys live in `data/settings.json` rather than `.env`, so nobody has to find and
edit a dotfile to get started. `.env` still works and is still read first; a key
saved here overrides it, because otherwise typing one into the app and pressing
Save would appear to do nothing.

The file is plain JSON. That is worth saying out loud: the keys are not
encrypted, because a local app has nowhere to hide a decryption key that an
attacker with read access to the same folder could not also reach. On Unix the
file is chmod 600; on Windows it inherits the folder's permissions. Treat it the
way you would treat `.env`.
"""

import json
import logging
import os
import stat

import config

log = logging.getLogger(__name__)

# What the app will accept from the settings form. Everything else in config
# stays where it is -- this is for the values a user actually needs to enter.
FIELDS = {
    "DEEPGRAM_API_KEY": {"label": "Deepgram API key", "secret": True},
    "SPEECHMATICS_API_KEY": {"label": "Speechmatics API key", "secret": True},
    "OPENROUTER_API_KEY": {"label": "OpenRouter API key", "secret": True},
    "TAVILY_API_KEY": {"label": "Tavily API key (optional, for web search)", "secret": True},
    "STT_PROVIDER": {"label": "Default transcriber", "secret": False},
    "OPENROUTER_MODEL": {"label": "Copilot model", "secret": False},
}

_ENV_AT_STARTUP = {name: (os.getenv(name) or "").strip() for name in FIELDS}


def path():
    return config.DATA_DIR / "settings.json"


def load() -> dict:
    """Whatever was saved, or an empty dict. A corrupt file must not stop the app."""
    try:
        with open(path(), encoding="utf-8") as handle:
            saved = json.load(handle)
    except FileNotFoundError:
        return {}
    except (ValueError, OSError):
        log.warning("could not read %s; ignoring it", path())
        return {}
    return {k: v for k, v in saved.items() if k in FIELDS and isinstance(v, str)}


def apply_to_config() -> None:
    """Push saved values onto the config module so the rest of the app sees them.

    Called at startup and again after every save, so a key entered in the app
    works for the next meeting without restarting anything.

    Only ever *applies* -- it never blanks a value it has nothing to say about.
    A config attribute may have been set some other way (by a test, or by
    embedding this app in something else), and importing a module should not
    quietly erase it. Explicit removal is handled in `save`.
    """
    saved = load()
    for name in FIELDS:
        value = saved.get(name, "").strip()
        if value:
            setattr(config, name, value)
        elif _ENV_AT_STARTUP.get(name):
            # Nothing saved, but .env has one: make sure that is what is in force.
            setattr(config, name, _ENV_AT_STARTUP[name])

    # A couple of values are derived, so they have to be recomputed.
    config.OPENROUTER_NOTES_MODEL = (
        (os.getenv("OPENROUTER_NOTES_MODEL") or "").strip() or config.OPENROUTER_MODEL
    )
    config.STT_PROVIDER = (config.STT_PROVIDER or "deepgram").strip().lower()


def save(updates: dict) -> list[str]:
    """Merge `updates` into the saved settings. Returns the names that changed.

    An empty string means "leave this alone", because the form only ever shows a
    masked value and saving the form must not wipe a key the user did not touch.
    To remove one, send the string "-".
    """
    saved = load()
    changed = []

    for name, spec in FIELDS.items():
        if name not in updates:
            continue
        value = str(updates[name] or "").strip()
        if not value:
            continue  # untouched
        if value == "-":
            if saved.pop(name, None) is not None:
                changed.append(name)
                # apply_to_config only applies, so an explicit removal has to
                # put the fallback in place itself.
                setattr(config, name, _ENV_AT_STARTUP.get(name, ""))
            continue
        if saved.get(name) != value:
            saved[name] = value
            changed.append(name)

    if changed:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        target = path()
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(saved, handle, indent=2, sort_keys=True)
        try:
            os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)  # 600 where supported
        except OSError:
            pass
        apply_to_config()
        # Deliberately never log the values.
        log.info("settings updated: %s", ", ".join(changed))

    return changed


def mask(value: str) -> str:
    """Enough to recognise a key by, never enough to use it."""
    value = (value or "").strip()
    if not value:
        return ""
    if len(value) <= 8:
        return "•" * len(value)
    return f"{'•' * 8}{value[-4:]}"


def describe() -> dict:
    """What the settings form shows: never a usable secret, and where it came from.

    A list rather than a dict so the order survives serialisation -- the Deepgram
    key belongs at the top, not wherever alphabetical sorting puts it.
    """
    saved = load()
    fields = []
    for name, spec in FIELDS.items():
        current = (getattr(config, name, "") or "").strip()
        if name in saved and saved[name].strip():
            source = "saved in the app"
        elif _ENV_AT_STARTUP.get(name):
            source = "from .env"
        else:
            source = ""
        fields.append({
            "name": name,
            "label": spec["label"],
            "secret": spec["secret"],
            "set": bool(current),
            "value": mask(current) if spec["secret"] else current,
            "source": source,
        })
    return {"fields": fields, "path": str(path())}
