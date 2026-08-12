"""Tests for entering API keys in the app instead of editing .env."""

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

_TMP = tempfile.TemporaryDirectory()
config.DATA_DIR = Path(_TMP.name)
config.DB_PATH = config.DATA_DIR / "settings-tests.sqlite3"

import settings  # noqa: E402


def _field(described: dict, name: str) -> dict:
    """describe() returns an ordered list; look one up by name."""
    return next(f for f in described["fields"] if f["name"] == name)


class TestFieldOrder(unittest.TestCase):
    def test_the_deepgram_key_comes_first(self):
        names = [f["name"] for f in settings.describe()["fields"]]
        self.assertEqual(names[0], "DEEPGRAM_API_KEY")
        self.assertLess(names.index("DEEPGRAM_API_KEY"), names.index("STT_PROVIDER"))


class SettingsCase(unittest.TestCase):
    def setUp(self):
        # Each test gets its own settings file and a clean config, and puts both
        # back afterwards -- other test modules share this config object.
        self.dir = Path(tempfile.mkdtemp())
        original_data_dir = config.DATA_DIR
        self.addCleanup(setattr, config, "DATA_DIR", original_data_dir)
        config.DATA_DIR = self.dir
        saved_env = dict(settings._ENV_AT_STARTUP)
        saved_config = {name: getattr(config, name, "") for name in settings.FIELDS}

        def restore():
            settings._ENV_AT_STARTUP.clear()
            settings._ENV_AT_STARTUP.update(saved_env)
            for name, value in saved_config.items():
                setattr(config, name, value)

        self.addCleanup(restore)
        for name in settings.FIELDS:
            settings._ENV_AT_STARTUP[name] = ""
            setattr(config, name, "")


class TestSavingAndLoading(SettingsCase):
    def test_a_saved_key_reaches_config_without_a_restart(self):
        self.assertEqual(config.DEEPGRAM_API_KEY, "")
        settings.save({"DEEPGRAM_API_KEY": "dg-live-key"})
        self.assertEqual(config.DEEPGRAM_API_KEY, "dg-live-key")

    def test_it_persists_to_disk(self):
        settings.save({"SPEECHMATICS_API_KEY": "sm-key"})
        with open(settings.path(), encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["SPEECHMATICS_API_KEY"], "sm-key")

    def test_reports_what_changed(self):
        self.assertEqual(settings.save({"DEEPGRAM_API_KEY": "one"}), ["DEEPGRAM_API_KEY"])
        self.assertEqual(settings.save({"DEEPGRAM_API_KEY": "one"}), [],
                         "saving the same value is not a change")
        self.assertEqual(settings.save({"DEEPGRAM_API_KEY": "two"}), ["DEEPGRAM_API_KEY"])

    def test_an_empty_field_leaves_the_existing_key_alone(self):
        """The form shows a mask, so a blank field means untouched. Treating it
        as 'delete' would wipe every key the user did not retype."""
        settings.save({"DEEPGRAM_API_KEY": "keep-me", "OPENROUTER_API_KEY": "or"})
        settings.save({"DEEPGRAM_API_KEY": "", "OPENROUTER_API_KEY": "  "})
        self.assertEqual(config.DEEPGRAM_API_KEY, "keep-me")
        self.assertEqual(config.OPENROUTER_API_KEY, "or")

    def test_a_dash_removes_a_key(self):
        settings.save({"DEEPGRAM_API_KEY": "bye"})
        self.assertEqual(settings.save({"DEEPGRAM_API_KEY": "-"}), ["DEEPGRAM_API_KEY"])
        self.assertEqual(config.DEEPGRAM_API_KEY, "")
        self.assertNotIn("DEEPGRAM_API_KEY", settings.load())

    def test_unknown_names_are_ignored(self):
        settings.save({"SOMETHING_ELSE": "x", "DEEPGRAM_API_KEY": "y"})
        self.assertEqual(settings.load(), {"DEEPGRAM_API_KEY": "y"})

    def test_values_are_trimmed(self):
        settings.save({"DEEPGRAM_API_KEY": "  spaced  "})
        self.assertEqual(config.DEEPGRAM_API_KEY, "spaced")

    def test_a_corrupt_file_is_ignored_rather_than_fatal(self):
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.path().write_text("{not json", encoding="utf-8")
        self.assertEqual(settings.load(), {})
        settings.apply_to_config()  # must not raise

    def test_a_missing_file_is_fine(self):
        self.assertEqual(settings.load(), {})

    @unittest.skipIf(os.name == "nt", "POSIX permissions only")
    def test_the_file_is_not_world_readable(self):
        settings.save({"DEEPGRAM_API_KEY": "secret"})
        mode = stat.S_IMODE(settings.path().stat().st_mode)
        self.assertEqual(mode & (stat.S_IRGRP | stat.S_IROTH), 0)


class TestEnvInteraction(SettingsCase):
    def test_env_is_used_when_nothing_is_saved(self):
        settings._ENV_AT_STARTUP["DEEPGRAM_API_KEY"] = "from-env"
        settings.apply_to_config()
        self.assertEqual(config.DEEPGRAM_API_KEY, "from-env")

    def test_a_saved_key_overrides_env(self):
        """Otherwise typing a key into the app and pressing Save would silently
        do nothing whenever .env already had one."""
        settings._ENV_AT_STARTUP["DEEPGRAM_API_KEY"] = "from-env"
        settings.save({"DEEPGRAM_API_KEY": "from-app"})
        self.assertEqual(config.DEEPGRAM_API_KEY, "from-app")

    def test_clearing_a_saved_key_falls_back_to_env(self):
        settings._ENV_AT_STARTUP["DEEPGRAM_API_KEY"] = "from-env"
        settings.save({"DEEPGRAM_API_KEY": "from-app"})
        settings.save({"DEEPGRAM_API_KEY": "-"})
        self.assertEqual(config.DEEPGRAM_API_KEY, "from-env")

    def test_the_notes_model_follows_the_copilot_model(self):
        settings.save({"OPENROUTER_MODEL": "qwen/qwen3.6-plus"})
        self.assertEqual(config.OPENROUTER_NOTES_MODEL, "qwen/qwen3.6-plus")

    def test_the_provider_is_normalised(self):
        settings.save({"STT_PROVIDER": "  LOCAL  "})
        self.assertEqual(config.STT_PROVIDER, "local")


class TestMasking(SettingsCase):
    def test_a_key_is_recognisable_but_not_usable(self):
        masked = settings.mask("dg_1234567890abcdef")
        self.assertTrue(masked.endswith("cdef"))
        self.assertNotIn("1234567890", masked)

    def test_short_values_reveal_nothing(self):
        self.assertEqual(settings.mask("abc"), "•••")

    def test_empty_stays_empty(self):
        self.assertEqual(settings.mask(""), "")
        self.assertEqual(settings.mask(None), "")

    def test_describe_never_returns_a_usable_secret(self):
        settings.save({"DEEPGRAM_API_KEY": "dg_supersecretvalue"})
        described = settings.describe()
        blob = json.dumps(described)
        self.assertNotIn("dg_supersecretvalue", blob)
        self.assertNotIn("supersecret", blob)
        field = _field(described, "DEEPGRAM_API_KEY")
        self.assertTrue(field["set"])
        self.assertEqual(field["source"], "saved in the app")

    def test_describe_shows_where_each_value_came_from(self):
        settings._ENV_AT_STARTUP["OPENROUTER_API_KEY"] = "env-key"
        settings.apply_to_config()
        settings.save({"DEEPGRAM_API_KEY": "app-key"})
        described = settings.describe()
        self.assertEqual(_field(described, "OPENROUTER_API_KEY")["source"], "from .env")
        self.assertEqual(_field(described, "DEEPGRAM_API_KEY")["source"], "saved in the app")
        self.assertEqual(_field(described, "SPEECHMATICS_API_KEY")["source"], "")

    def test_non_secret_fields_are_shown_in_full(self):
        settings.save({"OPENROUTER_MODEL": "deepseek/deepseek-v3.2"})
        field = _field(settings.describe(), "OPENROUTER_MODEL")
        self.assertFalse(field["secret"])
        self.assertEqual(field["value"], "deepseek/deepseek-v3.2")


class TestUnblockingTheApp(SettingsCase):
    def test_entering_keys_clears_the_missing_list(self):
        self.assertIn("DEEPGRAM_API_KEY", config.missing_keys("deepgram"))
        self.assertIn("OPENROUTER_API_KEY", config.missing_keys("deepgram"))

        settings.save({"DEEPGRAM_API_KEY": "dg", "OPENROUTER_API_KEY": "or"})
        self.assertEqual(config.missing_keys("deepgram"), [])

    def test_the_local_provider_only_needs_the_llm_key(self):
        settings.save({"OPENROUTER_API_KEY": "or"})
        self.assertEqual(config.missing_keys("local"), [])
        self.assertEqual(config.missing_keys("deepgram"), ["DEEPGRAM_API_KEY"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
