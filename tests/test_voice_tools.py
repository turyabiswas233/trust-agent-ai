import unittest
from unittest.mock import Mock, patch

from libs import voice
from tools.auto_switch_voice import auto_switch_voice
from tools.list_voice_models import list_voice_models


class VoiceSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_preferred = voice._preferred_english_voice
        self.original_active = voice._active_voice
        voice._preferred_english_voice = voice.DEFAULT_ENGLISH_VOICE
        voice._active_voice = None
        self.addCleanup(
            setattr, voice, "_preferred_english_voice", self.original_preferred
        )
        self.addCleanup(setattr, voice, "_active_voice", self.original_active)

    def test_bengali_switch_restores_preferred_english_voice(self) -> None:
        english = auto_switch_voice("en", "Ava")
        bengali = auto_switch_voice("bn")
        restored_english = auto_switch_voice("English")

        self.assertIn("Ava (en_US-amy-medium)", english)
        self.assertIn("Ananya (bn_BD-google-medium)", bengali)
        self.assertIn("Ava (en_US-amy-medium)", restored_english)

    def test_rejects_voice_for_the_wrong_language(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support"):
            voice.select_voice("bn", "Miles")

    def test_tool_lists_installed_models_with_assistant_names(self) -> None:
        models = list_voice_models()

        self.assertIn("Ananya - bn_BD-google-medium (Bengali)", models)
        self.assertIn("Ava - en_US-amy-medium (English)", models)

    def test_switch_tool_changes_voice_used_by_speak(self) -> None:
        auto_switch_voice("en", "Sage")
        loaded_voice = Mock()
        with (
            patch("libs.voice._load_voice", return_value=loaded_voice) as load_voice,
            patch("libs.voice.wave.open") as open_wave,
            patch("libs.voice.platform.system", return_value="Other"),
        ):
            voice.speak("Hello")

        load_voice.assert_called_once_with("en_US-sam-medium")
        loaded_voice.synthesize_wav.assert_called_once_with(
            "Hello", open_wave.return_value.__enter__.return_value
        )


if __name__ == "__main__":
    unittest.main()
