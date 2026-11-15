import asyncio
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import edge_tts
from edge_tts.typing import Voice


def detect_os() -> str:
    """Return a normalized OS name supported by edge-playback."""
    operating_system = platform.system()
    supported_systems = {
        "Windows": "windows",
        "Darwin": "macos",
        "Linux": "linux",
    }
    if operating_system not in supported_systems:
        raise RuntimeError(f"Unsupported operating system: {operating_system}")
    return supported_systems[operating_system]


def get_available_voices(locale: str | None = None) -> list[Voice]:
    """Retrieve Edge TTS voices, optionally filtering by locale (for example, en-US)."""
    voices = asyncio.run(edge_tts.list_voices())
    if locale is None:
        return voices
    return [voice for voice in voices if voice["Locale"] == locale]


def _playback_command(audio_file: str) -> list[str]:
    operating_system = detect_os()
    if operating_system == "macos":
        executable = shutil.which("afplay")
        if executable:
            return [executable, audio_file]
    elif operating_system == "linux":
        executable = shutil.which("mpv")
        if executable:
            return [executable, "--no-video", audio_file]
        executable = shutil.which("ffplay")
        if executable:
            return [
                executable,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "error",
                audio_file,
            ]

    requirements = {
        "macos": "afplay (included with macOS)",
        "linux": "mpv or ffplay",
    }
    raise RuntimeError(
        f"No supported audio player found on {operating_system}; "
        f"install or enable {requirements[operating_system]}."
    )


def _edge_playback_executable() -> str:
    executable_name = "edge-playback.exe"
    executable = shutil.which(executable_name)
    if executable:
        return executable

    executable_path = Path(sys.executable).with_name(executable_name)
    if executable_path.is_file():
        return str(executable_path)

    raise RuntimeError(
        "Could not find edge-playback. Install the project dependencies with "
        "`python -m pip install -r requirements.txt`."
    )


def speak(text: str, voice: str | None = None) -> None:
    """Speak text using an Edge TTS voice on Windows, macOS, or Linux."""
    if not text.strip():
        return

    operating_system = detect_os()
    available_voices = get_available_voices("en-US")
    if voice is None:
        voice = next(
            (
                item["ShortName"]
                for item in available_voices
                if item["Gender"] == "Female"
            ),
            None,
        )
        if voice is None:
            raise RuntimeError("No female en-US Edge TTS voices are available.")
    elif not any(item["ShortName"] == voice for item in available_voices):
        raise ValueError(f"Edge TTS voice is not available: {voice}")

    if operating_system == "windows":
        subprocess.run(
            [_edge_playback_executable(), "--voice", voice, "--text", text],
            check=True,
        )
        return

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
        audio_path = audio_file.name
    try:
        asyncio.run(edge_tts.Communicate(text, voice).save(audio_path))
        subprocess.run(_playback_command(audio_path), check=True)
    finally:
        Path(audio_path).unlink(missing_ok=True)


if __name__ == "__main__":
    text = input("Enter text to speak: ").strip()
    if text:
        speak(text)

 