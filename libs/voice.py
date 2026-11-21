import platform
import subprocess
import wave
from functools import lru_cache
from pathlib import Path

VOICE_DIR = Path(__file__).resolve().parents[1] / "voices"
DEFAULT_ENGLISH_VOICE = "en_US-lessac-medium"
BENGALI_VOICE = "bn_BD-google-medium"
VOICE_ASSISTANT_NAMES = {
    "bn_BD-google-medium": "Ananya",
    "en_US-hifi_tts_92-medium": "Harmony",
    "en_US-kristin-medium": "Kira",
    "en_US-lessac-medium": "Luna",
}

_preferred_english_voice = DEFAULT_ENGLISH_VOICE
_active_voice: str | None = None

def _available_voice_models() -> list[dict[str, str]]:
    models = []
    for model_path in sorted(VOICE_DIR.glob("*.onnx")):
        model_name = model_path.stem
        if model_name.startswith("bn_"):
            language = "Bengali"
            language_code = "bn"
        elif model_name.startswith("en_"):
            language = "English"
            language_code = "en"
        else:
            language = "Unknown"
            language_code = "und"
        models.append(
            {
                "model": model_name,
                "assistant_name": VOICE_ASSISTANT_NAMES.get(
                    model_name, model_name
                ),
                "language": language,
                "language_code": language_code,
            }
        )
    return models

def available_voice_models() -> list[dict[str, str]]:
    """Return voice model IDs and their user-facing assistant names."""
    return _available_voice_models()

def current_voice_model() -> str:
    """Return the active model or the configured English preference."""
    return _active_voice or _preferred_english_voice

def _language_code(language: str) -> str:
    normalized = language.strip().casefold().replace("_", "-")
    if normalized in {"bn", "bn-bd", "bangla", "bengali", "ben"}:
        return "bn"
    if normalized in {"en", "en-us", "english", "eng"}:
        return "en"
    raise ValueError(
        f"Unsupported voice language: {language!r}. Supported languages are English and Bengali."
    )

def _resolve_voice_model(voice_name: str) -> dict[str, str]:
    normalized = voice_name.strip().casefold()
    for model in _available_voice_models():
        if normalized in {
            model["model"].casefold(),
            f"{model['model']}.onnx".casefold(),
            model["assistant_name"].casefold(),
        }:
            return model
    available = ", ".join(
        f"{model['assistant_name']} ({model['model']})"
        for model in _available_voice_models()
    )
    raise ValueError(f"Unknown voice {voice_name!r}. Available voices: {available}.")

def select_voice(language: str, voice_name: str | None = None) -> dict[str, str]:
    """
    Select a voice only if a specific voice_name is provided.
    If no voice_name is provided, the current voice remains unchanged.
    """
    global _active_voice, _preferred_english_voice

    # Validation only: check if the language provided is valid
    _language_code(language) 

    if voice_name is not None and not voice_name.strip():
        raise ValueError("voice_name must not be empty.")

    # ONLY switch if a specific voice_name is explicitly requested
    if voice_name is not None:
        selected = _resolve_voice_model(voice_name)
        
        # Optional: Check if the chosen voice actually supports the language
        # Remove the check below if you want to allow any voice for any language
        lang_code = _language_code(language)
        if selected["language_code"] != lang_code:
            # We print a warning instead of crashing to be more flexible, 
            # or keep it as a ValueError if you want strict enforcement.
            print(f"Warning: Voice {voice_name} is for {selected['language']}, not {language}.")

        _active_voice = selected["model"]
        
        # Update preference if the user switched to a new English voice
        if lang_code == "en":
            _preferred_english_voice = selected["model"]
            
        return selected

    # If no voice_name is provided, we do nothing. 
    # The function returns the current state.
    return {
        "model": _active_voice or _preferred_english_voice,
        "assistant_name": VOICE_ASSISTANT_NAMES.get(_active_voice or _preferred_english_voice, "_active_voice"),
        "language": "Unknown", # Simplified as we aren't switching
        "language_code": "und"
    }

@lru_cache(maxsize=None)
def _load_voice(model_name: str):
    from piper import PiperVoice
    model_path = VOICE_DIR / f"{model_name}.onnx"
    if not model_path.is_file():
        raise FileNotFoundError(f"Voice model file not found: {model_path}")
    return PiperVoice.load(str(model_path))

def speak(text: str):
    if not text.strip():
        return

    model_name = current_voice_model()
    voice = _load_voice(model_name)
    output = Path("speech.wav")

    with wave.open(str(output), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    system = platform.system()
    if system == "Windows":
        import winsound
        winsound.PlaySound(str(output), winsound.SND_FILENAME)
    elif system == "Darwin":
        subprocess.run(["afplay", str(output)], check=True, text=True)
    elif system == "Linux":
        subprocess.run(["aplay", str(output)], check=True, text=True)

if __name__ == "__main__":
    while True:
        text = input("Enter text (exit to quit): ")
        if text.lower() == "exit":
            break
        speak(text)