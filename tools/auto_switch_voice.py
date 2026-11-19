from libs.voice import select_voice


def auto_switch_voice(language: str, voice_name: str | None = "en") -> str:
    """Choose a supported voice for the response language."""
    voice = select_voice(language, voice_name)
    return (
        f"Voice set to {voice['assistant_name']} ({voice['model']}) "
        f"for {voice['language']}."
    )
