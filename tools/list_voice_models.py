from libs.voice import VOICE_DIR, available_voice_models, current_voice_model


def list_voice_models() -> str:
    """List installed voice models and their assistant names."""
    models = available_voice_models()
    if not models:
        return f"No voice models found in {VOICE_DIR}."

    lines = ["Available voice models:"]
    for model in models:
        selected = " (current)" if model["model"] == current_voice_model() else ""
        lines.append(
            f"- {model['assistant_name']} - {model['model']} "
            f"({model['language']}){selected}"
        )
    return "\n".join(lines)
