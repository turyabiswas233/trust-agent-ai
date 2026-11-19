import json
import time
from typing import Any
from ollama import ResponseError, chat, list as olist

from libs.model_config import (
    AVAILABLE_TOOLS,
    CUSTOM_PARAMS,
    MAX_AGENT_STEPS,
    MAX_TOTAL_TOKENS,
    MODEL_NAME,
    TOOL_DEFINITIONS,
    VOICE_SUMMARY_PROMPT,
)
from libs.printer import print_response, print_streamed, print_tool_call
from libs.voice import speak

def has_model(model_name:str|None) :
    models = olist()
    flag = True
    if model_name:
        if (model_name not in (x.model for x in models.models)):
            print("Owner model is not available right now.")
            flag = False
        elif (MODEL_NAME not in (x.model for x in models.models)):
            print("Default model is not available right now.")
            flag = False
    elif (MODEL_NAME not in (x.model for x in models.models)):
            print("Default model is not available right now.")
            flag = False
    if not flag:
        print(f"Available models [default - {MODEL_NAME}]: ")
        for model in models.models:
            if model.size:
                print(f"{model.model} - {round(model.size / (1024 ** 3), 2)} GiB")
    return flag
    


def _tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return parsed


def _generate_voice_summary(text: str, model_name: str| None) -> str:
    if not model_name:
        return "sorry. no model is provided."
    response = chat(
        model=model_name,
        messages=[
            {"role": "system", "content": VOICE_SUMMARY_PROMPT},
            {"role": "user", "content": text},
        ],
        stream=False,
        think=False,
        options={
            "temperature": 0.4,
            "top_k": 10,
            "top_p": 0.95
        },
    )
    return (response.message.content or "").strip()


def run_agent_turn(
    messages: list[dict[str, Any]],
    allowed_tools: dict[str, Any] | None = None,
    max_steps: int = MAX_AGENT_STEPS,
    max_tokens: int = MAX_TOTAL_TOKENS,
    model_name: str | None = None,
    show_output: bool = True,
    show_tool_response: bool = False,
    options: dict[str, Any] | None = None,
    keep_alive: int | str | None = None,
    voice_enabled: bool = False,
    stream: bool = True,
) -> dict[str, Any]:
    tools_map = allowed_tools if allowed_tools is not None else AVAILABLE_TOOLS
    tools = [
        definition
        for definition in TOOL_DEFINITIONS
        if definition["function"]["name"] in tools_map
    ]
    total_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    truncated = False
    final_content = ""
    voice_summary = ""
    voice_error: str | None = None
    started_at = time.monotonic()

    def outcome(status: str, error: str | None) -> dict[str, Any]:
        return {
            "content": final_content,
            "tokens": total_tokens,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "truncated": truncated,
            "latency_s": time.monotonic() - started_at,
            "status": status,
            "error": error,
            "voice_summary": voice_summary,
            "voice_error": voice_error,
        }

    def speak_voice_summary() -> None:
        nonlocal voice_summary, voice_error
        if not voice_enabled or not final_content:
            return
        try:
            voice_summary = _generate_voice_summary(
                final_content, model_name or MODEL_NAME
            )
        except ResponseError as error:
            voice_error = f"Ollama voice summary error: {error.error}"
            if show_output:
                print(f"\n[Voice Error] {voice_error}")
            return
        if not voice_summary:
            voice_error = "Ollama returned an empty voice summary."
            if show_output:
                print(f"\n[Voice Error] {voice_error}")
            return
        try:
            speak(voice_summary)
        except MemoryError as error:
            voice_error = f"Voice playback failed: {error}"
            if show_output:
                print(f"\n[Voice Error] {voice_error}")

    for step in range(max_steps):
        try:
            chat_options = {
                "model": model_name or MODEL_NAME,
                "messages": messages,
                "stream": stream,
                "tools": tools,
                "think": True,
                "options": options or CUSTOM_PARAMS,
            }
            if keep_alive is not None:
                chat_options["keep_alive"] = keep_alive
            response = chat(
                **chat_options,
            )
            if stream:
                message, turn_tokens = print_streamed(
                    response,
                    show_output=show_output,
                    show_thinking=True,
                )
            else:
                message, turn_tokens = print_response(
                    response,
                    show_output=show_output,
                )

            total_tokens += turn_tokens
            total_input_tokens += message["prompt_tokens"]
            total_output_tokens += message["output_tokens"]
            truncated = truncated or (
                message["done_reason"] == "length"
                or message["output_tokens"] >= (options or CUSTOM_PARAMS).get(
                    "num_predict", 0
                )
            )
            messages.append(
                {
                    key: value
                    for key, value in message.items()
                    if key not in {"prompt_tokens", "output_tokens", "done_reason"}
                }
            )
        except ResponseError as error:
            err_msg = f"Ollama ResponseError: {error.error}"
            if show_output:
                print(f"\n[Error] {err_msg}")
            return outcome("error", err_msg)
        except Exception as error:
            err_msg = f"Unexpected model stream error: {error}"
            if show_output:
                print(f"\n[Error] {err_msg}")
            return outcome("error", err_msg)

        final_content = message.get("content") or ""
        if total_tokens >= max_tokens:
            if not message.get("tool_calls"):
                speak_voice_summary()
            if show_output:
                print(f"Total Token Used: {total_tokens} (cap: {max_tokens})")
            return outcome("token_limit", f"Total token cap of {max_tokens} reached.")
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            speak_voice_summary()
            return outcome("success", None)
        if step + 1 >= max_steps:
            return outcome("max_steps", f"Agent step cap of {max_steps} reached.")

        for tool_call in tool_calls:
            if show_output:
                print_tool_call(tool_call)
            function = tool_call.get("function", {})
            fn_name = function.get("name")
            if fn_name not in tools_map:
                err_msg = f"Unknown tool requested: {fn_name}"
                if show_output:
                    print(f"  [Tool Error] {err_msg}")
                return outcome("error", err_msg)
            try:
                arguments = _tool_arguments(function.get("arguments", {}))
                result = tools_map[fn_name](**arguments)
                if show_tool_response:
                    print(f"  [Tool Result] {result}")
            except Exception as error:
                err_msg = f"Tool '{fn_name}' execution failed: {error}"
                if show_output:
                    print(f"  [Tool Error] {err_msg}")
                return outcome("error", err_msg)
            messages.append({"role": "tool", "content": str(result), "token": 0})
        if show_output:
            print(f"TOKEN-COST: {turn_tokens}")
    return outcome("max_steps", f"Agent step cap of {max_steps} reached.")
