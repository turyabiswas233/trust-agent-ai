import json
import time
from typing import Any

from ollama import ResponseError, chat

from libs.model_config import (
    AVAILABLE_TOOLS,
    CUSTOM_PARAMS,
    MAX_AGENT_STEPS,
    MAX_TOTAL_TOKENS,
    MODEL_NAME,
    TOOL_DEFINITIONS,
)
from libs.printer import print_streamed, print_tool_call


def _tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return parsed


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
) -> dict[str, Any]:
    """Run a bounded Ollama tool loop and return content, usage, status, and errors."""
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
        }

    for step in range(max_steps):
        # print(f"Trying in [step]: {step}")
        try:
            chat_options = {
                "model": model_name or MODEL_NAME,
                "messages": messages,
                "stream": True,
                "tools": tools,
                "think": True,
                "options": options or CUSTOM_PARAMS,
            }
            if keep_alive is not None:
                chat_options["keep_alive"] = keep_alive
            stream = chat(
                **chat_options,
            )
            message, turn_tokens = print_streamed(stream, show_output=show_output)
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
            if show_output:
                print(final_content)
                print(f"Total Token Used: {total_tokens} (cap: {max_tokens})")
            return outcome("token_limit", f"Total token cap of {max_tokens} reached.")
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
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
