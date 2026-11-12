import json
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
) -> dict[str, Any]:
    """Run a bounded Ollama tool loop and return content, usage, status, and errors."""
    tools_map = allowed_tools if allowed_tools is not None else AVAILABLE_TOOLS
    tools = [definition for definition in TOOL_DEFINITIONS
             if definition["function"]["name"] in tools_map]
    total_tokens = 0
    final_content = ""

    for step in range(max_steps):
        try:
            stream = chat(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
                tools=tools,
                think=False,
                options=CUSTOM_PARAMS,
            )
            message, turn_tokens = print_streamed(stream)
            total_tokens += turn_tokens
            messages.append(message)
        except ResponseError as error:
            err_msg = f"Ollama ResponseError: {error.error}"
            print(f"\n[Error] {err_msg}")
            return {
                "content": final_content,
                "tokens": total_tokens,
                "status": "error",
                "error": err_msg,
            }
        except Exception as error:
            err_msg = f"Unexpected model stream error: {error}"
            print(f"\n[Error] {err_msg}")
            return {
                "content": final_content,
                "tokens": total_tokens,
                "status": "error",
                "error": err_msg,
            }

        final_content = message.get("content") or ""
        if total_tokens >= max_tokens:
            return {
                "content": final_content,
                "tokens": total_tokens,
                "status": "token_limit",
                "error": f"Total token cap of {max_tokens} reached.",
            }
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return {
                "content": final_content,
                "tokens": total_tokens,
                "status": "success",
                "error": None,
            }
        if step + 1 >= max_steps:
            return {
                "content": final_content,
                "tokens": total_tokens,
                "status": "max_steps",
                "error": f"Agent step cap of {max_steps} reached.",
            }

        for tool_call in tool_calls:
            print_tool_call(tool_call)
            function = tool_call.get("function", {})
            fn_name = function.get("name")
            if fn_name not in tools_map:
                err_msg = f"Unknown tool requested: {fn_name}"
                print(f"  [Tool Error] {err_msg}")
                return {
                    "content": final_content,
                    "tokens": total_tokens,
                    "status": "error",
                    "error": err_msg,
                }
            try:
                arguments = _tool_arguments(function.get("arguments", {}))
                result = tools_map[fn_name](**arguments)
            except Exception as error:
                err_msg = f"Tool '{fn_name}' execution failed: {error}"
                print(f"  [Tool Error] {err_msg}")
                return {
                    "content": final_content,
                    "tokens": total_tokens,
                    "status": "error",
                    "error": err_msg,
                }
            messages.append({"role": "tool", "content": str(result)})

    return {
        "content": final_content,
        "tokens": total_tokens,
        "status": "max_steps",
        "error": f"Agent step cap of {max_steps} reached.",
    }
