from ollama import chat, ResponseError
from libs.model_config import AVAILABLE_TOOLS, MODEL_NAME, CUSTOM_PARAMS
from libs.printer import print_streamed, print_tool_call


def run_agent_turn(messages: list, allowed_tools: dict | None = None) -> dict:
    """
    Runs a multi-step Ollama chat + tool-calling loop.
    Breaks cleanly on ResponseError or tool execution failures without crashing the app.
    Returns {"content": str, "tokens": int, "error": str | None}.
    """
    tools_map = allowed_tools if allowed_tools is not None else AVAILABLE_TOOLS
    tools_list = list(tools_map.values())
    total_tokens = 0
    final_content = ""

    while True:
        try:
            stream = chat(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
                tools=tools_list,
                think=False,
                options=CUSTOM_PARAMS,
            )
            message, turn_tokens = print_streamed(stream)
            total_tokens += turn_tokens
            messages.append(message)

        except ResponseError as e:
            err_msg = f"Ollama ResponseError: {e.error}"
            print(f"\n[Error] {err_msg} — skipping remaining steps for this prompt.")
            return {"content": "", "tokens": total_tokens, "error": err_msg}

        except Exception as e:
            err_msg = f"Unexpected stream error: {e}"
            print(f"\n[Error] {err_msg} — skipping remaining steps for this prompt.")
            return {"content": "", "tokens": total_tokens, "error": err_msg}

        if not message.get("tool_calls"):
            final_content = message.get("content") or ""
            break

        for tool_call in message["tool_calls"]:
            print_tool_call(tool_call)
            fn_name = tool_call["function"]["name"]
            fn_args = tool_call["function"].get("arguments", {})

            if fn_name not in tools_map:
                err_msg = f"Unknown tool requested: {fn_name}"
                print(f"  [Tool Error] {err_msg}")
                return {"content": "", "tokens": total_tokens, "error": err_msg}

            try:
                result = tools_map[fn_name](**fn_args)
            except Exception as e:
                err_msg = f"Tool '{fn_name}' execution failed: {e}"
                print(f"  [Tool Error] {err_msg} — breaking prompt execution.")
                return {"content": "", "tokens": total_tokens, "error": err_msg}

            messages.append({"role": "tool", "content": str(result)})

    return {"content": final_content, "tokens": total_tokens, "error": None}