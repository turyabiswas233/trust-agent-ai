from libs.chunk_msg import MarkdownTypewriter 


def print_streamed(stream, delay: float = 5 / 1e3) -> tuple[dict, int]:
    """Consumes an Ollama stream, prints tokens live with smooth typing and Rich Markdown, and returns (message_dict, token_count)."""
    final_text = ""
    tool_calls = []
    tokens_used = 0

    with MarkdownTypewriter(delay=delay) as writer:
        for chunk in stream:
            message = chunk.get("message", {})

            content = message.get("content") or ""
            final_text += content
            tool_calls.extend(message.get("tool_calls") or [])

            if content:
                writer.write(content) 

            # Ollama attaches token counts to the final chunk (done=True)
            if chunk.get("done"):
                tokens_used += (chunk.get("prompt_eval_count") or 0) + (chunk.get("eval_count") or 0)

    print()
    return (
        {
            "role": "assistant",
            "content": final_text,
            "tool_calls": tool_calls,
            "token": tokens_used
        },
        tokens_used,
    )


def print_tool_call(tool_call: dict) -> None:
    fn_name = tool_call["function"]["name"]
    fn_args = tool_call["function"].get("arguments", {})
    print(f"  [Tool] {fn_name}({fn_args})")

