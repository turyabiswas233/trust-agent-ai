from libs.chunk_msg import MarkdownTypewriter


def print_response(response, show_output: bool = False) -> tuple[dict, int]:
    """Normalize and display one non-streamed Ollama response."""
    response_data = (
        response.model_dump()
        if hasattr(response, "model_dump")
        else response
    )
    response_message = response_data.get("message", {})
    prompt_tokens = response_data.get("prompt_eval_count") or 0
    output_tokens = response_data.get("eval_count") or 0
    tokens_used = prompt_tokens + output_tokens
    content = response_message.get("content") or ""

    if content and show_output:
        print(content)

    return (
        {
            "role": response_message.get("role") or "assistant",
            "content": content,
            "tool_calls": response_message.get("tool_calls") or [],
            "token": tokens_used,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "done_reason": response_data.get("done_reason") or "",
        },
        tokens_used,
    )


def print_streamed(stream, show_output: bool = False) -> tuple[dict, int]:
    """Consumes an Ollama stream, prints tokens live with smooth typing and Rich Markdown, and returns (message_dict, token_count)."""
    final_text = ""
    tool_calls = []
    tokens_used = 0
    prompt_tokens = 0
    output_tokens = 0
    done_reason = ""

    # Set delay=0 if MarkdownTypewriter iterates char-by-char internally
    with MarkdownTypewriter(delay=0) as writer:
        for chunk in stream:
            message = chunk.get("message", {})
            content = message.get("content") or ""
            final_text += content
            tool_calls.extend(message.get("tool_calls") or [])

            if content and show_output:
                writer.write(content)

            # Ollama attaches token counts to the final chunk (done=True)
            if chunk.get("done"):
                done_reason = chunk.get("done_reason") or ""
                prompt_count = chunk.get("prompt_eval_count") or 0
                output_count = chunk.get("eval_count") or 0
                prompt_tokens += prompt_count
                output_tokens += output_count
                tokens_used += prompt_count + output_count

        print()

    return (
        {
            "role": "assistant",
            "content": final_text,
            "tool_calls": tool_calls,
            "token": tokens_used,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "done_reason": done_reason,
        },
        tokens_used,
    )



def print_tool_call(tool_call: dict) -> None:
    fn_name = tool_call["function"]["name"]
    fn_args = tool_call["function"].get("arguments", {})
    print(f"  [Tool] {fn_name}({fn_args})")
