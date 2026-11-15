import re
from libs.chunk_msg import MarkdownTypewriter


def print_streamed(
    stream,
    delay: float = 5 / 1e3,
    show_output: bool = True,
) -> tuple[dict, int]:
    """Consumes an Ollama stream, prints tokens live with smooth typing and Rich Markdown, and returns (message_dict, token_count)."""
    final_text = ""
    tool_calls = []
    tokens_used = 0
    prompt_tokens = 0
    output_tokens = 0
    done_reason = ""
    
    # Buffer to hold incomplete Bangla words/conjuncts
    bangla_buffer = ""
    # Boundary pattern: whitespace or common Bangla/English punctuation
    boundary_pattern = re.compile(r"([\s।,?!.;:\-\n]+)")

    # Set delay=0 if MarkdownTypewriter iterates char-by-char internally
    with MarkdownTypewriter(delay=0) as writer:
        for chunk in stream:
            message = chunk.get("message", {})

            content = message.get("content") or ""
            final_text += content
            tool_calls.extend(message.get("tool_calls") or [])

            if content and show_output:
                bangla_buffer += content
                
                # Split buffer whenever a space or punctuation mark arrives
                parts = boundary_pattern.split(bangla_buffer)
                
                # Everything except the last element is a complete word + delimiter
                if len(parts) > 1:
                    complete_chunk = "".join(parts[:-1])
                    bangla_buffer = parts[-1]
                    writer.write(complete_chunk)

            # Ollama attaches token counts to the final chunk (done=True)
            if chunk.get("done"):
                done_reason = chunk.get("done_reason") or ""
                prompt_count = chunk.get("prompt_eval_count") or 0
                output_count = chunk.get("eval_count") or 0
                prompt_tokens += prompt_count
                output_tokens += output_count
                tokens_used += prompt_count + output_count

        # Flush any remaining text at the end of the stream
        if bangla_buffer and show_output:
            writer.write(bangla_buffer)

    if show_output:
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


# def print_streamed(
#     stream,
#     delay: float = 5 / 1e3,
#     show_output: bool = True,
# ) -> tuple[dict, int]:
#     """Consumes an Ollama stream, prints tokens live with smooth typing and Rich Markdown, and returns (message_dict, token_count)."""
#     final_text = ""
#     tool_calls = []
#     tokens_used = 0
#     prompt_tokens = 0
#     output_tokens = 0
#     done_reason = ""

#     with MarkdownTypewriter(delay=delay) as writer:
#         for chunk in stream:
#             message = chunk.get("message", {})

#             content = message.get("content") or ""
#             final_text += content
#             tool_calls.extend(message.get("tool_calls") or [])

#             if content and show_output:
#                 writer.write(content) 

#             # Ollama attaches token counts to the final chunk (done=True)
#             if chunk.get("done"):
#                 done_reason = chunk.get("done_reason") or ""
#                 prompt_count = chunk.get("prompt_eval_count") or 0
#                 output_count = chunk.get("eval_count") or 0
#                 prompt_tokens += prompt_count
#                 output_tokens += output_count
#                 tokens_used += prompt_count + output_count

#     if show_output:
#         print()
#     return (
#         {
#             "role": "assistant",
#             "content": final_text,
#             "tool_calls": tool_calls,
#             "token": tokens_used,
#             "prompt_tokens": prompt_tokens,
#             "output_tokens": output_tokens,
#             "done_reason": done_reason,
#         },
#         tokens_used,
#     )


def print_tool_call(tool_call: dict) -> None:
    fn_name = tool_call["function"]["name"]
    fn_args = tool_call["function"].get("arguments", {})
    print(f"  [Tool] {fn_name}({fn_args})")
