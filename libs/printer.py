from libs.chunk_msg import print_typewriter


def print_streamed(stream, delay: float = 5 / 1e3) -> tuple[dict, int]:
    """Consumes an Ollama stream, prints tokens live, and returns (message_dict, token_count)."""
    final_text = ""
    tool_calls = []
    tokens_used = 0

    for chunk in stream:
        message = chunk.get("message", {})

        content = message.get("content") or ""
        final_text += content
        tool_calls.extend(message.get("tool_calls") or [])

        if content:
            print_typewriter(content, delay=delay)

        # Ollama attaches token counts to the final chunk (done=True)
        if chunk.get("done"):
            tokens_used += (chunk.get("prompt_eval_count") or 0) + (chunk.get("eval_count") or 0)

    print()
    return (
        {
            "role": "assistant",
            "content": final_text,
            "tool_calls": tool_calls,
        },
        tokens_used,
    )


def print_tool_call(tool_call: dict) -> None:
    fn_name = tool_call["function"]["name"]
    fn_args = tool_call["function"].get("arguments", {})
    print(f"  [Tool] {fn_name}({fn_args})")


def print_batch_prompt_header(index: int, total: int, prompt_text: str) -> None:
    print(f"\n[{index}/{total}] Prompt: {prompt_text}")
    print("(AI)>> ", end="")


def print_batch_summary(results: list[dict], grand_total_tokens: int) -> None:
    print("\n" + "=" * 60)
    print("BATCH EXECUTION TOKEN SUMMARY")
    print("=" * 60)
    for idx, item in enumerate(results, start=1):
        status_tag = "[OK]" if item.get("status") == "success" else "[ERR]"
        preview = item["prompt"][:40] + ("..." if len(item["prompt"]) > 40 else "")
        print(f"  {idx}. {status_tag:<5} {preview:<43} | {item['token_cost']} tokens")
    print("-" * 60)
    succeeded = sum(1 for r in results if r.get("status") == "success")
    print(f"  Completed              : {succeeded}/{len(results)} prompts succeeded")
    print(f"  Total Token Cost       : {grand_total_tokens} tokens")
    print("=" * 60 + "\n")