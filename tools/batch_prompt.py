import csv
import json
from libs.printer import print_batch_prompt_header, print_batch_summary


def run_csv_prompts(file_path: str, prompt_column: str = "prompt") -> str:
    """
    Reads a CSV file of English or Bangla prompts, streams each response live in the console,
    gracefully skips/records any prompt that throws an error, and returns JSON with results and token costs.
    """
    from libs.model_config import AVAILABLE_TOOLS, SYSTEM_PROMPT
    from libs.ollama_engine import run_agent_turn

    sub_tools_map = {k: v for k, v in AVAILABLE_TOOLS.items() if k != "run_csv_prompts"}

    with open(file_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and prompt_column not in reader.fieldnames:
            prompt_column = reader.fieldnames[0]

        prompts = [
            (row.get(prompt_column) or "").strip()
            for row in reader
            if (row.get(prompt_column) or "").strip()
        ]

    results = []
    grand_total_tokens = 0
    total_count = len(prompts)

    for idx, prompt_text in enumerate(prompts, start=1):
        print_batch_prompt_header(idx, total_count, prompt_text)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]

        turn_outcome = run_agent_turn(messages, allowed_tools=sub_tools_map)
        prompt_tokens = turn_outcome["tokens"]
        grand_total_tokens += prompt_tokens

        if turn_outcome["error"]:
            results.append({
                "prompt": prompt_text,
                "status": "failed",
                "result": None,
                "error": turn_outcome["error"],
                "token_cost": prompt_tokens,
            })
        else:
            results.append({
                "prompt": prompt_text,
                "status": "success",
                "result": turn_outcome["content"],
                "error": None,
                "token_cost": prompt_tokens,
            })

    print_batch_summary(results, grand_total_tokens)

    return json.dumps(
        {
            "total_prompts": total_count,
            "total_token_cost": grand_total_tokens,
            "results": results,
        },
        ensure_ascii=False,
        indent=2,
    )