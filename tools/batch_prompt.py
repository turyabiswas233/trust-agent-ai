import csv
import json
from ollama import chat

def run_csv_prompts(file_path: str, prompt_column: str = "prompt", model: str = "qwen3.5:4b") -> str:
    """
    Reads a CSV file containing English or Bangla prompts, runs each prompt through the LLM,
    and returns a JSON list of objects containing the result and total token cost.

    Args:
        file_path: Path to the input CSV file.
        prompt_column: Name of the column containing the prompts (defaults to 'prompt').
        model: Ollama model to run the prompts against (defaults to 'qwen3.5:4b').

    Returns:
        JSON string of [{"prompt": str, "result": str, "token_cost": int}, ...]
    """
    results = []

    with open(file_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        
        # Fallback to first column if specified column name isn't found
        if reader.fieldnames and prompt_column not in reader.fieldnames:
            prompt_column = reader.fieldnames[0]

        for row in reader:
            prompt_text = (row.get(prompt_column) or "").strip()
            if not prompt_text:
                continue

            # Non-streaming call returns exact token counts directly
            response = chat(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                stream=False
            )

            input_tokens = response.get("prompt_eval_count") or 0
            output_tokens = response.get("eval_count") or 0

            results.append({
                "prompt": prompt_text,
                "result": response["message"]["content"],
                "token_cost": input_tokens + output_tokens
            })

    return json.dumps(results, ensure_ascii=False, indent=2)