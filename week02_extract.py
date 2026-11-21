"""Week 02 — Structured field extraction from 50 domain notices.

Compares prompt-only (Method A) vs grammar-constrained (Method B) decoding
across 3 runs each. All calls go through the class 1 wrapper for cost tracking.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from libs.bulk_prompt import calculate_cost
from libs.model_config import CUSTOM_PARAMS, MODEL_NAME
from libs.ollama_engine import run_agent_turn

NOTICES_DIR = Path("data/domain_notices")
SCHEMA_PATH = Path("schemas/notice_schema_v1.json")
OUTPUT_DIR = Path("output/week02")

EXTRACTION_PROMPT = """You are a structured data extraction assistant.

Extract the following fields from the domain notice below and return ONLY valid JSON with these exact keys:

{
  "schema_version": 1,
  "notice_id": "<string: the notice/reference ID, e.g. DN-001>",
  "domain_name": "<string: the fully qualified domain name>",
  "registrar": "<string: the registrar name>",
  "notice_type": "<string: one of expiration_warning, renewal_notice, transfer_request, suspension_alert, whois_verification, dnssec_alert>",
  "notice_date": "<string: date issued in YYYY-MM-DD format>",
  "action_required": <boolean: true if action is required, false otherwise>,
  "deadline": "<string: deadline in YYYY-MM-DD format, or null if none>",
  "amount_due_usd": <number: amount in USD, or null if no payment required>
}

Rules:
- schema_version must always be 1.
- notice_type must be exactly one of: expiration_warning, renewal_notice, transfer_request, suspension_alert, whois_verification, dnssec_alert.
- Dates must be in YYYY-MM-DD format.
- If there is no deadline, set deadline to null.
- If there is no amount due, set amount_due_usd to null.
- Return ONLY the JSON object, no markdown, no explanation, no extra text.

--- DOMAIN NOTICE ---
{notice_text}
--- END NOTICE ---
"""


def load_notices() -> list[dict[str, str]]:
    """Load all 50 domain notice text files."""
    notices = []
    for path in sorted(NOTICES_DIR.glob("notice_*.txt")):
        text = path.read_text(encoding="utf-8")
        notices.append({"file": path.name, "text": text})
    return notices


def load_schema() -> dict[str, Any]:
    """Load the JSON schema for grammar-constrained decoding."""
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def run_extraction(
    notice_text: str,
    model_name: str,
    options: dict[str, Any],
    schema: dict[str, Any] | None = None,
    keep_alive: int = 300,
) -> dict[str, Any]:
    """Run a single extraction through the class 1 wrapper.

    Args:
        notice_text: The raw domain notice text.
        model_name: Ollama model tag.
        options: Generation options (temperature, seed, etc.).
        schema: If provided, passed as format= for grammar-constrained decoding.
        keep_alive: Ollama keep_alive setting.

    Returns:
        Dict with raw_output, input_tokens, output_tokens, cost_usd, latency_s, status, error.
    """
    prompt = EXTRACTION_PROMPT.format(notice_text=notice_text)
    messages = [{"role": "user", "content": prompt}]

    result = run_agent_turn(
        messages,
        allowed_tools={},
        max_steps=1,
        model_name=model_name,
        show_output=False,
        options=options,
        keep_alive=keep_alive,
        stream=False,
        thinking=False,
        format=schema,
    )

    input_tokens = int(result.get("input_tokens", 0))
    output_tokens = int(result.get("output_tokens", 0))

    return {
        "raw_output": result.get("content", ""),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": calculate_cost(input_tokens, output_tokens),
        "latency_s": float(result.get("latency_s", 0)),
        "status": result.get("status", "unknown"),
        "error": result.get("error"),
    }


def run_all_extractions(
    notices: list[dict[str, str]],
    model_name: str,
    schema: dict[str, Any],
    num_runs: int = 3,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run Method A and Method B extractions across all notices and runs.

    Returns list of extraction records.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "extractions_raw.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    methods = [
        ("prompt_only", None),
        ("grammar_constrained", schema),
    ]

    total_calls = len(notices) * len(methods) * num_runs
    call_num = 0

    with output_path.open("w", encoding="utf-8") as f:
        for method_name, method_schema in methods:
            for run_idx in range(1, num_runs + 1):
                seed = 42 + run_idx
                options = {
                    **CUSTOM_PARAMS,
                    "temperature": 0.2,
                    "seed": seed,
                    "num_predict": 512,
                    "num_ctx": 4096,
                }

                for notice in notices:
                    call_num += 1
                    print(
                        f"[{call_num}/{total_calls}] "
                        f"{method_name} run={run_idx} {notice['file']}"
                    )

                    result = run_extraction(
                        notice["text"],
                        model_name,
                        options,
                        schema=method_schema,
                    )

                    record = {
                        "method": method_name,
                        "run": run_idx,
                        "notice_file": notice["file"],
                        **result,
                    }
                    all_records.append(record)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

    print(f"\nAll {total_calls} extractions saved to {output_path}")
    return all_records


def run_extraction_pipeline(
    model_name: str | None = None,
    runs: int = 3,
    dry_run: bool = False,
    output_dir: Path = OUTPUT_DIR,
) -> list[dict[str, Any]]:
    model = model_name or MODEL_NAME
    notices = load_notices()
    schema = load_schema()

    if not notices:
        raise FileNotFoundError("No notices found in data/domain_notices/")

    print(f"Loaded {len(notices)} notices, schema v1, model={model}")

    if dry_run:
        notices = notices[:2]
        runs = 1
        print("DRY RUN: 2 notices, 1 run per method (4 calls total)")

    output_path = output_dir / "extractions_raw.jsonl"
    records = run_all_extractions(notices, model, schema, runs, output_path)

    total_cost = sum(r["cost_usd"] for r in records)
    total_tokens = sum(r["input_tokens"] + r["output_tokens"] for r in records)
    print(f"\nTotal extractions: {len(records)}")
    print(f"Total tokens:      {total_tokens}")
    print(f"Total cost:        ${total_cost:.6f}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=None, help="Ollama model tag (defaults to model_config)."
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per method.")
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on first 2 notices only, 1 run, for quick validation.",
    )
    args = parser.parse_args()
    run_extraction_pipeline(
        model_name=args.model,
        runs=args.runs,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
