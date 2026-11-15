"""Run CSV prompt datasets through the existing Trust AI agent."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from libs.model_config import CUSTOM_PARAMS, MAX_TOTAL_TOKENS, SYSTEM_PROMPT
from libs.ollama_engine import run_agent_turn

RESULT_FIELDS = [
    "id",
    "category",
    "language",
    "model",
    "prompt",
    "output",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "latency_s",
    "truncated",
    "status",
    "error",
]
REPORT_FIELDS = [
    "id",
    "category",
    "en_input_tokens",
    "bn_input_tokens",
    "input_token_ratio",
    "en_output_tokens",
    "bn_output_tokens",
    "en_total_cost",
    "bn_total_cost",
    "total_cost_ratio",
    "en_latency_s",
    "bn_latency_s",
    "en_truncated",
    "bn_truncated",
    "en_status",
    "bn_status",
]
LANGUAGE_NAMES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "bn": "bn",
    "ben": "bn",
    "bengali": "bn",
    "bangla": "bn",
}


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts must be non-negative.")
    return (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000


def _read_dataset(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        if not reader.fieldnames:
            raise ValueError(f"{path} must have a CSV header row.")
        rows = [
            {
                (key or "").strip().casefold(): (value or "").strip()
                for key, value in row.items()
            }
            for row in reader
        ]
    if not rows:
        raise ValueError(f"{path} contains no prompt rows.")
    return rows


def _prompt_items(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for row_number, row in enumerate(rows, start=1):
        pair_id = row.get("id") or str(row_number)
        category = row.get("category") or row.get("test_type") or ""

        if row.get("english") or row.get("bangla"):
            if not row.get("english") or not row.get("bangla"):
                raise ValueError(
                    f"Dataset row {row_number} must provide both english and bangla prompts."
                )
            items.extend(
                [
                    {
                        "id": pair_id,
                        "category": category,
                        "language": "en",
                        "prompt": row["english"],
                    },
                    {
                        "id": pair_id,
                        "category": category,
                        "language": "bn",
                        "prompt": row["bangla"],
                    },
                ]
            )
            continue

        prompt = row.get("prompt")
        if not prompt:
            raise ValueError(
                f"Dataset row {row_number} needs a prompt column, or english and bangla columns."
            )
        language = LANGUAGE_NAMES.get(row.get("language", "").casefold(), "und")
        items.append(
            {
                "id": pair_id,
                "category": category,
                "language": language,
                "prompt": prompt,
            }
        )
    return items


def _write_report(
    items: list[dict[str, str]],
    completed: dict[tuple[str, str], dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_language_pairs = any(
        (item["id"], "en") in completed and (item["id"], "bn") in completed
        for item in items
    )
    fields = REPORT_FIELDS if has_language_pairs else [
        "id",
        "category",
        "language",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "latency_s",
        "truncated",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fields)
        writer.writeheader()
        if has_language_pairs:
            pairs = dict.fromkeys((item["id"], item["category"]) for item in items)
            for pair_id, category in pairs:
                en = completed.get((pair_id, "en"))
                bn = completed.get((pair_id, "bn"))
                if en is None or bn is None:
                    continue
                en_cost = en["cost_usd"]
                bn_cost = bn["cost_usd"]
                writer.writerow(
                    {
                        "id": pair_id,
                        "category": category,
                        "en_input_tokens": en["input_tokens"],
                        "bn_input_tokens": bn["input_tokens"],
                        "input_token_ratio": (
                            f"{bn['input_tokens'] / en['input_tokens']:.8f}"
                            if en["input_tokens"]
                            else ""
                        ),
                        "en_output_tokens": en["output_tokens"],
                        "bn_output_tokens": bn["output_tokens"],
                        "en_total_cost": f"{en_cost:.12f}",
                        "bn_total_cost": f"{bn_cost:.12f}",
                        "total_cost_ratio": (
                            f"{bn_cost / en_cost:.8f}" if en_cost else ""
                        ),
                        "en_latency_s": f"{en['latency_s']:.6f}",
                        "bn_latency_s": f"{bn['latency_s']:.6f}",
                        "en_truncated": en["truncated"],
                        "bn_truncated": bn["truncated"],
                        "en_status": en["status"],
                        "bn_status": bn["status"],
                    }
                )
        else:
            for item in items:
                result = completed[(item["id"], item["language"])]
                writer.writerow(
                    {
                        "id": item["id"],
                        "category": item["category"],
                        "language": item["language"],
                        "input_tokens": result["input_tokens"],
                        "output_tokens": result["output_tokens"],
                        "cost_usd": f"{result['cost_usd']:.12f}",
                        "latency_s": f"{result['latency_s']:.6f}",
                        "truncated": result["truncated"],
                        "status": result["status"],
                    }
                )


def run_bulk_dataset(
    dataset_path: str | Path,
    model_name: str,
    result_path: str | Path = "result.csv",
    report_path: str | Path = "report.csv",
    max_output_tokens: int = 256,
) -> tuple[Path, Path]:
    if not model_name.strip():
        raise ValueError("An Ollama model name is required.")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive.")
    dataset_path = Path(dataset_path)
    result_path = Path(result_path)
    report_path = Path(report_path)
    if result_path.resolve() == report_path.resolve():
        raise ValueError("Result and report CSV paths must be different.")
    items = _prompt_items(_read_dataset(dataset_path))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    options = {
        **CUSTOM_PARAMS,
        "temperature": 0,
        "seed": 42,
        "num_predict": max_output_tokens,
    }
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    with result_path.open("w", encoding="utf-8", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for item in items:
            key = (item["id"], item["language"])
            if key in completed:
                raise ValueError(
                    f"Duplicate prompt for id={item['id']} language={item['language']}."
                )
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
            ]
            result = run_agent_turn(
                conversation,
                max_tokens=MAX_TOTAL_TOKENS,
                model_name=model_name,
                show_output=False,
                options=options,
                keep_alive=0,
            )
            input_tokens = int(result.get("input_tokens", 0))
            output_tokens = int(result.get("output_tokens", 0))
            record = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": calculate_cost(input_tokens, output_tokens),
                "latency_s": float(result.get("latency_s", 0)),
                "truncated": bool(result.get("truncated", False)),
                "status": result["status"],
            }
            completed[key] = record
            writer.writerow(
                {
                    **item,
                    "model": model_name,
                    "output": result.get("content", ""),
                    **record,
                    "cost_usd": f"{record['cost_usd']:.12f}",
                    "latency_s": f"{record['latency_s']:.6f}",
                    "error": result.get("error") or "",
                }
            )
            result_file.flush()
            print(f"Completed {item['id']} ({item['language']})")

    _write_report(items, completed, report_path)
    return result_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--result", type=Path, default=Path("result.csv"))
    parser.add_argument("--report", type=Path, default=Path("report.csv"))
    parser.add_argument("--max-output-tokens", type=int, default=256)
    args = parser.parse_args()
    result, report = run_bulk_dataset(
        args.dataset,
        args.model,
        result_path=args.result,
        report_path=args.report,
        max_output_tokens=args.max_output_tokens,
    )
    print(f"Detailed responses: {result}")
    print(f"Comparison report: {report}")


if __name__ == "__main__":
    main()
