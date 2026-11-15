import argparse
from pathlib import Path

from libs.bulk_prompt import run_bulk_dataset
from libs.model_config import SYSTEM_PROMPT
from libs.ollama_engine import run_agent_turn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Trust AI interactively or on a CSV dataset.",
        epilog=(
            "Bulk example: python main.py --dataset dataset.csv "
            "--model qwen3.5:9b"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="CSV file containing prompts to run in bulk.",
    )
    parser.add_argument("--model", help="Ollama model tag to use.")
    parser.add_argument("--result", type=Path, default=Path("result.csv"))
    parser.add_argument("--report", type=Path, default=Path("report.csv"))
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    args = parser.parse_args()

    if args.dataset:
        if not args.model:
            parser.error("--model is required when --dataset is provided.")
        result_path, report_path = run_bulk_dataset(
            args.dataset,
            args.model,
            result_path=args.result,
            report_path=args.report,
            max_output_tokens=args.max_output_tokens,
        )
        print(f"Detailed responses: {result_path}")
        print(f"Comparison report: {report_path}")
        return

    conversation: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            input_text = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not input_text:
            continue
        if input_text.lower() in {"/q", "/quit", "bye"}:
            for ctx in conversation:
                if ctx["role"]=='assistant':
                    print(f"(token): {ctx['token']}")
            break

        conversation.append({"role": "user", "content": input_text})
        print("(AI)>> ", end="")
        outcome = run_agent_turn(conversation, model_name=args.model, max_steps=5)
        if outcome["status"] != "success": 
            print(f"[{outcome['status']}] {outcome['error']}")


if __name__ == "__main__":
    main()