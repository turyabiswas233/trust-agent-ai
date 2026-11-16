import argparse
from pathlib import Path
import sys
from time import sleep
from libs.bulk_prompt import run_bulk_dataset
from libs.model_config import SYSTEM_PROMPT, MODEL_NAME
from libs.ollama_engine import run_agent_turn, has_model

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
    parser.add_argument("--voice-enabled", type=str, default="Y")
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream the model response (use --no-stream to wait for the full response).",
    )
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
            stream=args.stream,
        )
        print(f"Detailed responses: {result_path}")
        print(f"Comparison report: {report_path}")
        return

    if not has_model(args.model):
        return

    start_text = f"Selected model: {args.model if args.model else MODEL_NAME}"
    
    for char in start_text:
        sys.stdout.write(char)
        sys.stdout.flush()
        sleep(0.01)
    print()


    conversation: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            input_text = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not input_text:
            continue
        if input_text.lower() in {"/q", "quit", "bye", "exit"}:
            for ctx in conversation:
                if ctx["role"]=='assistant':
                    print(f"(token): {ctx['token']}")
            break

        conversation.append({"role": "user", "content": input_text})
        print("(AI)>> ", end="")
        outcome = run_agent_turn(
            conversation,
            model_name=args.model,
            max_steps=5,
            voice_enabled=(args.voice_enabled == "Y"),
            stream=args.stream,
        )
        if outcome["status"] != "success": 
            print(f"[{outcome['status']}] {outcome['error']}")


if __name__ == "__main__":
    main()