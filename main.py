import argparse
from pathlib import Path
import sys
from time import sleep
from libs.bulk_prompt import run_bulk_dataset
from libs.model_config import SYSTEM_PROMPT, MODEL_NAME, models, total_models_size
from libs.ollama_engine import run_agent_turn, has_model
from libs.week02_runner import run_week02_assignment, run_contract_tests

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Trust AI interactively, on a CSV dataset, or evaluate assignments.",
        epilog=(
            "Examples:\n"
            "  Interactive:       python main.py\n"
            "  Week 02 Benchmark: python main.py --week2\n"
            "  Week 02 Dry Run:   python main.py --week2 --dry-run\n"
            "  Contract Tests:    python main.py --test-contract\n"
            "  Bulk CSV:          python main.py --dataset dataset.csv --model qwen2.5:3b\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Week 02 Assignment options
    parser.add_argument(
        "--week2",
        action="store_true",
        help="Run Week 02 domain notices extraction, evaluation and report generation.",
    )
    parser.add_argument(
        "--assignment",
        type=int,
        choices=[1, 2],
        help="Run assignment: 1 for SQL agent eval, 2 for Week 02 structured extraction.",
    )
    parser.add_argument(
        "--test-contract",
        action="store_true",
        help="Run schema contract tests using pytest (additive passes, breaking fails).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per method for Week 02 benchmark (default: 3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="When running Week 02, extract 2 sample notices for 1 run.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip extraction and evaluate existing extractions_raw.jsonl.",
    )

    # General / interactive options
    parser.add_argument(
        "--dataset",
        type=Path,
        help="CSV file containing prompts to run in bulk.",
    )
    parser.add_argument("--model", help="Ollama model tag to use.")
    parser.add_argument("--result", type=Path, default=Path("result.csv"))
    parser.add_argument("--report", type=Path, default=Path("report.csv"))
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--voice", action=argparse.BooleanOptionalAction, default=False, help="use --no-voice to not to use voice assistant service.")
    parser.add_argument("--think",  action=argparse.BooleanOptionalAction, default=True, help="use --no-think to stop thinking.")
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream the model response (use --no-stream to wait for the full response).",
    )
    args = parser.parse_args()

    # 1. Handle schema contract tests only
    if args.test_contract and not (args.week2 or args.assignment == 2):
        run_contract_tests()
        return

    # 2. Handle Assignment 1
    if args.assignment == 1:
        from evaluate import run_evaluation as run_sql_eval
        print("Running Assignment 1 SQL Agent evaluation...")
        summary = run_sql_eval()
        import json
        print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
        return

    # 3. Handle Assignment 2 / Week 2
    if args.week2 or args.assignment == 2:
        run_week02_assignment(
            model_name=args.model,
            runs=args.runs,
            dry_run=args.dry_run,
            evaluate_only=args.evaluate_only,
            test_contract=args.test_contract or not args.evaluate_only,
        )
        return

    # 4. Handle bulk CSV dataset
    if args.dataset:
        if not args.model:
            parser.error("--model is required when --dataset is provided.")
        result_path, report_path = run_bulk_dataset(
            dataset_path=args.dataset,
            model_name= args.model,
            result_path=args.result,
            report_path=args.report,
            max_output_tokens=args.max_output_tokens,
            stream=args.stream
        )
        print(f"Detailed responses: {result_path}")
        print(f"Comparison report: {report_path}")
        return

    if not has_model(args.model):
        print("No valid LLM Model is chosen. Exiting...")
        sleep(1)
        return
    
    print(f"available Models ({total_models_size}): ")
    i = 1
    for model in models:
        print(f"{i}. {model.model}")
        i=i+1
    print(f"Selected model: {args.model if args.model else MODEL_NAME}")


    conversation: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            input_text = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not input_text:
            continue
        if input_text.lower() in {"/week2", "/eval-week2", "week2"}:
            run_week02_assignment(model_name=args.model, runs=args.runs)
            continue
        if input_text.lower() in {"/test-contract", "test-contract"}:
            run_contract_tests()
            continue
        if input_text.lower() in {"/q", "quit", "bye", "exit"}:
            conversation.append({"role": "user", "content": input_text})
            print("(AI)>> ", end="")
            outcome = run_agent_turn(
                conversation,
                model_name=args.model,
                voice_enabled=args.voice,
                stream=args.stream,
                show_tool_response=False
            )
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
            voice_enabled=args.voice,
            keep_alive=600,
            stream=args.stream,
            show_tool_response=False,
            thinking=args.think
        )
        if outcome["status"] != "success": 
            print(f"[{outcome['status']}] {outcome['error']}")


if __name__ == "__main__":
    main()