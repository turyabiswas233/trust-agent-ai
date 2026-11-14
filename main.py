from libs.model_config import SYSTEM_PROMPT
from libs.ollama_engine import run_agent_turn

def main() -> None:
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
                
                print(f"[{ctx["role"]}] -> (token): {ctx['token']}")
            break

        conversation.append({"role": "user", "content": input_text})
        print("(AI)>> ", end="")
        outcome = run_agent_turn(conversation)
        if outcome["status"] != "success": 
            print(f"[{outcome['status']}] {outcome['error']}")


if __name__ == "__main__":
    main()