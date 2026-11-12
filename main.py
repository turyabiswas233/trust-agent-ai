from libs.model_config import SYSTEM_PROMPT
from libs.ollama_engine import run_agent_turn


def main() -> None:
    conversation: list = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        print(">> ", end="")
        input_text = input().strip()
        if not input_text:
            continue

        conversation.append({"role": "user", "content": input_text})
        print("(AI)>> ", end="")
        run_agent_turn(conversation)

        if input_text.endswith("\\q") or input_text.lower().endswith("bye"):
            break


if __name__ == "__main__":
    main()