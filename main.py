# ollama run model-name

# global libs
from ollama import chat

#  local libs | tools
from libs.model_config import AVAILABLE_TOOLS, TOOLS_LIST, MODEL_NAME, SYSTEM_PROMPT, CUSTOM_PARAMS
from libs.chunk_msg import print_typewriter


def print_streamed(stream) -> dict:
    final_text = ''
    tool_calls = []


    for chunk in stream:
        message = chunk['message']

        # Accumulate streamed content instead of overwriting it
        content = message.get('content') or ''
        final_text += content

        tool_calls.extend(message.get('tool_calls') or [])

        if content:
            print_typewriter(content,delay=5/1e3)

    print()
            

    return {
        'role': 'assistant',
        'content': final_text,
        'tool_calls': tool_calls,
    }

def ollama_chat(messages: list) -> None:
 
    while True:
        stream = chat(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            tools=TOOLS_LIST,
            think=True,
            options=CUSTOM_PARAMS
        )
        message = print_streamed(stream)
        messages.append(message)

        if not message.get("tool_calls"):
            break

    
        for tool_call in message["tool_calls"]:
            fn_name = tool_call["function"]["name"]
            fn_args = tool_call["function"].get("arguments", {})

            for x in message["tool_calls"]:
                print(x)
                

            if fn_name in AVAILABLE_TOOLS:
                result = AVAILABLE_TOOLS[fn_name](**fn_args)
            else:
                result = f"Unknown tool: {fn_name}"

            tool_result = {"role": "tool", "content": str(result)}
            messages.append(tool_result)


    
    return


if __name__ == '__main__':
    conversation: list = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        print(">> ", end="")
        input_text = input().strip()

        if input_text.endswith("\\q") or input_text.lower().endswith("bye"):
            conversation.append({"role": "user", "content": input_text})
            ollama_chat(conversation)
            break

        conversation.append({"role": "user", "content": input_text})
        
        print("(AI)>> ", end="")
        ollama_chat(conversation)
        