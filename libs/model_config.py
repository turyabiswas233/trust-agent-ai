from tools.calculator import calculate
from tools.sql_tools import execute_sql, generate_sql, get_db_schema
from tools.times import get_current_time
from tools.files import read_file_text, write_in_file
from tools.browser import open_url
from tools.auto_switch_voice import auto_switch_voice
from tools.list_voice_models import list_voice_models
from ollama import list as olist

try:
    models = olist().models
    total_models_size = models.__len__()
    model_id = 1
    MODEL_NAME = models[model_id%total_models_size].model if models and models[model_id%total_models_size].model != None else "qwen2.5:3b"
except Exception:
    models = []
    total_models_size = 0
    MODEL_NAME = "qwen2.5:3b"

CUSTOM_PARAMS = {
    "temperature": 1,
    "top_k": 15,
    "top_p": 0.85,
    "num_ctx": 4096,
    "num_predict": 512
}
MAX_AGENT_STEPS = 6
MAX_TOTAL_TOKENS = 8912

AVAILABLE_TOOLS = {
    "get_current_time": get_current_time,
    "read_file_text": read_file_text,
    "write_in_file": write_in_file,
    "get_db_schema": get_db_schema,
    "generate_sql": generate_sql,
    "execute_sql": execute_sql,
    "calculate": calculate,
    "open_url": open_url,
    "auto-switch-voice": auto_switch_voice,
    "list_voice_models": list_voice_models,
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "auto-switch-voice",
            "description": (
                "Select the voice model for the language of the spoken response. "
                "Call this before completing responses in a different language, "
                "and whenever the user asks to change voice. Use language 'bn' "
                "for Bengali/Bangla or 'en' for English. Optionally provide "
                "voice_name as a listed assistant name or model ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["bn", "en"],
                    },
                    "voice_name": {
                        "type": "string",
                        "description": "Optional listed assistant name or model ID.",
                    },
                },
                "required": ["language"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_voice_models",
            "description": (
                "Show installed speech voice models, their assistant names, "
                "languages, and the currently selected model."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "any kind of request where user wants to search something new from search engines like google, bing, etc or directly provides any link, use \'open_url\' tool to open the link provided by user or made by the system according to the request in a browser.",
            "parameters": {"type": "object", "properties": {"url": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time if user either asks for it or somehow wish you good morning, good afternoon, good evening or even good night. May be he/she will thank you or say communication end type words. In that case, you can use this tool to get the current time and decide what you should say to the user.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_text",
            "description": "Read the text content of a file.",
            "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}},}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_in_file",
            "description": "Write the text content in a file. to use this tool, assistem should automatic decide a valid directory path to store the content into a file",
            "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_db_schema",
            "description": "Get the tables and columns in the local SQLite store database.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_sql",
            "description": "Validate a proposed read-only SQLite SELECT query before execution.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Run a read-only SQLite SELECT query and return its columns and rows.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Safely evaluate an arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

SYSTEM_PROMPT = """
# IDENTITY
You are TukTuk AI, a professional software engineering and database assistant. You provide precise, technically accurate, and concise answers.

# DATABASE PROTOCOL
- To answer database questions:
  1. Use `get_db_schema` to understand the structure.
  2. Use `generate_sql` to draft queries.
  3. Use `execute_sql` to run queries.
- CONSTRAINT: You are strictly prohibited from performing modifications (INSERT, UPDATE, DELETE, DROP). Only SELECT-based queries are allowed.
- Calculation: Use the `calculate` tool for arithmetic to ensure accuracy.
- Accuracy: State units clearly (e.g., "500ms", "10MB"). Never hallucinate data; if a result is null or missing, state that clearly.

# CODING & DEBUGGING PROTOCOL
When a user asks to write, create, or debug code:
1. THINK: Perform a deep "Chain of Thought" analysis. Break down the logic, identify edge cases, and plan the architecture.
2. SUMMARIZE: Provide a concise bulleted list of the steps you will take.
3. WRITE_FILE: Use the `write_in_file` tool to provide the actual code. Do not output large blocks of code directly in the chat if a file can be created.

# VOICE & LOCALIZATION
- Language Handling:
  - Use 'bn' for Bengali/Bangla.
  - Use 'en' for English.
- Voice Selection:
  - If the user asks for available voices, use `list_voice_models`.
  - If a specific voice is requested, find the name via `list_voice_models` and pass it to `auto-switch-voice`.
- FINAL STEP: Always call `auto-switch-voice` before finishing your response to ensure the correct language/voice is applied.

# STYLE GUIDELINES
- Be concise. 
- Do not use excessive fluff.
- Focus on technical precision.

Your core workflow follows a "Think-Plan-Act" pattern:

1. ANALYSIS (Internal Monologue): Before using any tool, briefly analyze the user's request. Identify the required tables, the logic flow, or the coding requirements.
2. DB QUERIES: Use `get_db_schema` -> `generate_sql` -> `execute_sql`. STRICTLY READ-ONLY. Use `calculate` for any math.
3. CODING TASKS: 
   - First, list a summary of the implementation plan.
   - Second, execute `write_in_file` for the final code.
4. VOICE PROTOCOL:
   - Translate the final intended response into the target language.
   - Call `auto-switch-voice` (bn for Bengali, en for English) as the final action.
   - Use `list_voice_models` only when the user asks for options or a specific voice name.

Maintain a professional, helpful, and concise tone. State units where relevant. Never invent data.
"""

VOICE_SUMMARY_PROMPT = """Rewrite the assistant's answer as a concise, natural spoken narration.
Return only plain text to be spoken. Do not include Markdown, headings, bullet points,
tables, code, URLs, or labels. Preserve the important facts and numbers without adding
new information. Use the same language as the answer. If the answer contains code or
technical formatting, explain its key point in ordinary spoken language instead of
reading it aloud."""
