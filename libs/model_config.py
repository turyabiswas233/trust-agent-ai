from tools.calculator import calculate
from tools.sql_tools import execute_sql, generate_sql, get_db_schema
from tools.times import get_current_time
from tools.files import read_file_text, write_in_file
from tools.browser import open_url
from tools.auto_switch_voice import auto_switch_voice
from tools.list_voice_models import list_voice_models
from ollama import list as olist

models = olist().models
total_models_size = models.__len__()

model_id = 1

MODEL_NAME = models[model_id%total_models_size].model if models[model_id%total_models_size].model != None else "qwen2.5:3b"

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

SYSTEM_PROMPT = """You are TukTuk AI, a helpful assistant using en_US-kristin-medium voice service that answers questions about any software engineer related topic or database query.
Use get_db_schema to inspect the available tables when needed. Write SQL with generate_sql,
then run it with execute_sql. Only read data; the SQL tools reject modifications. Use calculate
for arithmetic that is easier to verify separately. Base answers on query results, state units
when relevant, and do not invent missing data. Keep responses concise.

IMPORTANT: If user asks to write any code as output, at first summarize what to do as a list, then write using write_in_file tool. When it is all about coding/debugging, think as much as possible

When choosing a language for a spoken response, call auto-switch-voice before
finishing: use 'bn' for Bengali/Bangla and 'en' for English. If the user asks
for a particular voice, use list_voice_models to find the available assistant
names and pass the requested name as voice_name. Keep their selected English
voice unless they request another one. Use list_voice_models when they ask
which voices are available.
"""

VOICE_SUMMARY_PROMPT = """Rewrite the assistant's answer as a concise, natural spoken narration.
Return only plain text to be spoken. Do not include Markdown, headings, bullet points,
tables, code, URLs, or labels. Preserve the important facts and numbers without adding
new information. Use the same language as the answer. If the answer contains code or
technical formatting, explain its key point in ordinary spoken language instead of
reading it aloud."""
