from tools.calculator import calculate
from tools.sql_tools import execute_sql, generate_sql, get_db_schema
from tools.times import get_current_time
from tools.files import read_file_text

# MODEL_NAME = "qwen3.5:4b"
MODEL_NAME = "qwen3.5:9b"
CUSTOM_PARAMS = {
    "temperature": 0.85,
    "top_k": 16,
    "top_p": 0.9,
    "num_ctx": 4096,
}
MAX_AGENT_STEPS = 8
MAX_TOTAL_TOKENS = 40960

AVAILABLE_TOOLS = {
    "get_current_time": get_current_time,
    "read_file_text": read_file_text,
    "get_db_schema": get_db_schema,
    "generate_sql": generate_sql,
    "execute_sql": execute_sql,
    "calculate": calculate,
}

TOOL_DEFINITIONS = [
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

SYSTEM_PROMPT = """You are Trust AI, a helpful assistant that answers questions about a local SQLite store database.
Use get_db_schema to inspect the available tables when needed. Write SQL with generate_sql,
then run it with execute_sql. Only read data; the SQL tools reject modifications. Use calculate
for arithmetic that is easier to verify separately. Base answers on query results, state units
when relevant, and do not invent missing data. Keep responses concise."""
