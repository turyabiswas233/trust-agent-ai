import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "store.db"
MAX_RESULT_ROWS = 100
_WRITE_KEYWORDS = re.compile(
    r"\b(ATTACH|DETACH|ALTER|CREATE|DELETE|DROP|INSERT|REINDEX|REPLACE|"
    r"SAVEPOINT|UPDATE|VACUUM|BEGIN|COMMIT|ROLLBACK|PRAGMA)\b",
    re.IGNORECASE,
)


def _validate_select(sql: str) -> str:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL query must be a non-empty string.")

    statement = sql.strip()
    if statement.endswith(";"):
        statement = statement[:-1].rstrip()
    if ";" in statement:
        raise ValueError("Only one SQL statement is allowed.")
    if not re.match(r"^(SELECT|WITH)\b", statement, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")
    if _WRITE_KEYWORDS.search(statement):
        raise ValueError("The query contains a disallowed SQL keyword.")
    return statement


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else DB_PATH
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def get_db_schema() -> str:
    """Return user tables, columns, and primary-key markers as JSON."""
    with closing(_connect()) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        schema: dict[str, list[dict[str, Any]]] = {}
        for table_row in tables:
            table_name = table_row["name"]
            columns = connection.execute(
                f'PRAGMA table_info("{table_name.replace(chr(34), chr(34) * 2)}")'
            ).fetchall()
            schema[table_name] = [
                {
                    "name": column["name"],
                    "type": column["type"],
                    "primary_key": bool(column["pk"]),
                }
                for column in columns
            ]
    return json.dumps(schema, ensure_ascii=False)


def generate_sql(sql: str) -> str:
    """Validate and return a read-only SELECT query for the execute_sql tool."""
    return _validate_select(sql)


def execute_sql(sql: str) -> str:
    """Execute one read-only SQLite SELECT and return a bounded JSON result."""
    statement = _validate_select(sql)
    with closing(_connect()) as connection:
        cursor = connection.execute(statement)
        if cursor.description is None:
            raise ValueError("The query did not return rows.")
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)

    truncated = len(rows) > MAX_RESULT_ROWS
    rows = rows[:MAX_RESULT_ROWS]
    return json.dumps(
        {
            "columns": columns,
            "rows": [list(row) for row in rows],
            "row_count": len(rows),
            "truncated": truncated,
        },
        ensure_ascii=False,
        default=str,
    )
