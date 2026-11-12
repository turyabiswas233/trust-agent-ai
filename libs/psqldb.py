import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_FILE = _PROJECT_ROOT / ".env"
_LOCAL_ENV_FILE = _PROJECT_ROOT / ".env.local"
_connection: Any | None = None


def connect_to_postgres() -> str:
    """Connect using PGHOST, PGDATABASE, PGUSER, PGPASSWORD, and optional PGPORT."""
    global _connection

    env_file = _LOCAL_ENV_FILE if _LOCAL_ENV_FILE.is_file() else _ENV_FILE
    load_dotenv(env_file)
    required_variables = ("PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD")
    missing_variables = [
        variable for variable in required_variables if not os.environ.get(variable)
    ]
    if missing_variables:
        raise RuntimeError(
            "Set the PostgreSQL environment variables in the project .env.local "
            "(or .env) file: "
            + ", ".join(missing_variables)
        )

    try:
        import psycopg
    except ImportError as error:
        raise RuntimeError(
            "PostgreSQL support requires Psycopg 3. Install it with "
            "`python -m pip install \"psycopg[binary]\"`."
        ) from error

    connection = psycopg.connect(
        host=os.environ["PGHOST"],
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        port=int(os.environ.get("PGPORT", "5432")),
    )
    if _connection is not None:
        _connection.close()
    _connection = connection
    return "Connected to PostgreSQL."


def query_postgres(
    query: str,
    params: list[str | int | float | bool | None] | None = None,
) -> list[tuple[Any, ...]]:
    """Run a parameterized query on the connected database and return its rows."""
    if _connection is None:
        raise RuntimeError("Connect to PostgreSQL before querying data.")

    with _connection.cursor() as cursor:
        cursor.execute(query, params)
        if cursor.description is None:
            return []
        return cursor.fetchall()
