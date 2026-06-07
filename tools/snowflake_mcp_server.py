"""
tools/snowflake_mcp_server.py

Local FastMCP server that wraps Snowflake operations using snowflake-connector-python.

This replaces the old Snowflake Managed MCP (remote HTTP endpoint) architecture:

  OLD: FastAPI -> LangGraph -> SnowflakeMCPClient (HTTP/Streamable) -> Snowflake Managed MCP
  NEW: FastAPI -> LangGraph -> SnowflakeMCPClient (in-process FastMCP) -> FastMCP Server -> Snowflake

Usage (standalone for testing):
    python tools/snowflake_mcp_server.py

Required environment variables:
    SNOWFLAKE_ACCOUNT     e.g. VEIDJBV-BR57195 or VEIDJBV-BR57195.snowflakecomputing.com
    SNOWFLAKE_USER        Snowflake username
    SNOWFLAKE_PASSWORD    Snowflake password
    SNOWFLAKE_WAREHOUSE   Snowflake virtual warehouse name
    SNOWFLAKE_DATABASE    Default database (e.g. TPCH_DATA_PRODUCT)
    SNOWFLAKE_SCHEMA      Default schema   (e.g. ANALYTICS)
    SNOWFLAKE_ROLE        (optional) Snowflake role to assume

Removed environment variables (no longer needed):
    MCP_SERVER_URL        (was remote Snowflake Managed MCP endpoint)
    MCP_BEARER_TOKEN      (was bearer token for remote endpoint)
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import snowflake.connector
from fastmcp import FastMCP
from snowflake.connector import DictCursor
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

try:
    import structlog

    _log = structlog.get_logger(__name__)
    _USE_STRUCTLOG = True
except ImportError:
    _USE_STRUCTLOG = False
    _log = logger  # type: ignore[assignment]


def _info(event: str, **kw: Any) -> None:
    if _USE_STRUCTLOG:
        _log.info(event, **kw)
    else:
        logger.info("%s | %s", event, json.dumps(kw, default=str))


def _warning(event: str, **kw: Any) -> None:
    if _USE_STRUCTLOG:
        _log.warning(event, **kw)
    else:
        logger.warning("%s | %s", event, json.dumps(kw, default=str))


def _error(event: str, **kw: Any) -> None:
    if _USE_STRUCTLOG:
        _log.error(event, **kw)
    else:
        logger.error("%s | %s", event, json.dumps(kw, default=str))


# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------

mcp = FastMCP("Snowflake Analytics MCP")

# ---------------------------------------------------------------------------
# Snowflake connection helper
# ---------------------------------------------------------------------------

_TRANSIENT_EXCEPTIONS = (
    snowflake.connector.errors.OperationalError,
    snowflake.connector.errors.DatabaseError,
    snowflake.connector.errors.InterfaceError,
)


def _get_connection_params() -> Dict[str, Any]:
    """
    Build Snowflake connector keyword arguments from environment variables.

    The SNOWFLAKE_ACCOUNT value may include the full .snowflakecomputing.com
    suffix; the connector accepts either form, but we strip the suffix to keep
    it normalised (the connector docs recommend the bare account identifier).
    """
    account = os.environ.get("SNOWFLAKE_ACCOUNT", "")
    # Strip trailing .snowflakecomputing.com if present
    account = re.sub(r"\.snowflakecomputing\.com$", "", account, flags=re.IGNORECASE)

    params: Dict[str, Any] = {
        "account": account,
        "user": os.environ.get("SNOWFLAKE_USER", ""),
        "password": os.environ.get("SNOWFLAKE_PASSWORD", ""),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", ""),
        "database": os.environ.get("SNOWFLAKE_DATABASE", ""),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA", ""),
    }

    role = os.environ.get("SNOWFLAKE_ROLE", "")
    if role:
        params["role"] = role

    missing = [k for k, v in params.items() if not v and k != "role"]
    if missing:
        raise ValueError(
            f"Missing required Snowflake environment variables: "
            f"{[f'SNOWFLAKE_{k.upper()}' for k in missing]}"
        )

    return params


@contextmanager
def _get_connection() -> Generator[snowflake.connector.SnowflakeConnection, None, None]:
    """
    Context manager that opens a fresh Snowflake connection and always closes it.

    A new connection is created per tool call for simplicity and safety —
    no connection pooling is attempted so there are no stale-session issues.
    """
    conn: Optional[snowflake.connector.SnowflakeConnection] = None
    params = _get_connection_params()
    _info(
        "snowflake.connect",
        account=params["account"],
        user=params["user"],
        database=params["database"],
        schema=params["schema"],
    )
    try:
        conn = snowflake.connector.connect(**params)
        yield conn
    except Exception:
        _error("snowflake.connect.failed", account=params.get("account"))
        raise
    finally:
        if conn is not None:
            try:
                conn.close()
                _info("snowflake.connection.closed")
            except Exception as close_exc:
                _warning("snowflake.connection.close_error", error=str(close_exc))


# ---------------------------------------------------------------------------
# Tenacity retry decorator for transient Snowflake errors
# ---------------------------------------------------------------------------

_retry_on_transient = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_TRANSIENT_EXCEPTIONS),
    reraise=True,
)


def _run_with_retry(fn, *args, **kwargs):
    """
    Execute a callable with tenacity retry applied for transient Snowflake errors.

    tenacity's @retry decorator does not compose cleanly with FastMCP's
    decorator stack, so we use this thin helper instead of decorating each
    tool directly.
    """
    decorated = _retry_on_transient(fn)
    return decorated(*args, **kwargs)


# ---------------------------------------------------------------------------
# Internal SQL helpers
# ---------------------------------------------------------------------------

_SELECT_RE = re.compile(r"^\s*(?:WITH|SELECT)\b", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


def _apply_row_limit(sql: str, max_rows: int) -> str:
    """Append LIMIT to bare SELECT statements that have none."""
    stripped = sql.strip().rstrip(";")
    if _SELECT_RE.match(stripped) and not _LIMIT_RE.search(stripped):
        return f"{stripped} LIMIT {max_rows}"
    return sql


def _assert_select(sql: str) -> None:
    """Raise ValueError if sql does not start with SELECT (case-insensitive)."""
    if not _SELECT_RE.match(sql.strip()):
        raise ValueError(
            "Only SELECT statements are permitted via execute_query. "
            f"Received: {sql[:120]!r}"
        )


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def execute_query(sql: str, max_rows: int = 1000) -> dict:
    """
    Execute a SELECT query against Snowflake and return the results.

    Parameters
    ----------
    sql:
        A SELECT SQL statement. Non-SELECT statements raise ValueError.
    max_rows:
        Maximum rows returned. A LIMIT clause is appended automatically
        when the statement is a bare SELECT without one.

    Returns
    -------
    dict with keys:
        columns   – list of column name strings
        rows      – list of rows; each row is a list of values
        row_count – number of rows returned
        truncated – True when a synthetic LIMIT was applied
    """
    _assert_select(sql)
    guarded_sql = _apply_row_limit(sql, max_rows)
    truncated = guarded_sql != sql.strip().rstrip(";")

    _info("mcp_tool.execute_query", sql_preview=sql[:200], max_rows=max_rows)

    def _run():
        with _get_connection() as conn:
            with conn.cursor(DictCursor) as cur:
                cur.execute(guarded_sql)
                raw_rows = cur.fetchall()

        if not raw_rows:
            return {"columns": [], "rows": [], "row_count": 0, "truncated": truncated}

        columns = list(raw_rows[0].keys())
        rows = [list(row.values()) for row in raw_rows]
        _info(
            "mcp_tool.execute_query.done",
            row_count=len(rows),
            col_count=len(columns),
        )
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
        }

    return _run_with_retry(_run)


@mcp.tool()
def list_databases() -> list[str]:
    """
    Return the names of all databases accessible to the current Snowflake role.

    Uses SHOW DATABASES and returns a plain list of database name strings.
    """
    _info("mcp_tool.list_databases")

    def _run():
        with _get_connection() as conn:
            with conn.cursor(DictCursor) as cur:
                cur.execute("SHOW DATABASES")
                rows = cur.fetchall()

        # SHOW DATABASES returns rows with a "name" column (case may vary)
        names: List[str] = []
        for row in rows:
            # DictCursor keys may be upper or lower case depending on connector version
            name = row.get("name") or row.get("NAME") or ""
            if name:
                names.append(str(name))

        _info("mcp_tool.list_databases.done", count=len(names))
        return names

    return _run_with_retry(_run)


@mcp.tool()
def list_schemas(database: str) -> list[str]:
    """
    Return all schema names in the given database (excluding INFORMATION_SCHEMA).

    Parameters
    ----------
    database:
        The Snowflake database name (case-insensitive).
    """
    _info("mcp_tool.list_schemas", database=database)

    def _run():
        sql = (
            f"SELECT SCHEMA_NAME "
            f"FROM {database}.INFORMATION_SCHEMA.SCHEMATA "
            f"WHERE SCHEMA_NAME != 'INFORMATION_SCHEMA' "
            f"ORDER BY SCHEMA_NAME"
        )
        with _get_connection() as conn:
            with conn.cursor(DictCursor) as cur:
                cur.execute(sql)
                rows = cur.fetchall()

        names = [
            str(row.get("SCHEMA_NAME") or row.get("schema_name") or "")
            for row in rows
            if row.get("SCHEMA_NAME") or row.get("schema_name")
        ]
        _info("mcp_tool.list_schemas.done", database=database, count=len(names))
        return names

    return _run_with_retry(_run)


@mcp.tool()
def list_tables(database: str, schema: str) -> list[str]:
    """
    Return all table and view names in the given database.schema.

    Excludes the INFORMATION_SCHEMA schema itself.

    Parameters
    ----------
    database:
        The Snowflake database name.
    schema:
        The schema name within that database.
    """
    _info("mcp_tool.list_tables", database=database, schema=schema)

    def _run():
        sql = (
            f"SELECT TABLE_NAME "
            f"FROM {database}.INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA = '{schema}' "
            f"  AND TABLE_TYPE IN ('BASE TABLE', 'VIEW') "
            f"  AND TABLE_SCHEMA != 'INFORMATION_SCHEMA' "
            f"ORDER BY TABLE_NAME"
        )
        with _get_connection() as conn:
            with conn.cursor(DictCursor) as cur:
                cur.execute(sql)
                rows = cur.fetchall()

        names = [
            str(row.get("TABLE_NAME") or row.get("table_name") or "")
            for row in rows
            if row.get("TABLE_NAME") or row.get("table_name")
        ]
        _info(
            "mcp_tool.list_tables.done",
            database=database,
            schema=schema,
            count=len(names),
        )
        return names

    return _run_with_retry(_run)


@mcp.tool()
def describe_table(database: str, schema: str, table: str) -> list[dict]:
    """
    Return column metadata for the given table or view.

    Parameters
    ----------
    database:
        The Snowflake database name.
    schema:
        The schema name.
    table:
        The table or view name.

    Returns
    -------
    list of dicts, one per column, ordered by ORDINAL_POSITION:
        name     – column name
        type     – SQL data type string
        nullable – True / False
        comment  – column comment (may be empty string)
    """
    _info(
        "mcp_tool.describe_table",
        database=database,
        schema=schema,
        table=table,
    )

    def _run():
        sql = (
            f"SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COMMENT "
            f"FROM {database}.INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_SCHEMA = '{schema}' "
            f"  AND TABLE_NAME = '{table}' "
            f"ORDER BY ORDINAL_POSITION"
        )
        with _get_connection() as conn:
            with conn.cursor(DictCursor) as cur:
                cur.execute(sql)
                rows = cur.fetchall()

        columns: List[Dict[str, Any]] = []
        for row in rows:
            col_name = row.get("COLUMN_NAME") or row.get("column_name") or ""
            data_type = row.get("DATA_TYPE") or row.get("data_type") or ""
            is_nullable_raw = row.get("IS_NULLABLE") or row.get("is_nullable") or "NO"
            comment = row.get("COMMENT") or row.get("comment") or ""

            is_nullable = str(is_nullable_raw).upper() in ("YES", "TRUE", "1")
            columns.append(
                {
                    "name": str(col_name),
                    "type": str(data_type),
                    "nullable": is_nullable,
                    "comment": str(comment),
                }
            )

        _info(
            "mcp_tool.describe_table.done",
            database=database,
            schema=schema,
            table=table,
            col_count=len(columns),
        )
        return columns

    return _run_with_retry(_run)


@mcp.tool()
def discover_metadata(
    database: Optional[str] = None,
    schema: Optional[str] = None,
) -> dict:
    """
    Walk the metadata tree for a database/schema and return a nested dict.

    If database or schema are not provided, the values from the environment
    variables SNOWFLAKE_DATABASE and SNOWFLAKE_SCHEMA are used as defaults.

    Discovery is capped at 200 tables total to prevent runaway calls.

    Parameters
    ----------
    database:
        The Snowflake database to inspect. Defaults to SNOWFLAKE_DATABASE env var.
    schema:
        A specific schema to inspect. When omitted, all non-INFORMATION_SCHEMA
        schemas in the database are walked.

    Returns
    -------
    Nested dict:
        {
            "<database>": {
                "<schema>": {
                    "<table>": [
                        {"name": "COL1", "type": "TEXT", "nullable": True, "comment": ""},
                        ...
                    ]
                }
            }
        }
    """
    resolved_db = database or os.environ.get("SNOWFLAKE_DATABASE", "")
    resolved_schema = schema or os.environ.get("SNOWFLAKE_SCHEMA", "")

    if not resolved_db:
        raise ValueError(
            "database must be provided or SNOWFLAKE_DATABASE must be set in the environment."
        )

    _info(
        "mcp_tool.discover_metadata.start",
        database=resolved_db,
        schema=resolved_schema or "<all>",
    )

    metadata: Dict[str, Any] = {resolved_db: {}}

    # Determine which schemas to walk
    if resolved_schema:
        schemas_to_walk = [resolved_schema]
    else:
        try:
            schemas_to_walk = list_schemas(resolved_db)
        except Exception as exc:
            _error(
                "mcp_tool.discover_metadata.list_schemas_failed",
                database=resolved_db,
                error=str(exc),
            )
            return metadata

    _info(
        "mcp_tool.discover_metadata.schemas",
        database=resolved_db,
        schema_count=len(schemas_to_walk),
    )

    total_tables = 0
    _TABLE_CAP = 200

    for sch in schemas_to_walk:
        metadata[resolved_db][sch] = {}

        if total_tables >= _TABLE_CAP:
            _warning(
                "mcp_tool.discover_metadata.table_cap_reached",
                cap=_TABLE_CAP,
                schema=sch,
            )
            break

        try:
            tables = list_tables(resolved_db, sch)
        except Exception as exc:
            _warning(
                "mcp_tool.discover_metadata.list_tables_failed",
                database=resolved_db,
                schema=sch,
                error=str(exc),
            )
            continue

        for tbl in tables:
            if total_tables >= _TABLE_CAP:
                _warning(
                    "mcp_tool.discover_metadata.table_cap_reached",
                    cap=_TABLE_CAP,
                    table=tbl,
                )
                break

            try:
                cols = describe_table(resolved_db, sch, tbl)
            except Exception as exc:
                _warning(
                    "mcp_tool.discover_metadata.describe_table_failed",
                    database=resolved_db,
                    schema=sch,
                    table=tbl,
                    error=str(exc),
                )
                cols = []

            metadata[resolved_db][sch][tbl] = cols
            total_tables += 1

    _info(
        "mcp_tool.discover_metadata.done",
        database=resolved_db,
        total_tables=total_tables,
    )
    return metadata


# ---------------------------------------------------------------------------
# Standalone entry point (for testing / manual inspection)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _info("snowflake_mcp_server.starting", transport="stdio")
    mcp.run()
