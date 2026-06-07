"""
tools/mcp_client.py

Async MCP client wrapper for Snowflake — FastMCP in-process transport.

ARCHITECTURE:
  FastAPI -> LangGraph -> SnowflakeMCPClient (in-process FastMCP)
          -> FastMCP Server (tools/snowflake_mcp_server.py)
          -> snowflake-connector-python -> Snowflake

The client imports the FastMCP server object directly and calls tools in-process
via fastmcp.Client — no HTTP, no SSE, no bearer token required.

ENV VARS (consumed by the server, not the client directly):
  SNOWFLAKE_ACCOUNT    e.g. VEIDJBV-BR57195  (or full .snowflakecomputing.com form)
  SNOWFLAKE_USER
  SNOWFLAKE_PASSWORD
  SNOWFLAKE_WAREHOUSE
  SNOWFLAKE_DATABASE   default: TPCH_DATA_PRODUCT
  SNOWFLAKE_SCHEMA     default: ANALYTICS
  SNOWFLAKE_ROLE       (optional)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastmcp import Client
from tools.snowflake_mcp_server import mcp as _snowflake_mcp

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


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class MCPConnectionError(Exception):
    """Raised when the in-process FastMCP client cannot reach the server."""


class MCPToolError(Exception):
    """Raised when an MCP tool call returns an error or unexpected payload."""


class MCPQueryError(MCPToolError):
    """Raised specifically for SQL execution failures via MCP."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ToolInfo:
    """Metadata about a single MCP tool exposed by the server."""

    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Structured result of an execute_query MCP call."""

    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    sql: str = ""
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "sql": self.sql,
            "truncated": self.truncated,
        }


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------


class SnowflakeMCPClient:
    """
    Async client for the local FastMCP Snowflake server (in-process transport).

    Every public method opens a fresh in-process FastMCP session, performs the
    operation, and closes the session.  Connection details (account, user,
    password, etc.) are read from environment variables by the server module at
    import time — the client itself needs no credentials.

    Usage
    -----
    client = SnowflakeMCPClient.from_env()
    tools  = await client.list_available_tools()
    result = await client.execute_query("SELECT CURRENT_DATE()")
    """

    def __init__(self) -> None:
        _info("mcp_client.init", transport="fastmcp_in_process")

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "SnowflakeMCPClient":
        """
        Construct a client.  All Snowflake credentials are read from env
        by the server (tools/snowflake_mcp_server.py); the client itself
        requires no arguments.
        """
        return cls()

    # ------------------------------------------------------------------
    # snowflake_db_schema — backward compat for metadata_discovery_node
    # ------------------------------------------------------------------

    @property
    def snowflake_db_schema(self) -> tuple:
        """
        Return (database, schema) from environment variables.

        Kept for backward compatibility with metadata_discovery_node which
        calls ``db, schema = client.snowflake_db_schema``.
        """
        db = os.environ.get("SNOWFLAKE_DATABASE", "").upper()
        schema = os.environ.get("SNOWFLAKE_SCHEMA", "").upper()
        return db, schema

    # ------------------------------------------------------------------
    # Internal tool-call helper
    # ------------------------------------------------------------------

    async def _call_tool(self, tool_name: str, args: dict) -> Any:
        """
        Call a named tool on the in-process FastMCP server.

        Opens a fresh Client session for each call, extracts text from the
        first content item, JSON-parses if possible, and returns the result.
        Raises MCPToolError on errors.
        """
        _info("mcp_client.call_tool", tool=tool_name, args=args)
        try:
            async with Client(_snowflake_mcp) as client:
                result = await client.call_tool(tool_name, args)
        except MCPToolError:
            raise
        except Exception as exc:
            raise MCPToolError(
                f"Tool call '{tool_name}' failed: {exc}"
            ) from exc

        # FastMCP 3.x returns CallToolResult with .content list and .is_error bool
        if getattr(result, "is_error", False):
            err_text = ""
            for item in getattr(result, "content", []):
                if hasattr(item, "text"):
                    err_text = item.text or ""
                    break
            raise MCPToolError(f"Tool '{tool_name}' returned an error: {err_text}")

        content = getattr(result, "content", None)
        if not content:
            _warning("mcp_client.call_tool.empty_result", tool=tool_name)
            return None

        # Extract text from the first TextContent item
        first = content[0]
        if hasattr(first, "text"):
            text: Optional[str] = first.text
        else:
            _warning(
                "mcp_client.call_tool.no_text_attr",
                tool=tool_name,
                content_type=type(first).__name__,
            )
            return first

        if text is None:
            return None

        # Guard: surface obvious error strings rather than passing them as data
        _TOOL_ERROR_PREFIXES = (
            "MCP error",
            "Error parsing",
            "Error:",
            "Execution Error",
            "SQL compilation error",
        )
        stripped = text.strip()
        if any(stripped.startswith(p) for p in _TOOL_ERROR_PREFIXES):
            raise MCPToolError(
                f"Tool '{tool_name}' returned an error response: {stripped[:300]}"
            )

        # Attempt JSON decode; fall back to raw string
        try:
            parsed = json.loads(text)
            _info(
                "mcp_client.call_tool.success",
                tool=tool_name,
                result_type=type(parsed).__name__,
            )
            return parsed
        except json.JSONDecodeError:
            _info(
                "mcp_client.call_tool.success_raw",
                tool=tool_name,
                text_length=len(text),
            )
            return text

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    async def list_available_tools(self) -> List[ToolInfo]:
        """
        Query the in-process FastMCP server for all registered tools.
        Returns a list of ToolInfo dataclasses.
        """
        _info("mcp_client.list_available_tools")
        try:
            async with Client(_snowflake_mcp) as client:
                tools_result = await client.list_tools()
        except Exception as exc:
            raise MCPConnectionError(f"list_tools failed: {exc}") from exc

        tool_infos: List[ToolInfo] = []
        for tool in tools_result:
            # FastMCP 3.x Tool objects use .inputSchema (MCP spec field name)
            schema: Dict[str, Any] = {}
            for attr in ("inputSchema", "parameters", "input_schema"):
                raw = getattr(tool, attr, None)
                if raw and isinstance(raw, dict):
                    schema = raw
                    break
            tool_infos.append(
                ToolInfo(
                    name=tool.name,
                    description=getattr(tool, "description", "") or "",
                    input_schema=schema,
                )
            )

        _info(
            "mcp_client.list_available_tools.done",
            count=len(tool_infos),
            names=[t.name for t in tool_infos],
        )
        return tool_infos

    # ------------------------------------------------------------------
    # Metadata methods
    # ------------------------------------------------------------------

    async def list_databases(self) -> List[str]:
        """List all databases accessible to the configured Snowflake user."""
        _info("mcp_client.list_databases")
        result = await self._call_tool("list_databases", {})
        return self._extract_string_list(result, context="list_databases")

    async def list_schemas(self, database: str) -> List[str]:
        """List schemas in a given database."""
        _info("mcp_client.list_schemas", database=database)
        result = await self._call_tool("list_schemas", {"database": database})
        return self._extract_string_list(result, context="list_schemas")

    async def list_tables(self, database: str, schema: str) -> List[str]:
        """List tables in a given database and schema."""
        _info("mcp_client.list_tables", database=database, schema=schema)
        result = await self._call_tool(
            "list_tables", {"database": database, "schema": schema}
        )
        return self._extract_string_list(result, context="list_tables")

    async def describe_table(
        self, database: str, schema: str, table: str
    ) -> Dict[str, Any]:
        """
        Describe a table (columns, types, etc.).
        Returns a dict with at least a 'columns' key.
        """
        _info(
            "mcp_client.describe_table",
            database=database,
            schema=schema,
            table=table,
        )
        result = await self._call_tool(
            "describe_table",
            {"database": database, "schema": schema, "table": table},
        )
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"raw": result}
        return {"raw": result}

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    async def execute_query(
        self, sql: str, max_rows: int = 1000
    ) -> QueryResult:
        """
        Execute a SQL query via the in-process FastMCP server and return a
        QueryResult dataclass.

        Parameters
        ----------
        sql:
            The SQL statement to execute.
        max_rows:
            Maximum number of rows to return (passed to the server tool;
            a LIMIT clause is also appended when appropriate).
        """
        _info("mcp_client.execute_query", sql_preview=sql[:200])
        guarded_sql = self._apply_row_limit(sql, max_rows)
        try:
            result = await self._call_tool(
                "execute_query", {"sql": guarded_sql, "max_rows": max_rows}
            )
        except MCPToolError as exc:
            raise MCPQueryError(
                f"Query execution failed: {exc}\nSQL: {guarded_sql}"
            ) from exc
        return self._parse_query_result(result, original_sql=sql)

    # ------------------------------------------------------------------
    # Full metadata discovery
    # ------------------------------------------------------------------

    async def discover_metadata(
        self,
        database: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Walk the metadata tree: databases -> schemas -> tables -> columns.

        When ``database`` and/or ``schema`` are provided they are passed to the
        server tool so discovery can be scoped to a specific context.  When
        omitted the server uses its configured defaults.

        Returns a nested dict::

            {
                "database_name": {
                    "schema_name": {
                        "table_name": [{"name": ..., "type": ...}, ...]
                    }
                }
            }
        """
        _info(
            "mcp_client.discover_metadata.start",
            database=database,
            schema=schema,
        )
        args: Dict[str, Any] = {}
        if database is not None:
            args["database"] = database
        if schema is not None:
            args["schema"] = schema

        result = await self._call_tool("discover_metadata", args)

        if isinstance(result, dict):
            _info(
                "mcp_client.discover_metadata.done",
                databases=len(result),
            )
            return result

        _warning(
            "mcp_client.discover_metadata.unexpected_type",
            result_type=type(result).__name__,
        )
        return {}

    # ------------------------------------------------------------------
    # Private parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_string_list(result: Any, *, context: str = "") -> List[str]:
        """
        Normalise a raw MCP result to a flat list of strings.
        MCP servers may return lists, dicts with a 'result'/'data' key, etc.
        """
        if result is None:
            return []
        if isinstance(result, list):
            out: List[str] = []
            for item in result:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    for key in (
                        "name", "NAME", "database_name", "schema_name",
                        "table_name", "value",
                    ):
                        if key in item:
                            out.append(str(item[key]))
                            break
                    else:
                        for v in item.values():
                            if isinstance(v, str):
                                out.append(v)
                                break
            return out
        if isinstance(result, dict):
            for key in (
                "result", "data", "items", "values", "rows",
                "databases", "schemas", "tables",
            ):
                if key in result:
                    return SnowflakeMCPClient._extract_string_list(
                        result[key], context=context
                    )
            _warning(
                "mcp_client.extract_string_list.unexpected_dict",
                context=context,
                keys=list(result.keys()),
            )
            return []
        if isinstance(result, str):
            lines = [ln.strip() for ln in result.splitlines() if ln.strip()]
            return lines
        return []

    @staticmethod
    def _extract_columns(desc: Any) -> List[Dict[str, Any]]:
        """
        Extract a list of column dicts from a describe_table result.
        Handles multiple common shapes.
        """
        if desc is None:
            return []
        if isinstance(desc, list):
            return desc
        if isinstance(desc, dict):
            for key in ("columns", "fields", "schema", "result"):
                if key in desc and isinstance(desc[key], list):
                    return desc[key]
        return []

    @staticmethod
    def _parse_query_result(raw: Any, original_sql: str = "") -> QueryResult:
        """
        Convert the raw execute_query result to a QueryResult dataclass.

        Handles the shapes the FastMCP Snowflake server may produce:

        Shape 1  – standard:     {"columns": [...], "rows": [[...]]}
        Shape 2  – fields:       {"data": [...], "fields": [...]}
        Shape 3  – columnNames:  {"columnNames": [...], "rows": [[...]]}
        Shape 4  – result list:  {"result": [...]}
        Shape 5  – list of dicts: [{"COL": "val", ...}, ...]
        Shape 6  – raw string (re-parsed recursively)
        """
        if raw is None:
            return QueryResult(columns=[], rows=[], row_count=0, sql=original_sql)

        if isinstance(raw, str):
            try:
                reparsed = json.loads(raw)
                return SnowflakeMCPClient._parse_query_result(reparsed, original_sql)
            except (json.JSONDecodeError, ValueError):
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                if len(lines) == 1:
                    return QueryResult(
                        columns=["result"], rows=[[raw]], row_count=1, sql=original_sql
                    )
                return QueryResult(
                    columns=["result"],
                    rows=[[ln] for ln in lines],
                    row_count=len(lines),
                    sql=original_sql,
                )

        if isinstance(raw, dict):
            columns: List[str] = []
            rows: List[List[Any]] = []

            # Shape 3: columnNames + rows
            if "columnNames" in raw and "rows" in raw:
                columns = [str(c) for c in raw["columnNames"]]
                rows = raw["rows"]

            # Shape 1: columns + rows
            elif "columns" in raw and "rows" in raw:
                col_raw = raw["columns"]
                if col_raw and isinstance(col_raw[0], dict):
                    columns = [
                        c.get("name", c.get("NAME", str(i)))
                        for i, c in enumerate(col_raw)
                    ]
                else:
                    columns = [str(c) for c in col_raw]
                rows = raw["rows"]

            # Shape 2: fields + data
            elif "data" in raw and "fields" in raw:
                columns = [
                    f.get("name", f.get("label", str(i)))
                    if isinstance(f, dict) else str(f)
                    for i, f in enumerate(raw["fields"])
                ]
                rows = raw["data"]

            # Shape 4: result list
            elif "result" in raw and isinstance(raw["result"], list):
                inner = raw["result"]
                if inner and isinstance(inner[0], dict):
                    columns = list(inner[0].keys())
                    rows = [list(r.values()) for r in inner]
                else:
                    columns = []
                    rows = inner

            else:
                _warning(
                    "mcp_client.parse_query_result.unknown_dict_shape",
                    keys=list(raw.keys()),
                    raw_preview=str(raw)[:300],
                )
                return QueryResult(
                    columns=[], rows=[], row_count=0, sql=original_sql
                )

            # Rows may be dicts (name->value) rather than lists
            if rows and isinstance(rows[0], dict):
                if not columns:
                    columns = list(rows[0].keys())
                rows = [list(r.values()) for r in rows]

            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                sql=original_sql,
            )

        if isinstance(raw, list):
            if not raw:
                return QueryResult(
                    columns=[], rows=[], row_count=0, sql=original_sql
                )
            if isinstance(raw[0], dict):
                columns = list(raw[0].keys())
                rows = [list(r.values()) for r in raw]
            else:
                columns = []
                rows = [[item] for item in raw]
            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                sql=original_sql,
            )

        # Scalar result
        return QueryResult(
            columns=["result"],
            rows=[[raw]],
            row_count=1,
            sql=original_sql,
        )

    @staticmethod
    def _apply_row_limit(sql: str, max_rows: int) -> str:
        """
        Append a LIMIT clause to SELECT statements that don't already have one.
        Best-effort guard; the server tool also enforces max_rows.
        """
        stripped = sql.strip().rstrip(";")
        upper = stripped.upper()
        is_select = upper.lstrip().startswith("SELECT")
        has_limit = "LIMIT" in upper
        if is_select and not has_limit:
            return f"{stripped} LIMIT {max_rows}"
        return sql
