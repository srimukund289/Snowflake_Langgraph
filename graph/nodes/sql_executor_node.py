"""
graph/nodes/sql_executor_node.py

Executes validated SQL through the Snowflake MCP client and stores the
structured result in AgentState.query_results.

Responsibilities
----------------
1. Short-circuit if a prior node has already set state["error"].
2. Verify that sql_validator_node produced a valid SQLValidationResult.
3. Execute the SQL via SnowflakeMCPClient (SSE/MCP transport — no direct
   Snowflake connector used).
4. Convert the MCP client's QueryResult into graph.state.QueryResult.
5. Log timing, warn on 0-row results and slow queries (> 30 s).
6. Return partial state update: query_results, current_node, execution_logs.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from graph.state import AgentState, ExecutionLog, QueryResult
from tools.mcp_client import MCPConnectionError, MCPQueryError, SnowflakeMCPClient

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_NODE_NAME = "sql_executor_node"
_SLOW_QUERY_THRESHOLD_S = 30.0
_MAX_ROWS = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _make_log(
    status: str,
    message: str,
    duration_ms: float = 0.0,
    metadata: Dict[str, Any] | None = None,
) -> ExecutionLog:
    return ExecutionLog(
        timestamp=_now_iso(),
        node=_NODE_NAME,
        status=status,
        message=message,
        duration_ms=duration_ms,
        metadata=metadata or {},
    )


def _mcp_result_to_state(
    mcp_result: Any,
    sql: str,
    execution_time_ms: float,
) -> QueryResult:
    """
    Convert a tools.mcp_client.QueryResult into a graph.state.QueryResult.

    The MCP client stores rows as List[List[Any]]; the state's QueryResult
    uses data as List[Dict[str, Any]] (column-name keyed rows).
    """
    # mcp_result is tools.mcp_client.QueryResult
    columns: List[str] = mcp_result.columns or []
    raw_rows: List[List[Any]] = mcp_result.rows or []

    # Build list-of-dicts; fall back to index-keyed dicts if columns missing
    if columns:
        data = [
            {columns[i] if i < len(columns) else str(i): v
             for i, v in enumerate(row)}
            for row in raw_rows
        ]
    else:
        data = [{str(i): v for i, v in enumerate(row)} for row in raw_rows]

    return QueryResult(
        success=True,
        row_count=mcp_result.row_count,
        data=data,
        error=None,
        columns=columns,
        execution_time_ms=execution_time_ms,
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def sql_executor_node(state: AgentState) -> dict:
    """
    LangGraph node: execute generated SQL via Snowflake MCP.

    Returns a partial-state dict; query_results and execution_logs are
    Annotated[List, operator.add] fields so LangGraph appends them.
    """
    start_ts = time.time()

    start_log = _make_log(
        status="started",
        message="SQL executor node started",
        metadata={"sql_preview": (state.get("generated_sql") or "")[:200]},
    )

    # ------------------------------------------------------------------
    # 1. Short-circuit: propagate upstream error
    # ------------------------------------------------------------------
    if state.get("error"):
        elapsed_ms = (time.time() - start_ts) * 1000
        logger.warning(
            "%s: skipping execution — upstream error detected: %s",
            _NODE_NAME,
            state["error"],
        )
        skip_log = _make_log(
            status="error",
            message=f"Skipped SQL execution due to upstream error: {state['error']}",
            duration_ms=elapsed_ms,
        )
        return {
            "current_node": _NODE_NAME,
            "execution_logs": [start_log, skip_log],
        }

    # ------------------------------------------------------------------
    # 2. Read required state fields
    # ------------------------------------------------------------------
    generated_sql: str = state.get("generated_sql") or ""
    validation_result = state.get("validation_result")

    if not generated_sql:
        elapsed_ms = (time.time() - start_ts) * 1000
        msg = "No generated_sql found in state; cannot execute"
        logger.error("%s: %s", _NODE_NAME, msg)
        err_log = _make_log(
            status="error",
            message=msg,
            duration_ms=elapsed_ms,
        )
        return {
            "error": msg,
            "current_node": _NODE_NAME,
            "execution_logs": [start_log, err_log],
        }

    # ------------------------------------------------------------------
    # 3. Verify SQL validation passed
    # ------------------------------------------------------------------
    if validation_result is None or not validation_result.is_valid:
        elapsed_ms = (time.time() - start_ts) * 1000
        issues = (
            validation_result.issues if validation_result is not None else []
        )
        msg = (
            f"SQL validation did not pass — refusing to execute. "
            f"Issues: {issues}"
        )
        logger.error("%s: %s", _NODE_NAME, msg)
        err_log = _make_log(
            status="error",
            message=msg,
            duration_ms=elapsed_ms,
            metadata={"validation_issues": issues},
        )
        return {
            "error": msg,
            "current_node": _NODE_NAME,
            "execution_logs": [start_log, err_log],
        }

    # Log any validation warnings before proceeding
    if validation_result.warnings:
        logger.warning(
            "%s: SQL has validation warnings: %s",
            _NODE_NAME,
            validation_result.warnings,
        )

    # ------------------------------------------------------------------
    # 4. Execute via MCP
    # ------------------------------------------------------------------
    try:
        client = SnowflakeMCPClient.from_env()
    except (MCPConnectionError, ValueError) as exc:
        elapsed_ms = (time.time() - start_ts) * 1000
        msg = f"Failed to initialise MCP client: {exc}"
        logger.error("%s: %s", _NODE_NAME, msg)
        err_log = _make_log(
            status="error",
            message=msg,
            duration_ms=elapsed_ms,
        )
        failed_result = QueryResult(
            success=False,
            row_count=0,
            data=[],
            error=msg,
            columns=[],
            execution_time_ms=elapsed_ms,
        )
        return {
            "error": msg,
            "query_results": [failed_result],
            "current_node": _NODE_NAME,
            "execution_logs": [start_log, err_log],
        }

    query_start = time.time()
    try:
        logger.info("%s: executing SQL via MCP (max_rows=%d)", _NODE_NAME, _MAX_ROWS)
        mcp_result = await client.execute_query(generated_sql, max_rows=_MAX_ROWS)
        query_elapsed_s = time.time() - query_start
        query_elapsed_ms = query_elapsed_s * 1000

    except (MCPQueryError, MCPConnectionError, Exception) as exc:
        query_elapsed_ms = (time.time() - query_start) * 1000
        total_elapsed_ms = (time.time() - start_ts) * 1000
        msg = f"MCP query execution failed: {exc}"
        logger.error("%s: %s", _NODE_NAME, msg, exc_info=True)
        err_log = _make_log(
            status="error",
            message=msg,
            duration_ms=total_elapsed_ms,
            metadata={
                "sql_preview": generated_sql[:200],
                "query_duration_ms": round(query_elapsed_ms, 2),
                "error_type": type(exc).__name__,
            },
        )
        failed_result = QueryResult(
            success=False,
            row_count=0,
            data=[],
            error=msg,
            columns=[],
            execution_time_ms=query_elapsed_ms,
        )
        return {
            "error": msg,
            "query_results": [failed_result],
            "current_node": _NODE_NAME,
            "execution_logs": [start_log, err_log],
        }

    # ------------------------------------------------------------------
    # 5. Post-execution checks and result conversion
    # ------------------------------------------------------------------
    total_elapsed_ms = (time.time() - start_ts) * 1000

    # Warn on slow queries
    if query_elapsed_s > _SLOW_QUERY_THRESHOLD_S:
        logger.warning(
            "%s: query took %.1f s (threshold: %.1f s). SQL: %s",
            _NODE_NAME,
            query_elapsed_s,
            _SLOW_QUERY_THRESHOLD_S,
            generated_sql[:300],
        )

    # Warn on empty results
    if mcp_result.row_count == 0:
        logger.warning(
            "%s: query returned 0 rows. SQL: %s",
            _NODE_NAME,
            generated_sql[:300],
        )

    # Build the state-compatible QueryResult
    state_result = _mcp_result_to_state(
        mcp_result=mcp_result,
        sql=generated_sql,
        execution_time_ms=query_elapsed_ms,
    )

    logger.info(
        "%s: completed — row_count=%d, duration_ms=%.1f",
        _NODE_NAME,
        state_result.row_count,
        query_elapsed_ms,
    )

    done_log = _make_log(
        status="completed",
        message=(
            f"SQL executed successfully — {state_result.row_count} row(s) returned"
        ),
        duration_ms=total_elapsed_ms,
        metadata={
            "row_count": state_result.row_count,
            "columns": state_result.columns,
            "query_duration_ms": round(query_elapsed_ms, 2),
            "slow_query": query_elapsed_s > _SLOW_QUERY_THRESHOLD_S,
            "empty_result": state_result.row_count == 0,
            "validation_warnings": validation_result.warnings,
            "sql_preview": generated_sql[:200],
        },
    )

    return {
        "query_results": [state_result],
        "current_node": _NODE_NAME,
        "execution_logs": [start_log, done_log],
    }
