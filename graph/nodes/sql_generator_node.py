"""
graph/nodes/sql_generator_node.py

SQL Generator Node — generates Snowflake SQL from the investigation plan,
selected tables, and full table schemas discovered by the metadata pipeline.

This node:
1. Reads question, plan, selected_tables, and table_metadata from state.
2. Serialises table schemas into the prompt via build_sql_prompt().
3. Calls ChatOpenAI (gpt-4o) with JSON-mode structured output.
4. Parses the returned {sql, reasoning, assumptions, expected_columns}.
5. Strips markdown code fences from the SQL if present.
6. Writes generated_sql and sql_reasoning back to state.

No direct Snowflake connections are made here — SQL execution is deferred
to sql_executor_node which uses the MCP client.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from graph.state import AgentState, ExecutionLog, TableMetadata
from prompts.prompts import SQL_GENERATOR_SYSTEM_PROMPT, build_sql_prompt

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node constants
# ---------------------------------------------------------------------------

_NODE_NAME = "sql_generator_node"

# Load LLM config from environment
from config.llm_config import LLM_MODEL, LLM_TEMPERATURE_ANALYSIS

_LLM_MODEL = LLM_MODEL
_LLM_TEMPERATURE = LLM_TEMPERATURE_ANALYSIS

# Markdown code-fence patterns to strip from LLM output
_FENCE_RE = re.compile(
    r"```(?:sql|SQL)?\s*([\s\S]*?)```",
    re.IGNORECASE,
)

# JSON response schema expected from the LLM
_RESPONSE_SCHEMA = {
    "title": "SQLGeneratorResponse",
    "type": "object",
    "properties": {
        "sql": {
            "type": "string",
            "description": "Complete Snowflake SQL query (SELECT only, LIMIT required)",
        },
        "reasoning": {
            "type": "string",
            "description": "2-4 sentences explaining the query design choices",
        },
        "assumptions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of assumptions made while writing the query",
        },
        "expected_columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of column names that will appear in the result",
        },
    },
    "required": ["sql", "reasoning", "assumptions"],
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _make_log(
    node: str,
    status: str,
    message: str,
    duration_ms: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> ExecutionLog:
    """Construct an ExecutionLog entry."""
    return ExecutionLog(
        timestamp=_now_iso(),
        node=node,
        status=status,
        message=message,
        duration_ms=duration_ms,
        metadata=metadata or {},
    )


def _clean_sql(raw_sql: str) -> str:
    """
    Remove markdown code fences from a SQL string if present.

    Handles:
      ```sql SELECT ... ```
      ```SELECT ... ```
      ``` SELECT ... ```

    If no fences are found the input is returned stripped.
    """
    raw_sql = raw_sql.strip()
    match = _FENCE_RE.search(raw_sql)
    if match:
        return match.group(1).strip()
    return raw_sql


def _table_metadata_to_prompt_dicts(
    table_metadata: List[TableMetadata],
) -> List[Dict[str, Any]]:
    """
    Convert a list of TableMetadata dataclasses into the dict shape expected
    by build_sql_prompt():
      [{"table": "DB.SCHEMA.TABLE", "columns": [{"name": ..., "type": ..., "nullable": ...}]}]
    """
    result: List[Dict[str, Any]] = []
    for tbl in table_metadata:
        cols = [
            {
                "name": col.name,
                "type": col.data_type,
                "nullable": col.nullable,
                "comment": col.description,
            }
            for col in tbl.columns
        ]
        result.append(
            {
                "table": tbl.fully_qualified_name,
                "columns": cols,
            }
        )
    return result


def _parse_llm_response(raw: Any) -> Dict[str, Any]:
    """
    Parse the LLM response to a dict with keys: sql, reasoning, assumptions.

    Handles:
    - Already-parsed dict (structured output)
    - JSON string (raw completion)
    - AIMessage with .content attribute
    """
    # Case 1: structured output already returned a dict
    if isinstance(raw, dict):
        return raw

    # Case 2: AIMessage or similar object
    content: str = ""
    if hasattr(raw, "content"):
        content = raw.content
    elif isinstance(raw, str):
        content = raw
    else:
        content = str(raw)

    # Strip optional markdown fences around the JSON blob
    content = content.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content, re.IGNORECASE)
    if fence_match:
        content = fence_match.group(1).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"SQL generator LLM did not return valid JSON. "
            f"Raw content (first 500 chars): {content[:500]!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def sql_generator_node(state: AgentState) -> dict:
    """
    LangGraph node: generate Snowflake SQL from the investigation plan.

    Reads
    -----
    state.question          : original user question
    state.plan              : ordered list of analysis steps from planner_node
    state.selected_tables   : fully-qualified table names from dataset_selector_node
    state.table_metadata    : List[TableMetadata] with column details
    state.intent            : intent classification (used for time context)
    state.time_period       : time period string from intent_node
    state.dimensions        : dimensions list (used to confirm granularity)

    Writes
    ------
    state.generated_sql     : clean SQL string
    state.sql_reasoning     : explanation of query design
    state.current_node      : set to this node's name
    state.execution_logs    : appended with start + completion (or error) entries
    state.error             : set on failure (other fields left unchanged)
    """
    start_ts = time.time()
    logs: List[ExecutionLog] = []

    # Short-circuit: if a prior node already failed, propagate its error.
    if state.get("error"):
        prior_error = state["error"]
        logs.append(_make_log(_NODE_NAME, "skipped",
                              f"Skipping — upstream error: {prior_error[:120]}"))
        return {"current_node": _NODE_NAME, "execution_logs": logs}

    # --- Start log ---
    logs.append(
        _make_log(
            node=_NODE_NAME,
            status="started",
            message="SQL generation started",
            metadata={
                "selected_tables": state.get("selected_tables", []),
                "plan_steps": len(state.get("plan", [])),
            },
        )
    )
    logger.info(
        "%s | started | tables=%s plan_steps=%d",
        _NODE_NAME,
        state.get("selected_tables", []),
        len(state.get("plan", [])),
    )

    try:
        # ----------------------------------------------------------------
        # 1. Extract required fields from state
        # ----------------------------------------------------------------
        question: str = state.get("question", "")
        plan: List[str] = state.get("plan", [])
        selected_tables: List[str] = state.get("selected_tables", [])
        table_metadata: List[TableMetadata] = state.get("table_metadata", [])
        time_period: str = state.get("time_period", "unspecified")
        intent: str = state.get("intent", "")

        if not question:
            raise ValueError("State field 'question' is empty — cannot generate SQL.")

        if not selected_tables:
            raise ValueError(
                "No tables selected. Ensure dataset_selector_node ran successfully."
            )

        # ----------------------------------------------------------------
        # 2. Serialise table schemas for the prompt
        # ----------------------------------------------------------------
        prompt_table_dicts = _table_metadata_to_prompt_dicts(table_metadata)

        # Determine granularity hint from dimensions (best-effort)
        dimensions: List[str] = state.get("dimensions", [])
        granularity: str = "unspecified"
        time_keywords = {"daily", "weekly", "monthly", "quarterly", "yearly", "annual"}
        for dim in dimensions:
            if any(kw in dim.lower() for kw in time_keywords):
                granularity = dim.lower()
                break

        # ----------------------------------------------------------------
        # 3. Build the user-turn prompt
        # ----------------------------------------------------------------
        user_prompt = build_sql_prompt(
            question=question,
            plan=plan,
            selected_tables=selected_tables,
            table_metadata=prompt_table_dicts,
            time_period=time_period,
            granularity=granularity,
        )

        logger.debug(
            "%s | prompt_built | length=%d", _NODE_NAME, len(user_prompt)
        )

        # ----------------------------------------------------------------
        # 4. Call ChatOpenAI with JSON structured output
        # ----------------------------------------------------------------
        llm = ChatOpenAI(model=_LLM_MODEL, temperature=_LLM_TEMPERATURE)
        structured_llm = llm.with_structured_output(_RESPONSE_SCHEMA)

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=SQL_GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        logger.info("%s | calling LLM | model=%s", _NODE_NAME, _LLM_MODEL)
        llm_start = time.time()
        raw_response = await structured_llm.ainvoke(messages)
        llm_duration_ms = (time.time() - llm_start) * 1000
        logger.info(
            "%s | LLM responded | duration_ms=%.1f", _NODE_NAME, llm_duration_ms
        )

        # ----------------------------------------------------------------
        # 5. Parse the structured response
        # ----------------------------------------------------------------
        parsed = _parse_llm_response(raw_response)

        raw_sql: str = parsed.get("sql", "").strip()
        reasoning: str = parsed.get("reasoning", "").strip()
        assumptions: List[str] = parsed.get("assumptions", [])
        expected_columns: List[str] = parsed.get("expected_columns", [])

        if not raw_sql:
            raise ValueError(
                "LLM returned an empty SQL string. Full response: "
                + json.dumps(parsed, default=str)[:500]
            )

        # ----------------------------------------------------------------
        # 6. Clean the SQL (strip markdown fences if present)
        # ----------------------------------------------------------------
        clean_sql = _clean_sql(raw_sql)

        logger.info(
            "%s | sql_generated | length=%d assumptions=%d expected_cols=%d",
            _NODE_NAME,
            len(clean_sql),
            len(assumptions),
            len(expected_columns),
        )

        # ----------------------------------------------------------------
        # 7. Build completion log and return state updates
        # ----------------------------------------------------------------
        duration_ms = (time.time() - start_ts) * 1000
        logs.append(
            _make_log(
                node=_NODE_NAME,
                status="completed",
                message="SQL generation completed successfully",
                duration_ms=duration_ms,
                metadata={
                    "sql_length": len(clean_sql),
                    "assumptions": assumptions,
                    "expected_columns": expected_columns,
                    "llm_duration_ms": round(llm_duration_ms, 1),
                    "reasoning_preview": reasoning[:200],
                },
            )
        )

        return {
            "generated_sql": clean_sql,
            "sql_reasoning": reasoning,
            "current_node": _NODE_NAME,
            "execution_logs": logs,
        }

    except Exception as exc:
        duration_ms = (time.time() - start_ts) * 1000
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("%s | error | %s", _NODE_NAME, error_msg, exc_info=True)

        logs.append(
            _make_log(
                node=_NODE_NAME,
                status="error",
                message=error_msg,
                duration_ms=duration_ms,
                metadata={"error_type": type(exc).__name__},
            )
        )

        return {
            "error": error_msg,
            "current_node": _NODE_NAME,
            "execution_logs": logs,
        }
