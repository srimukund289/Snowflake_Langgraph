"""
graph/nodes/dataset_selector_node.py

Dataset selector node for the AI Data Analyst Agent workflow.

Reads the available_metadata discovered by metadata_discovery_node, then uses
gpt-4o (with JSON-mode output) to select the minimal, most-relevant set of
tables needed to answer the user's question.  For each selected table it builds
a rich TableMetadata object from the metadata catalog so downstream nodes have
full column-level detail without needing another MCP round-trip.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import AgentState, ColumnInfo, ExecutionLog, TableMetadata
from prompts.prompts import DATASET_SELECTOR_SYSTEM_PROMPT, build_selector_prompt

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NODE_NAME = "dataset_selector_node"
_MAX_TABLES = 5            # hard cap on selected tables (spec says top 5)

# Load LLM config from environment
from config.llm_config import LLM_MODEL, LLM_TEMPERATURE_ANALYSIS

_LLM_MODEL = LLM_MODEL
_LLM_TEMPERATURE = LLM_TEMPERATURE_ANALYSIS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _make_log(
    status: str,
    message: str,
    duration_ms: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> ExecutionLog:
    return ExecutionLog(
        timestamp=_now_iso(),
        node=_NODE_NAME,
        status=status,
        message=message,
        duration_ms=duration_ms,
        metadata=metadata or {},
    )


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """
    Attempt to parse JSON from the LLM response.

    The LLM is instructed to return raw JSON but may occasionally wrap it in
    markdown fences.  Strip those before parsing.
    """
    text = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first fence line and any trailing fence line
        inner_lines = []
        for i, line in enumerate(lines):
            if i == 0 and line.startswith("```"):
                continue
            if line.strip() == "```":
                continue
            inner_lines.append(line)
        text = "\n".join(inner_lines).strip()

    return json.loads(text)


def _extract_columns_from_metadata(
    db: str, schema: str, table: str, metadata: Dict[str, Any]
) -> List[ColumnInfo]:
    """
    Pull column definitions for a specific table out of the nested metadata dict.

    Metadata structure (set by metadata_discovery_node):
      {
        "DATABASE": {
          "SCHEMA": {
            "TABLE": [
              {"name": "col", "type": "VARCHAR", "nullable": True, "comment": "..."},
              ...
            ]
          }
        }
      }

    The lookup is case-insensitive on all three keys to handle Snowflake's
    convention of returning names in uppercase.
    """
    # Build a case-insensitive lookup for database names
    db_key: Optional[str] = None
    for k in metadata.keys():
        if k.upper() == db.upper():
            db_key = k
            break
    if db_key is None:
        return []

    schemas = metadata[db_key]
    if not isinstance(schemas, dict):
        return []

    schema_key: Optional[str] = None
    for k in schemas.keys():
        if k.upper() == schema.upper():
            schema_key = k
            break
    if schema_key is None:
        return []

    tables = schemas[schema_key]
    if not isinstance(tables, dict):
        return []

    table_key: Optional[str] = None
    for k in tables.keys():
        if k.upper() == table.upper():
            table_key = k
            break
    if table_key is None:
        return []

    raw_columns = tables[table_key]
    if not isinstance(raw_columns, list):
        return []

    columns: List[ColumnInfo] = []
    for col in raw_columns:
        if not isinstance(col, dict):
            continue
        name = col.get("name", col.get("NAME", col.get("column_name", "unknown")))
        data_type = col.get(
            "type",
            col.get("data_type", col.get("TYPE", col.get("DATA_TYPE", "UNKNOWN"))),
        )
        nullable_raw = col.get(
            "nullable",
            col.get("is_nullable", col.get("NULLABLE", col.get("IS_NULLABLE", True))),
        )
        # Normalise nullable to bool — MCP servers sometimes return "YES"/"NO"
        if isinstance(nullable_raw, str):
            nullable = nullable_raw.upper() not in ("NO", "FALSE", "NOT NULL", "N")
        else:
            nullable = bool(nullable_raw) if nullable_raw is not None else True

        description = col.get(
            "comment", col.get("description", col.get("COMMENT", ""))
        )
        columns.append(
            ColumnInfo(
                name=str(name),
                data_type=str(data_type),
                nullable=nullable,
                description=str(description) if description else "",
            )
        )

    return columns


def _parse_fully_qualified_name(fqn: str):
    """
    Split a fully-qualified table name into (database, schema, table).

    Accepts:
      - DATABASE.SCHEMA.TABLE
      - DATABASE.SCHEMA  (table defaults to "")
      - TABLE            (database and schema default to "")
    """
    parts = fqn.strip().split(".")
    if len(parts) >= 3:
        return parts[0], parts[1], ".".join(parts[2:])
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return "", "", parts[0]


def _build_table_metadata(
    selected_tables: List[str], metadata: Dict[str, Any]
) -> List[TableMetadata]:
    """
    Build a list of TableMetadata objects for each selected table.

    If a table cannot be found in the metadata catalog (e.g. the LLM
    hallucinated a name), it is still included but with an empty column list
    so the downstream sql_generator_node can still proceed.
    """
    result: List[TableMetadata] = []
    for fqn in selected_tables:
        db, schema, table = _parse_fully_qualified_name(fqn)
        columns = _extract_columns_from_metadata(db, schema, table, metadata)
        result.append(
            TableMetadata(
                database=db,
                schema=schema,
                table_name=table,
                columns=columns,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def dataset_selector_node(state: AgentState) -> dict:
    """
    LangGraph node: select the most relevant tables for the user's question.

    Reads
    -----
    - state["question"]            : original user question
    - state["plan"]                : ordered list of analysis steps
    - state["metrics"]             : extracted metric labels (for context)
    - state["dimensions"]          : extracted dimension labels (for context)
    - state["available_metadata"]  : full metadata tree from metadata_discovery_node

    Writes
    ------
    - selected_tables      : List[str]          fully-qualified DB.SCHEMA.TABLE names
    - table_metadata       : List[TableMetadata] per-table column details
    - selection_reasoning  : str                 LLM's explanation of the selection
    - current_node         : str
    - execution_logs       : List[ExecutionLog]  (appended via operator.add)
    - error                : str | None          set on failure
    """
    t_start = time.time()
    logs: List[ExecutionLog] = []

    # Short-circuit: if a prior node already failed, propagate its error.
    if state.get("error"):
        prior_error = state["error"]
        logs.append(_make_log("skipped", f"Skipping — upstream error: {prior_error[:120]}"))
        return {"current_node": _NODE_NAME, "execution_logs": logs}

    logs.append(_make_log("started", "dataset_selector_node started"))
    logger.info("[%s] started", _NODE_NAME)

    try:
        # ------------------------------------------------------------------
        # 1. Read required state fields
        # ------------------------------------------------------------------
        question: str = state.get("question", "")
        plan: List[str] = state.get("plan", [])
        metrics: List[str] = state.get("metrics", [])
        dimensions: List[str] = state.get("dimensions", [])
        available_metadata: Dict[str, Any] = state.get("available_metadata", {})

        if not question:
            raise ValueError("state['question'] is empty — cannot select tables")

        if not available_metadata:
            raise ValueError(
                "state['available_metadata'] is empty — "
                "metadata_discovery_node must run before dataset_selector_node"
            )

        logger.info(
            "[%s] inputs: metrics=%s, dimensions=%s, plan_steps=%d, "
            "metadata_dbs=%d",
            _NODE_NAME,
            metrics,
            dimensions,
            len(plan),
            len(available_metadata),
        )

        # ------------------------------------------------------------------
        # 2. Build prompt
        # ------------------------------------------------------------------
        user_prompt = build_selector_prompt(
            question=question,
            plan=plan,
            metadata=available_metadata,
        )

        # ------------------------------------------------------------------
        # 3. Call gpt-4o with JSON output
        # ------------------------------------------------------------------
        model = ChatOpenAI(
            model=_LLM_MODEL,
            temperature=_LLM_TEMPERATURE,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        messages = [
            SystemMessage(content=DATASET_SELECTOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        logger.info("[%s] invoking LLM (%s)", _NODE_NAME, _LLM_MODEL)
        llm_response = await model.ainvoke(messages)
        raw_content: str = llm_response.content

        # ------------------------------------------------------------------
        # 4. Parse LLM response
        # ------------------------------------------------------------------
        try:
            parsed: Dict[str, Any] = _parse_llm_json(raw_content)
        except json.JSONDecodeError as exc:
            logger.error(
                "[%s] LLM response is not valid JSON: %s | raw=%s",
                _NODE_NAME,
                exc,
                raw_content[:500],
            )
            raise ValueError(
                f"LLM returned non-JSON response: {exc}\n"
                f"Raw (first 500 chars): {raw_content[:500]}"
            ) from exc

        raw_selected: List[str] = parsed.get("selected_tables", [])
        join_strategy: str = parsed.get("join_strategy", "")
        reasoning: str = parsed.get("reasoning", "")
        excluded: List[Any] = parsed.get("excluded_tables", [])
        dq_flags: List[str] = parsed.get("data_quality_flags", [])

        logger.info(
            "[%s] LLM selected %d tables: %s",
            _NODE_NAME,
            len(raw_selected),
            raw_selected,
        )

        # ------------------------------------------------------------------
        # 5. Validate: must have at least one table.
        #
        # If the LLM returned an empty list it means the question is out of
        # scope for the configured datasets — surface the LLM's reasoning as
        # a clear error rather than blindly auto-selecting unrelated tables.
        # ------------------------------------------------------------------
        if not raw_selected:
            # Propagate the LLM's own explanation so the user sees why
            out_of_scope_msg = (
                f"The configured database does not contain data relevant to this question. "
                f"{reasoning or 'No matching tables were found.'} "
                f"Available tables: "
                + ", ".join(
                    f"{db}.{schema}.{table}"
                    for db, schemas in available_metadata.items()
                    for schema, tables in schemas.items()
                    for table in list(tables)[:10]
                )
            )
            raise ValueError(out_of_scope_msg)

        # (raw_selected already raised above if empty — this block is unreachable
        #  but kept as a safety net for the metadata-empty case)

        # ------------------------------------------------------------------
        # 6. Enforce top-5 limit
        # ------------------------------------------------------------------
        selected_tables = raw_selected[:_MAX_TABLES]
        if len(raw_selected) > _MAX_TABLES:
            logger.warning(
                "[%s] LLM returned %d tables; capping to top %d",
                _NODE_NAME,
                len(raw_selected),
                _MAX_TABLES,
            )

        # ------------------------------------------------------------------
        # 7. Build TableMetadata for each selected table
        # ------------------------------------------------------------------
        table_metadata = _build_table_metadata(selected_tables, available_metadata)

        # REJECT hallucinated tables (tables not in metadata)
        hallucinated = [tm for tm in table_metadata if not tm.columns]
        if hallucinated:
            hallucinated_names = ", ".join(tm.fully_qualified_name for tm in hallucinated)
            error_msg = (
                f"LLM hallucinated table name(s) not in the database: {hallucinated_names}. "
                f"Available tables: {', '.join(f'{db}.{schema}.{table}' for db, schemas in available_metadata.items() for schema, tables in schemas.items() for table in tables)}. "
                f"The question may be out of scope for the configured database."
            )
            raise ValueError(error_msg)

        # Warn if any selected table has no columns (this should not be reached due to check above)
        for tm in table_metadata:
            if not tm.columns:
                logger.warning(
                    "[%s] table '%s' has no columns in metadata catalog — "
                    "LLM may have hallucinated this table name",
                    _NODE_NAME,
                    tm.fully_qualified_name,
                )

        # ------------------------------------------------------------------
        # 8. Compose selection_reasoning (include join strategy + DQ flags)
        # ------------------------------------------------------------------
        reasoning_parts = [reasoning]
        if join_strategy:
            reasoning_parts.append(f"Join strategy: {join_strategy}")
        if dq_flags:
            reasoning_parts.append(
                "Data quality flags: " + "; ".join(dq_flags)
            )
        if excluded:
            excl_summary = "; ".join(
                f"{e.get('table', '?')} ({e.get('reason', '')})"
                if isinstance(e, dict)
                else str(e)
                for e in excluded[:5]
            )
            reasoning_parts.append(f"Excluded tables: {excl_summary}")
        selection_reasoning = "  |  ".join(p for p in reasoning_parts if p)

        # ------------------------------------------------------------------
        # 9. Emit completion log
        # ------------------------------------------------------------------
        duration_ms = (time.time() - t_start) * 1000
        logs.append(
            _make_log(
                "completed",
                f"Selected {len(selected_tables)} table(s): "
                + ", ".join(selected_tables),
                duration_ms=duration_ms,
                metadata={
                    "selected_tables": selected_tables,
                    "table_count": len(selected_tables),
                    "tables_with_columns": sum(
                        1 for tm in table_metadata if tm.columns
                    ),
                    "join_strategy": join_strategy,
                    "data_quality_flags": dq_flags,
                },
            )
        )

        logger.info(
            "[%s] completed in %.1f ms — %d table(s) selected",
            _NODE_NAME,
            duration_ms,
            len(selected_tables),
        )

        return {
            "selected_tables": selected_tables,
            "table_metadata": table_metadata,
            "selection_reasoning": selection_reasoning,
            "current_node": _NODE_NAME,
            "execution_logs": logs,
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        duration_ms = (time.time() - t_start) * 1000
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("[%s] error — %s", _NODE_NAME, error_msg, exc_info=True)

        logs.append(
            _make_log(
                "error",
                f"dataset_selector_node failed: {error_msg}",
                duration_ms=duration_ms,
                metadata={"error_type": type(exc).__name__},
            )
        )

        return {
            "error": error_msg,
            "current_node": _NODE_NAME,
            "execution_logs": logs,
        }
