"""
graph/nodes/metadata_discovery_node.py

Metadata Discovery Node — FastMCP / local Snowflake backend.

DATASET CONFIGURATION (in .env):

  Option A — single dataset:
    SNOWFLAKE_DATABASE=FINANCE_DW
    SNOWFLAKE_SCHEMA=REPORTING

  Option B — multiple datasets (LLM routes to the right one per question):
    SNOWFLAKE_DATASETS=[
      {"db":"FINANCE_DW","schema":"REPORTING","description":"P&L, budget vs actuals"},
      {"db":"SALES_DW","schema":"ANALYTICS","description":"Orders, customers, products"}
    ]

  When SNOWFLAKE_DATASETS is set it takes priority over SNOWFLAKE_DATABASE/SCHEMA.
  If multiple datasets are configured, GPT-4o reads the descriptions and picks the
  most relevant one for the user's question before running metadata discovery.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Semantic model (optional — used when SEMANTIC_MODEL_PATH is set in .env)
from tools.semantic_model_loader import get_semantic_model, format_semantic_model_for_prompt
from tools.mcp_client import SnowflakeMCPClient

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph.state import AgentState, ExecutionLog
from tools.mcp_client import SnowflakeMCPClient

logger = logging.getLogger(__name__)

NODE_NAME = "metadata_discovery_node"


# ---------------------------------------------------------------------------
# Dataset config helpers
# ---------------------------------------------------------------------------

def _load_datasets() -> List[Dict[str, str]]:
    """
    Parse dataset configuration from environment variables.

    Returns a list of dicts with keys: db, schema, description.
    Precedence: SNOWFLAKE_DATASETS (JSON) > SNOWFLAKE_DATABASE + SNOWFLAKE_SCHEMA.
    """
    raw = os.environ.get("SNOWFLAKE_DATASETS", "").strip()
    if raw:
        # python-dotenv stops at the first newline for unquoted values, so a
        # multi-line JSON array will arrive as a truncated fragment (e.g. just "[").
        # Strip all whitespace/newlines before parsing so both formats work:
        #   single-line: SNOWFLAKE_DATASETS=[{"db":"..."},{"db":"..."}]
        #   quoted:      SNOWFLAKE_DATASETS='[{"db":"..."},{"db":"..."}]'
        raw_clean = raw.replace("\n", "").replace("\r", "").strip().strip("'\"")
        try:
            datasets = json.loads(raw_clean)
            # Normalise keys: accept 'database'/'db', 'schema', 'description'
            normalised = []
            for d in datasets:
                db = (d.get("db") or d.get("database") or "").strip().upper()
                schema = (d.get("schema") or "").strip().upper()
                desc = (d.get("description") or "").strip()
                if db and schema:
                    normalised.append({"db": db, "schema": schema, "description": desc})
            if normalised:
                return normalised
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(
                "SNOWFLAKE_DATASETS could not be parsed as JSON: %s\n"
                "Raw value received: %r\n"
                "TIP: put the entire JSON on a single line in .env, or quote it:\n"
                "  SNOWFLAKE_DATASETS='[{\"db\":\"DB1\",\"schema\":\"S1\",\"description\":\"...\"}]'",
                exc, raw[:120],
            )

    # Fall back to single-dataset env vars.
    # Schema is optional — omitting it means discover all schemas in the database.
    db = os.environ.get("SNOWFLAKE_DATABASE", "").strip().upper()
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "").strip().upper()
    if db:
        return [{"db": db, "schema": schema, "description": ""}]

    return []


async def _route_dataset(
    question: str,
    plan: List[str],
    datasets: List[Dict[str, str]],
) -> Dict[str, str]:
    """
    Use GPT-4o to pick the most relevant dataset for the given question.
    Returns the chosen dataset dict {db, schema, description}.
    """
    catalog = "\n".join(
        f"{i+1}. {d['db']}.{d['schema']}"
        + (f" — {d['description']}" if d['description'] else "")
        for i, d in enumerate(datasets)
    )
    plan_str = "\n".join(f"  - {s}" for s in plan) if plan else "  (not yet generated)"

    prompt = f"""You are a data routing assistant. A user has asked a business question.
Select the single most relevant dataset from the catalog below.

USER QUESTION:
{question}

ANALYSIS PLAN:
{plan_str}

AVAILABLE DATASETS:
{catalog}

Return ONLY a JSON object with these keys:
  index        — 1-based index of the chosen dataset (integer)
  db           — exact database name as listed
  schema       — exact schema name as listed
  reasoning    — one sentence explaining the choice

Example: {{"index": 2, "db": "FINANCE_DW", "schema": "REPORTING", "reasoning": "The question is about P&L which lives in the financial reporting schema."}}"""

    from config.llm_config import get_analysis_llm_kwargs

    llm = ChatOpenAI(**get_analysis_llm_kwargs(include_json_mode=True))
    response = await llm.ainvoke([
        SystemMessage(content="You are a data routing assistant. Return only JSON."),
        HumanMessage(content=prompt),
    ])

    parsed = json.loads(response.content)
    idx = int(parsed.get("index", 1)) - 1
    idx = max(0, min(idx, len(datasets) - 1))   # clamp to valid range
    chosen = datasets[idx]
    logger.info(
        "dataset_routing: chose %s.%s — %s",
        chosen["db"], chosen["schema"],
        parsed.get("reasoning", ""),
    )
    return chosen


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _make_log(
    status: str,
    message: str,
    duration_ms: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> ExecutionLog:
    return ExecutionLog(
        timestamp=_now_iso(),
        node=NODE_NAME,
        status=status,
        message=message,
        duration_ms=duration_ms,
        metadata=metadata or {},
    )


def _error_return(err: str, logs: List[ExecutionLog],
                  tool_names: Optional[List[str]] = None) -> dict:
    logs.append(_make_log("error", err))
    return {
        "error": err,
        "available_metadata": {},
        "available_tool_names": tool_names or [],
        "current_node": NODE_NAME,
        "execution_logs": logs,
    }


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------

async def metadata_discovery_node(state: AgentState) -> dict:
    """
    LangGraph node: discover Snowflake metadata via the local FastMCP server.

    Writes to state:
        available_metadata   : {db: {schema: {table: [col_dicts]}}}
        available_tool_names : List[str]
        current_node         : str
        execution_logs       : List[ExecutionLog]
    """
    logs: List[ExecutionLog] = []
    node_start = time.time()

    # Short-circuit on upstream error
    if state.get("error"):
        logs.append(_make_log("skipped", f"Upstream error: {state['error'][:120]}"))
        return {"available_metadata": {}, "available_tool_names": [],
                "current_node": NODE_NAME, "execution_logs": logs}

    logs.append(_make_log("started", "Metadata discovery started"))

    # ------------------------------------------------------------------
    # 1a. Semantic model — prefer YAML over live INFORMATION_SCHEMA.
    #     Resolution order:
    #       1. SEMANTIC_MODEL_PATH env var (explicit path)
    #       2. Auto-discover *_semantic_model.yml matching db/schema
    #       3. Fall through to dynamic INFORMATION_SCHEMA discovery
    # ------------------------------------------------------------------
    _peek_db = os.environ.get("SNOWFLAKE_DATABASE", "").upper()
    _peek_schema = os.environ.get("SNOWFLAKE_SCHEMA", "").upper()
    try:
        semantic_model = get_semantic_model(db=_peek_db, schema=_peek_schema)
    except Exception as exc:
        return _error_return(f"Failed to load semantic model: {exc}", logs)

    if semantic_model is not None:
        logger.info(
            "metadata_discovery_node: using semantic model '%s' "
            "(%d tables, %d relationships)",
            semantic_model.name, len(semantic_model.tables), len(semantic_model.relationships),
        )
        metadata = semantic_model.to_metadata_dict()
        total_tables = sum(
            len(tbls)
            for schemas in metadata.values()
            for tbls in schemas.values()
        )
        logs.append(_make_log(
            "completed",
            f"Semantic model '{semantic_model.name}': "
            f"{total_tables} table(s), {len(semantic_model.relationships)} relationship(s)",
            metadata={
                "source": "semantic_model",
                "model_name": semantic_model.name,
                "tables": total_tables,
                "relationships": len(semantic_model.relationships),
            },
        ))
        # Still list MCP tools so downstream nodes know what's executable
        try:
            client = SnowflakeMCPClient.from_env()
            tool_infos = await client.list_available_tools()
            available_tool_names = [t.name for t in tool_infos]
        except Exception:
            available_tool_names = []
        return {
            "available_metadata": metadata,
            "available_tool_names": available_tool_names,
            "current_node": NODE_NAME,
            "execution_logs": logs,
        }

    # ------------------------------------------------------------------
    # 1b. Dynamic discovery — no semantic model configured.
    #     Load dataset configuration from env vars.
    # ------------------------------------------------------------------
    datasets = _load_datasets()
    if not datasets:
        return _error_return(
            "No dataset configured. Set SNOWFLAKE_DATASETS or "
            "SNOWFLAKE_DATABASE + SNOWFLAKE_SCHEMA in your .env file.",
            logs,
        )

    logs.append(_make_log(
        "info",
        f"Found {len(datasets)} configured dataset(s): "
        + ", ".join(f"{d['db']}.{d['schema']}" for d in datasets),
        metadata={"datasets": datasets},
    ))

    # ------------------------------------------------------------------
    # 2. Route to the right dataset (LLM picks when >1 configured)
    # ------------------------------------------------------------------
    if len(datasets) == 1:
        chosen = datasets[0]
        logs.append(_make_log(
            "info",
            f"Single dataset configured — using {chosen['db']}.{chosen['schema']}",
        ))
    else:
        try:
            question = state.get("question", "")
            plan = state.get("plan", [])
            chosen = await _route_dataset(question, plan, datasets)
            logs.append(_make_log(
                "completed",
                f"LLM routed to {chosen['db']}.{chosen['schema']}",
                metadata={"chosen": chosen, "total_datasets": len(datasets)},
            ))
        except Exception as exc:
            logger.warning("Dataset routing failed (%s) — defaulting to first dataset", exc)
            chosen = datasets[0]
            logs.append(_make_log(
                "info",
                f"Routing fallback → {chosen['db']}.{chosen['schema']}",
            ))

    db, schema = chosen["db"], chosen["schema"]
    logger.info("metadata_discovery_node.context  db=%s  schema=%s", db, schema)

    # ------------------------------------------------------------------
    # 3. Create MCP client and discover metadata
    # ------------------------------------------------------------------
    try:
        client = SnowflakeMCPClient.from_env()
    except Exception as exc:
        return _error_return(f"Failed to initialise MCP client: {exc}", logs)

    try:
        metadata: Dict[str, Any] = await client.discover_metadata(
            database=db, schema=schema or None   # None = discover all schemas
        )
    except Exception as exc:
        return _error_return(f"discover_metadata failed: {exc}", logs)

    total_tables = sum(
        len(tables)
        for schemas in metadata.values()
        for tables in schemas.values()
    )
    logs.append(_make_log(
        "completed",
        f"Discovered {total_tables} table(s) in {db}.{schema}",
        metadata={"database": db, "schema": schema, "tables": total_tables},
    ))

    # ------------------------------------------------------------------
    # 4. List available MCP tools
    # ------------------------------------------------------------------
    try:
        tool_infos = await client.list_available_tools()
        available_tool_names = [t.name for t in tool_infos]
    except Exception as exc:
        return _error_return(f"list_available_tools failed: {exc}", logs)

    logs.append(_make_log(
        "completed",
        f"Found {len(available_tool_names)} MCP tool(s): {available_tool_names}",
        metadata={"tools": available_tool_names},
    ))

    # ------------------------------------------------------------------
    # 5. Final summary
    # ------------------------------------------------------------------
    total_ms = (time.time() - node_start) * 1000
    logs.append(_make_log(
        "completed",
        f"Done — {total_tables} table(s) in {db}.{schema} | {total_ms:.0f} ms",
        duration_ms=total_ms,
        metadata={"database": db, "schema": schema,
                  "tables": total_tables, "tool_count": len(available_tool_names)},
    ))

    return {
        "available_metadata": metadata,
        "available_tool_names": available_tool_names,
        "current_node": NODE_NAME,
        "execution_logs": logs,
    }
