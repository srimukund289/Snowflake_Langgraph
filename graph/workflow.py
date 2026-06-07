"""
graph/workflow.py — LangGraph StateGraph assembling all 9 analysis nodes.

Linear flow:
  question -> intent -> planner -> metadata_discovery -> dataset_selector
           -> sql_generator -> sql_validator -> sql_executor -> analyst -> response

Conditional edge from sql_validator: if state["error"] is set after validation,
the workflow skips sql_executor and routes directly to analyst, which handles
the error case gracefully.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AgentState
from graph.nodes.intent_node import intent_node
from graph.nodes.planner_node import planner_node
from graph.nodes.metadata_discovery_node import metadata_discovery_node
from graph.nodes.dataset_selector_node import dataset_selector_node
from graph.nodes.sql_generator_node import sql_generator_node
from graph.nodes.sql_validator_node import sql_validator_node
from graph.nodes.sql_executor_node import sql_executor_node
from graph.nodes.analyst_node import analyst_node
from graph.nodes.response_node import response_node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional routing helper
# ---------------------------------------------------------------------------


def _route_after_validation(state: AgentState) -> str:
    """
    Route the graph after sql_validator_node completes.

    If validation set an error (invalid or dangerous SQL) we skip sql_executor
    and go straight to analyst_node, which is expected to handle the error
    state and produce a meaningful response.  Otherwise we proceed normally
    to sql_executor_node.
    """
    if state.get("error"):
        logger.warning(
            "SQL validation failed — routing directly to analyst_node. "
            "error=%s",
            state["error"],
        )
        return "analyst_node"
    return "sql_executor_node"


# ---------------------------------------------------------------------------
# Build the StateGraph
# ---------------------------------------------------------------------------

def _build_workflow() -> StateGraph:
    """Construct and return the compiled LangGraph StateGraph."""
    workflow = StateGraph(AgentState)

    # -- Add all 9 nodes --------------------------------------------------
    workflow.add_node("intent_node", intent_node)
    workflow.add_node("planner_node", planner_node)
    workflow.add_node("metadata_discovery_node", metadata_discovery_node)
    workflow.add_node("dataset_selector_node", dataset_selector_node)
    workflow.add_node("sql_generator_node", sql_generator_node)
    workflow.add_node("sql_validator_node", sql_validator_node)
    workflow.add_node("sql_executor_node", sql_executor_node)
    workflow.add_node("analyst_node", analyst_node)
    workflow.add_node("response_node", response_node)

    # -- Entry point -------------------------------------------------------
    workflow.set_entry_point("intent_node")

    # -- Linear edges (pre-validation) ------------------------------------
    workflow.add_edge("intent_node", "planner_node")
    workflow.add_edge("planner_node", "metadata_discovery_node")
    workflow.add_edge("metadata_discovery_node", "dataset_selector_node")
    workflow.add_edge("dataset_selector_node", "sql_generator_node")
    workflow.add_edge("sql_generator_node", "sql_validator_node")

    # -- Conditional edge from sql_validator_node -------------------------
    # Pass:  sql_validator_node -> sql_executor_node -> analyst_node
    # Fail:  sql_validator_node -> analyst_node  (skip execution)
    workflow.add_conditional_edges(
        "sql_validator_node",
        _route_after_validation,
        {
            "sql_executor_node": "sql_executor_node",
            "analyst_node": "analyst_node",
        },
    )

    # -- Linear edges (post-validation) -----------------------------------
    workflow.add_edge("sql_executor_node", "analyst_node")
    workflow.add_edge("analyst_node", "response_node")
    workflow.add_edge("response_node", END)

    return workflow


# ---------------------------------------------------------------------------
# Compile with MemorySaver checkpointer for observability / replay
# ---------------------------------------------------------------------------

_memory_checkpointer = MemorySaver()

compiled_graph = _build_workflow().compile(checkpointer=_memory_checkpointer)

logger.info("LangGraph workflow compiled successfully with MemorySaver checkpointer.")


# ---------------------------------------------------------------------------
# Public helper: run the full analysis workflow
# ---------------------------------------------------------------------------


async def run_analysis(question: str) -> Dict[str, Any]:
    """
    Run the full analysis workflow for a natural-language question.

    Parameters
    ----------
    question:
        The business / analytical question posed by the user.

    Returns
    -------
    dict with keys:
        answer          — executive-level narrative answer
        analysis        — detailed analytical findings
        sql             — generated SQL query
        tables          — list of fully-qualified table names used
        intent          — detected intent label
        plan            — ordered list of investigation steps
        error           — error message if any node failed, else None
    """
    thread_id = str(uuid.uuid4())
    config: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    initial_state: Dict[str, Any] = {
        "question": question,
        "current_node": "start",
        "execution_logs": [],
        "query_results": [],
    }

    logger.info(
        "Starting analysis workflow",
        extra={"thread_id": thread_id, "question": question},
    )

    try:
        result = await compiled_graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        logger.exception("Workflow invocation failed: %s", exc)
        return {
            "answer": "",
            "analysis": "",
            "sql": "",
            "tables": [],
            "intent": "",
            "plan": [],
            "error": f"Workflow failed: {exc}",
        }

    logger.info(
        "Analysis workflow completed",
        extra={
            "thread_id": thread_id,
            "has_error": bool(result.get("error")),
        },
    )

    return {
        "answer": result.get("answer", ""),
        "analysis": result.get("analysis", ""),
        "sql": result.get("generated_sql", ""),
        "tables": result.get("selected_tables", []),
        "intent": result.get("intent", ""),
        "plan": result.get("plan", []),
        "error": result.get("error"),
    }
