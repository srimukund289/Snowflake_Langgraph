"""
graph/nodes/intent_node.py

Intent extraction node for the AI Data Analyst Agent.

Reads the raw user question from AgentState, calls GPT-4o with a structured
JSON prompt, and writes the extracted intent fields back into state.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.llm_config import get_analysis_llm_kwargs
from graph.state import AgentState, ExecutionLog
from prompts.prompts import INTENT_SYSTEM_PROMPT, build_intent_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM – module-level singleton so the model is not re-instantiated per call
# ---------------------------------------------------------------------------
_llm = ChatOpenAI(**get_analysis_llm_kwargs(include_json_mode=True))

# Node name constant used in logs and state tracking
NODE_NAME = "intent_node"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_log(
    status: str,
    message: str,
    duration_ms: float = 0.0,
    metadata: Dict[str, Any] | None = None,
) -> ExecutionLog:
    return ExecutionLog(
        timestamp=datetime.now(timezone.utc).isoformat(),
        node=NODE_NAME,
        status=status,
        message=message,
        duration_ms=duration_ms,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

async def intent_node(state: AgentState) -> dict:
    """
    Extract structured business intent from the user's question.

    Reads:
        state["question"]

    Writes:
        intent, metrics, dimensions, time_period, intent_reasoning,
        current_node, execution_logs

    The LLM is invoked with response_format=json_object so the response is
    always parseable without additional stripping.  The JSON schema returned
    by the model is:

        {
          "intent":      "<intent classification>",
          "metrics":     ["..."],
          "dimensions":  ["..."],
          "time_period": "...",
          "filters":     ["..."],
          "granularity": "...",
          "reasoning":   "..."
        }

    Only the fields required by AgentState are written back; extras (filters,
    granularity) are stored in the execution log metadata for downstream nodes
    that may need them.
    """
    start_time = time.time()
    question: str = state.get("question", "").strip()

    logger.info(
        "intent_node started",
        extra={"node": NODE_NAME, "question_length": len(question)},
    )

    # Emit a "started" log immediately so the timeline is accurate
    started_log = _make_log("started", f"Extracting intent from question: {question[:120]!r}")

    if not question:
        error_msg = "No question provided in state."
        logger.error(error_msg, extra={"node": NODE_NAME})
        duration_ms = (time.time() - start_time) * 1000
        return {
            "error": error_msg,
            "current_node": NODE_NAME,
            "execution_logs": [
                started_log,
                _make_log("error", error_msg, duration_ms=duration_ms),
            ],
        }

    try:
        # Build messages
        user_prompt = build_intent_prompt(question)
        messages = [
            SystemMessage(content=INTENT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        logger.debug("Calling GPT-4o for intent extraction", extra={"node": NODE_NAME})
        response = await _llm.ainvoke(messages)

        # Parse the JSON response
        raw_content: str = response.content
        parsed: Dict[str, Any] = json.loads(raw_content)

        # Extract fields — provide safe defaults for every key
        intent: str = parsed.get("intent", "exploration")
        metrics: List[str] = parsed.get("metrics", [])
        dimensions: List[str] = parsed.get("dimensions", [])
        time_period: str = parsed.get("time_period", "unspecified")
        intent_reasoning: str = parsed.get("reasoning", "")

        # Extra fields not in AgentState but useful for downstream context
        filters: List[str] = parsed.get("filters", [])
        granularity: str = parsed.get("granularity", "unspecified")

        duration_ms = (time.time() - start_time) * 1000

        logger.info(
            "intent_node completed",
            extra={
                "node": NODE_NAME,
                "intent": intent,
                "metrics": metrics,
                "dimensions": dimensions,
                "time_period": time_period,
                "duration_ms": duration_ms,
            },
        )

        completed_log = _make_log(
            status="completed",
            message=(
                f"Intent extracted: '{intent}' | "
                f"metrics={metrics} | "
                f"dimensions={dimensions} | "
                f"time_period='{time_period}'"
            ),
            duration_ms=duration_ms,
            metadata={
                "intent": intent,
                "metrics": metrics,
                "dimensions": dimensions,
                "time_period": time_period,
                "filters": filters,
                "granularity": granularity,
                "reasoning": intent_reasoning,
            },
        )

        return {
            "intent": intent,
            "metrics": metrics,
            "dimensions": dimensions,
            "time_period": time_period,
            "intent_reasoning": intent_reasoning,
            "current_node": NODE_NAME,
            "execution_logs": [started_log, completed_log],
        }

    except json.JSONDecodeError as exc:
        duration_ms = (time.time() - start_time) * 1000
        error_msg = f"Failed to parse JSON from LLM response: {exc}"
        logger.exception(error_msg, extra={"node": NODE_NAME})
        return {
            "error": error_msg,
            "current_node": NODE_NAME,
            "execution_logs": [
                started_log,
                _make_log("error", error_msg, duration_ms=duration_ms, metadata={"exception": str(exc)}),
            ],
        }

    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000
        error_msg = f"intent_node encountered an unexpected error: {exc}"
        logger.exception(error_msg, extra={"node": NODE_NAME})
        return {
            "error": error_msg,
            "current_node": NODE_NAME,
            "execution_logs": [
                started_log,
                _make_log("error", error_msg, duration_ms=duration_ms, metadata={"exception": str(exc)}),
            ],
        }
