"""
graph/nodes/analyst_node.py

Analyzes SQL query results and generates structured business findings.

Reads query_results from state, formats the data as a readable table,
calls GPT-4o with JSON output to produce findings, data_summary, anomalies,
and key_metrics, then writes those fields back to state.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from graph.state import AgentState, ExecutionLog, QueryResult
from prompts.prompts import (
    ANALYST_SYSTEM_PROMPT,
    build_analyst_prompt,
    format_results_for_prompt,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM — shared module-level instance (model is stateless, safe to reuse)
# ---------------------------------------------------------------------------
from config.llm_config import get_analysis_llm_kwargs

_llm = ChatOpenAI(**get_analysis_llm_kwargs(include_json_mode=True))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _extract_result_data(query_results: List[QueryResult]) -> List[Dict[str, Any]]:
    """
    Pull row data from the first successful QueryResult in the list.
    Returns an empty list if there are no results or none succeeded.
    """
    for qr in query_results:
        if qr.success and qr.data:
            return qr.data
    return []


def _parse_analyst_json(raw: str) -> Dict[str, Any]:
    """
    Parse the LLM's raw JSON output into a dict.
    Strips markdown code-fences if the model wrapped the response.
    """
    text = raw.strip()
    # Remove optional ```json / ``` wrappers
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first and last fence lines
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

async def analyst_node(state: AgentState) -> dict:
    """
    LangGraph node: analyze SQL query results and produce business findings.

    Returns a partial state dict with:
        findings, data_summary, anomalies, current_node, execution_logs
    """
    node_name = "analyst_node"
    start_ts = _now_iso()
    t0 = time.time()

    logger.info(json.dumps({"node": node_name, "status": "started"}))

    # ------------------------------------------------------------------
    # Fast path: propagate upstream error
    # ------------------------------------------------------------------
    upstream_error: str | None = state.get("error")
    if upstream_error:
        duration_ms = (time.time() - t0) * 1000
        log_entry = ExecutionLog(
            timestamp=start_ts,
            node=node_name,
            status="error",
            message=f"Skipped — upstream error: {upstream_error}",
            duration_ms=duration_ms,
            metadata={"upstream_error": upstream_error},
        )
        logger.warning(
            json.dumps({"node": node_name, "status": "skipped", "reason": upstream_error})
        )
        return {
            "findings": [f"Analysis skipped due to upstream error: {upstream_error}"],
            "data_summary": f"No analysis performed. Upstream error: {upstream_error}",
            "anomalies": [],
            "current_node": node_name,
            "execution_logs": [log_entry],
        }

    # ------------------------------------------------------------------
    # Read relevant state fields
    # ------------------------------------------------------------------
    question: str = state.get("question", "")
    generated_sql: str = state.get("generated_sql", "")
    query_results: List[QueryResult] = state.get("query_results", [])
    plan: List[str] = state.get("plan", [])
    intent: str = state.get("intent", "")

    # ------------------------------------------------------------------
    # Handle empty query results
    # ------------------------------------------------------------------
    results_data = _extract_result_data(query_results)

    if not results_data:
        duration_ms = (time.time() - t0) * 1000
        log_entry = ExecutionLog(
            timestamp=start_ts,
            node=node_name,
            status="completed",
            message="Query executed but returned no data rows.",
            duration_ms=duration_ms,
            metadata={"row_count": 0},
        )
        logger.info(json.dumps({"node": node_name, "status": "no_data"}))
        return {
            "findings": ["No data returned from query."],
            "data_summary": (
                "Query executed but returned no results. "
                "This may indicate the filters are too restrictive, "
                "the time period has no data, or the tables are empty."
            ),
            "anomalies": [],
            "current_node": node_name,
            "execution_logs": [log_entry],
        }

    # ------------------------------------------------------------------
    # Build prompt and call GPT-4o
    # ------------------------------------------------------------------
    try:
        user_prompt = build_analyst_prompt(
            question=question,
            sql=generated_sql,
            results=results_data,
            plan=plan,
            intent=intent if intent else None,
        )

        messages = [
            {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        logger.debug(
            json.dumps(
                {
                    "node": node_name,
                    "action": "llm_call",
                    "row_count": len(results_data),
                }
            )
        )

        response = await _llm.ainvoke(messages)
        raw_content: str = response.content

        parsed = _parse_analyst_json(raw_content)

        findings: List[str] = parsed.get("findings", [])
        data_summary: str = parsed.get("data_summary", "")
        anomalies: List[str] = parsed.get("anomalies", [])
        key_metrics: Dict[str, str] = parsed.get("key_metrics", {})

        # Ensure non-empty defaults
        if not findings:
            findings = ["Analysis complete — no specific findings were highlighted."]
        if not data_summary:
            data_summary = "Analysis completed. Review the findings for details."

        duration_ms = (time.time() - t0) * 1000

        log_entry = ExecutionLog(
            timestamp=start_ts,
            node=node_name,
            status="completed",
            message=(
                f"Generated {len(findings)} finding(s), "
                f"{len(anomalies)} anomaly(ies), "
                f"confidence={parsed.get('confidence', 'UNKNOWN')}."
            ),
            duration_ms=duration_ms,
            metadata={
                "row_count": len(results_data),
                "findings_count": len(findings),
                "anomalies_count": len(anomalies),
                "confidence": parsed.get("confidence", ""),
                "key_metrics_keys": list(key_metrics.keys()),
            },
        )

        logger.info(
            json.dumps(
                {
                    "node": node_name,
                    "status": "completed",
                    "findings_count": len(findings),
                    "anomalies_count": len(anomalies),
                    "duration_ms": round(duration_ms, 1),
                }
            )
        )

        return {
            "findings": findings,
            "data_summary": data_summary,
            "anomalies": anomalies,
            "current_node": node_name,
            "execution_logs": [log_entry],
        }

    except json.JSONDecodeError as exc:
        duration_ms = (time.time() - t0) * 1000
        error_msg = f"Failed to parse analyst LLM response as JSON: {exc}"
        logger.error(json.dumps({"node": node_name, "status": "error", "error": error_msg}))
        log_entry = ExecutionLog(
            timestamp=start_ts,
            node=node_name,
            status="error",
            message=error_msg,
            duration_ms=duration_ms,
            metadata={"exception_type": "JSONDecodeError"},
        )
        return {
            "error": error_msg,
            "findings": ["Analysis failed — LLM returned unparseable output."],
            "data_summary": "Analysis could not be completed due to a parsing error.",
            "anomalies": [],
            "current_node": node_name,
            "execution_logs": [log_entry],
        }

    except Exception as exc:  # noqa: BLE001
        duration_ms = (time.time() - t0) * 1000
        error_msg = f"analyst_node error: {type(exc).__name__}: {exc}"
        logger.exception(json.dumps({"node": node_name, "status": "error", "error": error_msg}))
        log_entry = ExecutionLog(
            timestamp=start_ts,
            node=node_name,
            status="error",
            message=error_msg,
            duration_ms=duration_ms,
            metadata={"exception_type": type(exc).__name__},
        )
        return {
            "error": error_msg,
            "findings": ["Analysis failed due to an unexpected error."],
            "data_summary": "Analysis could not be completed.",
            "anomalies": [],
            "current_node": node_name,
            "execution_logs": [log_entry],
        }
