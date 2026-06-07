"""
graph/nodes/response_node.py

Generates the executive-ready final response from accumulated analysis state.

Responsibilities:
- Read all upstream results: question, plan, intent, findings, data_summary,
  anomalies, generated_sql, query_results, error.
- If an error is present, produce an error-aware report explaining what happened.
- Call ChatOpenAI gpt-4o with the RESPONSE_SYSTEM_PROMPT to write a structured
  Markdown report (Executive Summary / Key Findings / Root Cause Analysis /
  Recommendations / Next Steps for Analysis).
- Derive a concise 2-3 sentence "answer" from the Executive Summary section.
- Return updated fields: answer, analysis, current_node, execution_logs.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import AgentState, ExecutionLog
from prompts.prompts import RESPONSE_SYSTEM_PROMPT, build_response_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level LLM instance (shared; stateless)
# ---------------------------------------------------------------------------

from config.llm_config import get_response_llm_kwargs

_llm = ChatOpenAI(**get_response_llm_kwargs())

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ERROR_REPORT_TEMPLATE = """\
## Executive Summary

The analysis could not be completed due to a processing error. {error_summary} \
The pipeline was interrupted before producing final results. \
The sections below capture what was retrieved before the failure.

## Key Findings

- **Error encountered:** {error_detail}

## Root Cause Analysis

The workflow encountered an error that prevented full execution. \
This may be caused by a connectivity issue with the data source, an invalid or \
unsafe SQL query, missing schema metadata, or an unexpected data format. \
The original question was: *{question}*

## Recommendations

1. **Operations:** Review the execution logs for the specific error message and node that failed.
2. **Data Engineering:** Verify that the Snowflake MCP server is reachable and returning valid responses.
3. **Analytics:** Rephrase the question or narrow its scope, then retry the analysis.

## Next Steps for Analysis

- [ ] Re-run the query with explicit table and column names — to isolate a metadata discovery failure.
- [ ] Check MCP server connectivity and bearer token validity — to rule out authentication issues.
"""


_DATA_SCOPE_KEYWORDS = (
    "does not contain data relevant",
    "configured database does not contain",
    "No relevant tables",
    "out of scope",
)

_DATA_SCOPE_TEMPLATE = """\
## Executive Summary

The data needed to answer this question is not available in the configured database.

{error_detail}

## What You Can Do

1. **Ask a question about your available data** — the configured database contains: {tables}
2. **Add a dataset** — add a new entry to `SNOWFLAKE_DATASETS` in your `.env` file that \
points to a database containing the data you need.
3. **Rephrase the question** — if the data exists under a different name or metric, \
try rephrasing using terms that match your table and column names.

## Configured Database Contents

The agent found these tables — use them to guide your questions:

{table_list}
"""


def _is_data_scope_error(error: str) -> bool:
    return any(kw in error for kw in _DATA_SCOPE_KEYWORDS)


def _build_error_report(question: str, error: str,
                        available_metadata: dict = None) -> str:
    """Return a Markdown error report when the pipeline did not complete."""
    if _is_data_scope_error(error):
        # Build a helpful list of what IS available
        tables: list = []
        if available_metadata:
            for db, schemas in available_metadata.items():
                for schema, tbls in schemas.items():
                    for tbl in tbls:
                        tables.append(f"- `{db}.{schema}.{tbl}`")
        table_list = "\n".join(tables[:20]) if tables else "- (metadata not available)"
        tables_inline = ", ".join(t.strip("- `") for t in tables[:5])
        return _DATA_SCOPE_TEMPLATE.format(
            error_detail=error.replace("ValueError: ", ""),
            tables=tables_inline or "see list below",
            table_list=table_list,
        )

    short = error[:200] + "..." if len(error) > 200 else error
    return _ERROR_REPORT_TEMPLATE.format(
        error_summary="See the error detail below for specifics.",
        error_detail=short,
        question=question.strip(),
    )


def _extract_executive_summary(markdown: str) -> str:
    """
    Extract the Executive Summary paragraph from the Markdown report and return
    it as a clean 2-3 sentence "answer".

    Falls back to the first non-header, non-empty paragraph if the section
    header is not found.
    """
    lines = markdown.splitlines()
    in_section = False
    collected: List[str] = []

    for line in lines:
        stripped = line.strip()

        # Detect the Executive Summary header (handles ## or ###)
        if stripped.lower().startswith("#") and "executive summary" in stripped.lower():
            in_section = True
            continue

        # Stop at the next section header
        if in_section and stripped.startswith("#"):
            break

        if in_section and stripped:
            collected.append(stripped)

    if collected:
        paragraph = " ".join(collected)
        # Truncate to roughly 3 sentences
        sentences = paragraph.split(". ")
        answer = ". ".join(sentences[:3]).strip()
        if answer and not answer.endswith("."):
            answer += "."
        return answer

    # Fallback: return first non-empty, non-header line
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("-"):
            return stripped

    return markdown[:300].strip()


def _collect_query_data(state: AgentState) -> List[Dict[str, Any]]:
    """Flatten all successful query result rows for context in the prompt."""
    rows: List[Dict[str, Any]] = []
    for qr in (state.get("query_results") or []):
        if qr.success:
            rows.extend(qr.data or [])
    return rows


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

async def response_node(state: AgentState) -> dict:
    """
    LangGraph node: generate the executive-ready final response.

    Returns a partial state dict with:
        answer          - concise 2-3 sentence summary
        analysis        - full Markdown report
        current_node    - "response_node"
        execution_logs  - list containing one ExecutionLog entry
    """
    node_name = "response_node"
    start_ts = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    logger.info("response_node: starting", extra={"node": node_name})

    # ------------------------------------------------------------------
    # Read relevant state fields
    # ------------------------------------------------------------------
    question: str = state.get("question", "")
    plan: List[str] = state.get("plan") or []
    intent: str = state.get("intent", "")
    findings: List[str] = state.get("findings") or []
    data_summary: str = state.get("data_summary") or ""
    anomalies: List[str] = state.get("anomalies") or []
    generated_sql: str = state.get("generated_sql") or ""
    error: Optional[str] = state.get("error")

    # Analyst node may store additional context
    # (these keys may or may not be present depending on analyst_node version)
    key_metrics: Optional[Dict[str, str]] = state.get("key_metrics")  # type: ignore[assignment]
    root_cause_hypotheses: Optional[List[str]] = state.get("root_cause_hypotheses")  # type: ignore[assignment]
    caveats: Optional[List[str]] = state.get("caveats")  # type: ignore[assignment]

    try:
        # ------------------------------------------------------------------
        # Branch: error path
        # ------------------------------------------------------------------
        if error:
            logger.warning(
                "response_node: upstream error detected, generating error report",
                extra={"error_preview": error[:120]},
            )
            available_metadata = state.get("available_metadata") or {}
            analysis = _build_error_report(question, error, available_metadata)
            if _is_data_scope_error(error):
                answer = (
                    "The data needed to answer this question is not available in the "
                    "configured database. " + error.replace("ValueError: ", "")[:200]
                )
            else:
                answer = (
                    f"The analysis for your question could not be completed. "
                    f"An error was encountered during processing: {error[:150]}. "
                    f"Please review the execution logs and retry."
                )

        else:
            # ------------------------------------------------------------------
            # Build prompt and call LLM
            # ------------------------------------------------------------------
            user_prompt = build_response_prompt(
                question=question,
                plan=plan,
                findings=findings,
                data_summary=data_summary,
                anomalies=anomalies,
                key_metrics=key_metrics,
                root_cause_hypotheses=root_cause_hypotheses,
                caveats=caveats,
            )

            messages = [
                SystemMessage(content=RESPONSE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            logger.debug("response_node: invoking LLM", extra={"prompt_chars": len(user_prompt)})

            response = await _llm.ainvoke(messages)
            analysis: str = response.content.strip()  # type: ignore[assignment]

            # ------------------------------------------------------------------
            # Extract concise answer from Executive Summary
            # ------------------------------------------------------------------
            answer = _extract_executive_summary(analysis)

        duration_ms = (time.time() - t0) * 1000

        log_entry = ExecutionLog(
            timestamp=start_ts,
            node=node_name,
            status="completed",
            message=f"Generated executive report ({len(analysis)} chars). Answer: {answer[:80]}...",
            duration_ms=round(duration_ms, 2),
            metadata={
                "has_error": bool(error),
                "findings_count": len(findings),
                "anomalies_count": len(anomalies),
                "answer_length": len(answer),
                "analysis_length": len(analysis),
                "intent": intent,
            },
        )

        logger.info(
            "response_node: completed",
            extra={
                "duration_ms": round(duration_ms, 2),
                "answer_preview": answer[:100],
            },
        )

        return {
            "answer": answer,
            "analysis": analysis,
            "current_node": node_name,
            "execution_logs": [log_entry],
        }

    except Exception as exc:
        duration_ms = (time.time() - t0) * 1000
        error_msg = f"response_node failed: {exc}"
        logger.exception("response_node: unhandled exception", exc_info=exc)

        # Best-effort fallback report
        fallback_analysis = _build_error_report(
            question, error_msg
        )
        fallback_answer = (
            "An unexpected error occurred while generating the executive report. "
            f"Error: {str(exc)[:200]}. "
            "Please check execution logs for details."
        )

        log_entry = ExecutionLog(
            timestamp=start_ts,
            node=node_name,
            status="error",
            message=error_msg,
            duration_ms=round(duration_ms, 2),
            metadata={"exception_type": type(exc).__name__},
        )

        return {
            "answer": fallback_answer,
            "analysis": fallback_analysis,
            "error": error_msg,
            "current_node": node_name,
            "execution_logs": [log_entry],
        }
