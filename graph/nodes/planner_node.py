"""
graph/nodes/planner_node.py

Planner node: generates a structured, step-by-step analytical investigation
plan from the extracted intent metadata produced by intent_node.

Reads  : question, intent, metrics, dimensions, time_period from AgentState
Writes : plan (List[str]), current_node, execution_logs
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from config.llm_config import get_analysis_llm_kwargs
from graph.state import AgentState, ExecutionLog
from prompts.prompts import PLANNER_SYSTEM_PROMPT, build_planner_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM initialisation (module-level singleton; created once per worker process)
# ---------------------------------------------------------------------------
_llm = ChatOpenAI(**get_analysis_llm_kwargs(include_json_mode=True))


def _parse_steps(raw: str) -> List[str]:
    """
    Parse the LLM response into a list of plan step strings.

    The LLM is instructed to return a JSON object of the form:
        {"steps": ["step 1 ...", "step 2 ...", ...]}

    Falls back to splitting on numbered list patterns if JSON parsing fails.
    Returns between 1 and 10 non-empty strings.
    """
    raw = raw.strip()

    # --- Primary: expect valid JSON with a "steps" key ---
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "steps" in parsed:
            steps = parsed["steps"]
            if isinstance(steps, list):
                cleaned = [str(s).strip() for s in steps if str(s).strip()]
                if cleaned:
                    return cleaned
    except json.JSONDecodeError:
        pass

    # --- Fallback: try to extract JSON from a code-fenced response ---
    for fence in ("```json", "```"):
        if fence in raw:
            start = raw.find(fence) + len(fence)
            end = raw.find("```", start)
            if end != -1:
                block = raw[start:end].strip()
                try:
                    parsed = json.loads(block)
                    if isinstance(parsed, dict) and "steps" in parsed:
                        steps = parsed["steps"]
                        cleaned = [str(s).strip() for s in steps if str(s).strip()]
                        if cleaned:
                            return cleaned
                except json.JSONDecodeError:
                    pass

    # --- Last resort: split on numbered-list lines ("1. ...", "2. ...") ---
    import re
    lines = raw.splitlines()
    steps: List[str] = []
    numbered_re = re.compile(r"^\s*\d+[\.\)]\s+(.+)")
    for line in lines:
        m = numbered_re.match(line)
        if m:
            steps.append(m.group(1).strip())

    if steps:
        logger.warning("planner_node: fell back to regex parsing for plan steps")
        return steps

    # Absolute fallback — return the whole response as a single step so the
    # pipeline is not blocked.
    logger.error("planner_node: could not parse plan steps; using raw response as single step")
    return [raw[:500]] if raw else ["Investigate the question using available data."]


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

async def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: generate an analytical investigation plan.

    Reads extracted intent fields from *state*, calls GPT-4o with the planner
    prompt, parses the numbered step list, and returns the updated state slice.

    Returns:
        dict with keys: plan, current_node, execution_logs
    """
    node_name = "planner_node"
    start_ts = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    logger.info(
        "planner_node started",
        extra={
            "node": node_name,
            "intent": state.get("intent", ""),
            "metrics": state.get("metrics", []),
            "dimensions": state.get("dimensions", []),
            "time_period": state.get("time_period", ""),
        },
    )

    # -----------------------------------------------------------------------
    # 1. Read inputs from state
    # -----------------------------------------------------------------------
    question: str = state.get("question", "")
    intent: str = state.get("intent", "")
    metrics: List[str] = state.get("metrics", [])
    dimensions: List[str] = state.get("dimensions", [])
    time_period: str = state.get("time_period", "")

    if not question:
        err = "planner_node: 'question' is empty in state — cannot build plan"
        logger.error(err)
        duration_ms = (time.time() - t0) * 1000
        log_entry = ExecutionLog(
            timestamp=start_ts,
            node=node_name,
            status="error",
            message=err,
            duration_ms=duration_ms,
            metadata={"error": err},
        )
        return {
            "error": err,
            "current_node": node_name,
            "execution_logs": [log_entry],
        }

    # -----------------------------------------------------------------------
    # 2. Build the prompt
    # -----------------------------------------------------------------------
    user_message = build_planner_prompt(
        question=question,
        intent=intent,
        metrics=metrics,
        dimensions=dimensions,
        time_period=time_period,
    )

    logger.debug("planner_node: prompt built, invoking LLM")

    # -----------------------------------------------------------------------
    # 3. Call GPT-4o
    # -----------------------------------------------------------------------
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = await _llm.ainvoke(messages)
        raw_content: str = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        err = f"planner_node: LLM call failed — {exc}"
        logger.exception(err)
        duration_ms = (time.time() - t0) * 1000
        log_entry = ExecutionLog(
            timestamp=start_ts,
            node=node_name,
            status="error",
            message=err,
            duration_ms=duration_ms,
            metadata={"exception": str(exc)},
        )
        return {
            "error": err,
            "current_node": node_name,
            "execution_logs": [log_entry],
        }

    logger.debug("planner_node: LLM response received, parsing steps")

    # -----------------------------------------------------------------------
    # 4. Parse the response into a list of plan steps
    # -----------------------------------------------------------------------
    plan: List[str] = _parse_steps(raw_content)

    duration_ms = (time.time() - t0) * 1000

    logger.info(
        "planner_node completed",
        extra={
            "node": node_name,
            "plan_steps": len(plan),
            "duration_ms": round(duration_ms, 2),
        },
    )

    # -----------------------------------------------------------------------
    # 5. Build execution log entry
    # -----------------------------------------------------------------------
    log_entry = ExecutionLog(
        timestamp=start_ts,
        node=node_name,
        status="completed",
        message=f"Generated {len(plan)}-step analysis plan",
        duration_ms=round(duration_ms, 2),
        metadata={
            "plan_steps": len(plan),
            "intent": intent,
            "metrics": metrics,
            "dimensions": dimensions,
            "time_period": time_period,
            "first_step_preview": plan[0][:120] if plan else "",
        },
    )

    # -----------------------------------------------------------------------
    # 6. Return partial state update
    # -----------------------------------------------------------------------
    return {
        "plan": plan,
        "current_node": node_name,
        "execution_logs": [log_entry],
    }
