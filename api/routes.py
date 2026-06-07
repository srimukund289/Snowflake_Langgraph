"""
api/routes.py

FastAPI router exposing:
  POST /analyze  – run the LangGraph analysis workflow
  GET  /health   – service health check

Every request is tagged with a unique request-id injected via middleware
(see app.py where the middleware is registered).  The request-id is also
propagated into structured log entries so that all log lines for a single
HTTP call share the same correlation id.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging setup
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


def _error(event: str, **kw: Any) -> None:
    if _USE_STRUCTLOG:
        _log.error(event, **kw)
    else:
        logger.error("%s | %s", event, json.dumps(kw, default=str))


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Payload for POST /analyze."""

    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Natural-language question to analyse against Snowflake data.",
        examples=["What was the total revenue by region in Q4 2024?"],
    )


class AnalyzeResponse(BaseModel):
    """Response returned by POST /analyze."""

    answer: str = Field(
        default="",
        description="Executive-level plain-English answer to the question.",
    )
    analysis: str = Field(
        default="",
        description="Detailed analytical narrative produced by the analyst node.",
    )
    sql: str = Field(
        default="",
        description="The final SQL query that was executed against Snowflake.",
    )
    tables: List[str] = Field(
        default_factory=list,
        description="Fully-qualified table names (DB.SCHEMA.TABLE) used in the analysis.",
    )
    intent: str = Field(
        default="",
        description="High-level intent label extracted from the question.",
    )
    plan: List[str] = Field(
        default_factory=list,
        description="Ordered investigation steps produced by the planner node.",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Wall-clock time taken to process the request in milliseconds.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Non-null when the workflow completed with an error condition.",
    )


class HealthResponse(BaseModel):
    """Response returned by GET /health."""

    status: str = Field(default="healthy")
    timestamp: str = Field(description="ISO-8601 UTC timestamp of the health check.")
    version: str = Field(default="1.0.0")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()


# ---------------------------------------------------------------------------
# Helper: extract request-id from request state (set by middleware in app.py)
# ---------------------------------------------------------------------------


def _get_request_id(request: Request) -> str:
    """Return the request-id injected by the RequestID middleware, or a new one."""
    return getattr(request.state, "request_id", str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Run AI data analysis workflow",
    description=(
        "Accepts a natural-language question, runs it through the full LangGraph "
        "analysis pipeline (intent extraction -> planning -> metadata discovery -> "
        "table selection -> SQL generation & validation -> SQL execution -> "
        "analysis -> response generation) and returns a structured answer."
    ),
    responses={
        400: {"description": "Bad request – question too short / too long."},
        422: {"description": "Validation error in request payload."},
        500: {"description": "Internal error during workflow execution."},
    },
)
async def analyze(request: Request, body: AnalyzeRequest) -> AnalyzeResponse:
    """Run the LangGraph analysis workflow for the supplied question."""

    request_id = _get_request_id(request)
    start_time = time.monotonic()

    _info(
        "analyze.request.received",
        request_id=request_id,
        question_length=len(body.question),
        question_preview=body.question[:120],
    )

    # Lazy import to avoid circular imports at module load time and to keep
    # startup fast when graph assembly is expensive.
    try:
        from graph.workflow import run_analysis  # noqa: PLC0415
    except ImportError as exc:
        _error(
            "analyze.import_error",
            request_id=request_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Workflow module could not be loaded. Check server logs.",
        ) from exc

    try:
        final_state: Dict[str, Any] = await run_analysis(body.question)
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.monotonic() - start_time) * 1000
        _error(
            "analyze.workflow.error",
            request_id=request_id,
            error=str(exc),
            error_type=type(exc).__name__,
            elapsed_ms=round(elapsed_ms, 2),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {exc}",
        ) from exc

    elapsed_ms = (time.monotonic() - start_time) * 1000

    response = AnalyzeResponse(
        answer=final_state.get("answer", ""),
        analysis=final_state.get("analysis", ""),
        sql=final_state.get("sql", ""),
        tables=final_state.get("tables", []),
        intent=final_state.get("intent", ""),
        plan=final_state.get("plan", []),
        processing_time_ms=round(elapsed_ms, 2),
        error=final_state.get("error"),
    )

    _info(
        "analyze.request.completed",
        request_id=request_id,
        elapsed_ms=round(elapsed_ms, 2),
        tables_used=response.tables,
        has_error=response.error is not None,
    )

    return response


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service health check",
    description=(
        "Returns HTTP 200 with status='healthy' when the service is running. "
        "Optionally checks MCP client connectivity when the MCP_SERVER_URL "
        "environment variable is set."
    ),
)
async def health(request: Request) -> HealthResponse:
    """Return a liveness / readiness health response."""

    import datetime  # noqa: PLC0415

    request_id = _get_request_id(request)
    now_iso = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    _info("health.check", request_id=request_id, timestamp=now_iso)

    return HealthResponse(
        status="healthy",
        timestamp=now_iso,
        version="1.0.0",
    )
