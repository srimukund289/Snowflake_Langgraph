"""
app.py - FastAPI application entry point for AI Data Analyst Agent.

Responsibilities:
- Application factory with lifespan context manager
- Structured JSON logging setup
- CORS and request-logging middleware
- Global exception handler
- Pydantic Settings for environment-based configuration
- Router inclusion (api.routes)
"""

from __future__ import annotations

# Load .env into os.environ BEFORE any other imports so that langchain-openai
# and other libraries that read os.environ at import/instantiation time pick up
# the keys from the .env file.
from dotenv import load_dotenv
load_dotenv()

import logging
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Snowflake connection credentials
    snowflake_account: str = Field(..., description="Snowflake account identifier (e.g. VEIDJBV-BR57195)")
    snowflake_user: str = Field(..., description="Snowflake username")
    snowflake_password: str = Field(..., description="Snowflake password")
    snowflake_warehouse: str = Field(..., description="Snowflake virtual warehouse name")
    snowflake_role: str = Field(default="", description="Snowflake role (optional)")

    # Single-dataset fallback (used when SNOWFLAKE_DATASETS is not set)
    snowflake_database: str = Field(default="", description="Default Snowflake database")
    snowflake_schema: str = Field(default="", description="Default Snowflake schema")

    # Multi-dataset routing: JSON array of {db, schema, description}
    # When set, the LLM picks the right dataset per question.
    # Example: [{"db":"FINANCE_DW","schema":"REPORTING","description":"Financial P&L data"}]
    snowflake_datasets: str = Field(
        default="",
        description="JSON array of datasets with db, schema, description fields",
    )

    openai_api_key: str = Field(..., description="OpenAI API key for GPT-4o")
    log_level: str = Field(default="INFO", description="Logging level (DEBUG/INFO/WARNING/ERROR)")
    max_retries: int = Field(default=3, description="Maximum retries for MCP calls")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog with JSON output and stdlib integration."""

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle for the FastAPI application."""

    # ---- Startup ----
    settings: Settings = app.state.settings  # type: ignore[attr-defined]
    _configure_logging(settings.log_level)

    missing: list[str] = []
    if not settings.snowflake_account:
        missing.append("SNOWFLAKE_ACCOUNT")
    if not settings.snowflake_user:
        missing.append("SNOWFLAKE_USER")
    if not settings.snowflake_password:
        missing.append("SNOWFLAKE_PASSWORD")
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")

    if missing:
        logger.error(
            "startup_validation_failed",
            missing_env_vars=missing,
        )
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    logger.info(
        "application_startup",
        snowflake_account=settings.snowflake_account,
        snowflake_database=settings.snowflake_database,
        snowflake_schema=settings.snowflake_schema,
        log_level=settings.log_level,
        max_retries=settings.max_retries,
    )

    yield

    # ---- Shutdown ----
    logger.info("application_shutdown")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = Settings()  # type: ignore[call-arg]  # loaded from env

    app = FastAPI(
        title="AI Data Analyst Agent",
        version="1.0.0",
        description=(
            "Production-ready AI agent that answers natural-language data questions "
            "by querying Snowflake via MCP and synthesising executive-level insights."
        ),
        lifespan=lifespan,
    )

    # Attach settings so the lifespan handler (and routes) can access them.
    app.state.settings = settings

    # ---- CORS ----------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Request logging middleware ------------------------------------------
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next: Any) -> Response:
        start_time = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            client=request.client.host if request.client else "unknown",
        )
        return response

    # ---- Global exception handler -------------------------------------------
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled_exception",
            method=request.method,
            path=request.url.path,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc),
            },
        )

    # ---- Root redirect ------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/health")

    # ---- Include routers ----------------------------------------------------
    from api.routes import router  # noqa: PLC0415 — deferred to avoid import-time side-effects

    app.include_router(router)

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (used by uvicorn and tests)
# ---------------------------------------------------------------------------

app = create_app()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
