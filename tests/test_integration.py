"""
tests/test_integration.py

Integration tests for the full AI Data Analyst Agent workflow.

All external calls (MCP / OpenAI) are mocked so the suite runs without live
credentials or network access.

Test matrix
-----------
1.  test_full_analysis_flow              — mock every node's external call;
                                           verify end-to-end state flow.
2.  test_analyze_endpoint                — POST /analyze with mocked workflow.
3.  test_health_endpoint                 — GET /health returns 200 + correct keys.
4.  test_invalid_question                — empty / too-short question -> 422.
5.  test_mcp_client_retry                — tenacity retries 3x on transient failure.
6.  test_sql_validation_blocks_bad_sql   — validator rejects DROP; workflow
                                           routes to analyst without executing.
7.  test_metadata_discovery              — node calls list_tools first, then
                                           uses discovered names.

pytest.ini (repo root) sets:
    [pytest]
    asyncio_mode = auto
"""

from __future__ import annotations

import json
import os
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Environment stubs — must be set before any project import that reads env
# ---------------------------------------------------------------------------

os.environ.setdefault("MCP_SERVER_URL", "http://mock-mcp-server/sse")
os.environ.setdefault("MCP_BEARER_TOKEN", "mock-bearer-token")
os.environ.setdefault("OPENAI_API_KEY", "mock-openai-key")

# ---------------------------------------------------------------------------
# Project imports (after env stubs)
# ---------------------------------------------------------------------------

from graph.state import (
    AgentState,
    QueryResult,
    SQLValidationResult,
    make_initial_state,
)
from graph.nodes.sql_validator_node import sql_validator_node
from graph.nodes.metadata_discovery_node import metadata_discovery_node
from tools.mcp_client import (
    MCPConnectionError,
    SnowflakeMCPClient,
    ToolInfo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_info(name: str, description: str = "") -> ToolInfo:
    """Convenience factory for ToolInfo objects used in MCP mocks."""
    return ToolInfo(name=name, description=description, input_schema={})


def _make_state(**overrides: Any) -> AgentState:
    """Return a fully-initialised AgentState with optional field overrides."""
    base = make_initial_state(question="What was the total revenue in Q4 2024?")
    base.update(overrides)  # type: ignore[attr-defined]
    return base


def _llm_response(content: str) -> MagicMock:
    """Return a MagicMock that looks like a LangChain AIMessage with .content."""
    resp = MagicMock()
    resp.content = content
    return resp


# ---------------------------------------------------------------------------
# Standard MCP fixtures reused across several tests
# ---------------------------------------------------------------------------

_MOCK_TOOLS = [
    _make_tool_info("list_databases"),
    _make_tool_info("list_schemas"),
    _make_tool_info("list_tables"),
    _make_tool_info("describe_table"),
    _make_tool_info("execute_query"),
]

_MOCK_METADATA = {
    "ANALYTICS": {
        "PUBLIC": {
            "SALES_FACT": [
                {"name": "region", "type": "VARCHAR"},
                {"name": "revenue", "type": "NUMBER"},
                {"name": "quarter", "type": "VARCHAR"},
                {"name": "year", "type": "NUMBER"},
            ]
        }
    }
}


# ---------------------------------------------------------------------------
# 1. test_full_analysis_flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_analysis_flow():
    """
    End-to-end workflow: mock all LLM and MCP calls, invoke run_analysis(),
    and assert that the returned dict is correctly populated.
    """
    # LLM payloads
    intent_payload = json.dumps(
        {
            "intent": "Revenue Trend Analysis",
            "metrics": ["revenue", "sales"],
            "dimensions": ["region"],
            "time_period": "Q4 2024",
            "filters": [],
            "granularity": "monthly",
            "reasoning": "User wants regional revenue figures for Q4 2024.",
        }
    )
    planner_payload = json.dumps(
        {
            "plan": [
                "Identify revenue tables",
                "Filter by Q4 2024",
                "Aggregate by region",
            ]
        }
    )
    selector_payload = json.dumps(
        {
            "selected_tables": ["ANALYTICS.PUBLIC.SALES_FACT"],
            "reasoning": "SALES_FACT contains revenue and region columns.",
        }
    )
    sql_payload = json.dumps(
        {
            "sql": (
                "SELECT region, SUM(revenue) AS total_revenue "
                "FROM ANALYTICS.PUBLIC.SALES_FACT "
                "WHERE quarter = 'Q4' AND year = 2024 "
                "GROUP BY region LIMIT 100"
            ),
            "reasoning": "Aggregate revenue by region for Q4 2024.",
        }
    )
    analyst_payload = json.dumps(
        {
            "findings": ["North America led with $5M revenue."],
            "data_summary": "Revenue data for Q4 2024 by region.",
            "anomalies": [],
            "key_metrics": {"total_revenue": "$12M"},
            "confidence": "HIGH",
        }
    )

    # MCP execute_query return value
    mock_mcp_qr = MagicMock()
    mock_mcp_qr.columns = ["region", "total_revenue"]
    mock_mcp_qr.rows = [["North America", 5_000_000], ["EMEA", 4_000_000]]
    mock_mcp_qr.row_count = 2

    # dataset_selector_node and sql_generator_node construct ChatOpenAI locally,
    # so we patch the class in those modules.  intent_node, planner_node,
    # analyst_node, and response_node expose a module-level _llm singleton.
    local_llm_mock = MagicMock()
    local_llm_mock.with_structured_output = MagicMock(return_value=local_llm_mock)
    local_llm_mock.ainvoke = AsyncMock(
        side_effect=[
            _llm_response(selector_payload),
            _llm_response(sql_payload),
        ]
    )

    def _local_chat_openai(*_a: Any, **_kw: Any) -> MagicMock:
        return local_llm_mock

    with (
        patch("graph.nodes.intent_node._llm") as m_intent,
        patch("graph.nodes.planner_node._llm") as m_planner,
        patch("graph.nodes.analyst_node._llm") as m_analyst,
        patch("graph.nodes.response_node._llm") as m_response,
        patch("graph.nodes.dataset_selector_node.ChatOpenAI", side_effect=_local_chat_openai),
        patch("graph.nodes.sql_generator_node.ChatOpenAI", side_effect=_local_chat_openai),
        patch(
            "tools.mcp_client.SnowflakeMCPClient.list_available_tools",
            new_callable=AsyncMock,
            return_value=_MOCK_TOOLS,
        ),
        patch(
            "tools.mcp_client.SnowflakeMCPClient.discover_metadata",
            new_callable=AsyncMock,
            return_value=_MOCK_METADATA,
        ),
        patch(
            "tools.mcp_client.SnowflakeMCPClient.execute_query",
            new_callable=AsyncMock,
            return_value=mock_mcp_qr,
        ),
    ):
        m_intent.ainvoke = AsyncMock(return_value=_llm_response(intent_payload))
        m_planner.ainvoke = AsyncMock(return_value=_llm_response(planner_payload))
        m_analyst.ainvoke = AsyncMock(return_value=_llm_response(analyst_payload))
        m_response.ainvoke = AsyncMock(
            return_value=_llm_response(
                "## Executive Summary\n\n"
                "North America led revenue in Q4 2024.\n\n"
                "## Key Findings\n\n- $5M from North America.\n"
            )
        )

        from graph.workflow import run_analysis  # noqa: PLC0415

        result = await run_analysis("What was the total revenue in Q4 2024?")

    # Required top-level keys
    for key in ("answer", "analysis", "sql", "tables", "intent", "plan", "error"):
        assert key in result, f"Missing key in result: '{key}'"

    assert result["error"] is None, f"Unexpected workflow error: {result['error']}"
    assert result["intent"] == "Revenue Trend Analysis"
    assert "revenue" in result["sql"].lower()
    assert isinstance(result["tables"], list) and len(result["tables"]) > 0
    assert isinstance(result["plan"], list) and len(result["plan"]) > 0
    assert result["answer"], "answer must be non-empty"
    assert result["analysis"], "analysis must be non-empty"


# ---------------------------------------------------------------------------
# 2. test_analyze_endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_endpoint():
    """
    POST /analyze returns 200 with the correct response shape when
    run_analysis is mocked to return a canned result.
    """
    canned = {
        "answer": "North America led Q4 2024 with $5M revenue.",
        "analysis": "## Executive Summary\n\nRevenue analysis complete.",
        "generated_sql": "SELECT region, SUM(revenue) FROM sales GROUP BY region",
        "selected_tables": ["ANALYTICS.PUBLIC.SALES_FACT"],
        "intent": "Revenue Trend Analysis",
        "plan": ["Step 1", "Step 2"],
        "error": None,
    }

    with patch("graph.workflow.run_analysis", new_callable=AsyncMock, return_value=canned):
        from app import create_app  # noqa: PLC0415

        test_app = create_app()
        async with httpx.AsyncClient(app=test_app, base_url="http://testserver") as client:
            response = await client.post(
                "/analyze",
                json={"question": "What was the total revenue in Q4 2024?"},
            )

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )
    body = response.json()
    for field in ("answer", "analysis", "sql", "tables", "intent", "plan", "processing_time_ms"):
        assert field in body, f"Missing field in response body: '{field}'"

    assert body["intent"] == "Revenue Trend Analysis"
    assert isinstance(body["tables"], list)
    assert isinstance(body["processing_time_ms"], float)


# ---------------------------------------------------------------------------
# 3. test_health_endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint():
    """
    GET /health must return HTTP 200 with status='healthy', a timestamp,
    and a version field.
    """
    from app import create_app  # noqa: PLC0415

    test_app = create_app()
    async with httpx.AsyncClient(app=test_app, base_url="http://testserver") as client:
        response = await client.get("/health")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    body = response.json()
    assert body.get("status") == "healthy", f"Unexpected status: {body}"
    assert "timestamp" in body, "Response must include 'timestamp'"
    assert "version" in body, "Response must include 'version'"
    ts: str = body["timestamp"]
    assert "T" in ts or "-" in ts, f"timestamp does not look ISO-8601: {ts!r}"


# ---------------------------------------------------------------------------
# 4. test_invalid_question
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_question():
    """
    POST /analyze with an empty string or a question shorter than 5 characters
    must return HTTP 422 without ever calling run_analysis.
    """
    from app import create_app  # noqa: PLC0415

    test_app = create_app()

    with patch("graph.workflow.run_analysis", new_callable=AsyncMock) as mock_run:
        async with httpx.AsyncClient(app=test_app, base_url="http://testserver") as client:
            resp_empty = await client.post("/analyze", json={"question": ""})
            resp_short = await client.post("/analyze", json={"question": "hi"})

    assert resp_empty.status_code == 422, (
        f"Expected 422 for empty question, got {resp_empty.status_code}"
    )
    assert resp_short.status_code == 422, (
        f"Expected 422 for short question, got {resp_short.status_code}"
    )
    mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# 5. test_mcp_client_retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_client_retry():
    """
    Verify that the tenacity retry wrapper fires exactly 3 attempts before
    re-raising MCPConnectionError on persistent failures.
    """
    from tenacity import AsyncRetrying, stop_after_attempt, wait_none

    call_counter = {"n": 0}

    async def _always_fails() -> None:
        call_counter["n"] += 1
        raise MCPConnectionError("Simulated SSE failure")

    with pytest.raises(MCPConnectionError):
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_none(),
            reraise=True,
        ):
            with attempt:
                await _always_fails()

    assert call_counter["n"] == 3, (
        f"Expected exactly 3 retry attempts, got {call_counter['n']}"
    )


@pytest.mark.asyncio
async def test_mcp_client_retry_succeeds_on_third_attempt():
    """
    Verify that if the first two calls fail but the third succeeds,
    tenacity returns the successful result without raising.
    """
    from tenacity import (
        AsyncRetrying,
        retry_if_exception_type,
        stop_after_attempt,
        wait_none,
    )

    call_count = {"n": 0}

    async def _unstable_call() -> List[ToolInfo]:
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise MCPConnectionError(f"Transient failure #{call_count['n']}")
        return [ToolInfo(name="execute_query", description="run sql")]

    results: List[ToolInfo] = []
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_none(),
        retry=retry_if_exception_type(MCPConnectionError),
        reraise=True,
    ):
        with attempt:
            results = await _unstable_call()

    assert len(results) == 1
    assert results[0].name == "execute_query"
    assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# 6. test_sql_validation_blocks_bad_sql (node-level + workflow-level)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sql_validation_blocks_bad_sql():
    """
    sql_validator_node must set error and validation_result.is_valid=False
    when the generated SQL contains a forbidden DDL keyword (DROP).
    """
    state = _make_state(
        generated_sql="DROP TABLE ANALYTICS.PUBLIC.SALES_FACT",
        selected_tables=["ANALYTICS.PUBLIC.SALES_FACT"],
    )

    result = await sql_validator_node(state)

    assert result.get("error"), "sql_validator_node must set 'error' for bad SQL"
    assert result["validation_result"].is_valid is False
    assert len(result["validation_result"].issues) > 0

    issues_text = " ".join(result["validation_result"].issues).lower()
    assert "drop" in issues_text or "forbidden" in issues_text, (
        f"Expected DROP to be flagged; issues: {result['validation_result'].issues}"
    )


@pytest.mark.asyncio
async def test_sql_validation_blocks_injection():
    """
    sql_validator_node must flag a stray semicolon (injection vector).
    """
    state = _make_state(
        generated_sql="SELECT * FROM sales; DROP TABLE users",
        selected_tables=["sales"],
    )

    result = await sql_validator_node(state)

    assert result.get("error"), "Semicolon injection must produce an error"
    assert result["validation_result"].is_valid is False
    issues_text = " ".join(result["validation_result"].issues).lower()
    assert "semicolon" in issues_text or "drop" in issues_text


@pytest.mark.asyncio
async def test_sql_validation_passes_valid_select():
    """
    sql_validator_node must not set error for a safe SELECT statement.
    """
    state = _make_state(
        generated_sql=(
            "SELECT region, SUM(revenue) AS total "
            "FROM ANALYTICS.PUBLIC.SALES_FACT "
            "WHERE year = 2024 "
            "GROUP BY region "
            "LIMIT 100"
        ),
        selected_tables=["ANALYTICS.PUBLIC.SALES_FACT"],
    )

    result = await sql_validator_node(state)

    assert not result.get("error"), (
        f"Valid SQL must not produce an error; got: {result.get('error')}"
    )
    assert result["validation_result"].is_valid is True
    assert result["validation_result"].issues == []


@pytest.mark.asyncio
async def test_workflow_skips_executor_on_bad_sql():
    """
    When sql_validator_node blocks bad SQL the workflow must skip
    sql_executor_node entirely (execute_query is never called).
    """
    intent_payload = json.dumps(
        {
            "intent": "sabotage",
            "metrics": [],
            "dimensions": [],
            "time_period": "N/A",
            "filters": [],
            "granularity": "N/A",
            "reasoning": "test",
        }
    )
    planner_payload = json.dumps({"plan": ["test plan"]})
    selector_payload = json.dumps(
        {"selected_tables": ["ANALYTICS.PUBLIC.ORDERS"], "reasoning": "test"}
    )
    # SQL generator returns a DROP statement to trigger the validator
    sql_payload = json.dumps(
        {"sql": "DROP TABLE ANALYTICS.PUBLIC.ORDERS", "reasoning": "malicious"}
    )
    analyst_payload = json.dumps(
        {
            "findings": ["Validation blocked the query."],
            "data_summary": "No data analysed.",
            "anomalies": [],
            "key_metrics": {},
            "confidence": "N/A",
        }
    )

    bad_sql_metadata = {
        "ANALYTICS": {"PUBLIC": {"ORDERS": [{"name": "id", "type": "NUMBER"}]}}
    }
    bad_sql_tools = [
        _make_tool_info("list_databases"),
        _make_tool_info("list_schemas"),
        _make_tool_info("list_tables"),
        _make_tool_info("describe_table"),
        _make_tool_info("execute_query"),
    ]

    # Both dataset_selector_node and sql_generator_node build ChatOpenAI locally.
    local_llm = MagicMock()
    local_llm.with_structured_output = MagicMock(return_value=local_llm)
    local_llm.ainvoke = AsyncMock(
        side_effect=[
            _llm_response(selector_payload),
            _llm_response(sql_payload),
        ]
    )

    def _local_chat(*_a: Any, **_kw: Any) -> MagicMock:
        return local_llm

    with (
        patch("graph.nodes.intent_node._llm") as m_intent,
        patch("graph.nodes.planner_node._llm") as m_plan,
        patch("graph.nodes.analyst_node._llm") as m_analyst,
        patch("graph.nodes.response_node._llm") as m_resp,
        patch("graph.nodes.dataset_selector_node.ChatOpenAI", side_effect=_local_chat),
        patch("graph.nodes.sql_generator_node.ChatOpenAI", side_effect=_local_chat),
        patch(
            "tools.mcp_client.SnowflakeMCPClient.list_available_tools",
            new_callable=AsyncMock,
            return_value=bad_sql_tools,
        ),
        patch(
            "tools.mcp_client.SnowflakeMCPClient.discover_metadata",
            new_callable=AsyncMock,
            return_value=bad_sql_metadata,
        ),
        patch(
            "tools.mcp_client.SnowflakeMCPClient.execute_query",
            new_callable=AsyncMock,
        ) as mock_execute,
    ):
        m_intent.ainvoke = AsyncMock(return_value=_llm_response(intent_payload))
        m_plan.ainvoke = AsyncMock(return_value=_llm_response(planner_payload))
        m_analyst.ainvoke = AsyncMock(return_value=_llm_response(analyst_payload))
        m_resp.ainvoke = AsyncMock(
            return_value=_llm_response(
                "## Executive Summary\n\nValidation blocked the query.\n"
            )
        )

        from graph.workflow import run_analysis  # noqa: PLC0415

        result = await run_analysis("drop the orders table please")

    mock_execute.assert_not_called()
    assert isinstance(result, dict)
    assert "answer" in result


# ---------------------------------------------------------------------------
# 7. test_metadata_discovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metadata_discovery():
    """
    metadata_discovery_node must:
    - Call list_available_tools() to discover tool names.
    - Call discover_metadata() to walk the schema tree.
    - Populate available_tool_names and available_metadata without setting error.
    """
    mock_tool_infos = [
        _make_tool_info("list_databases", "List all databases"),
        _make_tool_info("list_schemas", "List schemas in a database"),
        _make_tool_info("list_tables", "List tables in a schema"),
        _make_tool_info("describe_table", "Describe a table"),
        _make_tool_info("execute_query", "Execute a SQL query"),
    ]

    mock_metadata = {
        "SALES_DB": {
            "PUBLIC": {
                "ORDERS": [
                    {"name": "order_id", "type": "NUMBER"},
                    {"name": "amount", "type": "FLOAT"},
                    {"name": "region", "type": "VARCHAR"},
                ]
            },
            "REPORTING": {
                "REVENUE_SUMMARY": [
                    {"name": "period", "type": "VARCHAR"},
                    {"name": "revenue", "type": "FLOAT"},
                ]
            },
        }
    }

    state = _make_state()

    with (
        patch(
            "tools.mcp_client.SnowflakeMCPClient.list_available_tools",
            new_callable=AsyncMock,
            return_value=mock_tool_infos,
        ) as mock_list_tools,
        patch(
            "tools.mcp_client.SnowflakeMCPClient.discover_metadata",
            new_callable=AsyncMock,
            return_value=mock_metadata,
        ) as mock_discover,
    ):
        result = await metadata_discovery_node(state)

    mock_list_tools.assert_called_once()
    mock_discover.assert_called_once()

    assert not result.get("error"), (
        f"metadata_discovery_node set an unexpected error: {result.get('error')}"
    )

    assert "available_tool_names" in result
    assert set(result["available_tool_names"]) == {
        "list_databases",
        "list_schemas",
        "list_tables",
        "describe_table",
        "execute_query",
    }

    assert "available_metadata" in result
    metadata = result["available_metadata"]
    assert "SALES_DB" in metadata
    assert "PUBLIC" in metadata["SALES_DB"]
    assert "ORDERS" in metadata["SALES_DB"]["PUBLIC"]

    assert "execution_logs" in result
    assert len(result["execution_logs"]) > 0
    for log in result["execution_logs"]:
        assert hasattr(log, "node")
        assert log.node == "metadata_discovery_node"


@pytest.mark.asyncio
async def test_metadata_discovery_handles_mcp_error():
    """
    When list_available_tools() raises MCPConnectionError,
    metadata_discovery_node must return an error dict without raising
    and must not call discover_metadata.
    """
    state = _make_state()

    with (
        patch(
            "tools.mcp_client.SnowflakeMCPClient.list_available_tools",
            new_callable=AsyncMock,
            side_effect=MCPConnectionError("Cannot reach MCP server"),
        ),
        patch(
            "tools.mcp_client.SnowflakeMCPClient.discover_metadata",
            new_callable=AsyncMock,
        ) as mock_discover,
    ):
        result = await metadata_discovery_node(state)

    assert result.get("error"), (
        "metadata_discovery_node must set error when list_available_tools fails"
    )
    error_lower = result["error"].lower()
    assert "cannot reach mcp server" in error_lower or "list" in error_lower
    mock_discover.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Additional edge-case / contract tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_endpoint_propagates_workflow_error():
    """
    When run_analysis returns a soft error (error key is set), POST /analyze
    must still return 200 and include the error in the response body.
    """
    canned_error = {
        "answer": "The analysis could not be completed due to a processing error.",
        "analysis": "## Executive Summary\n\nError occurred.",
        "generated_sql": "",
        "selected_tables": [],
        "intent": "",
        "plan": [],
        "error": "MCP connection failed: timeout after 30s",
    }

    with patch("graph.workflow.run_analysis", new_callable=AsyncMock, return_value=canned_error):
        from app import create_app  # noqa: PLC0415

        test_app = create_app()
        async with httpx.AsyncClient(app=test_app, base_url="http://testserver") as client:
            response = await client.post(
                "/analyze", json={"question": "What is the revenue trend?"}
            )

    assert response.status_code == 200
    body = response.json()
    assert body.get("error") is not None
    assert "MCP" in body["error"] or "error" in body["error"].lower()


@pytest.mark.asyncio
async def test_analyze_endpoint_missing_question_field():
    """
    POST /analyze with no 'question' field must return 422.
    """
    from app import create_app  # noqa: PLC0415

    test_app = create_app()
    async with httpx.AsyncClient(app=test_app, base_url="http://testserver") as client:
        response = await client.post("/analyze", json={})

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 9. Unit tests for state dataclasses
# ---------------------------------------------------------------------------


def test_make_initial_state_defaults():
    """
    make_initial_state() must populate every required AgentState field with
    a sensible default so that node tests can build states safely.
    """
    state = make_initial_state("How much revenue did we make?")

    assert state["question"] == "How much revenue did we make?"
    assert state["intent"] == ""
    assert state["metrics"] == []
    assert state["dimensions"] == []
    assert state["time_period"] == ""
    assert state["plan"] == []
    assert state["available_metadata"] == {}
    assert state["available_tool_names"] == []
    assert state["selected_tables"] == []
    assert state["table_metadata"] == []
    assert state["generated_sql"] == ""
    assert state["validation_result"] is None
    assert state["query_results"] == []
    assert state["findings"] == []
    assert state["data_summary"] == ""
    assert state["anomalies"] == []
    assert state["answer"] == ""
    assert state["analysis"] == ""
    assert state["error"] is None
    assert state["execution_logs"] == []


def test_query_result_dataclass():
    """QueryResult dataclass fields and defaults work as expected."""
    qr = QueryResult(success=True, row_count=2)
    assert qr.success is True
    assert qr.row_count == 2
    assert qr.data == []
    assert qr.error is None
    assert qr.columns == []
    assert qr.execution_time_ms == 0.0


def test_sql_validation_result_dataclass():
    """SQLValidationResult dataclass fields and defaults work as expected."""
    vr_ok = SQLValidationResult(is_valid=True)
    assert vr_ok.is_valid is True
    assert vr_ok.issues == []
    assert vr_ok.warnings == []

    vr_fail = SQLValidationResult(is_valid=False, issues=["Forbidden keyword: DROP."])
    assert vr_fail.is_valid is False
    assert len(vr_fail.issues) == 1
