"""
tests/conftest.py

Shared pytest fixtures for the AI Data Analyst Agent test suite.

All Snowflake access is mocked — no direct snowflake-connector-python
or live MCP connections are made during tests.

NOTE: OPENAI_API_KEY is set to a dummy value before any project imports so
that the module-level ChatOpenAI singleton in intent_node.py can be
instantiated without a real key.  No live OpenAI calls are made during tests.
"""

from __future__ import annotations

import asyncio
import os

# Set dummy env vars before any project module is imported.  Some nodes
# (intent_node, planner_node, etc.) create a ChatOpenAI singleton at
# module-level, which raises OpenAIError if the key is absent.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-for-unit-tests")
os.environ.setdefault("MCP_SERVER_URL", "http://localhost:8080/sse")
os.environ.setdefault("MCP_BEARER_TOKEN", "test-bearer-token")

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graph.state import (
    AgentState,
    ColumnInfo,
    ExecutionLog,
    QueryResult,
    SQLValidationResult,
    TableMetadata,
    make_initial_state,
)


# ---------------------------------------------------------------------------
# Simple value fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_question() -> str:
    """A realistic business question used across many tests."""
    return "Why did revenue drop in Q4?"


# ---------------------------------------------------------------------------
# Metadata fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """
    TPCH-like nested metadata tree mirroring what SnowflakeMCPClient
    returns from discover_metadata().

    Structure: {database: {schema: {table: [column_dicts]}}}
    """
    return {
        "SNOWFLAKE_SAMPLE_DATA": {
            "TPCH_SF1": {
                "ORDERS": [
                    {"name": "O_ORDERKEY", "type": "NUMBER(38,0)", "nullable": False},
                    {"name": "O_CUSTKEY", "type": "NUMBER(38,0)", "nullable": False},
                    {"name": "O_ORDERSTATUS", "type": "VARCHAR(1)", "nullable": False},
                    {"name": "O_TOTALPRICE", "type": "NUMBER(12,2)", "nullable": False},
                    {"name": "O_ORDERDATE", "type": "DATE", "nullable": False},
                    {"name": "O_ORDERPRIORITY", "type": "VARCHAR(15)", "nullable": False},
                    {"name": "O_SHIPPRIORITY", "type": "NUMBER(38,0)", "nullable": False},
                    {"name": "O_COMMENT", "type": "VARCHAR(79)", "nullable": True},
                ],
                "LINEITEM": [
                    {"name": "L_ORDERKEY", "type": "NUMBER(38,0)", "nullable": False},
                    {"name": "L_PARTKEY", "type": "NUMBER(38,0)", "nullable": False},
                    {"name": "L_SUPPKEY", "type": "NUMBER(38,0)", "nullable": False},
                    {"name": "L_LINENUMBER", "type": "NUMBER(38,0)", "nullable": False},
                    {"name": "L_QUANTITY", "type": "NUMBER(12,2)", "nullable": False},
                    {"name": "L_EXTENDEDPRICE", "type": "NUMBER(12,2)", "nullable": False},
                    {"name": "L_DISCOUNT", "type": "NUMBER(12,2)", "nullable": False},
                    {"name": "L_TAX", "type": "NUMBER(12,2)", "nullable": False},
                    {"name": "L_RETURNFLAG", "type": "VARCHAR(1)", "nullable": False},
                    {"name": "L_LINESTATUS", "type": "VARCHAR(1)", "nullable": False},
                    {"name": "L_SHIPDATE", "type": "DATE", "nullable": False},
                    {"name": "L_COMMITDATE", "type": "DATE", "nullable": False},
                    {"name": "L_RECEIPTDATE", "type": "DATE", "nullable": False},
                    {"name": "L_SHIPINSTRUCT", "type": "VARCHAR(25)", "nullable": False},
                    {"name": "L_SHIPMODE", "type": "VARCHAR(10)", "nullable": False},
                    {"name": "L_COMMENT", "type": "VARCHAR(44)", "nullable": True},
                ],
                "CUSTOMER": [
                    {"name": "C_CUSTKEY", "type": "NUMBER(38,0)", "nullable": False},
                    {"name": "C_NAME", "type": "VARCHAR(25)", "nullable": False},
                    {"name": "C_ADDRESS", "type": "VARCHAR(40)", "nullable": False},
                    {"name": "C_NATIONKEY", "type": "NUMBER(38,0)", "nullable": False},
                    {"name": "C_PHONE", "type": "VARCHAR(15)", "nullable": False},
                    {"name": "C_ACCTBAL", "type": "NUMBER(12,2)", "nullable": False},
                    {"name": "C_MKTSEGMENT", "type": "VARCHAR(10)", "nullable": False},
                    {"name": "C_COMMENT", "type": "VARCHAR(117)", "nullable": True},
                ],
            }
        }
    }


# ---------------------------------------------------------------------------
# QueryResult fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_query_result() -> QueryResult:
    """
    A realistic QueryResult as produced by sql_executor_node —
    represents quarterly revenue data used in revenue-drop analysis.
    """
    return QueryResult(
        success=True,
        row_count=4,
        columns=["QUARTER", "TOTAL_REVENUE", "ORDER_COUNT"],
        data=[
            {"QUARTER": "2024-Q1", "TOTAL_REVENUE": 4_250_000.00, "ORDER_COUNT": 1823},
            {"QUARTER": "2024-Q2", "TOTAL_REVENUE": 4_780_000.00, "ORDER_COUNT": 2015},
            {"QUARTER": "2024-Q3", "TOTAL_REVENUE": 4_620_000.00, "ORDER_COUNT": 1944},
            {"QUARTER": "2024-Q4", "TOTAL_REVENUE": 3_190_000.00, "ORDER_COUNT": 1402},
        ],
        error=None,
        execution_time_ms=312.5,
    )


# ---------------------------------------------------------------------------
# MCP client mock
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mcp_client() -> AsyncMock:
    """
    AsyncMock for SnowflakeMCPClient.

    Pre-configured with realistic return values so individual tests do not
    need to patch every method unless they want to override specific behaviour.
    """
    client = AsyncMock()

    # list_available_tools
    from tools.mcp_client import ToolInfo

    client.list_available_tools.return_value = [
        ToolInfo(name="list_databases", description="List all databases"),
        ToolInfo(name="list_schemas", description="List schemas in a database"),
        ToolInfo(name="list_tables", description="List tables in a schema"),
        ToolInfo(name="describe_table", description="Describe a table's columns"),
        ToolInfo(name="execute_query", description="Execute a SQL query"),
    ]

    # list_databases
    client.list_databases.return_value = ["SNOWFLAKE_SAMPLE_DATA"]

    # list_schemas
    client.list_schemas.return_value = ["TPCH_SF1"]

    # list_tables
    client.list_tables.return_value = ["ORDERS", "LINEITEM", "CUSTOMER"]

    # describe_table
    client.describe_table.return_value = {
        "columns": [
            {"name": "O_ORDERKEY", "type": "NUMBER(38,0)"},
            {"name": "O_TOTALPRICE", "type": "NUMBER(12,2)"},
            {"name": "O_ORDERDATE", "type": "DATE"},
        ]
    }

    # execute_query — returns a tools.mcp_client.QueryResult
    from tools.mcp_client import QueryResult as MCPQueryResult

    client.execute_query.return_value = MCPQueryResult(
        columns=["QUARTER", "TOTAL_REVENUE", "ORDER_COUNT"],
        rows=[
            ["2024-Q1", 4_250_000.00, 1823],
            ["2024-Q2", 4_780_000.00, 2015],
            ["2024-Q3", 4_620_000.00, 1944],
            ["2024-Q4", 3_190_000.00, 1402],
        ],
        row_count=4,
        sql="SELECT quarter, SUM(total_price) AS total_revenue FROM orders GROUP BY 1",
        truncated=False,
    )

    # discover_metadata
    client.discover_metadata.return_value = {
        "SNOWFLAKE_SAMPLE_DATA": {
            "TPCH_SF1": {
                "ORDERS": [
                    {"name": "O_ORDERKEY", "type": "NUMBER(38,0)"},
                    {"name": "O_TOTALPRICE", "type": "NUMBER(12,2)"},
                    {"name": "O_ORDERDATE", "type": "DATE"},
                ],
                "LINEITEM": [
                    {"name": "L_ORDERKEY", "type": "NUMBER(38,0)"},
                    {"name": "L_EXTENDEDPRICE", "type": "NUMBER(12,2)"},
                ],
            }
        }
    }

    return client


# ---------------------------------------------------------------------------
# AgentState fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_agent_state(sample_query_result: QueryResult) -> AgentState:
    """
    A fully-populated AgentState representing a workflow that has run
    successfully through all nodes for the revenue-drop question.
    """
    validation = SQLValidationResult(
        is_valid=True,
        issues=[],
        warnings=["No LIMIT / TOP clause detected — large result sets may affect performance."],
    )

    orders_table = TableMetadata(
        database="SNOWFLAKE_SAMPLE_DATA",
        schema="TPCH_SF1",
        table_name="ORDERS",
        columns=[
            ColumnInfo(name="O_ORDERKEY", data_type="NUMBER(38,0)", nullable=False),
            ColumnInfo(name="O_CUSTKEY", data_type="NUMBER(38,0)", nullable=False),
            ColumnInfo(name="O_TOTALPRICE", data_type="NUMBER(12,2)", nullable=False),
            ColumnInfo(name="O_ORDERDATE", data_type="DATE", nullable=False),
        ],
    )

    lineitem_table = TableMetadata(
        database="SNOWFLAKE_SAMPLE_DATA",
        schema="TPCH_SF1",
        table_name="LINEITEM",
        columns=[
            ColumnInfo(name="L_ORDERKEY", data_type="NUMBER(38,0)", nullable=False),
            ColumnInfo(name="L_EXTENDEDPRICE", data_type="NUMBER(12,2)", nullable=False),
            ColumnInfo(name="L_DISCOUNT", data_type="NUMBER(12,2)", nullable=False),
            ColumnInfo(name="L_SHIPDATE", data_type="DATE", nullable=False),
        ],
    )

    start_log = ExecutionLog(
        timestamp="2024-01-15T10:00:00+00:00",
        node="intent_node",
        status="started",
        message="Extracting intent from question",
    )
    completed_log = ExecutionLog(
        timestamp="2024-01-15T10:00:01+00:00",
        node="intent_node",
        status="completed",
        message="Intent extracted: 'Revenue Trend Analysis'",
        duration_ms=842.3,
        metadata={
            "intent": "Revenue Trend Analysis",
            "metrics": ["revenue"],
            "dimensions": ["quarter"],
        },
    )

    generated_sql = (
        "SELECT\n"
        "    DATE_TRUNC('quarter', O_ORDERDATE) AS quarter,\n"
        "    SUM(O_TOTALPRICE) AS total_revenue,\n"
        "    COUNT(*) AS order_count\n"
        "FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS\n"
        "WHERE O_ORDERDATE >= '2024-01-01'\n"
        "GROUP BY 1\n"
        "ORDER BY 1"
    )

    return AgentState(
        # Input
        question="Why did revenue drop in Q4?",
        # Intent
        intent="Revenue Trend Analysis",
        metrics=["revenue", "total_price"],
        dimensions=["quarter", "region"],
        time_period="Q4 2024",
        intent_reasoning=(
            "The question asks about a revenue decline in Q4, "
            "suggesting a trend analysis comparing quarters."
        ),
        # Plan
        plan=[
            "Identify revenue metrics in the data model.",
            "Aggregate revenue by quarter for 2024.",
            "Compare Q4 against preceding quarters.",
            "Surface anomalies and potential causal factors.",
        ],
        # Metadata
        available_metadata={
            "SNOWFLAKE_SAMPLE_DATA": {
                "TPCH_SF1": {
                    "ORDERS": [{"name": "O_TOTALPRICE", "type": "NUMBER(12,2)"}],
                    "LINEITEM": [{"name": "L_EXTENDEDPRICE", "type": "NUMBER(12,2)"}],
                }
            }
        },
        available_tool_names=[
            "list_databases",
            "list_schemas",
            "list_tables",
            "describe_table",
            "execute_query",
        ],
        # Dataset selection
        selected_tables=[
            "SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS",
            "SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.LINEITEM",
        ],
        table_metadata=[orders_table, lineitem_table],
        selection_reasoning=(
            "ORDERS contains O_TOTALPRICE and O_ORDERDATE, which are "
            "necessary to compute quarterly revenue totals."
        ),
        # SQL
        generated_sql=generated_sql,
        sql_reasoning=(
            "Grouping by DATE_TRUNC quarter and summing O_TOTALPRICE "
            "gives quarterly revenue; filtering from 2024-01-01 scopes "
            "the analysis to the relevant year."
        ),
        # Validation
        validation_result=validation,
        # Execution
        query_results=[sample_query_result],
        # Analysis
        findings=[
            "Q4 2024 revenue was $3,190,000 — a 31% drop vs Q3 ($4,620,000).",
            "Order count also fell from 1,944 (Q3) to 1,402 (Q4), a 28% decline.",
            "Q2 was the peak quarter at $4,780,000.",
        ],
        data_summary=(
            "Quarterly revenue for 2024: Q1=$4.25M, Q2=$4.78M, Q3=$4.62M, Q4=$3.19M. "
            "Revenue declined sharply in Q4 relative to all prior quarters."
        ),
        anomalies=[
            "Q4 order count drop of 28% is anomalously large compared to Q1-Q3 variance (<10%).",
        ],
        # Response
        answer=(
            "Revenue dropped 31% in Q4 2024, falling from $4.62M in Q3 to $3.19M. "
            "Order volumes declined in parallel (28%), suggesting reduced demand "
            "rather than a pricing or margin issue."
        ),
        analysis=(
            "The data shows a consistent revenue trend through Q1-Q3 2024, "
            "followed by a significant contraction in Q4. The simultaneous drop "
            "in both revenue and order count points to a demand-side driver. "
            "Recommended next steps: segment by customer cohort and product line "
            "to isolate the root cause."
        ),
        # Control flow
        error=None,
        current_node="response_node",
        execution_logs=[start_log, completed_log],
    )


# ---------------------------------------------------------------------------
# OpenAI mock
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_openai():
    """
    Patch ChatOpenAI so tests do not make live API calls.

    The mock returns a deterministic AIMessage whose content is a valid JSON
    string covering the fields expected by intent_node.
    """
    import json

    from langchain_core.messages import AIMessage

    deterministic_response = json.dumps(
        {
            "intent": "Revenue Trend Analysis",
            "metrics": ["revenue", "total_price"],
            "dimensions": ["quarter"],
            "time_period": "Q4 2024",
            "filters": [],
            "granularity": "quarterly",
            "reasoning": (
                "The question asks about a revenue decline in Q4, "
                "suggesting a trend analysis comparing quarters."
            ),
        }
    )

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=deterministic_response))

    with patch("graph.nodes.intent_node._llm", mock_llm):
        yield mock_llm
