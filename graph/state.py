"""
LangGraph TypedDict state model for AI Data Analyst Agent.

All state fields use TypedDict for LangGraph compatibility.
List fields that accumulate across nodes use Annotated with operator.add reducer.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from typing import Annotated

from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ColumnInfo:
    """Metadata for a single table column."""

    name: str
    data_type: str
    nullable: bool = True
    description: str = ""


@dataclass
class TableMetadata:
    """Metadata for a single database table."""

    database: str
    schema: str
    table_name: str
    columns: List[ColumnInfo] = field(default_factory=list)

    @property
    def fully_qualified_name(self) -> str:
        return f"{self.database}.{self.schema}.{self.table_name}"


@dataclass
class SQLValidationResult:
    """Result of SQL safety / correctness validation."""

    is_valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class QueryResult:
    """Result of a single SQL query execution via MCP."""

    success: bool
    row_count: int = 0
    data: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


@dataclass
class ExecutionLog:
    """Structured log entry emitted by each graph node."""

    timestamp: str          # ISO-8601 string
    node: str               # node name, e.g. "intent_node"
    status: str             # "started" | "completed" | "error"
    message: str
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main LangGraph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    Central state object threaded through every node in the analysis workflow.

    Accumulating list fields (query_results, execution_logs) are annotated with
    operator.add so LangGraph merges partial updates by concatenation rather
    than replacement.
    """

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    question: str

    # ------------------------------------------------------------------
    # Intent extraction  (intent_node writes)
    # ------------------------------------------------------------------
    intent: str                  # e.g. "Revenue Trend Analysis"
    metrics: List[str]           # e.g. ["revenue", "sales"]
    dimensions: List[str]        # e.g. ["region", "product"]
    time_period: str             # e.g. "Q4 2024"
    intent_reasoning: str

    # ------------------------------------------------------------------
    # Planning  (planner_node writes)
    # ------------------------------------------------------------------
    plan: List[str]              # ordered list of investigation steps

    # ------------------------------------------------------------------
    # Metadata discovery  (metadata_discovery_node writes)
    # ------------------------------------------------------------------
    available_metadata: Dict[str, Any]   # full metadata tree from MCP
    available_tool_names: List[str]      # actual MCP tool names discovered

    # ------------------------------------------------------------------
    # Dataset selection  (dataset_selector_node writes)
    # ------------------------------------------------------------------
    selected_tables: List[str]           # fully-qualified: DB.SCHEMA.TABLE
    table_metadata: List[TableMetadata]  # per-table column details
    selection_reasoning: str

    # ------------------------------------------------------------------
    # SQL generation  (sql_generator_node writes)
    # ------------------------------------------------------------------
    generated_sql: str
    sql_reasoning: str

    # ------------------------------------------------------------------
    # SQL validation  (sql_validator_node writes)
    # ------------------------------------------------------------------
    validation_result: Optional[SQLValidationResult]

    # ------------------------------------------------------------------
    # SQL execution  (sql_executor_node writes)
    # Annotated with operator.add: each execution appends to the list.
    # ------------------------------------------------------------------
    query_results: Annotated[List[QueryResult], operator.add]

    # ------------------------------------------------------------------
    # Analysis  (analyst_node writes)
    # ------------------------------------------------------------------
    findings: List[str]
    data_summary: str
    anomalies: List[str]

    # ------------------------------------------------------------------
    # Response generation  (response_node writes)
    # ------------------------------------------------------------------
    answer: str
    analysis: str

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------
    error: Optional[str]
    current_node: str
    # Annotated with operator.add: every node appends its log entry.
    execution_logs: Annotated[List[ExecutionLog], operator.add]


# ---------------------------------------------------------------------------
# Default / empty state factory (useful for testing and initialisation)
# ---------------------------------------------------------------------------

def make_initial_state(question: str) -> AgentState:
    """Return an AgentState with sensible defaults for all optional fields."""
    return AgentState(
        question=question,
        intent="",
        metrics=[],
        dimensions=[],
        time_period="",
        intent_reasoning="",
        plan=[],
        available_metadata={},
        available_tool_names=[],
        selected_tables=[],
        table_metadata=[],
        selection_reasoning="",
        generated_sql="",
        sql_reasoning="",
        validation_result=None,
        query_results=[],
        findings=[],
        data_summary="",
        anomalies=[],
        answer="",
        analysis="",
        error=None,
        current_node="",
        execution_logs=[],
    )
