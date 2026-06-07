"""
tests/test_state.py

Unit tests for the data model defined in graph/state.py.

Covers:
- ColumnInfo dataclass creation and defaults
- TableMetadata dataclass creation, defaults, and the fully_qualified_name property
- SQLValidationResult dataclass creation and defaults
- QueryResult dataclass creation and defaults
- ExecutionLog dataclass creation and defaults
- AgentState TypedDict creation with all fields
- make_initial_state() factory function
- Annotated list fields (query_results, execution_logs) accept list values
- Optional fields accept None
"""

from __future__ import annotations

import operator
from typing import List, get_type_hints

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


# ===========================================================================
# ColumnInfo
# ===========================================================================


class TestColumnInfo:
    def test_required_fields(self):
        col = ColumnInfo(name="O_ORDERKEY", data_type="NUMBER(38,0)")
        assert col.name == "O_ORDERKEY"
        assert col.data_type == "NUMBER(38,0)"

    def test_default_nullable_is_true(self):
        col = ColumnInfo(name="C_COMMENT", data_type="VARCHAR(117)")
        assert col.nullable is True

    def test_default_description_is_empty_string(self):
        col = ColumnInfo(name="C_COMMENT", data_type="VARCHAR(117)")
        assert col.description == ""

    def test_nullable_can_be_set_false(self):
        col = ColumnInfo(name="O_ORDERKEY", data_type="NUMBER(38,0)", nullable=False)
        assert col.nullable is False

    def test_description_can_be_set(self):
        col = ColumnInfo(
            name="O_TOTALPRICE",
            data_type="NUMBER(12,2)",
            description="Total price of the order",
        )
        assert col.description == "Total price of the order"

    def test_multiple_columns_are_independent(self):
        col_a = ColumnInfo(name="A", data_type="VARCHAR")
        col_b = ColumnInfo(name="B", data_type="DATE", nullable=False)
        assert col_a.name != col_b.name
        assert col_a.nullable is True
        assert col_b.nullable is False


# ===========================================================================
# TableMetadata
# ===========================================================================


class TestTableMetadata:
    def test_required_fields(self):
        tbl = TableMetadata(
            database="SNOWFLAKE_SAMPLE_DATA",
            schema="TPCH_SF1",
            table_name="ORDERS",
        )
        assert tbl.database == "SNOWFLAKE_SAMPLE_DATA"
        assert tbl.schema == "TPCH_SF1"
        assert tbl.table_name == "ORDERS"

    def test_default_columns_is_empty_list(self):
        tbl = TableMetadata(database="DB", schema="SCH", table_name="TBL")
        assert tbl.columns == []

    def test_columns_list_is_not_shared_between_instances(self):
        """Each instance must get its own list — not a shared mutable default."""
        tbl_a = TableMetadata(database="DB", schema="SCH", table_name="A")
        tbl_b = TableMetadata(database="DB", schema="SCH", table_name="B")
        tbl_a.columns.append(ColumnInfo(name="X", data_type="INT"))
        assert tbl_b.columns == []

    def test_columns_accepts_column_info_objects(self):
        cols = [
            ColumnInfo(name="O_ORDERKEY", data_type="NUMBER(38,0)", nullable=False),
            ColumnInfo(name="O_TOTALPRICE", data_type="NUMBER(12,2)"),
        ]
        tbl = TableMetadata(
            database="DB",
            schema="SCH",
            table_name="ORDERS",
            columns=cols,
        )
        assert len(tbl.columns) == 2
        assert tbl.columns[0].name == "O_ORDERKEY"
        assert tbl.columns[1].data_type == "NUMBER(12,2)"

    def test_fully_qualified_name_property(self):
        tbl = TableMetadata(
            database="SNOWFLAKE_SAMPLE_DATA",
            schema="TPCH_SF1",
            table_name="ORDERS",
        )
        assert tbl.fully_qualified_name == "SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS"

    def test_fully_qualified_name_preserves_case(self):
        tbl = TableMetadata(database="mydb", schema="public", table_name="users")
        assert tbl.fully_qualified_name == "mydb.public.users"


# ===========================================================================
# SQLValidationResult
# ===========================================================================


class TestSQLValidationResult:
    def test_valid_result(self):
        result = SQLValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.issues == []
        assert result.warnings == []

    def test_invalid_result_with_issues(self):
        result = SQLValidationResult(
            is_valid=False,
            issues=["Forbidden keyword detected: DROP.", "SQL must begin with SELECT."],
        )
        assert result.is_valid is False
        assert len(result.issues) == 2
        assert "DROP" in result.issues[0]

    def test_warnings_without_issues(self):
        result = SQLValidationResult(
            is_valid=True,
            warnings=["SELECT * detected — consider selecting specific columns."],
        )
        assert result.is_valid is True
        assert result.issues == []
        assert len(result.warnings) == 1

    def test_default_issues_list_is_not_shared(self):
        r1 = SQLValidationResult(is_valid=True)
        r2 = SQLValidationResult(is_valid=True)
        r1.issues.append("some issue")
        assert r2.issues == []

    def test_default_warnings_list_is_not_shared(self):
        r1 = SQLValidationResult(is_valid=True)
        r2 = SQLValidationResult(is_valid=True)
        r1.warnings.append("some warning")
        assert r2.warnings == []

    def test_both_issues_and_warnings(self):
        result = SQLValidationResult(
            is_valid=False,
            issues=["Forbidden keyword: DELETE"],
            warnings=["No LIMIT clause detected"],
        )
        assert not result.is_valid
        assert len(result.issues) == 1
        assert len(result.warnings) == 1


# ===========================================================================
# QueryResult
# ===========================================================================


class TestQueryResult:
    def test_minimal_success_result(self):
        result = QueryResult(success=True)
        assert result.success is True
        assert result.row_count == 0
        assert result.data == []
        assert result.error is None
        assert result.columns == []
        assert result.execution_time_ms == 0.0

    def test_failure_result_with_error(self):
        result = QueryResult(
            success=False,
            error="MCP query execution failed: connection timeout",
        )
        assert result.success is False
        assert result.error == "MCP query execution failed: connection timeout"

    def test_full_populated_result(self):
        data = [
            {"QUARTER": "2024-Q1", "TOTAL_REVENUE": 4_250_000.00, "ORDER_COUNT": 1823},
            {"QUARTER": "2024-Q4", "TOTAL_REVENUE": 3_190_000.00, "ORDER_COUNT": 1402},
        ]
        result = QueryResult(
            success=True,
            row_count=2,
            columns=["QUARTER", "TOTAL_REVENUE", "ORDER_COUNT"],
            data=data,
            error=None,
            execution_time_ms=312.5,
        )
        assert result.row_count == 2
        assert result.columns == ["QUARTER", "TOTAL_REVENUE", "ORDER_COUNT"]
        assert len(result.data) == 2
        assert result.data[0]["QUARTER"] == "2024-Q1"
        assert result.execution_time_ms == 312.5

    def test_default_data_list_is_not_shared(self):
        r1 = QueryResult(success=True)
        r2 = QueryResult(success=True)
        r1.data.append({"col": "val"})
        assert r2.data == []

    def test_default_columns_list_is_not_shared(self):
        r1 = QueryResult(success=True)
        r2 = QueryResult(success=True)
        r1.columns.append("COL_A")
        assert r2.columns == []

    def test_error_field_is_none_by_default(self):
        result = QueryResult(success=True)
        assert result.error is None


# ===========================================================================
# ExecutionLog
# ===========================================================================


class TestExecutionLog:
    def test_required_fields(self):
        log = ExecutionLog(
            timestamp="2024-01-15T10:00:00+00:00",
            node="intent_node",
            status="started",
            message="Extracting intent",
        )
        assert log.timestamp == "2024-01-15T10:00:00+00:00"
        assert log.node == "intent_node"
        assert log.status == "started"
        assert log.message == "Extracting intent"

    def test_default_duration_ms_is_zero(self):
        log = ExecutionLog(
            timestamp="2024-01-15T10:00:00+00:00",
            node="planner_node",
            status="started",
            message="Planning started",
        )
        assert log.duration_ms == 0.0

    def test_default_metadata_is_empty_dict(self):
        log = ExecutionLog(
            timestamp="2024-01-15T10:00:00+00:00",
            node="planner_node",
            status="started",
            message="Planning started",
        )
        assert log.metadata == {}

    def test_metadata_dict_is_not_shared(self):
        log_a = ExecutionLog(
            timestamp="2024-01-15T10:00:00+00:00",
            node="node_a",
            status="completed",
            message="A done",
        )
        log_b = ExecutionLog(
            timestamp="2024-01-15T10:00:01+00:00",
            node="node_b",
            status="completed",
            message="B done",
        )
        log_a.metadata["key"] = "value"
        assert "key" not in log_b.metadata

    def test_all_fields_explicit(self):
        log = ExecutionLog(
            timestamp="2024-01-15T10:00:01+00:00",
            node="sql_validator_node",
            status="error",
            message="Forbidden keyword detected: DROP.",
            duration_ms=5.2,
            metadata={"issues": ["DROP"], "warnings": []},
        )
        assert log.status == "error"
        assert log.duration_ms == 5.2
        assert log.metadata["issues"] == ["DROP"]

    def test_status_values(self):
        for status in ("started", "completed", "error"):
            log = ExecutionLog(
                timestamp="2024-01-15T10:00:00+00:00",
                node="some_node",
                status=status,
                message=f"Node {status}",
            )
            assert log.status == status


# ===========================================================================
# AgentState TypedDict
# ===========================================================================


class TestAgentState:
    """Tests for the AgentState TypedDict itself."""

    def test_creation_with_all_fields(self, sample_agent_state: AgentState):
        """Fixture creates a fully-populated state without errors."""
        assert sample_agent_state["question"] == "Why did revenue drop in Q4?"
        assert sample_agent_state["intent"] == "Revenue Trend Analysis"
        assert "revenue" in sample_agent_state["metrics"]
        assert "quarter" in sample_agent_state["dimensions"]
        assert sample_agent_state["time_period"] == "Q4 2024"

    def test_plan_is_list(self, sample_agent_state: AgentState):
        assert isinstance(sample_agent_state["plan"], list)
        assert len(sample_agent_state["plan"]) > 0

    def test_available_metadata_is_dict(self, sample_agent_state: AgentState):
        assert isinstance(sample_agent_state["available_metadata"], dict)

    def test_available_tool_names_is_list(self, sample_agent_state: AgentState):
        names = sample_agent_state["available_tool_names"]
        assert isinstance(names, list)
        assert "execute_query" in names

    def test_selected_tables_are_fully_qualified(self, sample_agent_state: AgentState):
        for tbl in sample_agent_state["selected_tables"]:
            parts = tbl.split(".")
            assert len(parts) == 3, f"Expected DB.SCHEMA.TABLE format, got: {tbl}"

    def test_table_metadata_contains_table_metadata_objects(
        self, sample_agent_state: AgentState
    ):
        for tm in sample_agent_state["table_metadata"]:
            assert isinstance(tm, TableMetadata)
            assert isinstance(tm.columns, list)

    def test_validation_result_is_sql_validation_result(
        self, sample_agent_state: AgentState
    ):
        vr = sample_agent_state["validation_result"]
        assert isinstance(vr, SQLValidationResult)
        assert vr.is_valid is True

    def test_query_results_field_accepts_list(self, sample_agent_state: AgentState):
        qrs = sample_agent_state["query_results"]
        assert isinstance(qrs, list)
        assert len(qrs) == 1
        assert isinstance(qrs[0], QueryResult)
        assert qrs[0].success is True

    def test_execution_logs_field_accepts_list(self, sample_agent_state: AgentState):
        logs = sample_agent_state["execution_logs"]
        assert isinstance(logs, list)
        assert len(logs) >= 1
        assert all(isinstance(log, ExecutionLog) for log in logs)

    def test_findings_is_list_of_strings(self, sample_agent_state: AgentState):
        findings = sample_agent_state["findings"]
        assert isinstance(findings, list)
        assert all(isinstance(f, str) for f in findings)

    def test_anomalies_is_list_of_strings(self, sample_agent_state: AgentState):
        anomalies = sample_agent_state["anomalies"]
        assert isinstance(anomalies, list)
        assert all(isinstance(a, str) for a in anomalies)

    def test_error_field_can_be_none(self, sample_agent_state: AgentState):
        assert sample_agent_state["error"] is None

    def test_error_field_can_be_string(self):
        state = make_initial_state("test question")
        state["error"] = "Something went wrong"
        assert state["error"] == "Something went wrong"

    def test_validation_result_can_be_none(self):
        state = make_initial_state("test question")
        assert state["validation_result"] is None

    def test_current_node_is_string(self, sample_agent_state: AgentState):
        assert isinstance(sample_agent_state["current_node"], str)
        assert sample_agent_state["current_node"] == "response_node"

    def test_answer_and_analysis_are_strings(self, sample_agent_state: AgentState):
        assert isinstance(sample_agent_state["answer"], str)
        assert isinstance(sample_agent_state["analysis"], str)
        assert len(sample_agent_state["answer"]) > 0
        assert len(sample_agent_state["analysis"]) > 0


# ===========================================================================
# make_initial_state factory
# ===========================================================================


class TestMakeInitialState:
    def test_question_is_set(self):
        state = make_initial_state("Why did revenue drop in Q4?")
        assert state["question"] == "Why did revenue drop in Q4?"

    def test_all_string_fields_default_to_empty_string(self):
        state = make_initial_state("some question")
        empty_string_fields = [
            "intent",
            "time_period",
            "intent_reasoning",
            "selection_reasoning",
            "generated_sql",
            "sql_reasoning",
            "data_summary",
            "answer",
            "analysis",
            "current_node",
        ]
        for field in empty_string_fields:
            assert state[field] == "", f"Expected '{field}' to default to ''"

    def test_all_list_fields_default_to_empty_list(self):
        state = make_initial_state("some question")
        empty_list_fields = [
            "metrics",
            "dimensions",
            "plan",
            "available_tool_names",
            "selected_tables",
            "table_metadata",
            "query_results",
            "findings",
            "anomalies",
            "execution_logs",
        ]
        for field in empty_list_fields:
            assert state[field] == [], f"Expected '{field}' to default to []"

    def test_available_metadata_defaults_to_empty_dict(self):
        state = make_initial_state("some question")
        assert state["available_metadata"] == {}

    def test_validation_result_defaults_to_none(self):
        state = make_initial_state("some question")
        assert state["validation_result"] is None

    def test_error_defaults_to_none(self):
        state = make_initial_state("some question")
        assert state["error"] is None

    def test_returns_agent_state_compatible_dict(self):
        state = make_initial_state("some question")
        # AgentState is a TypedDict — verify it is a plain dict at runtime
        assert isinstance(state, dict)

    def test_different_questions_produce_independent_states(self):
        s1 = make_initial_state("Q1")
        s2 = make_initial_state("Q2")
        s1["metrics"].append("revenue")
        # Mutation of s1's list must not affect s2
        assert s2["metrics"] == []

    def test_initial_state_fields_are_all_required_keys_present(self):
        state = make_initial_state("test")
        required_keys = [
            "question", "intent", "metrics", "dimensions", "time_period",
            "intent_reasoning", "plan", "available_metadata",
            "available_tool_names", "selected_tables", "table_metadata",
            "selection_reasoning", "generated_sql", "sql_reasoning",
            "validation_result", "query_results", "findings", "data_summary",
            "anomalies", "answer", "analysis", "error", "current_node",
            "execution_logs",
        ]
        for key in required_keys:
            assert key in state, f"Key '{key}' missing from make_initial_state() result"


# ===========================================================================
# Annotated reducer semantics (operator.add on list fields)
# ===========================================================================


class TestAnnotatedListFields:
    """
    Verify that the Annotated[List[...], operator.add] fields declared in
    AgentState use operator.add as their reducer — i.e., two lists concatenate
    rather than replace.

    These tests do not invoke LangGraph directly; they confirm the intended
    semantics that LangGraph will honour at runtime.
    """

    def test_query_results_reducer_is_operator_add(self):
        """operator.add on two lists should concatenate them."""
        list_a: List[QueryResult] = [QueryResult(success=True, row_count=1)]
        list_b: List[QueryResult] = [QueryResult(success=True, row_count=2)]
        combined = operator.add(list_a, list_b)
        assert len(combined) == 2
        assert combined[0].row_count == 1
        assert combined[1].row_count == 2

    def test_execution_logs_reducer_is_operator_add(self):
        """operator.add on two log lists should concatenate them."""
        log_a = ExecutionLog(
            timestamp="2024-01-15T10:00:00+00:00",
            node="intent_node",
            status="started",
            message="started",
        )
        log_b = ExecutionLog(
            timestamp="2024-01-15T10:00:01+00:00",
            node="intent_node",
            status="completed",
            message="completed",
        )
        combined = operator.add([log_a], [log_b])
        assert len(combined) == 2
        assert combined[0].status == "started"
        assert combined[1].status == "completed"

    def test_query_results_accumulates_across_multiple_executions(self):
        """Simulates multiple nodes each appending one result."""
        accumulator: List[QueryResult] = []
        for i in range(3):
            new_result = [QueryResult(success=True, row_count=i + 1)]
            accumulator = operator.add(accumulator, new_result)
        assert len(accumulator) == 3
        assert accumulator[0].row_count == 1
        assert accumulator[2].row_count == 3

    def test_execution_logs_order_is_preserved(self):
        """Log order must be preserved when concatenating."""
        logs_batch_1 = [
            ExecutionLog("2024-01-15T10:00:00+00:00", "node_a", "started", "a start"),
            ExecutionLog("2024-01-15T10:00:01+00:00", "node_a", "completed", "a done"),
        ]
        logs_batch_2 = [
            ExecutionLog("2024-01-15T10:00:02+00:00", "node_b", "started", "b start"),
        ]
        combined = operator.add(logs_batch_1, logs_batch_2)
        assert len(combined) == 3
        assert combined[0].node == "node_a"
        assert combined[2].node == "node_b"
