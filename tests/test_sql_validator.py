"""
tests/test_sql_validator.py — Unit tests for sql_validator_node.py

Covers:
  - check_select_only (read-only / injection enforcement)
  - check_table_whitelist (allowed-table enforcement)
  - extract_table_references (regex parser)
  - sql_validator_node (full async LangGraph node)
"""

import pytest

from graph.nodes.sql_validator_node import (
    check_select_only,
    check_table_whitelist,
    extract_table_references,
    sql_validator_node,
)
from graph.state import make_initial_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(sql: str, selected_tables=None) -> dict:
    """Build a minimal AgentState dict for the validator node."""
    state = make_initial_state("test question")
    state["generated_sql"] = sql
    state["selected_tables"] = selected_tables or []
    return state


# ===========================================================================
# check_select_only
# ===========================================================================


class TestCheckSelectOnly:

    # -----------------------------------------------------------------------
    # Valid queries — should return no issues
    # -----------------------------------------------------------------------

    def test_valid_select_query(self):
        sql = "SELECT id, name FROM orders WHERE status = 'active'"
        issues = check_select_only(sql)
        assert issues == [], f"Expected no issues, got: {issues}"

    def test_valid_cte_with_query(self):
        sql = (
            "WITH monthly AS ("
            "  SELECT DATE_TRUNC('month', order_date) AS month, SUM(amount) AS total"
            "  FROM orders"
            "  GROUP BY 1"
            ")"
            "SELECT month, total FROM monthly ORDER BY month"
        )
        issues = check_select_only(sql)
        assert issues == [], f"Expected no issues, got: {issues}"

    def test_union_select_is_allowed(self):
        """Plain UNION SELECT (no system-table references) is legitimate SQL."""
        sql = (
            "SELECT id, name FROM customers "
            "UNION SELECT id, name FROM archived_customers"
        )
        issues = check_select_only(sql)
        # The suspicious-UNION check only triggers for system-table references.
        assert not any("UNION" in i for i in issues), (
            "Plain UNION SELECT should not be blocked"
        )

    def test_sql_with_comments_is_handled(self):
        """SQL that starts with SELECT and also contains /* */ inline comments
        should pass all read-only / injection checks.

        Note: the validator does not strip leading -- line comments before the
        first-token check, so SQL that opens with a -- comment line will fail
        the 'must start with SELECT or WITH' rule.  A SELECT-first query with
        only inline /* */ comments passes cleanly.
        """
        sql = (
            "SELECT customer_id, total_spent\n"
            "FROM customers /* main table */\n"
            "WHERE total_spent > 1000\n"
            "LIMIT 100"
        )
        issues = check_select_only(sql)
        assert issues == [], f"Expected no issues, got: {issues}"

    # -----------------------------------------------------------------------
    # DML / DDL — must be blocked
    # -----------------------------------------------------------------------

    def test_insert_is_blocked(self):
        sql = "INSERT INTO orders (id, amount) VALUES (1, 100)"
        issues = check_select_only(sql)
        assert any("INSERT" in i for i in issues), "INSERT should be blocked"

    def test_update_is_blocked(self):
        sql = "UPDATE orders SET status = 'closed' WHERE id = 42"
        issues = check_select_only(sql)
        assert any("UPDATE" in i for i in issues), "UPDATE should be blocked"

    def test_delete_is_blocked(self):
        sql = "DELETE FROM orders WHERE id = 42"
        issues = check_select_only(sql)
        assert any("DELETE" in i for i in issues), "DELETE should be blocked"

    def test_drop_table_is_blocked(self):
        sql = "DROP TABLE orders"
        issues = check_select_only(sql)
        assert any("DROP" in i for i in issues), "DROP should be blocked"

    def test_create_table_is_blocked(self):
        sql = "CREATE TABLE new_orders AS SELECT * FROM orders"
        issues = check_select_only(sql)
        assert any("CREATE" in i for i in issues), "CREATE should be blocked"

    # -----------------------------------------------------------------------
    # Injection patterns
    # -----------------------------------------------------------------------

    def test_semicolon_injection_blocked(self):
        """A stray semicolon outside a string literal is flagged."""
        sql = "SELECT * FROM orders; DROP TABLE orders"
        issues = check_select_only(sql)
        assert any("Semicolon" in i or "semicolon" in i for i in issues), (
            "Semicolon injection should be detected"
        )

    def test_semicolon_inside_string_is_allowed(self):
        """A semicolon that lives inside a string literal is not injection."""
        sql = "SELECT id FROM orders WHERE note = 'paid; verified'"
        issues = check_select_only(sql)
        assert not any("Semicolon" in i or "semicolon" in i for i in issues), (
            "Semicolon inside string literal should not be flagged"
        )

    def test_suspicious_union_information_schema_blocked(self):
        sql = (
            "SELECT id FROM orders "
            "UNION SELECT table_name FROM information_schema.tables"
        )
        issues = check_select_only(sql)
        assert any("UNION" in i or "injection" in i.lower() for i in issues), (
            "UNION against information_schema should be blocked"
        )

    def test_empty_sql_blocked(self):
        """An empty string should always produce at least one issue."""
        for empty in ("", "   ", "\n\t"):
            issues = check_select_only(empty)
            assert issues, f"Empty SQL '{repr(empty)}' should have issues"

    def test_must_start_with_select_or_with(self):
        """A statement that does not start with SELECT or WITH is rejected
        even if it has no other forbidden keywords."""
        sql = "SHOW TABLES"
        issues = check_select_only(sql)
        assert issues, "SHOW statement should fail the SELECT/WITH check"


# ===========================================================================
# extract_table_references
# ===========================================================================


class TestExtractTableReferences:

    def test_simple_from(self):
        sql = "SELECT id FROM orders"
        refs = extract_table_references(sql)
        assert "ORDERS" in refs

    def test_join_tables(self):
        sql = (
            "SELECT o.id, c.name "
            "FROM orders o "
            "JOIN customers c ON o.customer_id = c.id "
            "LEFT JOIN products p ON o.product_id = p.id"
        )
        refs = extract_table_references(sql)
        assert "ORDERS" in refs
        assert "CUSTOMERS" in refs
        assert "PRODUCTS" in refs

    def test_qualified_name(self):
        sql = "SELECT * FROM mydb.public.orders"
        refs = extract_table_references(sql)
        # The extractor returns the whole qualified name uppercased.
        assert any("ORDERS" in r for r in refs)

    def test_union_select_both_tables(self):
        sql = (
            "SELECT id FROM customers "
            "UNION ALL SELECT id FROM archived_customers"
        )
        refs = extract_table_references(sql)
        assert "CUSTOMERS" in refs
        assert "ARCHIVED_CUSTOMERS" in refs

    def test_no_tables_in_select_expression(self):
        """A SELECT with no FROM returns an empty list."""
        sql = "SELECT 1 + 1 AS result"
        refs = extract_table_references(sql)
        assert refs == []

    def test_multiple_joins(self):
        sql = (
            "SELECT a.col1, b.col2, c.col3 "
            "FROM table_a a "
            "INNER JOIN table_b b ON a.id = b.a_id "
            "LEFT JOIN table_c c ON b.id = c.b_id"
        )
        refs = extract_table_references(sql)
        assert "TABLE_A" in refs
        assert "TABLE_B" in refs
        assert "TABLE_C" in refs


# ===========================================================================
# check_table_whitelist
# ===========================================================================


class TestCheckTableWhitelist:

    def test_table_in_whitelist_passes(self):
        sql = "SELECT id FROM orders"
        issues = check_table_whitelist(sql, ["orders", "customers"])
        assert issues == [], f"Whitelisted table should pass, got: {issues}"

    def test_table_not_in_whitelist_flagged(self):
        sql = "SELECT id FROM secret_table"
        issues = check_table_whitelist(sql, ["orders", "customers"])
        assert issues, "Non-whitelisted table should produce an issue"
        assert any("SECRET_TABLE" in i for i in issues)

    def test_empty_whitelist_disables_check(self):
        """When allowed_tables is empty the check is disabled (no issues)."""
        sql = "SELECT id FROM any_random_table"
        issues = check_table_whitelist(sql, [])
        assert issues == [], "Empty whitelist should disable the check"

    def test_case_insensitive_match(self):
        """Matching should be case-insensitive."""
        sql = "SELECT id FROM Orders"
        issues = check_table_whitelist(sql, ["ORDERS"])
        assert issues == [], "Case-insensitive match should pass"

    def test_fully_qualified_name_allowed_by_bare(self):
        """DB.SCHEMA.TABLE is allowed when the bare TABLE name is whitelisted."""
        sql = "SELECT id FROM mydb.public.orders"
        issues = check_table_whitelist(sql, ["orders"])
        assert issues == [], (
            "Fully-qualified reference should match bare whitelisted name"
        )

    def test_multiple_joins_all_in_whitelist(self):
        sql = (
            "SELECT o.id, c.name, p.title "
            "FROM orders o "
            "JOIN customers c ON o.customer_id = c.id "
            "JOIN products p ON o.product_id = p.id"
        )
        issues = check_table_whitelist(sql, ["orders", "customers", "products"])
        assert issues == [], f"All tables whitelisted — expected no issues: {issues}"

    def test_one_of_multiple_joins_not_whitelisted(self):
        sql = (
            "SELECT o.id, h.detail "
            "FROM orders o "
            "JOIN hidden_table h ON o.id = h.order_id"
        )
        issues = check_table_whitelist(sql, ["orders"])
        assert any("HIDDEN_TABLE" in i for i in issues), (
            "Non-whitelisted joined table should be flagged"
        )


# ===========================================================================
# sql_validator_node (async LangGraph node)
# ===========================================================================


@pytest.mark.asyncio
class TestSqlValidatorNode:

    async def test_valid_select_passes(self):
        state = _make_state(
            "SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY 1 LIMIT 100"
        )
        result = await sql_validator_node(state)

        assert "validation_result" in result
        assert result["validation_result"].is_valid is True
        assert result.get("error") is None
        assert result["current_node"] == "sql_validator_node"

    async def test_valid_cte_passes(self):
        state = _make_state(
            "WITH summary AS (SELECT region, SUM(sales) AS total FROM sales_fact GROUP BY 1) "
            "SELECT region, total FROM summary ORDER BY total DESC LIMIT 20"
        )
        result = await sql_validator_node(state)

        assert result["validation_result"].is_valid is True
        assert result.get("error") is None

    async def test_insert_blocked_by_node(self):
        state = _make_state("INSERT INTO orders (id) VALUES (99)")
        result = await sql_validator_node(state)

        assert result["validation_result"].is_valid is False
        assert result.get("error") is not None
        assert "INSERT" in result["error"].upper()

    async def test_update_blocked_by_node(self):
        state = _make_state("UPDATE orders SET amount = 0 WHERE id = 1")
        result = await sql_validator_node(state)

        assert result["validation_result"].is_valid is False
        assert result.get("error") is not None

    async def test_delete_blocked_by_node(self):
        state = _make_state("DELETE FROM orders WHERE id = 1")
        result = await sql_validator_node(state)

        assert result["validation_result"].is_valid is False

    async def test_drop_table_blocked_by_node(self):
        state = _make_state("DROP TABLE orders")
        result = await sql_validator_node(state)

        assert result["validation_result"].is_valid is False

    async def test_create_table_blocked_by_node(self):
        state = _make_state("CREATE TABLE shadow AS SELECT * FROM orders")
        result = await sql_validator_node(state)

        assert result["validation_result"].is_valid is False

    async def test_semicolon_injection_blocked_by_node(self):
        state = _make_state("SELECT id FROM orders; DROP TABLE orders")
        result = await sql_validator_node(state)

        assert result["validation_result"].is_valid is False
        issues = result["validation_result"].issues
        assert any("semicolon" in i.lower() for i in issues)

    async def test_empty_sql_blocked_by_node(self):
        state = _make_state("")
        result = await sql_validator_node(state)

        assert result["validation_result"].is_valid is False
        issues = result["validation_result"].issues
        assert any("empty" in i.lower() for i in issues)

    async def test_table_not_in_whitelist_flagged_by_node(self):
        state = _make_state(
            "SELECT id FROM forbidden_table",
            selected_tables=["orders", "customers"],
        )
        result = await sql_validator_node(state)

        assert result["validation_result"].is_valid is False
        issues = result["validation_result"].issues
        assert any("FORBIDDEN_TABLE" in i for i in issues)

    async def test_table_in_whitelist_passes_node(self):
        state = _make_state(
            "SELECT id, amount FROM orders LIMIT 50",
            selected_tables=["orders"],
        )
        result = await sql_validator_node(state)

        assert result["validation_result"].is_valid is True
        assert result.get("error") is None

    async def test_multiple_joins_all_whitelisted(self):
        sql = (
            "SELECT o.id, c.name, p.title "
            "FROM orders o "
            "JOIN customers c ON o.customer_id = c.id "
            "JOIN products p ON o.product_id = p.id "
            "LIMIT 200"
        )
        state = _make_state(sql, selected_tables=["orders", "customers", "products"])
        result = await sql_validator_node(state)

        assert result["validation_result"].is_valid is True

    async def test_union_select_not_blocked(self):
        """A plain UNION SELECT that does not reference system tables is valid."""
        sql = (
            "SELECT id, name FROM customers "
            "UNION ALL "
            "SELECT id, name FROM archived_customers "
            "LIMIT 100"
        )
        state = _make_state(
            sql,
            selected_tables=["customers", "archived_customers"],
        )
        result = await sql_validator_node(state)

        assert result["validation_result"].is_valid is True

    async def test_sql_with_comments_passes(self):
        """SQL that starts with SELECT and contains inline /* */ comments passes.

        The validator's warning system flags comment presence; the query must
        start with SELECT (not a -- comment line) to pass the first-token check.
        The warning about comments is still emitted.
        """
        sql = (
            "SELECT region, SUM(revenue) AS total_revenue\n"
            "FROM sales /* primary fact table */\n"
            "GROUP BY region\n"
            "ORDER BY total_revenue DESC\n"
            "LIMIT 10"
        )
        state = _make_state(sql, selected_tables=["sales"])
        result = await sql_validator_node(state)

        assert result["validation_result"].is_valid is True
        # The node should still emit a warning about the inline comment.
        warnings = result["validation_result"].warnings
        assert any("comment" in w.lower() for w in warnings)

    async def test_execution_logs_always_present(self):
        """The node must always return at least two execution log entries."""
        state = _make_state("SELECT 1")
        result = await sql_validator_node(state)

        logs = result.get("execution_logs", [])
        assert len(logs) >= 2, "Expected at least a 'started' and a result log entry"
        statuses = [log.status for log in logs]
        assert "started" in statuses

    async def test_no_selected_tables_skips_whitelist(self):
        """When selected_tables is empty the whitelist check is skipped —
        even an 'unknown' table name should pass."""
        state = _make_state(
            "SELECT id FROM any_table LIMIT 10",
            selected_tables=[],
        )
        result = await sql_validator_node(state)

        # Whitelist check is disabled; only read-only checks apply.
        assert result["validation_result"].is_valid is True
