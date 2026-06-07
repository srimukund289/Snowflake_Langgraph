"""
sql_validator_node.py — Validates SQL safety before execution.

Performs code-based (non-LLM) validation:
  - Must start with SELECT or WITH
  - Forbids DML / DDL keywords
  - Checks for injection patterns (stray semicolons, suspicious UNION usage)
  - Verifies all referenced tables are in the selected_tables whitelist
"""

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from graph.state import AgentState, ExecutionLog, SQLValidationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_NAME = "sql_validator_node"

# DML / DDL keywords that must never appear in analytical queries.
FORBIDDEN_KEYWORDS: List[str] = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "ALTER",
    "TRUNCATE",
    "MERGE",
    "GRANT",
    "REVOKE",
    "EXEC",
    "EXECUTE",
    "CALL",
]

# Regex that matches a FROM or JOIN clause followed by a table reference.
# Handles optional schema/database qualifiers (db.schema.table) and aliases.
_TABLE_REF_PATTERN = re.compile(
    r"""
    (?:FROM|JOIN)\s+           # FROM or JOIN keyword
    (                          # capture group: full (possibly qualified) name
        (?:[`"\[]?             # optional opening quote/bracket
        [A-Za-z_][A-Za-z0-9_]*
        [`"\]]?                # optional closing quote/bracket
        \.                     # dot separator
        ){0,2}                 # up to two qualifiers (db.schema.)
        [`"\[]?
        [A-Za-z_][A-Za-z0-9_]*
        [`"\]]?
    )
    (?:\s+(?:AS\s+)?[A-Za-z_][A-Za-z0-9_]*)?  # optional alias
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Pattern for a stray semicolon that is not inside a string literal.
# Strategy: strip obvious string literals first, then check for ;
_STRING_LITERAL_PATTERN = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")

# Suspicious UNION pattern — UNION followed by SELECT that references
# system tables or uses comment tricks.
_SUSPICIOUS_UNION_PATTERN = re.compile(
    r"UNION\s+(?:ALL\s+)?SELECT\s+.*?(?:information_schema|pg_|sys\.|dual)",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Helper functions (also exported for unit-testing)
# ---------------------------------------------------------------------------


def _strip_string_literals(sql: str) -> str:
    """Replace string literal contents with placeholder to avoid false positives."""
    return _STRING_LITERAL_PATTERN.sub("''", sql)


def extract_table_references(sql: str) -> List[str]:
    """
    Return a list of table/view names referenced after FROM or JOIN keywords.

    Names are returned in UPPER CASE and stripped of any quote characters.
    Subquery aliases (bare identifiers not qualified) that appear directly
    after FROM ( ... ) are not captured because the regex requires a word
    character immediately after FROM/JOIN, so they are skipped naturally.

    SQL functions that contain the word FROM (e.g. EXTRACT(YEAR FROM col),
    TRIM(LEADING x FROM y)) are stripped before matching to avoid false positives.
    """
    stripped = _strip_string_literals(sql)
    # Remove function calls that embed FROM so their FROM is not confused with
    # a table-source FROM.  Covers EXTRACT(field FROM expr), TRIM(... FROM ...), etc.
    stripped = re.sub(r'\b(?:EXTRACT|TRIM|OVERLAY|POSITION)\s*\([^)]*\)', '', stripped, flags=re.IGNORECASE)
    matches = _TABLE_REF_PATTERN.findall(stripped)
    results: List[str] = []
    for raw in matches:
        # Remove surrounding quotes / brackets and normalise to upper case.
        clean = re.sub(r'[`"\[\]]', "", raw).strip().upper()
        if clean:
            results.append(clean)
    return results


def check_select_only(sql: str) -> List[str]:
    """
    Return a list of validation issues related to read-only enforcement.

    Checks:
      1. SQL is non-empty.
      2. First significant token is SELECT or WITH.
      3. No forbidden DML/DDL keywords appear outside string literals.
      4. No stray semicolons outside string literals.
      5. No suspicious UNION patterns targeting system tables.
    """
    issues: List[str] = []

    if not sql or not sql.strip():
        issues.append("SQL query is empty.")
        return issues

    # Strip literals to avoid false positives inside string values.
    stripped = _strip_string_literals(sql)
    normalised = stripped.upper()

    # --- 1. Must start with SELECT or WITH ---
    first_token_match = re.match(r"\s*(\w+)", normalised)
    if not first_token_match:
        issues.append("SQL does not start with a recognisable keyword.")
    else:
        first_token = first_token_match.group(1)
        if first_token not in ("SELECT", "WITH"):
            issues.append(
                f"SQL must begin with SELECT or WITH; found '{first_token}'."
            )

    # --- 2. Forbidden keywords (whole-word match to avoid partial hits) ---
    for kw in FORBIDDEN_KEYWORDS:
        # Use word boundary to avoid matching e.g. CREATED inside a name.
        pattern = rf"\b{kw}\b"
        if re.search(pattern, normalised):
            issues.append(f"Forbidden keyword detected: {kw}.")

    # --- 3. Stray semicolons (trailing ; is valid SQL terminator — ignore it) ---
    if ";" in stripped.rstrip(";").rstrip():
        issues.append(
            "Semicolon detected outside a string literal — possible SQL injection."
        )

    # --- 4. Suspicious UNION injection ---
    if _SUSPICIOUS_UNION_PATTERN.search(stripped):
        issues.append(
            "Suspicious UNION SELECT pattern detected — possible injection attempt."
        )

    return issues


def _extract_cte_names(sql: str) -> set:
    """Return CTE alias names defined in the WITH clause (e.g. 'orders_1996' in 'WITH orders_1996 AS (')."""
    return {m.group(1).upper() for m in re.finditer(r'\b(\w+)\s+AS\s*\(', sql, re.IGNORECASE)}


def check_table_whitelist(sql: str, allowed_tables: List[str]) -> List[str]:
    """
    Return a list of issues for table references not in *allowed_tables*.

    Comparison is case-insensitive and handles both bare names and
    fully-qualified names (DB.SCHEMA.TABLE).  An empty *allowed_tables* list
    disables the whitelist check (returns no issues) so that callers in
    contexts where metadata discovery has not run do not block all queries.
    """
    if not allowed_tables:
        return []

    issues: List[str] = []
    referenced = extract_table_references(sql)

    # CTE aliases look like table references but aren't real tables — exclude them.
    cte_names = _extract_cte_names(sql)

    # Normalise allowed list to upper case for comparison.
    allowed_upper = {t.upper() for t in allowed_tables}

    # Build a set of bare table names (last component) from allowed list.
    allowed_bare = {t.split(".")[-1].upper() for t in allowed_upper}

    # Build a set of table aliases used in the query (e.g. "sp" in "FROM ... sp")
    # so that column-qualified references like sp.order_date are not flagged.
    alias_pat = re.compile(
        r'\b(?:FROM|JOIN)\s+\S+\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\b',
        re.IGNORECASE,
    )
    table_aliases = {m.group(1).upper() for m in alias_pat.finditer(sql)}

    for ref in referenced:
        parts = ref.upper().split(".")
        bare = parts[-1]
        first = parts[0] if len(parts) > 1 else ""

        if bare in cte_names:
            continue  # CTE alias
        if first in table_aliases:
            continue  # alias-qualified column ref (e.g. sp.order_date)
        # Accept if the full reference OR the bare table name matches the whitelist.
        if ref not in allowed_upper and bare not in allowed_bare:
            issues.append(
                f"Table '{ref}' is not in the allowed tables list: {sorted(allowed_upper)}."
            )

    return issues


def _build_warnings(sql: str) -> List[str]:
    """
    Return advisory warnings for SQL patterns that are valid but worth noting.
    """
    warnings: List[str] = []
    upper = sql.upper()

    if "SELECT *" in upper or "SELECT\n*" in upper:
        warnings.append(
            "SELECT * detected — consider selecting specific columns for performance."
        )

    if re.search(r"\bLIMIT\b", upper) is None and re.search(r"\bTOP\b", upper) is None:
        warnings.append(
            "No LIMIT / TOP clause detected — large result sets may affect performance."
        )

    comment_count = len(re.findall(r"--", sql)) + len(re.findall(r"/\*", sql))
    if comment_count > 0:
        warnings.append(f"SQL contains {comment_count} comment(s) — review for clarity.")

    return warnings


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------


async def sql_validator_node(state: AgentState) -> dict:
    """
    LangGraph node: validate generated SQL for safety and correctness.

    Reads:   generated_sql, selected_tables
    Writes:  validation_result, current_node, execution_logs
             (also writes error if validation fails)
    """
    start_time = time.time()

    # Short-circuit: if a prior node already failed, propagate its error
    # without overwriting it with a misleading "SQL query is empty" message.
    if state.get("error"):
        prior_error = state["error"]
        skip_log = ExecutionLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            node=NODE_NAME, status="skipped",
            message=f"Skipping — upstream error: {prior_error[:120]}",
        )
        return {"current_node": NODE_NAME, "execution_logs": [skip_log]}

    log_started = ExecutionLog(
        timestamp=datetime.now(timezone.utc).isoformat(),
        node=NODE_NAME,
        status="started",
        message="SQL validation started.",
    )

    logger.info("sql_validator_node started")

    try:
        generated_sql: str = state.get("generated_sql", "") or ""
        selected_tables: List[str] = state.get("selected_tables", []) or []

        # ---------------------------------------------------------------
        # Run all checks
        # ---------------------------------------------------------------
        issues: List[str] = []

        # Read-only / injection checks
        issues.extend(check_select_only(generated_sql))

        # Table whitelist check (only when we have a non-empty whitelist)
        if selected_tables:
            issues.extend(check_table_whitelist(generated_sql, selected_tables))

        # Advisory warnings (do not affect is_valid)
        warnings = _build_warnings(generated_sql)

        # ---------------------------------------------------------------
        # Build result
        # ---------------------------------------------------------------
        is_valid = len(issues) == 0
        validation_result = SQLValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
        )

        duration_ms = (time.time() - start_time) * 1000

        if is_valid:
            logger.info(
                "SQL validation passed",
                extra={"warnings": warnings, "duration_ms": duration_ms},
            )
            log_completed = ExecutionLog(
                timestamp=datetime.now(timezone.utc).isoformat(),
                node=NODE_NAME,
                status="completed",
                message="SQL validation passed.",
                duration_ms=duration_ms,
                metadata={
                    "warnings": warnings,
                    "table_count": len(extract_table_references(generated_sql)),
                },
            )
            return {
                "validation_result": validation_result,
                "current_node": NODE_NAME,
                "execution_logs": [log_started, log_completed],
            }
        else:
            error_msg = f"SQL validation failed: {'; '.join(issues)}"
            logger.warning(
                "SQL validation failed",
                extra={"issues": issues, "duration_ms": duration_ms},
            )
            log_failed = ExecutionLog(
                timestamp=datetime.now(timezone.utc).isoformat(),
                node=NODE_NAME,
                status="error",
                message=error_msg,
                duration_ms=duration_ms,
                metadata={"issues": issues, "warnings": warnings},
            )
            return {
                "validation_result": validation_result,
                "current_node": NODE_NAME,
                "error": error_msg,
                "execution_logs": [log_started, log_failed],
            }

    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000
        error_msg = f"Unexpected error in {NODE_NAME}: {exc}"
        logger.exception("Unexpected error during SQL validation")
        log_error = ExecutionLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            node=NODE_NAME,
            status="error",
            message=error_msg,
            duration_ms=duration_ms,
        )
        return {
            "error": error_msg,
            "current_node": NODE_NAME,
            "execution_logs": [log_started, log_error],
        }
