"""
tools/generate_semantic_model.py

CLI script: auto-generate semantic_model.yml by scanning a Snowflake schema.

Usage
-----
    python tools/generate_semantic_model.py \\
        --database TPCH_DATA_PRODUCT \\
        --schema   ANALYTICS \\
        --output   semantic_model.yml \\
        [--include-tables TABLE1,TABLE2]

Credentials are read from env vars:
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE (optional)

After generation, edit the YAML to add descriptions and synonyms —
those are what make the LLM routing accurate.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()          # loads .env from cwd or parent directories
except ImportError:
    pass                   # python-dotenv not installed — rely on env vars being set

try:
    import snowflake.connector
except ImportError:
    print("ERROR: snowflake-connector-python is not installed.")
    print("Run: pip install snowflake-connector-python")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Snowflake helpers
# ---------------------------------------------------------------------------

def _get_connection():
    account = os.environ.get("SNOWFLAKE_ACCOUNT", "")
    user = os.environ.get("SNOWFLAKE_USER", "")
    password = os.environ.get("SNOWFLAKE_PASSWORD", "")
    warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE", "")
    role = os.environ.get("SNOWFLAKE_ROLE", "")

    missing = [n for n, v in [("SNOWFLAKE_ACCOUNT", account), ("SNOWFLAKE_USER", user),
                               ("SNOWFLAKE_PASSWORD", password)] if not v]
    if missing:
        print(f"ERROR: Missing env vars: {missing}")
        sys.exit(1)

    # Strip .snowflakecomputing.com suffix if present
    account = re.sub(r"\.snowflakecomputing\.com$", "", account, flags=re.IGNORECASE)

    params: Dict[str, Any] = {
        "account": account, "user": user, "password": password,
    }
    if warehouse:
        params["warehouse"] = warehouse
    if role:
        params["role"] = role

    return snowflake.connector.connect(**params)


def _fetch_tables(conn, database: str, schema: str,
                  include_only: Optional[List[str]] = None) -> List[str]:
    sql = (
        f"SELECT TABLE_NAME FROM {database}.INFORMATION_SCHEMA.TABLES "
        f"WHERE TABLE_SCHEMA = '{schema}' "
        f"  AND TABLE_TYPE IN ('BASE TABLE', 'VIEW') "
        f"ORDER BY TABLE_NAME"
    )
    with conn.cursor() as cur:
        cur.execute(sql)
        names = [row[0] for row in cur.fetchall()]

    if include_only:
        upper_filter = {t.upper() for t in include_only}
        names = [n for n in names if n.upper() in upper_filter]

    return names


def _fetch_columns(conn, database: str, schema: str) -> Dict[str, List[Dict]]:
    """Return {TABLE_NAME: [{name, type, nullable, comment}]}."""
    sql = (
        f"SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COMMENT "
        f"FROM {database}.INFORMATION_SCHEMA.COLUMNS "
        f"WHERE TABLE_SCHEMA = '{schema}' "
        f"ORDER BY TABLE_NAME, ORDINAL_POSITION"
    )
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    result: Dict[str, List[Dict]] = defaultdict(list)
    for table_name, col_name, data_type, is_nullable, comment in rows:
        result[table_name].append({
            "name": col_name,
            "type": data_type or "",
            "nullable": (is_nullable or "YES").upper() == "YES",
            "comment": comment or "",
        })
    return dict(result)


# ---------------------------------------------------------------------------
# Relationship detection
# ---------------------------------------------------------------------------

def _strip_prefix(col_name: str) -> str:
    """Strip common 1-2 char table-initial prefix (e.g. O_CUSTKEY → CUSTKEY)."""
    m = re.match(r"^[A-Z]{1,2}_(.+)$", col_name.upper())
    return m.group(1) if m else col_name.upper()


def _is_key_column(col_name: str) -> bool:
    upper = col_name.upper()
    return (upper.endswith("KEY") or upper.endswith("_ID")
            or upper.endswith("_FK") or upper.endswith("_REF"))


def _detect_relationships(
    tables: List[str],
    columns_by_table: Dict[str, List[Dict]],
) -> List[Dict[str, Any]]:
    """
    Infer FK relationships by matching stripped column names across tables.

    Strategy:
    1. Build index: stripped_col_name → [(table, raw_col_name)]
    2. For each stripped name that appears in 2+ tables and looks like a key
       column, infer a relationship.
    3. Decide which side is PK vs FK:
       - If the table initials match the column prefix in one table → that's the PK side
    """
    # Build index
    index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for table in tables:
        for col in columns_by_table.get(table, []):
            stripped = _strip_prefix(col["name"])
            if _is_key_column(col["name"]) or _is_key_column(stripped):
                index[stripped].append((table, col["name"]))

    relationships: List[Dict[str, Any]] = []
    seen: set = set()

    for stripped_name, occurrences in index.items():
        if len(occurrences) < 2:
            continue

        # For each pair, decide PK vs FK
        for i in range(len(occurrences)):
            for j in range(i + 1, len(occurrences)):
                tbl_a, col_a = occurrences[i]
                tbl_b, col_b = occurrences[j]

                pair_key = tuple(sorted([(tbl_a, col_a), (tbl_b, col_b)]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                # Heuristic: if col prefix matches table initials → PK side
                prefix_a = re.match(r"^([A-Z]{1,2})_", col_a.upper())
                prefix_a_val = prefix_a.group(1) if prefix_a else ""
                tbl_a_initials = tbl_a[0].upper()

                if prefix_a_val == tbl_a_initials:
                    # col_a is likely the PK column (e.g. C_CUSTKEY in CUSTOMER)
                    pk_table, pk_col = tbl_a, col_a
                    fk_table, fk_col = tbl_b, col_b
                else:
                    pk_table, pk_col = tbl_b, col_b
                    fk_table, fk_col = tbl_a, col_a

                rel_name = f"{fk_table.lower()}_to_{pk_table.lower()}"
                relationships.append({
                    "name": rel_name,
                    "description": "Auto-detected relationship",
                    "left_table": fk_table,
                    "left_column": fk_col,
                    "right_table": pk_table,
                    "right_column": pk_col,
                    "join_type": "LEFT",
                    "cardinality": "many_to_one",
                })

    return relationships


# ---------------------------------------------------------------------------
# LLM-based description generation
# ---------------------------------------------------------------------------

def _fetch_sample_row(conn, database: str, schema: str, table: str) -> Optional[Dict]:
    """Fetch one sample row from the table. Returns None on failure."""
    try:
        with conn.cursor(snowflake.connector.DictCursor) as cur:
            cur.execute(f"SELECT * FROM {database}.{schema}.{table} LIMIT 1")
            rows = cur.fetchall()
            return dict(rows[0]) if rows else None
    except Exception:
        return None


def _generate_descriptions_llm(
    table: str,
    columns: List[Dict],
    sample_row: Optional[Dict],
    database: str,
    schema: str,
) -> Dict[str, Any]:
    """
    Call GPT-4o to generate table description, table synonyms, and per-column
    descriptions + synonyms based on column names and a sample data row.

    Returns a dict:
      {
        "table_description": "...",
        "table_synonyms": ["..."],
        "columns": {
          "COL_NAME": {"description": "...", "synonyms": [...], "is_measure": bool, "is_dimension": bool}
        }
      }
    """
    import json as _json
    try:
        from openai import OpenAI
    except ImportError:
        return {}

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {}

    client = OpenAI(api_key=api_key)

    # Format sample row for prompt
    sample_str = ""
    if sample_row:
        sample_lines = []
        for col_name, val in list(sample_row.items())[:30]:  # cap to 30 columns
            sample_lines.append(f"  {col_name}: {repr(val)}")
        sample_str = "Sample row:\n" + "\n".join(sample_lines)
    else:
        sample_str = "No sample data available."

    col_list = "\n".join(
        f"  - {c['name']} ({c['type']})"
        for c in columns[:80]  # cap to 80 columns
    )

    prompt = f"""You are a data catalog expert creating a semantic metadata layer.

Table: {table}
Database: {database}   Schema: {schema}

Columns:
{col_list}

{sample_str}

For each column, infer from its name and sample value:
1. A clear 1-sentence business description (what does this column represent?)
2. 3-5 business synonyms (terms a non-technical user would use)
3. is_measure: true if numeric and aggregatable (sum, avg makes sense), false otherwise
4. is_dimension: true if categorical / used for filtering/grouping, false otherwise

Also provide:
- A 1-2 sentence description of the whole table
- 3-5 synonyms for the table itself

Return ONLY valid JSON in this exact structure:
{{
  "table_description": "...",
  "table_synonyms": ["...", "..."],
  "columns": {{
    "COLUMN_NAME": {{
      "description": "...",
      "synonyms": ["...", "..."],
      "is_measure": false,
      "is_dimension": true
    }}
  }}
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data catalog expert. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return _json.loads(response.choices[0].message.content)
    except Exception as exc:
        print(f"    [LLM warning] {table}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# YAML builder
# ---------------------------------------------------------------------------

def _is_measure(col: Dict) -> bool:
    numeric_types = {"NUMBER", "INT", "INTEGER", "FLOAT", "DOUBLE", "DECIMAL",
                     "NUMERIC", "REAL", "BIGINT", "SMALLINT"}
    return col["type"].upper().split("(")[0] in numeric_types and not _is_key_column(col["name"])


def _build_yaml_dict(
    database: str,
    schema: str,
    tables: List[str],
    columns_by_table: Dict[str, List[Dict]],
    relationships: List[Dict],
    llm_enrichment: Optional[Dict[str, Dict]] = None,  # table → LLM output dict
) -> Dict[str, Any]:
    llm_enrichment = llm_enrichment or {}
    table_dicts = []
    for table in tables:
        cols = columns_by_table.get(table, [])
        llm = llm_enrichment.get(table, {})
        llm_cols = llm.get("columns", {})  # col_name → {description, synonyms, is_measure, is_dimension}

        col_dicts = []
        for col in cols:
            llm_col = llm_cols.get(col["name"], llm_cols.get(col["name"].upper(), {}))
            is_key = _is_key_column(col["name"])
            col_dicts.append({
                "name": col["name"],
                "description": (
                    llm_col.get("description")
                    or col["comment"]
                    or "TODO: describe this column"
                ),
                "data_type": col["type"],
                "synonyms": llm_col.get("synonyms", []),
                "is_primary_key": is_key,
                "is_measure": llm_col.get("is_measure", _is_measure(col)) if not is_key else False,
                "is_dimension": llm_col.get("is_dimension", not is_key and not _is_measure(col)) if not is_key else False,
            })
        table_dicts.append({
            "name": table,
            "description": llm.get("table_description", "TODO: describe what this table contains"),
            "database": database,
            "schema": schema,
            "synonyms": llm.get("table_synonyms", []),
            "columns": col_dicts,
        })

    return {
        "version": "1.0",
        "name": f"{database}.{schema} Semantic Model",
        "description": (
            "Auto-generated semantic model. "
            "Add descriptions and synonyms for best LLM results."
        ),
        "tables": table_dicts,
        "relationships": relationships,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-generate a semantic_model.yml from a Snowflake schema.",
    )
    parser.add_argument("--database", required=True, help="Snowflake database name")
    parser.add_argument("--schema", required=True, help="Snowflake schema name")
    parser.add_argument(
        "--output", default="",
        help=(
            "Output YAML file path. "
            "Defaults to {DATABASE}__{SCHEMA}_semantic_model.yml in the project root "
            "so existing files are never overwritten."
        ),
    )
    parser.add_argument(
        "--include-tables", default="",
        help="Comma-separated list of tables to include (default: all)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip GPT-4o description generation (faster, produces TODO placeholders)",
    )
    args = parser.parse_args()

    include_only = (
        [t.strip() for t in args.include_tables.split(",") if t.strip()]
        if args.include_tables else None
    )

    # Default output: DATABASE__SCHEMA_semantic_model.yml
    # Sanitise the database name (long names with double underscores are kept as-is)
    if args.output:
        output_path = args.output
    else:
        safe_db = re.sub(r"[^A-Za-z0-9_\-]", "_", args.database)
        safe_schema = re.sub(r"[^A-Za-z0-9_\-]", "_", args.schema)
        output_path = f"{safe_db}__{safe_schema}_semantic_model.yml"

    print(f"Connecting to Snowflake ({args.database}.{args.schema})...")
    conn = _get_connection()

    print("Fetching tables...")
    tables = _fetch_tables(conn, args.database, args.schema, include_only)
    if not tables:
        print("No tables found. Check database/schema names and role permissions.")
        sys.exit(1)
    print(f"  Found {len(tables)} table(s): {', '.join(tables)}")

    print("Fetching columns...")
    columns_by_table = _fetch_columns(conn, args.database, args.schema)
    # Filter to only the tables we're including
    columns_by_table = {t: v for t, v in columns_by_table.items() if t in tables}

    print("Detecting relationships...")
    relationships = _detect_relationships(tables, columns_by_table)
    print(f"  Detected {len(relationships)} relationship(s)")

    # LLM enrichment: fetch sample rows + call GPT-4o for descriptions/synonyms
    llm_enrichment: Dict[str, Dict] = {}
    use_llm = not args.no_llm and bool(os.environ.get("OPENAI_API_KEY", ""))
    if use_llm:
        print(f"Generating descriptions with GPT-4o ({len(tables)} table(s))...")
        for table in tables:
            print(f"  Describing {table}...", end=" ", flush=True)
            sample_row = _fetch_sample_row(conn, args.database, args.schema, table)
            enrichment = _generate_descriptions_llm(
                table,
                columns_by_table.get(table, []),
                sample_row,
                args.database,
                args.schema,
            )
            llm_enrichment[table] = enrichment
            print("done" if enrichment else "skipped (no LLM response)")
    elif not os.environ.get("OPENAI_API_KEY", ""):
        print("Skipping LLM descriptions (OPENAI_API_KEY not set). Use --no-llm to suppress this message.")
    else:
        print("Skipping LLM descriptions (--no-llm flag set).")

    conn.close()

    yaml_dict = _build_yaml_dict(
        args.database, args.schema, tables, columns_by_table, relationships,
        llm_enrichment=llm_enrichment,
    )

    # Write YAML with header comment
    header = (
        f"# Semantic Model — generated by tools/generate_semantic_model.py\n"
        f"# Database: {args.database}   Schema: {args.schema}\n"
        f"# Tables: {len(tables)}   Relationships: {len(relationships)}\n"
        f"#\n"
        f"# Next steps:\n"
        f"#   1. Replace 'TODO: describe...' with real descriptions\n"
        f"#   2. Add synonyms so the LLM recognises business terms\n"
        f"#   3. Set SEMANTIC_MODEL_PATH={output_path} in your .env\n"
        f"#   4. Verify auto-detected relationships are correct\n"
        f"#\n"
        f"# Regenerate: python tools/generate_semantic_model.py "
        f"--database {args.database} --schema {args.schema} --output {output_path}\n\n"
    )

    yaml_str = yaml.dump(yaml_dict, default_flow_style=False, allow_unicode=True, sort_keys=False)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write(yaml_str)

    print(f"\nDone! Written to: {output_path}")
    print(f"  Tables:        {len(tables)}")
    print(f"  Relationships: {len(relationships)}")
    print(f"\nNext: edit descriptions/synonyms, then set SEMANTIC_MODEL_PATH={output_path} in .env")


if __name__ == "__main__":
    main()
