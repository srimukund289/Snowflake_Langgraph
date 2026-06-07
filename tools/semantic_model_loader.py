"""
tools/semantic_model_loader.py

Loads, validates, and caches the semantic YAML model.
Provides formatting helpers so nodes can inject rich context into LLM prompts.

Usage
-----
Set SEMANTIC_MODEL_PATH in .env:
    SEMANTIC_MODEL_PATH=semantic_model.yml

Then in any node:
    from tools.semantic_model_loader import get_semantic_model
    model = get_semantic_model()   # None if env var not set
    if model:
        metadata = model.to_metadata_dict()
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Dict, List, Optional

import yaml

from config.semantic_model_schema import SemanticModel, TableDefinition, Relationship

logger = logging.getLogger(__name__)

# Module-level cache keyed by resolved file path
_cache: Dict[str, SemanticModel] = {}


# ---------------------------------------------------------------------------
# Core load / get helpers
# ---------------------------------------------------------------------------

def load_semantic_model(path: str) -> SemanticModel:
    """
    Load and validate a semantic model YAML file.

    Parameters
    ----------
    path : str
        Absolute or relative path to the YAML file.

    Raises
    ------
    FileNotFoundError  — file does not exist
    ValueError         — YAML is invalid or fails Pydantic validation
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Semantic model file not found: '{path}'\n"
            "Generate one with: python tools/generate_semantic_model.py "
            "--database DB --schema SCHEMA --output semantic_model.yml"
        )

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(
            f"'{path}' does not contain a valid YAML mapping. "
            "Expected a dict at the top level."
        )

    try:
        model = SemanticModel.model_validate(raw)
    except Exception as exc:
        raise ValueError(
            f"Semantic model validation failed for '{path}': {exc}"
        ) from exc

    logger.info(
        "semantic_model.loaded  name=%r  tables=%d  relationships=%d  path=%s",
        model.name, len(model.tables), len(model.relationships), path,
    )
    return model


def _discover_all_yaml_files() -> List[str]:
    """Return all *_semantic_model.yml paths in the project root (sorted)."""
    found = glob.glob("*_semantic_model.yml") + glob.glob("semantic_model.yml")
    # Deduplicate and sort
    return sorted(set(found))


def get_semantic_model(db: str = "", schema: str = "") -> Optional[SemanticModel]:
    """
    Return a merged SemanticModel covering ALL discovered YAML files.

    Resolution order:
      1. SEMANTIC_MODEL_PATH env var → load that single file.
      2. Auto-discover all *_semantic_model.yml files → merge into one model
         so the agent sees every database at once and GPT-4o picks the right tables.
      3. None → caller falls back to dynamic INFORMATION_SCHEMA discovery.

    Merging strategy: tables and relationships from all YAML files are combined.
    The merged model is cached under the key "__merged__".
    """
    explicit_path = os.environ.get("SEMANTIC_MODEL_PATH", "").strip()

    if explicit_path:
        if explicit_path in _cache:
            return _cache[explicit_path]
        model = load_semantic_model(explicit_path)
        _cache[explicit_path] = model
        return model

    # Auto-discover and merge all YAML files
    if "__merged__" in _cache:
        return _cache["__merged__"] if _cache["__merged__"] else None

    yaml_files = _discover_all_yaml_files()
    if not yaml_files:
        _cache["__merged__"] = None  # type: ignore[assignment]
        return None

    all_tables: List[TableDefinition] = []
    all_relationships: List[Relationship] = []
    names: List[str] = []

    for path in yaml_files:
        try:
            model = load_semantic_model(path)
            all_tables.extend(model.tables)
            all_relationships.extend(model.relationships)
            names.append(model.name)
            logger.info("semantic_model.discovered  path=%s  tables=%d", path, len(model.tables))
        except Exception as exc:
            logger.warning("semantic_model.skip  path=%s  error=%s", path, exc)

    if not all_tables:
        _cache["__merged__"] = None  # type: ignore[assignment]
        return None

    merged = SemanticModel(
        version="1.0",
        name=" | ".join(names),
        description=f"Merged semantic model from {len(yaml_files)} YAML file(s): {', '.join(yaml_files)}",
        tables=all_tables,
        relationships=all_relationships,
    )
    logger.info(
        "semantic_model.merged  files=%d  total_tables=%d  total_relationships=%d",
        len(yaml_files), len(all_tables), len(all_relationships),
    )
    _cache["__merged__"] = merged
    return merged


    return None


# ---------------------------------------------------------------------------
# Prompt-formatting helpers
# ---------------------------------------------------------------------------

def format_semantic_model_for_prompt(model: SemanticModel) -> str:
    """
    Format the full semantic model as a readable text block for LLM prompts.

    Includes table descriptions, synonyms, column metadata, and relationships.
    The output replaces the plain column-name-only metadata that dynamic
    discovery produces, giving the LLM business-level context.
    """
    lines: List[str] = []

    # Group tables by database / schema
    db_map: Dict[str, Dict[str, List[TableDefinition]]] = {}
    for tbl in model.tables:
        db_map.setdefault(tbl.database, {}).setdefault(tbl.schema_name, []).append(tbl)

    for db_name, schemas in db_map.items():
        lines.append(f"DATABASE: {db_name}")
        for schema_name, tables in schemas.items():
            lines.append(f"  SCHEMA: {schema_name}")
            for tbl in tables:
                fqn = tbl.fully_qualified_name
                lines.append(f"    TABLE: {tbl.name}  [{fqn}]")
                if tbl.description:
                    lines.append(f"      Description: {tbl.description}")
                if tbl.synonyms:
                    lines.append(f"      Synonyms: {', '.join(tbl.synonyms)}")

                for col in tbl.columns:
                    flags: List[str] = []
                    if col.is_primary_key:
                        flags.append("PK")
                    if col.is_measure:
                        flags.append("measure")
                    if col.is_dimension:
                        flags.append("dimension")
                    flag_str = f"  ({', '.join(flags)})" if flags else ""

                    desc_str = f"  — {col.description}" if col.description else ""
                    syn_str = (
                        f"  [synonyms: {', '.join(col.synonyms)}]"
                        if col.synonyms
                        else ""
                    )
                    lines.append(
                        f"      - {col.name:<30} {col.data_type:<15}"
                        f"{flag_str}{desc_str}{syn_str}"
                    )

    if model.relationships:
        lines.append("")
        lines.append("RELATIONSHIPS:")
        for rel in model.relationships:
            lines.append(
                f"  {rel.left_table}.{rel.left_column} → "
                f"{rel.right_table}.{rel.right_column}"
                f"  ({rel.join_type} JOIN, {rel.cardinality})"
            )
            if rel.description:
                lines.append(f"    {rel.description}")

    return "\n".join(lines)


def format_join_hints(model: SemanticModel, selected_tables: List[str]) -> str:
    """
    Return SQL JOIN hint comments for relationships between selected tables.

    Parameters
    ----------
    selected_tables : List[str]
        Fully-qualified table names chosen by the dataset selector node,
        e.g. ["TPCH_DATA_PRODUCT.ANALYTICS.ORDERS", "TPCH_DATA_PRODUCT.ANALYTICS.CUSTOMER"]

    Returns empty string when no relationships apply or model has none.
    """
    if not model.relationships:
        return ""

    # Extract bare table names (last component) for matching
    bare_selected = {fqn.split(".")[-1].upper() for fqn in selected_tables}

    applicable: List[Relationship] = [
        rel
        for rel in model.relationships
        if rel.left_table.upper() in bare_selected
        and rel.right_table.upper() in bare_selected
    ]

    if not applicable:
        return ""

    # Build a lookup from bare table name → fully-qualified name
    fqn_lookup: Dict[str, str] = {
        fqn.split(".")[-1].upper(): fqn for fqn in selected_tables
    }

    hints: List[str] = ["-- JOIN HINTS (verified relationships from semantic model):"]
    for rel in applicable:
        right_fqn = fqn_lookup.get(rel.right_table.upper(), rel.right_table)
        left_alias = rel.left_table[:2].lower()
        right_alias = rel.right_table[:2].lower()
        join_sql = rel.to_join_sql(left_alias, right_alias, right_fqn)
        hints.append(
            f"--   {join_sql}  -- {rel.cardinality}"
            + (f"  ({rel.description})" if rel.description else "")
        )

    return "\n".join(hints)
