from __future__ import annotations

from typing import Iterable

import duckdb


def _get_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    # DESCRIBE returns rows like: (column_name, column_type, null, key, default, extra)
    return [row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()]


def _assert_has_columns(
    con: duckdb.DuckDBPyConnection, table: str, required: Iterable[str]
) -> None:
    cols = _get_columns(con, table)
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"[{table}] Missing required columns: {missing}. Found: {cols}")


def validate_raw(con: duckdb.DuckDBPyConnection, table: str = "transactions_raw") -> None:
    """
    Minimal validation for the raw ingestion table.
    Raises ValueError if something looks wrong.
    """
    _assert_has_columns(con, table, required=["Time", "Amount", "Class"])

    rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if rows <= 0:
        raise ValueError(f"[{table}] Row count is {rows}, expected > 0")

    bad_class = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE Class IS NULL OR Class NOT IN (0, 1)"
    ).fetchone()[0]
    if bad_class != 0:
        raise ValueError(f"[{table}] Found {bad_class} rows with invalid Class values")

    null_time = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE Time IS NULL"
    ).fetchone()[0]
    if null_time != 0:
        raise ValueError(f"[{table}] Found {null_time} rows with NULL Time")

    bad_amount = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE Amount IS NULL OR Amount < 0"
    ).fetchone()[0]
    if bad_amount != 0:
        raise ValueError(f"[{table}] Found {bad_amount} rows with NULL/negative Amount")