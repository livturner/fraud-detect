from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd

from fraud_pipeline.config import get_settings

def build_features(con: duckdb.DuckDBPyConnection, raw_table: str = "transactions_raw", feature_table: str = "transactions_features") -> None:
    """
    Build a features table from the raw transactions table.

    """
    con.execute(f"DROP TABLE IF EXISTS {feature_table}")
    con.execute(f"""
        CREATE TABLE {feature_table} AS
        SELECT
            Time,
            Amount,
            Class,
            -- Simple derived feature: hour of day
            (CAST(Time / 3600 AS BIGINT) % 24) AS hour_of_day
        FROM {raw_table}
    """
    )
    rows = con.execute(f"SELECT COUNT(*) FROM {feature_table}").fetchone()[0]
    print(f"Built features table: {feature_table} with {rows} rows")

