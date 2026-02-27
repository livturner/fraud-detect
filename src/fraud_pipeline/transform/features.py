from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd

from fraud_pipeline.config import get_settings

def build_features(con: duckdb.DuckDBPyConnection, raw_table: str = "transactions_raw", feature_table: str = "transactions_features", window_rows: int = 1000) -> None:
    """
    Build a features table from the raw transactions table.

    """
    con.execute(f"DROP TABLE IF EXISTS {feature_table}")
    con.execute(f"""
        CREATE TABLE {feature_table} AS
        WITH base AS (
        SELECT
            Time,
            Amount,
            Class,
            V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
            V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
            V21, V22, V23, V24, V25, V26, V27, V28,
            -- Simple derived feature: log amount
            LOG(Amount + 1) AS log_amount -- add 1 to avoid log(0),
            row_number() OVER (ORDER BY Time, Amount) AS txn_id -- unique transaction ID
        FROM {raw_table}
        ),
        stats AS (
            SELECT
            *,
            avg(Amount) OVER (ORDER BY txn_id ROWS BETWEEN {window_rows} PRECEDING AND 1 PRECEEDING) AS amt_mean_1k, -- rolling mean of amount over last 1000 transactions
            stddev_samp(Amount) OVER (ORDER BY txn_id ROWS BETWEEN {window_rows} PRECEDING AND 1 PRECEEDING) AS amt_std_1k -- rolling stddev of amount over last 1000 transactions
            FROM base
        )
        SELECT
            Time,
            Amount,
            Class,
            V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
            V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
            V21, V22, V23, V24, V25, V26, V27, V28,
            log_amount,
            amt_mean_1k,
            amt_std_1k,
            (Amount - amt_mean_1k) / NULLIF(amt_std_1k, 0) AS amt_zscore_1k -- z-score of amount compared to last 1000 transactions
        FROM stats
        """
    )
    rows = con.execute(f"SELECT COUNT(*) FROM {feature_table}").fetchone()[0]
    null_z = con.execute(f"SELECT COUNT(*) FROM {feature_table} WHERE amt_zscore_1k IS NULL").fetchone()[0]
    if null_z > 0:
        print(f"Warning: {null_z} rows have NULL z-score (likely due to insufficient history for rolling stats)")
    print(f"Built features table: {feature_table} with {rows} rows")

