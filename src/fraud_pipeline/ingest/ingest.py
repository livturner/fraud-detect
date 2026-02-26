from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from fraud_pipeline.config import get_settings


def ingest_csv_to_duckdb(
    csv_path: Path,
    db_path: Path,
    table_name: str = "transactions_raw",
) -> None:
    """
    Load the credit card fraud CSV into DuckDB.

    Creates/overwrites `table_name` in `db_path`.
    Prints a few sanity checks after writing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Read CSV
    df = pd.read_csv(csv_path)

    # Basic sanity checks on expected columns
    expected_cols = {"Time", "Amount", "Class"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found: {list(df.columns)[:10]}...")

    # Write to DuckDB
    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.register("df_view", df)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_view")

        # Sanity queries
        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        fraud_count = con.execute(f"SELECT COUNT(*) FROM {table_name} WHERE Class = 1").fetchone()[0]
        time_min, time_max = con.execute(
            f"SELECT MIN(Time), MAX(Time) FROM {table_name}"
        ).fetchone()
        amt_min, amt_max = con.execute(
            f"SELECT MIN(Amount), MAX(Amount) FROM {table_name}"
        ).fetchone()

        print("✅ Ingestion complete")
        print(f"DB: {db_path}")
        print(f"Table: {table_name}")
        print(f"Rows: {row_count}")
        print(f"Fraud rows (Class=1): {fraud_count}")
        print(f"Time range: {time_min} → {time_max}")
        print(f"Amount range: {amt_min} → {amt_max}")

    finally:
        con.close()


def main() -> None:
    settings = get_settings()
    ingest_csv_to_duckdb(csv_path=settings.csv_path, db_path=settings.db_path)


if __name__ == "__main__":
    main()