from __future__ import annotations

import sys

import duckdb

from fraud_pipeline.config import get_settings
from fraud_pipeline.ingest.ingest import ingest_csv_to_duckdb
from fraud_pipeline.transform.features import build_features


def run_ingest() -> None:
    settings = get_settings()
    ingest_csv_to_duckdb(csv_path=settings.csv_path, db_path=settings.db_path)

def run_features() -> None:
    settings = get_settings()
    con = duckdb.connect(str(settings.db_path))
    try:
        build_features(con=con, raw_table="transactions_raw")      
    finally:
        con.close()

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m fraud_pipeline.run <step>")
        print("Steps: ingest, features")
        raise SystemExit(1)
    
    step = sys.argv[1].lower()

    if step == "ingest":
        run_ingest()
    elif step == "features":
        run_features()
    else:
        raise SystemExit(f"Unknown step: {step}. Valid steps: ingest, features")
    
if __name__ == "__main__":
    main()