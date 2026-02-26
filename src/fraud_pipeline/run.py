from __future__ import annotations

import sys

from fraud_pipeline.ingest.ingest import ingest_csv_to_duckdb
from fraud_pipeline.config import get_settings

def run_ingest() -> None:
    settings = get_settings()
    ingest_csv_to_duckdb(csv_path=settings.csv_path, db_path=settings.db_path)      

def main() -> None:
    print("ARGV:", sys.argv)
    if len(sys.argv) < 2:
        print("Usage: python -m fraud_pipeline.run <step>")
        print("Steps: ingest")
        raise SystemExit(1)
    
    step = sys.argv[1].lower()

    if step == "ingest":
        run_ingest()
    else:
        raise SystemExit(f"Unknown step: {step}. Valid steps: ingest")
    
if __name__ == "__main__":
    main()