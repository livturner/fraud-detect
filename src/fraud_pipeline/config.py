from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

@dataclass(frozen=True)
class Settings:
    csv_path: Path
    db_path: Path
    model_path: Path
    predictions_output_path: Path
    sample_csv_path: Path

def get_settings() -> Settings:
    """
    Loads environment variables (optionally from local .env file) 
    and returns resolved absolute paths.
    
    Required:
    - FRAUD_CSV_PATH: Path to the raw credit card CSV file.
    - FRAUD_DB_PATH: Path where the DuckDB database should be created.
    """
    load_dotenv()  # Load from .env if it exists

    csv_raw = os.getenv("FRAUD_CSV_PATH")
    db_raw = os.getenv("FRAUD_DB_PATH")
    

    if not csv_raw:
        raise RuntimeError("Missing env var: FRAUD_CSV_PATH")
    if not db_raw:
        raise RuntimeError("Missing env var: FRAUD_DB_PATH")
    
    csv_path = Path(csv_raw).expanduser().resolve()
    db_path = Path(db_raw).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[2]

    return Settings(csv_path=csv_path, 
                    db_path=db_path,
                    model_path=repo_root / "artifacts" / "models" / "random_forest.joblib",
                    predictions_output_path=repo_root / "artifacts" / "predictions.csv",
                    sample_csv_path=repo_root / "data" / "raw" / "sample_transactions.csv",)