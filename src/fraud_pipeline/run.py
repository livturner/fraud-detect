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

def run_model() -> None:
    from fraud_pipeline.model.train import train_and_evaluate
    settings = get_settings()
    train_and_evaluate(
        db_path=settings.db_path)
    
def run_predict() -> None:
    from fraud_pipeline.model.predict import predict
    settings = get_settings()
    predict(
        csv_path=settings.sample_csv_path,
        model_path=settings.model_path,
        threshold=0.1,
        output_path=settings.predictions_output_path
    )

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m fraud_pipeline.run <step>")
        print("Steps: ingest, features, model, predict")
        raise SystemExit(1)
    
    step = sys.argv[1].lower()

    if step == "ingest":
        run_ingest()
    elif step == "features":
        run_features()
    elif step == "model":
        run_model()
    elif step == "predict":
        run_predict()
    else:
        raise SystemExit(f"Unknown step: {step}. Valid steps: ingest, features, model, predict")
    
if __name__ == "__main__":
    main()