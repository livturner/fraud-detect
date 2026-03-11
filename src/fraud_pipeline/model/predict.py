from __future__ import annotations

from pathlib import Path

import duckdb
import joblib
import pandas as pd

from fraud_pipeline.ingest.ingest import ingest_csv_to_duckdb
from fraud_pipeline.transform.features import build_features

def predict(
    csv_path: Path,
    model_path: Path,
    threshold: float = 0.1,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Predict fraud on a new dataset.

    Args:
        csv_path: Path to the input CSV file.
        model_path: Path to the trained model file.
        threshold: Threshold for classifying a transaction as fraud.
        output_path: Optional path to save the predictions as a CSV file.

    Returns:
        A DataFrame containing the predictions.
    """
    model = joblib.load(model_path)

    con = duckdb.connect()
    try:
        ingest_csv_to_duckdb(csv_path=csv_path, con=con, is_predict=True)
        
        build_features(con=con, is_predict=True)
    
        df = con.execute("SELECT * FROM transactions_features").fetchdf()

        feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        X = df[feature_cols]
        df["fraud_probability"] = model.predict_proba(X)[:, 1]
        df["flagged"] = (df["fraud_probability"] >= threshold).astype(int)
        
        if output_path is not None:
                df.to_csv(output_path, index=False)
                print(f"Saved predictions to {output_path}")
        return df
    finally:
        con.close()





