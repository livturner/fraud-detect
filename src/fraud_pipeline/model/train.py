from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from fraud_pipeline.config import get_settings

def train_baseline(db_path: Path, feature_table: str = "transactions_features") -> None:
    # Load features and labels into a DataFrame
    con = duckdb.connect(str(db_path))
    try:
        df = con.execute(f"SELECT * FROM {feature_table}").fetchdf()
    finally:
        con.close()

    df = df[df['amt_zscore_1k'].notnull()].copy() # drop rows where z-score is null (due to insufficient history for rolling stats)

    df = df.sort_values(by=["Time", "Amount"]).reset_index(drop=True) # split by time/amount to avoid lookahead bias

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    baseline_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"] 

    X_train = train_df[baseline_cols]  # features only
    y_train = train_df["Class"]  # labels

    X_test = test_df[baseline_cols]  # features only
    y_test = test_df["Class"]  # labels

    model = LogisticRegression(class_weight="balanced", max_iter=1000) # use balanced class weights due to class imbalance

    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of positive class (fraud)
    y_pred = (y_pred_proba >= 0.5).astype(int)  # threshold at 0.5
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC: {auc:.4f}")

if __name__ == "__main__":
    settings = get_settings()
    train_baseline(db_path=settings.db_path)
    
