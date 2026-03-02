from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score

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

    model = LogisticRegression(class_weight="balanced", max_iter=5000) # use balanced class weights due to class imbalance

    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of positive class (fraud)
    
    print ("\nThreshold tuning for baseline model:")

    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0) # set zero_division=0 to avoid warnings when there are no positive predictions at high thresholds
        recall = recall_score(y_test, y_pred)

        flagged = y_pred.sum()
        total = len(y_pred) 
        flag_rate = flagged / total # percentage of transactions flagged as fraud at this threshold
        
        print(f"Threshold: {threshold:.2f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Flagged: {flagged}/{total} {flag_rate:.4f}")

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC: {auc:.4f}")

def train_enhanced_model(db_path: Path, feature_table: str = "transactions_features") -> None:
    # Similar to train_baseline but includes engineered features like log_amount, rolling mean/stddev, and z-score
    # This is a placeholder for future implementation
    con = duckdb.connect(str(db_path))
    try:
        df = con.execute(f"SELECT * FROM {feature_table}").fetchdf()
    finally:
        con.close()
    
    df = df[df['amt_zscore_1k'].notnull()].copy() # drop rows where z-score is null

    df = df.sort_values(by=["Time", "Amount"]).reset_index(drop=True) # split by time/amount to avoid lookahead bias

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "log_amount", "amt_mean_1k", "amt_std_1k", "amt_zscore_1k"]

    X_train = train_df[feature_cols]  # features including engineered ones
    y_train = train_df["Class"]  # labels

    X_test = test_df[feature_cols]  # features including engineered ones
    y_test = test_df["Class"]  # labels

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of positive class (fraud)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Enhanced Model AUC: {auc:.4f}")

    y_pred = (y_pred_proba >= 0.5).astype(int)
    print("\nClassification Report for Enhanced Model (threshold=0.5):")
    print(classification_report(y_test, y_pred, zero_division=0))
