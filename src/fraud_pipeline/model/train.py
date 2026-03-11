from __future__ import annotations

from pathlib import Path
import json


import joblib

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from fraud_pipeline.config import get_settings


def load_feature_data(
    db_path: Path,
    feature_table: str = "transactions_features",
    zscore_col: str = "amt_zscore_1k",
) -> pd.DataFrame:
    """
    Load model-ready feature data from DuckDB, drop warmup rows, and sort chronologically.
    """
    con = duckdb.connect(str(db_path))
    try:
        df = con.execute(f"SELECT * FROM {feature_table}").fetchdf()
    finally:
        con.close()

    if zscore_col in df.columns:
        df = df[df[zscore_col].notnull()].copy()

    df = df.sort_values(["Time", "Amount"]).reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test using chronological order.
    """
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def build_model_dict() -> dict:
    """
    Return the models to compare.
    """
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_thresholds(
    y_test: pd.Series,
    y_pred_proba,
    thresholds: list[float],
    model_name: str,
) -> list[dict]:
    """
    Evaluate precision/recall/flag rate across thresholds.
    """
    rows = []
    total = len(y_test)

    print(f"\nPrecision/Recall at different thresholds for {model_name}:")

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        flagged = int(y_pred.sum())
        flag_rate = flagged / total

        print(
            f"Threshold: {t:.2f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"Flagged: {flagged}/{total} ({flag_rate:.4%})"
        )

        rows.append(
            {
                "model": model_name,
                "threshold": t,
                "precision": precision,
                "recall": recall,
                "flagged": flagged,
                "total": total,
                "flag_rate": flag_rate,
            }
        )

    return rows

def save_models(
    trained_models: dict,
    artifacts_dir: Path,
) -> None:
    """
    Persist trained models to disk using joblib.
    """
    models_dir = artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, model in trained_models.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, models_dir / filename)
        print(f"Saved model: {models_dir / filename}")

def save_artifacts(
    metrics_rows: list[dict],
    threshold_rows: list[dict],
    artifacts_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save metrics and threshold results to disk.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics_rows)
    threshold_df = pd.DataFrame(threshold_rows)

    metrics_df.to_csv(artifacts_dir / "model_metrics.csv", index=False)
    threshold_df.to_csv(artifacts_dir / "threshold_results.csv", index=False)

    with open(artifacts_dir / "model_metrics.json", "w") as f:
        json.dump(metrics_rows, f, indent=2)

    print(f"\nSaved metrics to: {artifacts_dir / 'model_metrics.csv'}")
    print(f"Saved thresholds to: {artifacts_dir / 'threshold_results.csv'}")

    return metrics_df, threshold_df


def save_plots(curve_data: dict, threshold_df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Save ROC, PR, and Random Forest threshold tradeoff plots.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    plt.figure(figsize=(8, 6))
    for name, data in curve_data.items():
        plt.plot(data["fpr"], data["tpr"], label=name)

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curve.png", dpi=200)
    plt.close()

    # Precision-Recall curve
    plt.figure(figsize=(8, 6))
    for name, data in curve_data.items():
        plt.plot(data["pr_recall"], data["pr_precision"], label=name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "pr_curve.png", dpi=200)
    plt.close()

    # Threshold tradeoff for Random Forest
    rf_df = threshold_df[threshold_df["model"] == "Random Forest"].copy()

    if not rf_df.empty:
        plt.figure(figsize=(8, 6))
        plt.plot(rf_df["threshold"], rf_df["precision"], marker="o", label="Precision")
        plt.plot(rf_df["threshold"], rf_df["recall"], marker="o", label="Recall")
        plt.plot(rf_df["threshold"], rf_df["flag_rate"], marker="o", label="Flag rate")
        plt.xlabel("Threshold")
        plt.ylabel("Value")
        plt.title("Random Forest Threshold Trade-off")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "rf_threshold_tradeoff.png", dpi=200)
        plt.close()

    print(f"Saved plots to: {figures_dir}")


def train_and_evaluate(
    db_path: Path,
    feature_table: str = "transactions_features",
    zscore_col: str = "amt_zscore_1k",
) -> None:
    """
    Train and compare baseline models on the feature table.
    """
    # Paths for outputs
    repo_root = Path(__file__).resolve().parents[3]
    artifacts_dir = repo_root / "artifacts"
    figures_dir = repo_root / "report" / "figures"

    # Load and split data
    df = load_feature_data(db_path=db_path, feature_table=feature_table, zscore_col=zscore_col)
    train_df, test_df = time_split(df, train_frac=0.8)

    print(f"Total usable rows: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Fraud cases in test set: {int(test_df['Class'].sum())}")

    # Feature selection
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    X_train = train_df[feature_cols]
    y_train = train_df["Class"]

    X_test = test_df[feature_cols]
    y_test = test_df["Class"]

    # Models
    models = build_model_dict()
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]

    metrics_rows = []
    threshold_rows = []
    trained_models = {}
    curve_data = {}

    for name, model in models.items():
        print("\n" + "=" * 80)
        print(f"MODEL: {name}")
        print("=" * 80)

        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)

        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC : {pr_auc:.4f}")

        metrics_rows.append(
            {
                "model": name,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

        # Save curve data
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_pred_proba)

        curve_data[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "pr_precision": pr_precision,
            "pr_recall": pr_recall,
        }

        # Default threshold snapshot
        y_pred_05 = (y_pred_proba >= 0.5).astype(int)
        print(f"\nClassification Report for {name} (threshold=0.5):")
        print(classification_report(y_test, y_pred_05, zero_division=0, digits=4))

        # Threshold tuning
        threshold_rows.extend(
            evaluate_thresholds(
                y_test=y_test,
                y_pred_proba=y_pred_proba,
                thresholds=thresholds,
                model_name=name,
            )
        )

    # Save metrics + plots
    _, threshold_df = save_artifacts(
        metrics_rows=metrics_rows,
        threshold_rows=threshold_rows,
        artifacts_dir=artifacts_dir,
    )

    save_plots(
        curve_data=curve_data,
        threshold_df=threshold_df,
        figures_dir=figures_dir,
    )

    save_models(
        trained_models=trained_models,
        artifacts_dir=artifacts_dir,
    )
