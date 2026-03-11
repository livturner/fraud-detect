# Credit Card Fraud Detection Project

## Overview 

This project builds a machine learning pipeline to detect fraudulent credit card transactions.

The pipeline ingests raw transaction data, generates behavioural features using rolling statistics, and trains classification models to predict fraud. Model performance is evaluated using ROC-AUC and precision-recall metrics, with threshold tuning used to balance fraud detection and investigation workload.

---
## Dataset

The dataset contains anonymised credit card transactions with PCA-transformed features (V1-V28), transaction amount, and a binary fraud label.

Due to the highly imbalanced nature of fraud detection problems, evaluation focuses on precision-recall metrics rather than accuracy.

The raw dataset is not included due to licensing and file constraints.

Dataset source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading the dataset, place the file in:

        data/raw/creditcard.csv

---
## Pipeline 

The project implements a modular fraud detection pipeline.

### 1. Data ingestion
Raw transaction data is loaded from CSV and stored in DuckDB for efficient querying.

### 2. Feature engineering 
Additional behavioural features are generated using rolling statistics on transaction amounts, including:
- rolling mean
- rolling standard deviation
- transaction z-score 

### 3. Model training 
Two models are trained and compared:

**Logistic Regression**
- baseline linear classifier 
- class weighting used to address this class imbalance

**Random Forest**
- non-linear ensemble model
- captures more complex relationships between features

### 4. Model evaluation
Models are evaluated using:

- ROC-AUC
- Precision-Recall AUC

Precision-recall metrics are particularly important for fraud detection because the dataset is highly imbalanced.

### 5. Threshold tuning
Classification thresholds are evaluated to explore the trade-off between:

- catching fraudulent transactions
- limiting the number of flagged legitimate transactions 

Threshold experiments are saved as artifacts for analysis.

---
## Results

| Model | ROC-AUC | PR-AUC |
|------|------|------|
| Logistic Regression | ~0.986 | ~0.762 |
| Random Forest | ~0.964 | ~0.820 |

The Random Forest model improved precision-recall performance copared to the linear baseline, suggesting that non-linear interactions between features contribute to fraud detection.

---
## Operational Trade-offs

Fraud detection systems must balance two competing objectives:

**Recall** - catching as many fraudulent transactions as possible

**Precision** - avoiding unnecessary investigation of legitimate transactions

Lower classification thresholds increase recall but flag more transactions for investigation. Higher thresholds improve precision but risk missing fraudulent activity.

In practice, fraud detection systems often prefer **higher recall**, since the cost of missed fraud can exceed the cost of investigating additional transactions.

Threshold tuning experiments allow this balance to be adjusted depending on operational requirements.

---
## Evaluation

### ROC Curve

![ROC Curve](report/figures/roc_curve.png)

### Precision–Recall Curve

![PR Curve](report/figures/pr_curve.png)

### Threshold Trade-off

![Threshold Tradeoff](report/figures/rf_threshold_tradeoff.png)

---
## Running the Pipeline

1. Download the dataset from Kaggle.

2. Place the dataset in:

        dataset/raw/creditcard.csv

3. Run the pipeline stages:

        python -m fraud_pipeline.run ingest

        python -m fraud_pipeline.run features

        python -m fraud_pipeline.run model


    Artifacts and evaluation plots will be saved in:

        artifacts/

        reports/figures/

---
## Project structure

```
.
fraud-detect
├── artifacts   # saved model metrics and experiment outputs
├── data
│   ├── raw        # original dataset
│   └── interim    # DuckDB feature store 
├── report
│   └── figures # evaluation plots
└── src
    └── fraud_pipeline  # main ML pipeline
        ├── ingest
        ├── model
            └── train.py   
        ├── run.py  # CLI entry point
        ├── transform
            └── features.py  
        └── validate.py
```



