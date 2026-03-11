# Credit Card Fraud Detection Project

## Overview 

This project builds a machine learning pipeline to detect fraudulent credit card transactions.

The pipeline ingests raw transaction data, generates behavioural features using rolling statistics, and trains classification models to predict fraud. Model performance is evaluated using ROC-AUC and precision-recall metrics, with threshold tuning used to balance fraud detection and investigation workload.

## Dataset

The dataset contains anonymised credit card transactions with PCA-transformed features (V1-V28), transaction amount, and a binary fraud label.

Due to the highly imbalanced nature of fraud detection problems, evaluation focuses on precision-recall metrics rather than accuracy.

The raw dataset is not included due to licensing and file constraints.

Dataset source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading the dataset, place the file in:
data/raw/creditcard.csv

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


