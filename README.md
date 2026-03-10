# Credit Card Fraud Detection Project

## Project structure

```
├── README.md
├── artifacts
│   ├── model_metrics.csv
│   ├── model_metrics.json
│   └── threshold_results.csv
├── data
│   ├── interim
│   │   └── fraud.duckdb
│   └── raw
│       └── creditcard.csv
├── pyproject.toml
├── report
│   └── figures
│       ├── pr_curve.png
│       ├── rf_threshold_tradeoff.png
│       └── roc_curve.png
├── requirements.txt
└── src
    └── fraud_pipeline
        ├── __init__.py
        ├── config.py
        ├── ingest
        ├── model
        ├── run.py
        ├── transform
        └── validate.py
```


