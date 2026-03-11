# Credit Card Fraud Detection Project

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


