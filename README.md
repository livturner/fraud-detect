# fraud-detect

## project goal
1. ingest transaction data
2. engineer a few behaviour signals
3. apply detection logic
4. output flagged transactions with reason codes
5. reports basic metrics (precision/recall)

## dataset
- real transactions (anonymised)
- over ~2 days
- high class imbalance (fraud is rare)
- 2 raw columns: 
    - time (seconds since first transaction)
    - amount (transaction amount)
- Target label:
    - Class (1 = fraud, 0 = normal)

PCA - Principal Component Analysis
- takes several original features (merchant, country, device, etc)
- mixes them together into new variables (V1, V2, etc)
- preserves as much info as possible but destroys interpretability

Key Characteristics:
Fraud is ~0.17% of the data so:
- accuracy is a useless metric
- a dumb model that predicts "not fraud" will always be 99% accurate.
- instead look at:
    - precision 
    - recall
    - maybe PR-AUC

Amount Outlier (per behavioural baseline)
- transation amounts unsually large compared to typical transactions
    - define 'normal' spend for a baseline
    - flag transactions X standard deviations above baseline

Velocity Spike 
- Many transactions in a short time window
    - count transations in last N minutes 
    - flag if count exceeds a threshold

Model Risk Score
- a simple classifer that gives a high probability of fraud
    - train a basic model on your engineered features
    - flag transactions above risk threshold

