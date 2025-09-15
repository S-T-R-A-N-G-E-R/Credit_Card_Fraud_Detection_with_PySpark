# Data Directory

This folder holds datasets used in the **Credit Card Fraud Detection with PySpark** project.

## Structure
- `raw/` : Original raw CSV files (large, not tracked in git).
- `processed/` : Cleaned / preprocessed datasets (generated during pipeline).
- `external/` : Any external reference data (optional).

## Notes
- Raw dataset (`train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`) comes from [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection).
- **These files are intentionally excluded from git** to avoid large binary files in version control.
- To reproduce results, download the dataset from Kaggle and place the CSV files into `data/raw/`.

