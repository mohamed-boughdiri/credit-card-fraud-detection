# Credit Card Fraud Detection

Binary classification on the [Kaggle mlg-ulb credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraudulent (0.17%).

## Approach
- Train/test split performed before any preprocessing to prevent data leakage
- Scaling and modeling combined in a single `sklearn.Pipeline` per model, so the fitted transform is guaranteed identical at train and inference time
- Class imbalance handled via `class_weight='balanced'` / `scale_pos_weight` rather than naive resampling
- Model comparison (Logistic Regression, Random Forest, XGBoost) via stratified 5-fold cross-validation on the training set only
- Final evaluation on an untouched, realistically-imbalanced test set
- Decision threshold selected using the precision-recall curve rather than a default 0.5 cutoff

## Results
Random Forest / XGBoost achieve F1 ≈ 0.85 and average precision ≈ 0.85 on held-out test data, consistent with published benchmarks on this dataset.

## Stack
Python, pandas, scikit-learn, XGBoost, matplotlib/seaborn
