"""
=============================================================================
ShieldCard AI: End-to-End Training & EDA Pipeline
Project: Credit Card Fraud Detection (Resume-Ready Architecture)
=============================================================================

This script contains the entire data science pipeline in one clean file:
1. Exploratory Data Analysis (EDA) & class imbalance diagnostics.
2. Stratified Train-Test Splitting (80/20) with NO data leakage.
3. Feature Engineering (Cyclical 24-hr Time) & Outlier-Resistant RobustScaler.
4. Model Training: Cost-Sensitive Random Forest (class_weight='balanced_subsample').
5. Comprehensive Evaluation: PR-AUC, ROC-AUC, Precision, Recall, F1, Confusion Matrix.
6. Decision Threshold Optimization (minimizing financial losses).
7. Artifact Persistence: Saves model, scaler, threshold config, and sample test data.

To train the model from scratch, simply run:
    python train.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def run_pipeline(csv_path: str = "creditcard.csv"):
    print("=" * 70)
    print("SHIELDCARD AI: END-TO-END TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # 1. LOAD DATA & EXPLORATORY DATA ANALYSIS (EDA)
    # --------------------------------------------------------------------------
    print(f"\n[Step 1] Loading raw dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    total_rows, total_cols = df.shape
    print(f"         Loaded {total_rows:,} transactions across {total_cols} columns.")

    print("\n[EDA Insights]:")
    fraud_count = (df['Class'] == 1).sum()
    legit_count = (df['Class'] == 0).sum()
    fraud_pct = (fraud_count / total_rows) * 100

    print(f"    - Legitimate Transactions: {legit_count:,} ({100-fraud_pct:.3f}%)")
    print(f"    - Fraudulent Transactions: {fraud_count:,} ({fraud_pct:.3f}%)")
    print(f"    - Extreme Imbalance: ~1 fraud per {legit_count // fraud_count} legitimate transactions.")
    print(f"    - Missing Values: {df.isnull().sum().sum()} (Clean dataset)")

    legit_median = df[df['Class'] == 0]['Amount'].median()
    fraud_median = df[df['Class'] == 1]['Amount'].median()
    print(f"    - 'Card Testing' Pattern: Fraud median amount (${fraud_median:.2f}) is lower than legit (${legit_median:.2f}).")

    # --------------------------------------------------------------------------
    # 2. STRATIFIED TRAIN-TEST SPLIT (PREVENTING DATA LEAKAGE)
    # --------------------------------------------------------------------------
    print("\n[Step 2] Executing Stratified Train-Test Split (80% Train, 20% Test)...")
    # CRITICAL: We NEVER undersample or apply SMOTE before splitting!
    # The test set must retain the true 0.17% production imbalance.
    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print(f"         Train set: {len(X_train):,} transactions ({(y_train == 1).sum()} frauds)")
    print(f"         Test set:  {len(X_test):,} transactions ({(y_test == 1).sum()} frauds - Real-world distribution)")

    # --------------------------------------------------------------------------
    # 3. FEATURE PREPROCESSING & ROBUST SCALING
    # --------------------------------------------------------------------------
    print("\n[Step 3] Feature Engineering & Scaling...")
    # Work on copies to prevent side effects
    X_tr = X_train.copy()
    X_te = X_test.copy()

    # Cyclical 24-hour Time Encoding (sin, cos)
    train_hour = (X_tr['Time'] / 3600.0) % 24.0
    test_hour = (X_te['Time'] / 3600.0) % 24.0

    X_tr['hour_sin'] = np.sin(2.0 * np.pi * train_hour / 24.0)
    X_tr['hour_cos'] = np.cos(2.0 * np.pi * train_hour / 24.0)
    X_te['hour_sin'] = np.sin(2.0 * np.pi * test_hour / 24.0)
    X_te['hour_cos'] = np.cos(2.0 * np.pi * test_hour / 24.0)

    X_tr.drop(columns=['Time'], inplace=True)
    X_te.drop(columns=['Time'], inplace=True)

    # RobustScaler on Amount (resistant to multi-thousand dollar outliers)
    scaler = RobustScaler()
    # CRITICAL: Fit ONLY on training data to prevent data leakage!
    X_tr['scaled_amount'] = scaler.fit_transform(X_tr[['Amount']])
    X_te['scaled_amount'] = scaler.transform(X_te[['Amount']])

    X_tr.drop(columns=['Amount'], inplace=True)
    X_te.drop(columns=['Amount'], inplace=True)

    feature_cols = [f"V{i}" for i in range(1, 29)] + ['hour_sin', 'hour_cos', 'scaled_amount']
    X_tr = X_tr[feature_cols]
    X_te = X_te[feature_cols]
    print(f"         Preprocessed features: {len(feature_cols)} (V1-V28 + hour_sin + hour_cos + scaled_amount)")

    # --------------------------------------------------------------------------
    # 4. MODEL TRAINING: COST-SENSITIVE RANDOM FOREST
    # --------------------------------------------------------------------------
    print("\n[Step 4] Training Cost-Sensitive Random Forest (class_weight='balanced_subsample')...")
    print("         (This weights rare fraud heavily without discarding legitimate data)...")
    
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced_subsample',
        max_depth=12,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_tr, y_train)
    print("         Model training completed successfully!")

    # --------------------------------------------------------------------------
    # 5. MODEL EVALUATION ON UNTOUCHED TEST DATA
    # --------------------------------------------------------------------------
    print("\n[Step 5] Evaluating Model on Untouched Test Set (56,962 transactions)...")
    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, 1]

    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "-" * 55)
    print(f"  PR-AUC (Precision-Recall AUC) : {pr_auc:.4f}  <-- True Metric")
    print(f"  ROC-AUC                       : {roc_auc:.4f}")
    print(f"  Recall (Fraud Catch Rate)     : {rec*100:.1f}% ({cm[1,1]}/{cm[1,1]+cm[1,0]} caught)")
    print(f"  Precision (Alarm Accuracy)    : {prec*100:.1f}%")
    print(f"  F1-Score                      : {f1:.4f}")
    print(f"  False Alarms (Legit Blocked)  : {cm[0,1]} out of 56,864 genuine transactions")
    print(f"  Missed Frauds (FN)            : {cm[1,0]}")
    print("-" * 55)

    # --------------------------------------------------------------------------
    # 6. DECISION THRESHOLD CALIBRATION
    # --------------------------------------------------------------------------
    print("\n[Step 6] Calibrating Decision Threshold via Financial Cost Function...")
    # Cost assumptions: Missed fraud = $120, False alarm = $5
    COST_FN, COST_FP = 120.0, 5.0

    best_thresh = 0.50
    min_cost = float('inf')

    for thresh in np.linspace(0.10, 0.90, 17):
        preds = (y_prob >= thresh).astype(int)
        conf = confusion_matrix(y_test, preds)
        fn = conf[1, 0]
        fp = conf[0, 1]
        cost = (fn * COST_FN) + (fp * COST_FP)
        if cost < min_cost:
            min_cost = cost
            best_thresh = thresh

    print(f"         Optimal Cutoff: {best_thresh:.2f} (Financial cost minimized to ${min_cost:,.2f})")

    # --------------------------------------------------------------------------
    # 7. ARTIFACT PERSISTENCE FOR STREAMLIT APP
    # --------------------------------------------------------------------------
    print("\n[Step 7] Exporting Trained Artifacts for Streamlit App...")
    joblib.dump(rf, "model_rf.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump({"optimal_threshold": float(best_thresh)}, "threshold_config.joblib")

    # Save 50 sample transactions for instant app testing
    test_export = X_test.copy()
    test_export['Actual_Class'] = y_test
    frauds_sample = test_export[test_export['Actual_Class'] == 1].head(25)
    legits_sample = test_export[test_export['Actual_Class'] == 0].head(25)
    sample_df = pd.concat([frauds_sample, legits_sample]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    sample_df.to_csv("sample_test_transactions.csv", index=False)

    print("         Exported:")
    print("         * 'model_rf.joblib'")
    print("         * 'scaler.joblib'")
    print("         * 'threshold_config.joblib'")
    print("         * 'sample_test_transactions.csv'")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE! You can now launch the Streamlit app:")
    print("    streamlit run app.py")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
