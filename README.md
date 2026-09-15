# 💳 Credit Card Fraud Detection Platform

An interactive machine learning application built with **Python, Scikit-Learn, and Streamlit** to detect credit card fraud in real time on imbalanced transaction data.

---

## 🌟 Key Features

The application provides two clear options:

### 📁 Option 1: Batch CSV Upload
* Upload any transaction `.csv` file.
* Includes a **built-in downloadable sample test dataset** (`sample_test_transactions.csv`) and a **one-click "Quick Test" button** to test 50 verified transactions instantly.
* Predicts fraud probability for every transaction, calculates summary KPIs, and allows downloading an annotated results CSV.
* If ground-truth labels are present, it compares predictions against actual labels (Accuracy, Frauds Caught, False Alarms).

### ✍️ Option 2: Manual Value Input (V1 through V28)
* Enter transaction details manually:
  * **Amount ($ USD)**
  * **Time** (seconds elapsed)
  * **All 28 PCA features: `V1` through `V28`** in an organized 4-column input layout.
* Includes **Quick-Fill presets**:
  * 🟢 *Pre-fill Sample Legitimate Transaction*
  * 🔴 *Pre-fill Sample Fraudulent Transaction*
  * 🔄 *Reset All to Zero*
* Real-time prediction card with probability score, risk level, and comparison against typical fraud benchmarks.

---

## 📊 Model & Performance Highlights
* **Architecture:** Cost-Sensitive Random Forest (`class_weight='balanced_subsample'`).
* **PR-AUC (Average Precision):** **0.8345** (on 56,962 untouched test transactions).
* **False Alarms:** Slashed down to just **15** across all ~56,800 legitimate test transactions (Precision = **84.2%**).
* **Calibrated Decision Threshold:** Set to **0.25** to minimize financial chargeback loss while maintaining high fraud recall (**87.8%**).

---

## 🚀 How to Run the App

1. Make sure required libraries are installed:
```bash
pip install streamlit scikit-learn pandas numpy joblib
```

2. Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

---

## 📂 Project Structure
```
├── app.py                         # Streamlit application with the 2 prediction options
├── model_rf.joblib                # Trained champion Random Forest model
├── scaler.joblib                  # Fitted RobustScaler for Amount
├── threshold_config.joblib        # Optimal threshold configuration (0.25)
├── sample_test_transactions.csv   # 50 verified test transactions for instant demo
├── creditcard.csv                 # Raw dataset (284,807 transactions)
├── Credit_card_Project.ipynb      # Original exploratory notebook
└── README.md                      # Project documentation
```
