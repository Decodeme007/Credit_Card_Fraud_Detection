"""
=============================================================================
ShieldCard AI: Credit Card Fraud Detection Platform
Interactive Streamlit Application
=============================================================================
Provides two dedicated operational modes:
1. Batch CSV Upload: Upload any CSV or test with sample transactions.
2. Manual Transaction Input: Enter Time, Amount, and all features V1 to V28.
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .fraud-card {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
    .legit-card {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
    .result-header {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .fraud-text { color: #DC2626; }
    .legit-text { color: #16A34A; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. CACHED MODEL & SCALER LOADER
# ------------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("model_rf.joblib")
    scaler = joblib.load("scaler.joblib")
    thresh_cfg = joblib.load("threshold_config.joblib")
    return model, scaler, thresh_cfg.get("optimal_threshold", 0.25)

@st.cache_data
def load_sample_csv():
    if os.path.exists("sample_test_transactions.csv"):
        return pd.read_csv("sample_test_transactions.csv")
    return None

model, scaler, optimal_threshold = load_assets()
sample_df = load_sample_csv()

# ------------------------------------------------------------------------------
# 3. PREPROCESSING FUNCTION
# ------------------------------------------------------------------------------
def preprocess_features(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """Preprocesses Time (cyclical), Amount (RobustScaler), and orders V1-V28."""
    df_proc = df.copy()

    # Time transformation
    time_series = df_proc['Time'] if 'Time' in df_proc.columns else pd.Series([0.0] * len(df_proc))
    hour = (time_series / 3600.0) % 24.0
    df_proc['hour_sin'] = np.sin(2.0 * np.pi * hour / 24.0)
    df_proc['hour_cos'] = np.cos(2.0 * np.pi * hour / 24.0)

    # Amount transformation
    if 'Amount' in df_proc.columns:
        df_proc['scaled_amount'] = scaler.transform(df_proc[['Amount']])
    else:
        df_proc['scaled_amount'] = 0.0

    # Ensure all V1-V28 columns exist
    for i in range(1, 29):
        col = f"V{i}"
        if col not in df_proc.columns:
            df_proc[col] = 0.0

    feature_order = [f"V{i}" for i in range(1, 29)] + ['hour_sin', 'hour_cos', 'scaled_amount']
    return df_proc[feature_order]

# ------------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION & SETTINGS
# ------------------------------------------------------------------------------
st.sidebar.title("💳 Fraud Detection")
st.sidebar.markdown("**Select How You Want to Predict:**")

mode = st.sidebar.radio(
    "Choose Prediction Mode:",
    [
        "📁 Option 1: Upload CSV File",
        "✍️ Option 2: Enter Values Manually (V1 to V28)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Decision Settings")
threshold = st.sidebar.slider(
    "Fraud Decision Threshold",
    min_value=0.05,
    max_value=0.95,
    value=float(optimal_threshold),
    step=0.05,
    help="Transactions with predicted fraud probability ≥ threshold are flagged."
)
st.sidebar.caption(f"💡 Default optimal threshold is **{optimal_threshold:.2f}** to balance fraud catching vs customer friction.")

# ------------------------------------------------------------------------------
# 5. OPTION 1: BATCH CSV UPLOAD
# ------------------------------------------------------------------------------
if mode == "📁 Option 1: Upload CSV File":
    st.markdown('<div class="main-title">📁 Batch CSV Fraud Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a CSV file containing transactions, or test instantly with our pre-loaded sample dataset.</div>', unsafe_allow_html=True)

    # Section to download or load sample CSV
    st.subheader("1. Need a sample file to test?")
    col_dl1, col_dl2 = st.columns([1, 2])

    use_sample = False
    with col_dl1:
        if sample_df is not None:
            csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Sample Test CSV",
                data=csv_bytes,
                file_name="sample_test_transactions.csv",
                mime="text/csv",
                help="Download a clean test CSV containing verified fraud and legit transactions."
            )
    with col_dl2:
        if sample_df is not None:
            if st.button("⚡ Quick Test: Load 50 Sample Transactions Directly"):
                use_sample = True

    st.markdown("---")
    st.subheader("2. Upload Your CSV File")
    uploaded_file = st.file_uploader("Upload CSV containing transaction columns (Time, Amount, V1-V28):", type=["csv"])

    input_df = None
    if use_sample and sample_df is not None:
        input_df = sample_df.copy()
        st.info("Loaded pre-packaged sample dataset (50 transactions with known ground truth).")
    elif uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded '{uploaded_file.name}' ({len(input_df):,} rows).")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

    # Process and predict on input dataframe
    if input_df is not None:
        with st.spinner("Analyzing transactions with Random Forest model..."):
            X_prepared = preprocess_features(input_df, scaler)
            fraud_probs = model.predict_proba(X_prepared)[:, 1]
            predictions = (fraud_probs >= threshold).astype(int)

            result_df = input_df.copy()
            result_df['Fraud_Probability_%'] = np.round(fraud_probs * 100, 2)
            result_df['Prediction'] = np.where(predictions == 1, "FRAUD ⚠️", "LEGIT ✅")

        # Summary KPIs
        total_tx = len(result_df)
        total_fraud = int(predictions.sum())
        total_legit = total_tx - total_fraud
        fraud_pct = (total_fraud / total_tx) * 100

        st.markdown("### 📊 Prediction Summary")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Transactions", f"{total_tx:,}")
        kpi2.metric("Approved (Legit)", f"{total_legit:,}")
        kpi3.metric("Flagged as Fraud", f"{total_fraud:,}")
        kpi4.metric("Flagged Fraud Rate", f"{fraud_pct:.2f}%")

        # Display ground truth accuracy if available in CSV
        actual_col = None
        for candidate in ['Actual_Class', 'Class', 'class', 'actual']:
            if candidate in result_df.columns:
                actual_col = candidate
                break

        if actual_col:
            st.markdown("### 🎯 Ground Truth Comparison")
            actuals = result_df[actual_col].astype(int)
            correct = (predictions == actuals).sum()
            acc = (correct / total_tx) * 100
            tp = int(((predictions == 1) & (actuals == 1)).sum())
            fp = int(((predictions == 1) & (actuals == 0)).sum())
            fn = int(((predictions == 0) & (actuals == 1)).sum())
            total_actual_frauds = int(actuals.sum())
            recall = (tp / total_actual_frauds * 100) if total_actual_frauds > 0 else 0

            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Model Match Rate", f"{acc:.1f}%")
            g2.metric("Actual Frauds Caught", f"{tp}/{total_actual_frauds} ({recall:.1f}%)")
            g3.metric("False Alarms (FP)", f"{fp}")
            g4.metric("Missed Frauds (FN)", f"{fn}")

        st.markdown("### 📋 Detailed Predictions Table")
        st.dataframe(result_df, use_container_width=True)

        # Download predictions
        pred_csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=pred_csv,
            file_name="fraud_predictions_results.csv",
            mime="text/csv"
        )

# ------------------------------------------------------------------------------
# 6. OPTION 2: MANUAL INPUT FORM (Time, Amount, V1 to V28)
# ------------------------------------------------------------------------------
elif mode == "✍️ Option 2: Enter Values Manually (V1 to V28)":
    st.markdown('<div class="main-title">✍️ Manual Transaction Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter transaction details and all PCA components (V1 through V28) to get a real-time fraud verdict.</div>', unsafe_allow_html=True)

    # Quick pre-fill helper
    st.subheader("💡 Quick Fill Presets (Optional)")
    fill_col1, fill_col2, fill_col3 = st.columns(3)

    if 'manual_values' not in st.session_state:
        st.session_state.manual_values = {}

    with fill_col1:
        if st.button("🟢 Pre-fill Sample Legitimate Transaction"):
            if sample_df is not None:
                legit_row = sample_df[sample_df['Actual_Class'] == 0].iloc[0]
                st.session_state.manual_values['Time'] = float(legit_row['Time'])
                st.session_state.manual_values['Amount'] = float(legit_row['Amount'])
                for i in range(1, 29):
                    st.session_state.manual_values[f"V{i}"] = float(legit_row[f"V{i}"])
                st.rerun()

    with fill_col2:
        if st.button("🔴 Pre-fill Sample Fraudulent Transaction"):
            if sample_df is not None:
                fraud_row = sample_df[sample_df['Actual_Class'] == 1].iloc[0]
                st.session_state.manual_values['Time'] = float(fraud_row['Time'])
                st.session_state.manual_values['Amount'] = float(fraud_row['Amount'])
                for i in range(1, 29):
                    st.session_state.manual_values[f"V{i}"] = float(fraud_row[f"V{i}"])
                st.rerun()

    with fill_col3:
        if st.button("🔄 Reset All to Zero"):
            st.session_state.manual_values = {}
            st.rerun()

    st.markdown("---")

    # Form to input Time and Amount
    st.subheader("1. General Transaction Details")
    c_time, c_amt = st.columns(2)
    with c_time:
        input_time = st.number_input(
            "Transaction Time (seconds elapsed from start):",
            value=float(st.session_state.manual_values.get("Time", 3600.0)),
            step=60.0,
            help="Seconds since the start of recording (e.g. 0 to 172800 for 48 hours)."
        )
    with c_amt:
        input_amount = st.number_input(
            "Transaction Amount ($ USD):",
            value=float(st.session_state.manual_values.get("Amount", 50.0)),
            min_value=0.0,
            step=1.0,
            format="%.2f"
        )

    # Form to input V1 to V28
    st.subheader("2. PCA Feature Values (V1 through V28)")
    st.caption("These 28 numerical features represent customer spending behavior patterns extracted via Principal Component Analysis.")

    v_inputs = {}

    # Display V1 to V28 in 4 organized columns (7 features per column)
    v_cols = st.columns(4)

    for i in range(1, 29):
        col_idx = (i - 1) % 4
        with v_cols[col_idx]:
            feat_name = f"V{i}"
            default_val = float(st.session_state.manual_values.get(feat_name, 0.0))
            v_inputs[feat_name] = st.number_input(
                f"Feature {feat_name}:",
                value=default_val,
                step=0.1,
                format="%.4f",
                key=f"input_{feat_name}"
            )

    st.markdown("---")

    # Predict Button
    if st.button("🚀 Analyze & Predict Transaction", type="primary", use_container_width=True):
        # Create input dataframe
        row_dict = {"Time": input_time, "Amount": input_amount}
        row_dict.update(v_inputs)
        single_df = pd.DataFrame([row_dict])

        # Preprocess and score
        processed_input = preprocess_features(single_df, scaler)
        fraud_prob = float(model.predict_proba(processed_input)[0, 1])
        is_fraud = fraud_prob >= threshold

        # Display Result
        st.markdown("### 🎯 Fraud Detection Result")

        r1, r2 = st.columns([1, 1])

        with r1:
            if is_fraud:
                st.markdown(f"""
                <div class="fraud-card">
                    <div class="result-header fraud-text">⚠️ FRAUDULENT TRANSACTION DETECTED</div>
                    <p style="font-size: 1.1rem; color: #7F1D1D;">
                        This transaction has been flagged as <b>High Risk</b> and should be declined or sent for secondary verification.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-card">
                    <div class="result-header legit-text">✅ TRANSACTION APPROVED</div>
                    <p style="font-size: 1.1rem; color: #14532D;">
                        This transaction appears <b>Legitimate</b> and can be approved safely.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            st.metric("Predicted Fraud Probability", f"{fraud_prob * 100:.2f}%")
            st.progress(fraud_prob)
            st.caption(f"Decision Threshold: `{threshold * 100:.1f}%` (Scores ≥ {threshold*100:.1f}% trigger a fraud decline)")

            if fraud_prob >= 0.70:
                st.error("Risk Level: **CRITICAL / HIGH RISK**")
            elif fraud_prob >= threshold:
                st.warning("Risk Level: **MODERATE RISK (Threshold Exceeded)**")
            else:
                st.success("Risk Level: **LOW RISK (Safe)**")

        # Highlight primary discriminators
        st.markdown("---")
        st.subheader("🔍 Primary Decision Influencers")
        key_check = ['V14', 'V4', 'V10', 'V12', 'V11', 'V17']
        key_summary = pd.DataFrame({
            "Feature": key_check,
            "Your Input": [v_inputs[c] for c in key_check],
            "Typical Legit Benchmark": [0.03, -0.01, 0.01, 0.01, -0.01, 0.01],
            "Typical Fraud Benchmark": [-5.41, +4.54, -5.67, -6.25, +3.80, -6.66]
        })
        st.dataframe(key_summary.style.format({"Your Input": "{:.4f}", "Typical Legit Benchmark": "{:.2f}", "Typical Fraud Benchmark": "{:.2f}"}), use_container_width=True)
