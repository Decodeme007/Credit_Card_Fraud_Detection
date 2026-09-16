"""
=============================================================================
Credit Card Fraud Detection Platform
Enterprise Risk Management Application
=============================================================================
Provides two primary operational workflows:
1. Batch Assessment: Upload a transaction dataset or test with historical samples.
2. Single Transaction Assessment: Input transaction attributes (Time, Amount, V1-V28).
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
    page_title="Credit Card Fraud Assessment Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .fraud-card {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-left: 5px solid #DC2626;
        border-radius: 6px;
        padding: 18px 22px;
        margin-top: 10px;
    }
    .legit-card {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        border-left: 5px solid #16A34A;
        border-radius: 6px;
        padding: 18px 22px;
        margin-top: 10px;
    }
    .result-status {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 6px;
    }
    .fraud-status { color: #991B1B; }
    .legit-status { color: #166534; }
    .result-description {
        font-size: 0.95rem;
        color: #334155;
        margin: 0;
        line-height: 1.4;
    }
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
    """Transforms Time into cyclical components, applies RobustScaler to Amount, and aligns V1-V28."""
    df_proc = df.copy()

    # Time transformation (cyclical 24-hour representation)
    time_series = df_proc['Time'] if 'Time' in df_proc.columns else pd.Series([0.0] * len(df_proc))
    hour = (time_series / 3600.0) % 24.0
    df_proc['hour_sin'] = np.sin(2.0 * np.pi * hour / 24.0)
    df_proc['hour_cos'] = np.cos(2.0 * np.pi * hour / 24.0)

    # Amount transformation (outlier-resistant scaling)
    if 'Amount' in df_proc.columns:
        df_proc['scaled_amount'] = scaler.transform(df_proc[['Amount']])
    else:
        df_proc['scaled_amount'] = 0.0

    # Ensure all PCA features V1 through V28 exist
    for i in range(1, 29):
        col = f"V{i}"
        if col not in df_proc.columns:
            df_proc[col] = 0.0

    feature_order = [f"V{i}" for i in range(1, 29)] + ['hour_sin', 'hour_cos', 'scaled_amount']
    return df_proc[feature_order]

# ------------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS & SETTINGS
# ------------------------------------------------------------------------------
st.sidebar.title("Risk Management Console")
st.sidebar.markdown("**Operational Mode**")

mode = st.sidebar.radio(
    "Select Workflow:",
    [
        "Batch Assessment (CSV Upload)",
        "Single Transaction Assessment"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Decision Configuration")
threshold = st.sidebar.slider(
    "Action Threshold (Fraud Probability)",
    min_value=0.05,
    max_value=0.95,
    value=float(optimal_threshold),
    step=0.05,
    help="Transactions with an estimated fraud probability equal to or above this cutoff are flagged for review or declined."
)
st.sidebar.caption(f"Default calibrated threshold is **{optimal_threshold:.2f}**, chosen to optimize the financial tradeoff between chargeback prevention and cardholder friction.")

# ------------------------------------------------------------------------------
# 5. MODE 1: BATCH CSV UPLOAD
# ------------------------------------------------------------------------------
if mode == "Batch Assessment (CSV Upload)":
    st.markdown('<div class="main-title">Batch Transaction Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluate batches of incoming transaction records. Upload a CSV file or use the historical test dataset for validation.</div>', unsafe_allow_html=True)

    # Sample File Access
    st.subheader("1. Sample Dataset")
    col_dl1, col_dl2 = st.columns([1, 2])

    use_sample = False
    with col_dl1:
        if sample_df is not None:
            csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Sample CSV Template",
                data=csv_bytes,
                file_name="sample_test_transactions.csv",
                mime="text/csv",
                help="Download a clean benchmark sample containing 50 historical transactions."
            )
    with col_dl2:
        if sample_df is not None:
            if st.button("Load Pre-packaged Sample Records"):
                use_sample = True

    st.markdown("---")
    st.subheader("2. Upload Dataset")
    uploaded_file = st.file_uploader(
        "Select CSV file containing transaction records (Time, Amount, V1-V28):",
        type=["csv"]
    )

    input_df = None
    if use_sample and sample_df is not None:
        input_df = sample_df.copy()
        st.info("Loaded pre-packaged test dataset containing 50 verified transaction records.")
    elif uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"Imported '{uploaded_file.name}' ({len(input_df):,} total rows).")
        except Exception as e:
            st.error(f"Unable to read CSV file: {e}")

    # Inference & Metrics Display
    if input_df is not None:
        with st.spinner("Processing transactions through risk scoring model..."):
            X_prepared = preprocess_features(input_df, scaler)
            fraud_probs = model.predict_proba(X_prepared)[:, 1]
            predictions = (fraud_probs >= threshold).astype(int)

            result_df = input_df.copy()
            result_df['Fraud_Probability_%'] = np.round(fraud_probs * 100, 2)
            result_df['Assessment'] = np.where(predictions == 1, "Flagged (High Risk)", "Cleared (Legitimate)")

        # High-level KPIs
        total_tx = len(result_df)
        total_fraud = int(predictions.sum())
        total_legit = total_tx - total_fraud
        fraud_pct = (total_fraud / total_tx) * 100

        st.markdown("### Portfolio Assessment Summary")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Records Evaluated", f"{total_tx:,}")
        kpi2.metric("Cleared for Settlement", f"{total_legit:,}")
        kpi3.metric("Flagged for Review / Blocked", f"{total_fraud:,}")
        kpi4.metric("Flagged Ratio", f"{fraud_pct:.2f}%")

        # Ground truth comparison if historical labels are included
        actual_col = None
        for candidate in ['Actual_Class', 'Class', 'class', 'actual']:
            if candidate in result_df.columns:
                actual_col = candidate
                break

        if actual_col:
            st.markdown("### Benchmark Validation Against Ground Truth")
            actuals = result_df[actual_col].astype(int)
            correct = (predictions == actuals).sum()
            acc = (correct / total_tx) * 100
            tp = int(((predictions == 1) & (actuals == 1)).sum())
            fp = int(((predictions == 1) & (actuals == 0)).sum())
            fn = int(((predictions == 0) & (actuals == 1)).sum())
            total_actual_frauds = int(actuals.sum())
            recall = (tp / total_actual_frauds * 100) if total_actual_frauds > 0 else 0

            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Classification Accuracy", f"{acc:.1f}%")
            g2.metric("Known Frauds Intercepted", f"{tp}/{total_actual_frauds} ({recall:.1f}%)")
            g3.metric("False Alerts (Innocent Users)", f"{fp}")
            g4.metric("Unintercepted Frauds (FN)", f"{fn}")

        st.markdown("### Evaluated Transaction Records")
        st.dataframe(result_df, width="stretch")

        # Export predictions
        pred_csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Scored Results (CSV)",
            data=pred_csv,
            file_name="fraud_assessment_results.csv",
            mime="text/csv"
        )

# ------------------------------------------------------------------------------
# 6. MODE 2: SINGLE MANUAL TRANSACTION SCORING
# ------------------------------------------------------------------------------
elif mode == "Single Transaction Assessment":
    st.markdown('<div class="main-title">Single Transaction Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analyze individual transactions in real time by specifying transaction parameters and PCA behavioral components.</div>', unsafe_allow_html=True)

    # Preset profiles
    st.subheader("Reference Profiles")
    fill_col1, fill_col2, fill_col3 = st.columns(3)

    if 'manual_values' not in st.session_state:
        st.session_state.manual_values = {}

    with fill_col1:
        if st.button("Load Typical Legitimate Profile"):
            if sample_df is not None:
                legit_row = sample_df[sample_df['Actual_Class'] == 0].iloc[0]
                st.session_state.manual_values['Time'] = float(legit_row['Time'])
                st.session_state.manual_values['Amount'] = float(legit_row['Amount'])
                for i in range(1, 29):
                    st.session_state.manual_values[f"V{i}"] = float(legit_row[f"V{i}"])
                st.rerun()

    with fill_col2:
        if st.button("Load Confirmed High-Risk Profile"):
            if sample_df is not None:
                fraud_row = sample_df[sample_df['Actual_Class'] == 1].iloc[0]
                st.session_state.manual_values['Time'] = float(fraud_row['Time'])
                st.session_state.manual_values['Amount'] = float(fraud_row['Amount'])
                for i in range(1, 29):
                    st.session_state.manual_values[f"V{i}"] = float(fraud_row[f"V{i}"])
                st.rerun()

    with fill_col3:
        if st.button("Clear Inputs to Zero"):
            st.session_state.manual_values = {}
            st.rerun()

    st.markdown("---")

    # Primary attributes
    st.subheader("1. Primary Transaction Attributes")
    c_time, c_amt = st.columns(2)
    with c_time:
        input_time = st.number_input(
            "Transaction Timestamp (seconds elapsed):",
            value=float(st.session_state.manual_values.get("Time", 3600.0)),
            step=60.0,
            help="Elapsed seconds since recording start (0 to 172,800 covers a 48-hour recording window)."
        )
    with c_amt:
        input_amount = st.number_input(
            "Transaction Amount (USD):",
            value=float(st.session_state.manual_values.get("Amount", 50.0)),
            min_value=0.0,
            step=1.0,
            format="%.2f"
        )

    # PCA feature attributes
    st.subheader("2. Behavioral Vector Attributes (V1 through V28)")
    st.caption("Principal components capturing transaction velocity, cardholder location patterns, and device behavior.")

    v_inputs = {}
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

    # Evaluation action
    if st.button("Run Fraud Assessment", type="primary", width="stretch"):
        row_dict = {"Time": input_time, "Amount": input_amount}
        row_dict.update(v_inputs)
        single_df = pd.DataFrame([row_dict])

        # Preprocess and score
        processed_input = preprocess_features(single_df, scaler)
        fraud_prob = float(model.predict_proba(processed_input)[0, 1])
        is_fraud = fraud_prob >= threshold

        # Risk Assessment Presentation
        st.markdown("### Risk Evaluation Outcome")

        r1, r2 = st.columns([1, 1])

        with r1:
            if is_fraud:
                st.markdown(f"""
                <div class="fraud-card">
                    <div class="result-status fraud-status">High Risk: Flagged for Review</div>
                    <p class="result-description">
                        This transaction deviates significantly from baseline spending patterns. 
                        Recommended action: Decline transaction or trigger multi-factor step-up verification.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-card">
                    <div class="result-status legit-status">Low Risk: Cleared for Approval</div>
                    <p class="result-description">
                        This transaction conforms with standard cardholder behavioral profiles. 
                        Recommended action: Approve and route for regular authorization.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            st.metric("Estimated Fraud Probability", f"{fraud_prob * 100:.2f}%")
            st.progress(fraud_prob)
            st.caption(f"Operational threshold: {threshold * 100:.1f}%. Probability scores exceeding this cutoff trigger intervention.")

            if fraud_prob >= 0.70:
                st.error("Risk Classification: Critical Risk")
            elif fraud_prob >= threshold:
                st.warning("Risk Classification: Elevated Risk (Threshold Exceeded)")
            else:
                st.success("Risk Classification: Standard / Low Risk")

        # Feature benchmark analysis
        st.markdown("---")
        st.subheader("Key Influencing Indicators")
        key_check = ['V14', 'V4', 'V10', 'V12', 'V11', 'V17']
        key_summary = pd.DataFrame({
            "Indicator": key_check,
            "Submitted Value": [v_inputs[c] for c in key_check],
            "Typical Legitimate Median": [0.03, -0.01, 0.01, 0.01, -0.01, 0.01],
            "Typical Fraudulent Median": [-5.41, +4.54, -5.67, -6.25, +3.80, -6.66]
        })
        st.dataframe(
            key_summary.style.format({
                "Submitted Value": "{:.4f}",
                "Typical Legitimate Median": "{:.2f}",
                "Typical Fraudulent Median": "{:.2f}"
            }),
            width="stretch"
        )
