"""
Loan Default Prediction — Streamlit Web Application
Machine Learning in Finance · Milestone 4

Run locally:
    streamlit run app.py

Requirements: see requirements.txt
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os, warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-high   { background:#fee2e2; border-left:6px solid #dc2626;
                   padding:14px 18px; border-radius:6px; font-size:1.05rem; }
    .risk-low    { background:#dcfce7; border-left:6px solid #16a34a;
                   padding:14px 18px; border-radius:6px; font-size:1.05rem; }
    .metric-box  { background:#f1f5f9; border-radius:8px; padding:12px 16px;
                   text-align:center; }
    .section-hdr { font-size:1.15rem; font-weight:700; color:#1e3a5f;
                   margin-top:1.2rem; margin-bottom:.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Load artefacts ────────────────────────────────────────────────────────────
PIPELINE_PATH  = "best_pipeline.pkl"
EXPLAINER_PATH = "shap_explainer.pkl"

@st.cache_resource(show_spinner="Loading model…")
def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        return None
    return joblib.load(PIPELINE_PATH)

@st.cache_resource(show_spinner="Loading SHAP explainer…")
def load_explainer():
    if not os.path.exists(EXPLAINER_PATH):
        return None
    return joblib.load(EXPLAINER_PATH)

pipeline  = load_pipeline()
explainer = load_explainer()

FEATURES = [
    "int_rate", "dti", "fico_range_low", "revol_util", "annual_inc_log",
    "loan_amnt_log", "installment", "open_acc", "pub_rec", "delinq_2yrs",
    "credit_history_years", "income_to_loan_ratio", "installment_to_income",
    "total_interest_burden", "sub_grade_freq", "emp_stable",
    "revol_bal_log", "annual_inc",
]

FEATURE_LABELS = {
    "int_rate":               "Interest Rate (%)",
    "dti":                    "Debt-to-Income Ratio (%)",
    "fico_range_low":         "FICO Score (lower bound)",
    "revol_util":             "Revolving Credit Utilisation (%)",
    "annual_inc":             "Annual Income ($)",
    "installment":            "Monthly Installment ($)",
    "open_acc":               "Number of Open Accounts",
    "pub_rec":                "Public Derogatory Records",
    "delinq_2yrs":            "Delinquencies (past 2 years)",
    "credit_history_years":   "Credit History Length (years)",
    "loan_amnt_log":          "Loan Amount (log-transformed) — auto",
    "annual_inc_log":         "Annual Income (log-transformed) — auto",
    "income_to_loan_ratio":   "Income-to-Loan Ratio — auto",
    "installment_to_income":  "Installment-to-Income Ratio — auto",
    "total_interest_burden":  "Total Interest Burden ($) — auto",
    "sub_grade_freq":         "Sub-Grade Frequency — auto",
    "revol_bal_log":          "Revolving Balance (log) — auto",
    "emp_stable":             "Employed ≥ 5 Years — auto",
}

# ── Sidebar: applicant inputs ─────────────────────────────────────────────────
st.sidebar.header("📋 Applicant Information")

st.sidebar.markdown("**Loan Details**")
loan_amnt = st.sidebar.number_input("Loan Amount ($)",        1_000, 40_000, 12_000, 500)
int_rate  = st.sidebar.slider("Interest Rate (%)",            5.0,   30.0,   13.5,  0.1)
installment = st.sidebar.number_input("Monthly Installment ($)", 50, 1_500, 380, 10)
term_60   = st.sidebar.selectbox("Loan Term", ["36 months", "60 months"])

st.sidebar.markdown("**Borrower Profile**")
annual_inc   = st.sidebar.number_input("Annual Income ($)",     10_000, 500_000, 65_000, 1_000)
dti          = st.sidebar.slider("Debt-to-Income Ratio (%)",    0.0,    45.0,   18.0,   0.1)
fico         = st.sidebar.slider("FICO Score",                  580,    850,    700,    1)
revol_util   = st.sidebar.slider("Revolving Credit Utilisation (%)", 0.0, 100.0, 45.0, 0.1)
open_acc     = st.sidebar.number_input("Number of Open Accounts",  1, 40, 10, 1)
pub_rec      = st.sidebar.number_input("Public Derogatory Records", 0, 10, 0, 1)
delinq_2yrs  = st.sidebar.number_input("Delinquencies (past 2 yrs)", 0, 15, 0, 1)
cr_hist_yrs  = st.sidebar.slider("Credit History Length (years)", 0.0, 35.0, 8.0, 0.5)
revol_bal    = st.sidebar.number_input("Revolving Balance ($)",   0, 200_000, 15_000, 500)
emp_yrs      = st.sidebar.selectbox("Employment Length", ["< 1 year","1 year","2 years",
                                     "3 years","4 years","5 years","6 years",
                                     "7 years","8 years","9 years","10+ years"])
sub_grade_freq = st.sidebar.slider(
    "Sub-Grade Frequency (0–1, use 0.05 if unknown)", 0.01, 0.15, 0.05, 0.005)

# ── Derived / engineered features ─────────────────────────────────────────────
annual_inc_log        = np.log1p(annual_inc)
loan_amnt_log         = np.log1p(loan_amnt)
revol_bal_log         = np.log1p(revol_bal)
income_to_loan_ratio  = min(annual_inc / max(loan_amnt, 1), 100)
installment_to_income = min(installment / max(annual_inc / 12, 1), 5)
total_interest_burden = loan_amnt * int_rate / 100
emp_yrs_num           = 10 if emp_yrs == "10+ years" else (
                         0 if emp_yrs == "< 1 year" else
                         int(emp_yrs.split()[0]))
emp_stable            = float(emp_yrs_num >= 5)

input_dict = {
    "int_rate":               int_rate,
    "dti":                    dti,
    "fico_range_low":         fico,
    "revol_util":             revol_util,
    "annual_inc_log":         annual_inc_log,
    "loan_amnt_log":          loan_amnt_log,
    "installment":            installment,
    "open_acc":               float(open_acc),
    "pub_rec":                float(pub_rec),
    "delinq_2yrs":            float(delinq_2yrs),
    "credit_history_years":   cr_hist_yrs,
    "income_to_loan_ratio":   income_to_loan_ratio,
    "installment_to_income":  installment_to_income,
    "total_interest_burden":  total_interest_burden,
    "sub_grade_freq":         sub_grade_freq,
    "emp_stable":             emp_stable,
    "revol_bal_log":          revol_bal_log,
    "annual_inc":             float(annual_inc),
}

X_input = pd.DataFrame([input_dict])[FEATURES]

# ── Main page ─────────────────────────────────────────────────────────────────
st.title("🏦 Loan Default Prediction")
st.caption("Machine Learning in Finance — Milestone 4 | XGBoost Model")

if pipeline is None:
    st.error(
        "⚠️ **Model file not found.** "
        "Please run the Milestone 4 notebook first to generate `best_pipeline.pkl`."
    )
    st.info(
        "**Expected files in the same directory as app.py:**\n"
        "- `best_pipeline.pkl` (generated in Section 3.6 of the notebook)\n"
        "- `shap_explainer.pkl` (generated in Section 6.5 of the notebook)"
    )
    st.stop()

col1, col2 = st.columns([1.1, 1.9])

# ── Left column: prediction result ───────────────────────────────────────────
with col1:
    st.markdown('<div class="section-hdr">📊 Risk Assessment</div>',
                unsafe_allow_html=True)

    prob  = pipeline.predict_proba(X_input)[0, 1]
    label = pipeline.predict(X_input)[0]

    # Optimal threshold from notebook (≈ 0.35)
    THRESHOLD = 0.35
    high_risk = prob >= THRESHOLD

    if high_risk:
        st.markdown(
            f'<div class="risk-high">⚠️ <b>High Risk — Likely Default</b><br>'
            f'Predicted default probability: <b>{prob:.1%}</b></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="risk-low">✅ <b>Low Risk — Likely to Repay</b><br>'
            f'Predicted default probability: <b>{prob:.1%}</b></div>',
            unsafe_allow_html=True)

    st.markdown("---")

    # Probability gauge bar
    fig_gauge, ax_g = plt.subplots(figsize=(4.5, 0.9))
    ax_g.barh([0], [1], color="#e5e7eb", height=0.5)
    bar_color = "#dc2626" if high_risk else "#16a34a"
    ax_g.barh([0], [prob], color=bar_color, height=0.5)
    ax_g.axvline(THRESHOLD, color="#1e3a5f", lw=1.5, ls="--", label=f"Threshold ({THRESHOLD:.0%})")
    ax_g.set_xlim(0, 1)
    ax_g.set_yticks([])
    ax_g.set_xlabel("Default Probability")
    ax_g.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax_g.legend(fontsize=7, loc="upper right")
    ax_g.set_title("Risk Score", fontsize=10, fontweight="bold")
    fig_gauge.tight_layout()
    st.pyplot(fig_gauge, use_container_width=True)
    plt.close(fig_gauge)

    st.markdown("---")
    st.markdown('<div class="section-hdr">📋 Input Summary</div>',
                unsafe_allow_html=True)
    summary = pd.DataFrame({
        "Feature": ["Loan Amount", "Interest Rate", "FICO Score",
                    "DTI Ratio", "Annual Income", "Revolving Util.",
                    "Credit History", "Delinquencies"],
        "Value": [f"${loan_amnt:,.0f}", f"{int_rate:.1f}%", f"{fico}",
                  f"{dti:.1f}%", f"${annual_inc:,.0f}", f"{revol_util:.1f}%",
                  f"{cr_hist_yrs:.1f} yrs", f"{delinq_2yrs}"],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

# ── Right column: SHAP explanation ───────────────────────────────────────────
with col2:
    st.markdown('<div class="section-hdr">🔍 Prediction Explanation (SHAP)</div>',
                unsafe_allow_html=True)

    if explainer is None:
        st.warning(
            "SHAP explainer not found. Run Section 6.5 of the notebook to generate "
            "`shap_explainer.pkl`, then restart the app."
        )
    else:
        try:
            import shap
            shap_vals = explainer.shap_values(X_input)

            # Waterfall plot
            fig_wf, ax_wf = plt.subplots(figsize=(8, 5))

            feat_vals  = X_input.iloc[0].values
            sv         = shap_vals[0]
            base_value = explainer.expected_value

            # Build waterfall data
            order      = np.argsort(np.abs(sv))[::-1][:10]
            feats_top  = [FEATURES[i] for i in order]
            sv_top     = sv[order]
            fv_top     = feat_vals[order]

            colors = ["#dc2626" if v > 0 else "#16a34a" for v in sv_top]
            labels = [f"{f}\n= {fv:.2f}" for f, fv in zip(feats_top, fv_top)]

            bars = ax_wf.barh(labels[::-1], sv_top[::-1],
                              color=colors[::-1], edgecolor="white", height=0.65)
            ax_wf.axvline(0, color="black", lw=0.8)
            for bar, val in zip(bars, sv_top[::-1]):
                ax_wf.text(
                    val + (0.001 if val >= 0 else -0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.4f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=8
                )
            ax_wf.set_xlabel("SHAP Value (impact on default probability)")
            ax_wf.set_title(
                f"Top 10 Feature Contributions\n"
                f"Base rate: {base_value:.3f}  →  Predicted: {prob:.3f}",
                fontsize=11, fontweight="bold"
            )
            from matplotlib.patches import Patch
            ax_wf.legend(
                handles=[Patch(color="#dc2626", label="Increases risk"),
                         Patch(color="#16a34a", label="Decreases risk")],
                loc="lower right", fontsize=8
            )
            fig_wf.tight_layout()
            st.pyplot(fig_wf, use_container_width=True)
            plt.close(fig_wf)

            st.markdown("---")
            # Feature contribution table
            st.markdown("**Feature-Level SHAP Contributions**")
            shap_df = pd.DataFrame({
                "Feature":    [FEATURES[i] for i in np.argsort(np.abs(sv))[::-1]],
                "Value":      [f"{feat_vals[i]:.4f}"  for i in np.argsort(np.abs(sv))[::-1]],
                "SHAP Impact":[f"{sv[i]:+.4f}" for i in np.argsort(np.abs(sv))[::-1]],
                "Direction":  ["↑ Risk" if sv[i] > 0 else "↓ Risk"
                               for i in np.argsort(np.abs(sv))[::-1]],
            }).head(12)
            st.dataframe(shap_df, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"SHAP plot error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: XGBoost (tuned via GridSearchCV) | "
    "Dataset: LendingClub 2007–2018 | "
    "Imbalance handling: scale_pos_weight | "
    f"Decision threshold: {THRESHOLD:.0%} (F1-optimal)"
)
