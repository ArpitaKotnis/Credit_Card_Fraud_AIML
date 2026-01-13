import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium fintech styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2.5rem;
    }
    .fraud-card {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
    }
    .safe-card {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
    }
    .status-text {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .risk-badge {
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        margin-top: 1rem;
        display: inline-block;
    }
    .risk-high { background: #fee2e2; color: #991b1b; }
    .risk-medium { background: #fef3c7; color: #92400e; }
    .risk-low { background: #d1fae5; color: #065f46; }
    .input-panel {
        background: #1e293b;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #334155;
    }
    .panel-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #334155;
        padding-bottom: 0.75rem;
    }
    .metric-card {
        background: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
    }
    .metric-value {
        color: #f1f5f9;
        font-size: 1.5rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("fraud_model.pkl")
    return model

model = load_model()

# Header
st.markdown('<p class="main-header">Credit Card Fraud Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time Transaction Risk Analysis</p>', unsafe_allow_html=True)

# Input panel
st.markdown('<div class="input-panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Transaction Input Panel</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    time = st.number_input(
        "Transaction Time (seconds)",
        min_value=0.0,
        step=1.0,
        help="Seconds elapsed since the first recorded transaction"
    )

with col2:
    amount = st.number_input(
        "Transaction Amount (₹ INR)",  # ✅ FIX
        min_value=0.0,
        step=0.01,
        format="%.2f",
        help="Enter transaction amount in Indian Rupees"
    )

st.markdown("</div>", unsafe_allow_html=True)

# Analyze
if st.button("Analyze Transaction", use_container_width=True):
    transaction = np.array([[time, amount]])
    prediction = model.predict(transaction)[0]
    fraud_prob = model.predict_proba(transaction)[0][1] * 100

    if fraud_prob >= 70:
        risk, css = "High", "risk-high"
    elif fraud_prob >= 40:
        risk, css = "Medium", "risk-medium"
    else:
        risk, css = "Low", "risk-low"

    col = st.columns([1, 2, 1])[1]

    with col:
        if prediction == 1:
            st.markdown(f"""
                <div class="fraud-card">
                    <div class="status-text">FRAUD DETECTED</div>
                    <div class="risk-badge {css}">Risk Level: {risk}</div>
                </div>
            """, unsafe_allow_html=True)
            explanation = "Transaction deviates from normal spending patterns and matches known fraud indicators."
        else:
            st.markdown(f"""
                <div class="safe-card">
                    <div class="status-text">TRANSACTION SAFE</div>
                    <div class="risk-badge {css}">Risk Level: {risk}</div>
                </div>
            """, unsafe_allow_html=True)
            explanation = "Transaction aligns with expected customer spending behavior."

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Transaction Time</div>
                <div class="metric-value">{time:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Transaction Amount</div>
                <div class="metric-value">₹{amount:,.2f}</div>  <!-- ✅ FIX -->
            </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Fraud Probability</div>
                <div class="metric-value">{fraud_prob:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background:#1e293b;padding:1.5rem;border-radius:12px;margin-top:1.5rem;">
        <strong>Analysis Summary:</strong><br>{explanation}
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#64748b;font-size:0.85rem;'>"
    "Fraud Detection System | Machine Learning Based Risk Analysis"
    "</div>",
    unsafe_allow_html=True
)
