from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Credit Risk Recovery Dashboard", page_icon="💼", layout="wide")

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "credit_score_model.pkl"
SCALER_PATH = PROJECT_DIR / "scaler.pkl"
DATASET_PATH = PROJECT_DIR / "loan-recovery.csv"

FEATURES = [
    "Age",
    "Monthly_Income",
    "Loan_Amount",
    "Loan_Tenure",
    "Interest_Rate",
    "Collateral_Value",
    "Outstanding_Loan_Amount",
    "Monthly_EMI",
    "Num_Missed_Payments",
    "Days_Past_Due",
]

SEGMENT_ORDER = {
    0: "Moderate Income, High Loan Burden",
    1: "High Income, Low Default Risk",
    2: "Moderate Income, Medium Risk",
    3: "High Loan, Higher Default Risk",
}

light_theme = {
    "bg": "#f4f7fb",
    "panel": "#ffffff",
    "panel_soft": "#eef3fa",
    "text": "#0f172a",
    "muted": "#475569",
    "primary": "#1d4ed8",
    "primary_soft": "#bfdbfe",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "border": "rgba(15, 23, 42, 0.08)",
    "shadow": "rgba(15, 23, 42, 0.1)",
}

dark_theme = {
    "bg": "#071a2d",
    "panel": "#0d233d",
    "panel_soft": "#112d4a",
    "text": "#eaf3ff",
    "muted": "#9ab3c8",
    "primary": "#1b8ef2",
    "primary_soft": "#7dd3fc",
    "success": "#20c997",
    "warning": "#f7b267",
    "danger": "#ff5c7a",
    "border": "rgba(255,255,255,0.08)",
    "shadow": "rgba(0, 0, 0, 0.28)",
}


def get_theme():
    return st.session_state.get("theme", "dark")


def load_css() -> None:
    theme = dark_theme if get_theme() == "dark" else light_theme
    st.markdown(
        f"""
        <style>
            :root {{
                --bg: {theme['bg']};
                --panel: {theme['panel']};
                --panel-soft: {theme['panel_soft']};
                --text: {theme['text']};
                --muted: {theme['muted']};
                --primary: {theme['primary']};
                --primary-soft: {theme['primary_soft']};
                --success: {theme['success']};
                --warning: {theme['warning']};
                --danger: {theme['danger']};
                --border: {theme['border']};
                --shadow: {theme['shadow']};
            }}

            html, body, [data-testid="stAppViewContainer"] {{
                background: linear-gradient(135deg, var(--bg) 0%, color-mix(in srgb, var(--panel) 85%, white 15%) 100%);
                color: var(--text);
            }}

            .stApp {{
                background: transparent;
            }}

            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, var(--panel) 0%, var(--panel-soft) 100%);
                border-right: 1px solid var(--border);
            }}

            .sidebar-header {{
                padding: 1.2rem 1rem 0.4rem 1rem;
                font-size: 1.45rem;
                font-weight: 800;
                color: var(--text);
                letter-spacing: 0.04em;
            }}

            .sidebar-subtitle {{
                padding: 0 1rem 0.8rem 1rem;
                color: var(--muted);
                font-size: 0.82rem;
            }}

            .sidebar-mark {{
                margin: 0.5rem 0 1rem 0;
                padding: 0.7rem 0.9rem;
                border-radius: 12px;
                background: rgba(27, 142, 242, 0.08);
                border: 1px solid rgba(125, 211, 252, 0.25);
                color: var(--text);
                font-size: 0.86rem;
            }}

            .main .block-container {{
                padding-top: 2rem;
                padding-bottom: 2rem;
            }}

            .dashboard-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 0.75rem;
                margin-bottom: 1rem;
            }}

            .title-badge {{
                display: inline-flex;
                align-items: center;
                gap: 0.55rem;
                padding: 0.35rem 0.8rem;
                border-radius: 999px;
                background: rgba(27, 142, 242, 0.12);
                color: var(--primary);
                border: 1px solid rgba(125, 211, 252, 0.35);
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }}

            .main-title {{
                font-size: clamp(2rem, 2.8vw, 3rem);
                font-weight: 800;
                letter-spacing: -0.04em;
                margin: 0.5rem 0 0 0;
                color: var(--text);
            }}

            .subtitle {{
                color: var(--muted);
                font-size: 0.96rem;
                margin-top: 0.2rem;
            }}

            .metric-card {{
                background: linear-gradient(180deg, var(--panel) 0%, color-mix(in srgb, var(--panel) 85%, white 15%) 100%);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: 0 10px 24px var(--shadow);
                transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
                min-height: 120px;
            }}

            .metric-card:hover {{
                transform: translateY(-3px);
                border-color: rgba(125, 211, 252, 0.4);
                box-shadow: 0 18px 40px rgba(14, 93, 170, 0.25);
            }}

            .metric-label {{
                color: var(--muted);
                font-size: 0.74rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.8rem;
                display: block;
            }}

            .metric-value {{
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.1;
                color: var(--text);
            }}

            .metric-delta {{
                margin-top: 0.7rem;
                font-size: 0.82rem;
                color: var(--success);
                font-weight: 600;
            }}

            .stDataFrame {{
                background: rgba(11, 31, 49, 0.02);
                border: 1px solid var(--border);
                border-radius: 16px;
            }}

            .stPlotlyChart > div {{
                border-radius: 18px;
                overflow: hidden;
                box-shadow: 0 10px 24px var(--shadow);
            }}

            .section-box {{
                background: linear-gradient(180deg, var(--panel) 0%, var(--panel-soft) 100%);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1rem 1rem 0.5rem 1rem;
                margin: 1rem 0;
                box-shadow: 0 10px 24px var(--shadow);
            }}

            .stButton > button {{
                background: linear-gradient(135deg, var(--primary), #0f6de6);
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: 700;
                padding: 0.7rem 1.2rem;
                transition: transform 0.18s ease, filter 0.18s ease;
                box-shadow: 0 10px 22px rgba(27, 142, 242, 0.35);
            }}

            .stButton > button:hover {{
                transform: translateY(-2px);
                filter: brightness(1.04);
            }}

            .stDownloadButton > button {{
                background: linear-gradient(135deg, #04b57f, #18a168);
                border: none;
                color: white;
                font-weight: 700;
                border-radius: 12px;
                padding: 0.7rem 1.2rem;
                box-shadow: 0 10px 24px rgba(32, 201, 151, 0.28);
            }}

            .stForm {{
                background: rgba(10, 30, 46, 0.04);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1rem;
                box-shadow: 0 10px 24px var(--shadow);
            }}

            .footer {{
                margin-top: 2rem;
                padding-top: 1.25rem;
                border-top: 1px solid var(--border);
                color: var(--muted);
                font-size: 0.82rem;
                text-align: center;
            }}

            @media (max-width: 768px) {{
                .main .block-container {{
                    padding-left: 1rem;
                    padding-right: 1rem;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    return model, scaler


@st.cache_data
def load_dataset(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)

    return None


def get_risk_strategy(score: float) -> str:
    if score >= 0.75:
        return "Immediate legal notices & aggressive recovery attempts"
    if score >= 0.5:
        return "Settlement offers & repayment plans"
    return "Automated reminders & monitoring"


def risk_label(score: float) -> str:
    if score >= 0.75:
        return "High Risk"
    if score >= 0.5:
        return "Medium Risk"
    return "Low Risk"


def build_risk_dataframe(df: pd.DataFrame):
    if df is None or df.empty:
        return df

    data = df.copy()
    if "Recovery_Status" not in data.columns:
        data["Recovery_Status"] = data.get("High_Risk_Flag", 0).map({0: "Recovered", 1: "Not Recovered"})

    if "Borrower_Segment" not in data.columns:
        _, scaler = load_artifacts()
        if scaler is not None and set(FEATURES).issubset(data.columns):
            scaled = scaler.transform(data[FEATURES])
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled)
            data["Borrower_Segment"] = labels
            data["Segment_Name"] = data["Borrower_Segment"].map(SEGMENT_ORDER)
        else:
            data["Borrower_Segment"] = 0
            data["Segment_Name"] = "Unknown"

    return data


def plot_loan_distribution(df):
    if df is None or df.empty:
        return None
    fig = px.histogram(
        df,
        x="Loan_Amount",
        nbins=30,
        marginal="box",
        opacity=0.8,
        title="Loan Amount Distribution",
        labels={"Loan_Amount": "Loan Amount ($)"},
        color_discrete_sequence=["#3fa4f5"],
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_x=0.5,
        xaxis_title="Loan Amount ($)",
        yaxis_title="Count",
        font={"color": "#eaf3ff"},
        height=360,
    )
    return fig


def plot_income_vs_loan(df):
    if df is None or df.empty:
        return None
    if "Recovery_Status" in df.columns:
        color_col = "Recovery_Status"
        color_map = {"Recovered": "#20c997", "Not Recovered": "#ff5c7a"}
    else:
        color_col = None
        color_map = None

    fig = px.scatter(
        df,
        x="Monthly_Income",
        y="Loan_Amount",
        color=color_col,
        size="Loan_Amount",
        hover_data={"Monthly_Income": True, "Loan_Amount": True, "Recovery_Status": True},
        title="Monthly Income vs Loan Amount",
        labels={"Monthly_Income": "Monthly Income ($)", "Loan_Amount": "Loan Amount ($)"},
        color_discrete_map=color_map,
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_x=0.5,
        xaxis_title="Monthly Income ($)",
        yaxis_title="Loan Amount ($)",
        font={"color": "#eaf3ff"},
        legend_title_text="Recovery Status",
        height=360,
    )
    return fig


def plot_payment_history(df):
    if df is None or df.empty or "Payment_History" not in df.columns:
        return None
    fig = px.histogram(
        df,
        x="Payment_History",
        color="Recovery_Status",
        barmode="group",
        title="Payment History vs Recovery Status",
        labels={"Payment_History": "Payment History", "count": "Number of Loans"},
        color_discrete_map={"Recovered": "#20c997", "Not Recovered": "#ff5c7a"},
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_x=0.5,
        xaxis_title="Payment History",
        yaxis_title="Number of Loans",
        legend_title_text="Recovery Status",
        font={"color": "#eaf3ff"},
        height=360,
    )
    return fig


def make_risk_gauge(score: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Borrower Risk Score"},
            delta={"reference": 50, "increasing": {"color": "#ff5c7a"}, "decreasing": {"color": "#20c997"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#dfe9f4"},
                "bar": {"color": "#1b8ef2"},
                "bgcolor": "rgba(255,255,255,0.08)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.12)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(32, 201, 151, 0.32)"},
                    {"range": [50, 75], "color": "rgba(247, 178, 103, 0.40)"},
                    {"range": [75, 100], "color": "rgba(255, 92, 122, 0.42)"},
                ],
                "threshold": {"line": {"color": "#f8fafc", "width": 4}, "thickness": 0.85, "value": 75},
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=25, r=25, t=35, b=20),
        font={"color": "#eaf3ff"},
    )
    return fig


def make_risk_distribution_chart(df):
    if df is None or df.empty:
        return None

    if "Recovery_Status" in df.columns:
        counts = df["Recovery_Status"].value_counts().reindex(["Recovered", "Not Recovered"]).fillna(0)
        labels = ["Recovered", "Not Recovered"]
        values = [int(counts.get("Recovered", 0)), int(counts.get("Not Recovered", 0))]
        colors = ["#20c997", "#ff5c7a"]
    else:
        labels = ["Low Risk", "Medium Risk", "High Risk"]
        values = [40, 35, 25]
        colors = ["#20c997", "#f7b267", "#ff5c7a"]

    fig = px.pie(
        names=labels,
        values=values,
        title="Risk Distribution",
        color_discrete_sequence=colors,
        hole=0.45,
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_x=0.5,
        legend_title_text="Status",
        font={"color": "#eaf3ff"},
        height=320,
    )
    return fig


def make_feature_importance_chart(model):
    if not hasattr(model, "feature_importances_"):
        return None

    importance_df = pd.DataFrame(
        {"Feature": FEATURES, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        color="Importance",
        color_continuous_scale="Viridis",
        template="plotly_dark",
        text_auto=".2%",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_x=0.5,
        font={"color": "#eaf3ff"},
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=350,
    )
    return fig


def get_prediction_from_input(model, payload: dict) -> tuple[float, bool]:
    row = [payload[col] for col in FEATURES]
    probability = model.predict_proba([row])[0][1]
    predicted_flag = bool(probability > 0.5)
    return float(probability), predicted_flag


def metric_card(title: str, value: str, delta: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">{title}</span>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-delta">{delta}</div>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def dashboard_page():
    st.markdown(
        """
        <div class="dashboard-header">
            <div>
                <div class="title-badge">Finance Intelligence</div>
                <h1 class="main-title">Credit Risk Recovery Dashboard</h1>
                <div class="subtitle">Portfolio monitoring, risk intelligence, and borrower behavior analysis.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.sidebar.file_uploader("Upload a CSV dataset", type=["csv"], help="Upload a dataset matching the notebook schema.")
    df = load_dataset(uploaded_file)

    if df is None:
        st.warning("No CSV file found. Upload your dataset from the sidebar or place a file named loan-recovery.csv in the project folder.")
        st.stop()

    st.sidebar.markdown("<div class='sidebar-header'>💼 CreditLens</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-subtitle'>Risk analytics platform</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-mark'>📊 Overview dashboard<br>📈 Borrower scoring<br>💰 Recovery planning</div>", unsafe_allow_html=True)

    if "Recovery_Status" in df.columns:
        recovered = int(df["Recovery_Status"].eq("Recovered").sum())
        not_recovered = int(df["Recovery_Status"].eq("Not Recovered").sum())
    else:
        recovered = 0
        not_recovered = 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Accounts", f"{len(df):,}", "Portfolio size")
    with col2:
        metric_card("Recovered", f"{recovered:,}", "Collections success")
    with col3:
        metric_card("At Risk", f"{not_recovered:,}", "Monitoring required")
    with col4:
        metric_card("Dataset Columns", f"{len(df.columns):,}", "Signals tracked")

    st.sidebar.download_button(
        label="Export CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="credit_risk_export.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("Portfolio Overview")
    st.dataframe(df.head(10), use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(plot_loan_distribution(df), use_container_width=True)
    with chart_col2:
        st.plotly_chart(plot_income_vs_loan(df), use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        st.plotly_chart(plot_payment_history(df), use_container_width=True)
    with chart_col4:
        st.plotly_chart(make_risk_distribution_chart(df), use_container_width=True)

    if "Num_Missed_Payments" in df.columns and "Recovery_Status" in df.columns:
        missed = px.box(
            df,
            x="Recovery_Status",
            y="Num_Missed_Payments",
            title="Missed Payments vs Recovery Status",
            labels={"Recovery_Status": "Recovery Status", "Num_Missed_Payments": "Number of Missed Payments"},
            color="Recovery_Status",
            color_discrete_map={"Recovered": "#20c997", "Not Recovered": "#ff5c7a"},
            points="all",
            template="plotly_dark",
        )
        missed.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title_x=0.5,
            font={"color": "#eaf3ff"},
            height=340,
        )
        st.plotly_chart(missed, use_container_width=True)


def predict_page():
    st.markdown(
        """
        <div class="dashboard-header">
            <div>
                <div class="title-badge">Risk Engine</div>
                <h1 class="main-title">Borrower Risk Prediction</h1>
                <div class="subtitle">Professional scoring interface powered by the existing Random Forest model.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, _ = load_artifacts()

    with st.form("risk_form"):
        st.subheader("Borrower profile")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            monthly_income = st.number_input("Monthly Income", min_value=0.0, max_value=200000.0, value=6000.0, step=100.0)
            loan_amount = st.number_input("Loan Amount", min_value=0.0, max_value=1000000.0, value=150000.0, step=1000.0)
            loan_tenure = st.number_input("Loan Tenure (months)", min_value=1, max_value=360, value=36)
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=12.0, step=0.1)
        with c2:
            collateral_value = st.number_input("Collateral Value", min_value=0.0, max_value=2000000.0, value=250000.0, step=1000.0)
            outstanding_loan = st.number_input("Outstanding Loan Amount", min_value=0.0, max_value=1000000.0, value=120000.0, step=1000.0)
            monthly_emi = st.number_input("Monthly EMI", min_value=0.0, max_value=500000.0, value=3500.0, step=100.0)
            missed_payments = st.number_input("Number of Missed Payments", min_value=0, max_value=100, value=2)
            days_past_due = st.number_input("Days Past Due", min_value=0, max_value=3650, value=15)

        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    if submitted:
        payload = {
            "Age": age,
            "Monthly_Income": monthly_income,
            "Loan_Amount": loan_amount,
            "Loan_Tenure": loan_tenure,
            "Interest_Rate": interest_rate,
            "Collateral_Value": collateral_value,
            "Outstanding_Loan_Amount": outstanding_loan,
            "Monthly_EMI": monthly_emi,
            "Num_Missed_Payments": missed_payments,
            "Days_Past_Due": days_past_due,
        }

        score, predicted_flag = get_prediction_from_input(model, payload)

        display_col1, display_col2 = st.columns([1.3, 1.7])
        with display_col1:
            st.plotly_chart(make_risk_gauge(score), use_container_width=True)
        with display_col2:
            st.markdown(
                """
                <div class="section-box">
                    <h3 style="margin-top:0; color:#eef7ff;">Risk Summary</h3>
                    <p style="color:#dbeaf9;">The model evaluates borrower risk using financial strain, repayment behavior, and collateral coverage.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                metric_card("Risk Score", f"{score:.2%}", "Probability")
            with col_b:
                metric_card("Risk Level", risk_label(score), "Classification")
            with col_c:
                metric_card("Model", "High Risk" if predicted_flag else "Low Risk", "Decision")

        st.markdown(
            f"### Recommended Recovery Strategy: {get_risk_strategy(score)}"
        )

        if score >= 0.75:
            st.error("This borrower is classified as high-risk. Immediate action is recommended.")
        elif score >= 0.5:
            st.warning("This borrower is moderate-risk. Consider a structured settlement or repayment plan.")
        else:
            st.success("This borrower is low-risk and is likely to remain in good standing with standard follow-up.")

        report_text = "\n".join(
            [
                "Credit Risk Prediction Report",
                "===========================",
                f"Risk Score: {score:.2%}",
                f"Risk Level: {risk_label(score)}",
                f"Recommended Recovery Strategy: {get_risk_strategy(score)}",
                "",
                "Borrower Inputs:",
                *[f"- {key}: {value}" for key, value in payload.items()],
            ]
        )

        report_col1, report_col2 = st.columns([1.6, 1])
        with report_col1:
            st.json(payload)
        with report_col2:
            st.download_button(
                label="Download Prediction Report",
                data=report_text,
                file_name="credit_risk_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        if hasattr(model, "feature_importances_"):
            st.plotly_chart(make_feature_importance_chart(model), use_container_width=True)


def main():
    load_css()

    st.sidebar.title("📊 Credit Recovery")
    st.sidebar.markdown("---")

    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    with st.sidebar:
        nav = st.radio("Navigation", ["🏠 Overview", "📈 Risk Prediction"], index=0, key="nav_page")
        st.markdown("<div class='sidebar-subtitle'>Display</div>", unsafe_allow_html=True)
        theme = st.toggle(
            "Dark mode",
            value=(st.session_state.theme == "dark"),
            key="theme_toggle",
            help="Toggle between dark and light dashboard theme",
        )
        st.session_state.theme = "dark" if theme else "light"
        load_css()

    try:
        load_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Place credit_score_model.pkl in the project directory to enable scoring.")
        return

    if nav == "🏠 Overview":
        dashboard_page()
    else:
        predict_page()

    st.markdown(
        """
        <div class="footer">
            Credit Risk Recovery Dashboard • Built for portfolio monitoring and recovery strategy planning • Streamlit Cloud ready
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
