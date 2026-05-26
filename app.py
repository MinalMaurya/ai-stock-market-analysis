from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import download_ohlcv, load_csv, demo_data
from src.features import add_indicators, make_supervised, FEATURE_LIST
from src.modeling import get_model, evaluate, ts_split, walkforward_cv_acc
from src.backtest import backtest


# ---------------- UI SETUP ----------------
st.set_page_config(
    page_title="AI Stock Vision",
    page_icon="📈",
    layout="wide",
)

mode = st.sidebar.radio("Theme", ["Light", "Dark"], index=1)

theme = {
    "Dark": {
        "BG": "#0E1117",
        "FG": "#FAFAFA",
        "CARD": "#161B22",
        "INPUT": "#262730",
        "ACCENT": "#4DA3FF",
    },
    "Light": {
        "BG": "#FFFFFF",
        "FG": "#111111",
        "CARD": "#F5F7FA",
        "INPUT": "#FFFFFF",
        "ACCENT": "#1F4C8F",
    },
}

colors = theme[mode]

st.markdown(
    f"""
    <style>

    /* =========================
       GLOBAL APP
    ========================= */

    .stApp {{
        background-color: {colors["BG"]};
        color: {colors["FG"]};
    }}

    html, body, [class*="css"] {{
        color: {colors["FG"]} !important;
    }}

    /* =========================
       TOP HEADER / TOOLBAR
    ========================= */

    header {{
        background-color: {colors["BG"]} !important;
    }}

    [data-testid="stHeader"] {{
        background-color: {colors["BG"]} !important;
    }}

    [data-testid="stToolbar"] {{
        background-color: {colors["BG"]} !important;
    }}

    [data-testid="stDecoration"] {{
        background-color: {colors["BG"]} !important;
    }}

    /* =========================
       SIDEBAR
    ========================= */

    section[data-testid="stSidebar"] {{
        background-color: {colors["CARD"]} !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }}

    section[data-testid="stSidebar"] * {{
        color: {colors["FG"]} !important;
    }}

    /* =========================
       TEXT
    ========================= */

    h1, h2, h3, h4, h5, h6 {{
        color: {colors["FG"]} !important;
        font-weight: 700;
    }}

    p, span, label, div {{
        color: {colors["FG"]} !important;
    }}

    /* =========================
       INPUT BOXES
    ========================= */

    input,
    textarea {{
        background-color: {colors["INPUT"]} !important;
        color: {colors["FG"]} !important;
        border-radius: 10px !important;
        border: 1px solid rgba(150,150,150,0.45) !important;
    }}

    input::placeholder,
    textarea::placeholder {{
        color: rgba(200,200,200,0.6) !important;
    }}

    /* =========================
       SELECTBOX
    ========================= */

    div[data-baseweb="select"] > div {{
        background-color: {colors["INPUT"]} !important;
        color: {colors["FG"]} !important;
        border-radius: 10px !important;
        border: 1px solid rgba(150,150,150,0.45) !important;
    }}

    div[data-baseweb="select"] span {{
        color: {colors["FG"]} !important;
    }}

    div[data-baseweb="select"] input {{
        color: {colors["FG"]} !important;
    }}

    /* Dropdown menu */
    ul[role="listbox"] {{
        background-color: {colors["INPUT"]} !important;
    }}

    li[role="option"] {{
        background-color: {colors["INPUT"]} !important;
        color: {colors["FG"]} !important;
    }}

    li[role="option"]:hover {{
        background-color: rgba(100,100,100,0.25) !important;
    }}

    li[role="option"] div {{
        color: {colors["FG"]} !important;
    }}

    div[data-baseweb="popover"] {{
        background-color: {colors["INPUT"]} !important;
    }}

    div[data-baseweb="popover"] * {{
        color: {colors["FG"]} !important;
    }}

    /* =========================
       DATE PICKER / CALENDAR
    ========================= */

    .stDateInput input {{
        background-color: {colors["INPUT"]} !important;
        color: {colors["FG"]} !important;
    }}

    div[data-baseweb="calendar"],
    div[data-baseweb="calendar"] * {{
        background-color: {colors["INPUT"]} !important;
        color: {colors["FG"]} !important;
    }}

    div[data-baseweb="datepicker"],
    div[data-baseweb="datepicker"] * {{
        background-color: {colors["INPUT"]} !important;
        color: {colors["FG"]} !important;
    }}

    button[aria-label] {{
        color: {colors["FG"]} !important;
        background-color: {colors["CARD"]} !important;
    }}

    /* =========================
       FILE UPLOADER
    ========================= */

    section[data-testid="stFileUploaderDropzone"] {{
        background-color: {colors["INPUT"]} !important;
        border: 1px dashed rgba(150,150,150,0.6) !important;
        border-radius: 12px !important;
    }}

    section[data-testid="stFileUploaderDropzone"] * {{
        color: {colors["FG"]} !important;
    }}

    section[data-testid="stFileUploaderDropzone"] button {{
        background-color: {colors["ACCENT"]} !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }}

    /* =========================
       METRICS
    ========================= */

    div[data-testid="metric-container"] {{
        background-color: {colors["CARD"]} !important;
        border-radius: 16px;
        padding: 18px;
        border: 1px solid rgba(128,128,128,0.25);
        box-shadow: 0px 2px 12px rgba(0,0,0,0.08);
    }}

    /* =========================
       BUTTONS
    ========================= */

    .stButton > button,
    .stDownloadButton > button {{
        background-color: {colors["ACCENT"]} !important;
        color: white !important;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        opacity: 0.9;
    }}

    /* =========================
       ALERTS
    ========================= */

    .stAlert {{
        border-radius: 12px;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 AI Stock Vision: Intelligent Market Trend Analyzer")
st.caption(
    "AI-powered stock trend prediction, feature engineering, backtesting, CSV support, and user-friendly analysis."
)


# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("Setup")

market = st.sidebar.selectbox(
    "Market",
    ["US (Stooq/Yahoo)", "India (Stooq/Yahoo)"],
    index=0,
)

stock_list_us = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google / Alphabet": "GOOGL",
    "Amazon": "AMZN",
    "Meta": "META",
    "Nvidia": "NVDA",
    "Tesla": "TSLA",
    "Netflix": "NFLX",
    "Adobe": "ADBE",
    "Intel": "INTC",
    "AMD": "AMD",
    "JPMorgan Chase": "JPM",
    "Coca Cola": "KO",
    "Walmart": "WMT",
    "Disney": "DIS",
}

stock_list_in = {
    "Reliance Industries": "RELIANCE",
    "Tata Consultancy Services": "TCS",
    "Infosys": "INFY",
    "HDFC Bank": "HDFCBANK",
    "ICICI Bank": "ICICIBANK",
    "State Bank of India": "SBIN",
    "ITC": "ITC",
    "Larsen & Toubro": "LT",
    "Bharti Airtel": "BHARTIARTL",
    "Axis Bank": "AXISBANK",
    "Hindustan Unilever": "HINDUNILVR",
    "Maruti Suzuki": "MARUTI",
    "Tata Motors": "TATAMOTORS",
    "Wipro": "WIPRO",
    "Asian Paints": "ASIANPAINT",
}

stock_dict = stock_list_in if market.startswith("India") else stock_list_us

selected_stock_name = st.sidebar.selectbox(
    "Search / Select Stock",
    options=list(stock_dict.keys()),
    index=0,
    help="Select a company name. The app automatically uses the correct symbol.",
)

manual_ticker = st.sidebar.text_input(
    "Or type stock symbol manually",
    value=stock_dict[selected_stock_name],
    help="Example: RELIANCE, TCS, INFY, AAPL, MSFT",
)

ticker = manual_ticker.strip().upper()

source = st.sidebar.selectbox(
    "Data source",
    ["auto", "stooq", "yahoo"],
    help="Auto tries available sources and falls back safely.",
)

date_col1, date_col2 = st.sidebar.columns(2)
start_date = date_col1.date_input("Start date", value=date(2018, 1, 1))
end_date = date_col2.date_input("End date", value=date.today())

horizon = st.sidebar.slider(
    "Prediction horizon (days)",
    min_value=1,
    max_value=10,
    value=5,
)

threshold = st.sidebar.slider(
    "Trade threshold (probability ≥)",
    min_value=0.50,
    max_value=0.70,
    value=0.50,
    step=0.01,
)

uploaded = st.sidebar.file_uploader(
    "Or upload CSV",
    type=["csv"],
    help="CSV should contain Date, Open, High, Low, Close, Volume.",
)

use_demo = st.sidebar.checkbox("Use Demo Mode")

run = st.sidebar.button("▶ Run Analysis")


# ---------------- HELPERS ----------------
@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, start: str, end: str | None, source: str):
    return download_ohlcv(ticker, start, end, source=source)


def safe_date_str(d):
    return None if d in (None, "", "None") else pd.to_datetime(d).strftime("%Y-%m-%d")


def normalize_ticker(user_ticker, market, source):
    t = user_ticker.strip().upper().replace(" ", "")

    if market.startswith("India") and source in ["yahoo", "auto"]:
        if not t.endswith(".NS") and not t.endswith(".BO"):
            t = t + ".NS"

    if source == "stooq":
        t = t.replace(".NS", "").replace(".BO", "")

    return t


# ---------------- MAIN WORKFLOW ----------------
if run:
    st.subheader("1) Data")

    px = None
    start_str = safe_date_str(start_date)

    # APIs often treat end date as exclusive, so add 1 day
    adjusted_end_date = end_date + timedelta(days=1)
    end_str = safe_date_str(adjusted_end_date)

    try:
        if uploaded is not None:
            px = load_csv(uploaded)
            st.success("Loaded data from CSV.")

        elif use_demo:
            px = demo_data()
            st.info("Using synthetic demo data.")

        else:
            t = normalize_ticker(ticker, market, source)

            try:
                px = fetch_data(t, start_str, end_str, source=source)
                st.success(f"Loaded market data from **{px.attrs.get('source', 'unknown')}**.")

            except Exception as first_err:
                if source == "stooq":
                    t2 = t if not market.startswith("India") else t.replace(".NS", "") + ".NS"
                    try:
                        px = fetch_data(t2, start_str, end_str, source="yahoo")
                        st.success(f"Fallback loaded from **{px.attrs.get('source', 'unknown')}**.")
                    except Exception:
                        pass

                elif source == "yahoo":
                    t2 = t.replace(".NS", "").replace(".BO", "")
                    try:
                        px = fetch_data(t2, start_str, end_str, source="stooq")
                        st.success(f"Fallback loaded from **{px.attrs.get('source', 'unknown')}**.")
                    except Exception:
                        pass

                if px is None or px.empty:
                    st.warning("Online fetch failed. Upload a CSV or use Demo Mode.")
                    st.error(f"Details: {first_err}")
                    st.stop()

    except Exception as e:
        st.error(f"Details: {e}")
        st.stop()

    if px is None or px.empty:
        st.stop()

    latest_market_date = px.index.max().date()
    today_date = date.today()

    if latest_market_date < today_date:
        st.warning(
            f"Market data is available only up to {latest_market_date}. "
            f"Today may be a weekend, holiday, or market data may not be updated yet."
        )
    else:
        st.success(f"Latest market data available for today: {latest_market_date}")

    st.write(
        f"**Rows:** {len(px):,} | "
        f"**Range:** {px.index.min().date()} → {px.index.max().date()}"
    )

    close_chart = px["Close"]

    if isinstance(close_chart, pd.DataFrame):
        close_chart = close_chart.iloc[:, 0]

    close_chart = pd.to_numeric(close_chart, errors="coerce").dropna()
    close_chart.name = "Close"

    st.line_chart(close_chart)

    # ---------------- FEATURES ----------------
    st.subheader("2) Features & Target")

    with st.spinner("Engineering indicators…"):
        feat = add_indicators(
            px,
            rsi_window=14,
            sma_windows=[5, 10, 20],
            ema_fast=12,
            ema_slow=26,
            macd_signal=9,
            bb_window=20,
        )
        sup = make_supervised(feat, horizon_days=horizon)

    available_features = [c for c in FEATURE_LIST if c in sup.columns]

    X = sup[available_features].copy()
    y = sup["target_up"].copy()

    valid = X.dropna().index.intersection(y.dropna().index)

    X = X.loc[valid]
    y = y.loc[valid]
    close_valid = sup["Close"].reindex(valid)

    st.write("### Feature Engineering Summary")

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Total Features Used", len(X.columns))
    f2.metric("Total Records", len(X))
    f3.metric("Prediction Horizon", f"{horizon} days")
    f4.metric("Target Column", "target_up")

    st.info(
        f"The target column tells the model whether the stock price will go UP after "
        f"{horizon} day(s). If future price is higher, target = 1. Otherwise, target = 0."
    )

    target_counts = y.value_counts().rename(index={0: "Down / Hold", 1: "Up"})

    st.write("### Target Distribution")
    st.bar_chart(target_counts)

    with st.expander("View all features used by the model"):
        st.write(list(X.columns))

    with st.expander("Preview feature dataset"):
        preview_df = X.copy()
        preview_df["target_up"] = y
        st.dataframe(preview_df.tail(20), use_container_width=True)

    missing_features = [c for c in FEATURE_LIST if c not in sup.columns]
    if missing_features:
        st.warning(f"Some expected features were not created: {missing_features}")

    if len(X) < 200:
        st.warning("Not enough rows after feature engineering. Try a longer date range.")
        st.stop()

    # ---------------- TRAIN TEST ----------------
    st.subheader("3) Train / Test Split")

    Xtr, Xte, ytr, yte = ts_split(X, y, test_ratio=0.2)
    st.write(f"Train: **{len(Xtr):,}** | Test: **{len(Xte):,}**")

    # ---------------- CV ----------------
    st.subheader("4) Cross-Validation (Train)")

    with st.spinner("Walk-forward CV…"):
        cv_acc = walkforward_cv_acc(Xtr, ytr, n_splits=5, random_state=42)

    st.metric("CV Accuracy (train)", f"{cv_acc:.3f}")

    # ---------------- FINAL MODEL ----------------
    st.subheader("5) Final Model & Test Metrics")

    with st.spinner("Fitting final model…"):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", get_model(42)),
            ]
        )
        pipe.fit(Xtr, ytr)
        proba_te = pipe.predict_proba(Xte)[:, 1]
        cls = evaluate(yte, proba_te, threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{cls['accuracy']:.3f}")
    c2.metric("Precision", f"{cls['precision']:.3f}")
    c3.metric("Recall", f"{cls['recall']:.3f}")
    c4.metric("ROC-AUC", f"{cls['roc_auc']:.3f}")

    st.write("**Confusion Matrix:**", cls["confusion_matrix"])

    with st.expander("Full classification report"):
        st.code(cls["classification_report"])

    # ---------------- BACKTEST ----------------
    st.subheader("6) Backtest")

    with st.spinner("Simulating strategy…"):
        bt = backtest(
            close_valid.loc[Xte.index],
            pd.Series(proba_te, index=Xte.index),
            threshold=threshold,
            plot_file="equity_curve.png",
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("CAGR (Strategy)", f"{bt['cagr_strategy']:.2%}")
    m2.metric("CAGR (Buy&Hold)", f"{bt['cagr_buyhold']:.2%}")
    m3.metric("Sharpe (Strategy)", f"{bt['sharpe_strategy']:.3f}")

    n1, n2 = st.columns(2)
    n1.metric("Max Drawdown (Strategy)", f"{bt['maxdd_strategy']:.2%}")
    n2.metric("Max Drawdown (Buy&Hold)", f"{bt['maxdd_buyhold']:.2%}")

    st.image(
        "equity_curve.png",
        caption="Equity Curve (Strategy vs Buy&Hold)",
        use_container_width=True,
    )

    proba_last = float(proba_te[-1])
    decision = int(proba_last >= threshold)

    st.info(
        f"P(up) = **{proba_last:.3f}** → "
        f"**{'BUY (1)' if decision else 'HOLD (0)'}** "
        f"(threshold {threshold:.2f})"
    )

    close_for_pred = close_valid.loc[Xte.index]

    if isinstance(close_for_pred, pd.DataFrame):
        close_for_pred = close_for_pred.iloc[:, 0]

    close_for_pred = pd.Series(close_for_pred).reset_index(drop=True)
    proba_for_pred = pd.Series(proba_te).reset_index(drop=True)

    pred_df = pd.DataFrame(
        {
            "date": pd.Series(Xte.index).reset_index(drop=True),
            "close": close_for_pred,
            "proba_up": proba_for_pred,
            "signal_next_day": (proba_for_pred >= threshold).astype(int),
        }
    )

    st.download_button(
        "⬇ Download predictions (CSV)",
        pred_df.to_csv(index=False),
        "predictions_test.csv",
        "text/csv",
    )

    with open("equity_curve.png", "rb") as f:
        st.download_button(
            "⬇ Download equity curve (PNG)",
            f.read(),
            "equity_curve.png",
            "image/png",
        )

else:
    st.info("Set your stock, select settings, and press **Run Analysis**.")
    st.caption("For India stocks, you can type RELIANCE, TCS, INFY. The app adds .NS automatically.")