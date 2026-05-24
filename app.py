

# app.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import download_ohlcv, load_csv, demo_data
from src.features import add_indicators, make_supervised, FEATURE_LIST
from src.modeling import get_model, evaluate, ts_split, walkforward_cv_acc
from src.backtest import backtest

# ---------------- UI SETUP ----------------
st.set_page_config(
    page_title="AI Stock: Market Trend Analysis",
    page_icon="📈",
    layout="wide",
)
st.title("🧠 AI for Market Trend Analysis")
st.caption("Clear next-day (or 5-day) prediction, simple backtest, CSV fallback, and friendly labels.")

# Simple in-app theme toggle (affects charts/containers; Streamlit global theme stays as user/system)
mode = st.sidebar.radio("Theme", ["Light", "Dark"], index=1)
BG = "#0E1117" if mode == "Dark" else "white"
FG = "white" if mode == "Dark" else "black"
ACCENT = "#2E86C1" if mode == "Dark" else "#1F4C8F"

st.markdown(
    f"""
    <style>
        .main {{ background-color: {BG}; color: {FG}; }}
        .stMetric label, .stMarkdown, .stSelectbox div, .stRadio div, .stSlider label {{ color: {FG}; }}
        .stButton>button {{ border-radius: 8px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("Setup")

market = st.sidebar.selectbox(
    "Market",
    ["US (Stooq/Yahoo)", "India (Stooq/Yahoo)"],
    index=0
)

popular_us = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
popular_in = ["RELIANCE", "TCS", "SBIN", "INFY", "HDFCBANK", "ITC"]

ticker = st.sidebar.text_input(
    "Ticker",
    value="AAPL" if market.startswith("US") else "RELIANCE",
    help="US examples: AAPL, MSFT (Stooq/Yahoo). India examples: RELIANCE (Stooq), RELIANCE.NS (Yahoo)."
)

source = st.sidebar.selectbox(
    "Data source",
    ["auto", "stooq", "yahoo"],
    help="If Yahoo is blocked on your network, prefer 'stooq' or 'auto'."
)

date_col1, date_col2 = st.sidebar.columns(2)
start_date = date_col1.date_input("Start date", value=date(2018, 1, 1))
end_date = date_col2.date_input("End date (optional)", value=None)

horizon = st.sidebar.slider("Prediction horizon (days)", min_value=1, max_value=10, value=1)
threshold = st.sidebar.slider("Trade threshold (probability ≥)", min_value=0.50, max_value=0.70, value=0.50, step=0.01)

uploaded = st.sidebar.file_uploader(
    "Or upload CSV (Date,Open,High,Low,Close,Volume)",
    type=["csv"]
)
use_demo = st.sidebar.checkbox("Use Demo Mode (synthetic data)")

st.sidebar.caption("Tip: For India via Yahoo, use the .NS suffix (e.g., RELIANCE.NS). For Stooq, no suffix.")

run = st.sidebar.button("▶ Run Analysis")

# ---------------- HELPERS ----------------
@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, start: str, end: str | None, source: str):
    px = download_ohlcv(ticker, start, end, source=source)
    return px

def safe_date_str(d):
    return None if d in (None, "", "None") else pd.to_datetime(d).strftime("%Y-%m-%d")

# ---------------- MAIN WORKFLOW ----------------
if run:
    st.subheader("1) Data")
    px = None
    start_str = safe_date_str(start_date)
    end_str = safe_date_str(end_date) if end_date else None

    try:
        if uploaded is not None:
            px = load_csv(uploaded)
            st.success("Loaded data from CSV.")
        elif use_demo:
            px = demo_data()
            st.info("Using synthetic demo data.")
        else:
            # normalize ticker based on chosen source/market
            t = ticker.strip().upper()
            if source == "yahoo" and market.startswith("India") and not t.endswith(".NS"):
                t = t + ".NS"
            if source == "stooq" and t.endswith(".NS"):
                t = t.replace(".NS", "")

            # try chosen source
            try:
                px = fetch_data(t, start_str, end_str, source=source)
                st.success(f"Loaded market data from **{px.attrs.get('source','unknown')}**.")
            except Exception as first_err:
                # smart fallback across sources
                fallback_tried = False
                if source == "stooq":
                    # try Yahoo with .NS for India
                    t2 = t if not market.startswith("India") else (t if t.endswith(".NS") else t + ".NS")
                    try:
                        px = fetch_data(t2, start_str, end_str, source="yahoo")
                        st.success(f"Fallback loaded from **{px.attrs.get('source','unknown')}**.")
                        fallback_tried = True
                    except Exception:
                        pass
                elif source == "yahoo":
                    # try Stooq without .NS
                    t2 = t.replace(".NS", "")
                    try:
                        px = fetch_data(t2, start_str, end_str, source="stooq")
                        st.success(f"Fallback loaded from **{px.attrs.get('source','unknown')}**.")
                        fallback_tried = True
                    except Exception:
                        pass

                if px is None or px.empty:
                    if uploaded is None and not use_demo:
                        st.warning("Online fetch failed. You can either upload a CSV or tick Demo Mode.")
                    st.error(f"Details: {first_err}")
                    st.stop()

    except Exception as e:
        if uploaded is None and not use_demo:
            st.warning("Online fetch failed. You can either upload a CSV or tick Demo Mode.")
        st.error(f"Details: {e}")

    if px is None or px.empty:
        st.stop()

    #st.write(f"**Rows:** {len(px):,} &nbsp;&nbsp; **Range:** {px.index.min().date()} → {px.index.max().date()}")
    st.line_chart(px["Close"].rename("Close"))

    st.subheader("2) Features & Target")
    with st.spinner("Engineering indicators…"):
        feat = add_indicators(
            px,
            rsi_window=14,
            sma_windows=[5, 10, 20],
            ema_fast=12,
            ema_slow=26,
            macd_signal=9,
            bb_window=20
        )
        sup = make_supervised(feat, horizon_days=horizon)

    X = sup[[c for c in FEATURE_LIST if c in sup.columns]].copy()
    y = sup["target_up"].copy()

    valid = X.dropna().index.intersection(y.dropna().index)
    X, y = X.loc[valid], y.loc[valid]
    close_valid = sup["Close"].reindex(valid)

    if len(X) < 200:
        st.warning("Not enough rows after feature engineering. Try a longer date range.")
        st.stop()

    st.subheader("3) Train / Test Split")
    Xtr, Xte, ytr, yte = ts_split(X, y, test_ratio=0.2)
    st.write(f"Train: **{len(Xtr):,}** &nbsp;&nbsp; Test: **{len(Xte):,}**")

    st.subheader("4) Cross-Validation (Train)")
    with st.spinner("Walk-forward CV…"):
        cv_acc = walkforward_cv_acc(Xtr, ytr, n_splits=5, random_state=42)
    st.metric("CV Accuracy (train)", f"{cv_acc:.3f}")

    st.subheader("5) Final Model & Test Metrics")
    with st.spinner("Fitting final model…"):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", get_model(42))])
        pipe.fit(Xtr, ytr)
        proba_te = pipe.predict_proba(Xte)[:, 1]
        cls = evaluate(yte, proba_te, threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{cls['accuracy']:.3f}")
    c2.metric("Precision", f"{cls['precision']:.3f}")
    c3.metric("Recall", f"{cls['recall']:.3f}")
    c4.metric("ROC-AUC", f"{cls['roc_auc']:.3f}")

    st.write("**Confusion Matrix**:", cls["confusion_matrix"])
    with st.expander("Full classification report"):
        st.code(cls["classification_report"])

    st.subheader("6) Backtest (Test Window)")
    with st.spinner("Simulating strategy…"):
        bt = backtest(close_valid.loc[Xte.index], pd.Series(proba_te, index=Xte.index),
                      threshold=threshold, plot_file="equity_curve.png")

    m1, m2, m3 = st.columns(3)
    m1.metric("CAGR (Strategy)", f"{bt['cagr_strategy']:.2%}")
    m2.metric("CAGR (Buy&Hold)", f"{bt['cagr_buyhold']:.2%}")
    m3.metric("Sharpe (Strategy)", f"{bt['sharpe_strategy']:.3f}")

    n1, n2 = st.columns(2)
    n1.metric("Max Drawdown (Strategy)", f"{bt['maxdd_strategy']:.2%}")
    n2.metric("Max Drawdown (Buy&Hold)", f"{bt['maxdd_buyhold']:.2%}")

    st.image("equity_curve.png", caption="Equity Curve (Strategy vs Buy&Hold)", use_column_width=True)

    # Next-day decision preview (last test day -> next day)
    last_idx = Xte.index[-1]
    proba_last = float(proba_te[-1])
    decision = int(proba_last >= threshold)
    st.info(
        #f"**Next-day signal (from last test day {last_idx.date()}):**  "
        f"P(up) = **{proba_last:.3f}** → **{'BUY (1)' if decision else 'HOLD (0)'}** "
        f"(threshold {threshold:.2f})"
    )

    # Download buttons
    pred_df = pd.DataFrame({
        "date": Xte.index,
        "close": close_valid.loc[Xte.index].values,
        "proba_up": proba_te,
        "signal_next_day": (proba_te >= threshold).astype(int)
    })
    st.download_button("⬇ Download predictions (CSV)", pred_df.to_csv(index=False), "predictions_test.csv", "text/csv")
    with open("equity_curve.png", "rb") as f:
        st.download_button("⬇ Download equity curve (PNG)", f.read(), "equity_curve.png", "image/png")

else:
    st.info("Set your ticker and press **Run Analysis**. For India via Yahoo, remember `.NS` (e.g., RELIANCE.NS).")
    st.caption("If online fetch fails due to network restrictions, use CSV upload or switch to Demo Mode.")