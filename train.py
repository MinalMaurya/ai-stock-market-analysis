from __future__ import annotations
import argparse
import json
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# local imports
from src.data import download_ohlcv, load_csv, demo_data
from src.features import add_indicators, make_supervised, FEATURE_LIST
from src.modeling import get_model, ts_split, walkforward_cv_acc, evaluate
from src.backtest import backtest


# ---------- Config Loader ----------
def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)


# ---------- Main Script ----------
def main():
    parser = argparse.ArgumentParser(description="AiStock: Next-Day Trend Predictor")
    parser.add_argument("--ticker", type=str, help="Override ticker (e.g., AAPL, RELIANCE.NS)")
    parser.add_argument("--csv", type=str, help="If online fetch fails, use a local CSV file")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "yahoo", "stooq"],
                        help="Data source: yahoo, stooq, or auto (default: auto)")
    args = parser.parse_args()

    cfg = load_config()
    if args.ticker:
        cfg["ticker"] = args.ticker

    px = None  # initialize to avoid NameError

    print(f"1/6 Fetching data for {cfg['ticker']} …")
    try:
        px = download_ohlcv(cfg["ticker"], cfg["start_date"], cfg["end_date"], source=args.source)
        print(f"Source: {px.attrs.get('source', 'unknown')}")
        print(f"Data rows: {len(px):,} | Range: {px.index.min().date()} → {px.index.max().date()}")
    except Exception as e:
        if args.csv:
            print(f"   Online fetch failed ({e}). Loading CSV: {args.csv}")
            px = load_csv(args.csv)
            print(f"Source: {px.attrs.get('source', 'csv')}")
            print(f"Data rows: {len(px):,} | Range: {px.index.min().date()} → {px.index.max().date()}")
        else:
            print(f"   Online fetch failed ({e}). Using synthetic demo data.")
            px = demo_data()
            print(f"Source: {px.attrs.get('source', 'demo')}")
            print(f"Data rows: {len(px):,} | Range: {px.index.min().date()} → {px.index.max().date()}")

    # hard stop if no data
    if px is None or px.empty:
        print("No data available after all attempts. Please try a CSV or a different source.")
        sys.exit(1)

    # ---------- Build Features ----------
    print("2/6 Building indicators …")
    feat = add_indicators(
        px,
        rsi_window=cfg["rsi_window"],
        sma_windows=cfg["sma_windows"],
        ema_fast=cfg["ema_fast"],
        ema_slow=cfg["ema_slow"],
        macd_signal=cfg["macd_signal"],
        bb_window=cfg["bb_window"]
    )

    # ---------- Supervised Target ----------
    print("3/6 Making next-day target …")
    sup = make_supervised(feat, horizon_days=cfg["horizon_days"])

    # ---------- Prepare Dataset ----------
    print("4/6 Assembling dataset …")
    X = sup[[c for c in FEATURE_LIST if c in sup.columns]].copy()
    y = sup["target_up"].copy()

    # align indices and drop NaNs
    valid = X.dropna().index.intersection(y.dropna().index)
    X, y = X.loc[valid], y.loc[valid]
    close_valid = sup["Close"].reindex(valid)

    Xtr, Xte, ytr, yte = ts_split(X, y, cfg["test_ratio"])
    print(f"   Train: {len(Xtr):,}  |  Test: {len(Xte):,}")

    if len(Xtr) == 0 or len(Xte) == 0:
        print("Not enough data to train/test. Please try a longer date range or different ticker.")
        sys.exit(1)

    # ---------- Walk-Forward CV ----------
    print("5/6 Time-series CV on train …")
    cv_acc = walkforward_cv_acc(Xtr, ytr, cfg["n_splits_cv"], cfg["random_state"])
    print(f"   CV Accuracy (mean): {cv_acc:.4f}")

    # ---------- Final Model ----------
    print("6/6 Train final model & evaluate …")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", get_model(cfg["random_state"]))
    ])
    pipe.fit(Xtr, ytr)
    proba_te = pipe.predict_proba(Xte)[:, 1]

    # ---------- Classification Metrics ----------
    cls = evaluate(yte, proba_te, cfg["threshold"])
    print("\n=== TEST CLASSIFICATION ===")
    print(f"accuracy: {cls['accuracy']:.4f}")
    print(f"precision: {cls['precision']:.4f}")
    print(f"recall: {cls['recall']:.4f}")
    print(f"roc_auc: {cls['roc_auc']:.4f}")
    print("\nConfusion Matrix:", cls["confusion_matrix"])
    print("\nReport:\n", cls["classification_report"])

    # ---------- Backtest ----------
    bt = backtest(close_valid.loc[Xte.index], pd.Series(proba_te, index=Xte.index),
                  threshold=cfg["threshold"], plot_file=cfg["plot_file"])
    print("\n=== BACKTEST (TEST WINDOW) ===")
    for k, v in bt.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # ---------- Save Outputs ----------
    out = pd.DataFrame({
        "date": Xte.index,
        "close": close_valid.loc[Xte.index].values,
        "proba_up": proba_te,
        "signal_next_day": (proba_te >= cfg["threshold"]).astype(int)
    })
    out.to_csv("predictions_test.csv", index=False)
    print("\nSaved: predictions_test.csv, equity_curve.png")


if __name__ == "__main__":
    main()