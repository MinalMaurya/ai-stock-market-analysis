from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable

def _as_series(df: pd.DataFrame | pd.Series, col: str) -> pd.Series:
    if isinstance(df, pd.Series):
        return df
    if col in df.columns and not isinstance(df[col], pd.DataFrame):
        return df[col].astype(float)
    if col in df.columns and isinstance(df[col], pd.DataFrame):
        return df[col].iloc[:, 0].astype(float)
    if isinstance(df.columns, pd.MultiIndex):
        for c in df.columns:
            if (isinstance(c, tuple) and c[0] == col) or (c == col):
                s = df[c]
                return s.astype(float) if isinstance(s, pd.Series) else s.iloc[:, 0].astype(float)
    raise KeyError(f"Column '{col}' not found; got {df.columns}")

def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, window: int=14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(window).mean()
    dn = (-d.clip(upper=0)).rolling(window).mean()
    rs = up / (dn + 1e-12)
    return 100 - (100/(1+rs))

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, w: int=14) -> pd.Series:
    pc = close.shift(1)
    tr = np.maximum(high-low, np.maximum((high-pc).abs(), (low-pc).abs()))
    return tr.rolling(w).mean()

def add_indicators(df: pd.DataFrame,
                   rsi_window: int,
                   sma_windows: Iterable[int],
                   ema_fast: int,
                   ema_slow: int,
                   macd_signal: int,
                   bb_window: int) -> pd.DataFrame:
    out = df.copy()
    close = _as_series(out, "Close")
    high  = _as_series(out, "High")
    low   = _as_series(out, "Low")
    vol   = _as_series(out, "Volume")

    out["ret_1"] = close.pct_change(fill_method=None)
    out["ret_5"] = close.pct_change(5, fill_method=None)
    out["log_ret"] = np.log1p(out["ret_1"])
    out["ret_vol_5"] = out["log_ret"].rolling(5).std()
    out["ret_vol_20"] = out["log_ret"].rolling(20).std()

    for w in sma_windows:
        sma = close.rolling(w).mean()
        out[f"sma_{w}"] = sma
        out[f"prc_over_sma_{w}"] = close / (sma + 1e-12)

    out["ema_fast"] = _ema(close, ema_fast)
    out["ema_slow"] = _ema(close, ema_slow)
    out["macd"] = out["ema_fast"] - out["ema_slow"]
    out["macd_signal"] = _ema(out["macd"], macd_signal)
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    out["rsi"] = _rsi(close, rsi_window)

    mid = close.rolling(bb_window).mean()
    std = close.rolling(bb_window).std()
    up, dn = mid + 2*std, mid - 2*std
    out["bb_percB"] = (close - dn) / (up - dn + 1e-12)

    ll = low.rolling(14).min()
    hh = high.rolling(14).max()
    out["stoch_k"] = 100 * (close - ll) / (hh - ll + 1e-12)

    out["vol_z"] = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-12)
    out["atr_14"] = _atr(high, low, close, 14)
    return out

FEATURE_LIST = [
    "ret_1","ret_5","log_ret","ret_vol_5","ret_vol_20",
    "prc_over_sma_5","prc_over_sma_10","prc_over_sma_20",
    "ema_fast","ema_slow","macd","macd_signal","macd_hist",
    "rsi","bb_percB","stoch_k","vol_z","atr_14"
]

def make_supervised(frame: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    out = frame.copy()
    close = _as_series(out, "Close")
    out["future_close"] = close.shift(-horizon_days)
    out["target_up"] = (out["future_close"] > close).astype(int)
    return out