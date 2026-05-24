from __future__ import annotations
from datetime import datetime
import time
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from pathlib import Path
import logging
# add near the top of data.py
try:
    from nsepy import get_history as nse_get_history
    HAS_NSEPY = True
except Exception:
    HAS_NSEPY = False
try:
    yf.utils.get_yf_logger().setLevel(logging.CRITICAL)
except Exception:
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
# ---- helpers ----
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)
    # Ensure required columns exist
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            if c == "Volume":
                df[c] = 0
            else:
                return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()

def _valid(df: pd.DataFrame) -> bool:
    return isinstance(df, pd.DataFrame) and (not df.empty) and ("Close" in df.columns)

# ---- sources ----
def _download_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                     progress=False, threads=False, interval="1d")
    if not _valid(df):
        # second attempt via history()
        df = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    df = _normalize_cols(df)
    if _valid(df):
        df.attrs["source"] = f"yahoo:{ticker}"
    return df

def _download_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    tkr = ticker.replace(".NS", "")  # Stooq uses no .NS suffix
    df = pdr.DataReader(tkr, "stooq", start, end)
    df = _normalize_cols(df)
    if _valid(df):
        df.attrs["source"] = f"stooq:{tkr}"
    return df

# ---- public API ----
def download_ohlcv(ticker: str, start_date: str, end_date: str | None, source: str = "auto") -> pd.DataFrame:
    """
    source='yahoo' | 'stooq' | 'nsepy' | 'auto'
    """
    ...
    if source == "nsepy":
        df = _download_nsepy(ticker, start_date, end_date)
        if _valid(df): return df
        raise ValueError(f"NSEPy failed for {ticker}")
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    # explicit source
    if source == "yahoo":
        df = _download_yahoo(ticker, start_date, end_date)
        if not _valid(df) and not ticker.endswith(".NS"):
            df = _download_yahoo(f"{ticker}.NS", start_date, end_date)
        if _valid(df): return df
        raise ValueError(f"Yahoo failed for {ticker}")

    if source == "stooq":
        df = _download_stooq(ticker, start_date, end_date)
        if _valid(df): return df
        raise ValueError(f"Stooq failed for {ticker}")
        # ---- AUTO fallback order ----
    # 1) Stooq (no .NS)
    try:
        df = _download_stooq(ticker, start_date, end_date)
        if _valid(df): 
            return df
    except Exception:
        pass
    time.sleep(0.3)

    # 2) Yahoo with given symbol
    try:
        df = _download_yahoo(ticker, start_date, end_date)
        if _valid(df): 
            return df
    except Exception:
        pass
    time.sleep(0.3)

    # 3) Yahoo with .NS (common for India tickers)
    if not ticker.endswith(".NS"):
        try:
            df = _download_yahoo(f"{ticker}.NS", start_date, end_date)
            if _valid(df): 
                return df
        except Exception:
            pass
        time.sleep(0.3)

    # 4) NSEPy (direct from NSE; works with symbols without .NS)
    try:
        if HAS_NSEPY:
            # try as-is
            df = _download_nsepy(ticker, start_date, end_date)
            if _valid(df):
                return df
            # if user passed RELIANCE.NS, strip suffix for nsepy
            if ticker.endswith(".NS"):
                df = _download_nsepy(ticker.replace(".NS", ""), start_date, end_date)
                if _valid(df):
                    return df
    except Exception:
        pass

    raise ValueError(f"No data for '{ticker}' from Stooq/Yahoo/NSEPy right now.")

def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path,skiprows=[1])
    print(df.head())
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df = df.rename(columns=str.title)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            if c == "Volume": df[c] = 0
            else: raise ValueError(f"CSV missing {c}")
    df = df.sort_index()
    df.attrs["source"] = "csv"
    return df

def demo_data(n=800, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # generate daily returns and prices
    rets = rng.normal(0.0004, 0.02, n)
    close = (1 + pd.Series(rets)).cumprod().to_numpy() * 100.0

    # make OHLC arrays (no index alignment issues)
    high = close * (1 + np.abs(rng.normal(0.001, 0.01, n)))
    low  = close * (1 - np.abs(rng.normal(0.001, 0.01, n)))
    openp = np.concatenate([[close[0]], close[:-1]])  # yesterday’s close as today’s open
    vol = (rng.gamma(5., 1., n) * 1e6).astype(int)

    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {"Open": openp, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx
    )
    df.attrs["source"] = "demo"
    return df
def _download_nsepy(ticker: str, start: str, end: str) -> pd.DataFrame:
    if not HAS_NSEPY:
        raise RuntimeError("nsepy not installed")
    s = pd.to_datetime(start).date()
    e = pd.to_datetime(end).date()
    # NSEPy uses symbols without .NS
    df = nse_get_history(symbol=ticker.replace(".NS", ""), start=s, end=e)
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = _normalize_cols(df)
    if _valid(df):
        df.attrs["source"] = f"nsepy:{ticker.replace('.NS','')}"
    return df