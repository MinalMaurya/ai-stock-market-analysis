from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _to_series(x):
    if isinstance(x, pd.Series): return x
    if isinstance(x, pd.DataFrame): return x.iloc[:, 0]
    raise TypeError("Expected pandas Series/DataFrame")

def backtest(close, proba, threshold=0.5, plot_file="equity_curve.png") -> dict:
    close = _to_series(close)
    proba = _to_series(proba)

    signal = (proba >= threshold).astype(int).shift(1).reindex(close.index).fillna(0)
    rets = close.pct_change().fillna(0)
    strat = rets * signal

    def cagr(r):
        cum = (1+r).prod(); years = len(r)/252
        return cum**(1/years)-1 if years>0 else float("nan")

    def sharpe(r): return np.sqrt(252) * (r.mean() / (r.std()+1e-12))
    def maxdd(c): return (c / c.cummax() - 1).min()

    eq = (1+strat).cumprod()
    bh = (1+rets).cumprod()

    plt.figure(figsize=(9,4))
    eq.plot(label="Strategy"); bh.plot(label="Buy & Hold")
    plt.title("Equity Curve (Test)"); plt.legend(); plt.tight_layout()
    plt.savefig(plot_file, dpi=150); plt.close()

    return {
        "cagr_strategy": cagr(strat),
        "cagr_buyhold": cagr(rets),
        "sharpe_strategy": sharpe(strat),
        "maxdd_strategy": maxdd(eq),
        "maxdd_buyhold": maxdd(bh),
        "equity_curve_last": float(eq.iloc[-1]),
        "buyhold_curve_last": float(bh.iloc[-1])
    }