from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, classification_report)

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

def get_model(random_state: int):
    if HAS_XGB:
        return xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=random_state, n_jobs=-1
        )
    else:
        return GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            random_state=random_state
        )

def ts_split(X: pd.DataFrame, y: pd.Series, test_ratio: float) -> Tuple:
    n = len(X)
    n_test = int(math.ceil(n * test_ratio))
    cut = n - n_test
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

def walkforward_cv_acc(X: pd.DataFrame, y: pd.Series, n_splits: int, random_state: int) -> float:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr, va in tscv.split(X):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", get_model(random_state))])
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xva)[:, 1]
        preds = (proba >= 0.5).astype(int)
        scores.append(accuracy_score(yva, preds))
    return float(np.mean(scores))

def evaluate(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "roc_auc": (roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "classification_report": classification_report(y_true, preds, zero_division=0)
    }