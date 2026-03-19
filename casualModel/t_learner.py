from __future__ import annotations
import time
from typing import Iterable, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor
_model = None

def _get_X_from_df(df):
    cols = [f'x{i}' for i in range(1, 26)]
    return df[cols].to_numpy(dtype=float)

def fit_ihdp(train_df, random_state: int=42):
    global _model
    X = _get_X_from_df(train_df)
    t = train_df['t'].astype(int).to_numpy()
    y = train_df['y_factual'].astype(float).to_numpy()
    rf_params = dict(n_estimators=300, min_samples_leaf=5, random_state=random_state, n_jobs=1)
    mu0 = RandomForestRegressor(**rf_params)
    mu1 = RandomForestRegressor(**rf_params)
    mu0.fit(X[t == 0], y[t == 0])
    mu1.fit(X[t == 1], y[t == 1])
    _model = dict(mu0=mu0, mu1=mu1)
    return _model

def _require_model():
    if _model is None:
        raise RuntimeError('Call fit_ihdp(train_df) before inference.')
    return _model

def _to_x_array(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float).reshape(1, -1)
    if arr.shape[1] != 25:
        raise ValueError(f'Expected 25 features, got {arr.shape[1]}')
    return arr

def infer_y_value(x: Iterable[float], t: int) -> Tuple[float, float]:
    t0 = time.perf_counter_ns()
    M = _require_model()
    X = _to_x_array(x)
    if int(t) == 1:
        yhat = float(M['mu1'].predict(X)[0])
    else:
        yhat = float(M['mu0'].predict(X)[0])
    us = (time.perf_counter_ns() - t0) / 1000.0
    return (yhat, us)
