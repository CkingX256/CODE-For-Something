from __future__ import annotations
import os, argparse, importlib.util, sys, warnings, math, random
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Sequence, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
ACIC_TREAT_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
BASE_DOSE = 0.0
DEFAULT_MODELS = ['CAM', 'CAM++', 'VCG', 'VCG-Forest++', 'T-Learner', 'X-Learner', 'PSM-IPTW']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _read_csv_any(path: str) -> pd.DataFrame:
    for enc in (None, 'utf-8-sig', 'gbk'):
        try:
            return pd.read_csv(path) if enc is None else pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise FileNotFoundError(path)

def _resolve_w_cols(df: pd.DataFrame):
    mapping, missing = ({}, [])
    cols = list(df.columns)
    for w in ACIC_TREAT_LEVELS:
        sw = str(w)
        got = None
        for c in (sw, f'{sw}_x', f'{sw}_y', w):
            if c in cols:
                got = c
                break
        if got is None:
            for c in cols:
                try:
                    if abs(float(str(c).replace('_x', '').replace('_y', '')) - w) < 1e-12:
                        got = c
                        break
                except Exception:
                    pass
        if got is None:
            missing.append(w)
        else:
            mapping[w] = got
    return (mapping, missing)

def load_acic_split_and_merge(training_path: str, testing_path: str, predictions_path: Optional[str]=None):
    df_tr_raw = _read_csv_any(training_path)
    df_te_raw = _read_csv_any(testing_path)
    df_pr = _read_csv_any(predictions_path) if predictions_path and os.path.exists(predictions_path) else None
    if 'unitID' not in df_te_raw.columns or 'step' not in df_te_raw.columns:
        raise ValueError('testing_sample.csv must contain unitID and step')
    if 'weekID' not in df_tr_raw.columns:
        raise ValueError('training_sample.csv must contain weekID')
    numeric_covs = [c for c in ['X1', 'X2', 'X3'] if c in df_tr_raw.columns]
    cat_covs = [c for c in ['C1', 'C2', 'C3'] if c in df_tr_raw.columns]
    covs = numeric_covs + cat_covs
    train_cov = df_tr_raw[['unitID', 'weekID'] + covs].drop_duplicates(subset=['unitID', 'weekID'])
    df_te = df_te_raw.merge(train_cov, left_on=['unitID', 'step'], right_on=['unitID', 'weekID'], how='left', validate='m:1')
    for c in numeric_covs:
        if df_te[c].isna().any():
            df_te[c] = df_te[c].fillna(df_tr_raw[c].median())
    for c in cat_covs:
        if df_te[c].isna().any():
            mode = df_tr_raw[c].mode().iloc[0] if not df_tr_raw[c].mode().empty else '__UNK__'
            df_te[c] = df_te[c].fillna(mode)
    mapping, missing = _resolve_w_cols(df_te)
    if missing:
        if df_pr is None:
            raise ValueError(f'testing_sample is missing outcome columns {missing} and no predictions.csv was provided')
        mapping_pr, missing_pr = _resolve_w_cols(df_pr)
        if missing_pr:
            raise ValueError(f'predictions.csv is also missing outcome columns {missing_pr}')
        pr_small = df_pr[['unitID', 'step'] + [mapping_pr[w] for w in ACIC_TREAT_LEVELS]].drop_duplicates(['unitID', 'step'])
        df_te = df_te.merge(pr_small, on=['unitID', 'step'], how='left', validate='m:1', suffixes=('', '_pr'))
        for w in ACIC_TREAT_LEVELS:
            col = mapping_pr[w]
            if col in df_te.columns:
                df_te.rename(columns={col: w}, inplace=True)
            elif f'{col}_pr' in df_te.columns:
                df_te.rename(columns={f'{col}_pr': w}, inplace=True)
    else:
        df_te.rename(columns={mapping[w]: w for w in ACIC_TREAT_LEVELS}, inplace=True)
    return (df_tr_raw, df_te, numeric_covs, cat_covs)

class CovariateEncoder:

    def __init__(self, numeric_cols: Sequence[str], cat_cols: Sequence[str]):
        self.numeric_cols = list(numeric_cols)
        self.cat_cols = list(cat_cols)
        self.num_imp = SimpleImputer(strategy='median') if self.numeric_cols else None
        self.cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]) if self.cat_cols else None

    def fit(self, df: pd.DataFrame):
        if self.numeric_cols:
            self.num_imp.fit(df[self.numeric_cols])
        if self.cat_cols:
            self.cat_pipe.fit(df[self.cat_cols])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        mats = []
        if self.numeric_cols:
            mats.append(np.asarray(self.num_imp.transform(df[self.numeric_cols]), float))
        if self.cat_cols:
            mats.append(np.asarray(self.cat_pipe.transform(df[self.cat_cols]), float))
        return np.concatenate(mats, axis=1) if mats else np.zeros((len(df), 0), float)

    def row_to_evidence(self, row: pd.Series) -> Dict[str, Any]:
        out = {}
        idx = 0
        for c in self.numeric_cols:
            out[f'X{idx}'] = float(row[c]) if c in row and pd.notna(row[c]) else 0.0
            idx += 1
        for c in self.cat_cols:
            out[f'X{idx}'] = str(row[c]) if c in row and pd.notna(row[c]) else '__UNK__'
            idx += 1
        return out

def _safe_nanmean(arr):
    arr = np.asarray(arr, float)
    m = np.isfinite(arr)
    return float(np.mean(arr[m])) if m.any() else np.nan

def _kendall_like_tau_row(y_true_row, y_pred_row):
    yt = np.asarray(y_true_row, float)
    yp = np.asarray(y_pred_row, float)
    c = d = 0
    K = len(yt)
    for i in range(K):
        for j in range(i + 1, K):
            a = yt[j] - yt[i]
            b = yp[j] - yp[i]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            s = a * b
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    den = c + d
    return np.nan if den == 0 else float((c - d) / den)

def acic_metrics(y_pred_df: pd.DataFrame, y_true_df: pd.DataFrame) -> Dict[str, float]:
    Yp = y_pred_df[ACIC_TREAT_LEVELS].to_numpy(dtype=float)
    Yt = y_true_df[ACIC_TREAT_LEVELS].to_numpy(dtype=float)
    diff = Yp - Yt
    rmse = float(np.sqrt(_safe_nanmean(diff ** 2)))
    mae = float(_safe_nanmean(np.abs(diff)))
    ss_res = float(np.nansum(diff ** 2))
    ss_tot = float(np.nansum((Yt - np.nanmean(Yt)) ** 2))
    r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    se_list = []
    for i in range(len(ACIC_TREAT_LEVELS) - 1):
        for j in range(i + 1, len(ACIC_TREAT_LEVELS)):
            d_ = (Yp[:, j] - Yp[:, i] - (Yt[:, j] - Yt[:, i])) ** 2
            se_list.append(d_[np.isfinite(d_)])
    mpehe = float(np.sqrt(_safe_nanmean(np.concatenate(se_list)))) if se_list else np.nan
    ate_list = []
    for i in range(len(ACIC_TREAT_LEVELS) - 1):
        for j in range(i + 1, len(ACIC_TREAT_LEVELS)):
            ate_list.append(np.nanmean(Yp[:, j] - Yp[:, i] - (Yt[:, j] - Yt[:, i])))
    ate_mae = float(np.nanmean(np.abs(ate_list))) if ate_list else np.nan
    ate_maxae = float(np.nanmax(np.abs(ate_list))) if ate_list else np.nan
    accs, regrets, taus = ([], [], [])
    for i in range(Yt.shape[0]):
        yt_row, yp_row = (Yt[i], Yp[i])
        if np.count_nonzero(np.isfinite(yt_row) & np.isfinite(yp_row)) < 2:
            taus.append(np.nan)
            continue
        j_true = int(np.nanargmax(yt_row))
        j_pred = int(np.nanargmax(yp_row))
        accs.append(1.0 if j_true == j_pred else 0.0)
        regrets.append(float(yt_row[j_true] - yt_row[j_pred]))
        taus.append(_kendall_like_tau_row(yt_row, yp_row))
    return {'ACIC_RMSE': rmse, 'ACIC_MAE': mae, 'ACIC_R2': r2, 'mPEHE': mpehe, 'ATE_MAE': ate_mae, 'ATE_MAXAE': ate_maxae, 'Policy_Acc': float(np.nanmean(accs)) if accs else np.nan, 'Policy_Regret': float(np.nanmean(regrets)) if regrets else np.nan, 'RankTau': float(np.nanmean(taus)) if taus else np.nan}

def _load_module(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def _make_bins(y: np.ndarray, n_bins: int=12):
    y = np.asarray(y, float)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(y, qs)
    edges = np.unique(edges)
    if len(edges) <= 2:
        lo, hi = (float(np.min(y)), float(np.max(y) + 1e-06))
        edges = np.linspace(lo, hi, max(3, n_bins + 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return (edges, centers)

def _bin_onehot(y: float, edges: np.ndarray, centers: np.ndarray) -> np.ndarray:
    idx = int(np.clip(np.searchsorted(edges[1:-1], float(y), side='right'), 0, len(centers) - 1))
    out = np.zeros(len(centers), float)
    out[idx] = 1.0
    return out

def _expected_from_prob(prob: np.ndarray, centers: np.ndarray) -> float:
    p = np.asarray(prob, float).reshape(-1)
    p = p / max(p.sum(), 1e-12)
    return float(np.dot(p, centers))

def _hgb_reg(seed: int):
    return RandomForestRegressor(n_estimators=50, max_depth=12, min_samples_leaf=20, random_state=seed, n_jobs=-1)

def _stratified_cap(df: pd.DataFrame, per_dose_max: int, seed: int) -> pd.DataFrame:
    parts = []
    for i, w in enumerate(ACIC_TREAT_LEVELS):
        sub = df[np.isclose(df['treatment'], w)]
        if len(sub) > per_dose_max:
            sub = sub.sample(n=per_dose_max, random_state=seed + i, replace=False)
        parts.append(sub)
    return pd.concat(parts, axis=0).reset_index(drop=True) if parts else df

class BaseAdapter:

    def fit(self, train_df: pd.DataFrame):
        ...

    def predict_curve(self, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        ...

class TLearnerDoseAdapter(BaseAdapter):

    def __init__(self, encoder: CovariateEncoder, seed: int=42, name: str='T-Learner'):
        self.encoder = encoder
        self.seed = seed
        self.name = name
        self.models = {}

    def fit(self, train_df: pd.DataFrame):
        train_df = _stratified_cap(train_df, per_dose_max=30000, seed=self.seed)
        X = self.encoder.transform(train_df)
        t = train_df['treatment'].astype(float).to_numpy()
        y = train_df['outcome'].astype(float).to_numpy()
        for i, w in enumerate(ACIC_TREAT_LEVELS):
            mask = np.isclose(t, w)
            m = _hgb_reg(self.seed + i)
            m.fit(X[mask], y[mask])
            self.models[w] = m
        return self

    def predict_curve(self, test_df: pd.DataFrame):
        X = self.encoder.transform(test_df)
        t0 = perf_counter()
        out = {'unitID': test_df['unitID'].values, 'step': test_df['step'].values}
        for w in ACIC_TREAT_LEVELS:
            out[w] = self.models[w].predict(X)
        return (pd.DataFrame(out), (perf_counter() - t0) * 1000000.0 / max(len(test_df), 1))

class PSMIPTWDoseAdapter(TLearnerDoseAdapter):

    def __init__(self, encoder: CovariateEncoder, seed: int=42):
        super().__init__(encoder, seed, 'PSM-IPTW')
        self.prop = None

    def fit(self, train_df: pd.DataFrame):
        train_df = _stratified_cap(train_df, per_dose_max=30000, seed=self.seed)
        X = self.encoder.transform(train_df)
        t = train_df['treatment'].astype(float).to_numpy()
        y = train_df['outcome'].astype(float).to_numpy()
        t_idx = np.array([ACIC_TREAT_LEVELS.index(float(v)) for v in t], int)
        self.prop = LogisticRegression(max_iter=1000, random_state=self.seed, n_jobs=None)
        self.prop.fit(X, t_idx)
        P = np.clip(self.prop.predict_proba(X), 0.001, 1.0)
        for i, w in enumerate(ACIC_TREAT_LEVELS):
            wi = ACIC_TREAT_LEVELS.index(w)
            mask = t_idx == wi
            sw = 1.0 / P[:, wi]
            m = _hgb_reg(self.seed + i)
            m.fit(X[mask], y[mask], sample_weight=sw[mask])
            self.models[w] = m
        return self

class XLearnerDoseAdapter(BaseAdapter):

    def __init__(self, encoder: CovariateEncoder, seed: int=42):
        self.encoder = encoder
        self.seed = seed
        self.name = 'X-Learner'
        self.mu0 = None
        self.pair = {}

    def fit(self, train_df: pd.DataFrame):
        train_df = _stratified_cap(train_df, per_dose_max=25000, seed=self.seed)
        X_all = self.encoder.transform(train_df)
        t_all = train_df['treatment'].astype(float).to_numpy()
        y_all = train_df['outcome'].astype(float).to_numpy()
        self.mu0 = _hgb_reg(self.seed)
        mask0 = np.isclose(t_all, BASE_DOSE)
        self.mu0.fit(X_all[mask0], y_all[mask0])
        for i, w in enumerate(ACIC_TREAT_LEVELS):
            if np.isclose(w, BASE_DOSE):
                continue
            mask = np.isclose(t_all, BASE_DOSE) | np.isclose(t_all, w)
            X = X_all[mask]
            y = y_all[mask]
            t_pair = np.isclose(t_all[mask], w).astype(int)
            mu0 = _hgb_reg(self.seed + 10 + i)
            mu1 = _hgb_reg(self.seed + 20 + i)
            mu0.fit(X[t_pair == 0], y[t_pair == 0])
            mu1.fit(X[t_pair == 1], y[t_pair == 1])
            D1 = y[t_pair == 1] - mu0.predict(X[t_pair == 1])
            D0 = mu1.predict(X[t_pair == 0]) - y[t_pair == 0]
            tau1 = _hgb_reg(self.seed + 30 + i)
            tau0 = _hgb_reg(self.seed + 40 + i)
            tau1.fit(X[t_pair == 1], D1)
            tau0.fit(X[t_pair == 0], D0)
            prop = LogisticRegression(max_iter=1000, random_state=self.seed + 50 + i)
            prop.fit(X, t_pair)
            self.pair[w] = dict(tau0=tau0, tau1=tau1, prop=prop)
        return self

    def predict_curve(self, test_df: pd.DataFrame):
        X = self.encoder.transform(test_df)
        t0 = perf_counter()
        y0 = self.mu0.predict(X)
        out = {'unitID': test_df['unitID'].values, 'step': test_df['step'].values, 0.0: y0}
        for w in ACIC_TREAT_LEVELS:
            if np.isclose(w, BASE_DOSE):
                continue
            P = np.clip(self.pair[w]['prop'].predict_proba(X)[:, 1], 0.001, 1 - 0.001)
            tau = (1.0 - P) * self.pair[w]['tau1'].predict(X) + P * self.pair[w]['tau0'].predict(X)
            out[w] = y0 + tau
        return (pd.DataFrame(out), (perf_counter() - t0) * 1000000.0 / max(len(test_df), 1))

class CAMDoseAdapter(BaseAdapter):

    def __init__(self, encoder: CovariateEncoder, seed: int=42, path: str=None, n_atoms: int=96, n_bins: int=12):
        self.encoder = encoder
        self.seed = seed
        self.name = 'CAM'
        self.path = path or os.path.join(BASE_DIR, 'caxm.py')
        self.n_atoms = n_atoms
        self.n_bins = n_bins
        self.mod = None
        self.base = None
        self.pair = {}
        self.edges = None
        self.centers = None

    def fit(self, train_df: pd.DataFrame):
        self.mod = _load_module(self.path, f'cam_mod_{self.seed}')
        y = train_df['outcome'].astype(float).to_numpy()
        self.edges, self.centers = _make_bins(y, self.n_bins)
        base_rows = []
        sub0 = train_df[np.isclose(train_df['treatment'], BASE_DOSE)].copy()
        if len(sub0) > 3000:
            sub0 = sub0.sample(n=3000, random_state=self.seed)
        self.base = self.mod.get_cam(n=self.n_atoms, card=len(self.centers), seed=self.seed)
        for _, r in sub0.iterrows():
            base_rows.append({'X': 0, 'y': _bin_onehot(float(r['outcome']), self.edges, self.centers), 'evidence': self.encoder.row_to_evidence(r)})
        self.mod.cam_fit(self.base, base_rows, smoothing=1.0)
        for i, w in enumerate(ACIC_TREAT_LEVELS):
            if np.isclose(w, BASE_DOSE):
                continue
            sub = train_df[np.isclose(train_df['treatment'], BASE_DOSE) | np.isclose(train_df['treatment'], w)].copy()
            if len(sub) > 5000:
                sub = sub.sample(n=5000, random_state=self.seed + i)
            model = self.mod.get_cam(n=self.n_atoms, card=len(self.centers), seed=self.seed + i + 1)
            rows = []
            for _, r in sub.iterrows():
                x = 1 if abs(float(r['treatment']) - float(w)) < 1e-12 else 0
                rows.append({'X': x, 'y': _bin_onehot(float(r['outcome']), self.edges, self.centers), 'evidence': self.encoder.row_to_evidence(r)})
            self.mod.cam_fit(model, rows, smoothing=1.0)
            self.pair[w] = model
        return self

    def predict_curve(self, test_df: pd.DataFrame):
        t0 = perf_counter()
        out = {'unitID': test_df['unitID'].values, 'step': test_df['step'].values}
        y0 = []
        ys = {w: [] for w in self.pair.keys()}
        for _, r in test_df.iterrows():
            ev = self.encoder.row_to_evidence(r)
            p0, _ = self.mod.cam_infer(self.base, 'Y', {'X': 0}, ev)
            y0.append(_expected_from_prob(p0, self.centers))
            for w, model in self.pair.items():
                p1, _ = self.mod.cam_infer(model, 'Y', {'X': 1}, ev)
                ys[w].append(_expected_from_prob(p1, self.centers))
        out[0.0] = np.asarray(y0, float)
        for w, v in ys.items():
            out[w] = np.asarray(v, float)
        return (pd.DataFrame(out), (perf_counter() - t0) * 1000000.0 / max(len(test_df), 1))

class CAMPPDoseAdapter(CAMDoseAdapter):

    def __init__(self, encoder: CovariateEncoder, seed: int=42, path: str=None, n_atoms: int=48, n_bins: int=10):
        super().__init__(encoder, seed, path or os.path.join(BASE_DIR, 'caxmpp.py'), n_atoms, n_bins)
        self.name = 'CAM++'

    def fit(self, train_df: pd.DataFrame):
        self.mod = _load_module(self.path, f'campp_mod_{self.seed}')
        y = train_df['outcome'].astype(float).to_numpy()
        self.edges, self.centers = _make_bins(y, self.n_bins)
        sub0 = train_df[np.isclose(train_df['treatment'], BASE_DOSE)].copy()
        if len(sub0) > 10000:
            sub0 = sub0.sample(n=10000, random_state=self.seed)
        self.base = self.mod.get_cam(n=self.n_atoms, card=len(self.centers), r=16, d=16, seed=self.seed)
        rows = []
        for _, r in sub0.iterrows():
            rows.append({'T': 0, 'y': _bin_onehot(float(r['outcome']), self.edges, self.centers), 'evidence': self.encoder.row_to_evidence(r)})
        self.mod.campp_fit(self.base, rows, lr=0.005, epochs=1, seed=self.seed)
        for i, w in enumerate(ACIC_TREAT_LEVELS):
            if np.isclose(w, BASE_DOSE):
                continue
            sub = train_df[np.isclose(train_df['treatment'], BASE_DOSE) | np.isclose(train_df['treatment'], w)].copy()
            if len(sub) > 15000:
                sub = sub.sample(n=15000, random_state=self.seed + i)
            model = self.mod.get_cam(n=self.n_atoms, card=len(self.centers), r=16, d=16, seed=self.seed + i + 1)
            rows = []
            for _, r in sub.iterrows():
                x = 1 if abs(float(r['treatment']) - float(w)) < 1e-12 else 0
                rows.append({'T': x, 'y': _bin_onehot(float(r['outcome']), self.edges, self.centers), 'evidence': self.encoder.row_to_evidence(r)})
            self.mod.campp_fit(model, rows, lr=0.005, epochs=1, seed=self.seed + i + 1)
            self.pair[w] = model
        return self

    def predict_curve(self, test_df: pd.DataFrame):
        t0 = perf_counter()
        out = {'unitID': test_df['unitID'].values, 'step': test_df['step'].values}
        y0 = []
        ys = {w: [] for w in self.pair.keys()}
        for _, r in test_df.iterrows():
            ev = self.encoder.row_to_evidence(r)
            p0, _ = self.mod.cam_infer(self.base, 'Y', {'T': 0}, ev)
            y0.append(_expected_from_prob(p0, self.centers))
            for w, model in self.pair.items():
                p1, _ = self.mod.cam_infer(model, 'Y', {'T': 1}, ev)
                ys[w].append(_expected_from_prob(p1, self.centers))
        out[0.0] = np.asarray(y0, float)
        for w, v in ys.items():
            out[w] = np.asarray(v, float)
        return (pd.DataFrame(out), (perf_counter() - t0) * 1000000.0 / max(len(test_df), 1))

class VCGDoseAdapter(BaseAdapter):

    def __init__(self, encoder: CovariateEncoder, seed: int=42):
        self.encoder = encoder
        self.seed = seed
        self.name = 'VCG'
        self.model = None

    def fit(self, train_df: pd.DataFrame):
        train_df = _stratified_cap(train_df, per_dose_max=30000, seed=self.seed)
        X = self.encoder.transform(train_df)
        t = train_df['treatment'].astype(float).to_numpy().reshape(-1, 1)
        y = train_df['outcome'].astype(float).to_numpy()
        Z = np.concatenate([X, t], axis=1)
        self.model = _hgb_reg(self.seed)
        self.model.fit(Z, y)
        return self

    def predict_curve(self, test_df: pd.DataFrame):
        X = self.encoder.transform(test_df)
        t0 = perf_counter()
        out = {'unitID': test_df['unitID'].values, 'step': test_df['step'].values}
        for w in ACIC_TREAT_LEVELS:
            Z = np.concatenate([X, np.full((len(X), 1), w, float)], axis=1)
            out[w] = self.model.predict(Z)
        return (pd.DataFrame(out), (perf_counter() - t0) * 1000000.0 / max(len(test_df), 1))

class VCGForestDoseAdapter(BaseAdapter):

    def __init__(self, encoder: CovariateEncoder, seed: int=42):
        self.encoder = encoder
        self.seed = seed
        self.name = 'VCG-Forest++'
        self.blocks = []
        self.models = []
        self.weights = []

    def fit(self, train_df: pd.DataFrame):
        train_df = _stratified_cap(train_df, per_dose_max=20000, seed=self.seed)
        X = self.encoder.transform(train_df)
        t = train_df['treatment'].astype(float).to_numpy().reshape(-1, 1)
        y = train_df['outcome'].astype(float).to_numpy()
        nfeat = X.shape[1]
        idx_all = np.arange(nfeat)
        idx_num = np.arange(len(self.encoder.numeric_cols))
        idx_cat = np.arange(len(self.encoder.numeric_cols), nfeat)
        mid = nfeat // 2
        candidate_blocks = [idx_all, idx_num, idx_cat, np.arange(0, mid), np.arange(mid, nfeat)]
        self.blocks = [b for b in candidate_blocks if len(b) > 0]
        Xtr, Xva, ttr, tva, ytr, yva = train_test_split(X, t, y, test_size=0.15, random_state=self.seed)
        rmses = []
        for i, block in enumerate(self.blocks):
            Ztr = np.concatenate([Xtr[:, block], ttr], axis=1)
            Zva = np.concatenate([Xva[:, block], tva], axis=1)
            m = _hgb_reg(self.seed + i)
            m.fit(Ztr, ytr)
            pred = m.predict(Zva)
            rmse = math.sqrt(mean_squared_error(yva, pred))
            self.models.append(m)
            rmses.append(rmse)
        inv = 1.0 / np.clip(np.asarray(rmses, float), 1e-06, None)
        self.weights = (inv / inv.sum()).tolist()
        return self

    def predict_curve(self, test_df: pd.DataFrame):
        X = self.encoder.transform(test_df)
        t0 = perf_counter()
        out = {'unitID': test_df['unitID'].values, 'step': test_df['step'].values}
        for w in ACIC_TREAT_LEVELS:
            preds = []
            for block, model, wt in zip(self.blocks, self.models, self.weights):
                Z = np.concatenate([X[:, block], np.full((len(X), 1), w, float)], axis=1)
                preds.append(wt * model.predict(Z))
            out[w] = np.sum(preds, axis=0)
        return (pd.DataFrame(out), (perf_counter() - t0) * 1000000.0 / max(len(test_df), 1))

def build_adapter(name: str, encoder: CovariateEncoder, seed: int):
    if name == 'CAM':
        return CAMDoseAdapter(encoder, seed)
    if name == 'CAM++':
        return CAMPPDoseAdapter(encoder, seed)
    if name == 'VCG':
        return VCGDoseAdapter(encoder, seed)
    if name == 'VCG-Forest++':
        return VCGForestDoseAdapter(encoder, seed)
    if name == 'T-Learner':
        return TLearnerDoseAdapter(encoder, seed)
    if name == 'X-Learner':
        return XLearnerDoseAdapter(encoder, seed)
    if name == 'PSM-IPTW':
        return PSMIPTWDoseAdapter(encoder, seed)
    raise ValueError(name)

def preflight_models(models, encoder, train_df, test_df):
    probe_train = []
    for w in ACIC_TREAT_LEVELS:
        sub = train_df[np.isclose(train_df['treatment'], w)]
        n = min(300, len(sub))
        if n:
            probe_train.append(sub.sample(n=n, random_state=123))
    probe_train = pd.concat(probe_train, axis=0).reset_index(drop=True)
    probe_test = test_df.head(8).copy()
    ok, msgs = ({}, {})
    for m in models:
        try:
            a = build_adapter(m, encoder, seed=42)
            a.fit(probe_train)
            pred, us = a.predict_curve(probe_test)
            arr = pred[ACIC_TREAT_LEVELS].to_numpy(float)
            if not np.isfinite(arr).all():
                raise RuntimeError('model output contains NaN/Inf')
            ok[m] = True
            msgs[m] = f'ok, infer_us_pair≈{us:.3f}'
        except Exception as e:
            ok[m] = False
            msgs[m] = str(e)
    return (ok, msgs, [m for m, v in ok.items() if not v])

def main(training_path=None, testing_path=None, predictions_path=None, models=None, sample_n=None, seed=42):
    tr = training_path or os.path.join(BASE_DIR, 'training_sample.csv')
    te = testing_path or os.path.join(BASE_DIR, 'testing_sample.csv')
    pr = predictions_path or os.path.join(BASE_DIR, 'predictions.csv')
    df_tr, df_te, num_cols, cat_cols = load_acic_split_and_merge(tr, te, pr)
    encoder = CovariateEncoder(num_cols, cat_cols).fit(df_tr)
    model_list = models or DEFAULT_MODELS
    ok, msgs, bad = preflight_models(model_list, encoder, df_tr, df_te)
    print('[precheck]')
    for m in model_list:
        print(f'  - {m}: {msgs[m]}')
    if bad:
        raise RuntimeError(f'precheck failed: {bad}')
    if sample_n is not None and int(sample_n) < len(df_te):
        df_eval = df_te.sample(n=int(sample_n), random_state=seed, replace=False).copy()
    else:
        df_eval = df_te.copy()
    rows = {}
    for i, m in enumerate(model_list):
        print(f'\n[running] {m} ...', flush=True)
        adapter = build_adapter(m, encoder, seed=seed + 100 * i)
        adapter.fit(df_tr)
        pred, us = adapter.predict_curve(df_eval)
        metrics = acic_metrics(pred, df_eval[['unitID', 'step'] + ACIC_TREAT_LEVELS])
        metrics['Infer_us_pair'] = float(us)
        rows[m] = metrics
        print(pd.Series(metrics).round(6).to_string())
    out = pd.DataFrame.from_dict(rows, orient='index')
    out_path = os.path.join(BASE_DIR, 'acic_metrics.csv')
    out.to_csv(out_path, encoding='utf-8-sig')
    print(f'\nsaved -> {out_path}')
    print(out.round(6).to_string())
    return out
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', type=str, default=','.join(DEFAULT_MODELS))
    ap.add_argument('--sample-n', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    models = [x.strip() for x in args.models.split(',') if x.strip()]
    main(models=models, sample_n=args.sample_n, seed=args.seed)
