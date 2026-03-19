import os, time, warnings
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import psutil
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
warnings.filterwarnings('ignore', category=UserWarning)
RANDOM_SEED = 42
POLICY_RATIO = 0.3

def load_ihdp(path='ihdp_data.csv'):
    df = pd.read_csv(path)
    t = df['treatment'].astype(int).values
    yf = df['y_factual'].astype(float).values
    ycf = df['y_cfactual'].astype(float).values
    y0 = np.where(t == 1, ycf, yf).astype(float)
    y1 = np.where(t == 1, yf, ycf).astype(float)
    df = df.copy()
    df['t'] = t
    df['y0_true'] = y0
    df['y1_true'] = y1
    if 'x1' in df.columns:
        df['Z'] = (df['x1'] > df['x1'].median()).astype(int)
    else:
        df['Z'] = 0
    return df

def train_test_split_df(df: pd.DataFrame, test_ratio: float=0.3, seed: int=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = max(1, int(round(test_ratio * len(df))))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return (df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True))

def _feature_evidence_binary(row):
    ev = {}
    for i in range(25):
        key = f'x{i + 1}'
        v = row.get(key, 0)
        try:
            vv = float(v)
        except Exception:
            vv = 0.0
        ev[f'X{i}'] = 1 if vv >= 1 else 0
    return ev

def _feature_vector_raw(row):
    return np.array([float(row.get(f'x{i}', 0.0)) for i in range(1, 26)], dtype=float)

def _fit_outcome_scaler(train_df: pd.DataFrame):
    vals = train_df['y_factual'].astype(float).values
    mu = float(np.mean(vals))
    sd = float(np.std(vals) + 1e-06)
    return {'mu': mu, 'sd': sd}

def _continuous_to_soft_label(y: float, scaler: Dict[str, float]):
    z = (float(y) - scaler['mu']) / max(scaler['sd'], 1e-06)
    p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
    return np.array([1.0 - p1, p1], dtype=float)

def _build_cam_training_rows(train_df: pd.DataFrame, scaler: Dict[str, float], max_rows: int=2000):
    rows = []
    for _, r in train_df.head(max_rows).iterrows():
        evidence = _feature_evidence_binary(r)
        rows.append({'T': int(r['t']), 'y': _continuous_to_soft_label(float(r['y_factual']), scaler), 'evidence': evidence})
    return rows

def fit_linear_calibration(train_df, method, ctx, sample_cap=2000):
    coef = {}
    for t in (0, 1):
        sub = train_df[train_df['t'] == t]
        if len(sub) > sample_cap:
            sub = sub.sample(sample_cap, random_state=RANDOM_SEED)
        P, Y = ([], [])
        for _, row in sub.iterrows():
            val, _us, mode = infer_value(method, row, do_t=t, ctx=ctx)
            if mode != 'prob':
                continue
            P.append(val)
            Y.append(float(row['y_factual']))
        if len(P) < 2:
            coef[t] = (1.0, 0.0)
        else:
            P = np.asarray(P)
            Y = np.asarray(Y)
            A = np.vstack([P, np.ones_like(P)]).T
            a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
            coef[t] = (float(a), float(b))
    return coef

def _get_vcg_model(ctx):
    from vcg import get_model
    if '_vcg' not in ctx:
        ctx['_vcg'] = get_model(seed=RANDOM_SEED)
    return ctx['_vcg']

def _get_vcgf_model(ctx):
    from vcg_forest import get_model
    if '_vcgf' not in ctx:
        ctx['_vcgf'] = get_model(seed=RANDOM_SEED)
    return ctx['_vcgf']

def _ensure_trained_baselines(ctx):
    if ctx.get('_trained_baselines'):
        return
    import t_learner as TL
    import psm_iptw as PI
    import dr_aipw as DR
    TL.fit_ihdp(ctx['_train_df'], random_state=RANDOM_SEED)
    PI.fit_ihdp(ctx['_train_df'], random_state=RANDOM_SEED)
    DR.fit_ihdp(ctx['_train_df'], random_state=RANDOM_SEED)
    ctx['_trained_baselines'] = True

def infer_value(method: str, row: pd.Series, do_t: int, ctx=None):
    ctx = {} if ctx is None else ctx
    if method == 'CAM':
        from caxm import get_cam, cam_infer, cam_fit
        cam = ctx.get('_cam')
        if cam is None:
            cam = get_cam(n=25, card=2, seed=RANDOM_SEED)
            tr_rows = _build_cam_training_rows(ctx['_train_df'], ctx['_outcome_scaler'])
            cam_fit(cam, tr_rows, smoothing=1.0, max_rows=2000)
            ctx['_cam'] = cam
        evidence = _feature_evidence_binary(row)
        prob_vec, elapsed_us = cam_infer(cam, target='Y', do_dict={'T': int(do_t)}, evidence=evidence)
        return (float(np.asarray(prob_vec).ravel()[-1]), float(elapsed_us), 'prob')
    if method == 'CAM++':
        from caxmpp import get_cam, cam_infer, campp_fit
        cam = ctx.get('_campp')
        if cam is None:
            cam = get_cam(n=25, card=2, r=16, seed=RANDOM_SEED)
            tr_rows = _build_cam_training_rows(ctx['_train_df'], ctx['_outcome_scaler'])
            campp_fit(cam, tr_rows, epochs=3, lr=0.005, max_rows=2000)
            ctx['_campp'] = cam
        evidence = _feature_evidence_binary(row)
        prob_vec, elapsed_us = cam_infer(cam, target='Y', do_dict={'T': int(do_t)}, evidence=evidence)
        return (float(np.asarray(prob_vec).ravel()[-1]), float(elapsed_us), 'prob')
    if method == 'VCG-Forest++':
        from vcg_forest import infer
        model = _get_vcgf_model(ctx)
        evidence = _feature_evidence_binary(row)
        prob_vec, elapsed_us = infer(model, target='Y', do_dict={'T': int(do_t)}, evidence=evidence)
        return (float(np.asarray(prob_vec).ravel()[-1]), float(elapsed_us), 'prob')
    if method == 'VCG':
        from vcg import infer_y_from_features
        model = _get_vcg_model(ctx)
        evidence = _feature_evidence_binary(row)
        eps_seed = int(abs(hash(tuple((int(evidence[f'X{i}']) for i in range(25))))) % (2 ** 31 - 1))
        p, elapsed_us = infer_y_from_features(model, evidence, do_t=int(do_t), eps_seed=eps_seed)
        return (float(p), float(elapsed_us), 'prob')
    if method == 'KleinVC':
        from klein_vc import get_klein_vc, klein_infer
        if '_kvc' not in ctx:
            ctx['_kvc'] = get_klein_vc(n=32, card=2)
        prob_vec, elapsed_us = klein_infer(ctx['_kvc'], target='Y', do_dict={'T': int(do_t)}, evidence={'Z': int(row.get('Z', 0))})
        return (float(np.asarray(prob_vec).ravel()[-1]), float(elapsed_us), 'prob')
    if method == 'MobiusVC':
        if ctx.get('_mobius') is None:
            from mobius_vc import get_mobius_vc_model
            ctx['_mobius'] = get_mobius_vc_model(cache_capacity=128, low_rank_dim=2, seed=RANDOM_SEED)
        prob_vec, elapsed_us, _ = ctx['_mobius'].query(target=1, intervention={0: float(do_t)})
        return (float(np.asarray(prob_vec).ravel()[-1]), float(elapsed_us), 'prob')
    _ensure_trained_baselines(ctx)
    x = _feature_vector_raw(row)
    if method == 'T-Learner':
        import t_learner as M
        yhat, us = M.infer_y_value(x, int(do_t))
        return (float(yhat), float(us), 'continuous')
    if method == 'PSM-IPTW':
        import psm_iptw as M
        yhat, us = M.infer_y_value(x, int(do_t))
        return (float(yhat), float(us), 'continuous')
    if method == 'DR-AIPW':
        import dr_aipw as M
        yhat, us = M.infer_y_value(x, int(do_t))
        return (float(yhat), float(us), 'continuous')
    raise ValueError(f'Unknown method: {method}')

def infer_tau_special(method: str, row: pd.Series, ctx=None):
    if method != 'DR-AIPW':
        return None
    ctx = {} if ctx is None else ctx
    _ensure_trained_baselines(ctx)
    import dr_aipw as M
    x = _feature_vector_raw(row)
    tau, us = M.infer_tau_dr(x, int(row['t']), float(row['y_factual']))
    return (float(tau), float(us))

def evaluate_causal(df_eval, method, train_df, policy_ratio=POLICY_RATIO, ctx=None):
    ctx = {} if ctx is None else ctx
    ctx['_train_df'] = train_df
    ctx.setdefault('_outcome_scaler', _fit_outcome_scaler(train_df))
    print(f'\nEvaluating {method}...')
    calib = fit_linear_calibration(train_df, method, ctx) if method in {'CAM', 'CAM++', 'VCG-Forest++', 'VCG'} else None
    tau_hat_list, tau_true_list = ([], [])
    y1_hat_list, y0_hat_list = ([], [])
    time_single_list, time_pair_list = ([], [])
    for _, row in df_eval.iterrows():
        special = infer_tau_special(method, row, ctx)
        v1, us1, mode1 = infer_value(method, row, 1, ctx)
        v0, us0, mode0 = infer_value(method, row, 0, ctx)
        time_single_list.extend([us1, us0])
        time_pair_list.append(us1 + us0)
        if mode1 == 'prob' and calib is not None:
            a1, b1 = calib.get(1, (1.0, 0.0))
            a0, b0 = calib.get(0, (1.0, 0.0))
            y1_hat = a1 * v1 + b1
            y0_hat = a0 * v0 + b0
        else:
            y1_hat, y0_hat = (v1, v0)
        y1_hat_list.append(float(y1_hat))
        y0_hat_list.append(float(y0_hat))
        if special is not None:
            tau_hat = special[0]
        else:
            tau_hat = float(y1_hat - y0_hat)
        tau_hat_list.append(float(tau_hat))
        tau_true_list.append(float(row['y1_true'] - row['y0_true']))
    tau_hat = np.array(tau_hat_list)
    tau_true = np.array(tau_true_list)
    ate_hat = float(np.mean(tau_hat))
    ate_true = float(np.mean(tau_true))
    ate_abs_err = float(abs(ate_hat - ate_true))
    cate_mae = float(np.mean(np.abs(tau_hat - tau_true)))
    pehe = float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))
    n = len(df_eval)
    k = max(1, int(round(policy_ratio * n)))
    idx = np.argsort(-tau_hat)
    treat_mask = np.zeros(n, dtype=bool)
    treat_mask[idx[:k]] = True
    y_policy = np.where(treat_mask, df_eval['y1_true'].values, df_eval['y0_true'].values)
    policy_value = float(np.mean(y_policy))
    infer_us_single = float(np.median(time_single_list))
    infer_us_pair = float(np.median(time_pair_list))
    avg_mem_mb = float(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    return dict(ATE_hat=ate_hat, ATE_true=ate_true, ATE_abs_err=ate_abs_err, CATE_MAE=cate_mae, PEHE=pehe, PolicyValue=policy_value, Infer_us_single=infer_us_single, Infer_us_pair=infer_us_pair, Mem_MB=avg_mem_mb)

def main(data_path='ihdp_data.csv'):
    df = load_ihdp(data_path)
    train_df, test_df = train_test_split_df(df)
    ctx = {'_outcome_scaler': _fit_outcome_scaler(train_df)}
    model_list = ['CAM', 'CAM++', 'VCG-Forest++', 'VCG', 'DR-AIPW', 'PSM-IPTW', 'T-Learner']
    out = {}
    for m in model_list:
        try:
            out[m] = evaluate_causal(test_df, m, train_df=train_df, ctx=ctx)
        except Exception as e:
            out[m] = {'error': str(e)}
    df_out = pd.DataFrame.from_dict(out, orient='index')
    print(df_out.round(6))
    df_out.to_csv('ihdp_metrics.csv', encoding='utf-8-sig')
    return df_out
if __name__ == '__main__':
    main()
