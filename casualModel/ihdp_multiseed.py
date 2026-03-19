import importlib.util, time, os, math
from pathlib import Path
import numpy as np
import pandas as pd
BASE_PATH = Path(__file__).resolve().with_name('ihdp_run.py')
DATA_PATH = str(Path(__file__).resolve().with_name('ihdp_data.csv'))
OUT_RAW = str(Path(__file__).resolve().with_name('ihdp_multiseed_raw.csv'))
OUT_SUMMARY = str(Path(__file__).resolve().with_name('ihdp_multiseed_summary.csv'))
OUT_PAPER = str(Path(__file__).resolve().with_name('ihdp_multiseed_paper.csv'))

def load_base():
    spec = importlib.util.spec_from_file_location('ihdp_base', str(BASE_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def fmt(mean, std, digits=4):
    return f'{mean:.{digits}f} ± {std:.{digits}f}'

def metrics_from_preds(df_eval, tau_hat):
    tau_true = df_eval['y1_true'].to_numpy(dtype=float) - df_eval['y0_true'].to_numpy(dtype=float)
    tau_hat = np.asarray(tau_hat, dtype=float)
    ate_hat = float(np.mean(tau_hat))
    ate_true = float(np.mean(tau_true))
    ate_abs_err = float(abs(ate_hat - ate_true))
    cate_mae = float(np.mean(np.abs(tau_hat - tau_true)))
    pehe = float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))
    n = len(df_eval)
    k = max(1, int(round(0.3 * n)))
    idx = np.argsort(-tau_hat)
    treat_mask = np.zeros(n, dtype=bool)
    treat_mask[idx[:k]] = True
    y_policy = np.where(treat_mask, df_eval['y1_true'].to_numpy(dtype=float), df_eval['y0_true'].to_numpy(dtype=float))
    policy_value = float(np.mean(y_policy))
    return dict(ATE_hat=ate_hat, ATE_true=ate_true, ATE_abs_err=ate_abs_err, CATE_MAE=cate_mae, PEHE=pehe, PolicyValue=policy_value)

def feature_matrix(df):
    return df[[f'x{i}' for i in range(1, 26)]].to_numpy(dtype=float)

def eval_custom(base, method, train_df, test_df, seed):
    base.RANDOM_SEED = seed
    ctx = {'_outcome_scaler': base._fit_outcome_scaler(train_df), '_train_df': train_df}
    calib = base.fit_linear_calibration(train_df, method, ctx) if method in {'CAM', 'CAM++', 'VCG-Forest++', 'VCG', 'KleinVC', 'MobiusVC'} else None
    tau_hat = []
    for _, row in test_df.iterrows():
        special = base.infer_tau_special(method, row, ctx)
        v1, _, mode1 = base.infer_value(method, row, 1, ctx)
        v0, _, mode0 = base.infer_value(method, row, 0, ctx)
        if mode1 == 'prob' and calib is not None:
            a1, b1 = calib.get(1, (1.0, 0.0))
            a0, b0 = calib.get(0, (1.0, 0.0))
            y1_hat = a1 * v1 + b1
            y0_hat = a0 * v0 + b0
        else:
            y1_hat, y0_hat = (v1, v0)
        tau_hat.append(float(special[0] if special is not None else y1_hat - y0_hat))
    return metrics_from_preds(test_df, tau_hat)

def eval_t_learner(train_df, test_df, seed):
    import t_learner as M
    M.fit_ihdp(train_df, random_state=seed)
    X = feature_matrix(test_df)
    y0 = M._model['mu0'].predict(X)
    y1 = M._model['mu1'].predict(X)
    tau_hat = y1 - y0
    return metrics_from_preds(test_df, tau_hat)

def eval_psm_iptw(train_df, test_df, seed):
    import psm_iptw as M
    M.fit_ihdp(train_df, random_state=seed)
    X = feature_matrix(test_df)
    y0 = M._model['mu0'].predict(X)
    y1 = M._model['mu1'].predict(X)
    tau_hat = y1 - y0
    return metrics_from_preds(test_df, tau_hat)

def eval_dr_aipw(train_df, test_df, seed):
    import dr_aipw as M
    M.fit_ihdp(train_df, random_state=seed)
    X = feature_matrix(test_df)
    t = test_df['t'].to_numpy(dtype=int)
    yf = test_df['y_factual'].to_numpy(dtype=float)
    prop = M._model['prop']
    e = np.clip(prop.predict_proba(X)[:, 1], 0.001, 1 - 0.001)
    mu0 = M._model['mu0'].predict(X)
    mu1 = M._model['mu1'].predict(X)
    tau_hat = mu1 - mu0 + t * (yf - mu1) / e - (1 - t) * (yf - mu0) / (1 - e)
    return metrics_from_preds(test_df, tau_hat)

def main(seeds=None):
    if seeds is None:
        seeds = list(range(10))
    base = load_base()
    df_all = base.load_ihdp(DATA_PATH)
    rows = []
    methods = ['CAM', 'CAM++', 'VCG-Forest++', 'VCG', 'DR-AIPW', 'PSM-IPTW', 'T-Learner']
    for seed in seeds:
        t_seed = time.time()
        train_df, test_df = base.train_test_split_df(df_all, seed=seed)
        for method in methods:
            t0 = time.time()
            try:
                if method in {'CAM', 'CAM++', 'VCG-Forest++', 'VCG'}:
                    res = eval_custom(base, method, train_df, test_df, seed)
                elif method == 'DR-AIPW':
                    res = eval_dr_aipw(train_df, test_df, seed)
                elif method == 'PSM-IPTW':
                    res = eval_psm_iptw(train_df, test_df, seed)
                elif method == 'T-Learner':
                    res = eval_t_learner(train_df, test_df, seed)
                row = {'seed': seed, 'model': method, 'elapsed_sec_model': time.time() - t0}
                row.update(res)
                print(f"[seed {seed}] {method} done in {row['elapsed_sec_model']:.2f}s", flush=True)
            except Exception as e:
                row = {'seed': seed, 'model': method, 'error': str(e), 'elapsed_sec_model': time.time() - t0}
                print(f'[seed {seed}] {method} ERROR: {e}', flush=True)
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT_RAW, index=False, encoding='utf-8-sig')
        print(f'== seed {seed} finished in {time.time() - t_seed:.2f}s ==', flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT_RAW, index=False, encoding='utf-8-sig')
    ok = raw[raw.get('error').isna()].copy() if 'error' in raw.columns else raw.copy()
    metric_cols = ['ATE_hat', 'ATE_true', 'ATE_abs_err', 'CATE_MAE', 'PEHE', 'PolicyValue', 'elapsed_sec_model']
    summary = ok.groupby('model', as_index=False)[metric_cols].agg(['mean', 'std'])
    summary.columns = ['model' if c[0] == 'model' else f'{c[0]}_{c[1]}' for c in summary.columns.to_flat_index()]
    summary = summary.sort_values('PEHE_mean', ascending=True).reset_index(drop=True)
    paper = pd.DataFrame({'Model': summary['model']})
    for col in ['ATE_abs_err', 'CATE_MAE', 'PEHE', 'PolicyValue']:
        paper[col] = [fmt(m, s) for m, s in zip(summary[f'{col}_mean'], summary[f'{col}_std'])]
    summary.to_csv(OUT_SUMMARY, index=False, encoding='utf-8-sig')
    paper.to_csv(OUT_PAPER, index=False, encoding='utf-8-sig')
    print('\nPAPER TABLE\n', paper.to_string(index=False), flush=True)
if __name__ == '__main__':
    main()
