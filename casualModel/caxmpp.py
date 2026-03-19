from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import time
import hashlib
import numpy as np
__version__ = 'refined_v3_fixed'
ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...]]

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))

def _softmax(z: np.ndarray, axis: int=-1) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

def _normalize_prob(v: np.ndarray, eps: float=1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    s = float(v.sum())
    if not np.isfinite(s) or s <= eps:
        return np.full_like(v, 1.0 / max(1, v.size), dtype=float)
    v = np.clip(v / s, 0.0, 1.0)
    v = v / max(float(v.sum()), eps)
    return v

def _hash_key(evidence: Optional[Dict[str, Any]]) -> int:
    if not evidence:
        return 0
    items = sorted(((str(k), str(v)) for k, v in evidence.items()))
    s = '|'.join([f'{k}={v}' for k, v in items])
    return int(hashlib.sha1(s.encode('utf-8')).hexdigest()[:8], 16) % (2 ** 31 - 1)

def _evidence_to_likelihood(n: int, evidence: Optional[Dict[str, Any]], floor: float=0.15) -> np.ndarray:
    if not evidence:
        return np.ones(n, dtype=float)
    l = np.ones(n, dtype=float)
    idx = np.arange(n)
    for k, v in evidence.items():
        key = str(k)
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v)
            if arr.dtype == bool and arr.size == n:
                l *= arr.astype(float)
                continue
            if arr.size == n:
                l *= np.clip(arr.astype(float).reshape(-1), 0.0, 1.0)
                continue
        if key.upper().startswith('Z') and isinstance(v, (int, np.integer, float)) and (float(v) in (0.0, 1.0)):
            parity = int(float(v))
            l *= (idx % 2 == parity).astype(float)
            continue
        h = int(hashlib.sha1(f'{key}={v}'.encode('utf-8')).hexdigest()[:8], 16) % 10000 / 10000.0
        l *= 0.4 + 0.6 * h
    floor = float(np.clip(floor, 0.0, 0.5))
    l = floor + (1.0 - floor) * np.clip(l, 0.0, 1.0)
    return l

def _evidence_to_vec(d: int, evidence: Optional[Dict[str, Any]], seed: int=0) -> np.ndarray:
    if not evidence:
        return np.zeros(d, dtype=float)
    for k in ('ctx', 'context'):
        if k in evidence:
            v = np.asarray(evidence[k], dtype=float).reshape(-1)
            if v.size == d:
                return v
    vec = np.zeros(d, dtype=float)
    items = sorted(((str(k), str(v)) for k, v in evidence.items()))
    for k, v in items:
        if k in ('ctx', 'context'):
            continue
        h = int(hashlib.sha1(f'{k}={v}'.encode('utf-8')).hexdigest()[:8], 16) % (10 ** 9 + 7)
        rng_i = np.random.default_rng(h ^ seed * 1315423911)
        vec += rng_i.standard_normal(d).astype(float) * 0.25
    mu = float(vec.mean())
    sd = float(vec.std() + 1e-06)
    return (vec - mu) / sd

@dataclass
class CAMPP:
    n: int
    card: int
    d: int
    r: int
    p: np.ndarray
    E: np.ndarray
    U: np.ndarray
    V: np.ndarray
    b: np.ndarray
    W1: np.ndarray
    W2: np.ndarray
    gate_w: np.ndarray
    gate_b: float = 0.0
    do_emb: np.ndarray = None
    evidence_floor: float = 0.15
    temperature: float = 1.0
    cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

def get_cam(n: int=64, card: int=2, r: Optional[int]=None, d: Optional[int]=None, seed: Optional[int]=None) -> CAMPP:
    if n <= 0:
        raise ValueError('n must be positive')
    if card <= 1:
        raise ValueError('card must be >= 2')
    rng = np.random.default_rng(seed)
    d = int(32 if d is None else d)
    r = int(8 if r is None else r)
    p = _normalize_prob(rng.dirichlet(np.ones(n, dtype=float)))
    E = rng.standard_normal((n, d)).astype(float) / np.sqrt(d)
    U = rng.standard_normal((2, card, r)).astype(float) / np.sqrt(r)
    V = rng.standard_normal((2, r, d)).astype(float) / np.sqrt(d)
    b = np.zeros((2, card), dtype=float)
    W1 = rng.standard_normal((d, d)).astype(float) / np.sqrt(d)
    W2 = rng.standard_normal((d, d)).astype(float) / np.sqrt(d)
    gate_w = rng.standard_normal(d).astype(float) / np.sqrt(d)
    do_emb = rng.standard_normal(d).astype(float) / np.sqrt(d)
    return CAMPP(n=n, card=card, d=d, r=r, p=p, E=E, U=U, V=V, b=b, W1=W1, W2=W2, gate_w=gate_w, gate_b=0.0, do_emb=do_emb, evidence_floor=0.15, temperature=1.0)

def _forward_logits(model: CAMPP, do_t: int, context: np.ndarray) -> np.ndarray:
    h = context.astype(float)
    h = h + np.tanh(model.W1 @ h)
    h = h + np.tanh(model.W2 @ h)
    if model.do_emb is not None:
        h = h + (float(do_t) - 0.5) * model.do_emb
    gate = float(_sigmoid(model.gate_w @ h + model.gate_b + 0.8 * (do_t - 0.5)))
    h_eff = h * gate
    X = model.E * h_eff[None, :]
    Z = (model.V[do_t] @ X.T).T
    logits = (model.U[do_t] @ Z.T).T + model.b[do_t][None, :]
    return logits

def cam_infer(model: CAMPP, target: Union[str, int]='Y', do_dict: Optional[Dict[str, Any]]=None, evidence: Optional[Dict[str, Any]]=None) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter_ns()
    if isinstance(target, str):
        if target.strip().upper() != 'Y':
            raise ValueError(f"CAM++ only supports target='Y', got {target!r}")
    elif isinstance(target, (int, np.integer)):
        if int(target) != 0:
            raise ValueError(f'CAM++ only supports target index 0, got {target}')
    else:
        raise ValueError(f'Unsupported target type: {type(target)}')
    do_dict = do_dict or {}
    t = do_dict.get('T', do_dict.get('X', 0))
    do_t = int(float(t) >= 0.5)
    ehash = _hash_key(evidence)
    key = (do_t, ehash)
    if key in model.cache:
        mix = model.cache[key][1]
        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        return (mix.copy().astype(float), float(elapsed_us))
    l = _evidence_to_likelihood(model.n, evidence, floor=model.evidence_floor)
    w = _normalize_prob(model.p * l)
    ctx = _evidence_to_vec(model.d, evidence, seed=0)
    logits = _forward_logits(model, do_t, ctx)
    atom_prob = _softmax(logits, axis=1)
    mix = np.clip(w @ atom_prob, 1e-12, 1.0)
    mix = mix / mix.sum()
    if model.temperature != 1.0:
        mix = _softmax(np.log(mix) / max(model.temperature, 1e-06))
    model.cache[key] = (w.astype(float), mix.astype(float))
    elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
    return (mix.astype(float), float(elapsed_us))

def update_prior(model: CAMPP, evidence: Optional[Dict[str, Any]], lr: float=1.0, floor: float=1e-06) -> CAMPP:
    lr = float(lr)
    if not 0.0 < lr <= 1.0:
        raise ValueError('lr must be in (0,1]')
    l = _evidence_to_likelihood(model.n, evidence, floor=model.evidence_floor)
    p_post = _normalize_prob(model.p * l)
    p_new = _normalize_prob((1.0 - lr) * model.p + lr * p_post)
    p_new = np.clip(p_new, float(floor), 1.0)
    model.p = _normalize_prob(p_new)
    model.cache.clear()
    return model

def kl_divergence(p: ArrayLike, q: ArrayLike, eps: float=1e-12) -> float:
    p = _normalize_prob(np.asarray(p, dtype=float).reshape(-1), eps=eps)
    q = _normalize_prob(np.asarray(q, dtype=float).reshape(-1), eps=eps)
    return float(np.sum(p * (np.log(np.clip(p, eps, 1.0)) - np.log(np.clip(q, eps, 1.0)))))

def _scalar_to_soft_class(value: float) -> int:
    return int(float(value) >= 0.0)

def _row_to_evidence_dict(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ev = row.get('evidence', None)
    if ev is not None:
        return ev
    feat = {k: v for k, v in row.items() if isinstance(k, str) and (k.startswith('X') or k.startswith('Z') or k in ('ctx', 'context'))}
    return feat or None

def campp_fit(model: CAMPP, data: Iterable[Dict[str, Any]], lr: float=0.01, epochs: int=1, max_rows: Optional[int]=None, l2: float=0.0001, seed: int=0, ridge: Optional[float]=None, **_: Any) -> CAMPP:
    """Lightweight SGD fit with backwards-compatible kwargs."""
    rng = np.random.default_rng(seed)
    rows = list(data)
    if max_rows is not None:
        rows = rows[:int(max_rows)]
    if not rows:
        return model
    lr = float(lr)
    l2 = float(l2 if ridge is None else ridge)
    for _ in range(int(epochs)):
        rng.shuffle(rows)
        for row in rows:
            t = row.get('T', row.get('X', 0))
            do_t = int(float(t) >= 0.5)
            y = row.get('y', row.get('Y', None))
            if y is None:
                if do_t == 1 and 'y1_true' in row:
                    y = row['y1_true']
                elif do_t == 0 and 'y0_true' in row:
                    y = row['y0_true']
                else:
                    continue
            if isinstance(y, (list, tuple, np.ndarray)):
                y_arr = np.asarray(y, dtype=float).reshape(-1)
                if y_arr.size == model.card:
                    yi = int(np.argmax(y_arr))
                else:
                    yi = _scalar_to_soft_class(float(y_arr[0]))
            elif isinstance(y, (int, np.integer)) and 0 <= int(y) < model.card:
                yi = int(y)
            else:
                yi = _scalar_to_soft_class(float(y))
            evidence = _row_to_evidence_dict(row)
            l = _evidence_to_likelihood(model.n, evidence, floor=model.evidence_floor)
            w = _normalize_prob(model.p * l)
            ctx = _evidence_to_vec(model.d, evidence, seed=0)
            logits = _forward_logits(model, do_t, ctx)
            atom_prob = _softmax(logits, axis=1)
            grad_logits = atom_prob.copy()
            grad_logits[:, yi] -= 1.0
            grad_logits *= w[:, None]
            h = ctx.astype(float)
            h = h + np.tanh(model.W1 @ h)
            h = h + np.tanh(model.W2 @ h)
            if model.do_emb is not None:
                h = h + (float(do_t) - 0.5) * model.do_emb
            gate = float(_sigmoid(model.gate_w @ h + model.gate_b + 0.8 * (do_t - 0.5)))
            h_eff = h * gate
            X = model.E * h_eff[None, :]
            Z = (model.V[do_t] @ X.T).T
            dU = grad_logits.T @ Z
            db = grad_logits.sum(axis=0)
            dZ = grad_logits @ model.U[do_t]
            dV = dZ.T @ X
            dU += l2 * model.U[do_t]
            dV += l2 * model.V[do_t]
            db += l2 * model.b[do_t]
            model.U[do_t] -= lr * dU
            model.V[do_t] -= lr * dV
            model.b[do_t] -= lr * db
    model.cache.clear()
    return model

def cam_time_and_accuracy(n_trials: int=500, seed: int=0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    model = get_cam(n=256, card=2, seed=seed)
    us_list: List[float] = []
    evid0 = {'Z0': 0}
    evid1 = {'Z0': 1}
    p00, _ = cam_infer(model, 'Y', {'T': 0}, evid0)
    p10, _ = cam_infer(model, 'Y', {'T': 1}, evid0)
    do_sensitive = float(np.abs(p10[1] - p00[1]) > 1e-09)
    p01, _ = cam_infer(model, 'Y', {'T': 0}, evid1)
    evid_sensitive = float(np.abs(p01[1] - p00[1]) > 1e-09)
    for _ in range(int(n_trials)):
        t = int(rng.random() > 0.5)
        evid = {'Z0': int(rng.random() > 0.5), 'u': float(rng.random())}
        _, us = cam_infer(model, 'Y', {'T': t}, evid)
        us_list.append(us)
    return {'avg_infer_us': float(np.mean(us_list)) if us_list else 0.0, 'do_sensitive': do_sensitive, 'evidence_sensitive': evid_sensitive}
__all__ = ['CAMPP', 'get_cam', 'cam_infer', 'campp_fit', 'update_prior', 'kl_divergence', 'cam_time_and_accuracy']
