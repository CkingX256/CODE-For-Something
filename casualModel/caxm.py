from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import time
import os
import hashlib
import numpy as np
__version__ = 'refined_v3_fixed'
ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...]]

def _normalize_prob(v: np.ndarray, eps: float=1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    s = float(v.sum())
    if not np.isfinite(s) or s <= eps:
        return np.full_like(v, 1.0 / max(1, v.size), dtype=float)
    v = np.clip(v / s, 0.0, 1.0)
    v = v / max(float(v.sum()), eps)
    return v

def _softmax(logits: np.ndarray, axis: int=-1) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

def _safe_log(p: np.ndarray, eps: float=1e-12) -> np.ndarray:
    return np.log(np.clip(p, eps, 1.0))

def _ensure_1d(v: ArrayLike, n: int) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.size != n:
        raise ValueError(f'Expected vector length {n}, got {v.size}')
    return v

@dataclass
class CAM:
    """Binary-treatment CAxM state with prior mass and potential-outcome distributions."""
    p: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    card: int = 2
    temperature: float = 1.0
    bias: Optional[np.ndarray] = None
    evidence_floor: float = 0.15
    params_path: Optional[str] = None

    @property
    def n(self) -> int:
        return int(self.p.size)

def _default_params_path() -> str:
    return os.environ.get('CAM_PARAMS_PATH', 'cam_params.npz')

def _load_cam_params(cam: CAM) -> None:
    if not cam.params_path:
        return
    path = cam.params_path
    if not os.path.exists(path):
        return
    try:
        data = np.load(path, allow_pickle=False)
        if 'p' in data and data['p'].shape == cam.p.shape:
            cam.p = _normalize_prob(data['p'])
        if 'y0' in data and data['y0'].shape == cam.y0.shape:
            cam.y0 = np.clip(data['y0'], 1e-08, 1.0)
            cam.y0 = cam.y0 / cam.y0.sum(axis=1, keepdims=True)
        if 'y1' in data and data['y1'].shape == cam.y1.shape:
            cam.y1 = np.clip(data['y1'], 1e-08, 1.0)
            cam.y1 = cam.y1 / cam.y1.sum(axis=1, keepdims=True)
        if 'temperature' in data:
            cam.temperature = float(data['temperature'])
        if 'bias' in data:
            b = np.asarray(data['bias'], dtype=float).reshape(-1)
            if b.size == cam.card:
                cam.bias = b
    except Exception:
        return

def _save_cam_params(cam: CAM) -> None:
    if not cam.params_path:
        return
    path = cam.params_path
    try:
        np.savez(path, p=np.asarray(cam.p, dtype=float), y0=np.asarray(cam.y0, dtype=float), y1=np.asarray(cam.y1, dtype=float), temperature=float(cam.temperature), bias=np.asarray(cam.bias, dtype=float) if cam.bias is not None else np.array([], dtype=float))
    except Exception:
        return

def _likelihood_from_evidence(cam: CAM, evidence: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    evidence -> per-atom likelihood vector.
    Supported inputs:
      - likelihood vector of length n
      - boolean mask of length n
      - keys starting with 'Z' with binary values: parity-based illustrative likelihood
      - other key/value pairs: stable hash-based soft likelihood perturbation
    """
    n = cam.n
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
        s = f'{key}={v}'
        h = int(hashlib.sha1(s.encode('utf-8')).hexdigest()[:8], 16) % 10000 / 10000.0
        l *= 0.4 + 0.6 * h
    floor = float(np.clip(cam.evidence_floor, 0.0, 0.5))
    l = floor + (1.0 - floor) * np.clip(l, 0.0, 1.0)
    return l

def get_cam(n: int=16, card: int=2, seed: Optional[int]=None) -> CAM:
    if n <= 0:
        raise ValueError('n must be positive')
    if card <= 1:
        raise ValueError('card must be >= 2')
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(n, dtype=float))
    y0 = rng.random((n, card))
    y0 = y0 / y0.sum(axis=1, keepdims=True)
    y1 = rng.random((n, card))
    y1 = y1 / y1.sum(axis=1, keepdims=True)
    cam = CAM(p=_normalize_prob(p), y0=y0.astype(float), y1=y1.astype(float), card=int(card), temperature=1.0, bias=np.zeros(card, dtype=float), evidence_floor=0.15, params_path=None)
    _load_cam_params(cam)
    return cam

def cam_infer(cam: CAM, target: Union[str, int]='Y', do_dict: Optional[Dict[str, Any]]=None, evidence: Optional[Dict[str, Any]]=None) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter_ns()
    if isinstance(target, str):
        if target.strip().upper() != 'Y':
            raise ValueError(f"CAM only supports target='Y', got {target!r}")
    elif isinstance(target, (int, np.integer)):
        if int(target) != 0:
            raise ValueError(f'CAM only supports target index 0, got {target}')
    else:
        raise ValueError(f'Unsupported target type: {type(target)}')
    do_dict = do_dict or {}
    x = do_dict.get('X', do_dict.get('T', 0))
    x = int(float(x) >= 0.5)
    l = _likelihood_from_evidence(cam, evidence)
    w = _normalize_prob(cam.p * l)
    yx = cam.y1 if x == 1 else cam.y0
    mix = np.clip(w @ yx, 1e-12, 1.0)
    mix = mix / mix.sum()
    if cam.bias is not None and cam.bias.size == cam.card:
        logits = _safe_log(mix) / max(cam.temperature, 1e-06) + cam.bias
        mix = _softmax(logits)
    elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
    return (mix.astype(float), float(elapsed_us))

def update_prior(cam: CAM, evidence: Optional[Dict[str, Any]], lr: float=1.0, floor: float=1e-06) -> CAM:
    lr = float(lr)
    if not 0.0 < lr <= 1.0:
        raise ValueError('lr must be in (0,1]')
    l = _likelihood_from_evidence(cam, evidence)
    p_post = _normalize_prob(cam.p * l)
    p_new = _normalize_prob((1.0 - lr) * cam.p + lr * p_post)
    p_new = np.clip(p_new, float(floor), 1.0)
    cam.p = _normalize_prob(p_new)
    _save_cam_params(cam)
    return cam

def kl_divergence(p: ArrayLike, q: ArrayLike, eps: float=1e-12) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    if p.size != q.size:
        raise ValueError('p and q must have same length')
    p = _normalize_prob(p, eps=eps)
    q = _normalize_prob(q, eps=eps)
    return float(np.sum(p * (_safe_log(p, eps) - _safe_log(q, eps))))

def _scalar_to_soft_label(value: float, card: int=2) -> np.ndarray:
    if card != 2:
        raise ValueError('scalar soft-label conversion currently supports card=2 only')
    z = float(np.clip(value, -20.0, 20.0))
    p1 = 1.0 / (1.0 + np.exp(-z))
    return np.array([1.0 - p1, p1], dtype=float)

def _row_to_evidence_dict(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ev = row.get('evidence', None)
    if ev is not None:
        return ev
    feat = {k: v for k, v in row.items() if isinstance(k, str) and (k.startswith('X') or k.startswith('Z'))}
    return feat or None

def cam_fit(cam: CAM, data: Iterable[Dict[str, Any]], smoothing: float=1.0, max_rows: Optional[int]=None, ridge: Optional[float]=None, **_: Any) -> CAM:
    """
    Robust fit for CAM. Supports:
      - categorical labels via y / Y
      - soft labels via length-card vectors
      - continuous labels via y / Y scalar (mapped to binary soft labels)
    Additional unused kwargs (e.g. ridge) are accepted for backwards compatibility.
    """
    n, card = (cam.n, cam.card)
    y0_cnt = np.full((n, card), float(smoothing), dtype=float)
    y1_cnt = np.full((n, card), float(smoothing), dtype=float)
    num = 0
    for row in data:
        if max_rows is not None and num >= int(max_rows):
            break
        num += 1
        x = row.get('x', row.get('X', row.get('T', 0)))
        x = int(float(x) >= 0.5)
        y = row.get('y', row.get('Y', None))
        if y is None:
            if x == 1 and 'y1_true' in row:
                y = row['y1_true']
            elif x == 0 and 'y0_true' in row:
                y = row['y0_true']
            else:
                raise ValueError('Each row must contain y/Y, or y0_true/y1_true consistent with treatment.')
        if isinstance(y, (list, tuple, np.ndarray)):
            yv = np.asarray(y, dtype=float).reshape(-1)
            if yv.size == 1 and card == 2:
                yv = _scalar_to_soft_label(float(yv[0]), card=card)
            elif yv.size != card:
                raise ValueError(f'y vector length must be card={card}')
            else:
                yv = np.clip(yv, 0.0, 1.0)
                yv = yv / max(float(yv.sum()), 1e-12)
        elif isinstance(y, (int, np.integer)) and 0 <= int(y) < card:
            yi = int(y)
            yv = np.zeros(card, dtype=float)
            yv[yi] = 1.0
        else:
            yv = _scalar_to_soft_label(float(y), card=card)
        if 'w' in row and row['w'] is not None:
            w = _ensure_1d(row['w'], n)
            w = _normalize_prob(np.clip(w, 0.0, 1.0))
        elif 'omega' in row and row['omega'] is not None:
            oid = int(row['omega'])
            if not 0 <= oid < n:
                raise ValueError('omega out of range')
            w = np.zeros(n, dtype=float)
            w[oid] = 1.0
        else:
            ev = _row_to_evidence_dict(row)
            l = _likelihood_from_evidence(cam, ev)
            w = _normalize_prob(cam.p * l)
        if x == 0:
            y0_cnt += w[:, None] * yv[None, :]
        else:
            y1_cnt += w[:, None] * yv[None, :]
    cam.y0 = y0_cnt / y0_cnt.sum(axis=1, keepdims=True)
    cam.y1 = y1_cnt / y1_cnt.sum(axis=1, keepdims=True)
    _save_cam_params(cam)
    return cam

def cam_time_and_accuracy(n_trials: int=1000, seed: int=0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    cam = get_cam(n=64, card=2, seed=seed)
    us_list: List[float] = []
    evid0 = {'Z0': 0}
    evid1 = {'Z0': 1}
    p00, _ = cam_infer(cam, 'Y', {'X': 0}, evid0)
    p10, _ = cam_infer(cam, 'Y', {'X': 1}, evid0)
    do_sensitive = float(np.abs(p10[1] - p00[1]) > 1e-09)
    p01, _ = cam_infer(cam, 'Y', {'X': 0}, evid1)
    evid_sensitive = float(np.abs(p01[1] - p00[1]) > 1e-09)
    for _ in range(int(n_trials)):
        x = int(rng.random() > 0.5)
        evid = {'Z0': int(rng.random() > 0.5), 'misc': float(rng.random())}
        _, us = cam_infer(cam, 'Y', {'X': x}, evid)
        us_list.append(us)
    avg_us = float(np.mean(us_list)) if us_list else 0.0
    prob_valid = float(np.allclose(p00.sum(), 1.0, atol=1e-10) and np.all(p00 >= 0))
    return {'avg_infer_us': avg_us, 'prob_valid': prob_valid, 'do_sensitive': do_sensitive, 'evidence_sensitive': evid_sensitive}
__all__ = ['CAM', 'get_cam', 'cam_infer', 'update_prior', 'kl_divergence', 'cam_time_and_accuracy', 'cam_fit']
