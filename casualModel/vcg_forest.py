from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import time
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

def _balanced_split(n: int, k_max: int, emb: np.ndarray) -> List[np.ndarray]:
    """
    Recursive balanced partition. Returns a disjoint cover of global index blocks, each of size <= k_max.
    """
    if n <= 0:
        return []
    if k_max <= 0:
        raise ValueError('k_max must be positive')
    emb = np.asarray(emb, dtype=float)
    if emb.shape[0] != n:
        raise ValueError('emb must have shape (n,d)')
    all_idx = np.arange(n, dtype=int)

    def rec(idxs: np.ndarray) -> List[np.ndarray]:
        idxs = np.asarray(idxs, dtype=int)
        if idxs.size <= k_max:
            return [idxs]
        X = emb[idxs] - emb[idxs].mean(axis=0, keepdims=True)
        if X.shape[0] <= 2:
            mid = idxs.size // 2
            return [idxs[:mid], idxs[mid:]]
        try:
            _, _, vt = np.linalg.svd(X, full_matrices=False)
            v = vt[0]
        except Exception:
            v = np.ones(X.shape[1], dtype=float)
        proj = X @ v
        med = float(np.median(proj))
        left = idxs[proj <= med]
        right = idxs[proj > med]
        if left.size == 0 or right.size == 0:
            order = np.argsort(proj)
            idxs2 = idxs[order]
            mid = idxs2.size // 2
            left, right = (idxs2[:mid], idxs2[mid:])
        return rec(left) + rec(right)
    blocks = rec(all_idx)
    flat = np.concatenate(blocks) if blocks else np.array([], dtype=int)
    if flat.size != n or len(np.unique(flat)) != n:
        raise RuntimeError('balanced_split failed to produce a disjoint cover')
    return blocks

class TanhAct:

    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def finv(y: np.ndarray) -> np.ndarray:
        y = np.clip(y, -0.999999, 0.999999)
        return 0.5 * np.log((1.0 + y) / (1.0 - y))

@dataclass
class _Tree:
    idx: np.ndarray
    A: np.ndarray
    U: np.ndarray
    V: np.ndarray
    d: int
    r: int
    card: int
    R: np.ndarray
    b: np.ndarray

    def W_ij(self, i: int, j: int) -> np.ndarray:
        return self.U[i, j] @ self.V[j].T

    def forward(self, eps: np.ndarray, do: Dict[int, Union[float, np.ndarray]], u_override: Optional[Dict[int, np.ndarray]]=None):
        k = self.idx.size
        u = np.zeros((k, self.d), dtype=float)
        u_override = u_override or {}
        for j_local in range(k):
            g = int(self.idx[j_local])
            if g in do:
                v = do[g]
                if np.isscalar(v):
                    u[j_local] = float(v)
                else:
                    vv = np.asarray(v, dtype=float).reshape(-1)
                    if vv.size != self.d:
                        raise ValueError('do vector length mismatch')
                    u[j_local] = vv
                continue
            if g in u_override:
                u[j_local] = np.asarray(u_override[g], dtype=float).reshape(-1)
                continue
            pre = np.zeros(self.d, dtype=float)
            parents = np.where(self.A[:j_local, j_local])[0]
            for i_local in parents:
                pre += self.W_ij(i_local, j_local) @ u[i_local]
            pre += eps[j_local]
            u[j_local] = TanhAct.f(pre)
        if self.card == 2:
            logits = np.sum(self.R * u, axis=1) + self.b
            p1 = _sigmoid(logits)
            return (u, p1)
        logits = np.einsum('kcd,kd->kc', self.R, u) + self.b
        prob = _softmax(logits, axis=1)
        return (u, prob)

    def abduce_partial(self, evidence_u: Dict[int, np.ndarray], do_fact: Dict[int, Union[float, np.ndarray]]):
        k = self.idx.size
        eps = np.zeros((k, self.d), dtype=float)
        u_fact = np.zeros((k, self.d), dtype=float)
        for j_local in range(k):
            g = int(self.idx[j_local])
            if g in do_fact:
                v = do_fact[g]
                if np.isscalar(v):
                    u_fact[j_local] = float(v)
                else:
                    u_fact[j_local] = np.asarray(v, dtype=float).reshape(-1)
                eps[j_local] = 0.0
                continue
            pre = np.zeros(self.d, dtype=float)
            parents = np.where(self.A[:j_local, j_local])[0]
            for i_local in parents:
                pre += self.W_ij(i_local, j_local) @ u_fact[i_local]
            if g in evidence_u:
                u_obs = np.asarray(evidence_u[g], dtype=float).reshape(-1)
                if u_obs.size != self.d:
                    raise ValueError('evidence vector length mismatch')
                u_fact[j_local] = u_obs
                eps[j_local] = TanhAct.finv(u_obs) - pre
            else:
                eps[j_local] = 0.0
                u_fact[j_local] = TanhAct.f(pre)
        return (eps, u_fact)

@dataclass
class VCGForestPP:
    n: int
    d: int
    r: int
    card: int
    trees: List[_Tree]
    default_treatment_index: int = 25
    outcome_index: int = 26

    def _tree_of(self, gidx: int) -> Optional[_Tree]:
        for t in self.trees:
            if int(gidx) in set(t.idx.tolist()):
                return t
        return None

    def infer_target(self, target_idx: int, do: Dict[int, Union[float, np.ndarray]], evidence_u: Dict[int, np.ndarray]) -> np.ndarray:
        tree = self._tree_of(target_idx)
        if tree is None:
            return np.array([0.5, 0.5], dtype=float) if self.card == 2 else np.full(self.card, 1.0 / self.card, dtype=float)
        do_fact = {}
        eps, _ = tree.abduce_partial(evidence_u=evidence_u, do_fact=do_fact)
        _, prob = tree.forward(eps=eps, do=do)
        t_local = int(np.where(tree.idx == int(target_idx))[0][0])
        if tree.card == 2:
            p1 = float(prob[t_local])
            return np.array([1.0 - p1, p1], dtype=float)
        return np.asarray(prob[t_local], dtype=float)

def _x_from_evidence(model: VCGForestPP, evidence: Optional[Dict[str, Any]]):
    evidence_u: Dict[int, np.ndarray] = {}
    obs_mask: Dict[int, bool] = {}
    if not evidence:
        return (evidence_u, obs_mask)
    if 'u' in evidence and evidence['u'] is not None:
        U = np.asarray(evidence['u'], dtype=float)
        if U.shape == (model.n, model.d):
            for i in range(model.n):
                if np.all(np.isfinite(U[i])):
                    evidence_u[int(i)] = U[i].copy()
                    obs_mask[int(i)] = True
    for k, v in evidence.items():
        if k == 'u':
            continue
        key = str(k)
        if key == 'T':
            idx = int(model.default_treatment_index)
        elif key == 'Y':
            idx = int(model.outcome_index)
        elif key.startswith('X') and key[1:].isdigit():
            idx = int(key[1:])
        elif key.startswith('Y') and key[1:].isdigit():
            idx = int(key[1:])
        elif isinstance(k, (int, np.integer)):
            idx = int(k)
        else:
            continue
        if np.isscalar(v):
            evidence_u[idx] = np.full(model.d, float(v), dtype=float)
            obs_mask[idx] = True
        else:
            vv = np.asarray(v, dtype=float).reshape(-1)
            if vv.size == model.d:
                evidence_u[idx] = vv
                obs_mask[idx] = True
    return (evidence_u, obs_mask)

def get_model(n: Optional[int]=None, card: Optional[int]=None, seed: int=0) -> VCGForestPP:
    n = int(27 if n is None else n)
    card = int(2 if card is None else card)
    d = 8
    r = 4
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, d)).astype(float) / np.sqrt(d)
    blocks = _balanced_split(n=n, k_max=min(32, max(2, n)), emb=emb)
    treat = 25 if n > 26 else max(0, n - 2)
    outcome = 26 if n > 26 else n - 1
    trees: List[_Tree] = []
    for block in blocks:
        idx_sorted = np.sort(block.astype(int))
        idx_list = idx_sorted.tolist()
        if treat in idx_list:
            idx_list.remove(treat)
            idx_list.insert(0, treat)
        if outcome in idx_list:
            idx_list.remove(outcome)
            idx_list.append(outcome)
        idx = np.array(idx_list, dtype=int)
        k = idx.size
        A = rng.random((k, k)) < 0.25
        A = np.triu(A, k=1).astype(bool)
        treat_local = int(np.where(idx == treat)[0][0]) if treat in idx else -1
        y_local = int(np.where(idx == outcome)[0][0]) if outcome in idx else -1
        if treat_local >= 0 and y_local >= 0 and (treat_local < y_local):
            A[treat_local, y_local] = True
        U = rng.standard_normal((k, k, d, r)).astype(float) * (0.25 / np.sqrt(d))
        if treat_local >= 0 and y_local >= 0 and (treat_local < y_local):
            U[treat_local, y_local] = rng.standard_normal((d, r)).astype(float) * (0.9 / np.sqrt(d))
        V = rng.standard_normal((k, d, r)).astype(float) * (0.25 / np.sqrt(d))
        if card == 2:
            R = rng.standard_normal((k, d)).astype(float) / np.sqrt(d)
            b = np.zeros(k, dtype=float)
        else:
            R = rng.standard_normal((k, card, d)).astype(float) / np.sqrt(d)
            b = np.zeros((k, card), dtype=float)
        trees.append(_Tree(idx=idx, A=A, U=U, V=V, d=d, r=r, card=card, R=R, b=b))
    default_treatment_index = treat
    return VCGForestPP(n=n, d=d, r=r, card=card, trees=trees, default_treatment_index=default_treatment_index, outcome_index=outcome)

def _parse_do_dict(model: VCGForestPP, do_dict: Dict[str, Any]) -> Dict[int, Union[float, np.ndarray]]:
    do: Dict[int, Union[float, np.ndarray]] = {}
    for k, v in (do_dict or {}).items():
        key = str(k)
        if key == 'T':
            idx = int(model.default_treatment_index)
        elif key == 'Y':
            idx = int(model.outcome_index)
        elif key.startswith('X') and key[1:].isdigit():
            idx = int(key[1:])
        elif key.startswith('Y') and key[1:].isdigit():
            idx = int(key[1:])
        elif isinstance(k, (int, np.integer)):
            idx = int(k)
        else:
            continue
        if np.isscalar(v):
            do[idx] = float(v)
        else:
            vv = np.asarray(v, dtype=float).reshape(-1)
            if vv.size == model.d:
                do[idx] = vv
    return do

def _parse_target(model: VCGForestPP, target: Union[str, int]) -> int:
    if isinstance(target, (int, np.integer)):
        idx = int(target)
    else:
        t = str(target).strip()
        if t.upper() == 'Y':
            idx = int(model.outcome_index)
        elif t.upper() == 'T':
            idx = int(model.default_treatment_index)
        elif t.startswith('X') and t[1:].isdigit():
            idx = int(t[1:])
        elif t.startswith('Y') and t[1:].isdigit():
            idx = int(t[1:])
        else:
            idx = int(model.outcome_index)
    if not 0 <= idx < model.n:
        raise ValueError(f'target index out of range: {idx}')
    return idx

def infer(model: VCGForestPP, target: Union[str, int], do_dict: Dict[str, Any], evidence: Optional[Dict[str, Any]]=None):
    t0 = time.perf_counter_ns()
    do = _parse_do_dict(model, do_dict)
    target_idx = _parse_target(model, target)
    evidence_u, _ = _x_from_evidence(model, evidence)
    prob_vec = model.infer_target(target_idx=target_idx, do=do, evidence_u=evidence_u)
    elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
    return (prob_vec.astype(float), float(elapsed_us))

def infer_y_prob(evidence: Optional[Dict[str, Any]]=None, do_t: int=1) -> Tuple[float, float]:
    model = get_model()
    prob_vec, us = infer(model, 'Y', {'T': int(do_t)}, evidence)
    return (float(prob_vec[1] if prob_vec.size >= 2 else prob_vec[0]), float(us))
__all__ = ['_balanced_split', 'get_model', '_x_from_evidence', 'infer', 'infer_y_prob', 'VCGForestPP']
