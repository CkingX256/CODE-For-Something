from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import time
import numpy as np
import random as _random
__version__ = 'refined_v3_fixed'

def set_seed(seed: int) -> None:
    _random.seed(seed)
    np.random.seed(seed)

def kl_discrete(p: np.ndarray, q: np.ndarray, eps: float=1e-12) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    p = p / max(float(p.sum()), eps)
    q = q / max(float(q.sum()), eps)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))

class Act:

    def f(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def finv(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Identity(Act):

    def f(self, x: np.ndarray) -> np.ndarray:
        return x

    def finv(self, y: np.ndarray) -> np.ndarray:
        return y

class Tanh(Act):

    def f(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def finv(self, y: np.ndarray) -> np.ndarray:
        y = np.clip(y, -0.999999, 0.999999)
        return 0.5 * np.log((1.0 + y) / (1.0 - y))

def _random_dag(k: int, edge_prob: float=0.25, seed: Optional[int]=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.random((k, k)) < float(edge_prob)
    return np.triu(A, k=1).astype(bool)

@dataclass
class VCGModel:
    k: int
    d: int
    A: np.ndarray
    W: np.ndarray
    act: Act = Identity()
    noise_scale: float = 0.1
    R: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    treatment_index: int = 25
    outcome_index: int = 26

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=bool)
        self.W = np.asarray(self.W, dtype=float)
        if self.R is None:
            rng = np.random.default_rng(0)
            self.R = rng.standard_normal((self.k, self.d)).astype(float) / np.sqrt(self.d)
        if self.b is None:
            self.b = np.zeros(self.k, dtype=float)

    def sample_eps(self, seed: Optional[int]=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((self.k, self.d)).astype(float) * float(self.noise_scale)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, eps: Optional[np.ndarray]=None, do: Optional[Dict[int, Union[float, np.ndarray]]]=None) -> Tuple[np.ndarray, np.ndarray]:
        if eps is None:
            eps = self.sample_eps()
        eps = np.asarray(eps, dtype=float)
        if eps.shape != (self.k, self.d):
            raise ValueError(f'eps must have shape {(self.k, self.d)}')
        u = np.zeros((self.k, self.d), dtype=float)
        do = do or {}
        for j in range(self.k):
            if j in do:
                v = do[j]
                if np.isscalar(v):
                    u[j] = float(v)
                else:
                    vv = np.asarray(v, dtype=float).reshape(-1)
                    if vv.size != self.d:
                        raise ValueError(f'do[{j}] vector must have length d={self.d}')
                    u[j] = vv
                continue
            pre = np.zeros(self.d, dtype=float)
            parents = np.where(self.A[:j, j])[0]
            for i in parents:
                pre += self.W[i, j] @ u[i]
            pre += eps[j]
            u[j] = self.act.f(pre)
        logits = np.sum(self.R * u, axis=1) + self.b
        p = self._sigmoid(logits)
        return (u, p)

    def abduce(self, u_fact: np.ndarray, do_fact: Optional[Dict[int, Union[float, np.ndarray]]]=None) -> np.ndarray:
        u_fact = np.asarray(u_fact, dtype=float)
        if u_fact.shape != (self.k, self.d):
            raise ValueError(f'u_fact must have shape {(self.k, self.d)}')
        do_fact = do_fact or {}
        eps = np.zeros((self.k, self.d), dtype=float)
        for j in range(self.k):
            if j in do_fact:
                eps[j] = 0.0
                continue
            pre = np.zeros(self.d, dtype=float)
            parents = np.where(self.A[:j, j])[0]
            for i in parents:
                pre += self.W[i, j] @ u_fact[i]
            eps[j] = self.act.finv(u_fact[j]) - pre
        return eps

    def intervene(self, nodes: Sequence[int], val: Union[float, np.ndarray]=1.0, eps: Optional[np.ndarray]=None):
        do = {int(n): val for n in nodes}
        return self.forward(eps=eps, do=do)

    def counterfactual(self, fact_u: np.ndarray, nodes: Sequence[int], val: Union[float, np.ndarray]=1.0, do_fact: Optional[Dict[int, Union[float, np.ndarray]]]=None) -> Tuple[np.ndarray, np.ndarray]:
        eps = self.abduce(fact_u, do_fact=do_fact)
        do = {int(n): val for n in nodes}
        return self.forward(eps=eps, do=do)

def get_model(k: int=27, d: int=8, edge_prob: float=0.2, seed: int=0, activation: str='tanh', noise_scale: float=0.1, treatment_index: int=25, outcome_index: int=26) -> VCGModel:
    if k <= max(treatment_index, outcome_index):
        k = max(treatment_index, outcome_index) + 1
    rng = np.random.default_rng(seed)
    A = _random_dag(k, edge_prob=edge_prob, seed=seed)
    W = np.zeros((k, k, d, d), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            if A[i, j]:
                W[i, j] = rng.standard_normal((d, d)).astype(float) * (0.2 / np.sqrt(d))
    for i in range(min(25, outcome_index)):
        A[i, outcome_index] = True
        W[i, outcome_index] = rng.standard_normal((d, d)).astype(float) * (0.12 / np.sqrt(d))
    A[treatment_index, outcome_index] = True
    W[treatment_index, outcome_index] = rng.standard_normal((d, d)).astype(float) * (0.75 / np.sqrt(d))
    act: Act = Tanh() if activation.lower() == 'tanh' else Identity()
    R = rng.standard_normal((k, d)).astype(float) / np.sqrt(d)
    b = np.zeros(k, dtype=float)
    R[outcome_index] = rng.standard_normal(d).astype(float) * (0.9 / np.sqrt(d))
    return VCGModel(k=k, d=d, A=A, W=W, act=act, noise_scale=float(noise_scale), R=R, b=b, treatment_index=treatment_index, outcome_index=outcome_index)

def infer(task: str, **kwargs):
    m: VCGModel = kwargs.get('model', None) or get_model()
    if task == 'intervene':
        nodes = kwargs.get('nodes', [])
        val = kwargs.get('val', 1.0)
        eps = kwargs.get('eps', None)
        return m.intervene(nodes, val, eps=eps)
    if task == 'counterfactual':
        fact_u = kwargs['fact_u']
        nodes = kwargs.get('nodes', [])
        val = kwargs.get('val', 1.0)
        do_fact = kwargs.get('do_fact', None)
        return m.counterfactual(fact_u, nodes, val, do_fact=do_fact)
    raise ValueError(f'Unknown task: {task!r}')

def infer_y_from_features(model: VCGModel, x_evidence: Dict[str, Any], do_t: int, eps_seed: int=0) -> Tuple[float, float]:
    t0 = time.perf_counter_ns()
    do: Dict[int, Union[float, np.ndarray]] = {}
    for i in range(25):
        do[i] = float(int(x_evidence.get(f'X{i}', 0)) >= 1)
    do[model.treatment_index] = float(do_t)
    _, p = model.forward(eps=model.sample_eps(seed=eps_seed), do=do)
    elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
    return (float(np.clip(p[model.outcome_index], 1e-06, 1 - 1e-06)), float(elapsed_us))
__all__ = ['set_seed', 'kl_discrete', 'get_model', 'infer', 'infer_y_from_features', 'VCGModel']
