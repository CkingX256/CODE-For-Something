from __future__ import annotations
import hashlib
import math
import time
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
RANK = 128
S_SAMPLES = 32
PHI_DIM = 32
CACHE_HET_SLOTS = 4096
CACHE_HOM_SLOTS = 2048
DEFAULT_ENTROPY_H0 = 0.85
RIDGE_L2 = 1e-06
_RESERVED_CTX_KEYS = {'__u__', '_u', 'context_vec', '__topo__', '__topo_hash__'}

def _ns() -> int:
    return time.perf_counter_ns()

def _to_us(delta_ns: int) -> float:
    return float(delta_ns) / 1000.0

def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    m = float(np.max(x)) if x.size else 0.0
    e = np.exp(x - m)
    s = float(np.sum(e)) + 1e-12
    return (e / s).astype(np.float64)

def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = np.clip(p, 1e-12, 1.0)
    H = -float(np.sum(p * np.log(p)))
    K = max(2, int(p.size))
    return float(H / math.log(K))

def _stable_hash16(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:16]

def topo_hash_from_any(topology: Optional[Any]) -> str:
    if topology is None:
        return 'NA'
    if isinstance(topology, str):
        return topology
    try:
        if isinstance(topology, np.ndarray):
            b = topology.astype(np.int8, copy=False).tobytes() + str(topology.shape).encode('utf-8')
        else:
            b = repr(topology).encode('utf-8')
        return hashlib.sha1(b).hexdigest()[:16]
    except Exception:
        return hashlib.sha1(repr(topology).encode('utf-8')).hexdigest()[:16]

def _canonicalize_intervention(do_dict: Dict[Any, Any]) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    for k, v in do_dict.items():
        ks = str(k)
        if ks in _RESERVED_CTX_KEYS:
            continue
        try:
            fv = float(v)
        except Exception:
            hv = int(hashlib.sha1(repr(v).encode('utf-8')).hexdigest()[:8], 16)
            fv = hv % 1000000 / 1000000.0
        items.append((ks, fv))
    items.sort(key=lambda x: x[0])
    return items

def intervention_signature(do_dict: Dict[Any, Any]) -> str:
    items = _canonicalize_intervention(do_dict)
    s = '|'.join((f'{k}:{v:.8f}' for k, v in items))
    return _stable_hash16(s)

def _hash_vec32(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return hashlib.sha1(x.tobytes()).hexdigest()[:16]

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = int(max(1, capacity))
        self._od: 'OrderedDict[Any, Any]' = OrderedDict()

    def get(self, key: Any) -> Any:
        if key in self._od:
            self._od.move_to_end(key)
            return self._od[key]
        return None

    def put(self, key: Any, value: Any) -> None:
        self._od[key] = value
        self._od.move_to_end(key)
        if len(self._od) > self.capacity:
            self._od.popitem(last=False)

def M(mat: np.ndarray) -> np.ndarray:
    """Klein involution operator M(A) = -flipud(fliplr(A)).T with M^2 = Id."""
    return -np.flipud(np.fliplr(mat)).T

class KMCCluster:

    def __init__(self, K: int=32, dim: int=8, beta: float=8.0, gamma: float=0.95, seed: int=0):
        self.K = int(max(2, K))
        self.dim = int(max(1, dim))
        self.beta = float(beta)
        self.gamma = float(gamma)
        rng = np.random.RandomState(int(seed) & 4294967295)
        C = rng.normal(size=(self.K, self.dim)).astype(np.float64)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        self.C = C
        self._proj = rng.normal(size=(self.dim, self.dim)).astype(np.float64)

    def _embed(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        if u.size == self.dim:
            out = u
        else:
            tmp = np.zeros(self.dim, dtype=np.float64)
            m = min(self.dim, int(u.size))
            if m > 0:
                tmp[:m] = u[:m]
            out = self._proj @ tmp
        out /= np.linalg.norm(out) + 1e-12
        return out

    def membership(self, u: np.ndarray) -> np.ndarray:
        z = self.beta * (self.C @ self._embed(u))
        return _softmax(z)

    def signature(self, u: np.ndarray) -> int:
        return int(np.argmax(self.membership(u)))

    def entropy(self, u: np.ndarray) -> float:
        return _entropy(self.membership(u))

class PhiNet:
    """Three-layer tanh feature map with deterministic initialization."""

    def __init__(self, n_in: int, hidden: int, n_out: int, seed: int=0):
        rng = np.random.RandomState(int(seed) & 4294967295)
        self.W1 = (rng.normal(size=(n_in, hidden)) / np.sqrt(max(1, n_in))).astype(np.float32)
        self.W2 = (rng.normal(size=(hidden, hidden)) / np.sqrt(max(1, hidden))).astype(np.float32)
        self.W3 = (rng.normal(size=(hidden, n_out)) / np.sqrt(max(1, hidden))).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x0 = np.asarray(x, dtype=np.float32).reshape(1, -1)
        h1 = np.tanh(x0 @ self.W1)
        h2 = np.tanh(h1 @ self.W2)
        out = np.tanh(h2 @ self.W3).reshape(-1)
        return out.astype(np.float32)

class KleinLowRankCache:
    """Low-rank cache storing basis U and per-intervention alpha tables."""

    def __init__(self, n: int, rank: int, S: int=32, seed: int=0):
        self.n = int(n)
        self.rank = int(rank)
        self.S = int(S)
        rng = np.random.RandomState(int(seed) & 4294967295)
        self.U = (rng.normal(size=(self.n, self.rank)) * 0.5).astype(np.float32)
        self.alphas: Dict[str, np.ndarray] = {}

    def _init_alphas_deterministic(self, x_name: str) -> np.ndarray:
        """Use stable hash-based initialization for cache misses."""
        seed = int(hashlib.sha1(x_name.encode('utf-8')).hexdigest()[:8], 16) & 4294967295
        rng = np.random.RandomState(seed)
        a = rng.normal(size=(self.S, self.rank)).astype(np.float32)
        a *= 0.2
        return a

    def set_alphas(self, x_name: str, alphas: np.ndarray) -> None:
        alphas = np.asarray(alphas, dtype=np.float32)
        if alphas.shape != (self.S, self.rank):
            raise ValueError(f'alphas shape expected {(self.S, self.rank)}, got {alphas.shape}')
        self.alphas[str(x_name)] = alphas

    def query(self, x_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if x_name not in self.alphas:
            self.alphas[x_name] = self._init_alphas_deterministic(x_name)
        return (self.U, self.alphas[x_name])

@dataclass(frozen=True)
class CacheKey:
    inter_sig: str
    topo_hash: str
    model_version: str
    cluster_sig: int
    alpha_hash: str
    evid_hash: str

class KleinVC:

    def __init__(self, p: np.ndarray, Delta0: np.ndarray, Delta1: np.ndarray, gate: np.ndarray, hop_mask: np.ndarray, phi_net: PhiNet, cache: KleinLowRankCache, var_count: int=4, model_version: str='v1', topology: Optional[Any]=None, cluster: Optional[KMCCluster]=None, entropy_H0: float=DEFAULT_ENTROPY_H0, seed: int=0):
        self.p = np.asarray(p, dtype=np.float32).reshape(-1)
        self.Delta0 = np.asarray(Delta0, dtype=np.float32).reshape(-1)
        self.Delta1 = np.asarray(Delta1, dtype=np.float32).reshape(-1)
        self.gate = np.asarray(gate, dtype=np.float32).reshape(-1)
        self.hop_mask = np.asarray(hop_mask, dtype=np.float32).reshape(-1)
        self.phi_net = phi_net
        self.cache = cache
        self.var_count = int(var_count)
        self.version = str(model_version)
        self.topo_hash = topo_hash_from_any(topology)
        self.cluster = cluster if cluster is not None else KMCCluster(K=32, dim=8, seed=seed)
        self.entropy_H0 = float(entropy_H0)
        self.cache_het = LRUCache(CACHE_HET_SLOTS)
        self.cache_hom = LRUCache(CACHE_HOM_SLOTS)
        rng = np.random.RandomState(int(seed) & 4294967295)
        self.W_phi = (rng.normal(size=(PHI_DIM, RANK)) / np.sqrt(max(1, PHI_DIM))).astype(np.float32)

    def set_topology(self, topology: Any) -> None:
        self.topo_hash = topo_hash_from_any(topology)

    def set_model_version(self, v: str) -> None:
        self.version = str(v)

    def _extract_context_vec(self, do_dict: Dict[Any, Any], feature_vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if feature_vec is not None:
            try:
                return np.asarray(feature_vec, dtype=np.float64).reshape(-1)
            except Exception:
                return None
        for k in ('__u__', '_u', 'context_vec'):
            if k in do_dict:
                try:
                    return np.asarray(do_dict[k], dtype=np.float64).reshape(-1)
                except Exception:
                    return None
        return None

    def _make_query_vec(self, do_dict: Dict[Any, Any], evidence: Optional[Dict[Any, Any]], feature_vec: Optional[np.ndarray]) -> np.ndarray:
        u = self._extract_context_vec(do_dict, feature_vec)
        if u is not None and u.size:
            return u
        items = _canonicalize_intervention(do_dict)
        vals = np.array([v for _, v in items], dtype=np.float64) if items else np.zeros(0, dtype=np.float64)
        mean = float(vals.mean()) if vals.size else 0.0
        std = float(vals.std()) if vals.size else 0.0
        mx = float(vals.max()) if vals.size else 0.0
        mn = float(vals.min()) if vals.size else 0.0
        do_n = float(len(items))
        ev_n = float(len(evidence) if evidence is not None else 0)
        return np.array([do_n, ev_n, mean, std, mx, mn], dtype=np.float64)

    def _cluster_sig_and_entropy(self, qvec: np.ndarray) -> Tuple[int, float]:
        m = self.cluster.membership(qvec)
        return (int(np.argmax(m)), _entropy(m))

    def _forward_p1(self, alpha_vec: np.ndarray, evid_mask: np.ndarray, x_name: str) -> float:
        U, alphas = self.cache.query(x_name)
        delta = self.Delta0 + self.Delta1 * alpha_vec + self.gate * alpha_vec ** 2
        w = self.p * evid_mask
        if float(w.sum()) <= 0:
            w = self.p.copy()
        w = w / (float(w.sum()) + 1e-12)
        w *= 1.0 + 0.2 * self.hop_mask
        w = w / (float(w.sum()) + 1e-12)
        inp = (w * delta).reshape(1, -1)
        phi = self.phi_net(inp)
        phi_rank = phi @ self.W_phi
        Q = np.outer(delta, w).astype(np.float32)
        K = np.block([[Q, np.zeros_like(Q)], [np.zeros_like(Q), M(Q.T)]])
        reg = float(np.linalg.norm(K - M(K.T), ord='fro'))
        y_pred = float(np.tensordot(alphas, phi_rank.astype(np.float32), axes=(1, 0)).mean())
        y_pred = float(np.tanh(y_pred) * 0.5 + 0.5 + 0.1 * reg)
        y_pred = float(np.clip(y_pred, 0.0, 1.0))
        return y_pred

    def infer(self, target: str, do_dict: Dict[Any, Any], evidence: Optional[Dict[Any, Any]]=None, feature_vec: Optional[np.ndarray]=None) -> Tuple[np.ndarray, float]:
        """
        Returns: (prob_vec[np.ndarray shape (2,)], elapsed_us[float])
        """
        t0 = _ns()
        n = int(self.p.size)
        if evidence is None:
            evid_mask = np.ones(n, dtype=np.float32)
        else:
            evid_mask = np.ones(n, dtype=np.float32)
            for k, v in evidence.items():
                if isinstance(k, (int, str)) and (str(k).startswith('Z') or str(k) == 'Z' or k == 1):
                    evid_mask *= (np.arange(n) % 2 == int(v)).astype(np.float32)
            if float(evid_mask.sum()) == 0.0:
                evid_mask = np.ones(n, dtype=np.float32)
        alpha_vec = np.zeros(n, dtype=np.float32)
        items = _canonicalize_intervention(do_dict)
        var_keys = [k for k, _ in items]
        if len(var_keys) == 0:
            seg_size = n
        else:
            seg_size = max(1, n // max(1, len(var_keys)))
        for i, (ks, fv) in enumerate(items):
            seg = int(i)
            start = seg * seg_size
            end = min(n, (seg + 1) * seg_size)
            if start >= n:
                break
            alpha_vec[start:end] = float(fv)
        qvec = self._make_query_vec(do_dict, evidence, feature_vec)
        cluster_sig, H = self._cluster_sig_and_entropy(qvec)
        force_hom_only = bool(H > self.entropy_H0)
        inter_sig = intervention_signature(do_dict)
        alpha_h = _hash_vec32(alpha_vec)
        evid_h = _hash_vec32(evid_mask)
        if not force_hom_only:
            k_het = CacheKey(inter_sig, self.topo_hash, self.version, int(cluster_sig), alpha_h, evid_h)
            v = self.cache_het.get(k_het)
            if v is not None:
                p1 = float(v)
                y = np.array([1.0 - p1, p1], dtype=np.float32)
                return (y, _to_us(_ns() - t0))
        k_hom = CacheKey(inter_sig, self.topo_hash, self.version, -1, alpha_h, evid_h)
        v = self.cache_hom.get(k_hom)
        if v is not None:
            p1 = float(v)
            if not force_hom_only:
                k_het = CacheKey(inter_sig, self.topo_hash, self.version, int(cluster_sig), alpha_h, evid_h)
                self.cache_het.put(k_het, p1)
            y = np.array([1.0 - p1, p1], dtype=np.float32)
            return (y, _to_us(_ns() - t0))
        x_name = str(items)
        p1 = self._forward_p1(alpha_vec, evid_mask, x_name)
        self.cache_hom.put(k_hom, p1)
        if not force_hom_only:
            k_het = CacheKey(inter_sig, self.topo_hash, self.version, int(cluster_sig), alpha_h, evid_h)
            self.cache_het.put(k_het, p1)
        y = np.array([1.0 - float(p1), float(p1)], dtype=np.float32)
        return (y, _to_us(_ns() - t0))

def get_klein_vc(n: int, card: int, seed: int=0) -> KleinVC:
    """Create a KleinVC instance while preserving the original n/card interface."""
    n = int(max(2, n))
    rng = np.random.RandomState(int(seed) & 4294967295)
    p = rng.rand(n).astype(np.float32) + 0.1
    p /= float(p.sum())
    Delta0 = rng.randn(n).astype(np.float32) * 0.1
    Delta1 = rng.randn(n).astype(np.float32) * 0.05
    gate = rng.rand(n).astype(np.float32) * 0.1
    hop_mask = rng.rand(n).astype(np.float32)
    phi_net = PhiNet(n_in=n, hidden=8, n_out=PHI_DIM, seed=seed)
    cache = KleinLowRankCache(n=n, rank=RANK, S=S_SAMPLES, seed=seed)
    cluster = KMCCluster(K=32, dim=8, beta=8.0, gamma=0.95, seed=seed)
    return KleinVC(p=p, Delta0=Delta0, Delta1=Delta1, gate=gate, hop_mask=hop_mask, phi_net=phi_net, cache=cache, var_count=4, model_version='v1', topology=None, cluster=cluster, entropy_H0=DEFAULT_ENTROPY_H0, seed=seed)

def klein_infer(kvc: KleinVC, target: str, do_dict: Dict[Any, Any], evidence: Optional[Dict[Any, Any]]=None, feature_vec: Optional[np.ndarray]=None) -> Tuple[np.ndarray, float]:
    """
    Returns: (prob_vec[2], elapsed_us)
    """
    return kvc.infer(target=target, do_dict=do_dict, evidence=evidence, feature_vec=feature_vec)

def update_prior(kvc: KleinVC, obs_counts: np.ndarray) -> None:
    """
    Update-Prior：p <- Dirichlet(p + n_obs)
    O(n) update that preserves the cache format.
    """
    obs = np.asarray(obs_counts, dtype=np.float64).reshape(-1)
    if obs.size != kvc.p.size:
        raise ValueError(f'obs_counts size {obs.size} must match p size {kvc.p.size}')
    p = kvc.p.astype(np.float64) + obs
    p = np.clip(p, 1e-12, None)
    p /= float(p.sum())
    kvc.p = p.astype(np.float32)

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * np.log(p / q)))

def klein_time_and_accuracy(kvc: KleinVC, dataset: List[Tuple[Dict[Any, Any], Dict[Any, Any], int]], target: str='Y') -> Tuple[float, float]:
    """
    Simple evaluation helper over samples (do_dict, evidence, y_true).
    """
    times = []
    correct = 0
    total = 0
    for do_dict, evidence, y_true in dataset:
        y, us = klein_infer(kvc, target, do_dict, evidence=evidence)
        pred = int(np.argmax(y))
        correct += int(pred == int(y_true))
        total += 1
        times.append(float(us))
    avg_us = float(np.mean(times)) if times else 0.0
    acc = float(correct) / float(max(1, total))
    return (avg_us, acc)
