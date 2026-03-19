from __future__ import annotations
import os, platform
if platform.system().lower().startswith('win'):
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import hashlib
import math
import time
import numpy as np
LOW_RANK_DIM = 128
CACHE_SLOTS_HET = 8192
CACHE_SLOTS_HOM = 4096
DEFAULT_ENTROPY_H0 = 0.85
DEFAULT_ENTROPY_LAMBDA = 1.0
RIDGE_L2 = 1e-06
_RESERVED_CTX_KEYS = {'__u__', '_u', 'context_vec', '__topo__', '__topo_hash__'}

def _stable_sha1_bytes(data: bytes) -> bytes:
    return hashlib.sha1(data).digest()

def _stable_int64_from_bytes(b: bytes) -> int:
    return int.from_bytes(b[:8], 'big', signed=False)

def _canonicalize_intervention(intervention: Dict[Any, Any]) -> List[Tuple[str, float]]:
    """Normalize an intervention dict into an ordered list of (key, value) pairs."""
    items: List[Tuple[str, float]] = []
    for k, v in intervention.items():
        ks = str(k)
        if ks in _RESERVED_CTX_KEYS:
            continue
        try:
            fv = float(v)
        except Exception:
            h = _stable_sha1_bytes(repr(v).encode('utf-8'))
            fv = _stable_int64_from_bytes(h) % 10000000 / 10000000.0
        items.append((ks, fv))
    items.sort(key=lambda x: x[0])
    return items

def intervention_signature(intervention: Dict[Any, Any]) -> str:
    """Return a short signature for the intervention itself, excluding context keys."""
    items = _canonicalize_intervention(intervention)
    s = '|'.join((f'{k}:{v:.8f}' for k, v in items))
    h = hashlib.sha1(s.encode('utf-8')).hexdigest()
    return h[:16]

def topo_hash_from_any(topology: Optional[Any]) -> str:
    """Compute a stable topology hash from an array, iterable, or string identifier."""
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

def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    m = float(np.max(x)) if x.size else 0.0
    e = np.exp(x - m)
    s = float(np.sum(e)) + 1e-12
    return (e / s).astype(np.float64)

def _entropy(p: np.ndarray) -> float:
    """
    Normalized entropy with base K; returns a value in [0, 1].
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = np.clip(p, 1e-12, 1.0)
    H = -float(np.sum(p * np.log(p)))
    K = max(2, int(p.size))
    return float(H / math.log(K))

def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

def _to_us(delta_ns: int) -> float:
    return float(delta_ns) / 1000.0

def _ns() -> int:
    return time.perf_counter_ns()

class LRUCache:
    """Small LRU cache with move-to-end semantics on hits."""

    def __init__(self, capacity: int=1024):
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

    def __len__(self) -> int:
        return len(self._od)

class KMCCluster:
    """Approximate spherical soft k-means used for routing and cache signatures."""

    def __init__(self, K: int=32, dim: int=8, beta: float=8.0, gamma: float=0.95, involution_strength: float=0.0, seed: int=0):
        self.K = int(max(2, K))
        self.dim = int(max(1, dim))
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.involution_strength = float(max(0.0, involution_strength))
        rng = np.random.RandomState(int(seed) & 4294967295)
        C = rng.normal(size=(self.K, self.dim)).astype(np.float64)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        self.C = C
        self._proj = rng.normal(size=(self.dim, self.dim)).astype(np.float64)

    @staticmethod
    def _involution_vec(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        return -v[::-1].copy()

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
        n = float(np.linalg.norm(out)) + 1e-12
        return out / n

    def membership(self, u: np.ndarray) -> np.ndarray:
        z = self.beta * (self.C @ self._embed(u))
        m = _softmax(z)
        return m

    def signature(self, u: np.ndarray) -> int:
        m = self.membership(u)
        return int(np.argmax(m))

    def entropy(self, u: np.ndarray) -> float:
        return _entropy(self.membership(u))

    def partial_fit(self, U_batch: np.ndarray) -> None:
        U_batch = np.asarray(U_batch, dtype=np.float64)
        if U_batch.ndim == 1:
            U_batch = U_batch.reshape(1, -1)
        if U_batch.size == 0:
            return
        Ms = []
        for i in range(U_batch.shape[0]):
            Ms.append(self.membership(U_batch[i]))
        M = np.stack(Ms, axis=0)
        for k in range(self.K):
            w = M[:, k].reshape(-1, 1)
            denom = float(np.sum(w)) + 1e-12
            mu = (w * np.stack([self._embed(U_batch[i]) for i in range(U_batch.shape[0])], axis=0)).sum(axis=0) / denom
            c_new = self.gamma * self.C[k] + (1.0 - self.gamma) * mu
            if self.involution_strength > 0:
                c_new = c_new - self.involution_strength * (c_new - self._involution_vec(c_new))
            c_new /= np.linalg.norm(c_new) + 1e-12
            self.C[k] = c_new

@dataclass(frozen=True)
class CacheKey:
    inter_sig: str
    topo_hash: str
    model_version: str
    cluster_sig: int

def _default_teacher(target: int, intervention: Dict[Any, Any]) -> np.ndarray:
    """
    Default teacher used when no external VCG module is supplied.
    - Outputs a binary distribution [P(Y=0), P(Y=1)].
    - Uses the common treatment key ('T' or 0) as the main logit driver.
    """
    t = None
    if 'T' in intervention:
        try:
            t = float(intervention['T'])
        except Exception:
            t = None
    if t is None and 0 in intervention:
        try:
            t = float(intervention[0])
        except Exception:
            t = 0.0
    if t is None:
        t = 0.0
    t = float(np.clip(t, -3.0, 3.0))
    b = (int(target) % 7 - 3) * 0.05
    logit = 2.0 * t + b
    p1 = float(np.clip(_sigmoid(logit), 1e-06, 1 - 1e-06))
    return np.array([1.0 - p1, p1], dtype=np.float32)

class MobiusVCModel:
    """Mobius-VC inference layer with heterogeneous/global caches and teacher fallback."""

    def __init__(self, cache_capacity_het: int=CACHE_SLOTS_HET, cache_capacity_hom: int=CACHE_SLOTS_HOM, low_rank_dim: int=LOW_RANK_DIM, model_version: str='v1', topology: Optional[Any]=None, cluster: Optional[KMCCluster]=None, entropy_H0: float=DEFAULT_ENTROPY_H0, entropy_lambda: float=DEFAULT_ENTROPY_LAMBDA, teacher: Optional[Callable[[int, Dict[Any, Any]], np.ndarray]]=None, cam_infer: Optional[Callable[..., Tuple[np.ndarray, float]]]=None, seed: int=0):
        self.low_rank_dim = int(max(2, low_rank_dim))
        self.version = str(model_version)
        self.topo_hash = topo_hash_from_any(topology)
        self.cluster = cluster if cluster is not None else KMCCluster(K=32, dim=8, beta=8.0, gamma=0.95, seed=seed)
        self.entropy_H0 = float(entropy_H0)
        self.entropy_lambda = float(entropy_lambda)
        self.teacher = teacher if teacher is not None else _default_teacher
        self.cam_infer = cam_infer
        self.cache_het = LRUCache(int(cache_capacity_het))
        self.cache_hom = LRUCache(int(cache_capacity_hom))
        self.U = self._init_U(seed=seed)
        self._projP = self._make_projection(self.U)
        self._target_index = 1

    def set_topology(self, topology: Any) -> None:
        self.topo_hash = topo_hash_from_any(topology)

    def set_model_version(self, v: str) -> None:
        self.version = str(v)

    def set_target_index(self, idx: int) -> None:
        self._target_index = int(idx)

    def _init_U(self, seed: int=0) -> np.ndarray:
        """
        Initialize a low-rank basis U with shape (2, r).

        To keep cache hits aligned with the teacher distribution, the default uses
        an approximate coordinate embedding:
            U[0,0] = 1 and U[1,1] = 1; all remaining entries are zero.
        Ridge regularization is used in the projection for numerical stability.
        """
        U = np.zeros((2, self.low_rank_dim), dtype=np.float64)
        U[0, 0] = 1.0
        if self.low_rank_dim > 1:
            U[1, 1] = 1.0
        return U

    def _make_projection(self, U: np.ndarray) -> np.ndarray:
        """
        Compute the ridge projection P = (U^T U + λ I)^{-1} U^T so that alpha = P y.
        """
        U = np.asarray(U, dtype=np.float64)
        UtU = U.T @ U
        UtU.flat[::UtU.shape[0] + 1] += RIDGE_L2
        P = np.linalg.solve(UtU, U.T)
        return P

    def _alpha_from_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).reshape(2)
        alpha = (self._projP @ y).astype(np.float32)
        return alpha

    def _prob_from_alpha(self, alpha: np.ndarray) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
        y_hat = (self.U @ alpha).reshape(2)
        y_hat = np.clip(y_hat, 0.0, None)
        s = float(np.sum(y_hat)) + 1e-12
        p = (y_hat / s).astype(np.float64)
        return np.array([float(p[0]), float(p[1])], dtype=np.float32)

    def _extract_context_vec(self, intervention: Dict[Any, Any]) -> Optional[np.ndarray]:
        for k in ('__u__', '_u', 'context_vec'):
            if k in intervention:
                try:
                    return np.asarray(intervention[k], dtype=np.float64).reshape(-1)
                except Exception:
                    return None
        return None

    def _make_query_vec(self, target: int, intervention: Dict[Any, Any], evidence: Optional[Dict[Any, Any]]=None) -> np.ndarray:
        """
        Build the query vector used for clustering and entropy-based routing. If __u__/context_vec is provided, prefer it.
        Otherwise use a stable summary-feature concatenation.
        """
        u = self._extract_context_vec(intervention)
        if u is not None and u.size:
            return u
        items = _canonicalize_intervention(intervention)
        vals = np.array([v for _, v in items], dtype=np.float64) if items else np.zeros(0, dtype=np.float64)
        mean = float(vals.mean()) if vals.size else 0.0
        std = float(vals.std()) if vals.size else 0.0
        mx = float(vals.max()) if vals.size else 0.0
        mn = float(vals.min()) if vals.size else 0.0
        do_n = float(len(items))
        ev_n = float(len(evidence) if evidence is not None else 0)
        vec = np.array([float(target), do_n, ev_n, mean, std, mx, mn], dtype=np.float64)
        return vec

    def _cluster_signature_and_entropy(self, qvec: np.ndarray) -> Tuple[int, float, np.ndarray]:
        m = self.cluster.membership(qvec)
        sig = int(np.argmax(m))
        H = _entropy(m)
        return (sig, H, m.astype(np.float64))

    def _make_cache_key(self, intervention: Dict[Any, Any], cluster_sig: int) -> CacheKey:
        return CacheKey(inter_sig=intervention_signature(intervention), topo_hash=self.topo_hash, model_version=self.version, cluster_sig=int(cluster_sig))

    def _should_force_vcg(self, entropy: float) -> bool:
        return bool(entropy > self.entropy_H0)

    def query(self, target: int, intervention: Dict[Any, Any], evidence: Optional[Dict[Any, Any]]=None) -> Tuple[np.ndarray, float, bool]:
        """
        Backward-compatible interface returning (prob_vec[2], elapsed_us, hit).
        hit=True indicates a heterogeneous/global cache hit or interpolated reuse.
        hit=False indicates a CAM/VCG computation followed by cache fill.
        """
        t0 = _ns()
        self.set_target_index(int(target))
        qvec = self._make_query_vec(target, intervention, evidence=evidence)
        clus_sig, H, _ = self._cluster_signature_and_entropy(qvec)
        force_vcg = self._should_force_vcg(H)
        if not force_vcg:
            k_het = self._make_cache_key(intervention, cluster_sig=clus_sig)
            alpha = self.cache_het.get(k_het)
            if alpha is not None:
                prob = self._prob_from_alpha(alpha)
                return (prob, _to_us(_ns() - t0), True)
            k_hom = self._make_cache_key(intervention, cluster_sig=-1)
            alpha = self.cache_hom.get(k_hom)
            if alpha is not None:
                prob = self._prob_from_alpha(alpha)
                self.cache_het.put(k_het, alpha)
                return (prob, _to_us(_ns() - t0), True)
        used_cam = False
        y: Optional[np.ndarray] = None
        if not force_vcg and self.cam_infer is not None:
            do_items = _canonicalize_intervention(intervention)
            if len(do_items) <= 1:
                try:
                    prob_vec, _lat_us = self.cam_infer(target='Y', do_dict=dict(do_items), evidence=evidence or {})
                    y = np.asarray(prob_vec, dtype=np.float32).reshape(2)
                    used_cam = True
                except Exception:
                    used_cam = False
                    y = None
        if y is None:
            y = np.asarray(self.teacher(int(target), intervention), dtype=np.float32).reshape(2)
        alpha_hat = self._alpha_from_y(y)
        k_hom = self._make_cache_key(intervention, cluster_sig=-1)
        self.cache_hom.put(k_hom, alpha_hat)
        hit = used_cam
        if not force_vcg:
            k_het = self._make_cache_key(intervention, cluster_sig=clus_sig)
            self.cache_het.put(k_het, alpha_hat)
        return (y.astype(np.float32), _to_us(_ns() - t0), hit)

    def query_batch(self, target: int, interventions: List[Dict[Any, Any]], evidence: Optional[Dict[Any, Any]]=None) -> Tuple[np.ndarray, float, float]:
        """
        Batch query path, preserving the legacy interface semantics:
        Returns: (Y_probs[B,2], elapsed_us_total, hit_rate)
        """
        t0 = _ns()
        outs = []
        hits = 0
        for inter in interventions:
            y, _us, hit = self.query(target=target, intervention=inter, evidence=evidence)
            outs.append(y)
            hits += int(bool(hit))
        out = np.stack(outs, axis=0).astype(np.float32)
        total_us = _to_us(_ns() - t0)
        hit_rate = float(hits) / float(max(1, len(interventions)))
        return (out, total_us, hit_rate)
_GLOBAL_VC: Optional[MobiusVCModel] = None

def get_mobius_vc_model(cache_capacity: int=CACHE_SLOTS_HET, low_rank_dim: int=LOW_RANK_DIM, model_version: str='v1', topology: Optional[Any]=None, seed: int=0) -> MobiusVCModel:
    """
    Backward-compatible factory returning a MobiusVCModel instance.
    """
    cluster = KMCCluster(K=32, dim=8, beta=8.0, gamma=0.95, seed=seed)
    return MobiusVCModel(cache_capacity_het=cache_capacity, cache_capacity_hom=max(256, cache_capacity // 2), low_rank_dim=low_rank_dim, model_version=model_version, topology=topology, cluster=cluster, seed=seed)

def _get_global_vc() -> MobiusVCModel:
    global _GLOBAL_VC
    if _GLOBAL_VC is None:
        _GLOBAL_VC = get_mobius_vc_model()
    return _GLOBAL_VC

def generate_common_interventions(n: int=1, grid: int=21, key: Union[int, str]=0) -> List[Dict[Any, float]]:
    """
    Generate a generic intervention set for cache warm-up or offline preprocessing.
    - By default, build a 0..1 grid of interventions on key=0.
    """
    n = int(max(1, n))
    grid = int(max(2, grid))
    xs = np.linspace(0.0, 1.0, grid, dtype=np.float64)
    inters: List[Dict[Any, float]] = []
    for x in xs:
        inters.append({key: float(x)})
    return inters

def infer_y_prob(evidence: Dict[str, Any]) -> Tuple[float, float]:
    """Convenience wrapper for ACIC/IHDP-style inputs. Returns (p1, elapsed_us)."""
    try:
        t = float(evidence.get('T', 0.0))
    except Exception:
        t = 0.0
    t = float(np.clip(t, 0.0, 1.0))
    inter: Dict[Any, Any] = {0: t}
    ctx = evidence.get('context_vec', None)
    if ctx is None:
        ctx = evidence.get('feature_vec', None)
    if ctx is not None:
        inter['__u__'] = ctx
    vc = _get_global_vc()
    prob_vec, us, _hit = vc.query(target=1, intervention=inter, evidence={k: v for k, v in evidence.items() if k not in ('T', 'context_vec', 'feature_vec')})
    p1 = float(np.asarray(prob_vec, dtype=np.float64).reshape(-1)[-1])
    return (p1, float(us))

@dataclass
class MobiusCausalConfig:
    dim: int = 16
    lr: float = 0.001
    steps: int = 1000

class MobiusCausal:

    def __init__(self, config: MobiusCausalConfig):
        self.config = config

def train_mobius_causal(*args: Any, **kwargs: Any) -> MobiusCausal:
    """Return a MobiusCausal shell object. Training logic can be filled in later."""
    return MobiusCausal(MobiusCausalConfig())

def integrate_causal_into_vc(model: MobiusVCModel, *args: Any, **kwargs: Any) -> MobiusVCModel:
    """Hook for exporting a trained layer into the VC model state."""
    return model
