"""Archived finite-register numerical kernels, with unchanged computational definitions."""
from __future__ import annotations

import os
import os, json, math, time, itertools, hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def now_stamp() -> str:
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime('%Y%m%d_%H%M%S', time.localtime(t)) + f'_{ms:03d}'

def sanitize_filename(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in ('-', '_', '.', '+'):
            out.append(ch)
        elif ch in (' ', '/', '\\', ':', '|'):
            out.append('_')
        else:
            out.append('_')
    s2 = ''.join(out)
    while '__' in s2:
        s2 = s2.replace('__', '_')
    return s2.strip('_')[:240]

def dagger(A: np.ndarray) -> np.ndarray:
    return np.conjugate(A.T)

def hermitize(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.complex128)
    return 0.5 * (A + dagger(A))

def is_massless(m: float, tol: float=1e-12) -> bool:
    return abs(float(m)) <= tol

def project_zero_mean(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v - float(np.mean(v))

def _ensure_density(rho: np.ndarray, eps: float=1e-15) -> np.ndarray:
    rho = hermitize(rho)
    tr = float(np.trace(rho).real)
    if tr <= eps:
        raise ValueError('Density has nonpositive trace.')
    return rho / tr

def matrix_log_psd(rho: np.ndarray, eps: float=1e-12) -> np.ndarray:
    rhoH = hermitize(rho)
    w, U = np.linalg.eigh(rhoH)
    w = np.clip(w.real, eps, None)
    return U * np.log(w) @ dagger(U)

def matrix_sqrt_psd(A: np.ndarray, eps: float=0.0) -> np.ndarray:
    AH = hermitize(A)
    w, U = np.linalg.eigh(AH)
    w = np.clip(w.real, eps, None)
    return U * np.sqrt(w) @ dagger(U)

def effect_from_sqrt(sF: np.ndarray) -> np.ndarray:
    sF = np.asarray(sF, dtype=np.complex128)
    return hermitize(dagger(sF) @ sF)

def expm_hermitian(H: np.ndarray, t: float, factor: complex=-1j) -> np.ndarray:
    Hh = hermitize(H)
    w, U = np.linalg.eigh(Hh)
    ph = np.exp(factor * float(t) * w.real)
    return U * ph @ dagger(U)

def pdf_text_page(pdf: PdfPages, title: str, lines: List[str], dpi: int=200) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(title, fontsize=12, y=0.99)
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.02, 0.98, '\n'.join(lines), va='top', ha='left', family='monospace', fontsize=9)
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)

def save_npz_json(out_prefix: str, label: str, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> Tuple[str, str]:
    ts = now_stamp()
    base = sanitize_filename(f'{out_prefix}_{label}_{ts}')
    npz_path = os.path.abspath(base + '.npz')
    json_path = os.path.abspath(base + '.json')
    np.savez_compressed(npz_path, **arrays)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return (npz_path, json_path)

def stable_hash_int(*items: Any, mod: int=2 ** 31 - 1) -> int:
    h = hashlib.sha256()
    for it in items:
        h.update(str(it).encode('utf-8'))
        h.update(b'|')
    return int(h.hexdigest(), 16) % int(mod)

def estimate_joint_state_bytes(N: int, M: int, S: int, geometry_quantum: bool, dtype_bytes: int=16) -> int:
    N = int(N)
    M = int(M)
    S = int(S)
    d = 1 << N
    if geometry_quantum:
        elements = S * M * M * d * d
    else:
        elements = S * M * d * d
    return int(elements) * int(dtype_bytes)

def _bytes_to_gib(x: int) -> float:
    return float(x) / float(1024 ** 3)

def _sigmoid_real(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))

def op_from_eigh(U: np.ndarray, vals: np.ndarray) -> np.ndarray:
    return hermitize(U * vals @ dagger(U))

def op_sigmoid_from_eigh(U: np.ndarray, evals: np.ndarray, kappa: float, sgn: int) -> Tuple[np.ndarray, np.ndarray]:
    x = _sigmoid_real(float(kappa) * float(sgn) * np.asarray(evals, dtype=float))
    F = op_from_eigh(U, x)
    sF = op_from_eigh(U, np.sqrt(np.clip(x, 0.0, None)))
    return (F, sF)
I2 = np.eye(2, dtype=np.complex128)
X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

def tensor_kets(kets: List[np.ndarray]) -> np.ndarray:
    out = np.array([1.0 + 0j], dtype=np.complex128)
    for k in kets:
        out = np.kron(out, np.asarray(k, dtype=np.complex128))
    return out

def build_local_paulis(N: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    N = int(N)
    X_ops, Y_ops, Z_ops = ([], [], [])
    for i in range(N):
        opx = np.array([1.0 + 0j], dtype=np.complex128)
        opy = np.array([1.0 + 0j], dtype=np.complex128)
        opz = np.array([1.0 + 0j], dtype=np.complex128)
        for j in range(N):
            opx = np.kron(opx, X2 if j == i else I2)
            opy = np.kron(opy, Y2 if j == i else I2)
            opz = np.kron(opz, Z2 if j == i else I2)
        X_ops.append(opx)
        Y_ops.append(opy)
        Z_ops.append(opz)
    return (X_ops, Y_ops, Z_ops)

def partial_trace_qubits(rho: np.ndarray, keep: Sequence[int], N: int) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.complex128)
    N = int(N)
    keep = sorted([int(k) for k in keep])
    if any((k < 0 or k >= N for k in keep)):
        raise ValueError('partial_trace_qubits: invalid keep index.')
    trace_out = [i for i in range(N) if i not in keep]
    if not trace_out:
        return rho.copy()
    dims = [2] * N
    T = rho.reshape(dims + dims)
    for q in sorted(trace_out, reverse=True):
        T = np.trace(T, axis1=q, axis2=q + len(dims))
        dims.pop(q)
    d_keep = 2 ** len(keep)
    return T.reshape(d_keep, d_keep)

def von_neumann_entropy(rho: np.ndarray, eps: float=1e-12) -> float:
    rhoH = hermitize(rho)
    w = np.linalg.eigvalsh(rhoH).real
    w = np.clip(w, 0.0, None)
    s = float(np.sum(w))
    if s <= 0:
        return 0.0
    w = w / s
    w = np.clip(w, eps, None)
    return float(-np.sum(w * np.log(w)))

def relative_entropy(rho: np.ndarray, sigma: np.ndarray, eps: float=1e-12) -> float:
    rhoH = _ensure_density(rho)
    sigH = _ensure_density(sigma)
    return float(np.trace(rhoH @ (matrix_log_psd(rhoH, eps) - matrix_log_psd(sigH, eps))).real)

def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    D = hermitize(rho) - hermitize(sigma)
    w = np.linalg.eigvalsh(D).real
    return float(0.5 * np.sum(np.abs(w)))

def negativity_two_qubits(rho_2q: np.ndarray) -> float:
    rho = hermitize(rho_2q)
    rho4 = rho.reshape(2, 2, 2, 2)
    rho_pt = rho4.transpose(0, 3, 2, 1).reshape(4, 4)
    w = np.linalg.eigvalsh(hermitize(rho_pt)).real
    return float(0.5 * (np.sum(np.abs(w)) - 1.0))

def negativity_bipartition_qubits(rho: np.ndarray, A: Sequence[int], N: int) -> float:
    rho = hermitize(np.asarray(rho, dtype=np.complex128))
    N = int(N)
    A = sorted(set((int(x) for x in A)))
    if not A or len(A) >= N:
        return 0.0
    B = [i for i in range(N) if i not in A]
    dA = 2 ** len(A)
    dB = 2 ** len(B)
    perm = basis_perm_from_qubit_order(N, A + B)
    rho_perm = rho[np.ix_(perm, perm)]
    rho4 = rho_perm.reshape(dA, dB, dA, dB)
    rho_pt = rho4.transpose(2, 1, 0, 3).reshape(dA * dB, dA * dB)
    w = np.linalg.eigvalsh(hermitize(rho_pt)).real
    return float(0.5 * (np.sum(np.abs(w)) - 1.0))

def log_negativity_bipartition_qubits(rho: np.ndarray, A: Sequence[int], N: int) -> float:
    neg = float(negativity_bipartition_qubits(rho, A, N))
    return float(math.log2(2.0 * neg + 1.0)) if neg > 0.0 else 0.0

def l1_coherence_computational_basis(rho: np.ndarray) -> float:
    rho = np.asarray(rho, dtype=np.complex128)
    off = rho - np.diag(np.diag(rho))
    return float(np.sum(np.abs(off)))

def purity_trace(rho: np.ndarray) -> float:
    rhoH = hermitize(np.asarray(rho, dtype=np.complex128))
    return float(np.real(np.trace(rhoH @ rhoH)))

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float=1e-15) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, None)
    q = np.clip(np.asarray(q, dtype=float), eps, None)
    p = p / float(np.sum(p))
    q = q / float(np.sum(q))
    return float(np.sum(p * np.log(p / q)))

def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(float(np.sum(p)), 1e-15)
    q = q / max(float(np.sum(q)), 1e-15)
    return float(0.5 * np.sum(np.abs(p - q)))

def _log_mean_vec(Wi: np.ndarray, Wj: np.ndarray, eps: float=1e-15) -> np.ndarray:
    Wi = np.clip(np.asarray(Wi, dtype=float), eps, None)
    Wj = np.clip(np.asarray(Wj, dtype=float), eps, None)
    Wi, Wj = np.broadcast_arrays(Wi, Wj)
    num = Wi - Wj
    den = np.log(Wi) - np.log(Wj)
    scale = np.maximum(np.maximum(Wi, Wj), 1.0)
    close = np.abs(num) <= 1e-14 * scale
    out = np.empty_like(num)
    out[close] = 0.5 * (Wi[close] + Wj[close])
    out[~close] = num[~close] / den[~close]
    return np.clip(out, eps, None)

def bkm_sld_classical_quadratic_from_ref(ref_rho: np.ndarray, Delta: np.ndarray, eps: float=1e-12) -> Dict[str, float]:
    ref = _ensure_density(ref_rho)
    w, U = np.linalg.eigh(hermitize(ref))
    w = np.clip(w.real, eps, None)
    D = dagger(U) @ hermitize(Delta) @ U
    abs2 = (D.conjugate() * D).real
    Wi = w[:, None]
    Wj = w[None, :]
    LM = _log_mean_vec(Wi, Wj, eps=eps)
    g_bkm = float(np.sum(abs2 / LM))
    denom = np.clip(Wi + Wj, eps, None)
    g_sld = float(np.sum(2.0 * abs2 / denom))
    diag = np.real(np.diag(D))
    g_cls = float(np.sum(diag * diag / w))
    return {'gBKM': g_bkm, 'CE_BKM_quad': 0.5 * g_bkm, 'FI_SLD': g_sld, 'FI_classical_eigbasis': g_cls}

def canonical_energy_metrics(rho: np.ndarray, rho_ref: np.ndarray, eps: float=1e-12) -> Dict[str, float]:
    rhoH = _ensure_density(rho)
    refH = _ensure_density(rho_ref)
    w, U = np.linalg.eigh(refH)
    w = np.clip(w.real, eps, None)
    D = dagger(U) @ (rhoH - refH) @ U
    abs2 = (D.conjugate() * D).real
    Wi = w[:, None]
    Wj = w[None, :]
    LM = _log_mean_vec(Wi, Wj, eps=eps)
    g_bkm = float(np.sum(abs2 / LM))
    denom = np.clip(Wi + Wj, eps, None)
    g_sld = float(np.sum(2.0 * abs2 / denom))
    diag = np.real(np.diag(D))
    g_cls = float(np.sum(diag * diag / w))
    return {'Srel': float(relative_entropy(rhoH, refH, eps)), 'CE_BKM_quad': float(0.5 * g_bkm), 'FI_SLD': float(g_sld), 'FI_classical_eigbasis': float(g_cls)}

@dataclass
class GeometryGraph:
    name: str
    adjacency: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Node:
    name: str
    curvature_ref: float
    coord: Optional[np.ndarray] = None
    frame_su2: Optional[np.ndarray] = None

@dataclass
class TriangulatedSurface:
    points: np.ndarray
    triangles: np.ndarray
    name: str
    closed: bool = True

    def to_graph(self) -> GeometryGraph:
        P = np.asarray(self.points, dtype=float)
        T = np.asarray(self.triangles, dtype=np.int64)
        V = int(P.shape[0])
        A = np.zeros((V, V), dtype=float)
        for i, j, k in T:
            for a, b in ((i, j), (j, k), (k, i)):
                A[a, b] = 1.0
                A[b, a] = 1.0
        return GeometryGraph(name=f'{self.name}_graph', adjacency=A, meta={'V': V, 'T': int(T.shape[0]), 'closed': bool(self.closed)})

    def curvature_scalar(self) -> np.ndarray:
        P = np.asarray(self.points, dtype=float)
        T = np.asarray(self.triangles, dtype=np.int64)
        V = int(P.shape[0])
        angle_sum = np.zeros(V, dtype=float)

        def ang(u, v) -> float:
            nu = float(np.linalg.norm(u))
            nv = float(np.linalg.norm(v))
            if nu <= 1e-15 or nv <= 1e-15:
                return 0.0
            c = float(np.dot(u, v) / (nu * nv))
            c = max(min(c, 1.0), -1.0)
            return float(math.acos(c))
        for i, j, k in T:
            pi, pj, pk = (P[i], P[j], P[k])
            angle_sum[i] += ang(pj - pi, pk - pi)
            angle_sum[j] += ang(pi - pj, pk - pj)
            angle_sum[k] += ang(pi - pk, pj - pk)
        target = 2.0 * math.pi if self.closed else math.pi
        return (target - angle_sum).astype(float)

    def mean_edge_length(self) -> float:
        P = np.asarray(self.points, dtype=float)
        T = np.asarray(self.triangles, dtype=np.int64)
        edges = set()
        for i, j, k in T:
            for a, b in ((i, j), (j, k), (k, i)):
                aa, bb = (a, b) if a < b else (b, a)
                edges.add((aa, bb))
        if not edges:
            return 0.0
        return float(np.mean([np.linalg.norm(P[a] - P[b]) for a, b in edges]))

    def refine_midpoint(self, name_suffix: str='_ref') -> 'TriangulatedSurface':
        P = np.asarray(self.points, dtype=float)
        T = np.asarray(self.triangles, dtype=np.int64)
        V0 = int(P.shape[0])
        edge_mid: Dict[Tuple[int, int], int] = {}
        new_pts = [P[i].copy() for i in range(V0)]
        new_tris: List[List[int]] = []

        def mid(i: int, j: int) -> int:
            a, b = (i, j) if i < j else (j, i)
            key = (a, b)
            if key in edge_mid:
                return edge_mid[key]
            idx = len(new_pts)
            new_pts.append(0.5 * (P[a] + P[b]))
            edge_mid[key] = idx
            return idx
        for i, j, k in T:
            m_ij = mid(i, j)
            m_jk = mid(j, k)
            m_ki = mid(k, i)
            new_tris += [[i, m_ij, m_ki], [j, m_jk, m_ij], [k, m_ki, m_jk], [m_ij, m_jk, m_ki]]
        return TriangulatedSurface(points=np.array(new_pts, float), triangles=np.array(new_tris, np.int64), name=f'{self.name}{name_suffix}', closed=self.closed)

def surface_euler_characteristic(surface: TriangulatedSurface) -> Dict[str, int]:
    P = np.asarray(surface.points, float)
    T = np.asarray(surface.triangles, np.int64)
    V = int(P.shape[0])
    F = int(T.shape[0])
    edges = set()
    for i, j, k in T:
        for a, b in ((i, j), (j, k), (k, i)):
            aa, bb = (int(a), int(b))
            if aa > bb:
                aa, bb = (bb, aa)
            edges.add((aa, bb))
    E = int(len(edges))
    chi = V - E + F
    return {'V': V, 'E': E, 'F': F, 'chi': int(chi)}

def vertex_areas_barycentric(surface: TriangulatedSurface) -> np.ndarray:
    P = np.asarray(surface.points, float)
    T = np.asarray(surface.triangles, np.int64)
    V = int(P.shape[0])
    A_v = np.zeros(V, float)
    for i, j, k in T:
        pi, pj, pk = (P[i], P[j], P[k])
        area = 0.5 * float(np.linalg.norm(np.cross(pj - pi, pk - pi)))
        A_v[i] += area / 3.0
        A_v[j] += area / 3.0
        A_v[k] += area / 3.0
    return A_v

def gauss_bonnet_stats(surface: TriangulatedSurface, expected_total: Optional[float]=None) -> Dict[str, Any]:
    delta = np.asarray(surface.curvature_scalar(), float)
    total = float(np.sum(delta))
    topo = surface_euler_characteristic(surface)
    expected = float(2.0 * math.pi * topo['chi']) if expected_total is None else float(expected_total)
    abs_err = float(total - expected)
    rel_err = float(abs_err / max(abs(expected), 1e-12))
    A_v = vertex_areas_barycentric(surface)
    Kdens = np.zeros_like(delta)
    mask = A_v > 1e-15
    Kdens[mask] = delta[mask] / A_v[mask]
    return {'chi': topo['chi'], 'V': topo['V'], 'E': topo['E'], 'F': topo['F'], 'total_deficit': total, 'expected_total': expected, 'abs_error': abs_err, 'rel_error': rel_err, 'integral_KdA': float(np.sum(Kdens * A_v))}

def make_irregular_octahedron_surface(seed: int=3, perturb: float=0.2, scale_xyz: Tuple[float, float, float]=(1.0, 1.2, 0.85)) -> TriangulatedSurface:
    rng = np.random.default_rng(int(seed))
    P = np.array([[+1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, +1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, +1.0], [0.0, 0.0, -1.0]], dtype=float)
    S = np.diag(np.array(scale_xyz, dtype=float))
    P = P @ S.T + perturb * rng.normal(size=P.shape)
    T = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], dtype=np.int64)
    return TriangulatedSurface(points=P, triangles=T, name=f'irregular_octa_seed{seed}', closed=True)

def make_irregular_tetrahedron_surface(seed: int=7, perturb: float=0.18, scale_xyz: Tuple[float, float, float]=(1.0, 1.0, 1.0)) -> TriangulatedSurface:
    rng = np.random.default_rng(int(seed))
    P = np.array([[+1.0, +1.0, +1.0], [+1.0, -1.0, -1.0], [-1.0, +1.0, -1.0], [-1.0, -1.0, +1.0]], dtype=float)
    S = np.diag(np.array(scale_xyz, dtype=float))
    P = P @ S.T + perturb * rng.normal(size=P.shape)
    T = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int64)
    return TriangulatedSurface(points=P, triangles=T, name=f'irregular_tet_seed{seed}', closed=True)

def make_irregular_cube_surface(seed: int=11, perturb: float=0.12, scale_xyz: Tuple[float, float, float]=(1.0, 1.08, 0.95)) -> TriangulatedSurface:
    rng = np.random.default_rng(int(seed))
    P = np.array([[-1.0, -1.0, -1.0], [+1.0, -1.0, -1.0], [+1.0, +1.0, -1.0], [-1.0, +1.0, -1.0], [-1.0, -1.0, +1.0], [+1.0, -1.0, +1.0], [+1.0, +1.0, +1.0], [-1.0, +1.0, +1.0]], dtype=float)
    S = np.diag(np.array(scale_xyz, dtype=float))
    P = P @ S.T + perturb * rng.normal(size=P.shape)
    T = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 5, 1], [0, 4, 5], [3, 2, 6], [3, 6, 7], [0, 3, 7], [0, 7, 4], [1, 6, 2], [1, 5, 6]], dtype=np.int64)
    return TriangulatedSurface(points=P, triangles=T, name=f'irregular_cube_seed{seed}', closed=True)

def surface_to_nodes_graph(surface: TriangulatedSurface, name_prefix: str='v', K_ref_override: Optional[np.ndarray]=None, compute_frames: bool=True, frame_seed: int=0) -> Tuple[List[Node], GeometryGraph, np.ndarray]:
    if K_ref_override is None:
        K = np.asarray(surface.curvature_scalar(), dtype=float)
    else:
        K = np.asarray(K_ref_override, dtype=float).copy()
        if K.shape[0] != surface.points.shape[0]:
            raise ValueError('K_ref_override length mismatch with number of vertices.')
    frames = surface_vertex_frames_su2(surface, seed=frame_seed) if compute_frames else [None] * int(surface.points.shape[0])
    nodes = [Node(name=f'{name_prefix}{i}', curvature_ref=float(K[i]), coord=np.asarray(surface.points[i], dtype=float).copy(), frame_su2=frames[i]) for i in range(int(surface.points.shape[0]))]
    graph = surface.to_graph()
    return (nodes, graph, K)

def surface_edges(tris: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    tris = np.asarray(tris, dtype=np.int64)
    e2t: Dict[Tuple[int, int], List[int]] = {}
    for t_idx, (i, j, k) in enumerate(tris):
        for a, b in ((i, j), (j, k), (k, i)):
            aa, bb = (a, b) if a < b else (b, a)
            e2t.setdefault((aa, bb), []).append(int(t_idx))
    return e2t

def flippable_edges_2_2(surface: TriangulatedSurface) -> List[Tuple[int, int]]:
    T = np.asarray(surface.triangles, dtype=np.int64)
    e2t = surface_edges(T)
    existing = set(e2t.keys())
    cand = []
    for (a, b), tris_idx in e2t.items():
        if len(tris_idx) != 2:
            continue
        t1, t2 = tris_idx
        tri1 = set(T[t1].tolist())
        tri2 = set(T[t2].tolist())
        k = list(tri1 - {a, b})
        l = list(tri2 - {a, b})
        if len(k) != 1 or len(l) != 1:
            continue
        k = int(k[0])
        l = int(l[0])
        if k == l:
            continue
        kk, ll = (k, l) if k < l else (l, k)
        if (kk, ll) in existing:
            continue
        cand.append((a, b))
    return cand

def pachner_flip_2_2(surface: TriangulatedSurface, edge: Tuple[int, int]) -> TriangulatedSurface:
    P = np.asarray(surface.points, float)
    T = np.asarray(surface.triangles, np.int64).copy()
    a, b = (int(edge[0]), int(edge[1]))
    aa, bb = (a, b) if a < b else (b, a)
    e2t = surface_edges(T)
    if (aa, bb) not in e2t or len(e2t[aa, bb]) != 2:
        raise ValueError('Edge is not an interior (2-2) flippable edge.')
    t1, t2 = e2t[aa, bb]
    tri1 = set(T[t1].tolist())
    tri2 = set(T[t2].tolist())
    k = int(list(tri1 - {aa, bb})[0])
    l = int(list(tri2 - {aa, bb})[0])
    T_new = T.tolist()
    T_new[t1] = [k, l, aa]
    T_new[t2] = [k, l, bb]
    return TriangulatedSurface(points=P.copy(), triangles=np.array(T_new, np.int64), name=surface.name + '_pachner22', closed=surface.closed)

def edge_length_map(surface: TriangulatedSurface) -> Dict[Tuple[int, int], float]:
    P = np.asarray(surface.points, float)
    T = np.asarray(surface.triangles, np.int64)
    e2t = surface_edges(T)
    out: Dict[Tuple[int, int], float] = {}
    for a, b in e2t.keys():
        out[a, b] = float(np.linalg.norm(P[a] - P[b]))
    return out

def build_balls(graph: GeometryGraph, radius_hops: int) -> List[List[int]]:
    A = np.asarray(graph.adjacency, dtype=float) > 0.0
    N = int(A.shape[0])
    r = int(radius_hops)
    balls: List[List[int]] = []
    for v in range(N):
        seen = {v}
        frontier = {v}
        for _ in range(r):
            nxt = set()
            for u in frontier:
                nxt |= set(np.where(A[u])[0].tolist())
            nxt -= seen
            seen |= nxt
            frontier = nxt
        balls.append(sorted(list(seen)))
    return balls

def build_geometry_ensemble(K_ref: np.ndarray, levels_per_node: int, delta_scale: float, max_configs: int, seed: int, enforce_total_curvature: bool=False, total_curvature_target: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K_ref = np.asarray(K_ref, dtype=float)
    N = int(K_ref.shape[0])
    L = int(levels_per_node)
    rng = np.random.default_rng(int(seed))
    idx = np.arange(L, dtype=float)
    center = 0.5 * (L - 1)
    level_offsets = (idx - center) * float(delta_scale)
    ref_level = int(round(center))
    ref_code = np.full(N, ref_level, dtype=np.int64)
    codes: List[np.ndarray] = [ref_code.copy()]
    seen = {tuple(ref_code.tolist())}
    while len(codes) < int(max_configs):
        base = codes[int(rng.integers(0, len(codes)))]
        c = base.copy()
        v = int(rng.integers(0, N))
        step = int(rng.choice([-1, +1]))
        c[v] = int(c[v] + step)
        if c[v] < 0 or c[v] >= L:
            continue
        t = tuple(c.tolist())
        if t in seen:
            continue
        seen.add(t)
        codes.append(c)
    codes_arr = np.stack(codes, axis=0)
    K_cfg = np.array([K_ref + level_offsets[codes_arr[a]] for a in range(codes_arr.shape[0])], dtype=float)
    if enforce_total_curvature:
        target = float(np.sum(K_ref)) if total_curvature_target is None else float(total_curvature_target)
        drift = (np.sum(K_cfg, axis=1) - target) / max(float(N), 1.0)
        K_cfg = K_cfg - drift[:, None]
    return (K_cfg, codes_arr, level_offsets)

def _diff_vertex_and_sign(code_a: np.ndarray, code_b: np.ndarray) -> Optional[Tuple[int, int]]:
    d = code_a - code_b
    idx = np.where(d != 0)[0]
    if idx.size != 1:
        return None
    v = int(idx[0])
    if abs(int(d[v])) != 1:
        return None
    return (v, int(np.sign(int(d[v]))))

def build_config_graph(codes: np.ndarray) -> Tuple[np.ndarray, Dict[Tuple[int, int], Tuple[int, int]]]:
    codes = np.asarray(codes, dtype=np.int64)
    M, _N = codes.shape
    A = np.zeros((M, M), dtype=float)
    edge_info: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for a in range(M):
        for b in range(a + 1, M):
            ds = _diff_vertex_and_sign(codes[a], codes[b])
            if ds is None:
                continue
            A[a, b] = 1.0
            A[b, a] = 1.0
            v, sgn = ds
            edge_info[a, b] = (v, sgn)
            edge_info[b, a] = (v, -sgn)
    return (A, edge_info)

def build_response_G(graph: GeometryGraph, m: float, h: float) -> np.ndarray:
    A = np.asarray(graph.adjacency, dtype=float)
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return 1.0 / max(float(h * h), 1e-15) * L + float(m) * float(m) * np.eye(A.shape[0], dtype=float)

def slice_F(k: np.ndarray, J: np.ndarray, G: np.ndarray, alpha: float) -> float:
    k = np.asarray(k, dtype=float)
    J = np.asarray(J, dtype=float)
    G = np.asarray(G, dtype=float)
    return 0.5 * float(alpha) * float(k @ (G @ k)) + float(J @ k)

def gibbs_pi(F: np.ndarray, beta: float) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    beta = float(beta)
    x = -beta * (F - float(np.min(F)))
    x = np.clip(x, -700.0, 700.0)
    w = np.exp(x)
    w = np.clip(w, 0.0, None)
    return w / max(float(np.sum(w)), 1e-15)

def metropolis_rates(F: np.ndarray, A_cfg: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    A_cfg = np.asarray(A_cfg, dtype=float)
    M = int(F.shape[0])
    beta = float(beta)
    gamma = float(gamma)
    R = np.zeros((M, M), dtype=float)
    for a in range(M):
        for b in range(M):
            if a == b or A_cfg[a, b] <= 0.0:
                continue
            dF = float(F[a] - F[b])
            R[a, b] = gamma * math.exp(-0.5 * beta * dF)
    return R

def basis_perm_from_qubit_order(N: int, new_order: Sequence[int]) -> np.ndarray:
    N = int(N)
    new_order = [int(x) for x in new_order]
    if sorted(new_order) != list(range(N)):
        raise ValueError('new_order must be a permutation of 0..N-1')
    d = 1 << N
    idx = np.arange(d, dtype=np.uint64)
    bits = [(idx >> np.uint64(N - 1 - q) & np.uint64(1)).astype(np.uint64) for q in range(N)]
    new_idx = np.zeros(d, dtype=np.uint64)
    for q in new_order:
        new_idx = new_idx << np.uint64(1) | bits[q]
    return new_idx.astype(np.int64)

def embed_operator(op_sub: np.ndarray, keep: Sequence[int], N: int, *, hermitize_output: bool=True) -> np.ndarray:
    op_sub = np.asarray(op_sub, dtype=np.complex128)
    keep = sorted([int(k) for k in keep])
    N = int(N)
    k = len(keep)
    if k == 0:
        return np.eye(2 ** N, dtype=np.complex128)
    if op_sub.shape != (2 ** k, 2 ** k):
        raise ValueError('embed_operator: op_sub dimension mismatch.')
    rest = [i for i in range(N) if i not in keep]
    perm = basis_perm_from_qubit_order(N, keep + rest)
    op_perm = np.kron(op_sub, np.eye(2 ** (N - k), dtype=np.complex128))
    out = op_perm[np.ix_(perm, perm)]
    return hermitize(out) if hermitize_output else out

def inv_perm_from_swaps(N: int, swaps: Sequence[Tuple[int, int]]) -> np.ndarray:
    N = int(N)
    d = 1 << N
    x = np.arange(d, dtype=np.uint64)
    for i, j in swaps:
        i = int(i)
        j = int(j)
        if i == j:
            continue
        si = N - 1 - i
        sj = N - 1 - j
        bi = x >> np.uint64(si) & np.uint64(1)
        bj = x >> np.uint64(sj) & np.uint64(1)
        m = (bi ^ bj).astype(np.uint64)
        x = x ^ m << np.uint64(si) ^ m << np.uint64(sj)
    perm = x.astype(np.int64)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(d, dtype=np.int64)
    return inv

def su2_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(axis))
    if n <= 1e-15:
        return np.eye(2, dtype=np.complex128)
    axis = axis / n
    ax, ay, az = (float(axis[0]), float(axis[1]), float(axis[2]))
    a = float(angle) * 0.5
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[c - 1j * s * az, -1j * s * ax - s * ay], [-1j * s * ax + s * ay, c + 1j * s * az]], dtype=np.complex128)

def frame_from_coord(coord: np.ndarray, seed: int=0) -> np.ndarray:
    c = np.asarray(coord, dtype=float).reshape(-1)
    s = stable_hash_int(tuple(np.round(c, 8)), seed, mod=10000000)
    rng = np.random.default_rng(int(s))
    axis = c.copy()
    if float(np.linalg.norm(axis)) <= 1e-12:
        axis = rng.normal(size=3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-15)
    ang = 0.7 * float(np.linalg.norm(c)) + 0.35 * float(rng.uniform(-math.pi, math.pi))
    return su2_from_axis_angle(axis, ang)

def _normalize3(v: np.ndarray, eps: float=1e-15) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros(3, dtype=float)
    return v / n

def _so3_project(R: np.ndarray, eps: float=1e-15) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    if float(np.linalg.det(Rp)) < 0.0:
        U[:, -1] *= -1.0
        Rp = U @ Vt
    return Rp

def quaternion_from_so3(R: np.ndarray, eps: float=1e-15) -> Tuple[float, float, float, float]:
    R = _so3_project(R, eps=eps)
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = float(R[2, 1] - R[1, 2]) / s
        y = float(R[0, 2] - R[2, 0]) / s
        z = float(R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0)) * 2.0
        w = float(R[2, 1] - R[1, 2]) / max(s, eps)
        x = 0.25 * s
        y = float(R[0, 1] + R[1, 0]) / max(s, eps)
        z = float(R[0, 2] + R[2, 0]) / max(s, eps)
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0)) * 2.0
        w = float(R[0, 2] - R[2, 0]) / max(s, eps)
        x = float(R[0, 1] + R[1, 0]) / max(s, eps)
        y = 0.25 * s
        z = float(R[1, 2] + R[2, 1]) / max(s, eps)
    else:
        s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0)) * 2.0
        w = float(R[1, 0] - R[0, 1]) / max(s, eps)
        x = float(R[0, 2] + R[2, 0]) / max(s, eps)
        y = float(R[1, 2] + R[2, 1]) / max(s, eps)
        z = 0.25 * s
    nrm = math.sqrt(w * w + x * x + y * y + z * z)
    if nrm <= eps:
        return (1.0, 0.0, 0.0, 0.0)
    w, x, y, z = (w / nrm, x / nrm, y / nrm, z / nrm)
    if w < 0.0:
        w, x, y, z = (-w, -x, -y, -z)
    return (float(w), float(x), float(y), float(z))

def su2_from_so3(R: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_from_so3(R)
    return np.array([[w - 1j * z, -1j * x - y], [-1j * x + y, w + 1j * z]], dtype=np.complex128)

def surface_vertex_frames_su2(surface: TriangulatedSurface, seed: int=0) -> List[np.ndarray]:
    P = np.asarray(surface.points, dtype=float)
    T = np.asarray(surface.triangles, dtype=np.int64)
    V = int(P.shape[0])
    rng = np.random.default_rng(int(seed))
    neigh: List[set] = [set() for _ in range(V)]
    n_acc = np.zeros((V, 3), dtype=float)
    for i, j, k in T:
        i = int(i)
        j = int(j)
        k = int(k)
        neigh[i].update((j, k))
        neigh[j].update((i, k))
        neigh[k].update((i, j))
        e1 = P[j] - P[i]
        e2 = P[k] - P[i]
        n = np.cross(e1, e2)
        n_acc[i] += n
        n_acc[j] += n
        n_acc[k] += n
    frames: List[np.ndarray] = []
    for i in range(V):
        n = _normalize3(n_acc[i])
        if float(np.linalg.norm(n)) <= 1e-12:
            axis = rng.normal(size=3)
            n = _normalize3(axis)
        neigh_i = sorted(list(neigh[i]))
        t1 = np.zeros(3, dtype=float)
        for j in neigh_i:
            v = P[int(j)] - P[i]
            v = v - float(np.dot(v, n)) * n
            if float(np.linalg.norm(v)) > 1e-12:
                t1 = _normalize3(v)
                break
        if float(np.linalg.norm(t1)) <= 1e-12:
            v = rng.normal(size=3)
            v = v - float(np.dot(v, n)) * n
            t1 = _normalize3(v)
        t2 = _normalize3(np.cross(n, t1))
        t1 = _normalize3(np.cross(t2, n))
        R = np.stack([t1, t2, n], axis=1)
        U = su2_from_so3(R)
        frames.append(U)
    return frames

@dataclass
class EdgeChannel:
    i: int
    j: int
    p_swap: float
    inv_perm: np.ndarray
    Ui_full: Optional[np.ndarray] = None
    Uj_full: Optional[np.ndarray] = None

    def apply(self, rho: np.ndarray) -> np.ndarray:
        p = float(np.clip(self.p_swap, 0.0, 1.0))
        if p == 0.0:
            return rho
        sw = rho[np.ix_(self.inv_perm, self.inv_perm)]
        if self.Ui_full is not None and self.Uj_full is not None:
            tmp = self.Ui_full @ sw @ dagger(self.Ui_full)
            tmp = self.Uj_full @ tmp @ dagger(self.Uj_full)
        else:
            tmp = sw
        return (1.0 - p) * rho + p * tmp

@dataclass
class SpacelikeStep:
    edges: List[EdgeChannel]

    def apply(self, rho: np.ndarray, order: Optional[Sequence[int]]=None) -> np.ndarray:
        out = rho
        if order is None:
            order = range(len(self.edges))
        for k in order:
            out = self.edges[int(k)].apply(out)
        return out

@dataclass
class CausalStructure:
    N: int
    steps: List[SpacelikeStep]
    edge_list: List[EdgeChannel]
    meta: Dict[str, Any] = field(default_factory=dict)

    def apply_global(self, rho: np.ndarray) -> np.ndarray:
        out = rho
        for st in self.steps:
            out = st.apply(out)
        return out

    def order_independence_error(self, rho: np.ndarray, step_idx: int=0) -> float:
        if not self.steps:
            return 0.0
        st = self.steps[int(step_idx) % len(self.steps)]
        if len(st.edges) <= 1:
            return 0.0
        o1 = list(range(len(st.edges)))
        o2 = list(reversed(o1))
        r1 = st.apply(rho, o1)
        r2 = st.apply(rho, o2)
        return float(np.linalg.norm(r1 - r2))

    def no_signaling_error_for_edge(self, rho: np.ndarray, edge_idx: int=0) -> float:
        if not self.edge_list:
            return 0.0
        e = self.edge_list[int(edge_idx) % len(self.edge_list)]
        candidates = [q for q in range(self.N) if q not in (e.i, e.j)]
        if not candidates:
            return 0.0
        probe = candidates[0]
        rho0 = partial_trace_qubits(rho, keep=[probe], N=self.N)
        rho1 = partial_trace_qubits(e.apply(rho), keep=[probe], N=self.N)
        return float(np.linalg.norm(rho1 - rho0))

    def no_signaling_region_test(self, step_idx: int=0, region_A: Sequence[int]=(0,), region_B: Sequence[int]=(1,), n_trials: int=20, seed: int=0) -> Dict[str, Any]:
        rng = np.random.default_rng(int(seed))
        A = sorted(set((int(x) for x in region_A)))
        B = sorted(set((int(x) for x in region_B)))
        if any((x < 0 or x >= self.N for x in A + B)):
            raise ValueError('no_signaling_region_test: invalid region index.')
        if set(A) & set(B):
            raise ValueError('no_signaling_region_test: regions A and B must be disjoint.')
        if not self.steps:
            return {'ok': False, 'reason': 'no steps in causal structure'}
        st = self.steps[int(step_idx) % len(self.steps)]
        d = 2 ** self.N

        def random_unitary(dim: int) -> np.ndarray:
            X = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
            Q, R = np.linalg.qr(X)
            ph = np.diag(R)
            ph = ph / np.maximum(np.abs(ph), 1e-15)
            return Q * ph
        UA = random_unitary(2 ** len(A))
        UA_full = embed_operator(UA, keep=A, N=self.N, hermitize_output=False)
        errs = []
        for _ in range(int(n_trials)):
            rho = random_density_matrix(d, rng=rng, rank=min(d, 8))
            out0 = st.apply(rho)
            out1 = st.apply(UA_full @ rho @ dagger(UA_full))
            rhoB0 = partial_trace_qubits(out0, keep=B, N=self.N)
            rhoB1 = partial_trace_qubits(out1, keep=B, N=self.N)
            errs.append(float(np.linalg.norm(rhoB1 - rhoB0)))
        return {'ok': True, 'step_idx': int(step_idx), 'region_A': A, 'region_B': B, 'n_trials': int(n_trials), 'mean_fro_err': float(np.mean(errs)) if errs else 0.0, 'max_fro_err': float(np.max(errs)) if errs else 0.0}

def build_causal_structure(graph: GeometryGraph, p_swap: Union[float, Dict[Tuple[int, int], float]]=0.35, seed: int=0, gr_transport: bool=False, node_coords: Optional[List[np.ndarray]]=None, node_frames: Optional[List[np.ndarray]]=None, frame_seed: int=0) -> Tuple[CausalStructure, Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    A = np.asarray(graph.adjacency, dtype=float)
    N = int(A.shape[0])
    edges = [(i, j) for i in range(N) for j in range(i + 1, N) if A[i, j] > 0.0]
    gr_data: Dict[str, Any] = {}
    frames: Optional[List[np.ndarray]] = None
    W_edge: Dict[Tuple[int, int], np.ndarray] = {}
    if gr_transport:
        if node_frames is not None:
            if len(node_frames) != N:
                raise ValueError('gr_transport=True requires node_frames of length N.')
            frames = [np.asarray(node_frames[i], dtype=np.complex128) for i in range(N)]
        else:
            if node_coords is None or len(node_coords) != N:
                raise ValueError('gr_transport=True requires node_coords (fallback) or node_frames.')
            frames = [frame_from_coord(node_coords[i], seed=frame_seed) for i in range(N)]
        invs = []
        for i, j in edges:
            Wij = frames[j] @ dagger(frames[i])
            W_edge[i, j] = Wij
            W_edge[j, i] = dagger(Wij)
            invs.append(float(np.real(np.trace(Wij)) / 2.0))
        gr_data = {'enabled': True, 'frame_seed': int(frame_seed), 'frame_source': 'node_frames' if node_frames is not None else 'frame_from_coord_fallback', 'edge_invariant_Ie_mean': float(np.mean(invs)) if invs else 0.0, 'edge_invariant_Ie_std': float(np.std(invs)) if invs else 0.0}
    else:
        gr_data = {'enabled': False}
    edge_channels: List[EdgeChannel] = []
    for i, j in edges:
        if isinstance(p_swap, dict):
            key = (i, j) if i < j else (j, i)
            p = float(p_swap.get(key, 0.0))
        else:
            p = float(p_swap)
        inv = inv_perm_from_swaps(N, swaps=[(i, j)])
        Ui_full = Uj_full = None
        if gr_transport and frames is not None:
            Wij = W_edge[i, j]
            Wji = dagger(Wij)
            Ui_full = embed_operator(Wji, keep=[i], N=N, hermitize_output=False)
            Uj_full = embed_operator(Wij, keep=[j], N=N, hermitize_output=False)
        edge_channels.append(EdgeChannel(i=i, j=j, p_swap=p, inv_perm=inv, Ui_full=Ui_full, Uj_full=Uj_full))
    remaining = list(range(len(edge_channels)))
    steps: List[SpacelikeStep] = []
    while remaining:
        rng.shuffle(remaining)
        used = np.zeros(N, dtype=bool)
        sel, rem2 = ([], [])
        for idx in remaining:
            e = edge_channels[idx]
            if used[e.i] or used[e.j]:
                rem2.append(idx)
                continue
            used[e.i] = True
            used[e.j] = True
            sel.append(idx)
        steps.append(SpacelikeStep(edges=[edge_channels[i] for i in sel]))
        remaining = rem2
    meta = {'graph': graph.name, 'N': N, 'n_edges': len(edges), 'n_spacelike_steps': len(steps), 'p_swap_uniform': float(p_swap) if not isinstance(p_swap, dict) else None, 'p_swap_per_edge': isinstance(p_swap, dict), 'seed': int(seed), 'gr_transport': bool(gr_transport), 'note': 'Disjoint matchings; each step is a commuting product of local edge CPTP channels.'}
    return (CausalStructure(N=N, steps=steps, edge_list=edge_channels, meta=meta), gr_data)

def gauge_covariance_check_for_gr_channel(graph: GeometryGraph, node_coords: List[np.ndarray], node_frames: Optional[List[np.ndarray]], p_swap: Union[float, Dict[Tuple[int, int], float]], causal_seed: int, frame_seed: int, n_tests: int=10, seed: int=0) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    N = int(graph.adjacency.shape[0])
    d = 2 ** N
    C0, gr0 = build_causal_structure(graph, p_swap=p_swap, seed=causal_seed, gr_transport=True, node_coords=node_coords, node_frames=node_frames, frame_seed=frame_seed)

    def random_su2() -> np.ndarray:
        axis = rng.normal(size=3)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-15)
        ang = float(rng.uniform(-math.pi, math.pi))
        return su2_from_axis_angle(axis, ang)

    def embed_product(Gs: List[np.ndarray]) -> np.ndarray:
        U = np.eye(d, dtype=np.complex128)
        for i in range(N):
            U = embed_operator(Gs[i], keep=[i], N=N, hermitize_output=False) @ U
        return U
    errs = []
    for _ in range(int(n_tests)):
        X = rng.normal(size=(d, min(d, 8))) + 1j * rng.normal(size=(d, min(d, 8)))
        rho = _ensure_density(X @ dagger(X))
        Gs = [random_su2() for _ in range(N)]
        Ug = embed_product(Gs)
        rho_g = Ug @ rho @ dagger(Ug)
        frames0 = [np.asarray(node_frames[i], dtype=np.complex128) for i in range(N)] if node_frames is not None else [frame_from_coord(node_coords[i], seed=frame_seed) for i in range(N)]
        frames1 = [Gs[i] @ frames0[i] for i in range(N)]
        A = np.asarray(graph.adjacency, float)
        edges = [(i, j) for i in range(N) for j in range(i + 1, N) if A[i, j] > 0.0]
        edge_channels = []
        for i, j in edges:
            if isinstance(p_swap, dict):
                key = (i, j) if i < j else (j, i)
                p = float(p_swap.get(key, 0.0))
            else:
                p = float(p_swap)
            inv = inv_perm_from_swaps(N, swaps=[(i, j)])
            Wij = frames1[j] @ dagger(frames1[i])
            Ui_full = embed_operator(dagger(Wij), keep=[i], N=N, hermitize_output=False)
            Uj_full = embed_operator(Wij, keep=[j], N=N, hermitize_output=False)
            edge_channels.append(EdgeChannel(i=i, j=j, p_swap=p, inv_perm=inv, Ui_full=Ui_full, Uj_full=Uj_full))
        by_pair = {(min(e.i, e.j), max(e.i, e.j)): e for e in edge_channels}
        steps_prime: List[SpacelikeStep] = []
        for st in C0.steps:
            st_edges = []
            for e in st.edges:
                key = (min(e.i, e.j), max(e.i, e.j))
                st_edges.append(by_pair[key])
            steps_prime.append(SpacelikeStep(edges=st_edges))
        C1 = CausalStructure(N=N, steps=steps_prime, edge_list=edge_channels, meta=dict(C0.meta))
        out_ref = Ug @ C0.apply_global(rho) @ dagger(Ug)
        out_cov = C1.apply_global(rho_g)
        errs.append(float(trace_distance(out_ref, out_cov)))
    return {'ok': True, 'N': N, 'n_tests': int(n_tests), 'mean_trace_distance': float(np.mean(errs)) if errs else 0.0, 'max_trace_distance': float(np.max(errs)) if errs else 0.0, 'gr_meta': gr0, 'note': 'Lower is better; should be ~0 up to numerical error if covariance holds.'}

def boundary_edge_count(ball: List[int], graph: GeometryGraph) -> int:
    A = np.asarray(graph.adjacency, dtype=float) > 0.0
    S = set(ball)
    cnt = 0
    for i in ball:
        for j in np.where(A[i])[0].tolist():
            if j not in S:
                cnt += 1
    return int(cnt)

def area_proxy(ball: List[int], graph: GeometryGraph, K_eff: np.ndarray, K_ref: np.ndarray, a: float) -> float:
    bc = boundary_edge_count(ball, graph)
    curv = float(np.sum(np.abs(K_eff[ball] - K_ref[ball])))
    return float(a) * bc + 0.05 * curv

def first_law_check_on_balls(rho_ref: np.ndarray, balls: List[List[int]], KB_list: List[np.ndarray], N: int, eps_list: Sequence[float]=(0.0003, 0.001, 0.003, 0.01), seed: int=0) -> Dict[str, Any]:
    rho_ref = _ensure_density(rho_ref)
    rng = np.random.default_rng(int(seed))
    d = rho_ref.shape[0]
    H = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    H = hermitize(H)
    results = []
    Sref = []
    for v, B in enumerate(balls):
        rhoB = partial_trace_qubits(rho_ref, keep=B, N=N)
        Sref.append(von_neumann_entropy(rhoB))
    Sref = np.array(Sref, float)
    for eps in eps_list:
        U = expm_hermitian(H, t=float(eps), factor=-1j)
        rho = U @ rho_ref @ dagger(U)
        dS = np.zeros(len(balls), float)
        dK = np.zeros(len(balls), float)
        for v, B in enumerate(balls):
            rhoB = partial_trace_qubits(rho, keep=B, N=N)
            rhoBref = partial_trace_qubits(rho_ref, keep=B, N=N)
            dS[v] = von_neumann_entropy(rhoB) - Sref[v]
            dK[v] = float(np.trace((rhoB - rhoBref) @ KB_list[v]).real)
        err = dS - dK
        rms = float(math.sqrt(np.mean(err * err)))
        mx = float(np.max(np.abs(err)))
        corr = float(np.corrcoef(dS, dK)[0, 1]) if np.std(dS) > 1e-15 and np.std(dK) > 1e-15 else float('nan')
        denom = float(np.dot(dK, dK))
        slope = float(np.dot(dK, dS) / denom) if denom > 1e-15 else float('nan')
        results.append({'eps': float(eps), 'rms_err': rms, 'max_abs_err': mx, 'corr': corr, 'slope_dS_on_dK': slope})
    best = min(results, key=lambda r: r['rms_err'])
    return {'seed': int(seed), 'eps_list': list(map(float, eps_list)), 'results': results, 'chosen': best}

def random_density_matrix(d: int, rng: np.random.Generator, rank: Optional[int]=None) -> np.ndarray:
    d = int(d)
    if rank is None:
        rank = d
    rank = int(rank)
    X = rng.normal(size=(d, rank)) + 1j * rng.normal(size=(d, rank))
    rho = X @ dagger(X)
    return _ensure_density(rho)

def pachner_channel_invariance_check(surface: TriangulatedSurface, p_swap: float=0.35, causal_seed: int=0, n_tests: int=20, seed: int=0, gr_transport: bool=False) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    cand = flippable_edges_2_2(surface)
    if not cand:
        return {'ok': False, 'reason': 'no flippable edges for 2-2 move', 'surface': surface.name}
    edge = cand[int(rng.integers(0, len(cand)))]
    surf2 = pachner_flip_2_2(surface, edge=edge)
    nodes1, g1, _ = surface_to_nodes_graph(surface, name_prefix='v')
    nodes2, g2, _ = surface_to_nodes_graph(surf2, name_prefix='v')
    if g1.adjacency.shape != g2.adjacency.shape:
        return {'ok': False, 'reason': 'vertex count changed (should not happen in 2-2)', 'surface': surface.name}
    N = int(g1.adjacency.shape[0])
    d = 2 ** N
    coords1 = [n.coord for n in nodes1]
    coords2 = [n.coord for n in nodes2]
    frames1 = [n.frame_su2 for n in nodes1]
    frames2 = [n.frame_su2 for n in nodes2]
    C1, _ = build_causal_structure(g1, p_swap=float(p_swap), seed=int(causal_seed), gr_transport=gr_transport, node_coords=coords1, node_frames=frames1, frame_seed=111)
    C2, _ = build_causal_structure(g2, p_swap=float(p_swap), seed=int(causal_seed), gr_transport=gr_transport, node_coords=coords2, node_frames=frames2, frame_seed=111)
    T = np.asarray(surface.triangles, np.int64)
    e2t = surface_edges(T)
    a, b = edge
    aa, bb = (a, b) if a < b else (b, a)
    t1, t2 = e2t[aa, bb]
    tri1 = set(T[t1].tolist())
    tri2 = set(T[t2].tolist())
    k = int(list(tri1 - {aa, bb})[0])
    l = int(list(tri2 - {aa, bb})[0])
    affected = sorted({int(aa), int(bb), int(k), int(l)})
    unaffected = [i for i in range(N) if i not in affected]
    ds = []
    ds_unaff = []
    tr_err = []
    for _ in range(int(n_tests)):
        rho = random_density_matrix(d, rng=rng, rank=min(d, 8))
        o1 = C1.apply_global(rho)
        o2 = C2.apply_global(rho)
        ds.append(trace_distance(o1, o2))
        tr_err.append(abs(float(np.trace(o1).real - 1.0)) + abs(float(np.trace(o2).real - 1.0)))
        if unaffected:
            r1u = partial_trace_qubits(o1, keep=unaffected, N=N)
            r2u = partial_trace_qubits(o2, keep=unaffected, N=N)
            ds_unaff.append(trace_distance(r1u, r2u))
    return {'ok': True, 'surface': surface.name, 'edge_flipped': [int(edge[0]), int(edge[1])], 'affected_vertices': affected, 'N': N, 'p_swap': float(p_swap), 'gr_transport': bool(gr_transport), 'n_tests': int(n_tests), 'mean_trace_distance': float(np.mean(ds)) if ds else 0.0, 'max_trace_distance': float(np.max(ds)) if ds else 0.0, 'mean_trace_distance_unaffected_subsystem': float(np.mean(ds_unaff)) if ds_unaff else None, 'max_trace_distance_unaffected_subsystem': float(np.max(ds_unaff)) if ds_unaff else None, 'trace_preservation_drift_mean_abs': float(np.mean(tr_err)) if tr_err else 0.0, 'order_independence_error_ref_graph1': float(C1.order_independence_error(np.eye(d, dtype=np.complex128) / d)), 'order_independence_error_ref_graph2': float(C2.order_independence_error(np.eye(d, dtype=np.complex128) / d))}

def pachner_channel_invariance_check_matched_density(surface: TriangulatedSurface, p_swap_base: float=0.35, dt_density: float=0.03, causal_seed: int=0, n_tests: int=20, seed: int=0, gr_transport: bool=False) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    cand = flippable_edges_2_2(surface)
    if not cand:
        return {'ok': False, 'reason': 'no flippable edges for 2-2 move', 'surface': surface.name}
    edge_old = cand[int(rng.integers(0, len(cand)))]
    surf2 = pachner_flip_2_2(surface, edge=edge_old)
    nodes1, g1, _ = surface_to_nodes_graph(surface, name_prefix='v')
    nodes2, g2, _ = surface_to_nodes_graph(surf2, name_prefix='v')
    if g1.adjacency.shape != g2.adjacency.shape:
        return {'ok': False, 'reason': 'vertex count changed (should not happen in 2-2)', 'surface': surface.name}
    N = int(g1.adjacency.shape[0])
    d = 2 ** N
    L1 = edge_length_map(surface)
    L2 = edge_length_map(surf2)

    def pmap_from_kappa(L: Dict[Tuple[int, int], float], kappa: float) -> Dict[Tuple[int, int], float]:
        out: Dict[Tuple[int, int], float] = {}
        for (a, b), ell in L.items():
            l2 = max(float(ell * ell), 1e-12)
            p = 1.0 - math.exp(-float(kappa) * float(dt_density) / l2)
            out[a, b] = float(np.clip(p, 0.0, 1.0))
        return out
    Lvals = np.array(list(L1.values()), float)
    L2vals = np.clip(Lvals * Lvals, 1e-12, None)

    def mean_p(kappa: float) -> float:
        p = 1.0 - np.exp(-float(kappa) * float(dt_density) / L2vals)
        return float(np.mean(p))
    target = float(np.clip(p_swap_base, 1e-06, 1.0 - 1e-06))
    lo, hi = (0.0, 1000.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mean_p(mid) < target:
            lo = mid
        else:
            hi = mid
    kappa = 0.5 * (lo + hi)
    p1 = pmap_from_kappa(L1, kappa)
    p2 = pmap_from_kappa(L2, kappa)
    T = np.asarray(surface.triangles, np.int64)
    e2t = surface_edges(T)
    a, b = edge_old
    aa, bb = (a, b) if a < b else (b, a)
    t1, t2 = e2t[aa, bb]
    tri1 = set(T[t1].tolist())
    tri2 = set(T[t2].tolist())
    k = int(list(tri1 - {aa, bb})[0])
    l = int(list(tri2 - {aa, bb})[0])
    edge_new = (k, l) if k < l else (l, k)
    p_old_diag = float(p1.get((aa, bb), p_swap_base))
    p2[edge_new] = p_old_diag
    coords1 = [n.coord for n in nodes1]
    coords2 = [n.coord for n in nodes2]
    frames1 = [n.frame_su2 for n in nodes1]
    frames2 = [n.frame_su2 for n in nodes2]
    C1, _ = build_causal_structure(g1, p_swap=p1, seed=int(causal_seed), gr_transport=gr_transport, node_coords=coords1, node_frames=frames1, frame_seed=111)
    C2, _ = build_causal_structure(g2, p_swap=p2, seed=int(causal_seed), gr_transport=gr_transport, node_coords=coords2, node_frames=frames2, frame_seed=111)
    affected = sorted({int(aa), int(bb), int(k), int(l)})
    unaffected = [i for i in range(N) if i not in affected]
    ds = []
    ds_unaff = []
    for _ in range(int(n_tests)):
        rho = random_density_matrix(d, rng=rng, rank=min(d, 8))
        o1 = C1.apply_global(rho)
        o2 = C2.apply_global(rho)
        ds.append(trace_distance(o1, o2))
        if unaffected:
            r1u = partial_trace_qubits(o1, keep=unaffected, N=N)
            r2u = partial_trace_qubits(o2, keep=unaffected, N=N)
            ds_unaff.append(trace_distance(r1u, r2u))
    return {'ok': True, 'surface': surface.name, 'edge_flipped_old': [int(aa), int(bb)], 'edge_flipped_new': [int(edge_new[0]), int(edge_new[1])], 'affected_vertices': affected, 'N': N, 'p_swap_base': float(p_swap_base), 'dt_density': float(dt_density), 'kappa_calibrated': float(kappa), 'p_old_diag': float(p_old_diag), 'p_new_diag': float(p2[edge_new]), 'gr_transport': bool(gr_transport), 'n_tests': int(n_tests), 'mean_trace_distance': float(np.mean(ds)) if ds else 0.0, 'max_trace_distance': float(np.max(ds)) if ds else 0.0, 'mean_trace_distance_unaffected_subsystem': float(np.mean(ds_unaff)) if ds_unaff else None, 'max_trace_distance_unaffected_subsystem': float(np.max(ds_unaff)) if ds_unaff else None, 'note': 'Matches diagonal generator density across 2–2 move via p(new_diag)=p(old_diag) with p(e) from edge-length density.'}

def calibrate_kappa_for_mean_p(edge_lengths: Sequence[float], dt: float, target_p: float, iters: int=60) -> float:
    L = np.asarray(list(edge_lengths), dtype=float)
    L2 = np.clip(L * L, 1e-12, None)
    target = float(np.clip(target_p, 1e-06, 1.0 - 1e-06))

    def mean_p(kappa: float) -> float:
        p = 1.0 - np.exp(-float(kappa) * float(dt) / L2)
        return float(np.mean(p))
    lo, hi = (0.0, 1000.0)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        if mean_p(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def p_swap_map_edge_length_density(surface: TriangulatedSurface, dt: float, target_mean_p: float, kappa: Optional[float]=None) -> Tuple[Dict[Tuple[int, int], float], float]:
    L = edge_length_map(surface)
    if not L:
        return ({}, 0.0)
    if kappa is None:
        kappa = calibrate_kappa_for_mean_p(L.values(), dt=float(dt), target_p=float(target_mean_p))
    out: Dict[Tuple[int, int], float] = {}
    for (a, b), ell in L.items():
        l2 = max(float(ell * ell), 1e-12)
        p = 1.0 - math.exp(-float(kappa) * float(dt) / l2)
        out[a, b] = float(np.clip(p, 0.0, 1.0))
    return (out, float(kappa))

def full_dynamics_triangulation_invariance_suite(surface: TriangulatedSurface, mode: str, base_params_template: Dict[str, Any], n_flips: int=6, ball_radii: Sequence[int]=(1, 2), T_test: float=0.3, dt: float=0.03, store_every: int=1, seed: int=0, causal_mean_p_target: float=0.35, match_causal_density: bool=True, gr_transport: bool=False) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    seq = pachner_flip_sequence(surface, n_flips=int(n_flips), seed=int(seed))
    if len(seq) <= 1:
        return {'ok': False, 'reason': 'Could not generate flip sequence (no flippable edges).', 'surface': surface.name}
    K_ref_fixed = seq[0].curvature_scalar()
    tpl = dict(base_params_template)
    tpl.pop('nodes', None)
    tpl.pop('graph', None)
    results_by_radius: List[Dict[str, Any]] = []
    for r in ball_radii:
        r = int(r)
        sims: List[GTRQCSimulator] = []
        sums: List[Dict[str, Any]] = []
        mats: List[List[np.ndarray]] = []
        for idx, surf in enumerate(seq):
            nodes, graph, _ = surface_to_nodes_graph(surf, name_prefix=f'tri{idx}_v', K_ref_override=K_ref_fixed, compute_frames=True, frame_seed=0)
            params = dict(tpl)
            params['nodes'] = nodes
            params['graph'] = graph
            params['ball_radius_hops'] = r
            params['rate_mode'] = GTRQCSimulator._MODE_ALIASES.get(mode.strip().lower(), mode.strip().lower())
            if match_causal_density:
                pmap, kappa = p_swap_map_edge_length_density(surf, dt=float(dt), target_mean_p=float(causal_mean_p_target))
                params['causal_p_swap'] = pmap
                params['causal_gr_transport'] = bool(gr_transport)
                params['causal_apply_every'] = int(params.get('causal_apply_every', 1))
            else:
                params['causal_p_swap'] = float(causal_mean_p_target)
            sim = GTRQCSimulator(**params)
            label = f"TRI_SUITE r={r} idx={idx}/{len(seq) - 1} mode={params['rate_mode']}"
            summ = sim.run(T_final=float(T_test), dt=float(dt), store_every=int(store_every), label=label, progress_every=0, store_matter_states=True)
            sims.append(sim)
            sums.append(summ)
            assert sim.rho_m_hist is not None
            mats.append(sim.rho_m_hist)
        ref_states = mats[0]
        timegrid = np.array(sims[0].times, float)
        dists_final = []
        dists_max_over_time = []
        for idx in range(1, len(seq)):
            st = mats[idx]
            ds_t = [trace_distance(st[k], ref_states[k]) for k in range(len(timegrid))]
            dists_final.append(float(ds_t[-1]))
            dists_max_over_time.append(float(np.max(ds_t)))
        results_by_radius.append(dict(ball_radius_hops=r, n_triangulations=len(seq), n_flips_attempted=int(n_flips), T_test=float(T_test), dt=float(dt), store_every=int(store_every), mode=str(mode), match_causal_density=bool(match_causal_density), gr_transport=bool(gr_transport), mean_trace_distance_final=float(np.mean(dists_final)) if dists_final else 0.0, max_trace_distance_final=float(np.max(dists_final)) if dists_final else 0.0, mean_trace_distance_max_over_time=float(np.mean(dists_max_over_time)) if dists_max_over_time else 0.0, max_trace_distance_max_over_time=float(np.max(dists_max_over_time)) if dists_max_over_time else 0.0, per_triangulation_summaries=sums))
    return {'ok': True, 'surface': surface.name, 'sequence_length': len(seq), 'n_flips_requested': int(n_flips), 'ball_radii': list(map(int, ball_radii)), 'results_by_radius': results_by_radius, 'note': 'Compares full coupled dynamics across many 2-2 flips; holds K_ref fixed from initial triangulation; optionally matches causal density via edge-length-based p(e).'}

def pachner_flip_sequence(surface: TriangulatedSurface, n_flips: int, seed: int=0) -> List[TriangulatedSurface]:
    rng = np.random.default_rng(int(seed))
    seq = [surface]
    cur = surface
    for _ in range(int(n_flips)):
        cand = flippable_edges_2_2(cur)
        if not cand:
            break
        edge = cand[int(rng.integers(0, len(cand)))]
        cur = pachner_flip_2_2(cur, edge=edge)
        seq.append(cur)
    return seq

class GTRQCSimulator:
    _MODE_ALIASES = {'linear_fixed': 'fixed_projector_unsigned', 'linear_operator_functional': 'fixed_projector_signed', 'paper_instantaneous_j_gibbs': 'instantaneous_projector_unsigned', 'paper_instantaneous_j_gibbs_operator': 'instantaneous_projector_signed', 'linear_operator_projector': 'fixed_projector_signed', 'linear_operator_functional_calculus': 'fixed_functional_signed', 'operator_functional_calculus': 'fixed_functional_signed', 'paper_instantaneous_j_gibbs_functional': 'instantaneous_functional_signed', 'instantaneous_functional': 'instantaneous_functional_signed', 'embedded_j_gibbs': 'embedded_projector_unsigned', 'embedded_j_gibbs_operator': 'embedded_projector_signed', 'embedded_j_gibbs_functional': 'embedded_functional_signed', 'embedded_functional': 'embedded_functional_signed'}

    def apply_curvature_shift(self, delta: np.ndarray, *, rebuild_fixed_rates: bool=True) -> None:
        delta = np.asarray(delta, dtype=float).reshape(-1)
        if delta.shape[0] != self.N:
            raise ValueError('apply_curvature_shift: delta must have shape (N,).')
        if getattr(self, 'enforce_gauss_bonnet', False):
            delta = delta - float(np.mean(delta))
        self.K_curv_shift = delta.copy()
        self.K_cfg = self.K_cfg_base + delta[None, :]
        self.H_list = np.stack([self._H_for_K(self.K_cfg[a]) for a in range(self.M)], axis=0)
        for i in range(self.N):
            for a in range(self.M):
                self.gamma_deph[i, a] = self._gamma_i(i, float(self.K_cfg[a, i]))
        if not rebuild_fixed_rates:
            return
        J0 = np.zeros(self.N, dtype=float)
        F0 = np.array([slice_F(self._k(self.K_cfg[a]), J0, self.G, self.alpha_area) for a in range(self.M)], dtype=float)
        self.pi0 = gibbs_pi(F0, self.beta_geom)
        self.R0 = metropolis_rates(F0, self.A_cfg, self.beta_geom, self.gamma_geom_base)
        self.out0 = np.sum(self.R0, axis=0)
        if self.backreaction == 'embedded' and self.J_source is not None:
            piS = np.zeros((self.S, self.M), float)
            RS = np.zeros((self.S, self.M, self.M), float)
            outS = np.zeros((self.S, self.M), float)
            for s in range(self.S):
                Js = self.J_source[s]
                Fs = np.array([slice_F(self._k(self.K_cfg[a]), Js, self.G, self.alpha_area) for a in range(self.M)], dtype=float)
                piS[s] = gibbs_pi(Fs, self.beta_geom)
                RS[s] = metropolis_rates(Fs, self.A_cfg, self.beta_geom, self.gamma_geom_base)
                outS[s] = np.sum(RS[s], axis=0)
            self.pi_source_geom = piS
            self.R_source = RS
            self.out_source = outS

    def __init__(self, nodes: List[Node], graph: GeometryGraph, ball_radius_hops: int=1, alpha_area: float=1.0, regge_a: float=0.8, mass_m: float=0.7, mesh_h: float=1.0, beta_geom: float=6.0, gamma_geom_base: float=12.0, levels_per_node: int=5, deltaK_scale: float=0.18, max_configs: int=10, seed_geometry_ensemble: int=77, omega_0: float=2.0 * math.pi * 40.0, alpha_curv_freq: float=0.2, gamma0: float=0.3, gamma_curv: float=1.2, exchange_J_strength: float=2.5, transverse_hx: Optional[float]=None, eps_modular: float=0.002, eps_ref_rank: float=1e-06, reference_state: str='zeros', initial_state: str='ghz', initial_rho_override: Optional[np.ndarray]=None, negativity_pair: Tuple[int, int]=(0, -1), geometry_quantum: bool=False, geometry_initial: str='diagonal_gibbs', geom_mix_rate: float=1.0, geom_pointer_dephase: float=0.15, monitor_eta: float=0.9, causal_p_swap: Union[float, Dict[Tuple[int, int], float]]=0.35, causal_seed: int=123, causal_apply_every: int=1, causal_gr_transport: bool=False, causal_gr_frame_seed: int=111, functional_kappa: float=0.4, rate_mode: str='linear_operator_functional_calculus', source_dim: int=7, source_J_scale: float=0.5, source_measure_strength: float=1.0, source_observable_mix: float=1.0, source_init: str='delta_center', enforce_zero_mean_if_massless: bool=True, enforce_gauss_bonnet: bool=False, jump_substep_safety: float=0.9, trace_mode: str='hygiene', trace_tol: float=5e-11, trace_hygiene_tol: float=1e-06, memory_guard_gib: Optional[float]=2.0):
        self.nodes = nodes
        self.graph = graph
        self.N = len(nodes)
        self.d = 2 ** self.N
        self.ball_radius_hops = int(ball_radius_hops)
        self.balls = build_balls(graph, self.ball_radius_hops)
        self.alpha_area = float(alpha_area)
        self.regge_a = float(regge_a)
        self.mass_m = float(mass_m)
        self.mesh_h = float(mesh_h)
        self.beta_geom = float(beta_geom)
        self.gamma_geom_base = float(gamma_geom_base)
        self.omega_0 = float(omega_0)
        self.alpha_curv_freq = float(alpha_curv_freq)
        self.gamma0 = float(gamma0)
        self.gamma_curv = float(gamma_curv)
        self.exchange_J = float(exchange_J_strength) * np.asarray(graph.adjacency, dtype=float)
        self.exchange_J_strength = float(exchange_J_strength)
        if transverse_hx is None:
            self.transverse_hx = float(getattr(self, 'transverse_hx', 0.0))
        else:
            self.transverse_hx = float(transverse_hx)
        self.enforce_gauss_bonnet = bool(enforce_gauss_bonnet)
        self.total_curvature_target = float(np.sum([n.curvature_ref for n in nodes]))
        self.eps_modular = float(eps_modular)
        self.eps_ref_rank = float(eps_ref_rank)
        self.geometry_quantum = bool(geometry_quantum)
        self.geometry_initial = str(geometry_initial).lower().strip()
        if self.geometry_initial not in ('coherent_gibbs', 'diagonal_gibbs'):
            raise ValueError('geometry_initial must be coherent_gibbs or diagonal_gibbs')
        self.geom_mix_rate = float(geom_mix_rate)
        self.geom_pointer_dephase = float(geom_pointer_dephase)
        self.monitor_eta = float(monitor_eta)
        self.causal_apply_every = max(int(causal_apply_every), 0)
        self.causal_gr_transport = bool(causal_gr_transport)
        self.causal_gr_frame_seed = int(causal_gr_frame_seed)
        self.causal_seed = int(causal_seed)
        if isinstance(causal_p_swap, dict):
            pmap: Dict[Tuple[int, int], float] = {}
            for (i, j), p in causal_p_swap.items():
                ii, jj = (int(i), int(j))
                if ii == jj:
                    continue
                if ii > jj:
                    ii, jj = (jj, ii)
                pmap[ii, jj] = float(np.clip(float(p), 0.0, 1.0))
            self.causal_p_swap = pmap
        else:
            self.causal_p_swap = float(causal_p_swap)
        mode_in = str(rate_mode).strip().lower()
        mode = self._MODE_ALIASES.get(mode_in, mode_in)
        self.rate_mode = mode
        if mode.startswith('embedded_'):
            self.backreaction = 'embedded'
        elif mode.startswith('instantaneous_'):
            self.backreaction = 'instantaneous'
        else:
            self.backreaction = 'fixed'
        if 'functional' in mode:
            self.op_rate_family = 'functional'
        elif '_scalar_' in mode:
            self.op_rate_family = 'scalar'
        else:
            self.op_rate_family = 'projector'
        self.use_signed_monitors = mode.split('_')[-1] == 'signed'
        self.functional_kappa = float(functional_kappa)
        self.enforce_zero_mean_if_massless = bool(enforce_zero_mean_if_massless)
        self.volume_constraint_active = bool(self.enforce_zero_mean_if_massless and is_massless(self.mass_m))
        self.jump_substep_safety = float(jump_substep_safety)
        if not 0.0 < self.jump_substep_safety < 1.0:
            raise ValueError('jump_substep_safety must be in (0,1).')
        self.K_ref = np.array([n.curvature_ref for n in nodes], dtype=float)
        self.total_curvature_target = float(np.sum(self.K_ref))
        self.G = build_response_G(graph, m=self.mass_m, h=self.mesh_h)
        self.X_ops, self.Y_ops, self.Z_ops = build_local_paulis(self.N)
        self.K_cfg, self.codes, self.level_offsets = build_geometry_ensemble(self.K_ref, levels_per_node=int(levels_per_node), delta_scale=float(deltaK_scale), max_configs=int(max_configs), seed=int(seed_geometry_ensemble), enforce_total_curvature=bool(self.enforce_gauss_bonnet), total_curvature_target=float(self.total_curvature_target) if self.enforce_gauss_bonnet else None)
        self.K_cfg_base = np.array(self.K_cfg, dtype=float, copy=True)
        self.K_curv_shift = np.zeros(self.N, dtype=float)
        self.M = int(self.K_cfg.shape[0])
        self.A_cfg, self.edge_info = build_config_graph(self.codes)
        self.neigh_by_source: List[List[Tuple[int, int, int]]] = [[] for _ in range(self.M)]
        for b in range(self.M):
            for a in range(self.M):
                if a == b or self.A_cfg[a, b] <= 0.0:
                    continue
                v, sgn = self.edge_info[a, b]
                self.neigh_by_source[b].append((a, int(v), int(sgn)))
        self.HG = hermitize((self.geom_mix_rate * self.A_cfg).astype(np.complex128))
        self.rho_ref_pure = self._make_state(reference_state)
        self.rho_ref = (1.0 - self.eps_ref_rank) * self.rho_ref_pure + self.eps_ref_rank * (np.eye(self.d, dtype=np.complex128) / self.d)
        if initial_rho_override is not None:
            rho0 = _ensure_density(np.asarray(initial_rho_override, dtype=np.complex128))
            if rho0.shape != (self.d, self.d):
                raise ValueError('initial_rho_override has wrong shape.')
            self.rho0 = rho0
        else:
            self.rho0 = self._make_state(initial_state)
        self.KB_list = self._build_modulars(self.rho_ref_pure)
        self.Pi_full, self.M_full, self.monitor_obs_full = self._build_monitors_and_observables()
        self.effect_norm_bound = 1.0
        self.effect_eig_min = 0.0
        self.effect_eig_max = 1.0
        self.effect_eig_violation_count = 0
        try:
            max_eigs = []
            min_eigs = []
            for v in range(self.N):
                if self.op_rate_family == 'projector':
                    _, sF = self._select_F_and_sqrtF(v, +1)
                    E = effect_from_sqrt(sF)
                    w = np.linalg.eigvalsh(E).real
                    max_eigs.append(float(np.max(w)))
                    min_eigs.append(float(np.min(w)))
                else:
                    for sgn in (+1, -1):
                        _, sF = self._select_F_and_sqrtF(v, sgn)
                        E = effect_from_sqrt(sF)
                        w = np.linalg.eigvalsh(E).real
                        max_eigs.append(float(np.max(w)))
                        min_eigs.append(float(np.min(w)))
            if max_eigs:
                self.effect_norm_bound = max(1.0, float(np.max(max_eigs)))
                self.effect_eig_min = float(np.min(min_eigs))
                self.effect_eig_max = float(np.max(max_eigs))
                tol = 5e-12
                self.effect_eig_violation_count = int(np.sum(np.asarray(min_eigs) < -tol) + np.sum(np.asarray(max_eigs) > 1.0 + tol))
        except Exception:
            self.effect_norm_bound = 1.0
            self.effect_eig_min = float('nan')
            self.effect_eig_max = float('nan')
            self.effect_eig_violation_count = -1
        I = np.eye(self.d, dtype=np.complex128)
        self.Pi_bar_full = [hermitize(I - self.Pi_full[v]) for v in range(self.N)]
        self.M_bar_full = [matrix_sqrt_psd(self.Pi_bar_full[v], eps=0.0) for v in range(self.N)]
        self.F_plus_full: List[np.ndarray] = []
        self.F_minus_full: List[np.ndarray] = []
        self.sF_plus_full: List[np.ndarray] = []
        self.sF_minus_full: List[np.ndarray] = []
        for v in range(self.N):
            evals, U = np.linalg.eigh(hermitize(self.monitor_obs_full[v]))
            Fp, sFp = op_sigmoid_from_eigh(U, evals, self.functional_kappa, +1)
            Fm, sFm = op_sigmoid_from_eigh(U, evals, self.functional_kappa, -1)
            self.F_plus_full.append(Fp)
            self.sF_plus_full.append(sFp)
            self.F_minus_full.append(Fm)
            self.sF_minus_full.append(sFm)
        self.H_list = np.stack([self._H_for_K(self.K_cfg[a]) for a in range(self.M)], axis=0)
        self.gamma_deph = np.zeros((self.N, self.M), dtype=float)
        for i in range(self.N):
            for a in range(self.M):
                self.gamma_deph[i, a] = self._gamma_i(i, float(self.K_cfg[a, i]))
        J0 = np.zeros(self.N, dtype=float)
        F0 = np.array([slice_F(self._k(self.K_cfg[a]), J0, self.G, self.alpha_area) for a in range(self.M)], dtype=float)
        self.pi0 = gibbs_pi(F0, self.beta_geom)
        self.R0 = metropolis_rates(F0, self.A_cfg, self.beta_geom, self.gamma_geom_base)
        self.out0 = np.sum(self.R0, axis=0)
        self.source_dim = int(max(1, source_dim))
        self.source_J_scale = float(source_J_scale)
        self.source_measure_strength = float(source_measure_strength)
        self.source_observable_mix = float(source_observable_mix)
        self.source_init = str(source_init).lower().strip()
        self.S = 1
        self.J_source: Optional[np.ndarray] = None
        self.pi_source_geom: Optional[np.ndarray] = None
        self.R_source: Optional[np.ndarray] = None
        self.out_source: Optional[np.ndarray] = None
        self.sqrtE_source: Optional[List[np.ndarray]] = None
        if self.backreaction == 'embedded':
            self.S = int(self.source_dim)
            self.J_source = self._build_source_J_grid(self.S, self.source_J_scale)
            piS = np.zeros((self.S, self.M), float)
            RS = np.zeros((self.S, self.M, self.M), float)
            outS = np.zeros((self.S, self.M), float)
            for s in range(self.S):
                Js = self.J_source[s]
                Fs = np.array([slice_F(self._k(self.K_cfg[a]), Js, self.G, self.alpha_area) for a in range(self.M)], dtype=float)
                piS[s] = gibbs_pi(Fs, self.beta_geom)
                RS[s] = metropolis_rates(Fs, self.A_cfg, self.beta_geom, self.gamma_geom_base)
                outS[s] = np.sum(RS[s], axis=0)
            self.pi_source_geom = piS
            self.R_source = RS
            self.out_source = outS
            self.sqrtE_source = self._build_source_povm(self.S)
        self.trace_mode = str(trace_mode).lower().strip()
        if self.trace_mode not in ('cptp', 'hygiene', 'unnormalized'):
            raise ValueError("trace_mode must be one of {'cptp','hygiene','unnormalized'}")
        self.trace_tol = float(trace_tol)
        self.trace_hygiene_tol = float(trace_hygiene_tol)
        if self.trace_tol <= 0.0 or self.trace_hygiene_tol <= 0.0:
            raise ValueError('trace_tol and trace_hygiene_tol must be positive.')
        self._trace_renorm_count = 0
        est_bytes = estimate_joint_state_bytes(self.N, self.M, self.S, self.geometry_quantum, dtype_bytes=16)
        self.rho_bytes_est = int(est_bytes)
        self.rho_gib_est = _bytes_to_gib(est_bytes)
        if memory_guard_gib is not None:
            guard = float(memory_guard_gib)
            if self.rho_gib_est > guard:
                raise MemoryError(f'Estimated rho allocation {self.rho_gib_est:.2f} GiB exceeds guard {guard:.2f} GiB. Reduce N (qubits), M (configs), S (source_dim), or disable geometry_quantum.')
        coords = [n.coord if n.coord is not None else np.zeros(3) for n in nodes]
        frames = [n.frame_su2 for n in nodes]
        self.causal_graph_override: Optional[GeometryGraph] = None
        graph_for_causal = self.causal_graph_override if self.causal_graph_override is not None else self.graph
        self.causal, self.causal_gr_data = build_causal_structure(graph_for_causal, p_swap=self.causal_p_swap, seed=int(self.causal_seed), gr_transport=bool(self.causal_gr_transport), node_coords=coords, node_frames=frames if self.causal_gr_transport else None, frame_seed=int(self.causal_gr_frame_seed))
        self.rho = self._init_joint_state()
        p, q = (int(negativity_pair[0]), int(negativity_pair[1]))
        if q < 0:
            q = self.N - 1
        self.neg_pair = (p, q)
        self.causal_order_err = self.causal.order_independence_error(self.rho_ref, step_idx=0)
        self.causal_nosig_err_edge0 = self.causal.no_signaling_error_for_edge(self.rho_ref, edge_idx=0)
        self.times: List[float] = []
        self.K_eff_hist: List[np.ndarray] = []
        self.K_eq_hist: List[np.ndarray] = []
        self.res_hist: List[float] = []
        self.neg_hist: List[float] = []
        self.Srel_hist: List[float] = []
        self.Tdist_hist: List[float] = []
        self.Tdist_mix_hist: List[float] = []
        self.Tdist_step_hist: List[float] = []
        self.purity_hist: List[float] = []
        self.matter_entropy_hist: List[float] = []
        self.ent_entropyA_hist: List[float] = []
        self.ent_entropyB_hist: List[float] = []
        self.mutual_info_hist: List[float] = []
        self.log_negativity_AB_hist: List[float] = []
        self.coh_matter_hist: List[float] = []
        self.matter_min_eig_hist: List[float] = []
        self.CE_hist: List[float] = []
        self.FI_sld_hist: List[float] = []
        self.FI_cls_hist: List[float] = []
        self.KL_hist: List[float] = []
        self.TV_hist: List[float] = []
        self.coh_hist: List[float] = []
        self.p_hist: List[np.ndarray] = []
        self.piJ_hist: List[np.ndarray] = []
        self.J_hist: List[np.ndarray] = []
        self.J_est_hist: List[np.ndarray] = []
        self.pS_hist: List[np.ndarray] = []
        self.Sgen_hist: List[np.ndarray] = []
        self.trace_hist: List[float] = []
        self.CE_path_hist: List[float] = []
        self.FI_path_hist: List[float] = []
        self.rho_m_hist: Optional[List[np.ndarray]] = None

    def _make_state(self, which: str) -> np.ndarray:
        which = str(which).lower().strip()
        ket0 = np.array([1.0, 0.0], dtype=np.complex128)
        ket1 = np.array([0.0, 1.0], dtype=np.complex128)
        ketp = (ket0 + ket1) / math.sqrt(2.0)
        if which == 'zeros':
            psi = tensor_kets([ket0] * self.N)
        elif which == 'plus_product':
            psi = tensor_kets([ketp] * self.N)
        elif which == 'ghz':
            psi0 = tensor_kets([ket0] * self.N)
            psi1 = tensor_kets([ket1] * self.N)
            psi = (psi0 + psi1) / math.sqrt(2.0)
        else:
            raise ValueError(f"Unknown state '{which}' (valid: zeros, plus_product, ghz)")
        return np.outer(psi, psi.conj())

    def _build_modulars(self, rho_ref: np.ndarray) -> List[np.ndarray]:
        KBs = []
        for B in self.balls:
            rhoB = partial_trace_qubits(rho_ref, keep=B, N=self.N)
            dB = rhoB.shape[0]
            rho_reg = (1.0 - self.eps_modular) * rhoB + self.eps_modular * (np.eye(dB, dtype=np.complex128) / dB)
            KBs.append(-matrix_log_psd(rho_reg, eps=1e-12))
        return KBs

    def _build_monitors_and_observables(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        Pi_full, M_full, Mobs_full = ([], [], [])
        for v, B in enumerate(self.balls):
            KB = hermitize(self.KB_list[v])
            rhoB_ref = partial_trace_qubits(self.rho_ref_pure, keep=B, N=self.N)
            mu = float(np.trace(rhoB_ref @ KB).real)
            w, U = np.linalg.eigh(KB)
            x = self.monitor_eta * (w.real - mu)
            x = np.clip(x, -50.0, 50.0)
            p = 1.0 / (1.0 + np.exp(-x))
            PiB = hermitize(U * p @ dagger(U))
            MB = matrix_sqrt_psd(PiB, eps=0.0)
            Pi_full.append(embed_operator(PiB, keep=B, N=self.N))
            M_full.append(embed_operator(MB, keep=B, N=self.N))
            MobsB = hermitize(KB - mu * np.eye(KB.shape[0], dtype=np.complex128))
            Mobs_full.append(embed_operator(MobsB, keep=B, N=self.N))
        return (Pi_full, M_full, Mobs_full)

    def _omega_i(self, i: int, K_i: float) -> float:
        return self.omega_0 * (1.0 + self.alpha_curv_freq * (K_i - float(self.K_ref[i])))

    def _gamma_i(self, i: int, K_i: float) -> float:
        return self.gamma0 + self.gamma_curv * abs(K_i - float(self.K_ref[i]))

    def _H_for_K(self, Kvec: np.ndarray) -> np.ndarray:
        H = np.zeros((self.d, self.d), dtype=np.complex128)
        for i in range(self.N):
            H += 0.5 * self._omega_i(i, float(Kvec[i])) * self.Z_ops[i]
        hx = float(getattr(self, 'transverse_hx', 0.0))
        if hx != 0.0:
            for i in range(self.N):
                H += 0.5 * hx * self.X_ops[i]
        for i in range(self.N):
            for j in range(i + 1, self.N):
                Jij = float(self.exchange_J[i, j])
                if Jij != 0.0:
                    H += 0.5 * Jij * (self.X_ops[i] @ self.X_ops[j] + self.Y_ops[i] @ self.Y_ops[j])
        return hermitize(H)

    def _k(self, Kvec: np.ndarray) -> np.ndarray:
        k = np.asarray(Kvec, dtype=float) - self.K_ref
        return project_zero_mean(k) if self.volume_constraint_active else k

    def _build_source_J_grid(self, S: int, J_scale: float) -> np.ndarray:
        S = int(S)
        w, V = np.linalg.eigh(0.5 * (self.G + self.G.T))
        idx = np.argsort(w)
        V = V[:, idx]
        u = None
        for k in range(V.shape[1]):
            cand = V[:, k].copy()
            if self.volume_constraint_active:
                cand = project_zero_mean(cand)
            if float(np.linalg.norm(cand)) > 1e-09:
                u = cand
                break
        if u is None:
            u = np.ones(self.N, float)
            if self.volume_constraint_active:
                u = project_zero_mean(u)
        u = u / max(float(np.max(np.abs(u))), 1e-15)
        if S == 1:
            svals = np.array([0.0], float)
        else:
            svals = np.linspace(-float(J_scale), float(J_scale), S, dtype=float)
        Jsrc = np.stack([svals[s] * u for s in range(S)], axis=0).astype(float)
        return Jsrc

    def _build_source_povm(self, S: int) -> List[np.ndarray]:
        S = int(S)
        Q = np.zeros((self.d, self.d), dtype=np.complex128)
        for v in range(self.N):
            Q += self.monitor_obs_full[v]
        Q = hermitize(float(self.source_observable_mix) / max(float(self.N), 1.0) * Q)
        evals, U = np.linalg.eigh(Q)
        emin = float(np.min(evals.real))
        emax = float(np.max(evals.real))
        if S == 1:
            w = np.ones((1, evals.size), float)
        else:
            centers = np.linspace(emin, emax, S, dtype=float)
            base = max(emax - emin, 1e-09)
            sigma = base / max(float(S - 1), 1.0)
            sigma = sigma / max(float(self.source_measure_strength), 1e-09)
            sigma = max(float(sigma), 1e-06)
            diff = evals.real[None, :] - centers[:, None]
            w = np.exp(-0.5 * (diff / sigma) ** 2)
            w = np.clip(w, 0.0, None)
        denom = np.sum(w, axis=0, keepdims=True)
        denom = np.clip(denom, 1e-15, None)
        w = w / denom
        sqrtEs: List[np.ndarray] = []
        for s in range(S):
            vals = np.sqrt(np.clip(w[s], 0.0, None))
            sqrtEs.append(op_from_eigh(U, vals))
        return sqrtEs

    def _init_joint_state(self) -> np.ndarray:
        if self.backreaction == 'embedded':
            S = self.S
            if self.source_init == 'uniform':
                pS0 = np.full(S, 1.0 / S, dtype=float)
            else:
                mid = int(np.argmin(np.abs(np.linspace(-1.0, 1.0, S))))
                pS0 = np.zeros(S, float)
                pS0[mid] = 1.0
            assert self.pi_source_geom is not None
            if self.geometry_quantum:
                rho = np.zeros((S, self.M, self.M, self.d, self.d), dtype=np.complex128)
                if self.geometry_initial == 'coherent_gibbs':
                    for s in range(S):
                        ps = float(pS0[s])
                        if ps == 0.0:
                            continue
                        amp = np.sqrt(np.clip(self.pi_source_geom[s], 0.0, None))
                        amp = amp / max(float(np.linalg.norm(amp)), 1e-15)
                        for a in range(self.M):
                            for b in range(self.M):
                                rho[s, a, b] = ps * (amp[a] * amp[b]) * self.rho0
                else:
                    for s in range(S):
                        ps = float(pS0[s])
                        if ps == 0.0:
                            continue
                        for a in range(self.M):
                            rho[s, a, a] = ps * float(self.pi_source_geom[s, a]) * self.rho0
            else:
                rho = np.zeros((S, self.M, self.d, self.d), dtype=np.complex128)
                for s in range(S):
                    ps = float(pS0[s])
                    if ps == 0.0:
                        continue
                    for a in range(self.M):
                        rho[s, a] = ps * float(self.pi_source_geom[s, a]) * self.rho0
            return rho
        S = 1
        if self.geometry_quantum:
            rho = np.zeros((S, self.M, self.M, self.d, self.d), dtype=np.complex128)
            if self.geometry_initial == 'coherent_gibbs':
                amp = np.sqrt(np.clip(self.pi0, 0.0, None))
                amp = amp / max(float(np.linalg.norm(amp)), 1e-15)
                for a in range(self.M):
                    for b in range(self.M):
                        rho[0, a, b] = amp[a] * amp[b] * self.rho0
            else:
                for a in range(self.M):
                    rho[0, a, a] = float(self.pi0[a]) * self.rho0
        else:
            rho = np.zeros((S, self.M, self.d, self.d), dtype=np.complex128)
            for a in range(self.M):
                rho[0, a] = float(self.pi0[a]) * self.rho0
        return rho

    def matter_marginal(self) -> np.ndarray:
        if self.geometry_quantum:
            idx = np.arange(self.M)
            rho_m = np.sum(self.rho[:, idx, idx], axis=(0, 1))
        else:
            rho_m = np.sum(self.rho, axis=(0, 1))
        rho_m = hermitize(rho_m)
        tr = float(np.trace(rho_m).real)
        return rho_m / max(tr, 1e-15)

    def geom_prob(self) -> np.ndarray:
        if self.geometry_quantum:
            p = np.real(np.einsum('saaii->a', self.rho)).astype(float)
        else:
            p = np.real(np.einsum('saii->a', self.rho)).astype(float)
        p = np.clip(p, 0.0, None)
        return p / max(float(np.sum(p)), 1e-15)

    def source_prob(self) -> np.ndarray:
        if self.S == 1:
            return np.array([1.0], float)
        if self.geometry_quantum:
            pS = np.real(np.einsum('saaii->s', self.rho)).astype(float)
        else:
            pS = np.real(np.einsum('saii->s', self.rho)).astype(float)
        pS = np.clip(pS, 0.0, None)
        return pS / max(float(np.sum(pS)), 1e-15)

    def geom_coherence(self) -> float:
        if not self.geometry_quantum:
            return 0.0
        tot = 0.0
        for s in range(self.S):
            rhoG = np.einsum('abii->ab', self.rho[s])
            rhoG = hermitize(rhoG)
            off = rhoG - np.diag(np.diag(rhoG))
            tot += float(np.linalg.norm(off) / max(np.linalg.norm(rhoG), 1e-15))
        return float(tot / max(float(self.S), 1.0))

    def K_expect(self, p_geom: np.ndarray) -> np.ndarray:
        return (p_geom[:, None] * self.K_cfg).sum(axis=0)

    def modular_sources_J(self, rho_m: np.ndarray) -> np.ndarray:
        J = np.zeros(self.N, dtype=float)
        for v, B in enumerate(self.balls):
            rhoB = partial_trace_qubits(rho_m, keep=B, N=self.N)
            rhoBref = partial_trace_qubits(self.rho_ref_pure, keep=B, N=self.N)
            J[v] = float(np.trace(rhoB @ self.KB_list[v]).real - np.trace(rhoBref @ self.KB_list[v]).real)
        return J

    def generalized_entropy(self, rho_m: np.ndarray, K_eff: np.ndarray) -> np.ndarray:
        out = np.zeros(self.N, dtype=float)
        for v, B in enumerate(self.balls):
            rhoB = partial_trace_qubits(rho_m, keep=B, N=self.N)
            out[v] = self.alpha_area * area_proxy(B, self.graph, K_eff, self.K_ref, self.regge_a) + von_neumann_entropy(rhoB)
        return out

    def negativity(self, rho_m: np.ndarray) -> float:
        p, q = self.neg_pair
        return negativity_two_qubits(partial_trace_qubits(rho_m, keep=[p, q], N=self.N))

    def total_trace(self) -> float:
        if self.geometry_quantum:
            return float(np.real(np.einsum('saaii->', self.rho)))
        return float(np.real(np.einsum('saii->', self.rho)))

    def _apply_matter_unitary(self, dt: float) -> None:
        dt = float(dt)
        U_list = [expm_hermitian(self.H_list[a], dt, factor=-1j) for a in range(self.M)]
        if self.geometry_quantum:
            for s in range(self.S):
                for a in range(self.M):
                    Ua = U_list[a]
                    for b in range(self.M):
                        Ub = U_list[b]
                        self.rho[s, a, b] = Ua @ self.rho[s, a, b] @ dagger(Ub)
        else:
            for s in range(self.S):
                for a in range(self.M):
                    Ua = U_list[a]
                    self.rho[s, a] = Ua @ self.rho[s, a] @ dagger(Ua)

    def _apply_geometry_unitary(self, dt: float) -> None:
        if not self.geometry_quantum or abs(self.geom_mix_rate) <= 0.0:
            return
        dt = float(dt)
        UG = expm_hermitian(self.HG, dt, factor=-1j)
        for s in range(self.S):
            self.rho[s] = np.einsum('aA,ABij,bB->abij', UG, self.rho[s], np.conjugate(UG))

    def _apply_unitary_strang(self, dt_half: float) -> None:
        dt_half = float(dt_half)
        if self.geometry_quantum and abs(self.geom_mix_rate) > 0.0:
            self._apply_matter_unitary(0.5 * dt_half)
            self._apply_geometry_unitary(dt_half)
            self._apply_matter_unitary(0.5 * dt_half)
        else:
            self._apply_matter_unitary(dt_half)

    def _apply_conditional_dephasing_exact(self, dt: float) -> None:
        dt = float(dt)
        for i in range(self.N):
            Zi = self.Z_ops[i]
            g = np.clip(self.gamma_deph[i, :], 0.0, None)
            if self.geometry_quantum:
                srt = np.sqrt(g)
                s_a = srt[:, None]
                s_b = srt[None, :]
                lam_plus = np.exp(-0.5 * (s_a - s_b) ** 2 * dt)
                lam_minus = np.exp(-0.5 * (s_a + s_b) ** 2 * dt)
                alpha = 0.5 * (lam_plus + lam_minus)
                beta = 0.5 * (lam_plus - lam_minus)
                for s in range(self.S):
                    rho_new = np.zeros_like(self.rho[s])
                    for a in range(self.M):
                        for b in range(self.M):
                            rab = self.rho[s, a, b]
                            rho_new[a, b] = float(alpha[a, b]) * rab + float(beta[a, b]) * (Zi @ rab @ Zi)
                    self.rho[s] = rho_new
            else:
                p = 0.5 * (1.0 - np.exp(-2.0 * g * dt))
                p = np.clip(p, 0.0, 1.0)
                for s in range(self.S):
                    for a in range(self.M):
                        pa = float(p[a])
                        if pa != 0.0:
                            self.rho[s, a] = (1.0 - pa) * self.rho[s, a] + pa * (Zi @ self.rho[s, a] @ Zi)

    def _apply_geometry_pointer_dephase(self, dt: float) -> None:
        if not self.geometry_quantum:
            return
        rate = float(self.geom_pointer_dephase)
        if rate <= 0.0:
            return
        fac = math.exp(-rate * float(dt))
        for s in range(self.S):
            for a in range(self.M):
                for b in range(self.M):
                    if a != b:
                        self.rho[s, a, b] *= fac

    def _compute_rates_from_current_J(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho_m = self.matter_marginal()
        J = self.modular_sources_J(rho_m)
        FJ = np.array([slice_F(self._k(self.K_cfg[a]), J, self.G, self.alpha_area) for a in range(self.M)], dtype=float)
        piJ = gibbs_pi(FJ, self.beta_geom)
        R = metropolis_rates(FJ, self.A_cfg, self.beta_geom, self.gamma_geom_base)
        out_rate = np.sum(R, axis=0)
        return (R, out_rate, piJ)

    def _apply_source_update_measurement(self) -> None:
        if self.backreaction != 'embedded':
            return
        assert self.sqrtE_source is not None
        S = self.S
        sqrtE = self.sqrtE_source
        if self.geometry_quantum:
            rho_in = self.rho
            rho_out = np.zeros_like(rho_in)
            for s_new in range(S):
                M = sqrtE[s_new]
                for s_old in range(S):
                    for a in range(self.M):
                        for b in range(self.M):
                            rho_out[s_new, a, b] += M @ rho_in[s_old, a, b] @ dagger(M)
            self.rho = rho_out
        else:
            rho_in = self.rho
            rho_out = np.zeros_like(rho_in)
            for s_new in range(S):
                M = sqrtE[s_new]
                for s_old in range(S):
                    for a in range(self.M):
                        rho_out[s_new, a] += M @ rho_in[s_old, a] @ dagger(M)
            self.rho = rho_out

    def _select_F_and_sqrtF(self, v: int, sgn: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.op_rate_family == 'scalar':
            I = np.eye(self.d, dtype=np.complex128)
            return (I, I)
        if self.op_rate_family == 'projector':
            if self.use_signed_monitors:
                if sgn >= 0:
                    return (self.Pi_full[v], self.M_full[v])
                return (self.Pi_bar_full[v], self.M_bar_full[v])
            return (self.Pi_full[v], self.M_full[v])
        if self.use_signed_monitors:
            if sgn >= 0:
                return (self.F_plus_full[v], self.sF_plus_full[v])
            return (self.F_minus_full[v], self.sF_minus_full[v])
        return (self.F_plus_full[v], self.sF_plus_full[v])

    def kraus_completeness_error_geometry_jump(self, dt: float, n_cfg_samples: int=3, seed: int=0) -> float:
        rng = np.random.default_rng(int(seed))
        I = np.eye(self.d, dtype=np.complex128)
        if self.backreaction == 'instantaneous':
            R, _, _ = self._compute_rates_from_current_J()
        elif self.backreaction == 'embedded':
            assert self.R_source is not None
            pS = self.source_prob()
            s = int(np.argmax(pS))
            R = self.R_source[s]
        else:
            R = self.R0
        bs = rng.choice(self.M, size=min(int(n_cfg_samples), self.M), replace=False)
        errs = []
        for b in bs:
            b = int(b)
            S_op = np.zeros((self.d, self.d), dtype=np.complex128)
            for a, v, sgn in self.neigh_by_source[b]:
                r = float(R[a, b])
                if r <= 0.0:
                    continue
                _, sF = self._select_F_and_sqrtF(v, sgn)
                S_op += r * effect_from_sqrt(sF)
            Q = hermitize(I - float(dt) * S_op)
            K0 = matrix_sqrt_psd(Q, eps=0.0)
            comp = dagger(K0) @ K0 + float(dt) * S_op
            errs.append(float(np.linalg.norm(comp - I) / max(np.linalg.norm(I), 1e-15)))
        return float(np.max(errs)) if errs else 0.0

    def _apply_geometry_jump_kraus(self, dt: float) -> None:
        dt = float(dt)
        if dt <= 0.0:
            return
        I = np.eye(self.d, dtype=np.complex128)
        if self.backreaction == 'instantaneous':
            R_scalar, _out_rate, _ = self._compute_rates_from_current_J()
            R_list = [R_scalar for _ in range(self.S)]
        elif self.backreaction == 'embedded':
            assert self.R_source is not None
            R_list = [self.R_source[s] for s in range(self.S)]
        else:
            R_list = [self.R0 for _ in range(self.S)]
        any_rate = False
        for s in range(self.S):
            if np.any(R_list[s] > 0.0):
                any_rate = True
                break
        if not any_rate:
            return
        psd_tol = 1e-12
        comp_tol = 5e-10
        max_halvings = 60
        min_dt_sub = dt * 1e-14
        remaining = dt
        while remaining > 0.0:
            dt_sub = remaining
            S_ops: List[List[np.ndarray]] = [[None for _ in range(self.M)] for _ in range(self.S)]
            jump_edges: List[List[List[Tuple[int, float, np.ndarray]]]] = [[[] for _ in range(self.M)] for _ in range(self.S)]
            max_lambda = 0.0
            for s in range(self.S):
                R = R_list[s]
                for b in range(self.M):
                    S_op = np.zeros((self.d, self.d), dtype=np.complex128)
                    edges_b: List[Tuple[int, float, np.ndarray]] = []
                    for a, v, sgn in self.neigh_by_source[b]:
                        r = float(R[a, b])
                        if not r > 0.0:
                            continue
                        _, sF = self._select_F_and_sqrtF(int(v), int(sgn))
                        E = effect_from_sqrt(sF)
                        S_op += r * E
                        edges_b.append((int(a), r, sF))
                    S_op = hermitize(S_op)
                    S_ops[s][b] = S_op
                    jump_edges[s][b] = edges_b
                    if edges_b:
                        lam = float(np.max(np.linalg.eigvalsh(S_op).real))
                        if lam > max_lambda:
                            max_lambda = lam
            if max_lambda <= 0.0:
                return
            dt_cap = float(self.jump_substep_safety) / max(max_lambda, 1e-15)
            dt_sub = min(dt_sub, dt_cap)
            for _ in range(max_halvings):
                if dt_sub < min_dt_sub:
                    raise FloatingPointError(f'geometry_jump: dt_sub underflow ({dt_sub:.3e}) while enforcing CPTP. Consider reducing outer dt or lowering gamma_geom_base.')
                ok = True
                K0_lists: List[List[np.ndarray]] = [[None for _ in range(self.M)] for _ in range(self.S)]
                for s in range(self.S):
                    for b in range(self.M):
                        S_op = S_ops[s][b]
                        Q = hermitize(I - dt_sub * S_op)
                        w = np.linalg.eigvalsh(Q).real
                        if float(np.min(w)) < -psd_tol:
                            ok = False
                            break
                        K0 = matrix_sqrt_psd(Q, eps=0.0)
                        K0_lists[s][b] = K0
                        comp = hermitize(dagger(K0) @ K0 + dt_sub * S_op)
                        err = float(np.linalg.norm(comp - I) / max(np.linalg.norm(I), 1e-15))
                        if err > comp_tol:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    if self.geometry_quantum:
                        rho_in = self.rho
                        rho_out = np.zeros_like(rho_in)
                        for s in range(self.S):
                            K0_list = K0_lists[s]
                            for a in range(self.M):
                                Ka = K0_list[a]
                                for b in range(self.M):
                                    Kb = K0_list[b]
                                    rho_out[s, a, b] += Ka @ rho_in[s, a, b] @ dagger(Kb)
                            for b in range(self.M):
                                rho_bb = rho_in[s, b, b]
                                for a, r, sF in jump_edges[s][b]:
                                    Jop = math.sqrt(dt_sub * float(r)) * sF
                                    rho_out[s, a, a] += Jop @ rho_bb @ dagger(Jop)
                        self.rho = rho_out
                    else:
                        rho_in = self.rho
                        rho_out = np.zeros_like(rho_in)
                        for s in range(self.S):
                            K0_list = K0_lists[s]
                            for b in range(self.M):
                                rho_out[s, b] += K0_list[b] @ rho_in[s, b] @ dagger(K0_list[b])
                            for b in range(self.M):
                                rho_bb = rho_in[s, b]
                                for a, r, sF in jump_edges[s][b]:
                                    Jop = math.sqrt(dt_sub * float(r)) * sF
                                    rho_out[s, a] += Jop @ rho_bb @ dagger(Jop)
                        self.rho = rho_out
                    remaining -= dt_sub
                    break
                dt_sub *= 0.5
            else:
                raise FloatingPointError('geometry_jump: failed to find a PSD+complete dt_sub after halving. Try reducing dt or gamma_geom_base.')
        return

        def _max_lambda_Sop(R: np.ndarray) -> float:
            mx = 0.0
            for b in range(self.M):
                S_op = np.zeros((self.d, self.d), dtype=np.complex128)
                for a, v, sgn in self.neigh_by_source[b]:
                    r = float(R[a, b])
                    if r <= 0.0:
                        continue
                    _, sF = self._select_F_and_sqrtF(v, sgn)
                    S_op += r * effect_from_sqrt(sF)
                lam = float(np.max(np.linalg.eigvalsh(hermitize(S_op)).real))
                mx = max(mx, lam)
            return mx
        max_lambda = 0.0
        for s in range(self.S):
            max_lambda = max(max_lambda, _max_lambda_Sop(R_list[s]))
        if max_lambda <= 0.0:
            return
        dt_max = float(self.jump_substep_safety) / max(max_lambda, 1e-15)
        n_sub = int(math.ceil(dt / dt_max))
        dt_sub = dt / max(n_sub, 1)
        for _ in range(n_sub):
            if self.geometry_quantum:
                rho_in = self.rho
                rho_out = np.zeros_like(rho_in)
                for s in range(self.S):
                    R = R_list[s]
                    K0_list: List[np.ndarray] = []
                    for b in range(self.M):
                        S_op = np.zeros((self.d, self.d), dtype=np.complex128)
                        for a, v, sgn in self.neigh_by_source[b]:
                            r = float(R[a, b])
                            if r <= 0.0:
                                continue
                            _, sF = self._select_F_and_sqrtF(v, sgn)
                            S_op += r * effect_from_sqrt(sF)
                        Q = hermitize(I - dt_sub * S_op)
                        K0_list.append(matrix_sqrt_psd(Q, eps=0.0))
                    for a in range(self.M):
                        Ka = K0_list[a]
                        for b in range(self.M):
                            Kb = K0_list[b]
                            rho_out[s, a, b] += Ka @ rho_in[s, a, b] @ dagger(Kb)
                    for b in range(self.M):
                        rho_bb = rho_in[s, b, b]
                        for a, v, sgn in self.neigh_by_source[b]:
                            r = float(R[a, b])
                            if r <= 0.0:
                                continue
                            _, sF = self._select_F_and_sqrtF(v, sgn)
                            Jop = math.sqrt(dt_sub * r) * sF
                            rho_out[s, a, a] += Jop @ rho_bb @ dagger(Jop)
                self.rho = rho_out
            else:
                rho_in = self.rho
                rho_out = np.zeros_like(rho_in)
                for s in range(self.S):
                    R = R_list[s]
                    K0_list: List[np.ndarray] = []
                    for b in range(self.M):
                        S_op = np.zeros((self.d, self.d), dtype=np.complex128)
                        for a, v, sgn in self.neigh_by_source[b]:
                            r = float(R[a, b])
                            if r <= 0.0:
                                continue
                            F, _ = self._select_F_and_sqrtF(v, sgn)
                            S_op += r * F
                        Q = hermitize(I - dt_sub * S_op)
                        K0_list.append(matrix_sqrt_psd(Q, eps=0.0))
                    for b in range(self.M):
                        rho_out[s, b] += K0_list[b] @ rho_in[s, b] @ dagger(K0_list[b])
                    for b in range(self.M):
                        rho_bb = rho_in[s, b]
                        for a, v, sgn in self.neigh_by_source[b]:
                            r = float(R[a, b])
                            if r <= 0.0:
                                continue
                            _, sF = self._select_F_and_sqrtF(v, sgn)
                            Jop = math.sqrt(dt_sub * r) * sF
                            rho_out[s, a] += Jop @ rho_bb @ dagger(Jop)
                self.rho = rho_out

    def _apply_causal_channel(self) -> None:
        if self.causal_apply_every <= 0:
            return
        if self.geometry_quantum:
            for s in range(self.S):
                for a in range(self.M):
                    for b in range(self.M):
                        self.rho[s, a, b] = self.causal.apply_global(self.rho[s, a, b])
        else:
            for s in range(self.S):
                for a in range(self.M):
                    self.rho[s, a] = self.causal.apply_global(self.rho[s, a])

    def _post_step_trace_check(self, where: str) -> None:
        tr = self.total_trace()
        if not np.isfinite(tr) or tr <= 1e-15:
            raise FloatingPointError(f'Trace invalid at {where}: Tr={tr}')
        drift = float(tr - 1.0)
        if abs(drift) <= self.trace_tol:
            return
        if self.trace_mode == 'unnormalized':
            return
        if self.trace_mode == 'hygiene' and abs(drift) <= self.trace_hygiene_tol:
            self.rho /= tr
            self._trace_renorm_count += 1
            return
        raise FloatingPointError(f"Trace drift at {where}: Tr={tr:.16f} (drift={drift:+.3e}). Refuse renormalization in trace_mode='{self.trace_mode}'. Mitigation: reduce dt, increase jump_substep_safety, or inspect operator bounds.")

    def solve_G(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, float)
        if self.volume_constraint_active:
            b = project_zero_mean(b)
            G_reg = self.G + 1e-12 * np.eye(self.N)
            x = np.linalg.solve(G_reg, b)
            return project_zero_mean(x)
        return np.linalg.solve(self.G, b)

    def step_strang(self, dt: float) -> None:
        dt = float(dt)
        if dt <= 0.0:
            return
        self._apply_unitary_strang(0.5 * dt)
        self._apply_conditional_dephasing_exact(dt)
        self._apply_geometry_pointer_dephase(dt)
        if self.backreaction == 'embedded':
            self._apply_source_update_measurement()
        self._apply_geometry_jump_kraus(dt)
        self._apply_unitary_strang(0.5 * dt)
        self._post_step_trace_check(where='step_strang')

    def run(self, T_final: float, dt: float, store_every: int=1, label: str='', progress_every: int=25, store_matter_states: bool=False) -> Dict[str, Any]:
        T_final = float(T_final)
        dt = float(dt)
        store_every = int(store_every)
        n_steps = int(round(T_final / dt))
        t = 0.0
        self.times.clear()
        self.K_eff_hist.clear()
        self.K_eq_hist.clear()
        self.res_hist.clear()
        self.neg_hist.clear()
        self.Srel_hist.clear()
        self.Tdist_hist.clear()
        self.Tdist_mix_hist.clear()
        self.Tdist_step_hist.clear()
        self.purity_hist.clear()
        self.matter_entropy_hist.clear()
        self.ent_entropyA_hist.clear()
        self.ent_entropyB_hist.clear()
        self.mutual_info_hist.clear()
        self.log_negativity_AB_hist.clear()
        self.coh_matter_hist.clear()
        self.matter_min_eig_hist.clear()
        self.CE_hist.clear()
        self.FI_sld_hist.clear()
        self.FI_cls_hist.clear()
        self.KL_hist.clear()
        self.TV_hist.clear()
        self.coh_hist.clear()
        self.p_hist.clear()
        self.piJ_hist.clear()
        self.J_hist.clear()
        self.Sgen_hist.clear()
        self.trace_hist.clear()
        self.CE_path_hist.clear()
        self.FI_path_hist.clear()
        self.J_est_hist.clear()
        self.pS_hist.clear()
        self.kraus_tp_err0 = self.kraus_completeness_error_geometry_jump(dt=float(dt), n_cfg_samples=3, seed=123)
        self.rho_m_hist = [] if store_matter_states else None
        rho_prev_store: Optional[np.ndarray] = None
        CE_cum = 0.0
        FI_cum = 0.0
        t_wall0 = time.time()
        if progress_every and progress_every > 0:
            print(f'[RUN] {label}\n      n_steps={n_steps}, dt={dt:g}, store_every={store_every}, N={self.N}, M={self.M}, S={self.S}, geometry_quantum={self.geometry_quantum}, mode={self.rate_mode} (backreaction={self.backreaction}, op_rates={self.op_rate_family})', flush=True)
        for step in range(n_steps + 1):
            do_store = step % store_every == 0 or step == 0 or step == n_steps
            if do_store:
                rho_m = self.matter_marginal()
                J = self.modular_sources_J(rho_m)
                p_geom = self.geom_prob()
                K_eff = self.K_expect(p_geom)
                k_eq = self.solve_G(-1.0 / max(self.alpha_area, 1e-15) * J)
                if self.volume_constraint_active:
                    k_eq = project_zero_mean(k_eq)
                K_eq = self.K_ref + k_eq
                res = self.alpha_area * (self.G @ self._k(K_eff)) + J
                resn = float(np.linalg.norm(res) / math.sqrt(self.N))
                FJ = np.array([slice_F(self._k(self.K_cfg[a]), J, self.G, self.alpha_area) for a in range(self.M)], dtype=float)
                piJ = gibbs_pi(FJ, self.beta_geom)
                Sgen = self.generalized_entropy(rho_m, K_eff)
                neg = self.negativity(rho_m)
                ce = canonical_energy_metrics(rho_m, self.rho_ref)
                if rho_prev_store is None:
                    CE_cum = 0.0
                    FI_cum = 0.0
                    trdist_step = 0.0
                else:
                    Delta = rho_m - rho_prev_store
                    q = bkm_sld_classical_quadratic_from_ref(self.rho_ref, Delta, eps=1e-12)
                    CE_cum += float(q['CE_BKM_quad'])
                    FI_cum += float(q['FI_SLD'])
                    trdist_step = float(trace_distance(rho_m, rho_prev_store))
                rho_prev_store = rho_m.copy()
                pS = self.source_prob()
                if self.backreaction == 'embedded' and self.J_source is not None:
                    J_est = (pS[:, None] * self.J_source).sum(axis=0)
                else:
                    J_est = np.zeros_like(J)
                trdist_ref = float(trace_distance(rho_m, self.rho_ref))
                trdist_mix = float(trace_distance(rho_m, np.eye(self.d, dtype=np.complex128) / self.d))
                purity = float(purity_trace(rho_m))
                coh_m = float(l1_coherence_computational_basis(rho_m))
                S_matter = float(von_neumann_entropy(rho_m))
                if self.N >= 2:
                    A_part = list(range(max(1, self.N // 2)))
                    B_part = [i for i in range(self.N) if i not in A_part]
                    SA = float(von_neumann_entropy(partial_trace_qubits(rho_m, keep=A_part, N=self.N)))
                    SB = float(von_neumann_entropy(partial_trace_qubits(rho_m, keep=B_part, N=self.N)))
                    MI = float(SA + SB - S_matter)
                    LN = float(log_negativity_bipartition_qubits(rho_m, A_part, self.N))
                else:
                    SA = 0.0
                    SB = 0.0
                    MI = 0.0
                    LN = 0.0
                min_eig = float(np.min(np.linalg.eigvalsh(hermitize(rho_m)).real))
                if min_eig < -1e-10:
                    print(f'[CPTP WARNING] matter state min eigenvalue={min_eig:+.3e} at t={t:.6g}', flush=True)
                self.times.append(t)
                self.K_eff_hist.append(K_eff)
                self.K_eq_hist.append(K_eq)
                self.res_hist.append(resn)
                self.neg_hist.append(float(neg))
                self.Srel_hist.append(float(ce['Srel']))
                self.Tdist_hist.append(trdist_ref)
                self.Tdist_mix_hist.append(trdist_mix)
                self.Tdist_step_hist.append(float(trdist_step))
                self.purity_hist.append(purity)
                self.matter_entropy_hist.append(S_matter)
                self.ent_entropyA_hist.append(SA)
                self.ent_entropyB_hist.append(SB)
                self.mutual_info_hist.append(MI)
                self.log_negativity_AB_hist.append(LN)
                self.coh_matter_hist.append(coh_m)
                self.matter_min_eig_hist.append(min_eig)
                self.CE_hist.append(float(ce['CE_BKM_quad']))
                self.FI_sld_hist.append(float(ce['FI_SLD']))
                self.FI_cls_hist.append(float(ce['FI_classical_eigbasis']))
                self.KL_hist.append(kl_divergence(p_geom, piJ))
                self.TV_hist.append(total_variation(p_geom, piJ))
                self.coh_hist.append(self.geom_coherence())
                self.p_hist.append(p_geom)
                self.piJ_hist.append(piJ)
                self.J_hist.append(J)
                self.J_est_hist.append(J_est)
                self.pS_hist.append(pS)
                self.Sgen_hist.append(Sgen)
                self.trace_hist.append(self.total_trace())
                self.CE_path_hist.append(float(CE_cum))
                self.FI_path_hist.append(float(FI_cum))
                if self.rho_m_hist is not None:
                    self.rho_m_hist.append(rho_m.copy())
                if progress_every and progress_every > 0:
                    if step % progress_every == 0 or step == n_steps:
                        pct = 100.0 * float(step) / max(float(n_steps), 1.0)
                        dt_wall = time.time() - t_wall0
                        print(f"[RUN] step={step:5d}/{n_steps:5d} ({pct:6.2f}%)  t={t:8.4f}  res={resn:9.3e}  neg={neg:7.4f}  Srel={ce['Srel']:.3e}  CEhist={CE_cum:.3e}  Tr={self.trace_hist[-1]:.9f}  wall={dt_wall:7.2f}s", flush=True)
            if step == n_steps:
                break
            self.step_strang(dt)
            tr = self.total_trace()
            if not np.isfinite(tr) or tr <= 1e-15:
                raise FloatingPointError(f'Trace invalid after evolution: Tr={tr}')
            t += dt
            if self.causal_apply_every > 0 and (step + 1) % self.causal_apply_every == 0:
                self._apply_causal_channel()
        summary = {'label': label, 'mode': 'gTRQC_register_joint' if self.geometry_quantum else 'gTRQC_register_CQ', 'rate_mode': self.rate_mode, 'backreaction': self.backreaction, 'op_rate_family': self.op_rate_family, 'signed_monitors': bool(self.use_signed_monitors), 'geometry_quantum': bool(self.geometry_quantum), 'N': int(self.N), 'M': int(self.M), 'S': int(self.S), 'graph': self.graph.name, 't_final': float(self.times[-1]) if self.times else 0.0, 'residual_norm_final': float(self.res_hist[-1]) if self.res_hist else 0.0, 'Srel_final': float(self.Srel_hist[-1]) if self.Srel_hist else 0.0, 'CE_BKM_final': float(self.CE_hist[-1]) if self.CE_hist else 0.0, 'FI_SLD_final': float(self.FI_sld_hist[-1]) if self.FI_sld_hist else 0.0, 'FI_class_final': float(self.FI_cls_hist[-1]) if self.FI_cls_hist else 0.0, 'neg_final': float(self.neg_hist[-1]) if self.neg_hist else 0.0, 'KL_geom_final': float(self.KL_hist[-1]) if self.KL_hist else 0.0, 'TV_geom_final': float(self.TV_hist[-1]) if self.TV_hist else 0.0, 'dK_final_absmean': float(np.mean(np.abs(self.K_eff_hist[-1] - self.K_ref))) if self.K_eff_hist else 0.0, 'dK_final_absmax': float(np.max(np.abs(self.K_eff_hist[-1] - self.K_ref))) if self.K_eff_hist else 0.0, 'Sgen_final_mean': float(np.mean(self.Sgen_hist[-1])) if self.Sgen_hist else 0.0, 'Sgen_final_max': float(np.max(self.Sgen_hist[-1])) if self.Sgen_hist else 0.0, 'trdist_to_ref_final': float(self.Tdist_hist[-1]) if self.Tdist_hist else 0.0, 'trdist_to_mix_final': float(self.Tdist_mix_hist[-1]) if self.Tdist_mix_hist else 0.0, 'trdist_to_prev_final': float(self.Tdist_step_hist[-1]) if self.Tdist_step_hist else 0.0, 'trdist_ref_baseline_pure_to_mix': float(1.0 - 1.0 / max(float(self.d), 1.0)), 'purity_final': float(self.purity_hist[-1]) if self.purity_hist else 0.0, 'matter_entropy_final': float(self.matter_entropy_hist[-1]) if self.matter_entropy_hist else 0.0, 'ent_entropyA_final': float(self.ent_entropyA_hist[-1]) if self.ent_entropyA_hist else 0.0, 'ent_entropyB_final': float(self.ent_entropyB_hist[-1]) if self.ent_entropyB_hist else 0.0, 'mutual_info_final': float(self.mutual_info_hist[-1]) if self.mutual_info_hist else 0.0, 'log_negativity_AB_final': float(self.log_negativity_AB_hist[-1]) if self.log_negativity_AB_hist else 0.0, 'coherence_l1_matter_final': float(self.coh_matter_hist[-1]) if self.coh_matter_hist else 0.0, 'matter_min_eig_final': float(self.matter_min_eig_hist[-1]) if self.matter_min_eig_hist else 0.0, 'effect_eig_min': float(getattr(self, 'effect_eig_min', float('nan'))), 'effect_eig_max': float(getattr(self, 'effect_eig_max', float('nan'))), 'effect_eig_violation_count': int(getattr(self, 'effect_eig_violation_count', 0)), 'geom_coherence_final': float(self.coh_hist[-1]) if self.coh_hist else 0.0, 'trace_final': float(self.trace_hist[-1]) if self.trace_hist else 0.0, 'trace_drift_abs': float(abs((self.trace_hist[-1] if self.trace_hist else 0.0) - 1.0)), 'CE_BKM_history_final': float(self.CE_path_hist[-1]) if self.CE_path_hist else 0.0, 'FI_SLD_history_final': float(self.FI_path_hist[-1]) if self.FI_path_hist else 0.0, 'causal_apply_every': int(self.causal_apply_every), 'causal_structure_meta': self.causal.meta, 'causal_order_independence_error_on_ref': float(self.causal_order_err), 'causal_no_signaling_error_edge0_on_ref': float(self.causal_nosig_err_edge0), 'causal_gr_transport': bool(self.causal_gr_transport), 'causal_gr_data': self.causal_gr_data, 'mass_m': float(self.mass_m), 'volume_constraint_active': bool(self.volume_constraint_active), 'trace_mode': str(self.trace_mode), 'trace_tol': float(self.trace_tol), 'trace_hygiene_tol': float(self.trace_hygiene_tol), 'trace_hygiene_renorm_count': int(getattr(self, '_trace_renorm_count', 0)), 'rho_bytes_est': int(getattr(self, 'rho_bytes_est', 0)), 'kraus_completeness_err0': float(getattr(self, 'kraus_tp_err0', 0.0)), 'rho_gib_est': float(getattr(self, 'rho_gib_est', 0.0))}
        if progress_every and progress_every > 0:
            print(f"[RUN] DONE: {label}  residual={summary['residual_norm_final']:.3e}  Tr={summary['trace_final']:.9f}", flush=True)
        return summary

def canonical_energy_second_variation(base_params: Dict[str, Any], T_final: float, dt: float, store_every: int, lambda_val: float=0.002, seed: int=0, label: str='CE2') -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    sim0 = GTRQCSimulator(**base_params)
    d = sim0.d
    H = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    H = hermitize(H)
    U = expm_hermitian(H, t=float(lambda_val), factor=-1j)
    Udag = dagger(U)
    rho0 = sim0.rho0
    rho_p = U @ rho0 @ Udag
    rho_m = Udag @ rho0 @ U
    sum0 = sim0.run(T_final=T_final, dt=dt, store_every=store_every, label=f'{label} baseline', progress_every=0, store_matter_states=True)
    p_params = dict(base_params)
    p_params['initial_rho_override'] = rho_p
    m_params = dict(base_params)
    m_params['initial_rho_override'] = rho_m
    simp = GTRQCSimulator(**p_params)
    simm = GTRQCSimulator(**m_params)
    _ = simp.run(T_final=T_final, dt=dt, store_every=store_every, label=f'{label} +λ', progress_every=0, store_matter_states=True)
    _ = simm.run(T_final=T_final, dt=dt, store_every=store_every, label=f'{label} -λ', progress_every=0, store_matter_states=True)
    if sim0.rho_m_hist is None or simp.rho_m_hist is None or simm.rho_m_hist is None:
        raise RuntimeError('Expected stored matter states.')
    times = np.array(sim0.times, float)
    ce2 = np.zeros_like(times)
    srel_p = np.zeros_like(times)
    srel_m = np.zeros_like(times)
    for k in range(times.size):
        r0 = sim0.rho_m_hist[k]
        rp = simp.rho_m_hist[k]
        rm = simm.rho_m_hist[k]
        srel_p[k] = relative_entropy(rp, r0, eps=1e-12)
        srel_m[k] = relative_entropy(rm, r0, eps=1e-12)
        ce2[k] = (srel_p[k] + srel_m[k]) / max(float(lambda_val * lambda_val), 1e-30)
    return {'ok': True, 'lambda': float(lambda_val), 'seed': int(seed), 'times': times, 'Srel_plus': srel_p, 'Srel_minus': srel_m, 'CE2_second_variation': ce2, 'CE2_final': float(ce2[-1]) if ce2.size else 0.0, 'baseline_summary': sum0, 'note': 'Discrete second-derivative proxy for canonical energy defined as d^2/dλ^2 Srel(ρ_out(λ)||ρ_out(0))|_{λ=0}.'}

def plot_sim_timeseries(sim: GTRQCSimulator, title: str) -> List[plt.Figure]:
    t = np.array(sim.times, float)
    K = np.stack(sim.K_eff_hist, axis=0)
    Keq = np.stack(sim.K_eq_hist, axis=0)
    res = np.array(sim.res_hist, float)
    J = np.stack(sim.J_hist, axis=0)
    Jest = np.stack(sim.J_est_hist, axis=0) if sim.J_est_hist else np.zeros_like(J)
    Sgen = np.stack(sim.Sgen_hist, axis=0)
    neg = np.array(sim.neg_hist, float)
    Srel = np.array(sim.Srel_hist, float)
    Td = np.array(sim.Tdist_hist, float)
    CE = np.array(sim.CE_hist, float)
    FI = np.array(sim.FI_sld_hist, float)
    FIc = np.array(sim.FI_cls_hist, float)
    CEh = np.array(sim.CE_path_hist, float) if sim.CE_path_hist else np.zeros_like(t)
    FIh = np.array(sim.FI_path_hist, float) if sim.FI_path_hist else np.zeros_like(t)
    p = np.stack(sim.p_hist, axis=0)
    piJ = np.stack(sim.piJ_hist, axis=0)
    KL = np.array(sim.KL_hist, float)
    TV = np.array(sim.TV_hist, float)
    coh = np.array(sim.coh_hist, float)
    pS = np.stack(sim.pS_hist, axis=0) if sim.pS_hist else None
    figs: List[plt.Figure] = []
    fig, ax = plt.subplots(figsize=(11, 6))
    for i in range(sim.N):
        ax.plot(t, K[:, i], label=f'K_eff[{i}]')
        ax.plot(t, Keq[:, i], '--', alpha=0.6, label=f'K_eq[{i}]')
    ax.set_title(f'{title}: K_eff(t) vs diagnostic entanglement-equilibrium K_eq(t)')
    ax.set_xlabel('t')
    ax.set_ylabel('K')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    figs.append(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, res, color='C3')
    ax.set_title('Einstein/TRQC residual ||αG(K_eff-K_ref)+J||/sqrt(N)')
    ax.set_xlabel('t')
    ax.set_ylabel('residual')
    fig.tight_layout()
    figs.append(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i in range(sim.N):
        ax.plot(t, J[:, i], label=f'J[{i}]')
        if sim.backreaction == 'embedded':
            ax.plot(t, Jest[:, i], '--', alpha=0.75, label=f'J_est[{i}]')
    ax.set_title('Modular-energy sources J_v(t) (solid) and embedded estimate J_est (dashed)')
    ax.set_xlabel('t')
    ax.set_ylabel('J')
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    figs.append(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, np.mean(Sgen, axis=1), label='mean_v S_gen', color='C2')
    ax.fill_between(t, np.min(Sgen, axis=1), np.max(Sgen, axis=1), color='C2', alpha=0.15, label='min..max over balls')
    ax.set_title('Generalized entropy S_gen over TRQC balls')
    ax.set_xlabel('t')
    ax.set_ylabel('S_gen')
    ax.legend()
    fig.tight_layout()
    figs.append(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, neg, label='negativity', color='C0')
    ax2 = ax.twinx()
    ax2.plot(t, Srel, label='S(ρ||ρ_ref)', color='C1', alpha=0.85)
    ax2.plot(t, Td, label='0.5||ρ-ρ_ref||₁', color='C3', alpha=0.85)
    ax.set_title('Entanglement negativity and information distances')
    ax.set_xlabel('t')
    ax.set_ylabel('negativity')
    ax2.set_ylabel('distance')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    fig.tight_layout()
    figs.append(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, Srel, label='S(ρ||ρ_ref)', color='C1')
    ax.plot(t, CE, label='½ g_BKM(ρ-ρ_ref, ρ-ρ_ref)', color='C4')
    ax.plot(t, 0.25 * FI, label='¼ FI_SLD proxy', color='C5', alpha=0.9)
    ax.plot(t, FIc, label='FI_classical (eigbasis)', color='C6', alpha=0.8)
    ax.set_title('Instantaneous canonical-energy diagnostics vs Fisher proxies')
    ax.set_xlabel('t')
    ax.set_ylabel('value')
    ax.legend()
    fig.tight_layout()
    figs.append(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, CEh, label='CE_BKM_history (cumulative)', color='C4')
    ax2 = ax.twinx()
    ax2.plot(t, FIh, label='FI_SLD_history (cumulative)', color='C5', alpha=0.85)
    ax.set_title('History-level BKM path proxy (kept) at ρ_ref')
    ax.set_xlabel('t')
    ax.set_ylabel('CE_history')
    ax2.set_ylabel('FI_history')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    fig.tight_layout()
    figs.append(fig)
    mean_p = np.mean(p, axis=0)
    top = np.argsort(-mean_p)[:min(8, sim.M)]
    fig, ax = plt.subplots(figsize=(11, 5))
    for a in top:
        ax.plot(t, p[:, a], label=f'p[a={a}]')
        ax.plot(t, piJ[:, a], '--', alpha=0.55)
    ax.set_title('Geometry distribution p(a) vs diagnostic Gibbs π_J(a) (top configs)')
    ax.set_xlabel('t')
    ax.set_ylabel('probability')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    figs.append(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, KL, label='KL(p||π_J)', color='C4')
    ax.plot(t, TV, label='TV(p,π_J)', color='C5')
    ax.plot(t, coh, label='geometry coherence', color='C6')
    ax.set_title('Geometry diagnostics (π_J diagnostic)')
    ax.set_xlabel('t')
    ax.set_ylabel('metric')
    ax.legend()
    fig.tight_layout()
    figs.append(fig)
    if pS is not None and pS.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(11, 4))
        topS = np.argsort(-np.mean(pS, axis=0))[:min(6, pS.shape[1])]
        for s in topS:
            ax.plot(t, pS[:, s], label=f'pS[{s}]')
        ax.set_title('Source register distribution pS(s) (linear Markovian embedding)')
        ax.set_xlabel('t')
        ax.set_ylabel('probability')
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        figs.append(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, np.array(sim.trace_hist, float), color='C7')
    ax.set_title('Trace of joint state (should stay ~1 under CPTP splitting)')
    ax.set_xlabel('t')
    ax.set_ylabel('Tr(ρ)')
    fig.tight_layout()
    figs.append(fig)
    if hasattr(sim, 'gauss_bonnet_total_hist') and sim.gauss_bonnet_total_hist:
        gb_tot = np.array(sim.gauss_bonnet_total_hist, float)
        gb_exp = float(getattr(sim, 'gauss_bonnet_expected_total', 0.0))
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(np.array(sim.geom_times_hist, float), gb_tot, label='∑ δ(v) (total deficit)', color='C2')
        ax.axhline(gb_exp, color='k', ls='--', lw=1.0, label=f'2πχ = {gb_exp:.6g}')
        ax.set_title('Gauss–Bonnet check: total curvature invariant vs time')
        ax.set_xlabel('t')
        ax.set_ylabel('total deficit')
        ax.legend()
        fig.tight_layout()
        figs.append(fig)
    if hasattr(sim, 'wilson_angle_mean_hist') and sim.wilson_angle_mean_hist:
        wt = np.array(sim.wilson_tr_mean_hist, float)
        wa = np.array(sim.wilson_angle_mean_hist, float)
        tt = np.array(sim.geom_times_hist, float)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(tt, wt, label='mean Re(tr(WΔ)/2)', color='C4')
        ax2 = ax.twinx()
        ax2.plot(tt, wa, label='mean loop angle θ', color='C5', alpha=0.85)
        ax.set_title('Holonomy control: Wilson loops over triangles')
        ax.set_xlabel('t')
        ax.set_ylabel('Re(tr(W)/2)')
        ax2.set_ylabel('θ (radians)')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        fig.tight_layout()
        figs.append(fig)
    return figs

def geometry_builder(kind: str, seed: int) -> TriangulatedSurface:
    kind = kind.strip().lower()
    if kind == 'tetra':
        return make_irregular_tetrahedron_surface(seed=seed, perturb=0.18, scale_xyz=(1.0, 1.0, 1.0))
    if kind == 'octa':
        return make_irregular_octahedron_surface(seed=seed, perturb=0.2, scale_xyz=(1.0, 1.2, 0.85))
    if kind == 'cube':
        return make_irregular_cube_surface(seed=seed, perturb=0.12, scale_xyz=(1.0, 1.08, 0.95))
    raise ValueError(f"Unknown geometry kind '{kind}'")

def per_geometry_default_params(kind: str) -> Dict[str, Any]:
    kind = kind.strip().lower()
    if kind == 'tetra':
        return dict(max_configs=12, levels_per_node=5, deltaK_scale=0.18, geometry_quantum=True)
    if kind == 'octa':
        return dict(max_configs=10, levels_per_node=5, deltaK_scale=0.18, geometry_quantum=False)
    if kind == 'cube':
        return dict(max_configs=8, levels_per_node=5, deltaK_scale=0.16, geometry_quantum=False)
    return dict(max_configs=10, levels_per_node=5, deltaK_scale=0.18, geometry_quantum=False)

def per_mode_params(mode: str) -> Dict[str, Any]:
    mode = mode.strip().lower()
    alias = GTRQCSimulator._MODE_ALIASES.get(mode, mode)
    return dict(rate_mode=alias)

def summarize_folds(rows: List[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in keys:
        vals = []
        for r in rows:
            if k in r and isinstance(r[k], (int, float, np.floating)) and np.isfinite(r[k]):
                vals.append(float(r[k]))
        if vals:
            out[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 'n': len(vals)}
    return out

def _metric_mean(agg: Dict[str, Any], key: str, default: float=float('inf')) -> float:
    v = agg.get(key, None)
    if isinstance(v, dict) and 'mean' in v:
        try:
            return float(v['mean'])
        except Exception:
            return float(default)
    return float(default)

def block_meets_constraints(block: Dict[str, Any], constraints: Dict[str, float]) -> bool:
    agg = block.get('aggregate_metrics', {})
    for k, thr in constraints.items():
        val = _metric_mean(agg, k, default=float('inf'))
        if not np.isfinite(val) or val > float(thr):
            return False
    return True

def pareto_non_dominated_indices(values: np.ndarray) -> List[int]:
    values = np.asarray(values, dtype=float)
    n = int(values.shape[0])
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if np.all(values[j] <= values[i]) and np.any(values[j] < values[i]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]

def select_pareto_blocks(blocks: List[Dict[str, Any]], objectives: Sequence[str], constraints: Dict[str, float]) -> List[Dict[str, Any]]:
    feasible = [b for b in blocks if block_meets_constraints(b, constraints)]
    if not feasible:
        return []
    vals = np.array([[_metric_mean(b.get('aggregate_metrics', {}), k, default=float('inf')) for k in objectives] for b in feasible], dtype=float)
    idx = pareto_non_dominated_indices(vals)
    selected = [feasible[i] for i in idx]
    selected.sort(key=lambda b: tuple((_metric_mean(b.get('aggregate_metrics', {}), k, default=float('inf')) for k in objectives)))
    return selected
import copy
import math
import os
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def so3_from_axis_angle(axis: np.ndarray, angle: float, eps: float=1e-15) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).reshape(3)
    n = float(np.linalg.norm(axis))
    if n <= eps:
        return np.eye(3, dtype=float)
    a = axis / n
    ax, ay, az = (float(a[0]), float(a[1]), float(a[2]))
    K = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]], dtype=float)
    I = np.eye(3, dtype=float)
    s = math.sin(float(angle))
    c = math.cos(float(angle))
    return I + s * K + (1.0 - c) * (K @ K)

def copy_surface(surface: 'TriangulatedSurface', name_suffix: str='_copy') -> 'TriangulatedSurface':
    return TriangulatedSurface(points=np.array(surface.points, dtype=float, copy=True), triangles=np.array(surface.triangles, dtype=np.int64, copy=True), name=str(surface.name) + str(name_suffix), closed=bool(surface.closed))

def euler_characteristic(surface: TriangulatedSurface) -> int:
    V = int(surface.points.shape[0])
    F = int(surface.triangles.shape[0])
    edges = set()
    T = np.asarray(surface.triangles, dtype=np.int64)
    for i, j, k in T:
        for a, b in ((i, j), (j, k), (k, i)):
            aa, bb = (int(a), int(b))
            if aa > bb:
                aa, bb = (bb, aa)
            edges.add((aa, bb))
    E = len(edges)
    return V - E + F

def gauss_bonnet_target(surface: TriangulatedSurface) -> float:
    chi = euler_characteristic(surface)
    return 2.0 * math.pi * float(chi)

def _direction_to_sign(direction: str) -> int:
    d = str(direction).strip().lower()
    if d in ('cw', 'clockwise', 'right', 'neg', '-1', '-'):
        return -1
    if d in ('ccw', 'anticlockwise', 'counterclockwise', 'left', 'pos', '+1', '+'):
        return +1
    return +1

def edge_set_from_triangles(triangles: np.ndarray) -> set:
    T = np.asarray(triangles, dtype=np.int64)
    edges = set()
    for i, j, k in T:
        for a, b in ((i, j), (j, k), (k, i)):
            aa, bb = (int(a), int(b))
            if aa > bb:
                aa, bb = (bb, aa)
            edges.add((aa, bb))
    return edges

def _triangle_angle_at(P: np.ndarray, a: int, b: int, c: int) -> float:
    va = P[int(a)]
    vb = P[int(b)]
    vc = P[int(c)]
    u = vb - va
    v = vc - va
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu <= 1e-15 or nv <= 1e-15:
        return 0.0
    cang = float(np.dot(u, v) / (nu * nv))
    cang = max(min(cang, 1.0), -1.0)
    return float(math.acos(cang))

def delaunay_violation_amount(surface: TriangulatedSurface, edge: Tuple[int, int]) -> float:
    P = np.asarray(surface.points, dtype=float)
    T = np.asarray(surface.triangles, dtype=np.int64)
    a, b = (int(edge[0]), int(edge[1]))
    aa, bb = (a, b) if a < b else (b, a)
    e2t = surface_edges(T)
    if (aa, bb) not in e2t or len(e2t[aa, bb]) != 2:
        return 0.0
    t1, t2 = e2t[aa, bb]
    tri1 = T[t1].tolist()
    tri2 = T[t2].tolist()
    k = int(list(set(tri1) - {aa, bb})[0])
    l = int(list(set(tri2) - {aa, bb})[0])
    ang_k = _triangle_angle_at(P, k, aa, bb)
    ang_l = _triangle_angle_at(P, l, aa, bb)
    return float(ang_k + ang_l - math.pi)

def apply_local_delaunay_flips_inplace(surface: TriangulatedSurface, tol: float=1e-06, max_flips: int=64) -> Dict[str, Any]:
    flips: List[Tuple[int, int]] = []
    max_violation = 0.0
    for _ in range(int(max_flips)):
        cand = flippable_edges_2_2(surface)
        if not cand:
            break
        best_edge = None
        best_v = 0.0
        for e in cand:
            v = delaunay_violation_amount(surface, e)
            if v > best_v:
                best_v = v
                best_edge = e
        max_violation = max(max_violation, float(best_v))
        if best_edge is None or best_v <= float(tol):
            break
        surf2 = pachner_flip_2_2(surface, edge=best_edge)
        surface.triangles = np.array(surf2.triangles, dtype=np.int64, copy=True)
        flips.append((int(best_edge[0]), int(best_edge[1])))
    return {'n_flips': int(len(flips)), 'max_violation': float(max_violation), 'flips': flips}

def wilson_loop_triangle_stats(surface: TriangulatedSurface, frames: Sequence[np.ndarray]) -> Dict[str, float]:
    T = np.asarray(surface.triangles, dtype=np.int64)
    if T.size == 0:
        return {'mean_tr_over_2_real': 0.0, 'std_tr_over_2_real': 0.0, 'mean_loop_angle': 0.0, 'std_loop_angle': 0.0}
    tr_vals = []
    ang_vals = []
    for i, j, k in T:
        i = int(i)
        j = int(j)
        k = int(k)
        Fi = np.asarray(frames[i], dtype=np.complex128)
        Fj = np.asarray(frames[j], dtype=np.complex128)
        Fk = np.asarray(frames[k], dtype=np.complex128)
        Uij = Fj @ dagger(Fi)
        Ujk = Fk @ dagger(Fj)
        Uki = Fi @ dagger(Fk)
        W = Uij @ Ujk @ Uki
        tr2 = np.trace(W) / 2.0
        x = float(np.real(tr2))
        x = float(np.clip(x, -1.0, 1.0))
        theta = 2.0 * float(math.acos(x))
        tr_vals.append(x)
        ang_vals.append(theta)
    tr_vals = np.array(tr_vals, float)
    ang_vals = np.array(ang_vals, float)
    return {'mean_tr_over_2_real': float(np.mean(tr_vals)), 'std_tr_over_2_real': float(np.std(tr_vals)), 'mean_loop_angle': float(np.mean(ang_vals)), 'std_loop_angle': float(np.std(ang_vals))}

class AcceleratedRotatingSurfaceGTRQCSimulator(GTRQCSimulator):

    def __init__(self, *, surface: 'TriangulatedSurface', rotation_direction: str='ccw', time_dependent_noise: bool=True, noise_update_every: int=1, gauss_bonnet_tol: float=5e-06, rot_axis: Tuple[float, float, float]=(0.0, 0.0, 1.0), rot_omega0: float=0.0, rot_alpha: float=1.25, coriolis_scale: float=1.0, centrifugal_scale: float=0.25, euler_scale: float=0.25, drive_gain: float=6.0, spring_k: float=25.0, damping: float=2.0, grav_k: float=0.25, keep_scale: bool=True, max_radius: float=6.0, dynamic_exchange: bool=True, dynamic_exchange_power: float=2.0, dynamic_exchange_clip: Tuple[float, float]=(0.25, 4.0), geometry_update_every: int=1, frame_recompute_every: int=1, causal_rebuild_every: int=1, curvature_update_every: int=1, enable_time_dependent_curvature: bool=True, c_light: Optional[float]=None, causality_mode: str='warn', gauss_bonnet_check_every: int=1, gauss_bonnet_rel_tol: float=1e-09, gauss_bonnet_mode: str='warn', holonomy_check_every: int=1, remesh_every: int=0, remesh_delaunay_tol: float=1e-06, remesh_max_flips_per_call: int=64, remesh_p_swap_mapping: str='length_density', remesh_verbose: bool=False, frame_seed: int=0, **base_kwargs: Any):
        self.surface = copy_surface(surface, name_suffix='_accel_rot')
        self.rotation_direction = str(rotation_direction)
        self._rot_sign = _direction_to_sign(rotation_direction)
        self.rot_axis = np.asarray(rot_axis, dtype=float).reshape(3)
        if float(np.linalg.norm(self.rot_axis)) <= 1e-15:
            self.rot_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        self.rot_axis = self.rot_axis / max(float(np.linalg.norm(self.rot_axis)), 1e-15)
        self.rot_omega0 = float(rot_omega0)
        self.rot_alpha = float(rot_alpha)
        self.coriolis_scale = float(coriolis_scale)
        self.centrifugal_scale = float(centrifugal_scale)
        self.euler_scale = float(euler_scale)
        self.drive_gain = float(drive_gain)
        self.spring_k = float(spring_k)
        self.damping = float(damping)
        self.grav_k = float(grav_k)
        self.keep_scale = bool(keep_scale)
        self.max_radius = float(max_radius)
        self.dynamic_exchange = bool(dynamic_exchange)
        self.dynamic_exchange_power = float(dynamic_exchange_power)
        self.dynamic_exchange_clip = (float(dynamic_exchange_clip[0]), float(dynamic_exchange_clip[1]))
        self.geometry_update_every = max(int(geometry_update_every), 0)
        self.frame_recompute_every = max(int(frame_recompute_every), 0)
        self.causal_rebuild_every = max(int(causal_rebuild_every), 0)
        self.curvature_update_every = max(int(curvature_update_every), 0)
        self.enable_time_dependent_curvature = bool(enable_time_dependent_curvature)
        self.c_light = None if c_light is None else float(c_light)
        self.causality_mode = str(causality_mode).strip().lower()
        if self.causality_mode not in ('warn', 'raise', 'clip'):
            raise ValueError("causality_mode must be one of {'warn','raise','clip'}")
        self.gauss_bonnet_check_every = max(int(gauss_bonnet_check_every), 0)
        self.gauss_bonnet_rel_tol = float(gauss_bonnet_rel_tol)
        self.gauss_bonnet_mode = str(gauss_bonnet_mode).strip().lower()
        if self.gauss_bonnet_mode not in ('warn', 'raise', 'ignore'):
            raise ValueError("gauss_bonnet_mode must be one of {'warn','raise','ignore'}")
        self.holonomy_check_every = max(int(holonomy_check_every), 0)
        self.remesh_every = max(int(remesh_every), 0)
        self.remesh_delaunay_tol = float(remesh_delaunay_tol)
        self.remesh_max_flips_per_call = int(remesh_max_flips_per_call)
        self.remesh_p_swap_mapping = str(remesh_p_swap_mapping).strip().lower()
        if self.remesh_p_swap_mapping not in ('preserve', 'length_density', 'nearest_old'):
            raise ValueError("remesh_p_swap_mapping must be one of {'preserve','length_density','nearest_old'}")
        self.remesh_verbose = bool(remesh_verbose)
        self.causal_graph_override: Optional[GeometryGraph] = None
        self.frame_seed = int(frame_seed)
        self.geom_times_hist: List[float] = []
        self.geom_coords_hist: List[np.ndarray] = []
        self.geom_edge_lengths_hist: List[np.ndarray] = []
        self.geom_omega_hist: List[float] = []
        self.curvature_shift_hist: List[np.ndarray] = []
        self.gauss_bonnet_total_hist: List[float] = []
        self.gauss_bonnet_relerr_hist: List[float] = []
        self.wilson_tr_mean_hist: List[float] = []
        self.wilson_angle_mean_hist: List[float] = []
        self.causality_violations: List[Dict[str, Any]] = []
        self.remesh_events: List[Dict[str, Any]] = []
        self.gauss_bonnet_expected_total: float = 0.0
        self._geom_t = 0.0
        self._geom_step = 0
        self._pos = None
        self._vel = None
        self.geom_edges: List[Tuple[int, int]] = []
        self._L0: Optional[np.ndarray] = None
        self._L0_mean: float = 1.0
        self.exchange_J0: Optional[np.ndarray] = None
        self._H_Z_list: Optional[List[np.ndarray]] = None
        self._H_coup_dynamic: Optional[np.ndarray] = None
        super().__init__(**base_kwargs)
        self.time_dependent_noise = bool(time_dependent_noise)
        self.noise_update_every = max(int(noise_update_every), 0)
        self.gb_target = gauss_bonnet_target(self.surface) if self.surface.closed else float('nan')
        self.gauss_bonnet_tol = float(gauss_bonnet_tol)
        self.gb_total_hist: List[float] = []
        self.gb_err_hist: List[float] = []
        self.K_surface_hist: List[np.ndarray] = []
        if int(self.surface.points.shape[0]) != int(self.N):
            raise ValueError(f'RotatingSurface simulator requires surface vertex count == N. Got surface.V={self.surface.points.shape[0]} vs N={self.N}.')
        P = np.array(self.surface.points, dtype=float, copy=True)
        P = P - np.mean(P, axis=0, keepdims=True)
        self.surface.points[:] = P
        for i in range(self.N):
            self.nodes[i].coord = P[i].copy()
        frames0 = surface_vertex_frames_su2(self.surface, seed=int(self.frame_seed))
        for i in range(self.N):
            self.nodes[i].frame_su2 = frames0[i]
        gb0 = gauss_bonnet_stats(self.surface)
        self.gauss_bonnet_expected_total = float(gb0['expected_total'])
        A = np.asarray(self.graph.adjacency, dtype=float) > 0.0
        edges: List[Tuple[int, int]] = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if A[i, j]:
                    edges.append((i, j))
        self.geom_edges = edges
        if edges:
            L0 = np.array([float(np.linalg.norm(P[i] - P[j])) for i, j in edges], dtype=float)
            self._L0 = np.clip(L0, 1e-09, None)
            self._L0_mean = float(np.mean(self._L0))
        else:
            self._L0 = np.zeros((0,), dtype=float)
            self._L0_mean = 1.0
        self._pos = P.copy()
        self._vel = np.zeros_like(P)
        self.exchange_J0 = np.array(self.exchange_J, dtype=float, copy=True)
        self._H_Z_list = []
        for a in range(self.M):
            Hz = np.zeros((self.d, self.d), dtype=np.complex128)
            for i in range(self.N):
                Hz += 0.5 * self._omega_i(i, float(self.K_cfg[a, i])) * self.Z_ops[i]
            self._H_Z_list.append(hermitize(Hz))
        self._update_dynamic_exchange_and_coupling()
        self._refresh_causal_structure_if_needed()

    def apply_curvature_shift(self, delta: np.ndarray, *, rebuild_fixed_rates: bool=True) -> None:
        super().apply_curvature_shift(delta, rebuild_fixed_rates=rebuild_fixed_rates)
        if self._H_Z_list is not None:
            self._H_Z_list = []
            for a in range(self.M):
                Hz = np.zeros((self.d, self.d), dtype=np.complex128)
                for i in range(self.N):
                    Hz += 0.5 * self._omega_i(i, float(self.K_cfg[a, i])) * self.Z_ops[i]
                self._H_Z_list.append(hermitize(Hz))

    def _omega_vec(self, t: float) -> np.ndarray:
        w = self._rot_sign * (self.rot_omega0 + self.rot_alpha * float(t))
        return float(w) * self.rot_axis

    def _alpha_vec(self) -> np.ndarray:
        a = self._rot_sign * self.rot_alpha
        return float(a) * self.rot_axis

    def _current_edge_lengths(self) -> np.ndarray:
        if not self.geom_edges:
            return np.zeros((0,), dtype=float)
        assert self._pos is not None
        P = self._pos
        L = np.array([float(np.linalg.norm(P[i] - P[j])) for i, j in self.geom_edges], dtype=float)
        return np.clip(L, 1e-09, None)

    def _spring_forces(self) -> np.ndarray:
        assert self._pos is not None
        P = self._pos
        F = np.zeros_like(P)
        if not self.geom_edges or self._L0 is None:
            return F
        for idx, (i, j) in enumerate(self.geom_edges):
            rij = P[i] - P[j]
            dist = float(np.linalg.norm(rij))
            dist = max(dist, 1e-12)
            u = rij / dist
            ext = dist - float(self._L0[idx])
            f = -self.spring_k * ext
            Fi = f * u
            F[i] += Fi
            F[j] -= Fi
        return F

    def _update_dynamic_exchange_and_coupling(self) -> None:
        if not self.dynamic_exchange:
            self._H_coup_dynamic = None
            return
        if self.exchange_J0 is None:
            self._H_coup_dynamic = None
            return
        A = np.asarray(self.graph.adjacency, dtype=float) > 0.0
        L = self._current_edge_lengths()
        if self._L0 is None:
            self._H_coup_dynamic = None
            return
        if L.shape != self._L0.shape:
            self._H_coup_dynamic = None
            return
        p = float(self.dynamic_exchange_power)
        ratio = (self._L0 / L) ** p
        lo, hi = self.dynamic_exchange_clip
        ratio = np.clip(ratio, lo, hi)
        Jdyn = np.zeros((self.N, self.N), dtype=float)
        for idx, (i, j) in enumerate(self.geom_edges):
            Jij0 = float(self.exchange_J0[i, j])
            Jij = Jij0 * float(ratio[idx])
            Jdyn[i, j] = Jij
            Jdyn[j, i] = Jij
        Hc = np.zeros((self.d, self.d), dtype=np.complex128)
        for idx, (i, j) in enumerate(self.geom_edges):
            ii, jj = (int(i), int(j))
            Jij0 = float(self.exchange_J0[ii, jj])
            Jij = Jij0 * float(ratio[idx])
            if Jij == 0.0:
                continue
            Hc += 0.5 * Jij * (self.X_ops[ii] @ self.X_ops[jj] + self.Y_ops[ii] @ self.Y_ops[jj])
        self._H_coup_dynamic = hermitize(Hc)

    def _refresh_causal_structure_if_needed(self) -> None:
        if not bool(getattr(self, 'causal_gr_transport', False)):
            return
        coords = [n.coord if n.coord is not None else np.zeros(3, dtype=float) for n in self.nodes]
        frames = [n.frame_su2 for n in self.nodes]
        self.causal, self.causal_gr_data = build_causal_structure(self.graph, p_swap=self.causal_p_swap, seed=int(self.causal_seed), gr_transport=bool(self.causal_gr_transport), node_coords=coords, node_frames=frames, frame_seed=int(self.causal_gr_frame_seed))

    def _rebuild_after_remesh(self, *, new_graph: GeometryGraph) -> None:
        self.graph = new_graph
        self.exchange_J = float(self.exchange_J_strength) * np.asarray(self.graph.adjacency, dtype=float)
        self.balls = build_balls(self.graph, self.ball_radius_hops)
        self.KB_list = self._build_modulars(self.rho_ref_pure)
        self.Pi_full, self.M_full, self.monitor_obs_full = self._build_monitors_and_observables()
        I = np.eye(self.d, dtype=np.complex128)
        self.Pi_bar_full = [hermitize(I - self.Pi_full[v]) for v in range(self.N)]
        self.M_bar_full = [matrix_sqrt_psd(self.Pi_bar_full[v], eps=0.0) for v in range(self.N)]
        self.F_plus_full, self.F_minus_full = ([], [])
        self.sF_plus_full, self.sF_minus_full = ([], [])
        for v in range(self.N):
            evals, U = np.linalg.eigh(hermitize(self.monitor_obs_full[v]))
            Fp, sFp = op_sigmoid_from_eigh(U, evals, self.functional_kappa, +1)
            Fm, sFm = op_sigmoid_from_eigh(U, evals, self.functional_kappa, -1)
            self.F_plus_full.append(Fp)
            self.sF_plus_full.append(sFp)
            self.F_minus_full.append(Fm)
            self.sF_minus_full.append(sFm)
        self.G = build_response_G(self.graph, m=self.mass_m, h=self.mesh_h)
        J0 = np.zeros(self.N, dtype=float)
        F0 = np.array([slice_F(self._k(self.K_cfg[a]), J0, self.G, self.alpha_area) for a in range(self.M)], dtype=float)
        self.pi0 = gibbs_pi(F0, self.beta_geom)
        self.R0 = metropolis_rates(F0, self.A_cfg, self.beta_geom, self.gamma_geom_base)
        self.out0 = np.sum(self.R0, axis=0)
        if self.backreaction == 'embedded':
            self.J_source = self._build_source_J_grid(self.S, self.source_J_scale)
            piS = np.zeros((self.S, self.M), float)
            RS = np.zeros((self.S, self.M, self.M), float)
            outS = np.zeros((self.S, self.M), float)
            for s in range(self.S):
                Js = self.J_source[s]
                Fs = np.array([slice_F(self._k(self.K_cfg[a]), Js, self.G, self.alpha_area) for a in range(self.M)], dtype=float)
                piS[s] = gibbs_pi(Fs, self.beta_geom)
                RS[s] = metropolis_rates(Fs, self.A_cfg, self.beta_geom, self.gamma_geom_base)
                outS[s] = np.sum(RS[s], axis=0)
            self.pi_source_geom = piS
            self.R_source = RS
            self.out_source = outS
            self.sqrtE_source = self._build_source_povm(self.S)
        self.apply_curvature_shift(self.K_curv_shift, rebuild_fixed_rates=True)
        self._refresh_causal_structure_if_needed()

    def _geometry_integrate_one_dt(self, dt: float) -> None:
        dt = float(dt)
        if dt <= 0.0:
            return
        assert self._pos is not None and self._vel is not None
        P = self._pos
        V = self._vel
        P_prev = np.array(P, dtype=float, copy=True)
        omega = self._omega_vec(self._geom_t)
        alpha = self._alpha_vec()
        v_target = np.cross(omega[None, :], P)
        a_drive = self.drive_gain * (v_target - V)
        F_spring = self._spring_forces()
        a_spring = F_spring
        a_damp = -self.damping * V
        a_grav = -self.grav_k * P
        a_cor = self.coriolis_scale * (2.0 * np.cross(omega[None, :], V))
        a_cent = self.centrifugal_scale * np.cross(omega[None, :], np.cross(omega[None, :], P))
        a_eul = self.euler_scale * np.cross(alpha[None, :], P)
        A = a_drive + a_spring + a_damp + a_grav + a_cor + a_cent + a_eul
        V = V + A * dt
        P = P + V * dt
        P = P - np.mean(P, axis=0, keepdims=True)
        if self.keep_scale and self._L0 is not None and (self._L0.size > 0):
            L = self._current_edge_lengths()
            mL = float(np.mean(L)) if L.size else 1.0
            if mL > 1e-12:
                s = float(self._L0_mean) / mL
                s = float(np.clip(s, 0.85, 1.15))
                P *= s
                V *= s
        if self.max_radius > 0.0:
            rmax = float(np.max(np.linalg.norm(P, axis=1))) if P.size else 0.0
            if rmax > self.max_radius:
                s = float(self.max_radius) / max(rmax, 1e-12)
                P *= s
                V *= s
        if self.c_light is not None and float(self.c_light) > 0.0:
            disp = np.linalg.norm(P - P_prev, axis=1)
            vmax = float(np.max(disp)) / max(float(dt), 1e-15)
            if vmax > float(self.c_light):
                idx = int(np.argmax(disp))
                evt = {'t': float(self._geom_t), 'dt': float(dt), 'vmax': float(vmax), 'c_light': float(self.c_light), 'vertex': int(idx), 'disp': float(disp[idx])}
                if self.causality_mode == 'raise':
                    raise FloatingPointError(f'Strict causality violated: vmax={vmax:.6g} > c_light={float(self.c_light):.6g} (vertex={idx}, |Δx|={disp[idx]:.6g}, dt={dt:g}).')
                if self.causality_mode == 'clip':
                    scale = float(self.c_light) * float(dt) / max(float(disp[idx]), 1e-15)
                    scale = float(np.clip(scale, 0.0, 1.0))
                    P = P_prev + scale * (P - P_prev)
                    V = (P - P_prev) / max(float(dt), 1e-15)
                    evt['clipped_scale'] = float(scale)
                self.causality_violations.append(evt)
        self._pos = P
        self._vel = V
        self._geom_t += dt
        self._geom_step += 1
        self.surface.points[:] = P
        for i in range(self.N):
            self.nodes[i].coord = P[i].copy()
        self._Ksurf_last = self.surface.curvature_scalar().astype(float)
        self.K_surface_hist.append(self._Ksurf_last.copy())
        if np.isfinite(self.gb_target):
            gb_total = float(np.sum(self._Ksurf_last))
            gb_err = float(gb_total - self.gb_target)
            if abs(gb_err) > self.gauss_bonnet_tol:
                print(f'[Gauss–Bonnet WARNING] t={self._geom_t:.4f}  sum(deficit)={gb_total:.8f}, target={self.gb_target:.8f}, err={gb_err:+.3e}', flush=True)
        did_remesh = False
        if self.remesh_every > 0 and self._geom_step % self.remesh_every == 0:
            old_edge_set = edge_set_from_triangles(self.surface.triangles)
            res = apply_local_delaunay_flips_inplace(self.surface, tol=float(self.remesh_delaunay_tol), max_flips=int(self.remesh_max_flips_per_call))
            flips = res.get('flips', [])
            if flips:
                did_remesh = True
                new_edge_set = edge_set_from_triangles(self.surface.triangles)
                self.remesh_events.append({'t': float(self._geom_t), 'n_flips': int(res.get('n_flips', 0)), 'max_violation': float(res.get('max_violation', 0.0)), 'flips': flips, 'n_edges_old': int(len(old_edge_set)), 'n_edges_new': int(len(new_edge_set))})
                if self.remesh_verbose:
                    print(f"[REMESH] t={self._geom_t:.4f} flips={len(flips)} max_violation={res.get('max_violation', 0.0):.3e}", flush=True)
                new_graph = self.surface.to_graph()
                if isinstance(self.causal_p_swap, dict):
                    old_map = dict(self.causal_p_swap)
                    target_mean = float(np.mean(list(old_map.values()))) if old_map else 0.0
                    if self.remesh_p_swap_mapping == 'length_density':
                        new_map, _kappa = p_swap_map_edge_length_density(self.surface, dt=float(dt), target_mean_p=float(max(target_mean, 1e-06)))
                    else:
                        new_map = {e: float(target_mean) for e in new_edge_set}
                    for e in new_edge_set:
                        if e in old_map:
                            new_map[e] = float(old_map[e])
                    self.causal_p_swap = new_map
                old_L0_map: Dict[Tuple[int, int], float] = {}
                if self._L0 is not None:
                    for idx_e, e in enumerate(self.geom_edges):
                        ee = (int(e[0]), int(e[1]))
                        if ee[0] > ee[1]:
                            ee = (ee[1], ee[0])
                        old_L0_map[ee] = float(self._L0[idx_e])
                self.geom_edges = sorted(list(new_edge_set))
                if self.geom_edges:
                    L0_new = []
                    for i, j in self.geom_edges:
                        ii, jj = (int(i), int(j))
                        ee = (ii, jj) if ii < jj else (jj, ii)
                        if ee in old_L0_map:
                            L0_new.append(float(old_L0_map[ee]))
                        else:
                            L0_new.append(float(np.linalg.norm(P[ii] - P[jj])))
                    self._L0 = np.clip(np.array(L0_new, float), 1e-09, None)
                    self._L0_mean = float(np.mean(self._L0))
                else:
                    self._L0 = np.zeros((0,), dtype=float)
                    self._L0_mean = 1.0
                if self.exchange_J0 is not None:
                    J0_old = np.array(self.exchange_J0, dtype=float, copy=True)
                    J0_new = np.zeros_like(J0_old)
                    baseJ = float(getattr(self, 'exchange_J_strength', 0.0))
                    for i, j in self.geom_edges:
                        ii, jj = (int(i), int(j))
                        v0 = float(J0_old[ii, jj])
                        if v0 == 0.0:
                            v0 = baseJ
                        J0_new[ii, jj] = v0
                        J0_new[jj, ii] = v0
                    self.exchange_J0 = J0_new
        if did_remesh or (self.frame_recompute_every > 0 and self._geom_step % self.frame_recompute_every == 0):
            frames = surface_vertex_frames_su2(self.surface, seed=int(self.frame_seed))
            for i in range(self.N):
                self.nodes[i].frame_su2 = frames[i]
        self._update_dynamic_exchange_and_coupling()
        if did_remesh or (self.causal_rebuild_every > 0 and self._geom_step % self.causal_rebuild_every == 0):
            self._refresh_causal_structure_if_needed()

    def _apply_matter_unitary(self, dt: float) -> None:
        if not self.dynamic_exchange or self._H_Z_list is None or self._H_coup_dynamic is None:
            return super()._apply_matter_unitary(dt)
        dt = float(dt)
        Hc = self._H_coup_dynamic
        U_list = [expm_hermitian(self._H_Z_list[a] + Hc, dt, factor=-1j) for a in range(self.M)]
        if self.geometry_quantum:
            for s in range(self.S):
                for a in range(self.M):
                    Ua = U_list[a]
                    for b in range(self.M):
                        Ub = U_list[b]
                        self.rho[s, a, b] = Ua @ self.rho[s, a, b] @ dagger(Ub)
        else:
            for s in range(self.S):
                for a in range(self.M):
                    Ua = U_list[a]
                    self.rho[s, a] = Ua @ self.rho[s, a] @ dagger(Ua)

    def step_strang(self, dt: float) -> None:
        super().step_strang(dt)
        if self.geometry_update_every > 0:
            if self._geom_step % self.geometry_update_every == 0:
                self._geometry_integrate_one_dt(float(dt))
            else:
                self._geom_t += float(dt)
                self._geom_step += 1
        Ksurf = getattr(self, '_Ksurf_last', np.asarray(self.surface.curvature_scalar(), float))
        delta = np.asarray(Ksurf, float) - np.asarray(self.K_ref, float)
        update_every = self.curvature_update_every if self.curvature_update_every > 0 else self.noise_update_every
        if self.enable_time_dependent_curvature and self.time_dependent_noise and (update_every > 0) and (self._geom_step % update_every == 0):
            self.apply_curvature_shift(delta, rebuild_fixed_rates=True)
        gb_total = self.gauss_bonnet_total_hist[-1] if self.gauss_bonnet_total_hist else float('nan')
        gb_rel = self.gauss_bonnet_relerr_hist[-1] if self.gauss_bonnet_relerr_hist else float('nan')
        if self.gauss_bonnet_check_every > 0 and self._geom_step % self.gauss_bonnet_check_every == 0:
            gb = gauss_bonnet_stats(self.surface, expected_total=float(self.gauss_bonnet_expected_total))
            gb_total = float(gb['total_deficit'])
            gb_rel = float(gb['rel_error'])
            if self.gauss_bonnet_mode == 'raise' and gb_rel > float(self.gauss_bonnet_rel_tol):
                raise FloatingPointError(f'Gauss–Bonnet drift too large: rel_error={gb_rel:.3e} > tol={float(self.gauss_bonnet_rel_tol):.3e}')
            if self.gauss_bonnet_mode == 'warn' and gb_rel > float(self.gauss_bonnet_rel_tol):
                print(f'[GB] t={self._geom_t:.4f} rel_error={gb_rel:.3e} (expected {self.gauss_bonnet_expected_total:.6g})', flush=True)
        wtr = self.wilson_tr_mean_hist[-1] if self.wilson_tr_mean_hist else 0.0
        wang = self.wilson_angle_mean_hist[-1] if self.wilson_angle_mean_hist else 0.0
        if self.holonomy_check_every > 0 and self._geom_step % self.holonomy_check_every == 0:
            frames = [n.frame_su2 for n in self.nodes]
            holo = wilson_loop_triangle_stats(self.surface, frames)
            wtr = float(holo['mean_tr_over_2_real'])
            wang = float(holo['mean_loop_angle'])
        self.geom_times_hist.append(float(self._geom_t))
        assert self._pos is not None
        self.geom_coords_hist.append(np.array(self._pos, dtype=float, copy=True))
        self.geom_edge_lengths_hist.append(np.array(self._current_edge_lengths(), dtype=float, copy=True))
        self.geom_omega_hist.append(float(np.linalg.norm(self._omega_vec(self._geom_t))))
        self.curvature_shift_hist.append(delta.copy())
        self.gauss_bonnet_total_hist.append(float(gb_total))
        self.gauss_bonnet_relerr_hist.append(float(gb_rel))
        self.wilson_tr_mean_hist.append(float(wtr))
        self.wilson_angle_mean_hist.append(float(wang))

    def run(self, T_final: float, dt: float, store_every: int=1, label: str='', progress_every: int=25, store_matter_states: bool=False) -> Dict[str, Any]:
        self.geom_times_hist = [0.0]
        assert self._pos is not None
        self.geom_coords_hist = [np.array(self._pos, dtype=float, copy=True)]
        self.geom_edge_lengths_hist = [np.array(self._current_edge_lengths(), dtype=float, copy=True)]
        self.geom_omega_hist = [float(np.linalg.norm(self._omega_vec(0.0)))]
        self._geom_t = 0.0
        self._geom_step = 0
        delta0 = np.asarray(self.surface.curvature_scalar(), float) - np.asarray(self.K_ref, float)
        if self.enable_time_dependent_curvature:
            self.apply_curvature_shift(delta0, rebuild_fixed_rates=True)
        self.curvature_shift_hist = [delta0.copy()]
        gb0 = gauss_bonnet_stats(self.surface, expected_total=float(self.gauss_bonnet_expected_total))
        self.gauss_bonnet_total_hist = [float(gb0['total_deficit'])]
        self.gauss_bonnet_relerr_hist = [float(gb0['rel_error'])]
        holo0 = wilson_loop_triangle_stats(self.surface, [n.frame_su2 for n in self.nodes])
        self.wilson_tr_mean_hist = [float(holo0['mean_tr_over_2_real'])]
        self.wilson_angle_mean_hist = [float(holo0['mean_loop_angle'])]
        self.causality_violations = []
        self.remesh_events = []
        summ = super().run(T_final=T_final, dt=dt, store_every=store_every, label=label, progress_every=progress_every, store_matter_states=store_matter_states)
        summ = dict(summ)
        summ['geometry_motion_model'] = {'kind': 'accelerated_rotation_with_coriolis_like_and_springs', 'rotation_direction': str(self.rotation_direction), 'rot_axis': [float(x) for x in self.rot_axis.tolist()], 'rot_omega0': float(self.rot_omega0), 'rot_alpha': float(self.rot_alpha), 'coriolis_scale': float(self.coriolis_scale), 'centrifugal_scale': float(self.centrifugal_scale), 'euler_scale': float(self.euler_scale), 'drive_gain': float(self.drive_gain), 'spring_k': float(self.spring_k), 'damping': float(self.damping), 'grav_k': float(self.grav_k), 'dynamic_exchange': bool(self.dynamic_exchange), 'dynamic_exchange_power': float(self.dynamic_exchange_power), 'dynamic_exchange_clip': [float(self.dynamic_exchange_clip[0]), float(self.dynamic_exchange_clip[1])], 'geometry_update_every': int(self.geometry_update_every), 'frame_recompute_every': int(self.frame_recompute_every), 'causal_rebuild_every': int(self.causal_rebuild_every)}
        return summ

def canonical_energy_second_variation_for_simulator(sim_class: Any, base_params: Dict[str, Any], T_final: float, dt: float, store_every: int, lambda_val: float=0.002, seed: int=0, label: str='CE2') -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    sim0 = sim_class(**base_params)
    d = sim0.d
    H = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    H = hermitize(H)
    U = expm_hermitian(H, t=float(lambda_val), factor=-1j)
    Udag = dagger(U)
    rho0 = sim0.rho0
    rho_p = U @ rho0 @ Udag
    rho_m = Udag @ rho0 @ U
    sum0 = sim0.run(T_final=T_final, dt=dt, store_every=store_every, label=f'{label} baseline', progress_every=0, store_matter_states=True)
    p_params = dict(base_params)
    m_params = dict(base_params)
    p_params['initial_rho_override'] = rho_p
    m_params['initial_rho_override'] = rho_m
    simp = sim_class(**p_params)
    simm = sim_class(**m_params)
    _ = simp.run(T_final=T_final, dt=dt, store_every=store_every, label=f'{label} +λ', progress_every=0, store_matter_states=True)
    _ = simm.run(T_final=T_final, dt=dt, store_every=store_every, label=f'{label} -λ', progress_every=0, store_matter_states=True)
    if sim0.rho_m_hist is None or simp.rho_m_hist is None or simm.rho_m_hist is None:
        raise RuntimeError('Expected stored matter states for CE2 calculation.')
    times = np.array(sim0.times, float)
    ce2 = np.zeros_like(times)
    srel_p = np.zeros_like(times)
    srel_m = np.zeros_like(times)
    for k in range(times.size):
        r0 = sim0.rho_m_hist[k]
        rp = simp.rho_m_hist[k]
        rm = simm.rho_m_hist[k]
        srel_p[k] = relative_entropy(rp, r0, eps=1e-12)
        srel_m[k] = relative_entropy(rm, r0, eps=1e-12)
        ce2[k] = (srel_p[k] + srel_m[k]) / max(float(lambda_val * lambda_val), 1e-30)
    return {'ok': True, 'lambda': float(lambda_val), 'seed': int(seed), 'times': times, 'Srel_plus': srel_p, 'Srel_minus': srel_m, 'CE2_second_variation': ce2, 'CE2_final': float(ce2[-1]) if ce2.size else 0.0, 'baseline_summary': sum0, 'note': 'CE2 proxy for rotating-geometry simulator: d^2/dλ^2 Srel(ρ_out(λ)||ρ_out(0))|_{λ=0}.'}

def main_accelerated_rotation_models_only() -> None:
    OUT_PREFIX = os.environ.get('GTRQC_OUT_PREFIX', 'gTRQC_MANUSCRIPT_ACCEL_ROT_POC')
    DPI = int(os.environ.get('GTRQC_DPI', '200'))
    KFOLDS = int(os.environ.get('GTRQC_KFOLDS', '2'))
    GEOMS = [g.strip() for g in os.environ.get('GTRQC_GEOMS', 'tetra,octa,cube').split(',') if g.strip()]
    MODES = [m.strip() for m in os.environ.get('GTRQC_MODES', 'linear_fixed,linear_operator_functional,paper_instantaneous_j_gibbs,paper_instantaneous_j_gibbs_operator').split(',') if m.strip()]
    ROT_CASES = [c.strip() for c in os.environ.get('GTRQC_ROT_CASES', 'cw,ccw').split(',') if c.strip()]
    ROT_OMEGA0 = float(os.environ.get('GTRQC_ROT_OMEGA0', '0.0'))
    ROT_ALPHA = float(os.environ.get('GTRQC_ROT_ALPHA', '1.25'))
    ROT_AXIS_STR = os.environ.get('GTRQC_ROT_AXIS', '0,0,1')
    try:
        ROT_AXIS = tuple((float(x) for x in ROT_AXIS_STR.split(',')))
        if len(ROT_AXIS) != 3:
            ROT_AXIS = (0.0, 0.0, 1.0)
    except Exception:
        ROT_AXIS = (0.0, 0.0, 1.0)
    _LEVEL_TAGS = ('low', 'mid', 'high')
    DRIVE_GAIN_MID = float(os.environ.get('GTRQC_ROT_DRIVE_GAIN', '6.0'))
    SPRING_K_MID = float(os.environ.get('GTRQC_ROT_SPRING_K', '12.5'))
    GRAV_K_BASE = float(os.environ.get('GTRQC_ROT_GRAV_K', '0.25'))
    SPRING_LEVELS = [float(x) for x in os.environ.get('GTRQC_ROT_SPRING_K_LEVELS', f'{0.5 * SPRING_K_MID},{SPRING_K_MID},{2.0 * SPRING_K_MID}').split(',')]
    DRIVE_LEVELS = [float(x) for x in os.environ.get('GTRQC_ROT_DRIVE_GAIN_LEVELS', f'{0.5 * DRIVE_GAIN_MID},{DRIVE_GAIN_MID},{2.0 * DRIVE_GAIN_MID}').split(',')]
    GRAV_LEVELS = [float(x) for x in os.environ.get('GTRQC_ROT_GRAV_K_LEVELS', f'0.0,{GRAV_K_BASE},{2.0 * GRAV_K_BASE},{4.0 * GRAV_K_BASE}').split(',')]
    assert len(SPRING_LEVELS) == 3 and len(DRIVE_LEVELS) == 3 and (len(GRAV_LEVELS) >= 2), 'SPRING_K and DRIVE_GAIN sweeps must each have exactly 3 comma-separated values; GRAV_K_LEVELS must have >=2 values.'
    DAMPING = float(os.environ.get('GTRQC_ROT_DAMPING', '2.0'))
    COR_SCALE = float(os.environ.get('GTRQC_ROT_CORIOLIS_SCALE', '1.0'))
    CEN_SCALE = float(os.environ.get('GTRQC_ROT_CENTRIFUGAL_SCALE', '0.25'))
    EUL_SCALE = float(os.environ.get('GTRQC_ROT_EULER_SCALE', '0.25'))
    KEEP_SCALE = bool(int(os.environ.get('GTRQC_ROT_KEEP_SCALE', '1')))
    DYN_EXCH = bool(int(os.environ.get('GTRQC_ROT_DYNAMIC_EXCHANGE', '1')))
    DYN_POW = float(os.environ.get('GTRQC_ROT_DYNAMIC_EXCHANGE_POW', '2.0'))
    T_FINAL = float(os.environ.get('GTRQC_TFINAL', '2.0'))
    DT = float(os.environ.get('GTRQC_DT', '0.02'))
    STORE_EVERY = int(os.environ.get('GTRQC_STORE', '1'))
    HX = float(os.environ.get('GTRQC_OCTA_TRANSVERSE_HX', '0.25'))
    C_LIGHT_FACTOR = float(os.environ.get('GTRQC_C_LIGHT_FACTOR', '1.0'))
    ts0 = now_stamp()
    pdf_report = os.path.abspath(sanitize_filename(f'{OUT_PREFIX}_REPORT_{ts0}.pdf'))
    AGG_KEYS = ['residual_norm_final', 'dK_final_absmean', 'dK_final_absmax', 'KL_geom_final', 'TV_geom_final', 'Sgen_final_mean', 'Sgen_final_max', 'Srel_final', 'CE_BKM_final', 'CE_BKM_history_final', 'FI_SLD_final', 'FI_SLD_history_final', 'neg_final', 'trace_drift_abs']
    all_artifacts: List[Dict[str, str]] = []
    all_summaries: List[Dict[str, Any]] = []
    all_model_blocks: List[Dict[str, Any]] = []
    all_ce2: Optional[Dict[str, Any]] = None
    with PdfPages(pdf_report) as pdf:
        pdf_text_page(pdf, 'gTRQC — Accelerated-Rotation Geometry Models (CW/CCW) with Coriolis-like + Gravity-like Deformation', [f'Timestamp: {ts0}', '', 'This run executes ONLY the two new rotating-geometry cases:', '  • CW  (clockwise accelerated rotation)', '  • CCW (anticlockwise accelerated rotation)', '', 'Physics-inspired kinematics (phenomenological):', '  • ω(t) = ω0 + α t about a fixed axis', '  • spring constraints on graph edges', '  • Coriolis-like / centrifugal-like / Euler-like inertial terms', '  • gravity-like restoring field', '  • optional distance-dependent exchange Jij(t) ∝ (L0/L)^p', '', f'Geometries: {GEOMS}', f'Rate modes: {MODES}', f'Rotation cases: {ROT_CASES}', f'K folds: {KFOLDS}', f'T_final={T_FINAL}, dt={DT}, store_every={STORE_EVERY}', '', f'Rotation axis: {ROT_AXIS}', f'ω0={ROT_OMEGA0}, α={ROT_ALPHA}', f'drive_gain sweep levels (low/mid/high): {DRIVE_LEVELS}', f'spring_k sweep levels  (low/mid/high): {SPRING_LEVELS}', f'grav_k sweep levels: {GRAV_LEVELS}', f'damping={DAMPING}', f'coriolis_scale={COR_SCALE}, centrifugal_scale={CEN_SCALE}, euler_scale={EUL_SCALE}', f'dynamic_exchange={DYN_EXCH}, power={DYN_POW}'], dpi=DPI)
        pdf_text_page(pdf, 'Manuscript Traceability Matrix (Equation/Definition → Code)', [json.dumps(manuscript_traceability_matrix(), indent=2)], dpi=DPI)
        for geom_kind in GEOMS:
            surface0 = geometry_builder(geom_kind, seed=100)
            do_gr = True
            pachner_naive = pachner_channel_invariance_check(surface0, p_swap=0.35, causal_seed=123, n_tests=12, seed=999, gr_transport=do_gr)
            pachner_matched = pachner_channel_invariance_check_matched_density(surface0, p_swap_base=0.35, dt_density=DT, causal_seed=123, n_tests=12, seed=999, gr_transport=do_gr)
            pdf_text_page(pdf, f'Pachner 2–2 Channel Stability (static triangulation diagnostic) — geometry={geom_kind} (naive vs density-matched) + GR transport', ['NAIVE:', json.dumps(pachner_naive, indent=2), '', 'DENSITY-MATCHED:', json.dumps(pachner_matched, indent=2)], dpi=DPI)
            nodes0, g0, _ = surface_to_nodes_graph(surface0, name_prefix='v', compute_frames=True, frame_seed=0)
            coords0 = [n.coord for n in nodes0]
            frames0 = [n.frame_su2 for n in nodes0]
            cov = gauge_covariance_check_for_gr_channel(graph=g0, node_coords=coords0, node_frames=frames0, p_swap=0.35, causal_seed=123, frame_seed=111, n_tests=8, seed=2025)
            pdf_text_page(pdf, f'TRQC–GR Gauge Covariance Check (static) — geometry={geom_kind}', [json.dumps(cov, indent=2)], dpi=DPI)
            Ctmp, _ = build_causal_structure(g0, p_swap=0.35, seed=123, gr_transport=True, node_coords=coords0, node_frames=frames0, frame_seed=111)
            if Ctmp.steps and len(Ctmp.steps[0].edges) >= 2:
                eA = Ctmp.steps[0].edges[0]
                eB = Ctmp.steps[0].edges[1]
                region_A = [eA.i, eA.j]
                region_B = [eB.i, eB.j]
            else:
                eA = Ctmp.edge_list[0]
                region_A = [eA.i, eA.j]
                region_B = [q for q in range(Ctmp.N) if q not in region_A][:1]
            nosig = Ctmp.no_signaling_region_test(step_idx=0, region_A=region_A, region_B=region_B, n_trials=20, seed=2026)
            pdf_text_page(pdf, f'No-Signaling Region Test (static, step 0) — geometry={geom_kind}', [json.dumps(nosig, indent=2)], dpi=DPI)
            for mode in MODES:
                for rot_case in ROT_CASES:
                    for sk_idx, _sk in enumerate(SPRING_LEVELS):
                        SPRING_K = float(_sk)
                        sk_tag = _LEVEL_TAGS[sk_idx]
                        for dg_idx, _dg in enumerate(DRIVE_LEVELS):
                            DRIVE_GAIN = float(_dg)
                            dg_tag = _LEVEL_TAGS[dg_idx]
                            for gk_idx, _gk in enumerate(GRAV_LEVELS):
                                GRAV_K = float(_gk)
                                gk_tag = f'g{gk_idx}'
                                fold_summaries: List[Dict[str, Any]] = []
                                representative_figs: Optional[List[plt.Figure]] = None
                                for fold in range(KFOLDS):
                                    seed_surface = 1000 + 17 * fold + stable_hash_int('surface', geom_kind, rot_case, mod=997)
                                    seed_ensemble = 2000 + 29 * fold + stable_hash_int('ensemble', mode, rot_case, mod=991)
                                    surf = geometry_builder(geom_kind, seed=int(seed_surface))
                                    nodes, graph, _ = surface_to_nodes_graph(surf, name_prefix=f'{geom_kind[0]}', compute_frames=True, frame_seed=0)
                                    base_params = dict(nodes=nodes, graph=graph, ball_radius_hops=1, alpha_area=1.0, regge_a=0.8, mass_m=0.7, mesh_h=1.0, beta_geom=6.0, gamma_geom_base=12.0, omega_0=2.0 * math.pi * 40.0, alpha_curv_freq=0.2, gamma0=0.3, gamma_curv=1.2, exchange_J_strength=2.5, transverse_hx=HX, eps_modular=0.002, eps_ref_rank=1e-06, reference_state='zeros', initial_state='plus_product', negativity_pair=(0, len(nodes) - 1), geometry_initial='coherent_gibbs' if per_geometry_default_params(geom_kind).get('geometry_quantum', False) else 'diagonal_gibbs', geom_mix_rate=1.2 if per_geometry_default_params(geom_kind).get('geometry_quantum', False) else 0.0, geom_pointer_dephase=0.15, monitor_eta=0.9, causal_p_swap=0.35, causal_seed=123 + fold, causal_apply_every=1, causal_gr_transport=True, causal_gr_frame_seed=111, functional_kappa=0.4, source_dim=7, source_J_scale=0.5, source_measure_strength=1.0, source_observable_mix=1.0, source_init='delta_center', enforce_zero_mean_if_massless=True, enforce_gauss_bonnet=True, seed_geometry_ensemble=int(seed_ensemble), trace_mode='hygiene', trace_tol=5e-11, trace_hygiene_tol=1e-06, memory_guard_gib=2.0)
                                    base_params.update(per_geometry_default_params(geom_kind))
                                    base_params.update(per_mode_params(mode))
                                    if str(base_params['rate_mode']).startswith('embedded_') and base_params.get('geometry_quantum', False) and (len(nodes) >= 6):
                                        base_params['geometry_quantum'] = False
                                        base_params['geometry_initial'] = 'diagonal_gibbs'
                                        base_params['geom_mix_rate'] = 0.0
                                    label = f"{geom_kind.upper()} | {base_params['rate_mode']} | ROT={rot_case.upper()} | SK({sk_tag})={SPRING_K:g} DG({dg_tag})={DRIVE_GAIN:g} GK({gk_tag})={GRAV_K:g} | fold={fold + 1}/{KFOLDS}"
                                    print(f'\n=== {label} ===', flush=True)
                                    c_light = float(C_LIGHT_FACTOR) * float(surf.mean_edge_length()) / max(float(DT), 1e-15)
                                    sim = AcceleratedRotatingSurfaceGTRQCSimulator(surface=surf, rotation_direction=rot_case, rot_axis=ROT_AXIS, rot_omega0=ROT_OMEGA0, rot_alpha=ROT_ALPHA, coriolis_scale=COR_SCALE, centrifugal_scale=CEN_SCALE, euler_scale=EUL_SCALE, drive_gain=DRIVE_GAIN, spring_k=SPRING_K, damping=DAMPING, grav_k=GRAV_K, keep_scale=KEEP_SCALE, dynamic_exchange=DYN_EXCH, dynamic_exchange_power=DYN_POW, geometry_update_every=1, frame_recompute_every=1, causal_rebuild_every=1, frame_seed=0, c_light=c_light, causality_mode='warn', **base_params)
                                    fl = first_law_check_on_balls(rho_ref=sim.rho_ref_pure, balls=sim.balls, KB_list=sim.KB_list, N=sim.N, eps_list=(0.0003, 0.001, 0.003, 0.01), seed=10 + fold)
                                    summary = sim.run(T_final=T_FINAL, dt=DT, store_every=STORE_EVERY, label=label, progress_every=max(10, int(round(T_FINAL / DT / 8))), store_matter_states=False)
                                    summary['first_law_check'] = fl['chosen']
                                    arrays = dict(times=np.array(sim.times, float), K_eff=np.stack(sim.K_eff_hist, axis=0), K_eq=np.stack(sim.K_eq_hist, axis=0), residual_norm=np.array(sim.res_hist, float), negativity=np.array(sim.neg_hist, float), rel_entropy=np.array(sim.Srel_hist, float), trace_distance=np.array(sim.Tdist_hist, float), trace_distance_to_mix=np.array(sim.Tdist_mix_hist, float), trace_distance_step=np.array(sim.Tdist_step_hist, float), purity=np.array(sim.purity_hist, float), matter_entropy=np.array(sim.matter_entropy_hist, float), ent_entropyA=np.array(sim.ent_entropyA_hist, float), ent_entropyB=np.array(sim.ent_entropyB_hist, float), mutual_info=np.array(sim.mutual_info_hist, float), log_negativity_AB=np.array(sim.log_negativity_AB_hist, float), matter_coherence_l1=np.array(sim.coh_matter_hist, float), matter_min_eig=np.array(sim.matter_min_eig_hist, float), CE_BKM=np.array(sim.CE_hist, float), FI_SLD=np.array(sim.FI_sld_hist, float), FI_classical=np.array(sim.FI_cls_hist, float), CE_BKM_history=np.array(sim.CE_path_hist, float), FI_SLD_history=np.array(sim.FI_path_hist, float), geom_p=np.stack(sim.p_hist, axis=0), geom_piJ=np.stack(sim.piJ_hist, axis=0), geom_KL=np.array(sim.KL_hist, float), geom_TV=np.array(sim.TV_hist, float), geom_coherence=np.array(sim.coh_hist, float), J=np.stack(sim.J_hist, axis=0), J_est=np.stack(sim.J_est_hist, axis=0), pS=np.stack(sim.pS_hist, axis=0), Sgen=np.stack(sim.Sgen_hist, axis=0), trace=np.array(sim.trace_hist, float), K_ref=np.array(sim.K_ref, float), K_configs=np.array(sim.K_cfg, float), A_cfg=np.array(sim.A_cfg, float), codes=np.array(sim.codes, np.int64), geom_times_all=np.array(sim.geom_times_hist, float), geom_coords_all=np.stack(sim.geom_coords_hist, axis=0), geom_edge_lengths_all=np.stack(sim.geom_edge_lengths_hist, axis=0), geom_edges=np.array(sim.geom_edges, np.int64), geom_omega=np.array(sim.geom_omega_hist, float), curvature_shift_all=np.stack(sim.curvature_shift_hist, axis=0), gauss_bonnet_total=np.array(sim.gauss_bonnet_total_hist, float), gauss_bonnet_relerr=np.array(sim.gauss_bonnet_relerr_hist, float), wilson_tr_mean=np.array(sim.wilson_tr_mean_hist, float), wilson_angle_mean=np.array(sim.wilson_angle_mean_hist, float))
                                    meta = {'timestamp': now_stamp(), 'geometry_kind': geom_kind, 'mode': str(base_params['rate_mode']), 'rotation_case': str(rot_case), 'fold': fold, 'label': label, 'summary': summary, 'pachner_naive_check': pachner_naive, 'pachner_density_matched_check': pachner_matched, 'params': {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in base_params.items() if k not in ('nodes', 'graph')}, 'rotation_params': {'rot_axis': [float(x) for x in ROT_AXIS], 'rot_omega0': float(ROT_OMEGA0), 'rot_alpha': float(ROT_ALPHA), 'drive_gain': float(DRIVE_GAIN), 'spring_k': float(SPRING_K), 'damping': float(DAMPING), 'grav_k': float(GRAV_K), 'coriolis_scale': float(COR_SCALE), 'centrifugal_scale': float(CEN_SCALE), 'euler_scale': float(EUL_SCALE), 'keep_scale': bool(KEEP_SCALE), 'dynamic_exchange': bool(DYN_EXCH), 'dynamic_exchange_power': float(DYN_POW), 'sweep_tags': {'spring_k': sk_tag, 'drive_gain': dg_tag, 'grav_k': gk_tag}, 'sweep_levels': {'spring_k': SPRING_LEVELS, 'drive_gain': DRIVE_LEVELS, 'grav_k': GRAV_LEVELS}}, 'graph_meta': graph.meta, 'causality_violations': getattr(sim, 'causality_violations', []), 'remesh_events': getattr(sim, 'remesh_events', [])}
                                    npz_path, json_path = save_npz_json(OUT_PREFIX, sanitize_filename(label)[:140], arrays, meta)
                                    all_artifacts.append({'label': label, 'npz': npz_path, 'json': json_path})
                                    all_summaries.append(summary)
                                    fold_summaries.append(summary)
                                    print(f'[ARTIFACT SAVED] {label}', flush=True)
                                    print(f'    NPZ : {npz_path}', flush=True)
                                    print(f'    JSON: {json_path}', flush=True)
                                    if representative_figs is None:
                                        representative_figs = plot_sim_timeseries(sim, title=label)
                                        if all_ce2 is None:
                                            ce2 = canonical_energy_second_variation_for_simulator(sim_class=AcceleratedRotatingSurfaceGTRQCSimulator, base_params=dict(surface=surf, rotation_direction=rot_case, rot_axis=ROT_AXIS, rot_omega0=ROT_OMEGA0, rot_alpha=ROT_ALPHA, coriolis_scale=COR_SCALE, centrifugal_scale=CEN_SCALE, euler_scale=EUL_SCALE, drive_gain=float(DRIVE_GAIN), spring_k=float(SPRING_K), damping=DAMPING, grav_k=float(GRAV_K), keep_scale=KEEP_SCALE, dynamic_exchange=DYN_EXCH, dynamic_exchange_power=DYN_POW, geometry_update_every=1, frame_recompute_every=1, causal_rebuild_every=1, frame_seed=0, **base_params), T_final=min(T_FINAL, 0.9), dt=DT, store_every=STORE_EVERY, lambda_val=0.002, seed=31415, label=f"CE2 | {geom_kind} | {base_params['rate_mode']} | ROT={rot_case} | SK={SPRING_K:g} DG={DRIVE_GAIN:g} GK={GRAV_K:g}")
                                            all_ce2 = ce2
                                agg = summarize_folds(fold_summaries, AGG_KEYS)
                                block = {'geometry_kind': geom_kind, 'mode': str(per_mode_params(mode)['rate_mode']), 'rotation_case': str(rot_case), 'k_folds': KFOLDS, 'sweep_point': {'spring_k': float(SPRING_K), 'drive_gain': float(DRIVE_GAIN), 'grav_k': float(GRAV_K), 'tags': {'spring_k': sk_tag, 'drive_gain': dg_tag, 'grav_k': gk_tag}}, 'aggregate_metrics': agg, 'fold_summaries': fold_summaries}
                                all_model_blocks.append(block)
                                pdf_text_page(pdf, f"K-fold Summary — geometry={geom_kind}  mode={block['mode']}  ROT={rot_case.upper()}  SK({sk_tag}) DG({dg_tag}) GK({gk_tag})", [json.dumps(block, indent=2)], dpi=DPI)
                                if representative_figs is not None:
                                    for fig in representative_figs:
                                        pdf.savefig(fig, dpi=DPI)
                                        plt.close(fig)
                    agg = summarize_folds(fold_summaries, AGG_KEYS)
                    block = {'geometry_kind': geom_kind, 'mode': str(per_mode_params(mode)['rate_mode']), 'rotation_case': str(rot_case), 'k_folds': KFOLDS, 'aggregate_metrics': agg, 'fold_summaries': fold_summaries}
                    pdf_text_page(pdf, f"K-fold Summary — geometry={geom_kind}  mode={block['mode']}  ROT={rot_case.upper()}", [json.dumps(block, indent=2)], dpi=DPI)
                    if representative_figs is not None:
                        for fig in representative_figs:
                            pdf.savefig(fig, dpi=DPI)
                            plt.close(fig)
        SEL_KL_MAX = float(os.environ.get('GTRQC_SEL_KL_MAX', 'inf'))
        SEL_TV_MAX = float(os.environ.get('GTRQC_SEL_TV_MAX', 'inf'))
        SEL_SGEN_MAX = float(os.environ.get('GTRQC_SEL_SGEN_MAX', 'inf'))
        constraints: Dict[str, float] = {}
        if np.isfinite(SEL_KL_MAX):
            constraints['KL_geom_final'] = float(SEL_KL_MAX)
        if np.isfinite(SEL_TV_MAX):
            constraints['TV_geom_final'] = float(SEL_TV_MAX)
        if np.isfinite(SEL_SGEN_MAX):
            constraints['Sgen_final_mean'] = float(SEL_SGEN_MAX)
        objectives = ('residual_norm_final', 'dK_final_absmean', 'dK_final_absmax')
        pareto_blocks = select_pareto_blocks(all_model_blocks, objectives=objectives, constraints=constraints)
        pareto_report = {'objectives_minimize': list(objectives), 'constraints_max': constraints, 'n_candidates': int(len(all_model_blocks)), 'n_feasible': int(len([b for b in all_model_blocks if block_meets_constraints(b, constraints)])), 'n_pareto': int(len(pareto_blocks)), 'pareto_models': [{'geometry_kind': b.get('geometry_kind'), 'mode': b.get('mode'), 'rotation_case': b.get('rotation_case'), 'sweep_point': b.get('sweep_point'), 'aggregate_means': {k: _metric_mean(b.get('aggregate_metrics', {}), k, default=float('inf')) for k in list(objectives) + list(constraints.keys())}} for b in pareto_blocks[:min(len(pareto_blocks), 40)]], 'note': 'Pareto set computed on fold-aggregated means; constraints applied before Pareto filtering.'}
        pdf_text_page(pdf, 'Pareto-Optimal Selection (Residual vs dK, constrained KL/TV/Sgen)', [json.dumps(pareto_report, indent=2)], dpi=DPI)
        lines = ['Artifacts written (NPZ + JSON per fold):', '']
        for a in all_artifacts[:min(len(all_artifacts), 80)]:
            lines += [f"- {a['label']}", f"    NPZ : {a['npz']}", f"    JSON: {a['json']}", '']
        if len(all_artifacts) > 80:
            lines += [f'... ({len(all_artifacts) - 80} more artifacts omitted for brevity)']
        lines += ['', f'Consolidated PDF report: {pdf_report}']
        pdf_text_page(pdf, 'Artifact Index', lines, dpi=DPI)
    print('\n=== gTRQC Accelerated-Rotation Models COMPLETE ===')
    print(f'PDF report: {pdf_report}')
    print(f'Total fold artifacts: {len(all_artifacts)}')
    print('Artifacts written (NPZ + JSON per fold):')
    for a in all_artifacts:
        print(f"  [{a['label']}]")
        print(f"    NPZ : {a['npz']}")
        print(f"    JSON: {a['json']}")
from dataclasses import dataclass
from typing import Callable

class RelativisticDelayBuffer:

    def __init__(self, maxlen: int, shape: Tuple[int, ...], init: float=0.0):
        self.maxlen = int(maxlen)
        self.shape = tuple((int(s) for s in shape))
        self.init = float(init)
        self._buf: List[np.ndarray] = []
        self.reset()

    def reset(self) -> None:
        self._buf = [np.full(self.shape, self.init, dtype=float)]

    def push(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float).reshape(self.shape)
        self._buf.append(x.copy())
        if len(self._buf) > self.maxlen:
            self._buf = self._buf[-self.maxlen:]

    def _exp_weights(self, L: int, dt: float, tau: float) -> np.ndarray:
        L = int(L)
        dt = float(dt)
        tau = float(max(tau, 1e-12))
        k = np.arange(L, dtype=float)
        w = np.exp(-k * dt / tau)
        w = w / max(float(np.sum(w)), 1e-15)
        return w

    def effective(self, *, delay_steps: int=0, kernel: str='delta', dt: float=1.0, tau: Optional[float]=None) -> np.ndarray:
        delay_steps = max(int(delay_steps), 0)
        kernel = str(kernel).strip().lower()
        if kernel == 'delta':
            idx = max(0, len(self._buf) - 1 - delay_steps)
            return np.asarray(self._buf[idx], dtype=float).reshape(self.shape).copy()
        if kernel == 'exp':
            if tau is None:
                raise ValueError('RelativisticDelayBuffer: exp kernel requires tau.')
            L = len(self._buf)
            w = self._exp_weights(L, dt=float(dt), tau=float(tau))
            out = np.zeros(self.shape, dtype=float)
            for k in range(L):
                out += float(w[k]) * np.asarray(self._buf[-1 - k], dtype=float).reshape(self.shape)
            return out
        raise ValueError("RelativisticDelayBuffer: kernel must be one of {'delta','exp'}.")

@dataclass
class NonlinearGKSLRun:
    times: np.ndarray
    rho_hist: List[np.ndarray]
    K_hist: List[np.ndarray]
    J_hist: List[np.ndarray]
    min_eig_hist: List[float]

class SemiclassicalNonlinearGKSLSimulator:

    def __init__(self, *, nodes: List[Node], graph: GeometryGraph, surface: Optional[TriangulatedSurface]=None, ball_radius_hops: int=1, alpha_area: float=1.0, mass_m: float=0.7, mesh_h: float=1.0, enforce_zero_mean_if_massless: bool=True, omega_0: float=2.0 * math.pi * 40.0, alpha_curv_freq: float=0.2, gamma0: float=0.3, gamma_curv: float=1.2, exchange_J_strength: float=2.5, transverse_hx: float=0.0, eps_modular: float=0.002, eps_ref_rank: float=1e-06, reference_state: str='zeros', initial_state: str='ghz', initial_rho_override: Optional[np.ndarray]=None, edge_curv_strength: float=0.15, edge_curv_clip: Tuple[float, float]=(0.25, 4.0), collective_jump_base: float=0.0, collective_jump_kappa: float=2.0, collective_jump_safety: float=0.9, delay_steps: int=0, memory_kernel: str='delta', kernel_tau: Optional[float]=None, feedback_mode: str='modular', energy_feedback_scale: float=1.0):
        self.nodes = nodes
        self.graph = graph
        self.surface = surface
        self.transverse_hx = float(transverse_hx)
        self.N = int(len(nodes))
        self.d = 2 ** self.N
        self.K_ref = np.array([n.curvature_ref for n in nodes], dtype=float)
        self.alpha_area = float(alpha_area)
        self.mass_m = float(mass_m)
        self.mesh_h = float(mesh_h)
        self.enforce_zero_mean_if_massless = bool(enforce_zero_mean_if_massless)
        self.volume_constraint_active = bool(self.enforce_zero_mean_if_massless and is_massless(self.mass_m))
        self.G = build_response_G(self.graph, m=self.mass_m, h=self.mesh_h)
        self.omega_0 = float(omega_0)
        self.alpha_curv_freq = float(alpha_curv_freq)
        self.gamma0 = float(gamma0)
        self.gamma_curv = float(gamma_curv)
        self.exchange_J = float(exchange_J_strength) * np.asarray(self.graph.adjacency, dtype=float)
        self.transverse_hx = float(transverse_hx)
        self.edge_curv_strength = float(edge_curv_strength)
        self.edge_curv_clip = (float(edge_curv_clip[0]), float(edge_curv_clip[1]))
        self.transverse_hx = float(transverse_hx)
        self.collective_jump_base = float(collective_jump_base)
        self.collective_jump_kappa = float(collective_jump_kappa)
        self.collective_jump_safety = float(collective_jump_safety)
        self.delay_steps = max(int(delay_steps), 0)
        self.memory_kernel = str(memory_kernel).strip().lower()
        self.kernel_tau = kernel_tau
        self.feedback_mode = str(feedback_mode).strip().lower()
        if self.feedback_mode not in ('modular', 'energy'):
            raise ValueError("feedback_mode must be one of {'modular','energy'}.")
        self.energy_feedback_scale = float(energy_feedback_scale)
        self.balls = build_balls(self.graph, int(ball_radius_hops))
        self.eps_modular = float(eps_modular)
        self.eps_ref_rank = float(eps_ref_rank)
        self.X_ops, self.Y_ops, self.Z_ops = build_local_paulis(self.N)
        self.rho_ref_pure = self._make_state(reference_state)
        self.rho_ref = (1.0 - self.eps_ref_rank) * self.rho_ref_pure + self.eps_ref_rank * (np.eye(self.d, dtype=np.complex128) / self.d)
        if initial_rho_override is not None:
            rho0 = _ensure_density(np.asarray(initial_rho_override, dtype=np.complex128))
            if rho0.shape != (self.d, self.d):
                raise ValueError('initial_rho_override has wrong shape.')
            self.rho = rho0
        else:
            self.rho = self._make_state(initial_state)
        self.KB_list = self._build_modulars(self.rho_ref_pure)
        coords = np.array([n.coord if n.coord is not None else np.zeros(3, dtype=float) for n in nodes], dtype=float)
        A = np.asarray(self.graph.adjacency, dtype=float) > 0.0
        self._edges: List[Tuple[int, int]] = [(i, j) for i in range(self.N) for j in range(i + 1, self.N) if A[i, j]]
        self._edge_len = np.array([float(np.linalg.norm(coords[i] - coords[j])) for i, j in self._edges], dtype=float)
        self._edge_len = np.clip(self._edge_len, 1e-12, None)
        if self.surface is not None:
            Av = vertex_areas_barycentric(self.surface)
            Av = np.clip(np.asarray(Av, dtype=float), 1e-12, None)
        else:
            Av = np.ones(self.N, dtype=float)
        self._A_vertex = Av
        self._Kdens_ref = self.K_ref / self._A_vertex
        self._sigma_minus = [0.5 * (self.X_ops[i] + 1j * self.Y_ops[i]) for i in range(self.N)]
        self._regions = self._default_regions_for_octa(coords)
        self._H_energy_ref = self._H_from_K_edge_integrated(self.K_ref)
        self._E_ref = float(np.trace(self.rho_ref_pure @ self._H_energy_ref).real)
        self._u_energy = self._energy_source_direction()
        self._J_buffer = RelativisticDelayBuffer(maxlen=max(8, self.delay_steps + 4), shape=(self.N,), init=0.0)

    def _make_state(self, which: str) -> np.ndarray:
        which = str(which).lower().strip()
        ket0 = np.array([1.0, 0.0], dtype=np.complex128)
        ket1 = np.array([0.0, 1.0], dtype=np.complex128)
        ketp = (ket0 + ket1) / math.sqrt(2.0)
        if which == 'zeros':
            psi = tensor_kets([ket0] * self.N)
        elif which == 'plus_product':
            psi = tensor_kets([ketp] * self.N)
        elif which == 'ghz':
            psi0 = tensor_kets([ket0] * self.N)
            psi1 = tensor_kets([ket1] * self.N)
            psi = (psi0 + psi1) / math.sqrt(2.0)
        else:
            raise ValueError(f"Unknown state '{which}' (valid: zeros, plus_product, ghz)")
        return np.outer(psi, psi.conj())

    def _build_modulars(self, rho_ref: np.ndarray) -> List[np.ndarray]:
        KBs = []
        for B in self.balls:
            rhoB = partial_trace_qubits(rho_ref, keep=B, N=self.N)
            dB = rhoB.shape[0]
            rho_reg = (1.0 - self.eps_modular) * rhoB + self.eps_modular * (np.eye(dB, dtype=np.complex128) / dB)
            KBs.append(-matrix_log_psd(rho_reg, eps=1e-12))
        return KBs

    def _default_regions_for_octa(self, coords: np.ndarray) -> List[List[int]]:
        regions: List[List[int]] = [list(B) for B in self.balls]
        used = set()
        pairs = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dij = float(np.linalg.norm(coords[i] - coords[j]))
                pairs.append((dij, i, j))
        pairs.sort(reverse=True, key=lambda x: x[0])
        for _d, i, j in pairs:
            if i in used or j in used:
                continue
            used.add(i)
            used.add(j)
            regions.append([int(i), int(j)])
            if len(used) >= min(self.N, 6):
                break
        return regions

    def _energy_source_direction(self) -> np.ndarray:
        u = np.ones(self.N, dtype=float)
        if self.volume_constraint_active:
            u = project_zero_mean(u)
        n = float(np.linalg.norm(u))
        return u / max(n, 1e-15)

    def modular_sources_J(self, rho_m: np.ndarray) -> np.ndarray:
        J = np.zeros(self.N, dtype=float)
        for v, B in enumerate(self.balls):
            rhoB = partial_trace_qubits(rho_m, keep=B, N=self.N)
            rhoBref = partial_trace_qubits(self.rho_ref_pure, keep=B, N=self.N)
            J[v] = float(np.trace(rhoB @ self.KB_list[v]).real - np.trace(rhoBref @ self.KB_list[v]).real)
        return J

    def energy_feedback_J(self, rho_m: np.ndarray) -> np.ndarray:
        E = float(np.trace(rho_m @ self._H_energy_ref).real)
        return float(self.energy_feedback_scale) * float(E - self._E_ref) * self._u_energy

    def solve_G(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, float)
        if self.volume_constraint_active:
            b = project_zero_mean(b)
            G_reg = self.G + 1e-12 * np.eye(self.N)
            x = np.linalg.solve(G_reg, b)
            return project_zero_mean(x)
        return np.linalg.solve(self.G, b)

    def _omega_i(self, i: int, K_i: float) -> float:
        return self.omega_0 * (1.0 + self.alpha_curv_freq * (K_i - float(self.K_ref[i])))

    def _gamma_i(self, i: int, K_i: float) -> float:
        return self.gamma0 + self.gamma_curv * abs(K_i - float(self.K_ref[i]))

    def _H_from_K_edge_integrated(self, Kvec: np.ndarray) -> np.ndarray:
        Kvec = np.asarray(Kvec, dtype=float).reshape(self.N)
        Kdens = Kvec / self._A_vertex
        dKdens = Kdens - self._Kdens_ref
        H = np.zeros((self.d, self.d), dtype=np.complex128)
        for i in range(self.N):
            H += 0.5 * self._omega_i(i, float(Kvec[i])) * self.Z_ops[i]
        hx = float(getattr(self, 'transverse_hx', 0.0))
        if hx != 0.0:
            for i in range(self.N):
                H += 0.5 * hx * self.X_ops[i]
        lo, hi = self.edge_curv_clip
        for idx, (i, j) in enumerate(self._edges):
            Jij0 = float(self.exchange_J[i, j])
            if Jij0 == 0.0:
                continue
            integral = 0.5 * float(dKdens[i] + dKdens[j]) * float(self._edge_len[idx])
            factor = 1.0 + float(self.edge_curv_strength) * integral
            factor = float(np.clip(factor, lo, hi))
            Jij = Jij0 * factor
            H += 0.5 * Jij * (self.X_ops[i] @ self.X_ops[j] + self.Y_ops[i] @ self.Y_ops[j])
        return hermitize(H)

    def _collective_jump_ops(self, Kvec: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        if self.collective_jump_base <= 0.0:
            return []
        Kvec = np.asarray(Kvec, dtype=float).reshape(self.N)
        ops: List[Tuple[float, np.ndarray]] = []
        for reg in self._regions:
            reg = [int(x) for x in reg]
            dK = float(np.sum(Kvec[reg] - self.K_ref[reg])) / max(float(len(reg)), 1.0)
            strength = 1.0 - math.exp(-float(self.collective_jump_kappa) * (dK * dK))
            gamma = float(self.collective_jump_base) * float(np.clip(strength, 0.0, 1.0))
            if gamma <= 0.0:
                continue
            L = np.zeros((self.d, self.d), dtype=np.complex128)
            for i in reg:
                L += self._sigma_minus[int(i)]
            L = L / math.sqrt(max(len(reg), 1))
            ops.append((gamma, L))
        return ops

    def _apply_collective_jump_channel(self, rho: np.ndarray, dt: float, Kvec: np.ndarray) -> np.ndarray:
        ops = self._collective_jump_ops(Kvec)
        if not ops:
            return rho
        I = np.eye(self.d, dtype=np.complex128)
        S_op = np.zeros((self.d, self.d), dtype=np.complex128)
        for gamma, L in ops:
            S_op += float(gamma) * hermitize(dagger(L) @ L)
        S_op = hermitize(S_op)
        lam = float(np.max(np.linalg.eigvalsh(S_op).real))
        if lam <= 0.0:
            return rho
        dt_cap = float(self.collective_jump_safety) / max(lam, 1e-15)
        n_sub = int(math.ceil(float(dt) / dt_cap))
        dt_sub = float(dt) / max(n_sub, 1)
        out = rho
        for _ in range(n_sub):
            rho_in = out
            Q = hermitize(I - dt_sub * S_op)
            K0 = matrix_sqrt_psd(Q, eps=0.0)
            rho_out = K0 @ rho_in @ dagger(K0)
            for gamma, L in ops:
                Jop = math.sqrt(dt_sub * float(gamma)) * L
                rho_out += Jop @ rho_in @ dagger(Jop)
            out = hermitize(rho_out) / max(float(np.trace(rho_out).real), 1e-15)
        return out

    def _feedback_source_J(self, rho_m: np.ndarray) -> np.ndarray:
        if self.feedback_mode == 'energy':
            return self.energy_feedback_J(rho_m)
        return self.modular_sources_J(rho_m)

    def step(self, dt: float, *, dt_for_kernel: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = float(dt)
        dtK = float(dt if dt_for_kernel is None else dt_for_kernel)
        rho_m = _ensure_density(self.rho)
        J_now = self._feedback_source_J(rho_m)
        self._J_buffer.push(J_now)
        J_eff = self._J_buffer.effective(delay_steps=int(self.delay_steps), kernel=str(self.memory_kernel), dt=float(dtK), tau=self.kernel_tau)
        k_eq = self.solve_G(-1.0 / max(self.alpha_area, 1e-15) * J_eff)
        if self.volume_constraint_active:
            k_eq = project_zero_mean(k_eq)
        K = self.K_ref + k_eq
        H = self._H_from_K_edge_integrated(K)
        U = expm_hermitian(H, 0.5 * dt, factor=-1j)
        rho = U @ rho_m @ dagger(U)
        for i in range(self.N):
            g = float(np.clip(self._gamma_i(i, float(K[i])), 0.0, None))
            p = 0.5 * (1.0 - math.exp(-2.0 * g * dt))
            if p != 0.0:
                Zi = self.Z_ops[i]
                rho = (1.0 - p) * rho + p * (Zi @ rho @ Zi)
        rho = self._apply_collective_jump_channel(rho, dt=dt, Kvec=K)
        rho = U @ rho @ dagger(U)
        rho = _ensure_density(rho)
        self.rho = rho
        min_eig = float(np.min(np.linalg.eigvalsh(hermitize(rho)).real))
        return (rho, J_eff, K)

    def run(self, T_final: float, dt: float, store_every: int=1) -> NonlinearGKSLRun:
        T_final = float(T_final)
        dt = float(dt)
        store_every = max(int(store_every), 1)
        n_steps = int(round(T_final / dt))
        times: List[float] = []
        rho_hist: List[np.ndarray] = []
        K_hist: List[np.ndarray] = []
        J_hist: List[np.ndarray] = []
        min_eig_hist: List[float] = []
        self._J_buffer.reset()
        t = 0.0
        for step in range(n_steps + 1):
            if step % store_every == 0 or step == 0 or step == n_steps:
                rho = _ensure_density(self.rho)
                times.append(t)
                rho_hist.append(rho.copy())
                J_now = self._feedback_source_J(rho)
                self._J_buffer.push(J_now)
                J_eff = self._J_buffer.effective(delay_steps=int(self.delay_steps), kernel=str(self.memory_kernel), dt=float(dt), tau=self.kernel_tau)
                k_eq = self.solve_G(-1.0 / max(self.alpha_area, 1e-15) * J_eff)
                if self.volume_constraint_active:
                    k_eq = project_zero_mean(k_eq)
                K = self.K_ref + k_eq
                J_hist.append(J_eff.copy())
                K_hist.append(K.copy())
                min_eig_hist.append(float(np.min(np.linalg.eigvalsh(hermitize(rho)).real)))
            if step == n_steps:
                break
            self.step(dt, dt_for_kernel=dt)
            t += dt
        return NonlinearGKSLRun(times=np.array(times, dtype=float), rho_hist=rho_hist, K_hist=K_hist, J_hist=J_hist, min_eig_hist=min_eig_hist)

class OCTAReviewerExtensionsGTRQCSimulator(GTRQCSimulator):

    def __init__(self, *, surface: Optional[TriangulatedSurface]=None, delay_steps: int=0, memory_kernel: str='delta', kernel_tau: Optional[float]=None, edge_curv_strength: float=0.15, edge_curv_clip: Tuple[float, float]=(0.25, 4.0), transverse_hx: float=0.0, collective_jump_base: float=0.0, collective_jump_kappa: float=2.0, collective_jump_safety: float=0.9, **base_kwargs: Any):
        self._surface_for_interp = surface
        self.delay_steps = max(int(delay_steps), 0)
        self.memory_kernel = str(memory_kernel).strip().lower()
        self.kernel_tau = kernel_tau
        self.transverse_hx = float(transverse_hx)
        self.edge_curv_strength = float(edge_curv_strength)
        self.edge_curv_clip = (float(edge_curv_clip[0]), float(edge_curv_clip[1]))
        self.collective_jump_base = float(collective_jump_base)
        self.collective_jump_kappa = float(collective_jump_kappa)
        self.collective_jump_safety = float(collective_jump_safety)
        self._runtime_dt_for_kernel = 1.0
        super().__init__(**base_kwargs)
        coords = np.array([n.coord if n.coord is not None else np.zeros(3, dtype=float) for n in self.nodes], dtype=float)
        A = np.asarray(self.graph.adjacency, dtype=float) > 0.0
        self._edges = [(i, j) for i in range(self.N) for j in range(i + 1, self.N) if A[i, j]]
        self._edge_len = np.array([float(np.linalg.norm(coords[i] - coords[j])) for i, j in self._edges], dtype=float)
        self._edge_len = np.clip(self._edge_len, 1e-12, None)
        self._edge_J0 = np.array([float(self.exchange_J[i, j]) for i, j in self._edges], dtype=float)
        if self._surface_for_interp is not None:
            Av = vertex_areas_barycentric(self._surface_for_interp)
            Av = np.clip(np.asarray(Av, dtype=float), 1e-12, None)
        else:
            Av = np.ones(self.N, dtype=float)
        self._A_vertex = Av
        self._Kdens_ref = self.K_ref / self._A_vertex
        self._sigma_minus = [0.5 * (self.X_ops[i] + 1j * self.Y_ops[i]) for i in range(self.N)]
        self._regions = self._default_regions_for_octa(coords)
        self._J_buffer = RelativisticDelayBuffer(maxlen=max(8, self.delay_steps + 4), shape=(self.N,), init=0.0)
        self._H_list_edge = self._build_H_list_edge_integrated()

    def _default_regions_for_octa(self, coords: np.ndarray) -> List[List[int]]:
        regions: List[List[int]] = [list(B) for B in self.balls]
        used = set()
        pairs = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dij = float(np.linalg.norm(coords[i] - coords[j]))
                pairs.append((dij, i, j))
        pairs.sort(reverse=True, key=lambda x: x[0])
        for _d, i, j in pairs:
            if i in used or j in used:
                continue
            used.add(i)
            used.add(j)
            regions.append([int(i), int(j)])
            if len(used) >= min(self.N, 6):
                break
        return regions

    def _build_H_list_edge_integrated(self) -> np.ndarray:
        lo, hi = self.edge_curv_clip
        H_list = []
        for a in range(self.M):
            Kvec = np.asarray(self.K_cfg[a], dtype=float).reshape(self.N)
            Kdens = Kvec / self._A_vertex
            dKdens = Kdens - self._Kdens_ref
            H = np.zeros((self.d, self.d), dtype=np.complex128)
            for i in range(self.N):
                H += 0.5 * self._omega_i(i, float(Kvec[i])) * self.Z_ops[i]
            hx = float(getattr(self, 'transverse_hx', 0.0))
            if hx != 0.0:
                for i in range(self.N):
                    H += 0.5 * hx * self.X_ops[i]
            for idx, (i, j) in enumerate(self._edges):
                Jij0 = float(self._edge_J0[idx])
                if Jij0 == 0.0:
                    continue
                integral = 0.5 * float(dKdens[i] + dKdens[j]) * float(self._edge_len[idx])
                factor = 1.0 + float(self.edge_curv_strength) * integral
                factor = float(np.clip(factor, lo, hi))
                Jij = Jij0 * factor
                H += 0.5 * Jij * (self.X_ops[i] @ self.X_ops[j] + self.Y_ops[i] @ self.Y_ops[j])
            H_list.append(hermitize(H))
        return np.stack(H_list, axis=0)

    def apply_curvature_shift(self, delta: np.ndarray, *, rebuild_fixed_rates: bool=True) -> None:
        super().apply_curvature_shift(delta, rebuild_fixed_rates=rebuild_fixed_rates)
        self._H_list_edge = self._build_H_list_edge_integrated()

    def _compute_rates_from_current_J(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho_m = self.matter_marginal()
        J_now = self.modular_sources_J(rho_m)
        self._J_buffer.push(J_now)
        J_eff = self._J_buffer.effective(delay_steps=int(self.delay_steps), kernel=str(self.memory_kernel), dt=float(self._runtime_dt_for_kernel), tau=self.kernel_tau)
        FJ = np.array([slice_F(self._k(self.K_cfg[a]), J_eff, self.G, self.alpha_area) for a in range(self.M)], dtype=float)
        piJ = gibbs_pi(FJ, self.beta_geom)
        R = metropolis_rates(FJ, self.A_cfg, self.beta_geom, self.gamma_geom_base)
        out_rate = np.sum(R, axis=0)
        return (R, out_rate, piJ)

    def _apply_matter_unitary(self, dt: float) -> None:
        dt = float(dt)
        U_list = [expm_hermitian(self._H_list_edge[a], dt, factor=-1j) for a in range(self.M)]
        if self.geometry_quantum:
            for s in range(self.S):
                for a in range(self.M):
                    Ua = U_list[a]
                    for b in range(self.M):
                        Ub = U_list[b]
                        self.rho[s, a, b] = Ua @ self.rho[s, a, b] @ dagger(Ub)
        else:
            for s in range(self.S):
                for a in range(self.M):
                    Ua = U_list[a]
                    self.rho[s, a] = Ua @ self.rho[s, a] @ dagger(Ua)

    def _collective_jump_ops(self, K_eff: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        if self.collective_jump_base <= 0.0:
            return []
        K_eff = np.asarray(K_eff, dtype=float).reshape(self.N)
        ops: List[Tuple[float, np.ndarray]] = []
        for reg in self._regions:
            reg = [int(x) for x in reg]
            dK = float(np.sum(K_eff[reg] - self.K_ref[reg])) / max(float(len(reg)), 1.0)
            strength = 1.0 - math.exp(-float(self.collective_jump_kappa) * (dK * dK))
            gamma = float(self.collective_jump_base) * float(np.clip(strength, 0.0, 1.0))
            if gamma <= 0.0:
                continue
            L = np.zeros((self.d, self.d), dtype=np.complex128)
            for i in reg:
                L += self._sigma_minus[int(i)]
            L = L / math.sqrt(max(len(reg), 1))
            ops.append((gamma, L))
        return ops

    def _apply_collective_curvature_jumps_kraus(self, dt: float) -> None:
        dt = float(dt)
        if dt <= 0.0:
            return
        p_geom = self.geom_prob()
        K_eff = self.K_expect(p_geom)
        ops = self._collective_jump_ops(K_eff)
        if not ops:
            return
        I = np.eye(self.d, dtype=np.complex128)
        S_op = np.zeros((self.d, self.d), dtype=np.complex128)
        for gamma, L in ops:
            S_op += float(gamma) * hermitize(dagger(L) @ L)
        S_op = hermitize(S_op)
        lam = float(np.max(np.linalg.eigvalsh(S_op).real))
        if lam <= 0.0:
            return
        dt_cap = float(self.collective_jump_safety) / max(lam, 1e-15)
        n_sub = int(math.ceil(dt / dt_cap))
        dt_sub = dt / max(n_sub, 1)

        def apply_to_block(X: np.ndarray) -> np.ndarray:
            X_in = X
            Q = hermitize(I - dt_sub * S_op)
            K0 = matrix_sqrt_psd(Q, eps=0.0)
            Y = K0 @ X_in @ dagger(K0)
            for gamma, L in ops:
                Jop = math.sqrt(dt_sub * float(gamma)) * L
                Y += Jop @ X_in @ dagger(Jop)
            return hermitize(Y)
        for _ in range(n_sub):
            if self.geometry_quantum:
                rho_in = self.rho
                rho_out = np.zeros_like(rho_in)
                for s in range(self.S):
                    for a in range(self.M):
                        for b in range(self.M):
                            rho_out[s, a, b] = apply_to_block(rho_in[s, a, b])
                self.rho = rho_out
            else:
                rho_in = self.rho
                rho_out = np.zeros_like(rho_in)
                for s in range(self.S):
                    for a in range(self.M):
                        rho_out[s, a] = apply_to_block(rho_in[s, a])
                self.rho = rho_out
            tr = self.total_trace()
            if np.isfinite(tr) and tr > 1e-15:
                self.rho /= tr

    def step_strang(self, dt: float) -> None:
        dt = float(dt)
        if dt <= 0.0:
            return
        self._apply_unitary_strang(0.5 * dt)
        self._apply_conditional_dephasing_exact(dt)
        self._apply_geometry_pointer_dephase(dt)
        if self.backreaction == 'embedded':
            self._apply_source_update_measurement()
        self._apply_geometry_jump_kraus(dt)
        self._apply_collective_curvature_jumps_kraus(dt)
        self._apply_unitary_strang(0.5 * dt)
        self._post_step_trace_check(where='step_strang+reviewer_extensions')

    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        dt = float(kwargs.get('dt', 1.0)) if 'dt' in kwargs else float(args[1])
        self._runtime_dt_for_kernel = float(dt)
        self._J_buffer.reset()
        return super().run(*args, **kwargs)

def force_geometry_delta(sim: GTRQCSimulator, a0: int=0, s0: int=0) -> None:
    a0 = int(a0)
    s0 = int(s0)
    sim.rho[...] = 0.0
    if sim.geometry_quantum:
        sim.rho[s0, a0, a0] = sim.rho0
    else:
        sim.rho[s0, a0] = sim.rho0

def force_geometry_uniform(sim: GTRQCSimulator, s0: int=0) -> None:
    s0 = int(s0)
    sim.rho[...] = 0.0
    w = 1.0 / max(float(sim.M), 1.0)
    if sim.geometry_quantum:
        for a in range(sim.M):
            sim.rho[s0, a, a] = w * sim.rho0
    else:
        for a in range(sim.M):
            sim.rho[s0, a] = w * sim.rho0

def octa_thermodynamic_limit_suite() -> Dict[str, Any]:
    T_FINAL = float(os.environ.get('GTRQC_TFINAL', '2.0'))
    DT = float(os.environ.get('GTRQC_DT', '0.02'))
    STORE_EVERY = int(os.environ.get('GTRQC_STORE', '1'))
    HX = float(os.environ.get('GTRQC_OCTA_TRANSVERSE_HX', '0.25'))
    OUT_PREFIX = os.environ.get('GTRQC_OUT_PREFIX', 'gTRQC_OCTA_SUITE')
    SAVE_ARTIFACTS = bool(int(os.environ.get('GTRQC_SAVE_ARTIFACTS', '1')))
    artifacts: List[Dict[str, str]] = []
    M_LIST = [int(x) for x in os.environ.get('GTRQC_OCTA_M_LIST', '10,20,40').split(',') if x.strip()]
    LEVELS_PER_NODE = int(os.environ.get('GTRQC_OCTA_LEVELS_PER_NODE', '7'))
    DELTAK = float(os.environ.get('GTRQC_OCTA_DELTAK', '0.12'))
    DELAY_STEPS = int(os.environ.get('GTRQC_OCTA_DELAY_STEPS', '0'))
    KERNEL = os.environ.get('GTRQC_OCTA_KERNEL', 'delta').strip().lower()
    KERNEL_TAU = float(os.environ.get('GTRQC_OCTA_KERNEL_TAU', '0.20'))
    COLLECTIVE_BASE = float(os.environ.get('GTRQC_OCTA_COLLECTIVE_JUMP_BASE', '0.0'))
    COLLECTIVE_KAPPA = float(os.environ.get('GTRQC_OCTA_COLLECTIVE_JUMP_KAPPA', '2.0'))
    EDGE_CURV_STRENGTH = float(os.environ.get('GTRQC_OCTA_EDGE_CURV_STRENGTH', '0.15'))
    KFOLDS = int(os.environ.get('GTRQC_KFOLDS', '3'))
    MODES = [m.strip() for m in os.environ.get('GTRQC_MODES', 'linear_fixed,linear_operator_functional,paper_instantaneous_j_gibbs,paper_instantaneous_j_gibbs_operator').split(',') if m.strip()]
    MODE_ALIASES = [GTRQCSimulator._MODE_ALIASES.get(m.strip().lower(), m.strip().lower()) for m in MODES]
    fold_baselines: List[Dict[str, Any]] = []
    for fold in range(KFOLDS):
        surf = geometry_builder('octa', seed=100 + 17 * fold)
        nodes, graph, _ = surface_to_nodes_graph(surf, name_prefix='o', compute_frames=False, frame_seed=0)
        nl = SemiclassicalNonlinearGKSLSimulator(nodes=nodes, graph=graph, surface=surf, ball_radius_hops=1, alpha_area=1.0, mass_m=0.7, mesh_h=1.0, omega_0=2.0 * math.pi * 40.0, alpha_curv_freq=0.2, gamma0=0.3, gamma_curv=1.2, exchange_J_strength=2.5, transverse_hx=HX, eps_modular=0.002, eps_ref_rank=1e-06, reference_state='zeros', initial_state='plus_product', edge_curv_strength=EDGE_CURV_STRENGTH, collective_jump_base=COLLECTIVE_BASE, collective_jump_kappa=COLLECTIVE_KAPPA, delay_steps=DELAY_STEPS, memory_kernel=KERNEL, kernel_tau=KERNEL_TAU if KERNEL == 'exp' else None, feedback_mode='modular', energy_feedback_scale=1.0)
        nl_run = nl.run(T_final=T_FINAL, dt=DT, store_every=STORE_EVERY)
        fold_baselines.append(dict(fold=int(fold), surface=surf, nodes=nodes, graph=graph, nl_run=nl_run))
        if SAVE_ARTIFACTS:
            arrays_nl = {'nl_times': np.asarray(nl_run.times, dtype=float), 'nl_rho_hist': np.stack(nl_run.rho_hist, axis=0).astype(np.complex128), 'nl_K_hist': np.stack(nl_run.K_hist, axis=0).astype(float), 'nl_J_hist': np.stack(nl_run.J_hist, axis=0).astype(float), 'nl_min_eig_hist': np.asarray(nl_run.min_eig_hist, dtype=float)}
            meta_nl = {'timestamp': now_stamp(), 'suite': 'octa_thermodynamic_limit_suite', 'kind': 'nonlinear_baseline', 'geometry': 'octa', 'fold': int(fold), 'T_final': float(T_FINAL), 'dt': float(DT), 'store_every': int(STORE_EVERY), 'delay_steps': int(DELAY_STEPS), 'kernel': str(KERNEL), 'kernel_tau': float(KERNEL_TAU) if KERNEL == 'exp' else None, 'collective_jump_base': float(COLLECTIVE_BASE), 'collective_jump_kappa': float(COLLECTIVE_KAPPA), 'edge_curv_strength': float(EDGE_CURV_STRENGTH), 'transverse_hx': float(HX), 'note': 'Semiclassical nonlinear GKSL baseline (CPTP per-step construction), saved per fold.'}
            npz_path, json_path = save_npz_json(OUT_PREFIX, f'OCTA_NONLINEAR_BASELINE_fold{fold}', arrays_nl, meta_nl)
            artifacts.append({'label': f'OCTA_NONLINEAR_BASELINE_fold{fold}', 'npz': npz_path, 'json': json_path})
            print(f'[ARTIFACT SAVED] OCTA_NONLINEAR_BASELINE_fold{fold}', flush=True)
            print(f'    NPZ : {npz_path}', flush=True)
            print(f'    JSON: {json_path}', flush=True)
    comparisons: List[Dict[str, Any]] = []
    aggregate_by_config: List[Dict[str, Any]] = []
    AGG_KEYS = ['final_trace_distance', 'mean_trace_distance', 'max_trace_distance', 'residual_norm_final', 'trace_drift_abs']

    def _group_key(mode_alias: str, M: int, geom_quantum: bool) -> Tuple[str, int, int]:
        return (str(mode_alias), int(M), int(bool(geom_quantum)))
    grouped: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    for mode_in, mode_alias in zip(MODES, MODE_ALIASES):
        for M in M_LIST:
            for geom_quantum in (True, False):
                for fold in range(KFOLDS):
                    fb = fold_baselines[fold]
                    surf = fb['surface']
                    nodes = fb['nodes']
                    graph = fb['graph']
                    nl_run = fb['nl_run']
                    seed_ensemble = int(77 + 97 * int(M) + (1000 if bool(geom_quantum) else 0) + 13 * int(fold) + stable_hash_int('octa', mode_alias, int(M), int(bool(geom_quantum)), int(fold), mod=10000))
                    sim = OCTAReviewerExtensionsGTRQCSimulator(surface=surf, delay_steps=DELAY_STEPS, memory_kernel=KERNEL, kernel_tau=KERNEL_TAU if KERNEL == 'exp' else None, edge_curv_strength=EDGE_CURV_STRENGTH, collective_jump_base=COLLECTIVE_BASE, collective_jump_kappa=COLLECTIVE_KAPPA, nodes=nodes, graph=graph, ball_radius_hops=1, alpha_area=1.0, regge_a=0.8, mass_m=0.7, mesh_h=1.0, beta_geom=8.0, gamma_geom_base=40.0, levels_per_node=LEVELS_PER_NODE, deltaK_scale=DELTAK, max_configs=int(M), seed_geometry_ensemble=int(seed_ensemble), omega_0=2.0 * math.pi * 40.0, alpha_curv_freq=0.2, gamma0=0.3, gamma_curv=1.2, exchange_J_strength=2.5, transverse_hx=HX, eps_modular=0.002, eps_ref_rank=1e-06, reference_state='zeros', initial_state='plus_product', negativity_pair=(0, len(nodes) - 1), rate_mode=str(mode_alias), geometry_quantum=bool(geom_quantum), geometry_initial='coherent_gibbs' if geom_quantum else 'diagonal_gibbs', geom_mix_rate=1.0 if geom_quantum else 0.0, geom_pointer_dephase=2.0 if geom_quantum else 0.0, enforce_zero_mean_if_massless=True, enforce_gauss_bonnet=True, causal_apply_every=0, trace_mode='hygiene', trace_tol=5e-11, trace_hygiene_tol=1e-06, memory_guard_gib=2.0)
                    force_geometry_delta(sim, a0=0)
                    n_steps = int(round(T_FINAL / DT))
                    progress_every = max(1, n_steps // 10)
                    label = f'OCTA k-fold | mode={mode_alias} (in={mode_in}) | M={int(M)} | geom_quantum={bool(geom_quantum)} | fold={int(fold) + 1}/{int(KFOLDS)} | delay={int(DELAY_STEPS)} | kernel={KERNEL}'
                    summ = sim.run(T_final=T_FINAL, dt=DT, store_every=STORE_EVERY, label=label, progress_every=progress_every, store_matter_states=True)
                    assert sim.rho_m_hist is not None
                    L = min(len(sim.rho_m_hist), len(nl_run.rho_hist))
                    err_t = np.array([trace_distance(sim.rho_m_hist[k], nl_run.rho_hist[k]) for k in range(L)], dtype=float)
                    row = dict(mode=str(mode_alias), mode_in=str(mode_in), fold=int(fold), KFOLDS=int(KFOLDS), M=int(M), geometry_quantum=bool(geom_quantum), delay_steps=int(DELAY_STEPS), kernel=str(KERNEL), kernel_tau=float(KERNEL_TAU) if KERNEL == 'exp' else None, mean_trace_distance=float(np.mean(err_t)) if err_t.size else 0.0, max_trace_distance=float(np.max(err_t)) if err_t.size else 0.0, final_trace_distance=float(err_t[-1]) if err_t.size else 0.0, residual_norm_final=float(summ.get('residual_norm_final', 0.0)), trace_drift_abs=float(summ.get('trace_drift_abs', 0.0)), gtrqc_summary=summ)
                    comparisons.append(row)
                    grouped.setdefault(_group_key(mode_alias, int(M), bool(geom_quantum)), []).append(row)
                    if SAVE_ARTIFACTS:
                        run_label = f'OCTA_mode{sanitize_filename(str(mode_alias))}_M{int(M)}_gq{int(bool(geom_quantum))}_fold{int(fold)}_delay{int(DELAY_STEPS)}_kernel{sanitize_filename(str(KERNEL))}'
                        arrays_run = {'times': np.asarray(sim.times, dtype=float), 'residual_norm': np.asarray(sim.res_hist, dtype=float), 'trace_to_ref': np.asarray(sim.Tdist_hist, dtype=float), 'trace': np.asarray(sim.trace_hist, dtype=float), 'J': np.stack(sim.J_hist, axis=0).astype(float), 'K_eff': np.stack(sim.K_eff_hist, axis=0).astype(float), 'K_eq': np.stack(sim.K_eq_hist, axis=0).astype(float), 'rho_m_hist': np.stack(sim.rho_m_hist[:L], axis=0).astype(np.complex128), 'nl_times': np.asarray(nl_run.times[:L], dtype=float), 'nl_rho_hist': np.stack(nl_run.rho_hist[:L], axis=0).astype(np.complex128), 'trace_distance_to_nonlinear': np.asarray(err_t, dtype=float)}
                        meta_run = {'timestamp': now_stamp(), 'suite': 'octa_thermodynamic_limit_suite', 'geometry': 'octa', 'mode': str(mode_alias), 'mode_in': str(mode_in), 'fold': int(fold), 'KFOLDS': int(KFOLDS), 'M': int(M), 'geometry_quantum': bool(geom_quantum), 'T_final': float(T_FINAL), 'dt': float(DT), 'store_every': int(STORE_EVERY), 'delay_steps': int(DELAY_STEPS), 'kernel': str(KERNEL), 'kernel_tau': float(KERNEL_TAU) if KERNEL == 'exp' else None, 'collective_jump_base': float(COLLECTIVE_BASE), 'collective_jump_kappa': float(COLLECTIVE_KAPPA), 'edge_curv_strength': float(EDGE_CURV_STRENGTH), 'transverse_hx': float(HX), 'gtrqc_summary': summ, 'note': 'Per-fold artifact for OCTA k-fold × multi-mode comparison against nonlinear baseline.'}
                        npz_path, json_path = save_npz_json(OUT_PREFIX, run_label, arrays_run, meta_run)
                        artifacts.append({'label': run_label, 'npz': npz_path, 'json': json_path})
                        print(f'[ARTIFACT SAVED] {run_label}', flush=True)
                        print(f'    NPZ : {npz_path}', flush=True)
                        print(f'    JSON: {json_path}', flush=True)
    for (mode_alias, M, gq_int), rows in grouped.items():
        agg = summarize_folds(rows, AGG_KEYS)
        aggregate_by_config.append(dict(mode=str(mode_alias), M=int(M), geometry_quantum=bool(gq_int), KFOLDS=int(KFOLDS), aggregate_metrics=agg, n_rows=int(len(rows))))
    return {'ok': True, 'geometry': 'octa', 'T_final': float(T_FINAL), 'dt': float(DT), 'store_every': int(STORE_EVERY), 'KFOLDS': int(KFOLDS), 'modes_in': list(map(str, MODES)), 'modes_alias': list(map(str, MODE_ALIASES)), 'M_list': [int(x) for x in M_LIST], 'delay_steps': int(DELAY_STEPS), 'kernel': str(KERNEL), 'kernel_tau': float(KERNEL_TAU) if KERNEL == 'exp' else None, 'collective_jump_base': float(COLLECTIVE_BASE), 'collective_jump_kappa': float(COLLECTIVE_KAPPA), 'edge_curv_strength': float(EDGE_CURV_STRENGTH), 'transverse_hx': float(HX), 'nonlinear_baseline': {'feedback_mode': 'modular', 'per_fold_min_eig_min': [float(np.min(np.array(fb['nl_run'].min_eig_hist, float))) if fb['nl_run'].min_eig_hist else 0.0 for fb in fold_baselines]}, 'comparisons': comparisons, 'aggregate_by_config': aggregate_by_config, 'artifacts': artifacts, 'note': 'OCTA k-fold × multi-mode suite: for each fold we build an OCTA triangulation, run a nonlinear semiclassical GKSL baseline, then run gTRQC-with-register for each requested mode and (M,geometry_quantum) setting. Comparisons use trace distance between stored matter marginals at identical store times. All prior artifact saving + printing behavior is preserved; labels now include mode+fold.'}

def main_octa_reviewer_suite() -> None:
    print('[OCTA SUITE] START', flush=True)
    out = octa_thermodynamic_limit_suite()
    print('[OCTA SUITE] DONE', flush=True)
    print(json.dumps(out, indent=2), flush=True)
