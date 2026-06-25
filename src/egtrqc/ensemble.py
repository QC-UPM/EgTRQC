"""Geometry-ensemble utilities migrated from the OCTA notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class GeometryEnsemble:
    """Discrete geometry ensemble and its configuration graph.

    Attributes:
        curvature_configurations: One curvature vector per retained configuration.
        codes: Integer level code for each configuration and node.
        level_offsets: Physical curvature offsets associated with the discrete levels.
        adjacency: Configuration-graph adjacency.
        edge_info: Mapping from directed config edges to `(vertex, sign)`.
    """

    curvature_configurations: FloatArray
    codes: IntArray
    level_offsets: FloatArray
    adjacency: FloatArray
    edge_info: dict[tuple[int, int], tuple[int, int]]



def build_geometry_ensemble(
    curvature_reference: FloatArray,
    *,
    levels_per_node: int,
    delta_scale: float,
    max_configs: int,
    seed: int,
    enforce_total_curvature: bool = False,
    total_curvature_target: float | None = None,
) -> tuple[FloatArray, IntArray, FloatArray]:
    """Build the notebook-style discrete curvature ensemble.

    Args:
        curvature_reference: Reference curvature vector.
        levels_per_node: Number of discrete levels per node.
        delta_scale: Distance between adjacent levels.
        max_configs: Maximum number of retained configurations.
        seed: Deterministic seed.
        enforce_total_curvature: Whether to enforce constant total curvature.
        total_curvature_target: Optional explicit target total curvature.

    Returns:
        Tuple `(curvature_configurations, codes, level_offsets)`.
    """
    curvature_reference_arr = np.asarray(curvature_reference, dtype=np.float64)
    num_nodes = int(curvature_reference_arr.shape[0])
    levels = int(levels_per_node)
    if levels < 1:
        raise ValueError("levels_per_node must be positive.")
    if max_configs < 1:
        raise ValueError("max_configs must be positive.")

    rng = np.random.default_rng(int(seed))
    idx = np.arange(levels, dtype=np.float64)
    center = 0.5 * (levels - 1)
    level_offsets = (idx - center) * float(delta_scale)

    reference_level = int(round(center))
    reference_code = np.full(num_nodes, reference_level, dtype=np.int64)
    codes: list[IntArray] = [reference_code.copy()]
    seen = {tuple(reference_code.tolist())}

    while len(codes) < int(max_configs):
        candidate = reference_code.copy()
        touched = rng.integers(1, max(2, min(4, num_nodes + 1)))
        vertices = rng.choice(num_nodes, size=int(touched), replace=False)
        for vertex in np.atleast_1d(vertices):
            direction = int(rng.choice([-1, 1]))
            candidate[int(vertex)] = int(np.clip(candidate[int(vertex)] + direction, 0, levels - 1))
        key = tuple(candidate.tolist())
        if key in seen:
            continue
        seen.add(key)
        codes.append(candidate)

    codes_arr = np.stack(codes, axis=0)
    curvature_configurations = np.array(
        [curvature_reference_arr + level_offsets[codes_arr[idx_code]] for idx_code in range(codes_arr.shape[0])],
        dtype=np.float64,
    )

    if enforce_total_curvature:
        target = (
            float(np.sum(curvature_reference_arr))
            if total_curvature_target is None
            else float(total_curvature_target)
        )
        drift = (np.sum(curvature_configurations, axis=1) - target) / max(float(num_nodes), 1.0)
        curvature_configurations = curvature_configurations - drift[:, None]

    return curvature_configurations, codes_arr, level_offsets



def diff_vertex_and_sign(code_a: IntArray, code_b: IntArray) -> tuple[int, int] | None:
    """Return the single-node level change between two configurations.

    Args:
        code_a: First configuration code.
        code_b: Second configuration code.

    Returns:
        `(vertex, sign)` when the configurations differ by one unit at one node,
        otherwise `None`.
    """
    diff = np.asarray(code_a, dtype=np.int64) - np.asarray(code_b, dtype=np.int64)
    indices = np.where(diff != 0)[0]
    if indices.size != 1:
        return None
    vertex = int(indices[0])
    if abs(int(diff[vertex])) != 1:
        return None
    return vertex, int(np.sign(int(diff[vertex])))



def build_config_graph(codes: IntArray) -> tuple[FloatArray, dict[tuple[int, int], tuple[int, int]]]:
    """Build the configuration graph from integer level codes.

    Args:
        codes: Configuration codes.

    Returns:
        Tuple `(adjacency, edge_info)`.
    """
    codes_arr = np.asarray(codes, dtype=np.int64)
    num_configs = int(codes_arr.shape[0])
    adjacency = np.zeros((num_configs, num_configs), dtype=np.float64)
    edge_info: dict[tuple[int, int], tuple[int, int]] = {}
    for idx_a in range(num_configs):
        for idx_b in range(idx_a + 1, num_configs):
            delta = diff_vertex_and_sign(codes_arr[idx_a], codes_arr[idx_b])
            if delta is None:
                continue
            adjacency[idx_a, idx_b] = 1.0
            adjacency[idx_b, idx_a] = 1.0
            vertex, sign = delta
            edge_info[(idx_a, idx_b)] = (vertex, sign)
            edge_info[(idx_b, idx_a)] = (vertex, -sign)
    return adjacency, edge_info



def slice_free_energy(curvature: FloatArray, source: FloatArray, response: FloatArray, alpha_area: float) -> float:
    """Compute the notebook free-energy slice `F = 0.5*alpha*k^T G k + J^T k`.

    Args:
        curvature: Curvature perturbation vector.
        source: Source vector.
        response: Response operator `G`.
        alpha_area: Geometric prefactor.

    Returns:
        Scalar free energy.
    """
    curvature_arr = np.asarray(curvature, dtype=np.float64)
    source_arr = np.asarray(source, dtype=np.float64)
    response_arr = np.asarray(response, dtype=np.float64)
    return 0.5 * float(alpha_area) * float(curvature_arr @ (response_arr @ curvature_arr)) + float(source_arr @ curvature_arr)



def gibbs_distribution(free_energies: FloatArray, beta: float) -> FloatArray:
    """Compute the Gibbs distribution over geometry configurations.

    Args:
        free_energies: Free-energy values.
        beta: Inverse temperature.

    Returns:
        Normalized Gibbs weights.
    """
    free_energies_arr = np.asarray(free_energies, dtype=np.float64)
    shifted = -float(beta) * (free_energies_arr - float(np.min(free_energies_arr)))
    shifted = np.clip(shifted, -700.0, 700.0)
    weights = np.exp(shifted)
    weights = np.clip(weights, 0.0, None)
    return weights / max(float(np.sum(weights)), 1e-15)



def metropolis_rate_matrix(free_energies: FloatArray, adjacency: FloatArray, beta: float, gamma: float) -> FloatArray:
    """Build the notebook-style Metropolis rate matrix.

    Args:
        free_energies: Free-energy values.
        adjacency: Configuration-graph adjacency.
        beta: Inverse temperature.
        gamma: Base jump-rate scale.

    Returns:
        Directed rate matrix.
    """
    free_energies_arr = np.asarray(free_energies, dtype=np.float64)
    adjacency_arr = np.asarray(adjacency, dtype=np.float64)
    num_configs = int(free_energies_arr.shape[0])
    rates = np.zeros((num_configs, num_configs), dtype=np.float64)
    for idx_a in range(num_configs):
        for idx_b in range(num_configs):
            if idx_a == idx_b or adjacency_arr[idx_a, idx_b] <= 0.0:
                continue
            delta_energy = float(free_energies_arr[idx_a] - free_energies_arr[idx_b])
            rates[idx_a, idx_b] = float(gamma) * np.exp(-0.5 * float(beta) * delta_energy)
    return rates



def make_geometry_ensemble(
    curvature_reference: FloatArray,
    *,
    levels_per_node: int,
    delta_scale: float,
    max_configs: int,
    seed: int,
    enforce_total_curvature: bool = False,
    total_curvature_target: Optional[float] = None,
) -> GeometryEnsemble:
    """Build a full geometry ensemble and its configuration graph.

    Args:
        curvature_reference: Reference curvature vector.
        levels_per_node: Number of discrete levels per node.
        delta_scale: Distance between adjacent levels.
        max_configs: Maximum number of retained configurations.
        seed: Deterministic seed.
        enforce_total_curvature: Whether to enforce constant total curvature.
        total_curvature_target: Optional explicit target total curvature.

    Returns:
        Complete geometry ensemble package.
    """
    curvature_configurations, codes, level_offsets = build_geometry_ensemble(
        curvature_reference,
        levels_per_node=levels_per_node,
        delta_scale=delta_scale,
        max_configs=max_configs,
        seed=seed,
        enforce_total_curvature=enforce_total_curvature,
        total_curvature_target=total_curvature_target,
    )
    adjacency, edge_info = build_config_graph(codes)
    return GeometryEnsemble(
        curvature_configurations=curvature_configurations,
        codes=codes,
        level_offsets=level_offsets,
        adjacency=adjacency,
        edge_info=edge_info,
    )
