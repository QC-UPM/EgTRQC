"""Geometry primitives for graph-based EgTRQC experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GeometryGraph:
    """Discrete geometry graph used by graph-based backreaction models.

    Attributes:
        adjacency: Symmetric adjacency matrix.
        curvature_reference: Reference curvature value on each node.
        node_positions: Optional Cartesian node coordinates.
    """

    adjacency: FloatArray
    curvature_reference: FloatArray
    node_positions: FloatArray | None = None

    def __post_init__(self) -> None:
        """Validate graph consistency."""
        adjacency = np.asarray(self.adjacency, dtype=np.float64)
        curvature_reference = np.asarray(self.curvature_reference, dtype=np.float64)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a square matrix.")
        if curvature_reference.shape != (adjacency.shape[0],):
            raise ValueError("curvature_reference must match the number of nodes.")
        if not np.allclose(adjacency, adjacency.T):
            raise ValueError("adjacency must be symmetric.")
        if self.node_positions is not None:
            node_positions = np.asarray(self.node_positions, dtype=np.float64)
            if node_positions.shape[0] != adjacency.shape[0]:
                raise ValueError("node_positions must match the number of nodes.")

    @property
    def num_nodes(self) -> int:
        """Return the number of graph nodes."""
        return int(self.adjacency.shape[0])

    @property
    def degree_vector(self) -> FloatArray:
        """Return the weighted node-degree vector."""
        return np.sum(self.adjacency, axis=1)

    def laplacian(self) -> FloatArray:
        """Return the weighted graph Laplacian."""
        return np.diag(self.degree_vector) - np.asarray(self.adjacency, dtype=np.float64)


def build_balls(graph: GeometryGraph, radius_hops: int) -> list[list[int]]:
    """Build graph neighborhoods for the OCTA-inspired graph construction.

    Args:
        graph: Geometry graph.
        radius_hops: Neighborhood radius in graph hops.

    Returns:
        One sorted node list per graph vertex.
    """
    adjacency = np.asarray(graph.adjacency, dtype=np.float64) > 0.0
    num_nodes = graph.num_nodes
    radius = max(int(radius_hops), 0)

    balls: list[list[int]] = []
    for vertex in range(num_nodes):
        seen = {vertex}
        frontier = {vertex}
        for _ in range(radius):
            nxt: set[int] = set()
            for node in frontier:
                nxt.update(np.where(adjacency[node])[0].tolist())
            nxt.difference_update(seen)
            seen.update(nxt)
            frontier = nxt
        balls.append(sorted(seen))
    return balls


def build_response_operator(graph: GeometryGraph, mass_m: float, mesh_h: float) -> FloatArray:
    """Build the discrete response operator `G = (1/h^2)L + m^2 I`.

    Args:
        graph: Geometry graph.
        mass_m: Mass-like parameter.
        mesh_h: Mesh spacing.

    Returns:
        Dense response operator.
    """
    if mesh_h <= 0.0:
        raise ValueError("mesh_h must be strictly positive.")
    laplacian = graph.laplacian()
    return (laplacian / (mesh_h * mesh_h)) + ((mass_m * mass_m) * np.eye(graph.num_nodes, dtype=np.float64))


def project_zero_mean(values: FloatArray) -> FloatArray:
    """Project a vector onto the zero-mean subspace.

    Args:
        values: Input vector.

    Returns:
        Zero-mean vector.
    """
    arr = np.asarray(values, dtype=np.float64)
    return arr - float(np.mean(arr))


def build_octa_graph() -> GeometryGraph:
    """Build a small OCTA-like graph for migration experiments.

    Returns:
        Six-node octahedral graph with unit-weight edges.
    """
    adjacency = np.array(
        [
            [0, 1, 1, 1, 1, 0],
            [1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 1],
            [0, 1, 1, 1, 1, 0],
        ],
        dtype=np.float64,
    )
    curvature_reference = np.zeros(adjacency.shape[0], dtype=np.float64)
    node_positions = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    return GeometryGraph(adjacency=adjacency, curvature_reference=curvature_reference, node_positions=node_positions)
