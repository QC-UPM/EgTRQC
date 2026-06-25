"""Tests for the geometry ensemble utilities."""

from __future__ import annotations

import numpy as np

from egtrqc.ensemble import (
    build_config_graph,
    build_geometry_ensemble,
    gibbs_distribution,
    make_geometry_ensemble,
    metropolis_rate_matrix,
    slice_free_energy,
)
from egtrqc.geometry import build_octa_graph
from egtrqc.octa_model import GraphBackreactionModel, GraphBackreactionModelConfig, GeometryRegisterConfig


def test_build_geometry_ensemble_keeps_reference_config_first() -> None:
    """The first configuration should be the reference geometry."""
    graph = build_octa_graph()
    k_cfg, codes, offsets = build_geometry_ensemble(
        graph.curvature_reference,
        levels_per_node=3,
        delta_scale=0.2,
        max_configs=8,
        seed=7,
        enforce_total_curvature=True,
    )

    np.testing.assert_allclose(k_cfg[0], graph.curvature_reference)
    assert codes.shape == (8, graph.num_nodes)
    assert offsets.shape == (3,)


def test_build_config_graph_connects_single_level_moves() -> None:
    """Configuration graph edges should encode one-node one-level changes."""
    codes = np.array([[1, 1], [2, 1], [1, 2], [2, 2]], dtype=np.int64)
    adjacency, edge_info = build_config_graph(codes)

    assert adjacency[0, 1] == 1.0
    assert adjacency[0, 2] == 1.0
    assert adjacency[0, 3] == 0.0
    assert edge_info[(0, 1)] == (0, -1)


def test_gibbs_distribution_normalizes_probabilities() -> None:
    """Gibbs weights should be normalized and non-negative."""
    probs = gibbs_distribution(np.array([0.0, 1.0, 2.0], dtype=np.float64), beta=2.0)

    assert np.all(probs >= 0.0)
    assert abs(float(np.sum(probs)) - 1.0) <= 1e-12
    assert probs[0] > probs[1] > probs[2]


def test_graph_model_exposes_geometry_register_thermodynamics() -> None:
    """The graph model should compute free energies, Gibbs weights, and rates."""
    graph = build_octa_graph()
    model = GraphBackreactionModel(
        graph,
        GraphBackreactionModelConfig(
            source_bias=0.0,
            geometry_register=GeometryRegisterConfig(max_configs=10, levels_per_node=3, seed_geometry_ensemble=5),
        ),
    )
    source = model.modular_sources(model.initial_state())
    free_energies = model.free_energies(source)
    probs = model.gibbs_distribution(source)
    rates = model.metropolis_rate_matrix(source)

    assert free_energies.shape == (10,)
    assert probs.shape == (10,)
    assert rates.shape == (10, 10)
    assert abs(float(np.sum(probs)) - 1.0) <= 1e-12
    assert np.all(rates >= 0.0)


def test_slice_free_energy_matches_manual_quadratic_form() -> None:
    """Free-energy slices should match the manual quadratic expression."""
    k = np.array([1.0, -1.0], dtype=np.float64)
    j = np.array([0.5, 0.25], dtype=np.float64)
    g = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64)

    value = slice_free_energy(k, j, g, alpha_area=1.5)

    expected = 0.5 * 1.5 * float(k @ (g @ k)) + float(j @ k)
    assert abs(value - expected) <= 1e-12
