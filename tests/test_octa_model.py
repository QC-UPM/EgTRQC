"""Tests for the OCTA-ready graph model."""

from __future__ import annotations

import numpy as np

from egtrqc.config import DelayConfig, KernelConfig, ReproducibilityConfig, SimulationConfig
from egtrqc.diagnostics import audit_delay_definition, audit_single_advance_per_step
from egtrqc.geometry import build_balls, build_octa_graph, build_response_operator, project_zero_mean
from egtrqc.octa_model import GraphBackreactionModel, GraphBackreactionModelConfig
from egtrqc.simulator import DelayAwareSimulator


def test_graph_backreaction_model_runs_with_delay_audits() -> None:
    """The OCTA-ready graph model should preserve the same delay guarantees."""
    model = GraphBackreactionModel(build_octa_graph(), GraphBackreactionModelConfig())
    config = SimulationConfig(
        total_time=0.25,
        dt=0.05,
        delay=DelayConfig(delay_steps=2, kernel=KernelConfig(kind="delta")),
        reproducibility=ReproducibilityConfig(seed=19, store_every=1),
    )

    result = DelayAwareSimulator(model=model, config=config).run()

    assert audit_delay_definition(result).passed
    assert audit_single_advance_per_step(result).passed
    assert result.records[0].state.shape[0] == 6


def test_build_balls_matches_radius_one_octa_neighborhoods() -> None:
    """Graph balls should reproduce radius-1 neighborhoods from the notebook."""
    graph = build_octa_graph()
    balls = build_balls(graph, radius_hops=1)

    assert len(balls) == graph.num_nodes
    assert balls[0] == [0, 1, 2, 3, 4]
    assert balls[5] == [1, 2, 3, 4, 5]


def test_response_operator_matches_notebook_formula() -> None:
    """The response operator should implement `(1/h^2)L + m^2 I`."""
    graph = build_octa_graph()
    response = build_response_operator(graph, mass_m=0.7, mesh_h=1.0)
    laplacian = graph.laplacian()

    np.testing.assert_allclose(response, laplacian + (0.49 * np.eye(graph.num_nodes)))


def test_massless_response_enforces_zero_mean_curvature() -> None:
    """Massless solves should remain in the zero-mean subspace."""
    graph = build_octa_graph()
    model = GraphBackreactionModel(
        graph,
        GraphBackreactionModelConfig(mass_m=0.0, enforce_zero_mean_if_massless=True, source_bias=0.0),
    )
    rhs = np.array([2.0, -1.0, 0.0, 1.0, -2.0, 0.0], dtype=np.float64)

    solved = model.solve_response(rhs)

    assert abs(float(np.mean(solved))) <= 1e-12
    np.testing.assert_allclose(project_zero_mean(solved), solved)


def test_modular_source_is_centered_against_reference_state() -> None:
    """The modular-source proxy should vanish on the reference state up to bias."""
    config = GraphBackreactionModelConfig(source_gain=0.8, source_bias=0.0)
    model = GraphBackreactionModel(build_octa_graph(), config)

    source = model.modular_sources(model.initial_state())

    np.testing.assert_allclose(source, np.zeros_like(source))
