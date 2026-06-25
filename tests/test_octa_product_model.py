"""Tests for the product-state OCTA model."""

from __future__ import annotations

import numpy as np

from egtrqc.config import DelayConfig, KernelConfig, ReproducibilityConfig, SimulationConfig
from egtrqc.diagnostics import audit_delay_definition, audit_single_advance_per_step
from egtrqc.geometry import build_octa_graph
from egtrqc.octa_model import GeometryRegisterConfig
from egtrqc.octa_product_model import ProductStateOCTAModel, ProductStateOCTAModelConfig
from egtrqc.quantum import partial_trace_qubits, product_density_matrix
from egtrqc.simulator import DelayAwareSimulator


def test_product_state_modular_source_vanishes_at_reference_state() -> None:
    """Notebook-style modular sources should vanish at the reference state."""
    model = ProductStateOCTAModel(
        build_octa_graph(),
        ProductStateOCTAModelConfig(reference_excitation=0.15, initial_excitation=0.35),
    )

    source = model.modular_sources(model.initial_reference_state())

    np.testing.assert_allclose(source, np.zeros_like(source), atol=1e-12)


def test_product_state_modular_source_changes_under_state_perturbation() -> None:
    """Modular sources should respond nontrivially away from the reference state."""
    model = ProductStateOCTAModel(
        build_octa_graph(),
        ProductStateOCTAModelConfig(reference_excitation=0.1, initial_excitation=0.35),
    )
    state = model.initial_reference_state().copy()
    state[0] = 0.55

    source = model.modular_sources(state)

    assert float(np.linalg.norm(source)) > 1e-8


def test_product_state_ball_partial_trace_dimension_is_correct() -> None:
    """Ball marginals should have the expected Hilbert-space dimension."""
    graph = build_octa_graph()
    model = ProductStateOCTAModel(graph, ProductStateOCTAModelConfig())
    rho = product_density_matrix(model.initial_state())
    first_ball = model.balls[0]

    rho_ball = partial_trace_qubits(rho, keep=first_ball, num_qubits=graph.num_nodes)

    expected_dim = 2 ** len(first_ball)
    assert rho_ball.shape == (expected_dim, expected_dim)


def test_product_state_model_runs_with_delay_audits() -> None:
    """The product-state OCTA model should preserve the delay guarantees too."""
    model = ProductStateOCTAModel(
        build_octa_graph(),
        ProductStateOCTAModelConfig(
            geometry_register=GeometryRegisterConfig(max_configs=8, levels_per_node=3, seed_geometry_ensemble=9)
        ),
    )
    config = SimulationConfig(
        total_time=0.2,
        dt=0.05,
        delay=DelayConfig(delay_steps=2, kernel=KernelConfig(kind="delta")),
        reproducibility=ReproducibilityConfig(seed=13, store_every=1),
    )

    result = DelayAwareSimulator(model=model, config=config).run()

    assert audit_delay_definition(result).passed
    assert audit_single_advance_per_step(result).passed
