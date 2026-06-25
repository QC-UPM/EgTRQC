"""Tests for the reduced dense-state OCTA model."""

from __future__ import annotations

import numpy as np

from egtrqc.config import DelayConfig, KernelConfig, ReproducibilityConfig, SimulationConfig
from egtrqc.diagnostics import audit_delay_definition, audit_single_advance_per_step
from egtrqc.geometry import build_octa_graph
from egtrqc.octa_dense_model import DenseReducedOCTAModel, DenseReducedOCTAModelConfig
from egtrqc.simulator import DelayAwareSimulator


def test_dense_model_modular_source_vanishes_at_reference_state() -> None:
    """Exact modular sources should vanish on the reference dense state."""
    model = DenseReducedOCTAModel(
        build_octa_graph(),
        DenseReducedOCTAModelConfig(reference_state="zeros", initial_state="plus_product"),
    )

    source = model.modular_sources(model.reference_state())

    np.testing.assert_allclose(source, np.zeros_like(source), atol=1e-12)


def test_dense_model_evolution_preserves_density_properties() -> None:
    """One dense-state step should preserve trace and Hermiticity."""
    model = DenseReducedOCTAModel(build_octa_graph(), DenseReducedOCTAModelConfig())
    state = model.initial_state()
    source = model.modular_sources(state)

    updated = model.advance(state, source, 0.01, np.random.default_rng(0))

    assert abs(float(np.trace(updated).real) - 1.0) <= 1e-12
    np.testing.assert_allclose(updated, np.conjugate(updated.T), atol=1e-12)


def test_dense_model_runs_with_delay_audits() -> None:
    """The reduced dense-state model should preserve the verified delay guarantees."""
    model = DenseReducedOCTAModel(build_octa_graph(), DenseReducedOCTAModelConfig())
    config = SimulationConfig(
        total_time=0.1,
        dt=0.05,
        delay=DelayConfig(delay_steps=2, kernel=KernelConfig(kind="delta")),
        reproducibility=ReproducibilityConfig(seed=17, store_every=1),
    )

    result = DelayAwareSimulator(model=model, config=config).run()

    assert audit_delay_definition(result).passed
    assert audit_single_advance_per_step(result).passed
    assert result.records[0].state.shape == (64, 64)
