"""Integration tests for reviewer-facing audits."""

from __future__ import annotations

from egtrqc.config import DelayConfig, KernelConfig, ReproducibilityConfig, SimulationConfig
from egtrqc.diagnostics import audit_delay_definition, audit_single_advance_per_step
from egtrqc.model import LinearBackreactionModel, LinearBackreactionModelConfig
from egtrqc.simulator import DelayAwareSimulator


def test_reference_simulation_passes_delay_audit() -> None:
    """The reference simulator should satisfy reviewer concern C2."""
    config = SimulationConfig(
        total_time=0.2,
        dt=0.05,
        delay=DelayConfig(delay_steps=2, kernel=KernelConfig(kind="delta")),
        reproducibility=ReproducibilityConfig(seed=11, store_every=1),
    )
    model = LinearBackreactionModel(LinearBackreactionModelConfig())

    result = DelayAwareSimulator(model, config).run()
    report = audit_delay_definition(result)

    assert report.passed
    assert report.max_abs_error <= 1e-12


def test_reference_simulation_advances_buffer_once_per_step() -> None:
    """Diagnostics should not modify the physical delay history."""
    config = SimulationConfig(
        total_time=0.3,
        dt=0.05,
        delay=DelayConfig(delay_steps=1, kernel=KernelConfig(kind="delta")),
        reproducibility=ReproducibilityConfig(seed=5, store_every=1),
    )
    model = LinearBackreactionModel(LinearBackreactionModelConfig())

    result = DelayAwareSimulator(model, config).run()
    report = audit_single_advance_per_step(result)

    assert report.passed
    assert report.expected_advances == report.observed_advances == config.num_steps
