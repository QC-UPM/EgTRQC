"""Audit routines for reviewer-facing delay semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from egtrqc.simulator import SimulationResult


@dataclass(frozen=True, slots=True)
class DelayAuditReport:
    """Report for reviewer concern C2.

    Attributes:
        passed: Whether the physical log matches the mathematical delay rule.
        max_abs_error: Maximum absolute mismatch found in the audit.
        message: Human-readable summary.
    """

    passed: bool
    max_abs_error: float
    message: str


@dataclass(frozen=True, slots=True)
class SingleAdvanceAuditReport:
    """Report for reviewer concern C3.

    Attributes:
        passed: Whether the delay line advanced exactly once per physical step.
        expected_advances: Expected number of physical advances.
        observed_advances: Observed number of physical advances.
        message: Human-readable summary.
    """

    passed: bool
    expected_advances: int
    observed_advances: int
    message: str


def audit_delay_definition(result: SimulationResult) -> DelayAuditReport:
    """Verify the pure mathematical delay rule against the physical log.

    For the delta kernel the expected rule is:

    - `J_eff(t_n) = J(t_n)` for `delay_steps = 0`,
    - `J_eff(t_n) = J(t_{n-d})` for `n >= d`,
    - `J_eff(t_n) = J(t_0)` for `n < d`.

    Args:
        result: Simulation output to audit.

    Returns:
        Delay audit report.
    """
    if result.config.delay.kernel.kind != "delta":
        return DelayAuditReport(
            passed=True,
            max_abs_error=0.0,
            message="Delay-definition audit skipped because the kernel is not delta.",
        )

    delay_steps = result.config.delay.delay_steps
    initial_source = result.buffer_initial_source
    sources = [record.source for record in result.physical_log]

    max_abs_error = 0.0
    for index, record in enumerate(result.physical_log):
        if delay_steps == 0:
            expected = record.source
        elif index < delay_steps:
            expected = initial_source
        else:
            expected = sources[index - delay_steps]
        err = float(np.max(np.abs(record.effective_source - expected)))
        max_abs_error = max(max_abs_error, err)

    passed = max_abs_error <= 1e-12
    message = (
        "Effective source matches the mathematical pure-delay rule."
        if passed
        else "Effective source deviates from the mathematical pure-delay rule."
    )
    return DelayAuditReport(passed=passed, max_abs_error=max_abs_error, message=message)


def audit_single_advance_per_step(result: SimulationResult) -> SingleAdvanceAuditReport:
    """Verify that diagnostics did not advance the physical feedback history.

    Args:
        result: Simulation output to audit.

    Returns:
        Report checking that the delay buffer advanced once per physical step.
    """
    expected_advances = result.config.num_steps
    observed_advances = result.total_buffer_advances
    passed = expected_advances == observed_advances
    message = (
        "Delay history advanced exactly once per physical time step."
        if passed
        else "Delay history was advanced an unexpected number of times."
    )
    return SingleAdvanceAuditReport(
        passed=passed,
        expected_advances=expected_advances,
        observed_advances=observed_advances,
        message=message,
    )
