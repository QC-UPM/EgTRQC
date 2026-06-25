"""Unit tests for delay-buffer semantics."""

from __future__ import annotations

import numpy as np

from egtrqc.delay import DelayBuffer


def test_delay_buffer_uses_initial_source_as_prehistory() -> None:
    """Early-time delayed values should reuse `J(t0)` rather than zeros."""
    initial = np.array([3.5], dtype=np.float64)
    buffer = DelayBuffer(initial_value=initial, max_history=8)

    effective_step_0 = buffer.effective(np.array([3.5]), delay_steps=2)
    buffer.advance(np.array([3.5]))

    effective_step_1 = buffer.effective(np.array([7.0]), delay_steps=2)

    np.testing.assert_allclose(effective_step_0, initial)
    np.testing.assert_allclose(effective_step_1, initial)


def test_delay_buffer_update_order_matches_mathematical_definition() -> None:
    """Stored history should reproduce `J(t_{n-d})` exactly."""
    buffer = DelayBuffer(initial_value=np.array([5.0]), max_history=8)
    current_sources = [5.0, 7.0, 11.0, 13.0]

    outputs: list[float] = []
    for source in current_sources:
        outputs.append(float(buffer.effective(np.array([source]), delay_steps=2)[0]))
        buffer.advance(np.array([source]))

    assert outputs == [5.0, 5.0, 5.0, 7.0]
