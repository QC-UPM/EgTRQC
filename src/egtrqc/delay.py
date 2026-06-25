"""Delay-buffer primitives with explicit physical-history semantics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BufferSnapshot:
    """Immutable snapshot of the physical source history.

    Attributes:
        initial_value: Prehistory value representing `J(t0)` for early times.
        history: Stored physical source samples in chronological order.
        advance_count: Number of physical advances applied to the buffer.
    """

    initial_value: FloatArray
    history: tuple[FloatArray, ...]
    advance_count: int


class DelayBuffer:
    """Delay line that separates pure observation from physical advancement.

    The mathematical contract is:

    - diagnostics may inspect the effective source without mutating history,
    - the physical history advances exactly once per physical time step,
    - early-time prehistory is initialized with `J(t0)`, not zeros.
    """

    def __init__(self, initial_value: FloatArray, max_history: int) -> None:
        """Initialize the delay buffer.

        Args:
            initial_value: Initial source value `J(t0)` used as the prehistory.
            max_history: Maximum number of physical source samples to retain.
        """
        if max_history < 1:
            raise ValueError("max_history must be positive.")
        self._initial_value = np.asarray(initial_value, dtype=np.float64).copy()
        self._history: Deque[FloatArray] = deque(maxlen=max_history)
        self._advance_count = 0

    @property
    def initial_value(self) -> FloatArray:
        """Return the configured prehistory value."""
        return self._initial_value.copy()

    @property
    def advance_count(self) -> int:
        """Return the number of physical advances already applied."""
        return self._advance_count

    def snapshot(self) -> BufferSnapshot:
        """Return an immutable view of the current history.

        Returns:
            Snapshot containing the prehistory value, chronological samples,
            and the physical advance counter.
        """
        return BufferSnapshot(
            initial_value=self.initial_value,
            history=tuple(sample.copy() for sample in self._history),
            advance_count=self._advance_count,
        )

    def advance(self, sample: FloatArray) -> None:
        """Append one physical source sample to the history.

        Args:
            sample: Source value associated with the current physical time step.
        """
        arr = np.asarray(sample, dtype=np.float64).copy()
        self._history.append(arr)
        self._advance_count += 1

    def effective(
        self,
        current_value: FloatArray,
        *,
        delay_steps: int,
        kernel_kind: str = "delta",
        dt: float = 1.0,
        tau: float | None = None,
    ) -> FloatArray:
        """Evaluate the effective source without mutating history.

        Args:
            current_value: Source sample evaluated from the current physical state.
            delay_steps: Pure-delay lag in discrete time steps.
            kernel_kind: Memory kernel family, either `"delta"` or `"exp"`.
            dt: Discrete time step used by the exponential kernel.
            tau: Memory time constant used by the exponential kernel.

        Returns:
            Effective source entering the physical update at the current time.
        """
        current = np.asarray(current_value, dtype=np.float64).copy()
        if delay_steps < 0:
            raise ValueError("delay_steps must be non-negative.")
        if kernel_kind == "delta":
            return self._effective_delta(current_value=current, delay_steps=delay_steps)
        if kernel_kind == "exp":
            return self._effective_exp(current_value=current, dt=dt, tau=tau)
        raise ValueError("kernel_kind must be either 'delta' or 'exp'.")

    def _effective_delta(self, current_value: FloatArray, delay_steps: int) -> FloatArray:
        """Evaluate the pure-delay effective source.

        Args:
            current_value: Current source `J(t_n)`.
            delay_steps: Discrete delay `d`.

        Returns:
            `J(t_n)` for `d=0`, `J(t_{n-d})` for late times, or `J(t_0)` during
            the early-time prehistory interval.
        """
        if delay_steps == 0:
            return np.asarray(current_value, dtype=np.float64).copy()
        if len(self._history) >= delay_steps:
            return self._history[-delay_steps].copy()
        return self.initial_value

    def _effective_exp(
        self,
        current_value: FloatArray,
        *,
        dt: float,
        tau: float | None,
    ) -> FloatArray:
        """Evaluate an exponential-memory effective source.

        Args:
            current_value: Current source `J(t_n)`.
            dt: Discrete time step.
            tau: Exponential memory time constant.

        Returns:
            Weighted discrete exponential average including the current value.
        """
        if tau is None:
            raise ValueError("Exponential memory requires tau.")
        if dt <= 0.0:
            raise ValueError("dt must be strictly positive.")

        series = [np.asarray(current_value, dtype=np.float64).copy()]
        series.extend(sample.copy() for sample in reversed(self._history))
        k = np.arange(len(series), dtype=np.float64)
        weights = np.exp(-(k * dt) / tau)
        weights /= np.sum(weights)

        out = np.zeros_like(series[0])
        for weight, sample in zip(weights, series, strict=True):
            out += float(weight) * sample
        return out
