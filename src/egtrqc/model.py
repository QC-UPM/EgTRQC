"""Reference dynamical models for delayed semiclassical backreaction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LinearBackreactionModelConfig:
    """Configuration for the reference linearized backreaction model.

    Attributes:
        dimension: Dimension of the reduced matter-state vector.
        damping: Linear damping coefficient.
        feedback_gain: Coupling between effective source and state update.
        source_gain: Coupling between state and instantaneous source.
        curvature_gain: Mapping from effective source to curvature proxy.
        noise_scale: Optional deterministic-noise scale controlled by the RNG seed.
    """

    dimension: int = 3
    damping: float = 0.35
    feedback_gain: float = 0.45
    source_gain: float = 0.75
    curvature_gain: float = 1.2
    noise_scale: float = 0.0

    def __post_init__(self) -> None:
        """Validate the model configuration."""
        if self.dimension < 1:
            raise ValueError("dimension must be positive.")
        if self.damping < 0.0:
            raise ValueError("damping must be non-negative.")
        if self.noise_scale < 0.0:
            raise ValueError("noise_scale must be non-negative.")


class LinearBackreactionModel:
    """Small, extensible delayed-feedback model used as a clean PoC core.

    The model is intentionally lightweight. Its goal is to provide a reproducible
    and fully auditable reference implementation of delay semantics before the
    full OCTA-specific physics is reintroduced.
    """

    def __init__(self, config: LinearBackreactionModelConfig) -> None:
        """Store the model configuration and build deterministic operators.

        Args:
            config: Model parameters.
        """
        self._config = config
        self._a_matrix = -config.damping * np.eye(config.dimension, dtype=np.float64)
        self._b_matrix = config.feedback_gain * np.eye(config.dimension, dtype=np.float64)
        self._c_matrix = config.source_gain * np.eye(config.dimension, dtype=np.float64)
        self._curvature_matrix = config.curvature_gain * np.eye(config.dimension, dtype=np.float64)
        self._bias = np.linspace(0.15, 0.15 * config.dimension, config.dimension, dtype=np.float64)

    @property
    def config(self) -> LinearBackreactionModelConfig:
        """Return the model configuration."""
        return self._config

    def initial_state(self) -> FloatArray:
        """Return the deterministic initial matter state.

        Returns:
            Initial reduced state vector.
        """
        return np.linspace(1.0, 1.0 + 0.2 * (self._config.dimension - 1), self._config.dimension)

    def source(self, state: FloatArray) -> FloatArray:
        """Compute the instantaneous source.

        Args:
            state: Current matter-state vector.

        Returns:
            Instantaneous source `J(t)`.
        """
        return (self._c_matrix @ np.asarray(state, dtype=np.float64)) + self._bias

    def curvature(self, effective_source: FloatArray) -> FloatArray:
        """Compute a curvature proxy from the delayed source.

        Args:
            effective_source: Effective delayed source.

        Returns:
            Curvature proxy used in diagnostics and evolution.
        """
        return self._curvature_matrix @ np.asarray(effective_source, dtype=np.float64)

    def advance(self, state: FloatArray, effective_source: FloatArray, dt: float, rng: Generator) -> FloatArray:
        """Advance the reduced state by one explicit Euler step.

        Args:
            state: Current matter-state vector.
            effective_source: Effective source entering the physical update.
            dt: Time-step size.
            rng: Random generator used for deterministic perturbations.

        Returns:
            Updated matter-state vector.
        """
        drift = (self._a_matrix @ state) + (self._b_matrix @ effective_source)
        noise = np.zeros_like(drift)
        if self._config.noise_scale > 0.0:
            noise = self._config.noise_scale * rng.normal(size=drift.shape)
        return np.asarray(state, dtype=np.float64) + (dt * drift) + noise
