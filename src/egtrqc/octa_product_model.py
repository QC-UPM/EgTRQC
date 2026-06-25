"""Product-state OCTA model with notebook-style modular-source construction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from egtrqc.ensemble import GeometryEnsemble, gibbs_distribution, make_geometry_ensemble, metropolis_rate_matrix, slice_free_energy
from egtrqc.geometry import GeometryGraph, build_balls, build_response_operator, project_zero_mean
from egtrqc.octa_model import GeometryRegisterConfig
from egtrqc.quantum import ensure_density, matrix_log_psd, partial_trace_qubits, product_density_matrix

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class ProductStateOCTAModelConfig:
    """Configuration for the product-state OCTA model.

    Attributes:
        damping: Local damping on excitation probabilities.
        feedback_gain: Curvature-to-state feedback strength.
        source_bias: Constant source offset.
        alpha_area: Geometric prefactor in the response residual.
        mass_m: Mass-like parameter in the response operator.
        mesh_h: Mesh spacing in the response operator.
        ball_radius_hops: Radius of graph balls used for modular Hamiltonians.
        eps_modular: Regularization applied before matrix logs.
        reference_excitation: Reference local excitation probability.
        initial_excitation: Initial local excitation probability.
        enforce_zero_mean_if_massless: Whether to project the massless solve.
        geometry_register: Discrete geometry-register configuration.
        noise_scale: Optional deterministic perturbation scale.
    """

    damping: float = 0.15
    feedback_gain: float = 0.25
    source_bias: float = 0.0
    alpha_area: float = 1.0
    mass_m: float = 0.7
    mesh_h: float = 1.0
    ball_radius_hops: int = 1
    eps_modular: float = 2e-3
    reference_excitation: float = 0.1
    initial_excitation: float = 0.35
    enforce_zero_mean_if_massless: bool = True
    geometry_register: GeometryRegisterConfig = field(default_factory=GeometryRegisterConfig)
    noise_scale: float = 0.0

    def __post_init__(self) -> None:
        """Validate model settings."""
        if self.damping < 0.0:
            raise ValueError("damping must be non-negative.")
        if self.alpha_area <= 0.0:
            raise ValueError("alpha_area must be strictly positive.")
        if self.mesh_h <= 0.0:
            raise ValueError("mesh_h must be strictly positive.")
        if self.ball_radius_hops < 0:
            raise ValueError("ball_radius_hops must be non-negative.")
        if not (0.0 < self.eps_modular < 1.0):
            raise ValueError("eps_modular must lie in (0, 1).")
        if self.noise_scale < 0.0:
            raise ValueError("noise_scale must be non-negative.")


class ProductStateOCTAModel:
    """OCTA-ready model using ball marginals and modular Hamiltonians.

    The state is a vector of local excitation probabilities, but modular sources
    are computed from product-state density matrices and ball marginals in the
    same structural way as the notebook implementation.
    """

    def __init__(self, graph: GeometryGraph, config: ProductStateOCTAModelConfig) -> None:
        """Initialize the product-state OCTA model.

        Args:
            graph: Discrete geometry graph.
            config: Model configuration.
        """
        self._graph = graph
        self._config = config
        self._balls = build_balls(graph, config.ball_radius_hops)
        self._response = build_response_operator(graph, config.mass_m, config.mesh_h)
        self._edge_smoother = np.eye(graph.num_nodes, dtype=np.float64) + 0.15 * graph.adjacency
        self._reference_state = self.initial_reference_state()
        self._reference_density = product_density_matrix(self._reference_state)
        self._modular_hamiltonians = self._build_modular_hamiltonians(self._reference_density)
        self._reference_ball_expectations = self._ball_modular_expectations(self._reference_density)
        self._geometry_ensemble = make_geometry_ensemble(
            graph.curvature_reference,
            levels_per_node=config.geometry_register.levels_per_node,
            delta_scale=config.geometry_register.delta_scale,
            max_configs=config.geometry_register.max_configs,
            seed=config.geometry_register.seed_geometry_ensemble,
            enforce_total_curvature=config.geometry_register.enforce_total_curvature,
        )

    @property
    def graph(self) -> GeometryGraph:
        """Return the underlying geometry graph."""
        return self._graph

    @property
    def balls(self) -> list[list[int]]:
        """Return the modular balls used in the source construction."""
        return [list(ball) for ball in self._balls]

    @property
    def geometry_ensemble(self) -> GeometryEnsemble:
        """Return the discrete geometry ensemble used by the register layer."""
        return self._geometry_ensemble

    @property
    def volume_constraint_active(self) -> bool:
        """Return whether the massless zero-mean constraint is active."""
        return self._config.enforce_zero_mean_if_massless and abs(self._config.mass_m) <= 1e-12

    def initial_reference_state(self) -> FloatArray:
        """Return the reference local excitation probabilities."""
        return np.full(self._graph.num_nodes, float(self._config.reference_excitation), dtype=np.float64)

    def initial_state(self) -> FloatArray:
        """Return the initial local excitation probabilities."""
        return np.full(self._graph.num_nodes, float(self._config.initial_excitation), dtype=np.float64)

    def source(self, state: FloatArray) -> FloatArray:
        """Compute notebook-style modular sources from ball marginals."""
        rho = product_density_matrix(np.asarray(state, dtype=np.float64))
        expectations = self._ball_modular_expectations(rho)
        centered = expectations - self._reference_ball_expectations
        return centered + float(self._config.source_bias)

    def modular_sources(self, state: FloatArray) -> FloatArray:
        """Alias exposing the modular-source interpretation explicitly."""
        return self.source(state)

    def solve_response(self, rhs: FloatArray) -> FloatArray:
        """Solve the response equation `G k = rhs`."""
        rhs_arr = np.asarray(rhs, dtype=np.float64)
        if self.volume_constraint_active:
            rhs_arr = project_zero_mean(rhs_arr)
            solved = np.linalg.solve(self._response + (1e-12 * np.eye(self._graph.num_nodes)), rhs_arr)
            return project_zero_mean(solved)
        return np.linalg.solve(self._response, rhs_arr)

    def curvature(self, effective_source: FloatArray) -> FloatArray:
        """Compute curvature from the delayed effective source."""
        k_eq = self.solve_response((-1.0 / self._config.alpha_area) * np.asarray(effective_source, dtype=np.float64))
        return self._graph.curvature_reference + k_eq

    def advance(self, state: FloatArray, effective_source: FloatArray, dt: float, rng: Generator) -> FloatArray:
        """Advance the local excitation probabilities by one time step."""
        state_arr = np.asarray(state, dtype=np.float64)
        curvature = self.curvature(effective_source)
        smoothed_curvature = self._edge_smoother @ (curvature - self._graph.curvature_reference)
        drift = (-self._config.damping * state_arr) + (self._config.feedback_gain * smoothed_curvature)
        noise = np.zeros_like(drift)
        if self._config.noise_scale > 0.0:
            noise = self._config.noise_scale * rng.normal(size=drift.shape)
        updated = np.clip(state_arr + (dt * drift) + noise, 0.0, 1.0)
        return updated

    def free_energies(self, effective_source: FloatArray) -> FloatArray:
        """Compute free energies for all discrete geometry configurations."""
        source_arr = np.asarray(effective_source, dtype=np.float64)
        k_configs = self._geometry_ensemble.curvature_configurations - self._graph.curvature_reference[None, :]
        return np.array(
            [
                slice_free_energy(k_configs[idx], source_arr, self._response, self._config.alpha_area)
                for idx in range(k_configs.shape[0])
            ],
            dtype=np.float64,
        )

    def gibbs_distribution(self, effective_source: FloatArray) -> FloatArray:
        """Compute Gibbs weights over discrete geometry configurations."""
        return gibbs_distribution(self.free_energies(effective_source), self._config.geometry_register.beta_geom)

    def metropolis_rate_matrix(self, effective_source: FloatArray) -> FloatArray:
        """Compute the Metropolis rate matrix for the geometry register."""
        return metropolis_rate_matrix(
            self.free_energies(effective_source),
            self._geometry_ensemble.adjacency,
            self._config.geometry_register.beta_geom,
            self._config.geometry_register.gamma_geom_base,
        )

    def _build_modular_hamiltonians(self, rho_ref: ComplexArray) -> list[ComplexArray]:
        """Build one modular Hamiltonian per graph ball from the reference state."""
        modulars: list[ComplexArray] = []
        for ball in self._balls:
            rho_ball = partial_trace_qubits(rho_ref, keep=ball, num_qubits=self._graph.num_nodes)
            dim_ball = rho_ball.shape[0]
            regularized = ((1.0 - self._config.eps_modular) * rho_ball) + (
                self._config.eps_modular * (np.eye(dim_ball, dtype=np.complex128) / dim_ball)
            )
            modulars.append(-matrix_log_psd(regularized, eps=1e-12))
        return modulars

    def _ball_modular_expectations(self, rho: ComplexArray) -> FloatArray:
        """Compute modular expectations on each ball."""
        rho_density = ensure_density(rho)
        expectations = np.zeros(self._graph.num_nodes, dtype=np.float64)
        for index, ball in enumerate(self._balls):
            rho_ball = partial_trace_qubits(rho_density, keep=ball, num_qubits=self._graph.num_nodes)
            expectations[index] = float(np.trace(rho_ball @ self._modular_hamiltonians[index]).real)
        return expectations
