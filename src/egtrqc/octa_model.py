"""Graph-based backreaction models closer to the OCTA notebooks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from egtrqc.ensemble import GeometryEnsemble, gibbs_distribution, make_geometry_ensemble, metropolis_rate_matrix, slice_free_energy
from egtrqc.geometry import GeometryGraph, build_balls, build_response_operator, project_zero_mean

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GeometryRegisterConfig:
    """Configuration for the discrete geometry register ensemble.

    Attributes:
        levels_per_node: Number of curvature levels retained per graph node.
        delta_scale: Curvature spacing between adjacent levels.
        max_configs: Number of retained configurations.
        seed_geometry_ensemble: Deterministic seed for ensemble generation.
        enforce_total_curvature: Whether to keep total curvature fixed across configs.
        beta_geom: Inverse temperature for the Gibbs distribution.
        gamma_geom_base: Base Metropolis jump-rate scale.
    """

    levels_per_node: int = 3
    delta_scale: float = 0.15
    max_configs: int = 16
    seed_geometry_ensemble: int = 7
    enforce_total_curvature: bool = True
    beta_geom: float = 8.0
    gamma_geom_base: float = 40.0

    def __post_init__(self) -> None:
        """Validate geometry-register settings."""
        if self.levels_per_node < 1:
            raise ValueError("levels_per_node must be positive.")
        if self.max_configs < 1:
            raise ValueError("max_configs must be positive.")
        if self.beta_geom < 0.0:
            raise ValueError("beta_geom must be non-negative.")
        if self.gamma_geom_base < 0.0:
            raise ValueError("gamma_geom_base must be non-negative.")


@dataclass(frozen=True, slots=True)
class GraphBackreactionModelConfig:
    """Configuration for the graph-based OCTA-ready reference model.

    Attributes:
        damping: Node-local damping.
        feedback_gain: Coupling between curvature and matter update.
        source_gain: Coupling between modular-source amplitudes and source.
        edge_curvature_strength: Edge-smoothed curvature modulation weight.
        source_bias: Constant source offset.
        noise_scale: Deterministic noise scale controlled by the run seed.
        alpha_area: Geometric-response prefactor from the notebook residual.
        mass_m: Mass-like parameter in the response operator.
        mesh_h: Mesh spacing in the response operator.
        ball_radius_hops: Graph-ball radius used for modular sources.
        enforce_zero_mean_if_massless: Enforce zero-mean curvature perturbations when `mass_m ~= 0`.
        geometry_register: Discrete geometry-register configuration used for thermodynamic ensemble calculations.
    """

    damping: float = 0.2
    feedback_gain: float = 0.35
    source_gain: float = 0.8
    edge_curvature_strength: float = 0.15
    source_bias: float = 0.1
    noise_scale: float = 0.0
    alpha_area: float = 1.0
    mass_m: float = 0.7
    mesh_h: float = 1.0
    ball_radius_hops: int = 1
    enforce_zero_mean_if_massless: bool = True
    geometry_register: GeometryRegisterConfig = field(default_factory=GeometryRegisterConfig)

    def __post_init__(self) -> None:
        """Validate the model configuration."""
        if self.damping < 0.0:
            raise ValueError("damping must be non-negative.")
        if self.noise_scale < 0.0:
            raise ValueError("noise_scale must be non-negative.")
        if self.alpha_area <= 0.0:
            raise ValueError("alpha_area must be strictly positive.")
        if self.mesh_h <= 0.0:
            raise ValueError("mesh_h must be strictly positive.")
        if self.ball_radius_hops < 0:
            raise ValueError("ball_radius_hops must be non-negative.")


class GraphBackreactionModel:
    """Graph-based delayed backreaction model for OCTA migration.

    This model mirrors the notebook flow more closely than the initial linear toy
    model: graph balls define per-node modular sources, the response operator is
    the notebook-style `G = (1/h^2)L + m^2 I`, and curvature is obtained from the
    equilibrium solve `k_eq = -(1/alpha) G^-1 J`.
    """

    def __init__(self, graph: GeometryGraph, config: GraphBackreactionModelConfig) -> None:
        """Initialize the graph-based model.

        Args:
            graph: Discrete geometry graph.
            config: Model parameters.
        """
        self._graph = graph
        self._config = config
        self._balls = build_balls(graph, config.ball_radius_hops)
        self._response = build_response_operator(graph, config.mass_m, config.mesh_h)
        self._edge_smoother = np.eye(graph.num_nodes, dtype=np.float64) + config.edge_curvature_strength * graph.adjacency
        self._reference_state = self.initial_state()
        self._reference_ball_means = self._ball_means(self._reference_state)
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
    def config(self) -> GraphBackreactionModelConfig:
        """Return the model configuration."""
        return self._config

    @property
    def balls(self) -> list[list[int]]:
        """Return the modular neighborhoods used to build the source."""
        return [list(ball) for ball in self._balls]

    @property
    def volume_constraint_active(self) -> bool:
        """Return whether zero-mean curvature projection is active."""
        return self._config.enforce_zero_mean_if_massless and abs(self._config.mass_m) <= 1e-12

    @property
    def geometry_ensemble(self) -> GeometryEnsemble:
        """Return the discrete geometry ensemble used by the register layer."""
        return self._geometry_ensemble

    def initial_state(self) -> FloatArray:
        """Return the deterministic initial node state.

        Returns:
            Initial node-amplitude vector.
        """
        return np.linspace(1.0, 1.0 + 0.1 * (self._graph.num_nodes - 1), self._graph.num_nodes)

    def source(self, state: FloatArray) -> FloatArray:
        """Compute the notebook-shaped modular source proxy.

        Args:
            state: Current node-state amplitudes.

        Returns:
            Node-wise source vector.
        """
        ball_means = self._ball_means(state)
        centered = ball_means - self._reference_ball_means
        return (self._config.source_gain * centered) + self._config.source_bias

    def modular_sources(self, state: FloatArray) -> FloatArray:
        """Alias exposing the modular-source interpretation explicitly.

        Args:
            state: Current node-state amplitudes.

        Returns:
            Node-wise modular source vector.
        """
        return self.source(state)

    def solve_response(self, rhs: FloatArray) -> FloatArray:
        """Solve the notebook-style response equation.

        Args:
            rhs: Right-hand side of the response system.

        Returns:
            Curvature perturbation vector.
        """
        rhs_arr = np.asarray(rhs, dtype=np.float64)
        if self.volume_constraint_active:
            rhs_arr = project_zero_mean(rhs_arr)
            solved = np.linalg.solve(self._response + (1e-12 * np.eye(self._graph.num_nodes)), rhs_arr)
            return project_zero_mean(solved)
        return np.linalg.solve(self._response, rhs_arr)

    def curvature(self, effective_source: FloatArray) -> FloatArray:
        """Compute the graph curvature proxy.

        Args:
            effective_source: Delayed effective source.

        Returns:
            Curvature vector around the reference geometry.
        """
        effective_source_arr = np.asarray(effective_source, dtype=np.float64)
        k_eq = self.solve_response((-1.0 / self._config.alpha_area) * effective_source_arr)
        return self._graph.curvature_reference + k_eq

    def advance(self, state: FloatArray, effective_source: FloatArray, dt: float, rng: Generator) -> FloatArray:
        """Advance the graph state by one time step.

        Args:
            state: Current node-state amplitudes.
            effective_source: Delayed or memory-averaged source.
            dt: Time-step size.
            rng: Random generator used for deterministic perturbations.

        Returns:
            Updated node-state amplitudes.
        """
        state_arr = np.asarray(state, dtype=np.float64)
        curvature = self.curvature(effective_source)
        smoothed_curvature = self._edge_smoother @ (curvature - self._graph.curvature_reference)
        drift = (-self._config.damping * state_arr) + (self._config.feedback_gain * smoothed_curvature)
        noise = np.zeros_like(drift)
        if self._config.noise_scale > 0.0:
            noise = self._config.noise_scale * rng.normal(size=drift.shape)
        return state_arr + (dt * drift) + noise



    def free_energies(self, effective_source: FloatArray) -> FloatArray:
        """Compute free energies for all discrete geometry configurations.

        Args:
            effective_source: Delayed effective source.

        Returns:
            Free-energy value for each retained configuration.
        """
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
        """Compute the geometry Gibbs distribution for the effective source.

        Args:
            effective_source: Delayed effective source.

        Returns:
            Gibbs weights over retained configurations.
        """
        return gibbs_distribution(self.free_energies(effective_source), self._config.geometry_register.beta_geom)

    def metropolis_rate_matrix(self, effective_source: FloatArray) -> FloatArray:
        """Compute the geometry Metropolis rate matrix for the effective source.

        Args:
            effective_source: Delayed effective source.

        Returns:
            Directed configuration-jump rate matrix.
        """
        return metropolis_rate_matrix(
            self.free_energies(effective_source),
            self._geometry_ensemble.adjacency,
            self._config.geometry_register.beta_geom,
            self._config.geometry_register.gamma_geom_base,
        )

    def _ball_means(self, state: FloatArray) -> FloatArray:
        """Compute one mean value per graph ball.

        Args:
            state: Current node-state amplitudes.

        Returns:
            Ball-mean vector aligned with graph vertices.
        """
        state_arr = np.asarray(state, dtype=np.float64)
        means = np.zeros(self._graph.num_nodes, dtype=np.float64)
        for index, ball in enumerate(self._balls):
            means[index] = float(np.mean(state_arr[ball]))
        return means
