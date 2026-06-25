"""Reduced dense-state OCTA model for small-system controlled studies."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from egtrqc.ensemble import GeometryEnsemble, gibbs_distribution, make_geometry_ensemble, metropolis_rate_matrix, slice_free_energy
from egtrqc.geometry import GeometryGraph, build_balls, build_response_operator, project_zero_mean
from egtrqc.octa_model import GeometryRegisterConfig
from egtrqc.quantum import (
    ComplexArray,
    FloatArray,
    Z2,
    build_local_paulis,
    build_named_density,
    ensure_density,
    expm_hermitian,
    hermitize,
    matrix_log_psd,
    partial_trace_qubits,
)


@dataclass(frozen=True, slots=True)
class DenseReducedOCTAModelConfig:
    """Configuration for the reduced dense-state OCTA model.

    Attributes:
        alpha_area: Geometric prefactor in the response residual.
        mass_m: Mass-like parameter in the response operator.
        mesh_h: Mesh spacing in the response operator.
        ball_radius_hops: Radius of graph balls used for modular Hamiltonians.
        eps_modular: Regularization applied before matrix logs.
        reference_state: Named reference pure state.
        initial_state: Named initial pure state.
        omega_0: Base local precession frequency.
        alpha_curv_freq: Curvature dependence of the local Hamiltonian.
        gamma0: Baseline dephasing rate.
        gamma_curv: Curvature dependence of dephasing.
        transverse_hx: Optional transverse field.
        enforce_zero_mean_if_massless: Whether to project the massless solve.
        geometry_register: Discrete geometry-register configuration.
    """

    alpha_area: float = 1.0
    mass_m: float = 0.7
    mesh_h: float = 1.0
    ball_radius_hops: int = 1
    eps_modular: float = 2e-3
    reference_state: str = "zeros"
    initial_state: str = "plus_product"
    omega_0: float = 2.0 * math.pi * 40.0
    alpha_curv_freq: float = 0.2
    gamma0: float = 0.3
    gamma_curv: float = 1.2
    transverse_hx: float = 0.0
    enforce_zero_mean_if_massless: bool = True
    geometry_register: GeometryRegisterConfig = field(default_factory=GeometryRegisterConfig)

    def __post_init__(self) -> None:
        """Validate model settings."""
        if self.alpha_area <= 0.0:
            raise ValueError("alpha_area must be strictly positive.")
        if self.mesh_h <= 0.0:
            raise ValueError("mesh_h must be strictly positive.")
        if self.ball_radius_hops < 0:
            raise ValueError("ball_radius_hops must be non-negative.")
        if not (0.0 < self.eps_modular < 1.0):
            raise ValueError("eps_modular must lie in (0, 1).")
        if self.gamma0 < 0.0 or self.gamma_curv < 0.0:
            raise ValueError("Dephasing rates must be non-negative.")


class DenseReducedOCTAModel:
    """Small dense-state OCTA model with notebook-style modular sources.

    This model is intended for controlled small-system studies where a full
    density matrix is still tractable. It keeps the notebook structure for
    modular sources while using a compact CPTP evolution rule.
    """

    def __init__(self, graph: GeometryGraph, config: DenseReducedOCTAModelConfig) -> None:
        """Initialize the dense-state OCTA model."""
        self._graph = graph
        self._config = config
        self._num_qubits = graph.num_nodes
        self._dim = 2 ** self._num_qubits
        self._balls = build_balls(graph, config.ball_radius_hops)
        self._response = build_response_operator(graph, config.mass_m, config.mesh_h)
        self._x_ops, self._y_ops, self._z_ops = build_local_paulis(self._num_qubits)
        self._reference_density = build_named_density(self._num_qubits, config.reference_state)
        self._initial_density = build_named_density(self._num_qubits, config.initial_state)
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

    def initial_state(self) -> ComplexArray:
        """Return the initial dense matter state."""
        return self._initial_density.copy()

    def reference_state(self) -> ComplexArray:
        """Return the reference dense matter state."""
        return self._reference_density.copy()

    def source(self, state: ComplexArray) -> FloatArray:
        """Compute notebook-style modular sources from reduced ball marginals."""
        rho = ensure_density(state)
        expectations = self._ball_modular_expectations(rho)
        return expectations - self._reference_ball_expectations

    def modular_sources(self, state: ComplexArray) -> FloatArray:
        """Alias exposing the modular-source interpretation explicitly."""
        return self.source(state)

    def solve_response(self, rhs: FloatArray) -> FloatArray:
        """Solve the response equation `G k = rhs`."""
        rhs_arr = np.asarray(rhs, dtype=np.float64)
        if self.volume_constraint_active:
            rhs_arr = project_zero_mean(rhs_arr)
            solved = np.linalg.solve(self._response + (1e-12 * np.eye(self._num_qubits)), rhs_arr)
            return project_zero_mean(solved)
        return np.linalg.solve(self._response, rhs_arr)

    def curvature(self, effective_source: FloatArray) -> FloatArray:
        """Compute curvature from the delayed effective source."""
        k_eq = self.solve_response((-1.0 / self._config.alpha_area) * np.asarray(effective_source, dtype=np.float64))
        return self._graph.curvature_reference + k_eq

    def advance(self, state: ComplexArray, effective_source: FloatArray, dt: float, rng: Generator) -> ComplexArray:
        """Advance the dense state by one CPTP step."""
        del rng
        rho = ensure_density(state)
        curvature = self.curvature(effective_source)
        hamiltonian = self._hamiltonian_from_curvature(curvature)
        unitary = expm_hermitian(hamiltonian, 0.5 * float(dt), factor=-1j)
        rho_next = unitary @ rho @ np.conjugate(unitary.T)
        for qubit in range(self._num_qubits):
            gamma = self._gamma_i(qubit, float(curvature[qubit]))
            probability = 0.5 * (1.0 - math.exp(-2.0 * gamma * float(dt)))
            if probability != 0.0:
                z_op = self._z_ops[qubit]
                rho_next = ((1.0 - probability) * rho_next) + (probability * (z_op @ rho_next @ z_op))
        rho_next = unitary @ rho_next @ np.conjugate(unitary.T)
        return ensure_density(hermitize(rho_next))

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
        modulars: list[ComplexArray] = []
        for ball in self._balls:
            rho_ball = partial_trace_qubits(rho_ref, keep=ball, num_qubits=self._num_qubits)
            dim_ball = rho_ball.shape[0]
            regularized = ((1.0 - self._config.eps_modular) * rho_ball) + (
                self._config.eps_modular * (np.eye(dim_ball, dtype=np.complex128) / dim_ball)
            )
            modulars.append(-matrix_log_psd(regularized, eps=1e-12))
        return modulars

    def _ball_modular_expectations(self, rho: ComplexArray) -> FloatArray:
        expectations = np.zeros(self._num_qubits, dtype=np.float64)
        for index, ball in enumerate(self._balls):
            rho_ball = partial_trace_qubits(rho, keep=ball, num_qubits=self._num_qubits)
            expectations[index] = float(np.trace(rho_ball @ self._modular_hamiltonians[index]).real)
        return expectations

    def _omega_i(self, qubit: int, curvature_value: float) -> float:
        return self._config.omega_0 * (1.0 + self._config.alpha_curv_freq * (curvature_value - float(self._graph.curvature_reference[qubit])))

    def _gamma_i(self, qubit: int, curvature_value: float) -> float:
        del qubit
        return self._config.gamma0 + self._config.gamma_curv * abs(curvature_value)

    def _hamiltonian_from_curvature(self, curvature: FloatArray) -> ComplexArray:
        hamiltonian = np.zeros((self._dim, self._dim), dtype=np.complex128)
        for qubit in range(self._num_qubits):
            hamiltonian += 0.5 * self._omega_i(qubit, float(curvature[qubit])) * self._z_ops[qubit]
            if self._config.transverse_hx != 0.0:
                hamiltonian += 0.5 * self._config.transverse_hx * self._x_ops[qubit]
        return hermitize(hamiltonian)
