"""Quantum helper routines for modular-source calculations."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

I2 = np.eye(2, dtype=np.complex128)
X2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y2 = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
Z2 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def dagger(operator: ComplexArray) -> ComplexArray:
    """Return the conjugate transpose of an operator."""
    return np.conjugate(np.asarray(operator, dtype=np.complex128).T)


def hermitize(operator: ComplexArray) -> ComplexArray:
    """Return the Hermitian part of an operator."""
    op = np.asarray(operator, dtype=np.complex128)
    return 0.5 * (op + dagger(op))


def ensure_density(rho: ComplexArray, eps: float = 1e-15) -> ComplexArray:
    """Normalize a density matrix and validate its trace."""
    rho_h = hermitize(rho)
    trace = float(np.trace(rho_h).real)
    if trace <= eps:
        raise ValueError("Density matrix has nonpositive trace.")
    return rho_h / trace


def matrix_log_psd(rho: ComplexArray, eps: float = 1e-12) -> ComplexArray:
    """Compute the matrix logarithm of a PSD operator."""
    rho_h = hermitize(rho)
    evals, evecs = np.linalg.eigh(rho_h)
    evals = np.clip(evals.real, eps, None)
    return (evecs * np.log(evals)) @ dagger(evecs)


def tensor_product(operators: list[ComplexArray]) -> ComplexArray:
    """Compute the Kronecker product of a list of operators."""
    out = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for operator in operators:
        out = np.kron(out, np.asarray(operator, dtype=np.complex128))
    return out


def tensor_kets(kets: list[ComplexArray]) -> ComplexArray:
    """Compute the Kronecker product of a list of state vectors."""
    out = np.array([1.0 + 0.0j], dtype=np.complex128)
    for ket in kets:
        out = np.kron(out, np.asarray(ket, dtype=np.complex128))
    return out


def density_from_statevector(statevector: ComplexArray) -> ComplexArray:
    """Return the pure-state density matrix associated with a ket."""
    psi = np.asarray(statevector, dtype=np.complex128)
    return np.outer(psi, np.conjugate(psi))


def build_named_density(num_qubits: int, state_name: str) -> ComplexArray:
    """Build a small named multi-qubit pure state density matrix."""
    ket0 = np.array([1.0, 0.0], dtype=np.complex128)
    ket1 = np.array([0.0, 1.0], dtype=np.complex128)
    ketp = (ket0 + ket1) / math.sqrt(2.0)
    normalized_name = str(state_name).strip().lower()
    if normalized_name == "zeros":
        psi = tensor_kets([ket0] * int(num_qubits))
    elif normalized_name == "plus_product":
        psi = tensor_kets([ketp] * int(num_qubits))
    elif normalized_name == "ghz":
        psi0 = tensor_kets([ket0] * int(num_qubits))
        psi1 = tensor_kets([ket1] * int(num_qubits))
        psi = (psi0 + psi1) / math.sqrt(2.0)
    else:
        raise ValueError("Unknown named state. Supported states are zeros, plus_product, and ghz.")
    return density_from_statevector(psi)


def build_local_paulis(num_qubits: int) -> tuple[list[ComplexArray], list[ComplexArray], list[ComplexArray]]:
    """Build embedded single-qubit Pauli operators for each qubit."""
    x_ops: list[ComplexArray] = []
    y_ops: list[ComplexArray] = []
    z_ops: list[ComplexArray] = []
    for target in range(int(num_qubits)):
        op_x = np.array([[1.0 + 0.0j]], dtype=np.complex128)
        op_y = np.array([[1.0 + 0.0j]], dtype=np.complex128)
        op_z = np.array([[1.0 + 0.0j]], dtype=np.complex128)
        for qubit in range(int(num_qubits)):
            op_x = np.kron(op_x, X2 if qubit == target else I2)
            op_y = np.kron(op_y, Y2 if qubit == target else I2)
            op_z = np.kron(op_z, Z2 if qubit == target else I2)
        x_ops.append(op_x)
        y_ops.append(op_y)
        z_ops.append(op_z)
    return x_ops, y_ops, z_ops


def expm_hermitian(operator: ComplexArray, time: float, factor: complex = -1j) -> ComplexArray:
    """Exponentiate a Hermitian operator by eigendecomposition."""
    hermitian = hermitize(operator)
    evals, evecs = np.linalg.eigh(hermitian)
    phases = np.exp(factor * float(time) * evals.real)
    return (evecs * phases) @ dagger(evecs)


def local_density_from_excitation(excitation: float) -> ComplexArray:
    """Build a single-qubit diagonal density matrix from an excitation probability."""
    probability = float(np.clip(excitation, 0.0, 1.0))
    return np.array([[1.0 - probability, 0.0], [0.0, probability]], dtype=np.complex128)


def product_density_matrix(excitations: FloatArray) -> ComplexArray:
    """Build a product-state density matrix from local excitation probabilities."""
    local_states = [local_density_from_excitation(value) for value in np.asarray(excitations, dtype=np.float64)]
    return tensor_product(local_states)


def partial_trace_qubits(rho: ComplexArray, keep: list[int], num_qubits: int) -> ComplexArray:
    """Trace out all qubits except the ones listed in `keep`."""
    rho_arr = np.asarray(rho, dtype=np.complex128)
    keep_sorted = sorted(int(index) for index in keep)
    if any(index < 0 or index >= num_qubits for index in keep_sorted):
        raise ValueError("partial_trace_qubits received an invalid qubit index.")
    trace_out = [index for index in range(num_qubits) if index not in keep_sorted]
    if not trace_out:
        return rho_arr.copy()
    dims = [2] * int(num_qubits)
    tensor = rho_arr.reshape(dims + dims)
    current_n = int(num_qubits)
    for qubit in sorted(trace_out, reverse=True):
        tensor = np.trace(tensor, axis1=qubit, axis2=qubit + current_n)
        current_n -= 1
    dim_keep = 2 ** len(keep_sorted)
    return tensor.reshape(dim_keep, dim_keep)
