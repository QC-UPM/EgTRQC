"""Delay-aware simulation engine with reproducibility metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from egtrqc.config import SimulationConfig
from egtrqc.delay import DelayBuffer
from egtrqc.model import LinearBackreactionModel

FloatArray = NDArray[np.float64]
StateArray = NDArray[np.float64] | NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class StepRecord:
    """Immutable record for one stored simulation time.

    Attributes:
        step_index: Physical time-step index.
        time: Physical time.
        state: Reduced matter state, possibly vector- or matrix-valued.
        source: Instantaneous source `J(t_n)`.
        effective_source: Delayed or memory-averaged source used in the update.
        curvature: Curvature proxy derived from the effective source.
        history_length: Number of physical source samples stored before advancing.
        buffer_advance_count: Physical advance counter at storage time.
    """

    step_index: int
    time: float
    state: StateArray
    source: FloatArray
    effective_source: FloatArray
    curvature: FloatArray
    history_length: int
    buffer_advance_count: int


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Simulation output and reproducibility metadata.

    Attributes:
        config: Effective simulation configuration.
        model_name: Human-readable model identifier.
        records: Stored records according to the snapshot cadence.
        physical_log: Full per-step records used for reviewer audits.
        buffer_initial_source: Initial source value used for prehistory.
        total_buffer_advances: Number of physical advances applied to the delay line.
    """

    config: SimulationConfig
    model_name: str
    records: tuple[StepRecord, ...]
    physical_log: tuple[StepRecord, ...]
    buffer_initial_source: FloatArray
    total_buffer_advances: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the result into a JSON-serializable dictionary."""

        def convert(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                if np.iscomplexobj(value):
                    return {"real": value.real.tolist(), "imag": value.imag.tolist()}
                return value.tolist()
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(val) for key, val in value.items()}
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, list):
                return [convert(item) for item in value]
            if hasattr(value, "__dataclass_fields__"):
                return {key: convert(val) for key, val in asdict(value).items()}
            return value

        return {
            "config": convert(self.config),
            "model_name": self.model_name,
            "records": convert(self.records),
            "physical_log": convert(self.physical_log),
            "buffer_initial_source": convert(self.buffer_initial_source),
            "total_buffer_advances": self.total_buffer_advances,
        }


class DelayAwareSimulator:
    """Reference simulator that preserves delay semantics by construction."""

    def __init__(self, model: LinearBackreactionModel, config: SimulationConfig) -> None:
        """Store the model and the configuration.

        Args:
            model: Dynamical model used for state evolution.
            config: Global simulation configuration.
        """
        self._model = model
        self._config = config

    def run(self) -> SimulationResult:
        """Run the simulation.

        Returns:
            Structured simulation result with audit-friendly physical logs.
        """
        state = self._model.initial_state()
        current_source = self._model.source(state)
        max_history = self._config.delay.max_history or max(self._config.delay.delay_steps + 2, 8)
        buffer = DelayBuffer(initial_value=current_source, max_history=max_history)
        rng = default_rng(self._config.reproducibility.seed)

        stored_records: list[StepRecord] = []
        physical_log: list[StepRecord] = []

        for step_index in range(self._config.num_steps + 1):
            current_source = self._model.source(state)
            effective_source = buffer.effective(
                current_source,
                delay_steps=self._config.delay.delay_steps,
                kernel_kind=self._config.delay.kernel.kind,
                dt=self._config.dt,
                tau=self._config.delay.kernel.tau,
            )
            curvature = self._model.curvature(effective_source)
            snapshot = buffer.snapshot()

            record = StepRecord(
                step_index=step_index,
                time=step_index * self._config.dt,
                state=np.array(state, copy=True),
                source=np.asarray(current_source, dtype=np.float64).copy(),
                effective_source=np.asarray(effective_source, dtype=np.float64).copy(),
                curvature=np.asarray(curvature, dtype=np.float64).copy(),
                history_length=len(snapshot.history),
                buffer_advance_count=snapshot.advance_count,
            )
            physical_log.append(record)

            if self._should_store(step_index):
                stored_records.append(record)

            if step_index == self._config.num_steps:
                break

            next_state = self._model.advance(state, effective_source, self._config.dt, rng)
            buffer.advance(current_source)
            state = next_state

        return SimulationResult(
            config=self._config,
            model_name=type(self._model).__name__,
            records=tuple(stored_records),
            physical_log=tuple(physical_log),
            buffer_initial_source=buffer.initial_value,
            total_buffer_advances=buffer.advance_count,
        )

    def _should_store(self, step_index: int) -> bool:
        """Return whether a given step should be persisted."""
        cadence = self._config.reproducibility.store_every
        return step_index == 0 or step_index == self._config.num_steps or step_index % cadence == 0
