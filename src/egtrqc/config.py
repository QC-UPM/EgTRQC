"""Configuration models for the EgTRQC reference simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class KernelConfig:
    """Configuration for delayed-source post-processing.

    Attributes:
        kind: Memory kernel family. Supported values are `"delta"` and `"exp"`.
        tau: Time constant for the exponential kernel. It is ignored for `"delta"`.
    """

    kind: str = "delta"
    tau: float | None = None

    def __post_init__(self) -> None:
        """Validate the kernel configuration."""
        if self.kind not in {"delta", "exp"}:
            raise ValueError("Kernel kind must be either 'delta' or 'exp'.")
        if self.kind == "exp" and self.tau is None:
            raise ValueError("Exponential kernels require a non-null tau.")
        if self.tau is not None and self.tau <= 0.0:
            raise ValueError("Kernel tau must be strictly positive.")


@dataclass(frozen=True, slots=True)
class DelayConfig:
    """Configuration of the feedback delay line.

    Attributes:
        delay_steps: Pure-delay lag measured in discrete physical time steps.
        kernel: Kernel configuration used to evaluate the effective source.
        max_history: Optional explicit cap for stored physical source samples.
    """

    delay_steps: int = 0
    kernel: KernelConfig = field(default_factory=KernelConfig)
    max_history: int | None = None

    def __post_init__(self) -> None:
        """Validate delay settings."""
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be non-negative.")
        if self.max_history is not None and self.max_history < 1:
            raise ValueError("max_history must be positive when provided.")


@dataclass(frozen=True, slots=True)
class ReproducibilityConfig:
    """Configuration for deterministic execution and artifact storage.

    Attributes:
        seed: Seed used for all pseudo-random components.
        store_every: Snapshot cadence for persisted records.
        output_dir: Optional artifact directory for JSON exports.
    """

    seed: int = 7
    store_every: int = 1
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate reproducibility settings."""
        if self.store_every < 1:
            raise ValueError("store_every must be at least 1.")


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Top-level simulation configuration.

    Attributes:
        total_time: Final physical time.
        dt: Fixed time step.
        delay: Delay-line settings.
        reproducibility: Reproducibility and artifact settings.
        label: Human-readable experiment label.
    """

    total_time: float = 1.0
    dt: float = 0.05
    delay: DelayConfig = field(default_factory=DelayConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    label: str = "reference-delay-simulation"

    def __post_init__(self) -> None:
        """Validate temporal discretization."""
        if self.total_time <= 0.0:
            raise ValueError("total_time must be strictly positive.")
        if self.dt <= 0.0:
            raise ValueError("dt must be strictly positive.")

    @property
    def num_steps(self) -> int:
        """Return the number of physical steps.

        Returns:
            Number of steps implied by `total_time / dt`.
        """
        return int(round(self.total_time / self.dt))
