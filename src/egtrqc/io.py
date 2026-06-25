"""Artifact serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path

from egtrqc.simulator import SimulationResult


def write_result_json(result: SimulationResult, output_path: Path) -> Path:
    """Write a simulation result as JSON.

    Args:
        result: Simulation result to serialize.
        output_path: Target JSON path.

    Returns:
        The written path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return output_path
