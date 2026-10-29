"""Controllers for deterministic, manifest-driven benchmark execution."""
from __future__ import annotations

import json
from pathlib import Path

from .controlled_studies import StudyConfig, run_pair

STUDIES = ("source_controls", "active_short_lag", "register_scaling",
           "full_robustness", "long_window", "active_refinement", "population_refinement")


def run_manifest(root,study,recompute=False):
    """Execute or reuse every matched pair in a study manifest.

    Args:
        root: Repository root containing configs and results.
        study: Name of a supported JSON configuration manifest.
        recompute: Replace cached trajectories when true.

    Returns:
        Ordered per-pair summaries written to the study index."""
    root=Path(root)
    configs=json.loads((root/f'configs/{study}.json').read_text())
    rows=[run_pair(StudyConfig(**c),root/f'results/{study}',overwrite=recompute) for c in configs]
    (root/f'results/{study}/index.json').write_text(json.dumps(rows,indent=2))
    return rows


