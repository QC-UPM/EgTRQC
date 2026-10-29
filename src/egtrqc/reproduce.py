"""Command-line reproduction of the integrated numerical studies."""
from __future__ import annotations
import argparse
from pathlib import Path
from .study_controller import STUDIES, run_manifest
from .study_reporting import summarize, figures, frozen_checks, curvature_support_checks, validate_release
from .topology_studies import run_topology_study


def main():
    """Parse command-line options and dispatch the requested study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--study', choices=(*STUDIES, 'reduced_topology', 'all', 'report', 'validate'), default='validate')
    parser.add_argument('--recompute', action='store_true', help='Replace cached trajectories by newly executed trajectories.')
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / 'configs').exists():
        parser.error('--root must identify the repository root')
    if args.study in (*STUDIES, 'all'):
        for name in (STUDIES if args.study == 'all' else (args.study,)):
            run_manifest(root, name, recompute=args.recompute)
    if args.study in ('reduced_topology', 'all'):
        if args.recompute or not (root / 'results/reduced_topology/index.json').exists():
            run_topology_study(root / 'results/reduced_topology')
    if args.study in ('report', 'all'):
        summarize(root)
        for name in ('source_controls', 'active_short_lag'):
            frozen_checks(root, name)
        curvature_support_checks(root)
        figures(root)
    if args.study in ('validate', 'report', 'all'):
        import json
        print(json.dumps(validate_release(root), indent=2))


if __name__ == '__main__':
    main()
