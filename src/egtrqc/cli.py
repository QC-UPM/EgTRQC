"""Command-line interface for the EgTRQC reference MVP."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from egtrqc.analysis import DenseAnalysisConfig, run_dense_delay_analysis
from egtrqc.config import DelayConfig, KernelConfig, ReproducibilityConfig, SimulationConfig
from egtrqc.diagnostics import audit_delay_definition, audit_single_advance_per_step
from egtrqc.io import write_result_json
from egtrqc.model import LinearBackreactionModel, LinearBackreactionModelConfig
from egtrqc.simulator import DelayAwareSimulator


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description='EgTRQC reference delay-aware simulator')
    subparsers = parser.add_subparsers(dest='command', required=True)

    sweep = subparsers.add_parser('run-sweep', help='Run the reference reproducibility sweep.')
    sweep.add_argument('--output-dir', type=Path, required=True, help='Directory where JSON files are saved.')
    sweep.add_argument('--delays', type=int, nargs='*', default=[0, 1, 2, 5], help='Delay values to run.')
    sweep.add_argument('--total-time', type=float, default=0.5, help='Final physical time.')
    sweep.add_argument('--dt', type=float, default=0.05, help='Discrete time step.')
    sweep.add_argument('--seed', type=int, default=7, help='Deterministic seed.')

    dense = subparsers.add_parser(
        'analyze-dense-delay',
        help='Run dense-state delay analysis with tables, Plotly HTML files, and Markdown report.',
    )
    dense.add_argument('--output-dir', type=Path, required=True, help='Directory where analysis artifacts are saved.')
    dense.add_argument('--delays', type=int, nargs='*', default=[0, 1, 2, 5], help='Delay values to analyze.')
    dense.add_argument('--total-time', type=float, default=0.5, help='Final physical time.')
    dense.add_argument('--dt', type=float, default=0.05, help='Discrete time step.')
    dense.add_argument('--seed', type=int, default=7, help='Deterministic seed.')
    return parser


def run_reference_sweep(output_dir: Path, delays: list[int], total_time: float, dt: float, seed: int) -> None:
    """Execute the standard sweep used for reproducibility packaging.

    Args:
        output_dir: Directory where artifacts are written.
        delays: Delay values to sweep.
        total_time: Final physical time.
        dt: Discrete time step.
        seed: Deterministic seed.
    """
    model = LinearBackreactionModel(LinearBackreactionModelConfig())
    manifest: list[dict[str, object]] = []

    for delay in delays:
        config = SimulationConfig(
            total_time=total_time,
            dt=dt,
            delay=DelayConfig(delay_steps=delay, kernel=KernelConfig(kind='delta')),
            reproducibility=ReproducibilityConfig(seed=seed, store_every=1, output_dir=output_dir),
            label=f'delay-{delay}',
        )
        result = DelayAwareSimulator(model=model, config=config).run()
        delay_report = audit_delay_definition(result)
        advance_report = audit_single_advance_per_step(result)
        artifact_path = write_result_json(result, output_dir / f'delay_{delay}.json')
        manifest.append(
            {
                'delay_steps': delay,
                'artifact': str(artifact_path),
                'delay_audit': asdict(delay_report),
                'single_advance_audit': asdict(advance_report),
            }
        )

    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def main() -> None:
    """Run the CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'run-sweep':
        run_reference_sweep(
            output_dir=args.output_dir,
            delays=list(args.delays),
            total_time=float(args.total_time),
            dt=float(args.dt),
            seed=int(args.seed),
        )
    elif args.command == 'analyze-dense-delay':
        artifacts = run_dense_delay_analysis(
            DenseAnalysisConfig(
                delays=tuple(int(x) for x in args.delays),
                total_time=float(args.total_time),
                dt=float(args.dt),
                seed=int(args.seed),
                output_dir=args.output_dir,
            )
        )
        print(json.dumps({
            'output_dir': str(artifacts.output_dir),
            'manifest_path': str(artifacts.manifest_path),
            'timeseries_csv': str(artifacts.timeseries_csv),
            'summary_csv': str(artifacts.summary_csv),
            'report_markdown': str(artifacts.report_markdown),
            'plot_files': [str(path) for path in artifacts.plot_files],
        }, indent=2))


if __name__ == '__main__':
    main()
