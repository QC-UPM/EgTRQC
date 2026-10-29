"""Portable entry points for the reference notebook experiments and stored displays."""
from __future__ import annotations
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from io import StringIO
import json
import os
import numpy as np
import pandas as pd
from . import notebook_kernel as kernel


@contextmanager
def temporary_environment(values):
    """Temporarily set environment variables for the archived suite.

    Args:
        values: Environment variable names and temporary values.

    Yields:
        None while the temporary environment is active.
    """
    old = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update({key: str(value) for key, value in values.items()})
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_reference_suite(root, delays=(0,), profile='smoke', recompute=False):
    """Execute or reuse the historical reference scheduler.

    Args:
        root: Repository root.
        delays: Integer delay indices to execute.
        profile: Smoke uses M=2 and 0.04 s; reference uses M=10,20,40 and 2 s.
        recompute: Replace existing reference results when true.

    Returns:
        Archived suite summaries annotated with profile and protocol.
    """
    if profile not in ('smoke', 'reference'):
        raise ValueError("profile must be 'smoke' or 'reference'")
    root = Path(root)
    results = []
    for delay in delays:
        folder = root / 'results' / f'reference_{profile}' / f'delay{int(delay)}'
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / 'index.json'
        if target.exists() and not recompute:
            results.append(json.loads(target.read_text()))
            continue
        values = {
            'GTRQC_TFINAL': '.04' if profile == 'smoke' else '2.0',
            'GTRQC_DT': '.02', 'GTRQC_STORE': '1',
            'GTRQC_KFOLDS': '1' if profile == 'smoke' else '3',
            'GTRQC_OCTA_M_LIST': '2' if profile == 'smoke' else '10,20,40',
            'GTRQC_OCTA_DELAY_STEPS': int(delay), 'GTRQC_OCTA_KERNEL': 'delta',
            'GTRQC_MODES': 'fixed_projector_signed,fixed_projector_unsigned,instantaneous_projector_signed,instantaneous_projector_unsigned',
            'GTRQC_OUT_PREFIX': 'reference',
            'GTRQC_SAVE_ARTIFACTS': '1',
        }
        previous_directory = Path.cwd()
        try:
            os.chdir(folder)
            with temporary_environment(values), (folder / 'execution.log').open('w') as log, redirect_stdout(log):
                result = kernel.octa_thermodynamic_limit_suite()
        finally:
            os.chdir(previous_directory)
        for artifact in result.get('artifacts', []):
            for key in ('npz', 'json', 'pdf', 'csv'):
                if artifact.get(key):
                    candidate = Path(artifact[key])
                    artifact[key] = candidate.name
        result['execution_profile'] = profile
        result['protocol'] = 'archived_notebook_schedule'
        target.write_text(json.dumps(result, indent=2))
        results.append(result)
    return results


def reference_summary(results):
    """Flatten archived reference comparisons into a summary table.

    Args:
        results: Result dictionaries returned by run_reference_suite.

    Returns:
        DataFrame of comparisons without nested full-register summaries.
    """
    rows = []
    for result in results:
        for row in result['comparisons']:
            rows.append({key: value for key, value in row.items() if key != 'gtrqc_summary'})
    return pd.DataFrame(rows)


def stored_displays(root, relative_notebook):
    """Load recorded display payloads exported from a historical notebook.

    Args:
        root: Repository root.
        relative_notebook: Historical path relative to artifacts/notebooks.

    Returns:
        Original MIME display dictionaries from the corresponding JSON export."""
    path = Path(root) / 'results' / 'legacy_reference' / Path(relative_notebook).with_suffix('.displays.json')
    return json.loads(path.read_text())


def reference_curve_plot(root, delay, profile='smoke'):
    """Plot a reference curve from its recorded raw trajectory.

    Args:
        root: Repository root.
        delay: Integer reference delay index.
        profile: Smoke or full reference profile.

    Returns:
        Matplotlib figure for the first available instantaneous signed trajectory.
    """
    import matplotlib.pyplot as plt
    folder = Path(root) / 'results' / f'reference_{profile}' / f'delay{int(delay)}'
    candidates = sorted(folder.glob('*modeinstantaneous_projector_signed*M2*gq0*.npz'))
    if not candidates:
        candidates = sorted(folder.glob('*modeinstantaneous_projector_signed*gq0*.npz'))
    if not candidates:
        raise FileNotFoundError('Run the corresponding reference experiment first.')
    with np.load(candidates[0], allow_pickle=False) as data:
        time = data['times']; distance = data['trace_distance_to_nonlinear']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time[:len(distance)], distance, marker='o')
    ax.set(xlabel='Time (s)', ylabel='Trace distance',
           title=f'Reference protocol; {profile} profile; delay index {delay}')
    fig.tight_layout()
    return fig
