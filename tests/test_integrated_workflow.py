"""Integration checks for manifest execution and notebook-free reference outputs."""

import json
from dataclasses import asdict

from egtrqc.controlled_studies import StudyConfig
from egtrqc.reference_workflow import stored_displays
from egtrqc.study_controller import run_manifest


def test_manifest_execution_reuses_matching_trajectory(tmp_path, monkeypatch):
    """A rerun reuses the configured pair without executing physical dynamics."""
    config = StudyConfig(M=5, duration=.16, delay=.1)
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/source_controls.json').write_text(json.dumps([asdict(config)]))
    first = run_manifest(tmp_path, 'source_controls')
    trajectory = tmp_path / 'results/source_controls' / config.identifier / 'trajectory.npz'
    original_bytes = trajectory.read_bytes()

    def unexpected_execution(*args, **kwargs):
        raise AssertionError('A cached manifest must not construct physical models')

    monkeypatch.setattr('egtrqc.controlled_studies.make_models', unexpected_execution)
    second = run_manifest(tmp_path, 'source_controls')
    assert first == second
    assert trajectory.read_bytes() == original_bytes
    assert json.loads((tmp_path / 'results/source_controls/index.json').read_text()) == first
    assert first[0]['fr_buffer_advances'] == config.steps
    assert first[0]['source_index_error_fr'] == 0.


def test_historical_display_reader_uses_exported_json(tmp_path):
    """Recorded MIME displays remain readable after source notebooks are removed."""
    folder = tmp_path / 'results/legacy_reference/delay0'
    folder.mkdir(parents=True)
    payload = [{'text/plain': ['recorded output'], 'text/html': ['<b>result</b>']}]
    (folder / 'Visualization_delay0.displays.json').write_text(json.dumps(payload))
    assert stored_displays(tmp_path, 'delay0/Visualization_delay0.ipynb') == payload
