#!/usr/bin/env bash
# Test, validate cached studies, regenerate reports, and build documentation.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
cd "${ROOT_DIR}"
"${PYTHON_BIN}" -m pytest -q
"${PYTHON_BIN}" -m egtrqc.reproduce --root "${ROOT_DIR}" --study all "$@"
"${PYTHON_BIN}" -m sphinx -W --keep-going -b html docs/source docs/_build/html
