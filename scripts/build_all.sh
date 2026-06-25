#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jordieres/soft/vpy/bin/python}"
export PYTHONPATH="${ROOT_DIR}/src"

log() {
  printf '
[%s] %s
' "egtrqc" "$1"
}

run_cmd() {
  log "$1"
  shift
  "$@"
}

cd "${ROOT_DIR}"

run_cmd "Running test suite"   "${PYTHON_BIN}" -m pytest -q

run_cmd "Building reference sweep artifacts"   "${PYTHON_BIN}" -m egtrqc.cli run-sweep --output-dir artifacts/reference_sweep_vpy

run_cmd "Building dense delay analysis artifacts"   "${PYTHON_BIN}" -m egtrqc.cli analyze-dense-delay --output-dir artifacts/dense_delay_analysis

run_cmd "Building Sphinx documentation"   make -C docs html

log "Build completed successfully"
log "Artifacts: artifacts/reference_sweep_vpy, artifacts/dense_delay_analysis"
log "Docs: docs/_build/html/index.html"
