"""Sphinx configuration for the EgTRQC documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "EgTRQC"
copyright = "2026, Codex for EgTRQC"
author = "Codex for EgTRQC"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
graphviz_output_format = "svg"

html_theme = "nature"
html_static_path = ["_static"]
html_title = "EgTRQC Documentation"
