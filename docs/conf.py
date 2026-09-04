"""Sphinx configuration for the REMFlow documentation."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
    PROJECT_VERSION = tomllib.load(pyproject_file)["project"]["version"]

project = "REMFlow"
author = "REMFlow contributors"
copyright = "2026, REMFlow contributors"
version = PROJECT_VERSION
release = PROJECT_VERSION

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"REMFlow {release}"

nitpicky = False
