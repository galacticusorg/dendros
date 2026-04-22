# Configuration file for the Sphinx documentation builder.
#
# For a full list of configuration options see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the src directory to the path so sphinx-autodoc can find the package.
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -------------------------------------------------------

project = "Dendros"
author = "Andrew Benson"
copyright = "2026, Andrew Benson"

# Read the version from the installed package to avoid duplication.
try:
    from importlib.metadata import version as _version
    release = _version("dendros")
except Exception:
    release = "0.1.1"
version = ".".join(release.split(".")[:2])

# -- General configuration -----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output ---------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Intersphinx mappings ------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
}
