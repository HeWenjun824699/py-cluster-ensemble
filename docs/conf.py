# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'PCE'
copyright = '2026, The PCE Team'
author = 'The PCE Team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_static_path = ['_static']


# 1. Add source directory to sys.path
sys.path.insert(0, os.path.abspath('../src'))

# 2. Enable necessary extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme'
]
autosummary_generate = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# 3. Set the HTML theme
html_theme = 'sphinx_rtd_theme'

# 4. Configure Napoleon settings for NumPy style docstrings
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# 5. Mock heavy dependencies for Read the Docs
autodoc_mock_imports = [
    'igraph',
    'leidenalg',
    'community',
    'pyreadr',
    'fastcluster'
]
