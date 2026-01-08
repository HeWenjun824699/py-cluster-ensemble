# Configuration file for the Sphinx documentation builder.
import os
import sys

# -- Project information -----------------------------------------------------
project = 'PCE'
copyright = '2026, The PCE Team'
author = 'The PCE Team'

# -- General configuration ---------------------------------------------------
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_static_path = ['_static']
html_logo = "_static/docs.png"
html_theme_options = {
    'logo_only': True
}


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
