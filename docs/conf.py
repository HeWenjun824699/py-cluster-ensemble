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

extensions = []

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


# 1. 指向你的源码路径（假设从 docs/source 出发，src 目录在两层之上）
sys.path.insert(0, os.path.abspath('../src'))

# 2. 启用必要的扩展
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary'
]
autosummary_generate = True

# 3. 设置主题
html_theme = 'sphinx_rtd_theme'

# 4. 配置 Napoleon 以支持 NumPy 风格
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
