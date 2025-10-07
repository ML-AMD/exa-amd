import sys
import os
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, os.path.abspath('../..'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'exa-AMD'
copyright = 'Copyright 2025. Iowa State University & © 2025. Triad National Security, LLC'
author = 'ML-AMD team'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon'
]

autodoc_mock_imports = [
    'torch', 'torchvision',
    'parsl',
    'pymatgen',
    'spglib',
    'monty',
    'mp_api',
    'ternary',
    'matplotlib', 'matplotlib.pyplot',
    'scipy', 'scipy.spatial',
    'sklearn',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'inherited-members': False,
}

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
