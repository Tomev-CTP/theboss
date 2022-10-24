# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'theboss'
copyright = '2022, Tomasz Rybotycki'
author = 'Tomasz Rybotycki'
release = '4.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser'  # Markdown support
]

templates_path = ['_templates']
exclude_patterns = ['*GuanCodes*', 'quick_tests*', 'tests*', 'setup*']
add_module_names = False  # Don't display full module path

# -- Markdown support --------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/markdown.html
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}
myst_heading_anchors = 2

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['_static/width_fix.css', 'width_fix.css']

# -- Ignore selected modules -------------------------------------------------
# https://stackoverflow.com/questions/23543886/sphinx-adds-documentation-for-numpy-modules-to-my-docs

class Mock(object):

    __all__ = []

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return Mock()

    @classmethod
    def __getattr__(cls, name):
        if name in ('__file__', '__path__'):
            return '/dev/null'
        elif name[0] == name[0].upper():
            mockType = type(name, (), {})
            mockType.__module__ = __name__
            return mockType
        else:
            return Mock()

MOCK_MODULES = ['numpy', 'numpy.random', 'numpy.linalg', 'theboss.GuanCodes', 'GuanCodes', 'sklearn', 'sklearn.cluster', 'scipy', 'scipy.stats', 'scipy.special', 'quick_tests']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()