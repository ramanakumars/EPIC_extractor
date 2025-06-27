# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'EPIC_extractor'
copyright = '2025, Ramanakumar Sankar'
author = 'Ramanakumar Sankar'
release = '0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "myst_parser",
    "nbsphinx",
]

# napolean options
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    "repository_url": "https://github.com/ramanakumars/epicpy",
    "repository_branch": "main",
    "use_repository_button": True,
}
html_static_path = ['_static']


# -- copy over the notebook examples
def all_but_ipynb(dir, contents):
    result = []
    for c in contents:
        if os.path.isfile(os.path.join(dir, c)) and (not c.endswith(".ipynb")):
            result += [c]
    return result


project_root = "../../"
shutil.rmtree(os.path.join(project_root, "docs/source/notebooks"), ignore_errors=True)
shutil.copytree(
    os.path.join(project_root, "examples"),
    os.path.join(project_root, "docs/source/notebooks"),
    ignore=all_but_ipynb,
)
