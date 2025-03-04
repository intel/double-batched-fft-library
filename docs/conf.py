# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Double-Batched FFT Library'
copyright = '2025, Intel Corporation'
author = 'Intel Corporation'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['breathe']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

breathe_default_project = 'api'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_theme_options = {
    'repository_provider': 'github',
    'repository_url': 'https://github.com/intel/double-batched-fft-library',
    'use_repository_button': True,
    'use_issues_button': True,
    'use_source_button': True,
    'use_edit_page_button': True,
    'path_to_docs': 'docs',
    'navigation_with_keys': False
}
html_css_files = ['fix-scrollbar-bug.css']
