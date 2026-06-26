import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'quat'
copyright = '2026'
author = 'quat contributors'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

html_theme = 'furo'
html_title = 'quat Documentation'
html_static_path = []

autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
