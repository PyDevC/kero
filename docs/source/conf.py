import os
import sys
import pathlib
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def get_kero_version():
    version = pathlib.Path(os.path.dirname(__file__), '..', '..', 'version.txt').read_text().strip()
    revision = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(__file__)
    )
    git_sha = revision.decode('ascii').strip()
    return version + '+git' + git_sha[:7]


project = 'kero'
copyright = '2025, PyDevC'
author = 'PyDevC'
release = get_kero_version()

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
]

myst_enable_extensions = [
    'colon_fence',
    'deflist',
]

source_suffix = {
    '.md': 'markdown',
    '.rst': 'restructuredtext',
}

autodoc_mock_imports = [
    'kero._engine',
    'kero._engine._kero',
    'kero._engine._kero._mlir_libs',
    'kero._engine._kero._mlir_libs._keroEngine',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
