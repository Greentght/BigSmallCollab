"""Resolve and inject the three upstream model repos onto sys.path.

The framework treats MIRepNet / CBraMod / LaBraM as *read-only model sources*:
their code is imported, never copied. Default locations are ``~/<Repo>`` but can
be overridden per-repo with environment variables (useful when an adapter runs
inside that repo's own conda env on a different layout):

    MIREPNET_REPO=/path/to/MIRepNet
    CBRAMOD_REPO=/path/to/CBraMod
    LABRAM_REPO=/path/to/LaBraM

``add_repo(name)`` is idempotent and prepends the repo to ``sys.path`` so the
upstream package layout (e.g. ``from model.IFNet import IFNet``) resolves.
"""
import os
import sys

_HOME = os.path.expanduser('~')

REPOS = {
    'mirepnet': os.environ.get('MIREPNET_REPO', os.path.join(_HOME, 'MIRepNet')),
    'cbramod': os.environ.get('CBRAMOD_REPO', os.path.join(_HOME, 'CBraMod')),
    'labram': os.environ.get('LABRAM_REPO', os.path.join(_HOME, 'LaBraM')),
}


def repo_path(name):
    """Absolute path of an upstream repo; raises if it is missing on disk."""
    key = name.lower()
    if key not in REPOS:
        raise KeyError(f'unknown repo {name!r}; known: {list(REPOS)}')
    path = REPOS[key]
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f'{name} repo not found at {path}. Set {key.upper()}_REPO to override.')
    return path


def add_repo(name):
    """Prepend an upstream repo to sys.path (idempotent). Returns its path."""
    path = repo_path(name)
    if path not in sys.path:
        sys.path.insert(0, path)
    return path
