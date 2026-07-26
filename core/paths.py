"""Framework-internal path resolution.

The framework is self-contained: model *code* lives under ``backbones/`` (vendored)
and ``models/``, and pretrained *weights* are resolved under ``weights/`` (git-ignored
``*.pth`` symlinks pointing at the unified store ``/data1/llx/pretrained_weights/`` —
real files on the stable data disk, decoupled from the upstream repos). This module
resolves those weight files; nothing here reaches into external repos.

Per-model weight files can be overridden with an environment variable, e.g. when
running against a freshly fine-tuned checkpoint:

    MIREPNET_WEIGHT=/path/to/MIRepNet.pth
    CBRAMOD_WEIGHT=/path/to/pretrained_weights.pth
    LABRAM_WEIGHT=/path/to/labram-base.pth
"""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(_ROOT, 'weights')

# model name -> (default filename under weights/, override env var)
_WEIGHTS = {
    'mirepnet': ('mirepnet.pth', 'MIREPNET_WEIGHT'),
    'cbramod': ('cbramod.pth', 'CBRAMOD_WEIGHT'),
    'labram': ('labram-base.pth', 'LABRAM_WEIGHT'),
}


def weight_path(name):
    """Absolute path to a model's pretrained weights; raises if missing.

    Honors the per-model override env var, else falls back to ``weights/<file>``.
    """
    key = name.lower()
    if key not in _WEIGHTS:
        raise KeyError(f'unknown model {name!r}; known: {list(_WEIGHTS)}')
    fname, env = _WEIGHTS[key]
    path = os.environ.get(env, os.path.join(WEIGHTS_DIR, fname))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{name} weights not found at {path}. Set {env} to override, or add '
            f'the file/symlink under {WEIGHTS_DIR}/.')
    return path
