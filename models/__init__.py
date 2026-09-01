"""Model registry: ``name -> ModelAdapter``.

Every model lives in its own sub-package ``models/<name>/`` holding the network
definition + its adapter (and any model-specific helpers). Each adapter module
must export a single ``ADAPTER`` class (the ``ModelAdapter`` subclass); the
registry imports it on demand by convention ``models/<name>/adapter.py``, so
adding a model = adding a folder — no registry edit. Imports stay lazy/per-name,
so an environment with only one model's dependencies (each big model runs in its
own conda env) can still construct that adapter without importing the others
(whose deps — timm, einops, mne — may be absent).
"""
import importlib

KNOWN = ('ifnet', 'eegnet', 'adfcnn', 'mirepnet', 'cbramod', 'labram')

BIG_MODELS = ('mirepnet', 'cbramod', 'labram')
SMALL_MODELS = ('ifnet', 'eegnet', 'adfcnn')


def get_adapter(name, device='cpu', **cfg):
    """Instantiate the adapter for ``name`` with the given device + config."""
    n = name.lower()
    if n not in KNOWN:
        raise KeyError(f'unknown model {name!r}; known: {KNOWN}')
    mod = importlib.import_module(f'models.{n}.adapter')
    return mod.ADAPTER(device=device, **cfg)
