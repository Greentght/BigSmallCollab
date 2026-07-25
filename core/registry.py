"""name -> ModelAdapter factory.

Imports are lazy/per-name so that an environment which only has, say, MIRepNet's
dependencies can still instantiate the small + mirepnet adapters without importing
CBraMod/LaBraM (whose deps — timm, h5py, einops — may be absent). Each model runs
in its own conda env; the registry only constructs the adapter you ask for.
"""

# Declared so callers / configs can validate names without importing anything.
KNOWN = ('ifnet', 'eegnet', 'adfcnn', 'mirepnet', 'cbramod', 'cbramod_native',
         'labram')

BIG_MODELS = ('mirepnet', 'cbramod', 'cbramod_native', 'labram')
SMALL_MODELS = ('ifnet', 'eegnet', 'adfcnn')


def get_adapter(name, device='cpu', **cfg):
    """Instantiate the adapter for ``name`` with the given device + config."""
    n = name.lower()
    if n == 'ifnet':
        from adapters.small import IFNetAdapter
        return IFNetAdapter(device=device, **cfg)
    if n == 'eegnet':
        from adapters.small import EEGNetAdapter
        return EEGNetAdapter(device=device, **cfg)
    if n == 'adfcnn':
        from adapters.small import ADFCNNAdapter
        return ADFCNNAdapter(device=device, **cfg)
    if n == 'mirepnet':
        from adapters.mirepnet import MIRepNetAdapter
        return MIRepNetAdapter(device=device, **cfg)
    if n == 'cbramod':
        from adapters.cbramod import CBraModAdapter
        return CBraModAdapter(device=device, **cfg)
    if n == 'cbramod_native':
        from adapters.cbramod_native import CBraModNativeAdapter
        return CBraModNativeAdapter(device=device, **cfg)
    if n == 'labram':
        from adapters.labram import LaBraMAdapter
        return LaBraMAdapter(device=device, **cfg)
    raise KeyError(f'unknown model {name!r}; known: {KNOWN}')
