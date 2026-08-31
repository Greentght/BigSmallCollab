"""Model registry: ``name -> ModelAdapter``.

Every model lives in its own sub-package ``models/<name>/`` holding the network
definition + its adapter (and any model-specific helpers). Imports are lazy/per-name
so an environment with only one model's dependencies (each big model runs in its own
conda env) can still construct that adapter without importing the others (whose deps
— timm, einops, mne — may be absent).
"""
KNOWN = ('ifnet', 'eegnet', 'adfcnn', 'mirepnet', 'cbramod', 'labram')

BIG_MODELS = ('mirepnet', 'cbramod', 'labram')
SMALL_MODELS = ('ifnet', 'eegnet', 'adfcnn')

def get_adapter(name, device='cpu', **cfg):
    """Instantiate the adapter for ``name`` with the given device + config."""
    n = name.lower()
    if n == 'ifnet':
        from models.ifnet.adapter import IFNetAdapter
        return IFNetAdapter(device=device, **cfg)
    if n == 'eegnet':
        from models.eegnet.adapter import EEGNetAdapter
        return EEGNetAdapter(device=device, **cfg)
    if n == 'adfcnn':
        from models.adfcnn.adapter import ADFCNNAdapter
        return ADFCNNAdapter(device=device, **cfg)
    if n == 'mirepnet':
        from models.mirepnet.adapter import MIRepNetAdapter
        return MIRepNetAdapter(device=device, **cfg)
    if n == 'cbramod':
        from models.cbramod.adapter import CBraModAdapter
        return CBraModAdapter(device=device, **cfg)
    if n == 'labram':
        from models.labram.adapter import LaBraMAdapter
        return LaBraMAdapter(device=device, **cfg)
    raise KeyError(f'unknown model {name!r}; known: {KNOWN}')
