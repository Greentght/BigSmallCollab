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

# Back-compat alias: ``cbramod_native`` was the interim name of the final settled
# CBraMod teacher while an older avg-pool variant still existed. That variant is
# gone; ``cbramod`` now *is* the native teacher. The alias is kept so existing
# pipeline scripts and cached artifacts under ``.../cbramod_native/`` keep working.
_ALIASES = {'cbramod_native': 'cbramod'}


def get_adapter(name, device='cpu', **cfg):
    """Instantiate the adapter for ``name`` with the given device + config."""
    n = _ALIASES.get(name.lower(), name.lower())
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
        from models.cbramod.adapter_native import CBraModNativeAdapter
        return CBraModNativeAdapter(device=device, **cfg)
    if n == 'labram':
        from models.labram.adapter import LaBraMAdapter
        return LaBraMAdapter(device=device, **cfg)
    raise KeyError(f'unknown model {name!r}; known: {KNOWN}')
