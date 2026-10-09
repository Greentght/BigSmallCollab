"""Framework config: YAML dataset/model specs + pretrained weight resolution.

Configs live in ``configs/datasets/<name>.yaml`` and ``configs/models/<name>.yaml``.
Dataset config carries dataset facts and shared split policy (num_classes,
num_subjects, seeds, val_split). Model config carries model identity/static shape
settings plus ``finetune`` hyperparameters resolved by dataset + protocol.

Pretrained weights are resolved from ``/data1/llx/pre_weight/`` by default, so
the benchmark is decoupled from upstream repo copies under ``/home/lixinli``.
Per-model weight files can be overridden with an environment variable, e.g. when
running against a freshly fine-tuned checkpoint:

    MIREPNET_WEIGHT=/path/to/mirepnet.pth
    CBRAMOD_WEIGHT=/path/to/cbramod.pth
    LABRAM_WEIGHT=/path/to/labram-base.pth
    CODEBRAIN_WEIGHT=/path/to/CodeBrain.pth
"""
import os

import yaml

from experiments.storage import WEIGHTS_ROOT, external_path, resolve_local_file

_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(_ROOT, 'configs')
WEIGHTS_DIR = '/data1/llx/pre_weight'

# model name -> (default filename under WEIGHTS_DIR, override env var)
_WEIGHTS = {
    'mirepnet': ('mirepnet.pth', 'MIREPNET_WEIGHT'),
    'cbramod': ('cbramod.pth', 'CBRAMOD_WEIGHT'),
    'labram': ('labram-base.pth', 'LABRAM_WEIGHT'),
    'codebrain': ('codebrain.pth', 'CODEBRAIN_WEIGHT'),
}
_FINETUNE_KEY = 'finetune'
_LEGACY_FINETUNE_KEYS = (
    'epochs', 'lr', 'batch_size', 'weight_decay',
    'dropout', 'scale', 'label_smoothing',
)
_PROTOCOL_ALIASES = {'within': 'fewshot'}


def _canonical_protocol(protocol):
    p = str(protocol).lower()
    return _PROTOCOL_ALIASES.get(p, p)


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset_config(name):
    path = os.path.join(CONFIG_DIR, 'datasets', f'{name}.yaml')
    return _load_yaml(path)


def load_loso_dataset_config(name):
    """Current source/session and baseline locations for a registered task."""
    dataset = load_dataset_config(name)
    if 'loso' not in dataset:
        raise KeyError(f'no current LOSO source registered for {name!r}')
    return dict(dataset['loso'])


def normalize_loso_model_config(name, cfg):
    """Normalize explicit descriptions of defaults in completed baselines.

    The former runners omitted these values from their manifests while using
    exactly these settings. Strip a field only when its value matches that
    historical effective default. A changed value remains in the comparison,
    so changing LR, filtering, or any other setting still rejects reuse.
    """
    documented = {
        'optimizer': 'adam' if name == 'mirepnet' else 'adamw',
        'lr_schedule': 'epoch_cosine', 'schedule_update': 'epoch_end',
        'duration_seconds': 4.0, 'class_weights': False,
        'warmup_epochs': 0, 'min_lr': 0.0, 'label_smoothing': 0.0,
    }
    if name == 'mirepnet':
        documented.update(l_freq=8.0, h_freq=30.0, apply_EA=True,
                          ea_scope='per_subject',
                          test_ea_policy='all_unlabeled_held_out_subject_trials',
                          channel_mapping='inverse_distance_to_45_channels')
    elif name == 'ifnet':
        documented['filter_bank_hz'] = [[4.0, 16.0], [16.0, 40.0]]
    elif name in ('eegnet', 'adfcnn'):
        documented.update(l_freq=8.0, h_freq=32.0, filter_order=4,
                          filter_phase='zero_phase', preprocessing_stage='cached_input')
    if name in ('ifnet', 'eegnet', 'adfcnn'):
        documented['skip_preprocess'] = False
    result = dict(cfg)
    for key, value in documented.items():
        if key in result and result[key] == value:
            result.pop(key)
    return result


def equivalent_loso_model_config(name, recorded, current):
    return normalize_loso_model_config(name, recorded) == normalize_loso_model_config(name, current)


def load_model_yaml(name):
    """Raw model YAML, including the nested ``finetune`` table."""
    path = os.path.join(CONFIG_DIR, 'models', f'{name}.yaml')
    return _load_yaml(path)


def resolve_finetune_config(model_cfg, dataset=None, protocol=None):
    """Resolve model finetune params for ``dataset`` + ``protocol``.

    Merge order is ``finetune.defaults`` < ``finetune.<dataset>.<protocol>``.
    If dataset/protocol are omitted, only defaults are returned for legacy
    callers that still need a model's natural training schedule.
    """
    ft = model_cfg.get(_FINETUNE_KEY) or {}
    resolved = dict(ft.get('defaults') or {})

    if dataset is None and protocol is None:
        if not resolved:
            resolved.update({k: model_cfg[k] for k in _LEGACY_FINETUNE_KEYS
                             if k in model_cfg})
        return resolved
    if dataset is None or protocol is None:
        raise ValueError('dataset and protocol must be passed together')

    protocol = _canonical_protocol(protocol)
    ds_cfg = ft.get(dataset)
    if ds_cfg is None:
        raise KeyError(f'missing finetune config for dataset {dataset!r}')
    dataset_defaults = {
        k: v for k, v in ds_cfg.items() if k not in ('fewshot', 'loso', 'within')
    }
    resolved.update(dataset_defaults)
    proto_cfg = ds_cfg.get(protocol)
    if proto_cfg is None:
        raise KeyError(
            f'missing finetune config for {dataset!r} protocol {protocol!r}')
    resolved.update(proto_cfg or {})
    return resolved


def load_model_config(name, dataset=None, protocol=None):
    """Model config for runtime adapter construction.

    Returns static model keys plus resolved finetune hyperparameters. The raw
    nested ``finetune`` table is intentionally stripped before configs flow into
    adapters.
    """
    cfg = load_model_yaml(name)
    runtime = {k: v for k, v in cfg.items() if k != _FINETUNE_KEY}
    runtime.update(resolve_finetune_config(cfg, dataset, protocol))
    return runtime


def weight_path(name):
    """Absolute path to a model's pretrained weights; raises if missing.

    Honors the per-model override env var, else falls back to ``WEIGHTS_DIR/<file>``.
    """
    key = name.lower()
    if key not in _WEIGHTS:
        raise KeyError(f'unknown model {name!r}; known: {list(_WEIGHTS)}')
    fname, env = _WEIGHTS[key]
    default_path = (
        str(WEIGHTS_ROOT / fname)
        if key == 'codebrain'
        else os.path.join(WEIGHTS_DIR, fname)
    )
    path = str(external_path(os.environ.get(env, default_path)))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{name} weights not found at {path}. Set {env} to override, or add '
            f'the weight file at {default_path}.')
    return str(resolve_local_file(path))
