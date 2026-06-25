"""Tiny YAML config loader for dataset / model specs.

Configs live in ``configs/datasets/<name>.yaml`` and ``configs/models/<name>.yaml``.
Model config keys flow straight into the adapter's ``cfg`` (epochs, lr, scale,
emb_size, ...); dataset config carries num_classes / channels / seeds / val_split.
CLI flags override config values in the driver scripts (CLI > config > default).
"""
import os

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(_ROOT, 'configs')


def load_dataset_config(name):
    path = os.path.join(CONFIG_DIR, 'datasets', f'{name}.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_config(name):
    path = os.path.join(CONFIG_DIR, 'models', f'{name}.yaml')
    with open(path) as f:
        return yaml.safe_load(f)
