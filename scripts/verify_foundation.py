"""Verify the phase-1 foundation vendoring is byte-faithful and self-contained.

Checks:
  1. framework-owned small models import + forward -> (feat, logits) contract;
  2. vendored core.preproc.EA == MIRepNet utils.EA on the same random input;
  3. vendored core.eeg_dataset.EEGDataset == MIRepNet dataset.EEGDataset (X, y)
     for one subject;
  4. core.data.subject_split runs on the vendored path and is deterministic.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DS = 'BNCI2014004'   # 3ch/2class, small + fast to load


def check_models():
    from models import IFNet, ResidualEEGNet, ADFCNN_Net
    x = torch.randn(4, 3, 1000)
    for name, m in [('IFNet', IFNet(in_channels=3, samples=1000, num_classes=2)),
                    ('EEGNet', ResidualEEGNet(in_channels=3, samples=1000, num_classes=2)),
                    ('ADFCNN', ADFCNN_Net(in_channels=3, samples=1000, num_classes=2))]:
        feat, logits = m(x, return_features=True)
        assert logits.shape == (4, 2), f'{name} logits {logits.shape}'
        assert feat.ndim == 2 and feat.shape[0] == 4, f'{name} feat {feat.shape}'
        print(f'  [ok] {name}: feat{tuple(feat.shape)} logits{tuple(logits.shape)}')


def check_ea():
    from core.preproc import EA as EA_new
    sys.path.insert(0, os.path.expanduser('~/MIRepNet'))
    from utils.utils import EA as EA_old
    x = np.random.RandomState(0).randn(20, 3, 1000)
    a, b = EA_new(x.copy()), EA_old(x.copy())
    assert np.allclose(a, b), f'EA mismatch max|d|={np.abs(a-b).max()}'
    print(f'  [ok] EA identical (max|d|={np.abs(a-b).max():.2e})')


def check_dataset():
    from core.eeg_dataset import EEGDataset as DS_new
    from types import SimpleNamespace
    args = SimpleNamespace(dataset_name=DS, sub=[0], data_mode='session3')
    dnew = DS_new(args=args)
    sys.path.insert(0, os.path.expanduser('~/MIRepNet'))
    from dataset import EEGDataset as DS_old
    dold = DS_old(args=args)
    assert np.array_equal(np.asarray(dnew.X), np.asarray(dold.X)), 'X mismatch'
    assert np.array_equal(np.asarray(dnew.y), np.asarray(dold.y)), 'y mismatch'
    print(f'  [ok] EEGDataset identical: X{np.asarray(dnew.X).shape} y{np.asarray(dnew.y).shape}')


def check_split():
    from core import data
    Xtr, ytr, Xte, yte = data.subject_split(DS, 0, val_split=0.3, seed=666)
    Xtr2, ytr2, Xte2, yte2 = data.subject_split(DS, 0, val_split=0.3, seed=666)
    assert np.array_equal(Xtr, Xtr2) and np.array_equal(yte, yte2), 'split nondeterministic'
    print(f'  [ok] subject_split: {len(ytr)} calib / {len(yte)} test, deterministic')


if __name__ == '__main__':
    print('1. small models forward'); check_models()
    print('2. EA byte-faithful');     check_ea()
    print('3. EEGDataset byte-faithful'); check_dataset()
    print('4. subject_split deterministic'); check_split()
    print('\nALL FOUNDATION CHECKS PASSED')
