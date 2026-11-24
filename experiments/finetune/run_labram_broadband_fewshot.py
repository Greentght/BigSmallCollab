"""LaBraM Base supervised few-shot adaptation on validated broadband data.

Each process trains one subject/seed cell. The split matches the framework's
subject-wise 30% calibration / 70% test protocol. The optimization settings
adapt the historical native LaBraM recipe to this fixed-final-epoch protocol.
"""
from __future__ import annotations

import os

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if str(Path(p).resolve()) != _SCRIPT_DIR]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from scipy.signal import resample
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset

import config
from collab.seed import set_seed
from data import channels, loso_source
from data.preproc import bandpass, notch
from experiments.distill.run_fewshot_structure_kd import _split_subject
from experiments.finetune.labram_alexmi_broadband import load_alexmi_broadband
from experiments.storage import REPORTS_ROOT, RESULTS_ROOT, require_external_output, require_report_output
from models.labram.adapter import LaBraMAdapter
from models.labram.optim_factory import LayerDecayValueAssigner, get_parameter_groups

DATASETS = ('BNCI2014001', 'BNCI2014001-4', 'BNCI2014004', 'BNCI2015001', 'AlexMI')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_FILES = (
    'experiments/finetune/run_labram_broadband_fewshot.py',
    'experiments/finetune/labram_alexmi_broadband.py',
    'experiments/distill/run_fewshot_structure_kd.py', 'data/loso_source.py',
    'data/split.py', 'data/preproc.py', 'data/channels.py', 'collab/seed.py',
    'models/labram/adapter.py', 'models/labram/modeling_finetune.py',
    'models/labram/montage.py', 'models/labram/optim_factory.py',
    'models/base.py', 'models/__init__.py', 'experiments/storage.py', 'config.py',
    'configs/models/labram.yaml',
)


class BroadbandLaBraMAdapter(LaBraMAdapter):
    """Use the existing canonical montages along with the native adapters."""

    def _ch_names(self):
        if self.cfg['dataset_name'] == 'BNCI2015001':
            return channels.BNCI2015001_chn_names
        if self.cfg['dataset_name'] == 'AlexMI':
            return channels.AlexMI_chn_names
        return super()._ch_names()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, encoding='utf-8', delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def write_predictions(path, trial_uids, labels, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, encoding='utf-8', newline='', delete=False) as stream:
        writer = csv.writer(stream)
        writer.writerow(('trial_uid', 'y_true', 'y_pred'))
        writer.writerows((uid, int(label), int(pred))
                         for uid, label, pred in zip(trial_uids, labels, predictions))
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def build_optimizer(model, learning_rate, weight_decay, layer_decay):
    layers = model.get_num_layers()
    assigner = LayerDecayValueAssigner(
        [layer_decay ** (layers + 1 - index) for index in range(layers + 2)])
    skip = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else set()
    groups = get_parameter_groups(model, weight_decay, skip,
                                  assigner.get_layer_id, assigner.get_scale)
    for group in groups:
        group['lr'] = learning_rate * group.get('lr_scale', 1.0)
    return torch.optim.AdamW(groups, lr=learning_rate, eps=1e-8)


def scheduler_for(optimizer, epochs, warmup, learning_rate, min_lr):
    warmup = min(warmup, epochs)
    min_factor = min_lr / learning_rate

    def factor(epoch_index):
        step = epoch_index + 1
        if warmup and step <= warmup:
            return max(step / warmup, min_factor)
        progress = (step - warmup) / max(1, epochs - warmup)
        cosine = .5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return min_factor + (1 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def preprocess(adapter, raw, dataset):
    if dataset == 'AlexMI':
        # AlexMI contains three genuine seconds at 512 Hz, including the
        # endpoint. Drop that endpoint before resampling to three 1-s patches.
        raw = notch(np.asarray(raw, dtype=np.float64)[..., :1536], 512, 50.)
        raw = resample(raw, 600, axis=-1)
        raw = (raw / float(adapter.cfg['scale'])).astype(np.float32)
        return torch.as_tensor(raw.reshape(len(raw), raw.shape[1], 3, 200))
    # 001 and 5001 sources already carry a verified 0.1-75 Hz export filter.
    # 004's wider 0-120 Hz source is band-limited at epoch level for LaBraM.
    if dataset == 'BNCI2014004':
        raw = bandpass(np.asarray(raw, dtype=np.float64), 250, .1, 75.)
    raw = notch(np.asarray(raw, dtype=np.float64), 250, 50.)
    return adapter.preprocess(raw)


def weight_inventory(model, pretrained):
    checkpoint = torch.load(pretrained, map_location='cpu')
    weights = checkpoint.get('model', checkpoint)
    weights = {key[len('student.'):]: value for key, value in weights.items()
               if key.startswith('student.')}
    state = model.state_dict()
    matched = sorted(key for key, value in weights.items()
                     if key in state and value.shape == state[key].shape)
    return {'loaded_names': matched,
            'missing_names': sorted(set(state) - set(matched)),
            'discarded_names': sorted(set(weights) - set(matched)),
            'loaded_count': len(matched), 'model_tensor_count': len(state)}


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.gpu is not None and not torch.cuda.is_available():
        raise RuntimeError('GPU requested but CUDA is unavailable')
    device = torch.device('cpu' if args.gpu is None else f'cuda:{args.gpu}')
    dataset_cfg = config.load_dataset_config(args.dataset)
    if not 0 <= args.subject < int(dataset_cfg['num_subjects']):
        raise ValueError('subject must be a valid zero-based subject index')
    source = (load_alexmi_broadband() if args.dataset == 'AlexMI'
              else loso_source.load_dataset(args.dataset, model=None))
    raw = _split_subject(source, args.dataset, args.subject, args.seed, .3, .7)
    train_uids = raw['trial_uid_cal'].tolist()
    test_uids = raw['trial_uid_test'].tolist()
    if (len(set(train_uids)) != len(train_uids) or len(set(test_uids)) != len(test_uids)
            or set(train_uids) & set(test_uids)):
        raise ValueError('training and test trial UIDs must be unique and disjoint')
    weight_path = Path(config.weight_path('labram')).resolve()
    model_cfg = config.load_model_config('labram', args.dataset, 'fewshot')
    epochs = 1 if args.smoke else int(model_cfg['epochs'])
    training = {'epochs': epochs, 'optimizer': 'AdamW', 'lr': float(model_cfg['lr']),
                'batch_size': int(model_cfg['batch_size']),
                'weight_decay': float(model_cfg['weight_decay']), 'optimizer_eps': 1e-8,
                'layer_decay': .9, 'label_smoothing': .1, 'drop_path': .1,
                'warmup_epochs': min(5, epochs), 'min_lr': 1e-6,
                'scheduler': 'warmup_cosine_epoch', 'gradient_clip_norm': 3.,
                'evaluation': 'fixed_final_epoch', 'dataloader_workers': 0,
                'torch_intraop_threads': 1, 'torch_interop_threads': 1}
    preprocessing = {'source_variant': source.identity.get('source_variant'),
                     'canonical_window': ('3 s at 512 Hz; exclude endpoint before resampling'
                                          if args.dataset == 'AlexMI'
                                          else '4 s at 250 Hz via data.loso_source'),
                     'additional_bandpass_hz': [0.1, 75.0] if args.dataset == 'BNCI2014004' else None,
                     'notch_hz': 50.0, 'notch_q': 30, 'resample_hz': 200,
                     'notch_input_fs_hz': 512 if args.dataset == 'AlexMI' else 250,
                     'scale_divisor': float(model_cfg['scale']), 'patch_seconds': 1,
                     'patch_samples': 200, 'spatial_alignment': 'none'}
    code_files = (*CODE_FILES, f'configs/datasets/{args.dataset}.yaml')
    identity = {'dataset': args.dataset, 'subject': args.subject, 'seed': args.seed,
                'protocol': 'subject_session_stratified_30_70', 'smoke': bool(args.smoke),
                'num_classes': int(dataset_cfg['num_classes']),
                'source_manifest_sha256': source.source_manifest_sha256,
                'pretrained_sha256': file_hash(weight_path),
                'code_sha256': {name: file_hash(PROJECT_ROOT / name) for name in code_files},
                'train_uid_sha256': digest_json(train_uids),
                'test_uid_sha256': digest_json(test_uids),
                'training': training, 'model_config': model_cfg, 'preprocessing': preprocessing}
    identity_hash = digest_json(identity)
    root = require_external_output(RESULTS_ROOT / 'labram_broadband_fewshot' / args.run_id /
                                   args.dataset / f'subject{args.subject + 1:02d}' / f'seed{args.seed}')
    prediction_path = require_report_output(REPORTS_ROOT / 'fewshot/labram_broadband' / args.run_id /
                                            args.dataset / f'subject{args.subject + 1:02d}' /
                                            f'seed{args.seed}' / 'predictions.csv')
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'cell.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completion = root / 'completed.json'
        if completion.exists():
            record = json.loads(completion.read_text(encoding='utf-8'))
            if record.get('identity_sha256') != identity_hash or record.get('identity') != identity:
                raise ValueError('completed cell identity does not match requested experiment')
            if not prediction_path.is_file() or record.get('predictions_sha256') != file_hash(prediction_path):
                raise ValueError('completed cell predictions are missing or changed')
            print(f'[skip] validated completed cell {completion}', flush=True)
            return
        adapter_cfg = dict(model_cfg, dataset_name=args.dataset,
                           in_channels=int(raw['x_cal_raw'].shape[1]),
                           samples=1536 if args.dataset == 'AlexMI' else 1000)
        adapter = BroadbandLaBraMAdapter(device=str(device), **adapter_cfg)
        x_train = preprocess(adapter, raw['x_cal_raw'], args.dataset)
        x_test = preprocess(adapter, raw['x_test_raw'], args.dataset)
        y_train = torch.as_tensor(raw['y_cal'], dtype=torch.long)
        labels = np.asarray(raw['y_test'], dtype=np.int64)
        set_seed(args.seed)
        model = adapter.build(int(dataset_cfg['num_classes']))
        inventory = weight_inventory(model, weight_path)
        if inventory['loaded_count'] == 0:
            raise ValueError('no pretrained tensors were loaded')
        optimizer = build_optimizer(model, training['lr'], training['weight_decay'], training['layer_decay'])
        scheduler = scheduler_for(optimizer, epochs, training['warmup_epochs'], training['lr'], training['min_lr'])
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=training['label_smoothing'])
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=training['batch_size'],
                            shuffle=True, num_workers=0, generator=generator)
        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(adapter.forward(model, xb)[1], yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip_norm'])
                optimizer.step()
            scheduler.step()
            print(f'[train] {args.dataset} S{args.subject + 1:02d} seed={args.seed} '
                  f'epoch={epoch + 1}/{epochs} loss={float(loss):.6f}', flush=True)
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(x_test), training['batch_size']):
                logits = adapter.forward(model, x_test[start:start + training['batch_size']].to(device))[1]
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
        accuracy = float(np.mean(np.asarray(predictions) == labels) * 100)
        balanced_accuracy = float(balanced_accuracy_score(labels, predictions) * 100)
        kappa = float(cohen_kappa_score(labels, predictions))
        write_predictions(prediction_path, test_uids, labels, predictions)
        record = {'identity': identity, 'identity_sha256': identity_hash,
                  'accuracy_pct': accuracy, 'balanced_accuracy_pct': balanced_accuracy,
                  'kappa': kappa, 'n_train': len(train_uids), 'n_test': len(test_uids),
                  'train_trial_uids': train_uids, 'test_trial_uids': test_uids,
                  'predictions_path': str(prediction_path), 'predictions_sha256': file_hash(prediction_path),
                  'pretrained_inventory': inventory,
                  'runtime': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                              'device': str(device), 'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}}
        write_json(completion, record)
        print(json.dumps({'dataset': args.dataset, 'subject': args.subject + 1, 'seed': args.seed,
                          'accuracy_pct': accuracy, 'balanced_accuracy_pct': balanced_accuracy,
                          'kappa': kappa, 'n_train': len(train_uids),
                          'n_test': len(test_uids)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, default='BNCI2014001')
    parser.add_argument('--subject', type=int, required=True, help='zero-based subject index')
    parser.add_argument('--seed', type=int, choices=(666, 667, 668), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        parser.error('run-id must be a single path component')
    run(args)


if __name__ == '__main__':
    main()
