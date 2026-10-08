"""Scratch-train IFNet, EEGNet and ADFCNN on the shared broadband 001 NPY.

The task trial order and label IDs are read from the completed wideband
MIRepNet/CBraMod input manifests. Model-specific filtering is then applied
before fitting the same fixed-epoch LOSO recipe as the existing small-model
baseline.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from models import get_adapter
from experiments.finetune import run_loso_small_baselines as small_baseline
from experiments.finetune import run_loso_five_datasets as protocol
from experiments.storage import require_external_output, resolve_local_file


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / 'configs/reproductions/loso_001_all_models_wideband_v1.yaml'
SOURCE_ROOT = Path('/data1/llx/BNCI2014001/broadband_0p1_75hz')
INPUT_ROOT = Path('/data1/llx/BigSmallcollab/cache/eegfm_alignment_v2/model_inputs/wideband_npy_v3')
RESULTS_ROOT = Path('/data1/llx/BigSmallcollab/results/reproductions/loso_config_alignment_v2/wideband_npy_v3')
RECIPE = 'scratch_supervised_loso_wideband_v1'
MODELS = ('ifnet', 'eegnet', 'adfcnn')
DATASETS = ('BNCI2014001', 'BNCI2014001-4')
SOURCE_MANIFEST = SOURCE_ROOT / 'manifest.json'
SOURCE_META = SOURCE_ROOT / 'meta.csv'
SOURCE_X = SOURCE_ROOT / 'X.npy'
TRIAL_DIR = 'all_sessions_source_train_session'


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='zero-based held-out subject IDs; default is all nine')
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _source_and_task_manifests():
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    expected = source_manifest['files']['X.npy']['sha256']
    if _sha256(SOURCE_X) != expected:
        raise RuntimeError('shared broadband X.npy hash does not match its manifest')
    trial_paths = {}
    trial_dfs = {}
    task_manifests = {}
    for dataset in DATASETS:
        for model in ('mirepnet', 'cbramod'):
            folder = INPUT_ROOT / dataset / model / 'all_sessions_source_train_session'
            trial_path = folder / 'trials.csv'
            task_manifest_path = folder / 'manifest.json'
            if not trial_path.is_file() or not task_manifest_path.is_file():
                raise FileNotFoundError(f'missing completed wideband trial manifest: {folder}')
            frame = pd.read_csv(trial_path)
            task_manifest = json.loads(task_manifest_path.read_text())
            trial_dfs[(dataset, model)] = frame
            trial_paths[(dataset, model)] = trial_path
            task_manifests[(dataset, model)] = task_manifest
        left = trial_dfs[(dataset, 'mirepnet')]
        right = trial_dfs[(dataset, 'cbramod')]
        if not left.equals(right):
            raise RuntimeError(f'{dataset}: MIRepNet and CBraMod trial lists differ')
    return source_manifest, trial_paths, trial_dfs, task_manifests


def _input_snapshot():
    source_manifest, trial_paths, trial_dfs, task_manifests = _source_and_task_manifests()
    files = {}
    for path in (SOURCE_X, SOURCE_META, SOURCE_MANIFEST):
        digest = _sha256(path)
        expected = source_manifest['files'].get(path.name, {}).get('sha256')
        if expected and digest != expected:
            raise RuntimeError(f'shared source hash mismatch for {path}')
        files[path.name] = {'path': str(path), 'sha256': digest,
                            'bytes': path.stat().st_size}
    task_hashes = {}
    for dataset in DATASETS:
        path = trial_paths[(dataset, 'cbramod')]
        task_hashes[dataset] = {
            'path': str(path), 'sha256': _sha256(path), 'bytes': path.stat().st_size,
            'mi_manifest_sha256': _sha256(INPUT_ROOT / dataset / 'mirepnet' / TRIAL_DIR / 'manifest.json'),
            'cbr_manifest_sha256': _sha256(INPUT_ROOT / dataset / 'cbramod' / TRIAL_DIR / 'manifest.json'),
        }
    return {
        'source_manifest': source_manifest,
        'files': files,
        'task_hashes': task_hashes,
        'task_manifests': task_manifests,
        'task_frames': trial_dfs,
        'spec_sha256': _sha256(SPEC_PATH),
    }


def _load_trials(dataset, snapshot):
    x_source = np.load(SOURCE_X, mmap_mode='r')
    meta = pd.read_csv(SOURCE_META, dtype={'session': str, 'trial_uid': str})
    if x_source.ndim != 3 or tuple(x_source.shape) != (5184, 22, 1001):
        raise RuntimeError(f'unexpected broadband source shape: {x_source.shape}')
    task_rows = snapshot['task_frames'][(dataset, 'cbramod')].copy()
    task_manifest = snapshot['task_manifests'][(dataset, 'cbramod')]
    classes = task_manifest['class_mapping']
    expected_classes = ({'left_hand': 0, 'right_hand': 1} if dataset == 'BNCI2014001'
                        else {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3})
    if classes != expected_classes:
        raise RuntimeError(f'{dataset}: new source task labels do not match expected mapping: {classes}')
    selected_meta = meta.loc[meta['session'] == '0train'].copy()
    selected_meta['subject_zero_based'] = selected_meta['subject'].astype(int) - 1
    source_lookup = dict(zip(selected_meta['trial_uid'].astype(str),
                             selected_meta.index.astype(int)))
    task_rows['trial_uid'] = task_rows['trial_uid'].astype(str)
    missing = [uid for uid in task_rows['trial_uid'] if uid not in source_lookup]
    if missing:
        raise RuntimeError(f'{dataset}: {len(missing)} task trials missing from broadband source')
    if not np.array_equal(task_rows['subject_zero_based'].to_numpy(dtype=int),
                          np.asarray([int(selected_meta.loc[source_lookup[u], 'subject']) - 1
                                      for u in task_rows['trial_uid']], dtype=int)):
        raise RuntimeError(f'{dataset}: trial manifest subject IDs do not match source metadata')
    source_rows = np.asarray([source_lookup[u] for u in task_rows['trial_uid']], dtype=int)
    source_label_names = np.asarray([
        str(selected_meta.loc[row, 'class_name']) for row in source_rows
    ])
    expected_y = np.asarray([classes[name] for name in source_label_names], dtype=np.int64)
    if not np.array_equal(task_rows['label_id'].to_numpy(dtype=np.int64), expected_y):
        raise RuntimeError(f'{dataset}: task labels do not match source class names')
    x = np.asarray(x_source[source_rows, :, :1000], dtype=np.float32)
    y = task_rows['label_id'].to_numpy(dtype=np.int64)
    subjects = task_rows['subject_zero_based'].to_numpy(dtype=np.int64)
    uids = task_rows['trial_uid'].to_numpy(dtype=str)
    local_indices = np.zeros(len(y), dtype=np.int64)
    for subject in range(9):
        positions = np.flatnonzero(subjects == subject)
        local_indices[positions] = np.arange(len(positions), dtype=np.int64)
    local_uids = np.column_stack((subjects, local_indices))
    expected_per_subject = 144 if dataset == 'BNCI2014001' else 288
    if len(y) != expected_per_subject * 9 or not np.array_equal(
            np.bincount(subjects, minlength=9), np.full(9, expected_per_subject)):
        raise RuntimeError(f'{dataset}: invalid selected trial counts')
    return x, y, subjects, uids, local_uids, task_rows.to_dict('records'), meta, None


def _prepare_source(model_name, x):
    if model_name == 'ifnet':
        return x, 'IFNet adapter filterbank: 4-16 Hz and 16-40 Hz directly from broadband NPY'
    from data.preproc import bandpass
    filtered = bandpass(np.asarray(x, dtype=np.float64), 250, 8.0, 32.0)
    return filtered.astype(np.float32), 'epoch Butterworth order-4 zero-phase bandpass 8-32 Hz'


def _config_for(model_name, dataset):
    cfg = config.load_model_config(model_name, dataset, 'loso')
    cfg.update(dataset_name=dataset, in_channels=22, samples=1000, sample_rate=250)
    return cfg


def _profile(model_name, dataset, cfg):
    preprocessing = ('IFNet adapter filterbank 4-16 Hz and 16-40 Hz from broadband NPY'
                     if model_name == 'ifnet' else
                     'epoch-level order-4 zero-phase Butterworth 8-32 Hz')
    return {
        'name': RECIPE,
        'parameter_source': f'configs/models/{model_name}.yaml::finetune.{dataset}.loso',
        'initialization': 'random_from_scratch_each_fold_and_seed',
        'optimizer': cfg.get('optimizer', cfg.get('optimizer_type', 'adamw')),
        'lr_schedule': 'cosine_annealing_per_epoch',
        'warmup_epochs': 0,
        'preprocessing': preprocessing,
        'validation_policy': 'none; evaluate fixed final epoch',
        'resolved_config': {k: v for k, v in cfg.items() if k != 'dataset_name'},
    }


def _preflight(model_name, datasets, snapshot, device):
    for dataset in datasets:
        x, y, *_ = _load_trials(dataset, snapshot)
        x, preprocessing = _prepare_source(model_name, x[:32])
        cfg = _config_for(model_name, dataset)
        adapter = get_adapter(model_name, device=device, **cfg)
        xb = adapter.preprocess(x)
        model = adapter.build(len(snapshot['task_manifests'][(dataset, 'cbramod')]['class_mapping']))
        model.train()
        yb = torch.as_tensor(y[:len(xb)], dtype=torch.long, device=device)
        logits = adapter.forward(model, xb.to(device))[1]
        if tuple(logits.shape) != (len(xb), len(snapshot['task_manifests'][(dataset, 'cbramod')]['class_mapping'])):
            raise RuntimeError(f'{model_name}/{dataset}: invalid logits {tuple(logits.shape)}')
        loss = nn.CrossEntropyLoss()(logits, yb)
        loss.backward()
        small_baseline._optimizer(model, cfg).step()
        print(f'[preflight-ok] {model_name} {dataset} preprocessing={preprocessing} '
              f'input={tuple(xb.shape)} logits={tuple(logits.shape)}', flush=True)
        del x, y, xb, model, adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run(args, snapshot, device):
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('--seeds must be non-empty and unique')
    source_payload = {
        'source': snapshot['files'],
        'task_manifests': snapshot['task_hashes'],
        'config_sha256': snapshot['spec_sha256'],
        'runner_sha256': _sha256(Path(__file__)),
    }
    for dataset in args.datasets:
        task_manifest = snapshot['task_manifests'][(dataset, 'cbramod')]
        classes = task_manifest['class_mapping']
        ds_spec = {'classes': classes, 'subjects': 9,
                   'per_subject_trials': [144 if dataset == 'BNCI2014001' else 288] * 9,
                   'native_channels': 22, 'canonical_fs_hz': 250,
                   'source_dataset': 'BNCI2014001', 'source_shape': [5184, 22, 1001],
                   'source_fs_hz': 250, 'selected_session': '0train',
                   'window': 'first_1000_native_samples_from_broadband_source'}
        x, y, subject_ids, source_uids, local_uids, _records, _meta, _labels = _load_trials(dataset, snapshot)
        x, preprocessing = _prepare_source(args.model, x)
        cfg = _config_for(args.model, dataset)
        profile = _profile(args.model, dataset, cfg)
        output_dir = RESULTS_ROOT / dataset / args.model / RECIPE
        output_dir = require_external_output(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        n_subjects = 9
        folds = list(range(n_subjects)) if args.folds is None else args.folds
        if any(f < 0 or f >= n_subjects for f in folds):
            raise ValueError(f'{dataset}: fold IDs must be in [0,8]')
        split_manifest = {
            'protocol': 'leave_one_subject_out', 'dataset': dataset,
            'model': args.model, 'recipe': RECIPE,
            'training_profile': profile,
            'dataset_source': 'BNCI2014001/broadband_0p1_75hz',
            'source_files': source_payload,
            'selected_session': '0train (legacy session_T)',
            'class_mapping': classes, 'num_classes': len(classes),
            'num_subjects': n_subjects, 'total_trials': len(y),
            'subject_trial_counts': ds_spec['per_subject_trials'],
            'window': {'samples': 1000, 'sampling_rate_hz': 250,
                       'window_label': '4s_from_acquired_signal'},
            'test_policy': 'one full subject held out; fixed final epoch once',
            'validation_policy': 'none_in_fixed_recipe_baseline',
            'initialization': 'random_from_scratch_each_fold_and_seed',
            'seeds': [0, 1, 2],
        }
        split_path = require_external_output(output_dir / 'split_manifest.json')
        if split_path.exists():
            old = json.loads(split_path.read_text())
            if old != split_manifest:
                raise RuntimeError(f'output directory belongs to another experiment: {split_path}')
        temporary_split = split_path.with_name(f'{split_path.name}.{os.getpid()}.tmp')
        temporary_split.write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + '\n')
        os.replace(temporary_split, split_path)
        print(f'[dataset] {args.model} {dataset} trials={len(y)} '
              f'preprocessing={preprocessing} device={device}', flush=True)
        for fold in folds:
            train_mask = subject_ids != fold
            test_mask = subject_ids == fold
            x_train, y_train = x[train_mask], y[train_mask]
            x_test, y_test = x[test_mask], y[test_mask]
            uid_train, uid_test = source_uids[train_mask], source_uids[test_mask]
            local_test = local_uids[test_mask]
            for seed in args.seeds:
                cell, result_path, checkpoint_path, manifest_path, history_path = \
                    protocol._cell_paths(output_dir, fold, seed)
                cell = require_external_output(cell)
                cell.mkdir(parents=True, exist_ok=True)
                cached = protocol._cell_is_complete(result_path, checkpoint_path,
                                                    manifest_path, history_path,
                                                    cfg, source_payload, None, False)
                if cached is not None:
                    print(f'[skip] {args.model} {dataset} S{fold + 1} seed={seed}', flush=True)
                    continue
                protocol._set_seed(seed)
                adapter = get_adapter(args.model, device=device, **cfg)
                model = adapter.build(len(classes))
                started = time.time()
                model, history = small_baseline._fit(adapter, model, x_train, y_train, cfg, seed)
                feats, logits = adapter.infer(model, x_test)
                metrics, probs, pred, cm = protocol._metrics(y_test, logits)
                metrics.update(model=args.model, dataset=dataset,
                               test_subject=fold + 1, seed=seed,
                               n_train=len(y_train), n_test=len(y_test),
                               elapsed_sec=round(time.time() - started, 2),
                               artifact_origin='trained_from_scratch_updated_wideband')
                result = {'y': y_test, 'pred': pred, 'probs': probs,
                          'logits': logits, 'feats': feats,
                          'sample_uid': uid_test,
                          'local_sample_uid': local_test,
                          'confusion_matrix': cm,
                          'metrics_json': np.asarray(json.dumps(
                              metrics, sort_keys=True, allow_nan=True))}
                manifest = {
                    'protocol': 'leave_one_subject_out', 'dataset': dataset,
                    'model': args.model, 'recipe': RECIPE,
                    'training_profile': profile,
                    'artifact_origin': 'trained_from_scratch_updated_wideband',
                    'dataset_source': 'BNCI2014001/broadband_0p1_75hz',
                    'source_files': source_payload,
                    'source_trial_selection': {
                        'session': '0train', 'legacy_session': 'session_T',
                        'window': 'first_1000_native_samples'},
                    'initialization': 'random_from_scratch',
                    'pretrained_checkpoint': None, 'pretrained_sha256': None,
                    'model_config': cfg,
                    'environment_snapshot': protocol._environment_snapshot(device),
                    'num_classes': len(classes), 'label_values': classes,
                    'test_subject': fold + 1,
                    'train_subjects': sorted(np.unique(subject_ids[train_mask]).astype(int).tolist()),
                    'seed': seed, 'n_train': len(y_train), 'n_test': len(y_test),
                    'train_sample_uids': uid_train.tolist(),
                    'test_sample_uids': uid_test.tolist(),
                    'local_test_sample_uids': local_test.tolist(),
                    'window': split_manifest['window'],
                    'selection_policy': 'fixed_final_epoch_no_validation',
                    'preprocessing': preprocessing,
                    'test_metrics': metrics,
                }
                protocol._save_cell(result_path, checkpoint_path, manifest_path,
                                    history_path, model, result, manifest, history)
                print(f'[done] {args.model} {dataset} S{fold + 1} seed={seed} '
                      f'acc={metrics["accuracy"]:.4f} '
                      f'bacc={metrics["balanced_accuracy"]:.4f} '
                      f'kappa={metrics["kappa"]:.4f}', flush=True)
                del model, adapter, feats, logits
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del x_train, y_train, x_test, y_test, uid_train, uid_test, local_test
        del x, y, subject_ids, source_uids, local_uids
        gc.collect()


def main():
    args = _args()
    if args.gpu == 0:
        raise ValueError('GPU 0 is prohibited for this experiment')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('--seeds must be non-empty and unique')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '4')))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    protocol.PROTOCOL = 'loso_001_all_models_wideband_v1'
    device = f'cuda:{args.gpu}'
    snapshot = _input_snapshot()
    if args.preflight_only:
        _preflight(args.model, args.datasets, snapshot, device)
    else:
        _run(args, snapshot, device)


if __name__ == '__main__':
    main()
