"""Run the session_3, four-second BNCI2014004 LOSO reproduction.

Run once per foundation-model environment:

    conda run -n mirepnet python experiments/finetune/run_loso004_reproduction.py \
        --model mirepnet --gpu 6
    conda run -n cbramod python experiments/finetune/run_loso004_reproduction.py \
        --model cbramod --gpu 6

The outer folds exactly hold out one subject. There is no validation set; the
fixed final epoch is evaluated once. Results are written to a new, protocol-
specific directory so they cannot be confused with older ``*_loso`` artifacts.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
import data
from models import get_adapter
from experiments.storage import external_path, require_external_output, resolve_local_file


DATASET = 'BNCI2014004'
PROJECT_PROTOCOL = 'loso_benchmark_session3_4s_v1'
EEGFM_CB_FULL_PROTOCOL = 'loso_eegfm_cbfull_session3_4s_v1'
RESULTS_ROOT = Path('/data1/llx/BigSmallCollab_results') / 'reproductions'
EEGFM_CONFIG = Path('/home/lixinli/EEG-FM-Benchmark/config/BNCI2014004.json')
SEEDS = (666, 667, 668)
EEGFM_SEEDS = (0, 1, 2)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('mirepnet', 'cbramod'), required=True)
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(9)),
                        help='zero-based held-out subject IDs (default: all 9)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='seed list; defaults to the selected profile defaults')
    parser.add_argument('--cbramod-profile', choices=('project_loso', 'eegfm_full'),
                        default='project_loso',
                        help='CBraMod hyperparameters: current LOSO baseline or '
                             'EEG-FM-Benchmark BNCI2014004 full-finetuning config')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--force', action='store_true',
                        help='overwrite cells already complete in this new protocol directory')
    return parser.parse_args()


def _sha256(path):
    path = resolve_local_file(path)
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def _model_config(model_name, cbramod_profile='project_loso'):
    cfg = config.load_model_config(model_name, DATASET, 'loso')
    if model_name == 'mirepnet':
        cfg.update(dataset_name=DATASET, in_channels=3, samples=1000,
                   skip_preprocess=True)
    else:
        # Match the benchmark's intended 004 preprocessing, with a true 4 s
        # crop: resample -> 0.3-75 Hz -> 60 Hz notch -> no EA/CAR -> 4 patches.
        cfg.update(dataset_name=DATASET, in_channels=3, samples=1000,
                   target_fs=200, l_freq=0.3, h_freq=75.0, notch_freq=60.0,
                   norm_method=None, apply_EA=False, scale=1.0,
                   feature_head='flatten', dropout=0.1)
        if cbramod_profile == 'eegfm_full':
            if not EEGFM_CONFIG.is_file():
                raise FileNotFoundError(
                    f'EEG-FM-Benchmark config not found: {EEGFM_CONFIG}')
            benchmark = json.loads(resolve_local_file(EEGFM_CONFIG).read_text())['CBraMod']['full']
            # Transfer only the full-finetuning hyperparameters and signal
            # filters. Keep the shared LOSO protocol's four-second window.
            cfg.update(
                epochs=int(benchmark['epochs']),
                lr=float(benchmark['lr']),
                batch_size=int(benchmark['batch_size']),
                weight_decay=float(benchmark['weight_decay']),
                dropout=float(benchmark['dropout_rate']),
                label_smoothing=float(benchmark['label_smoothing']),
                optimizer=str(benchmark['optimizer_type']).lower(),
                optimizer_eps=float(benchmark['opt_eps']),
                warmup_epochs=int(benchmark['warmup_epochs']),
                min_lr=float(benchmark['min_lr']),
                target_fs=int(benchmark['target_fs']),
                l_freq=float(benchmark['l_freq']),
                h_freq=float(benchmark['h_freq']),
                notch_freq=float(benchmark['notch_freq']),
                norm_method=benchmark['norm_method'],
                apply_EA=bool(benchmark['apply_EA']),
                scale=1.0,
            )
    return cfg


def _profile_metadata(model_name, cbramod_profile):
    if model_name != 'cbramod' or cbramod_profile == 'project_loso':
        return {
            'name': 'project_loso',
            'description': 'Current repository CBraMod LOSO configuration',
        }
    benchmark = json.loads(resolve_local_file(EEGFM_CONFIG).read_text())['CBraMod']['full']
    return {
        'name': 'eegfm_benchmark_full_transferred_to_loso',
        'source_config': str(EEGFM_CONFIG),
        'source_config_sha256': _sha256(EEGFM_CONFIG),
        'source_section': 'CBraMod.full',
        'source_task_mode': benchmark['task_mode'],
        'source_train_percentage': benchmark['train_percentage'],
        'source_parameters': benchmark,
        'transfer_notes': [
            'Uses all trials from the eight LOSO training subjects; the source '
            'config itself is Fewshot with train_percentage=0.3.',
            'Keeps the reproduction protocol four-second crop; source '
            'time_length=5.0 repeat-padding is not applied.',
            'Uses the shared runner flatten-plus-dropout-plus-linear head.',
            'Uses step-indexed warmup and cosine LR from the source schedule, '
            'applied before each optimizer update.',
        ],
    }


def _load_trials():
    xs, ys, subjects, uids = [], [], [], []
    expected_per_subject = {1: 160, 2: 120, **{s: 160 for s in range(3, 10)}}
    for subject in range(9):
        x, y = data.load_subject_raw(DATASET, subject, data_mode='session3')
        if x.ndim != 3 or x.shape[1] != 3 or x.shape[2] < 1000:
            raise ValueError(
                f'S{subject + 1}: expected at least 1000 raw samples on 3 channels, '
                f'got {x.shape}')
        if len(y) != expected_per_subject[subject + 1]:
            raise ValueError(
                f'S{subject + 1}: expected {expected_per_subject[subject + 1]} '
                f'session_3 trials, got {len(y)}')
        if set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError(f'S{subject + 1}: expected binary labels 0/1, got {np.unique(y)}')
        xs.append(np.asarray(x[:, :, :1000], dtype=np.float32))
        ys.append(np.asarray(y, dtype=np.int64))
        subjects.append(np.full(len(y), subject, dtype=np.int64))
        uids.append(np.column_stack((np.full(len(y), subject, dtype=np.int64),
                                     np.arange(len(y), dtype=np.int64))))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(subjects),
            np.concatenate(uids))


def _preprocess_mirepnet(x, subject_ids, adapter):
    from data.preproc import bandpass

    x = bandpass(np.asarray(x, dtype=np.float64), 250, 8.0, 30.0)
    return adapter.ea_pad_per_subject(x, subject_ids)


def _make_adapter(model_name, cfg, device, protocol):
    adapter = get_adapter(model_name, device=device, **cfg)
    adapter.name = protocol + '_' + model_name
    return adapter


def _make_model(model_name, cfg, device, num_classes, protocol):
    adapter = _make_adapter(model_name, cfg, device, protocol)
    return adapter, adapter.build(num_classes)


def _train_cbramod(adapter, model, x, y, cfg, seed, profile):
    _set_seed(seed)
    x_tensor = adapter.preprocess(x)
    y_tensor = torch.as_tensor(y, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor),
                        batch_size=int(cfg['batch_size']), shuffle=True,
                        generator=generator, num_workers=0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg['lr']),
        weight_decay=float(cfg['weight_decay']),
        eps=float(cfg.get('optimizer_eps', 1e-8)))
    if profile == 'eegfm_full':
        lr_schedule = adapter._cosine_schedule(
            float(cfg['lr']), float(cfg['min_lr']), int(cfg['epochs']),
            len(loader), warmup_epochs=int(cfg['warmup_epochs']))
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg['epochs']), eta_min=0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg['label_smoothing']))
    global_step = 0
    for epoch in range(int(cfg['epochs'])):
        model.train()
        loss_total = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(adapter.device)
            yb = yb.to(adapter.device)
            if profile == 'eegfm_full':
                adapter._apply_lr_schedule(optimizer, lr_schedule, global_step)
            optimizer.zero_grad(set_to_none=True)
            logits = adapter.forward(model, xb)[1]
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            global_step += 1
            loss_total += float(loss.detach()) * len(yb)
            seen += len(yb)
        if profile == 'project_loso':
            scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == int(cfg['epochs']):
            current_lr = (float(lr_schedule[min(global_step, len(lr_schedule) - 1)])
                          if profile == 'eegfm_full'
                          else float(scheduler.get_last_lr()[0]))
            print(f'  epoch={epoch + 1:02d}/{cfg["epochs"]} '
                  f'loss={loss_total / max(seen, 1):.5f} '
                  f'lr={current_lr:.3g}', flush=True)
    return model


def _metrics(y, logits):
    from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                                 roc_auc_score)

    probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    pred = probs.argmax(axis=1)
    return {
        'accuracy': float((pred == y).mean()),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'kappa': float(cohen_kappa_score(y, pred)),
        'auroc': float(roc_auc_score(y, probs[:, 1])),
    }, probs, pred


def _run_cell(model_name, fold, seed, cfg, device, x_train, y_train,
              x_test, y_test, uid_test, output_dir, weight_sha, force,
              protocol, profile, profile_meta):
    run_dir = output_dir / model_name / f'subject_{fold + 1:02d}' / f'seed_{seed}'
    result_path = run_dir / 'result.npz'
    checkpoint_path = run_dir / 'model.pt'
    manifest_path = run_dir / 'manifest.json'
    if result_path.exists() and checkpoint_path.exists() and manifest_path.exists() and not force:
        with np.load(resolve_local_file(result_path)) as saved:
            return json.loads(str(saved['metrics_json'].item()))
    run_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(seed)
    adapter, model = _make_model(model_name, cfg, device, 2, protocol)
    started = time.time()
    if model_name == 'mirepnet':
        model = adapter.finetune(model, x_train, y_train, 2)
    else:
        model = _train_cbramod(
            adapter, model, x_train, y_train, cfg, seed, profile)
    feats, logits = adapter.infer(model, x_test)
    metrics, probs, pred = _metrics(y_test, logits)
    metrics.update(model=model_name, test_subject=fold + 1, seed=seed,
                   n_train=int(len(y_train)), n_test=int(len(y_test)),
                   elapsed_sec=round(time.time() - started, 2))

    result_tmp = run_dir / 'result.npz.tmp'
    with result_tmp.open('wb') as stream:
        np.savez_compressed(stream, y=y_test, pred=pred, probs=probs,
                            logits=logits, feats=feats, sample_uid=uid_test,
                            metrics_json=np.asarray(json.dumps(metrics, sort_keys=True)))
    os.replace(result_tmp, result_path)
    checkpoint_tmp = run_dir / 'model.pt.tmp'
    torch.save(model.state_dict(), require_external_output(checkpoint_tmp))
    os.replace(checkpoint_tmp, checkpoint_path)
    manifest = {
        'protocol': protocol,
        'training_profile': profile_meta,
        'dataset': DATASET,
        'model': model_name,
        'test_subject': fold + 1,
        'train_subjects': [s + 1 for s in range(9) if s != fold],
        'seed': seed,
        'input_window': 'first_1000_samples_4s',
        'sample_rate_hz': 250,
        'labels': {'left_hand': 0, 'right_hand': 1},
        'train_selection': 'all_session_3_trials_of_8_subjects',
        'test_selection': 'all_session_3_trials_of_held_out_subject',
        'model_config': cfg,
        'pretrained_sha256': weight_sha,
        'test_sample_uids': uid_test.tolist(),
        'test_metrics': metrics,
        'selection_policy': 'fixed_final_epoch_no_validation',
        'ea_policy': ('per_subject_covariance; test uses its unlabeled full-fold '
                      'covariance (transductive)'
                      if model_name == 'mirepnet' else 'disabled'),
    }
    manifest_tmp = run_dir / 'manifest.json.tmp'
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    os.replace(manifest_tmp, manifest_path)
    print(f'[done] {model_name} S{fold + 1} seed={seed} '
          f'acc={metrics["accuracy"]:.4f} '
          f'bacc={metrics["balanced_accuracy"]:.4f} '
          f'kappa={metrics["kappa"]:.4f}', flush=True)
    del model, adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _write_summary(output_dir, model_name, rows, protocol, profile_meta):
    output_dir = require_external_output(output_dir)
    csv_path = output_dir / model_name / 'summary.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ['model', 'test_subject', 'seed', 'n_train', 'n_test', 'accuracy',
               'balanced_accuracy', 'kappa', 'auroc', 'elapsed_sec']
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row['seed'], row['test_subject'])))

    by_seed = {}
    for seed in sorted({int(row['seed']) for row in rows}):
        seed_rows = [row for row in rows if int(row['seed']) == seed]
        by_seed[str(seed)] = {
            metric: {
                'mean': float(np.mean([row[metric] for row in seed_rows])),
                'std_subjects': float(np.std([row[metric] for row in seed_rows], ddof=1))
                if len(seed_rows) > 1 else 0.0,
            }
            for metric in ('accuracy', 'balanced_accuracy', 'kappa', 'auroc')
        }
    summary = {
        'protocol': protocol,
        'training_profile': profile_meta,
        'model': model_name,
        'fold_seed_cells_completed': len(rows),
        'per_seed_subject_mean': by_seed,
        'across_fold_seed_mean': {
            metric: {
                'mean': float(np.mean([row[metric] for row in rows])),
                'std_fold_seed_cells': float(np.std([row[metric] for row in rows], ddof=1))
                if len(rows) > 1 else 0.0,
            }
            for metric in ('accuracy', 'balanced_accuracy', 'kappa', 'auroc')
        } if rows else {},
    }
    (output_dir / model_name / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')


def main():
    args = _parse_args()
    if args.model != 'cbramod' and args.cbramod_profile != 'project_loso':
        raise ValueError('--cbramod-profile eegfm_full applies only to CBraMod')
    profile = (args.cbramod_profile if args.model == 'cbramod'
               else 'project_loso')
    protocol = (EEGFM_CB_FULL_PROTOCOL if profile == 'eegfm_full'
                else PROJECT_PROTOCOL)
    seeds = args.seeds if args.seeds is not None else list(
        EEGFM_SEEDS if profile == 'eegfm_full' else SEEDS)
    output_dir = require_external_output(args.output or (RESULTS_ROOT / protocol))
    profile_meta = _profile_metadata(args.model, profile)
    if any(fold < 0 or fold >= 9 for fold in args.folds):
        raise ValueError('--folds values must be zero-based integers in [0, 8]')
    if len(set(args.folds)) != len(args.folds):
        raise ValueError('--folds contains duplicates')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('--seeds must be a non-empty list without duplicates')

    weight_path = Path(config.weight_path(args.model)).resolve()
    if not weight_path.is_file() or weight_path.stat().st_size == 0:
        raise FileNotFoundError(f'pretrained checkpoint is missing or empty: {weight_path}')
    weight_sha = _sha256(weight_path)
    device = (f'cuda:{args.gpu}' if args.gpu is not None and torch.cuda.is_available()
              else 'cpu')
    if args.gpu is not None and not torch.cuda.is_available():
        raise RuntimeError('--gpu was specified but CUDA is unavailable in this environment')
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '4')))

    cfg = _model_config(args.model, profile)
    x, y, subject_ids, sample_uid = _load_trials()
    from data.eeg_dataset import _DATA_ROOT
    source_root = Path(_DATA_ROOT) / DATASET
    data_sources = {
        name: {'path': str(source_root / name), 'sha256': _sha256(source_root / name)}
        for name in ('X.npy', 'labels.npy', 'meta004.csv')
    }
    split_manifest = {
        'protocol': protocol,
        'training_profile': profile_meta,
        'dataset': DATASET,
        'subjects': 9,
        'selected_session': 'session_3 (MOABB session 3test)',
        'total_trials': int(len(y)),
        'subject_trial_counts': {
            str(s + 1): int((subject_ids == s).sum()) for s in range(9)},
        'window': {'samples': 1000, 'seconds': 4, 'sample_rate_hz': 250},
        'label_values': {'left_hand': 0, 'right_hand': 1},
        'validation_subject': None,
        'test_policy': 'one full subject held out; evaluate final epoch once',
        'models_share_trial_uids': True,
        'mirepnet_ea': 'per subject; held-out test covariance uses unlabeled trials',
        'source_files': data_sources,
        'seeds': seeds,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / 'split_manifest.json'
    if split_path.exists():
        old = json.loads(resolve_local_file(split_path).read_text())
        old_seeds = old.pop('seeds', [])
        expected = dict(split_manifest)
        expected.pop('seeds', None)
        if old != expected:
            raise RuntimeError(f'output directory has a different split manifest: {split_path}')
        split_manifest['seeds'] = sorted(set(old_seeds) | set(seeds))
        split_path.write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + '\n')
    else:
        split_path.write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + '\n')

    print(f'[run] protocol={protocol} model={args.model} device={device} '
          f'folds={[f + 1 for f in args.folds]} seeds={seeds}', flush=True)
    print(f'[data] n={len(y)} per_subject='
          f'{[split_manifest["subject_trial_counts"][str(s)] for s in range(1, 10)]}', flush=True)
    print(f'[weights] {weight_path} sha256={weight_sha}', flush=True)
    print(f'[config] {json.dumps(cfg, sort_keys=True)}', flush=True)

    rows = []
    for fold in args.folds:
        train_mask = subject_ids != fold
        test_mask = subject_ids == fold
        train_subject_ids = subject_ids[train_mask]
        if set(np.unique(train_subject_ids).tolist()) != set(range(9)) - {fold}:
            raise RuntimeError(f'LOSO train subject set is invalid for fold {fold}')
        if np.any(train_mask & test_mask):
            raise RuntimeError(f'LOSO train/test overlap for fold {fold}')
        x_train_raw, y_train = x[train_mask], y[train_mask]
        x_test_raw, y_test = x[test_mask], y[test_mask]
        uid_test = sample_uid[test_mask]

        adapter = _make_adapter(args.model, cfg, device, protocol)
        if args.model == 'mirepnet':
            x_train = _preprocess_mirepnet(x_train_raw, train_subject_ids, adapter)
            x_test = _preprocess_mirepnet(
                x_test_raw, np.full(len(y_test), fold, dtype=np.int64), adapter)
        else:
            x_train = x_train_raw
            x_test = x_test_raw
        del adapter

        for seed in seeds:
            row = _run_cell(args.model, fold, seed, cfg, device,
                            x_train, y_train, x_test, y_test, uid_test,
                            output_dir, weight_sha, args.force, protocol,
                            profile, profile_meta)
            rows.append(row)
        del x_train, x_test, x_train_raw, x_test_raw

    # Include already-completed cells when a run is resumed or sharded by fold.
    rows = []
    for fold in range(9):
        # A resumed/sharded run may supply only a subset of seeds. The split
        # manifest accumulates the full seed set for this output directory.
        for seed in split_manifest['seeds']:
            path = (output_dir / args.model / f'subject_{fold + 1:02d}' /
                    f'seed_{seed}' / 'result.npz')
            if path.exists():
                with np.load(resolve_local_file(path)) as saved:
                    rows.append(json.loads(str(saved['metrics_json'].item())))
    _write_summary(output_dir, args.model, rows, protocol, profile_meta)
    print(f'[summary] wrote {output_dir / args.model / "summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
