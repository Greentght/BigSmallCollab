"""Export frozen LOSO teachers on source trials and validate reusable KD targets.

Caches preserve each teacher's native penultimate feature.  The held-out subject
is never inferred for this export; each cache belongs to one teacher/fold/seed.
This module is importable from the student environment without loading either
teacher's model implementation.  Run the exporter in the teacher environment.
"""
import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from experiments.finetune import run_loso_five_datasets as protocol


ROOT = protocol.ROOT
PROTOCOL = 'loso_five_settings_kd_feature_warmup10_v1'
CACHE_ROOT = ROOT / 'results/distill' / PROTOCOL / 'teacher_cache'
TEACHERS = ('mirepnet', 'cbramod')
CACHE_VERSION = 1
_HASHES = {}
_CODE_FILES = (
    'experiments/distill/loso_teacher_cache.py',
    'experiments/finetune/run_loso_five_datasets.py',
    'data/eeg_dataset.py', 'data/preproc.py', 'data/channels.py',
    'models/base.py', 'models/mirepnet/adapter.py', 'models/mirepnet/mlm.py',
    'models/cbramod/adapter.py', 'models/cbramod/cbramod.py',
    'models/cbramod/criss_cross_transformer.py',
)


def _sha256(path):
    """Memoize hashes only while the file's identity/timestamps/size match."""
    path = Path(path).resolve()
    stat = path.stat()
    signature = (stat.st_dev, stat.st_ino, stat.st_size,
                 stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _HASHES.get(str(path))
    if cached is not None and cached[0] == signature:
        return cached[1]
    digest = protocol._sha256(path)
    _HASHES[str(path)] = (signature, digest)
    return digest


def _array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_request(teacher, dataset, fold, seed, spec):
    if teacher not in TEACHERS or dataset not in protocol.DATASET_NAMES:
        raise ValueError(f'unknown teacher/dataset: {teacher}/{dataset}')
    if not isinstance(fold, (int, np.integer)) or not 0 <= fold < spec['datasets'][dataset]['subjects']:
        raise ValueError(f'{dataset}: invalid zero-based fold {fold}')
    if not isinstance(seed, (int, np.integer)):
        raise ValueError(f'invalid seed {seed}')


def teacher_recipe(teacher, dataset, spec):
    """Return the canonical teacher runner flag: main or eegfm_full."""
    if teacher not in TEACHERS or dataset not in spec['datasets']:
        raise ValueError(f'unknown teacher/dataset: {teacher}/{dataset}')
    return 'eegfm_full' if teacher == 'cbramod' and dataset != 'AlexMI' else 'main'


def teacher_cell(teacher, dataset, fold, seed, spec):
    """Return the completed canonical teacher directory (fold is zero-based)."""
    _validate_request(teacher, dataset, fold, seed, spec)
    recipe = teacher_recipe(teacher, dataset, spec)
    name = (spec['optional_cbramod_reference_recipe']['name']
            if recipe == 'eegfm_full' else spec['main_recipes'][teacher]['name'])
    return protocol.RESULTS_ROOT / dataset / teacher / name / f'subject_{fold+1:02d}' / f'seed_{seed}'


def cache_paths(teacher, dataset, fold, seed):
    """Return (cache directory, cache.npz, cache_manifest.json)."""
    if teacher not in TEACHERS or dataset not in protocol.DATASET_NAMES:
        raise ValueError(f'unknown teacher/dataset: {teacher}/{dataset}')
    if not isinstance(fold, (int, np.integer)) or fold < 0:
        raise ValueError(f'invalid zero-based fold {fold}')
    cell = CACHE_ROOT / dataset / teacher / f'subject_{fold+1:02d}' / f'seed_{seed}'
    return cell, cell / 'cache.npz', cell / 'cache_manifest.json'


@contextmanager
def _lock(cell, exclusive):
    if exclusive:
        cell.mkdir(parents=True, exist_ok=True)
    with (cell / 'cache.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _expected_arrays(expected_uid, expected_y, fold, num_classes):
    uid = np.asarray(expected_uid)
    y = np.asarray(expected_y)
    if uid.ndim != 2 or uid.shape[1] != 2 or uid.dtype.kind not in 'iu':
        raise RuntimeError('expected sample UID must be an integer [N,2] array')
    if y.ndim != 1 or y.dtype.kind not in 'iu' or len(y) != len(uid) or not len(y):
        raise RuntimeError('expected labels must be a nonempty integer [N] array')
    uid, y = uid.astype(np.int64), y.astype(np.int64)
    if np.any(uid[:, 0] == fold) or np.any(uid < 0):
        raise RuntimeError('teacher cache request includes the held-out subject or invalid UID')
    if len(np.unique(uid, axis=0)) != len(uid):
        raise RuntimeError('duplicate training sample UID')
    if np.any(y < 0) or np.any(y >= num_classes):
        raise RuntimeError('training labels disagree with the canonical class mapping')
    return uid, y


def _source_payload(dataset, spec, snapshot):
    ds_source = spec['datasets'][dataset]['source_dataset']
    files = {}
    for name, record in snapshot['sources'][ds_source]['files'].items():
        path = Path(protocol._DATA_ROOT) / ds_source / name
        digest = _sha256(path)
        if digest != record['sha256']:
            raise RuntimeError(f'{dataset}: source file changed: {path}')
        files[name] = {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}
    trial_sha = _sha256(protocol.TRIALS_PATH)
    if trial_sha != snapshot['manifests']['trial_manifest']['sha256']:
        raise RuntimeError('canonical trial manifest differs from frozen snapshot')
    return {'dataset_source': ds_source, 'files': files,
            'trial_manifest_sha256': trial_sha, 'spec_sha256': snapshot['spec_sha256']}


def _teacher_context(teacher, dataset, fold, seed, uid, y, spec, snapshot):
    source_cell = teacher_cell(teacher, dataset, fold, seed, spec)
    cp, mp = source_cell / 'model.pt', source_cell / 'manifest.json'
    if not all(p.is_file() for p in (cp, mp, source_cell / 'result.npz')):
        raise FileNotFoundError(f'complete teacher artifacts missing: {source_cell}')
    manifest = json.loads(mp.read_text())
    recipe = source_cell.parent.parent.name
    recorded_recipe = manifest.get('recipe')
    if (dataset == 'BNCI2014004' and recorded_recipe == 'main'
            and manifest.get('artifact_origin') == 'reused_004_main_verified'):
        recorded_recipe = spec['main_recipes'][teacher]['name']
    if recorded_recipe != recipe:
        raise RuntimeError(f'{mp}: teacher recipe mismatch')
    checks = {'protocol': protocol.PROTOCOL, 'model': teacher, 'dataset': dataset,
              'test_subject': fold + 1, 'seed': seed,
              'num_classes': len(spec['datasets'][dataset]['classes']),
              'label_values': spec['datasets'][dataset]['classes'],
              'selection_policy': 'fixed_final_epoch_no_validation'}
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f'{mp}: teacher {key} mismatch')
    if not np.array_equal(np.asarray(manifest['train_sample_uids'], dtype=np.int64), uid):
        raise RuntimeError(f'{mp}: teacher train UIDs do not match requested source trials')
    test_uid = np.asarray(manifest['test_sample_uids'], dtype=np.int64)
    if (manifest['n_train'] != len(y) or test_uid.ndim != 2 or test_uid.shape[1] != 2
            or len(test_uid) != manifest['n_test'] or not np.all(test_uid[:, 0] == fold)):
        raise RuntimeError(f'{mp}: invalid teacher source/target partition')
    source_files = _source_payload(dataset, spec, snapshot)
    if manifest['source_files'] != source_files:
        raise RuntimeError(f'{mp}: teacher source metadata differs from canonical data')
    if manifest['pretrained_sha256'] != snapshot['pretrained_weights'][teacher]['sha256']:
        raise RuntimeError(f'{mp}: teacher pretrained origin mismatch')
    cfg = manifest['model_config']
    ds = spec['datasets'][dataset]
    if (cfg.get('dataset_name') != dataset or cfg.get('in_channels') != ds['native_channels']
            or cfg.get('samples') != 1000):
        raise RuntimeError(f'{mp}: teacher input shape configuration mismatch')
    if teacher == 'cbramod' and cfg.get('feature_head') != 'flatten':
        raise RuntimeError(f'{mp}: expected native flatten CBraMod task head')
    if teacher == 'mirepnet' and not cfg.get('skip_preprocess'):
        raise RuntimeError(f'{mp}: MIRepNet must use canonical per-subject preprocessing')
    feature_dim = int(cfg.get('emb_size', 256)) if teacher == 'mirepnet' else ds['native_channels'] * 4 * 200
    context = {'cache_version': CACHE_VERSION, 'protocol': PROTOCOL,
               'teacher_protocol': protocol.PROTOCOL, 'teacher': teacher, 'dataset': dataset,
               'fold_zero_based': int(fold), 'test_subject': int(fold + 1), 'seed': int(seed),
               'teacher_recipe': recipe, 'teacher_recipe_flag': teacher_recipe(teacher, dataset, spec),
               'teacher_checkpoint': {'path': str(cp), 'sha256': _sha256(cp)},
               'teacher_checkpoint_sha256': _sha256(cp),
               'teacher_manifest': {'path': str(mp), 'sha256': _sha256(mp)},
               'teacher_manifest_sha256': _sha256(mp),
               'model_config': cfg, 'model_config_sha256': protocol._json_hash(cfg),
               'source_files': source_files,
               'source_files_sha256': protocol._json_hash(source_files),
               'source_snapshot_sha256': _sha256(protocol.SNAPSHOT_PATH),
               'num_classes': len(ds['classes']), 'class_mapping': ds['classes'],
               'feature_dim': feature_dim, 'feature_transform': 'native_penultimate_identity',
               'n_train': len(y), 'sample_uid_sha256': _array_hash(uid),
               'labels_sha256': _array_hash(y),
               'train_subjects_zero_based': sorted(np.unique(uid[:, 0]).tolist()),
               'target_data_policy': 'source_trials_only_no_held_out_teacher_inference',
               'code_files': {path: _sha256(ROOT / path) for path in _CODE_FILES}}
    context['fingerprint_sha256'] = protocol._json_hash(context)
    return context, manifest


def _validate_data(data, uid, y, metadata):
    if set(data) != {'logits', 'feats', 'y', 'sample_uid'}:
        raise RuntimeError('teacher cache array keys mismatch')
    if (data['y'].dtype.kind not in 'iu' or data['sample_uid'].dtype.kind not in 'iu'
            or not np.array_equal(data['sample_uid'], uid) or not np.array_equal(data['y'], y)):
        raise RuntimeError('teacher cache sample UIDs or labels mismatch')
    for name, shape in (('logits', (len(y), metadata['num_classes'])),
                        ('feats', (len(y), metadata['feature_dim']))):
        if data[name].shape != shape or data[name].dtype.kind != 'f' or not np.isfinite(data[name]).all():
            raise RuntimeError(f'teacher cache {name} dimensions/dtype/finiteness mismatch')


def _load_cache_unlocked(teacher, dataset, fold, seed, expected_uid, expected_y):
    _, cp, mp = cache_paths(teacher, dataset, fold, seed)
    if not cp.is_file() or not mp.is_file():
        raise FileNotFoundError(f'teacher cache is not complete: {cp.parent}')
    spec, snapshot = protocol._load_spec()
    uid, y = _expected_arrays(expected_uid, expected_y, fold, len(spec['datasets'][dataset]['classes']))
    context, _ = _teacher_context(teacher, dataset, fold, seed, uid, y, spec, snapshot)
    metadata = json.loads(mp.read_text())
    for key, value in context.items():
        if metadata.get(key) != value:
            raise RuntimeError(f'{mp}: cache provenance mismatch: {key}')
    if metadata.get('cache_file_sha256') != _sha256(cp):
        raise RuntimeError(f'{cp}: cache file hash mismatch')
    with np.load(cp, allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    _validate_data(data, uid, y, metadata)
    return data, metadata


def load_cache(teacher, dataset, fold, seed, expected_uid, expected_y):
    """Return (arrays, metadata), rejecting changed/incomplete/misaligned caches."""
    cell, cp, mp = cache_paths(teacher, dataset, fold, seed)
    if not cell.is_dir():
        raise FileNotFoundError(f'teacher cache directory is missing: {cell}')
    with _lock(cell, exclusive=False):
        return _load_cache_unlocked(teacher, dataset, fold, seed, expected_uid, expected_y)


def _atomic_save(cell, data, metadata):
    cp, mp = cell / 'cache.npz', cell / 'cache_manifest.json'
    temporary_paths = []
    try:
        with tempfile.NamedTemporaryFile(dir=cell, prefix='cache.', suffix='.npz.tmp', delete=False) as stream:
            temporary_paths.append(Path(stream.name))
            np.savez_compressed(stream, **data)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = dict(metadata, cache_file_sha256=_sha256(temporary_paths[-1]))
        os.replace(temporary_paths[-1], cp)
        with tempfile.NamedTemporaryFile(mode='w', dir=cell, prefix='manifest.', suffix='.json.tmp', delete=False) as stream:
            temporary_paths.append(Path(stream.name))
            json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_paths[-1], mp)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


@torch.inference_mode()
def _infer_preprocessed(adapter, model, x):
    feats, logits = [], []
    bs = int(adapter.cfg['batch_size'])
    model.eval()
    for start in range(0, len(x), bs):
        feat, logit = adapter.forward(model, x[start:start + bs].to(adapter.device))
        feats.append(feat.cpu().numpy().astype(np.float32, copy=False))
        logits.append(logit.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(feats), np.concatenate(logits)


def export(args):
    spec, snapshot = protocol._load_spec()
    device = f'cuda:{args.gpu}' if args.gpu is not None else 'cpu'
    if args.gpu is not None and not torch.cuda.is_available():
        raise RuntimeError('--gpu specified but CUDA is unavailable')
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '4')))
    for dataset in args.datasets:
        x, y, subjects, uids, _, _, _, _ = protocol._load_trials(dataset, spec)
        folds = list(range(spec['datasets'][dataset]['subjects'])) if args.folds is None else args.folds
        prepared = None
        preprocessing_config_hash = None
        for fold in folds:
            train = subjects != fold
            uid_train, y_train = _expected_arrays(uids[train], y[train], fold, len(spec['datasets'][dataset]['classes']))
            for seed in args.seeds:
                _validate_request(args.teacher, dataset, fold, seed, spec)
                cell, cp, mp = cache_paths(args.teacher, dataset, fold, seed)
                with _lock(cell, exclusive=True):
                    if cp.is_file() or mp.is_file():
                        try:
                            cached, _ = _load_cache_unlocked(args.teacher, dataset, fold, seed, uid_train, y_train)
                            del cached
                            print(f'[cache-skip] {args.teacher} {dataset} S{fold+1} seed={seed}', flush=True)
                            continue
                        except (FileNotFoundError, RuntimeError, ValueError, OSError, KeyError) as exc:
                            print(f'[cache-rebuild] {cell}: {exc}', flush=True)
                    metadata, teacher_manifest = _teacher_context(args.teacher, dataset, fold, seed,
                                                                 uid_train, y_train, spec, snapshot)
                    cfg = teacher_manifest['model_config']
                    if preprocessing_config_hash is not None and preprocessing_config_hash != metadata['model_config_sha256']:
                        raise RuntimeError(f'{dataset}: teachers do not share the same preprocessing configuration')
                    protocol._set_seed(seed)
                    adapter = protocol._make_adapter(args.teacher, cfg, device, PROTOCOL, dataset,
                                                     teacher_recipe(args.teacher, dataset, spec))
                    if prepared is None:
                        started = time.time()
                        if args.teacher == 'mirepnet':
                            prepared = adapter.preprocess(protocol._prepare_mirepnet(x, subjects, adapter))
                        else:
                            prepared = adapter.preprocess(x)
                        preprocessing_config_hash = metadata['model_config_sha256']
                        print(f'[preprocessed] {args.teacher} {dataset} shape={tuple(prepared.shape)} '
                              f'seconds={time.time()-started:.1f}', flush=True)
                    model = adapter.build(metadata['num_classes'])
                    state = torch.load(metadata['teacher_checkpoint']['path'], map_location='cpu', weights_only=True)
                    model.load_state_dict(state, strict=True)
                    del state
                    model.requires_grad_(False)
                    model.eval()
                    started = time.time()
                    feats, logits = _infer_preprocessed(adapter, model, prepared[train])
                    data = {'logits': logits, 'feats': feats, 'y': y_train, 'sample_uid': uid_train}
                    _validate_data(data, uid_train, y_train, metadata)
                    metadata.update(export_elapsed_sec=round(time.time()-started, 3),
                                    export_environment=protocol._environment_snapshot(device))
                    _atomic_save(cell, data, metadata)
                    print(f'[cache-done] {args.teacher} {dataset} S{fold+1} seed={seed} '
                          f'logits={logits.shape} feats={feats.shape}', flush=True)
                    del model, adapter, data, feats, logits
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        del x, y, subjects, uids, prepared
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', choices=TEACHERS, required=True)
    parser.add_argument('--datasets', nargs='+', choices=protocol.DATASET_NAMES,
                        default=list(protocol.DATASET_NAMES))
    parser.add_argument('--seeds', nargs='+', type=int, default=[666, 667, 668])
    parser.add_argument('--folds', nargs='+', type=int, default=None,
                        help='zero-based held-out subject IDs; omit for all')
    parser.add_argument('--gpu', type=int, default=None)
    args = parser.parse_args()
    for name in ('datasets', 'seeds', 'folds'):
        values = getattr(args, name)
        if values is not None and len(values) != len(set(values)):
            parser.error(f'--{name} contains duplicates')
    export(args)


if __name__ == '__main__':
    main()
