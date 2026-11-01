"""Read verified MIRepNet LOSO teachers and UID-aligned student inputs.

This module never trains a teacher and never replaces an existing input cache.
Only genuinely missing, registered model-input caches may be prepared under the
existing per-model lock.  Historical teacher training knobs come from manifests.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from data import loso_source
from experiments.storage import LOSO_CACHE_ROOT, resolve_local_file


DATASETS = ('BNCI2014004', 'BNCI2014001', 'BNCI2014001-4', 'BNCI2015001', 'AlexMI')
STUDENTS = ('ifnet', 'eegnet', 'adfcnn')
SUBJECTS = {'BNCI2014004': 9, 'BNCI2014001': 9, 'BNCI2014001-4': 9,
            'BNCI2015001': 12, 'AlexMI': 8}


def _array_sha(array) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(json.dumps(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _load_input(dataset: str, model: str, auto_prepare: bool):
    """Validate existing inputs; prepare a registered cache only if absent."""
    if dataset == 'AlexMI':
        return loso_source.load_dataset(dataset, model)
    from experiments.distill import loso_target_cache as cache

    folder = LOSO_CACHE_ROOT / 'model_inputs' / dataset / model
    if folder.exists() or folder.is_symlink():
        # A mismatch is an error, never permission to archive/rebuild live inputs.
        return loso_source.load_dataset(dataset, model)
    if auto_prepare and cache._has_registered_input_recipe(dataset, model):
        with cache._input_lock(dataset, model, LOSO_CACHE_ROOT):
            if not (folder.exists() or folder.is_symlink()):
                cache._prepare_registered_input(dataset, model, LOSO_CACHE_ROOT)
        cache._clear_dataset_memory_cache(dataset, model)
    # 001 student caches have no registered preparation: the source reader
    # returns the current task's source view, filtered once for EEGNet/ADFCNN.
    return loso_source.load_dataset(dataset, model)


def ensure_uid_alignment(teacher_data, student_data):
    """Return teacher rows reordered to the student's physical UID order."""
    if teacher_data.class_mapping != student_data.class_mapping:
        raise ValueError('teacher and student class mappings differ')
    if teacher_data.source_manifest_sha256 != student_data.source_manifest_sha256:
        raise ValueError('teacher and student source manifest identities differ')
    teacher_uids = np.asarray(teacher_data.trial_uid).astype(str)
    student_uids = np.asarray(student_data.trial_uid).astype(str)
    if (len(set(teacher_uids)) != len(teacher_uids)
            or len(set(student_uids)) != len(student_uids)
            or set(teacher_uids) != set(student_uids)):
        raise ValueError('teacher/student UIDs are duplicated or do not describe the same trials')
    if not np.array_equal(teacher_uids, student_uids):
        lookup = {uid: i for i, uid in enumerate(teacher_uids)}
        order = np.asarray([lookup[uid] for uid in student_uids], dtype=np.int64)
        teacher_data = replace(teacher_data, x=np.asarray(teacher_data.x[order]),
            y=np.asarray(teacher_data.y[order]), subjects=np.asarray(teacher_data.subjects[order]),
            trial_uid=teacher_uids[order], sample_uid=np.asarray(teacher_data.sample_uid[order]),
            identity={**teacher_data.identity, 'reindexed_by_physical_trial_uid': True})
    for name in ('y', 'subjects', 'sample_uid'):
        if not np.array_equal(getattr(teacher_data, name), getattr(student_data, name)):
            raise ValueError(f'teacher/student {name} differ after physical UID alignment')
    return teacher_data, student_data


def load_inputs(dataset: str, student: str, auto_prepare: bool = True):
    """Return model-ready teacher EEG and the student's standard input view.

    MIRepNet data are always [N,45,1000], including preserved AlexMI inputs.
    Call the student's own adapter.preprocess exactly once on student_data.x.
    EEGNet/ADFCNN adapters are identity for their already filtered source view;
    IFNet applies its configured filter bank in that single call.
    """
    if dataset not in DATASETS or student not in STUDENTS:
        raise ValueError(f'unsupported dataset/student: {dataset}/{student}')
    teacher_data = _load_input(dataset, 'mirepnet', auto_prepare)
    student_data = _load_input(dataset, student, auto_prepare)
    if dataset == 'AlexMI':
        # The historical LOSO teacher recipe owns this transform. No new filter
        # parameters or new model cache are introduced by the expansion.
        from experiments.finetune import run_loso_five_datasets as legacy
        from models import get_adapter

        adapter = get_adapter('mirepnet', device='cpu', dataset_name=dataset,
                              skip_preprocess=True)
        ready = legacy._prepare_mirepnet(
            teacher_data.x, teacher_data.subjects, adapter)
        teacher_data = replace(teacher_data, x=np.asarray(ready, dtype=np.float32),
            identity={**teacher_data.identity,
                'teacher_input_transform': 'historical _prepare_mirepnet: 8-30 Hz, per-subject EA, 45-channel mapping',
                'teacher_input_ready': True})
    if tuple(teacher_data.x.shape[1:]) != (45, 1000):
        raise ValueError(f'{dataset}: MIRepNet model inputs must be [N,45,1000], got {teacher_data.x.shape}')
    expected = np.arange(SUBJECTS[dataset])
    if not np.array_equal(np.unique(teacher_data.subjects), expected):
        raise ValueError(f'{dataset}: source subjects differ from the expected complete LOSO set')
    return ensure_uid_alignment(teacher_data, student_data)


def _pretrain_record(manifest: dict):
    recorded = manifest.get('pretrained_checkpoint')
    if isinstance(recorded, dict):
        path = recorded.get('path')
        expected = recorded.get('sha256')
    else:
        path = recorded
        expected = manifest.get('pretrained_sha256', manifest.get('reference_pretrained_sha256'))
    if not path or not expected:
        raise ValueError('teacher manifest lacks the pretrained checkpoint path/hash')
    resolved = Path(resolve_local_file(path))
    if loso_source.sha256_file(resolved) != expected:
        raise ValueError(f'canonical MIRepNet pretrained checkpoint hash changed: {resolved}')
    return resolved, expected


def _check_partition_counts(manifest: dict, data, fold: int):
    train = np.flatnonzero(data.subjects != fold)
    test = np.flatnonzero(data.subjects == fold)
    n_train = manifest.get('n_train', manifest.get('train_trials'))
    n_test = manifest.get('n_test', manifest.get('test_trials'))
    if int(n_train if n_train is not None else -1) != len(train):
        raise ValueError('teacher source-training trial count differs from standard LOSO split')
    if int(n_test if n_test is not None else -1) != len(test):
        raise ValueError('teacher held-out trial count differs from standard LOSO split')
    if not len(train) or not len(test) or set(data.trial_uid[train]) & set(data.trial_uid[test]):
        raise ValueError('teacher standard LOSO split is empty or has overlapping UIDs')
    return train, test


def teacher_record(dataset: str, seed: int, fold: int, teacher_data) -> dict:
    """Read an existing exact-fold teacher; fail on missing/invalid artifacts.

    No call to ensure_teacher_checkpoint is made. The source model is never
    retrained as a side effect of reading this record.
    """
    if dataset not in DATASETS or int(seed) not in (666, 667, 668):
        raise ValueError(f'unsupported teacher dataset/seed: {dataset}/{seed}')
    if not 0 <= int(fold) < SUBJECTS[dataset]:
        raise ValueError(f'{dataset}: fold out of range: {fold}')
    if dataset == 'AlexMI':
        from data import loso_alexmi as legacy

        manifest_path, manifest = legacy._find_baseline_manifest('mirepnet', seed, fold)
        (_, _, _, _, source, _, _, _) = legacy._validate_data(
            manifest, teacher_data, str(manifest_path))
        legacy._validate_split_manifest(manifest, manifest_path, 'mirepnet', seed, fold, source)
        # The preserved teacher-target metadata contains the fine-tuned
        # checkpoint SHA omitted from the original baseline manifest.
        targets = legacy.load_teacher_targets('mirepnet', seed, fold, teacher_data)
        checkpoint = Path(targets['_teacher_checkpoint_path'])
        expected_sha = targets['_teacher_checkpoint_sha256']
        archived_config = dict(manifest['model_config'])
        model_config = dict(archived_config)
        model_config.update(dataset_name=dataset, in_channels=45, samples=1000,
                            skip_preprocess=True)
        target_manifest_path = targets['_manifest_path']
    else:
        manifest_path = loso_source._baseline_teacher_manifest(dataset, 'mirepnet', seed, fold)
        manifest = loso_source._json(manifest_path)
        loso_source._validate_baseline_manifest(manifest, manifest_path, teacher_data,
            dataset, 'mirepnet', seed, fold, check_prediction_hash=False)
        if manifest.get('status') != 'complete':
            raise ValueError(f'teacher is not complete: {manifest_path}')
        if manifest.get('input_manifest_sha256') != teacher_data.input_manifest_sha256:
            raise ValueError('teacher input manifest hash differs from loaded canonical inputs')
        recorded_input_files = manifest.get('input_files') or {}
        for name, expected in teacher_data.input_manifest.get('files', {}).items():
            saved = recorded_input_files.get(name) or {}
            if any(saved.get(field) != expected.get(field) for field in ('bytes', 'sha256')):
                raise ValueError(f'teacher recorded input file differs from canonical bytes: {name}')
            if saved.get('path') and Path(resolve_local_file(saved['path'])) != Path(
                    resolve_local_file(teacher_data.input_manifest_path.parent / name)):
                raise ValueError(f'teacher input path does not resolve to its canonical file: {name}')
        recorded_input_manifest = recorded_input_files.get('manifest.json')
        if (recorded_input_manifest is not None and recorded_input_manifest.get('sha256')
                != teacher_data.input_manifest_sha256):
            raise ValueError('teacher input manifest file record differs from canonical manifest')
        checkpoint = manifest_path.parent / 'final_model.pt'
        expected_sha = manifest.get('final_model_sha256')
        target_manifest_path = None
        if dataset in ('BNCI2014001', 'BNCI2014001-4'):
            archived_config = dict(manifest['resolved_training_config'])
            # Archived 001 result.json identifies the fixed 256-D, depth-6
            # MIRepNet adapter recipe via source hashes and strict state load.
            if manifest.get('profile') != 'wideband_npy_v3':
                raise ValueError('001 teacher does not belong to the current wideband source profile')
            model_config = {**archived_config, 'emb_size': 256, 'depth': 6,
                'dataset_name': dataset, 'in_channels': 45, 'samples': 1000,
                'skip_preprocess': True}
            prediction_path = manifest_path.parent / 'test_predictions.npz'
            if (not prediction_path.is_file() or loso_source.sha256_file(prediction_path)
                    != manifest.get('test_predictions_sha256')):
                raise ValueError('001 teacher prediction artifact/hash is missing or changed')
            with np.load(prediction_path, allow_pickle=False) as saved:
                expected_rows = np.flatnonzero(teacher_data.subjects == fold)
                if not np.array_equal(saved['row_ids'], expected_rows):
                    raise ValueError('001 baseline target row IDs differ from the current source split')
                if not np.array_equal(saved['labels'], teacher_data.y[expected_rows]):
                    raise ValueError('001 baseline target labels differ from canonical UID labels')
        else:
            archived_config = dict(manifest['model_config'])
            model_config = dict(archived_config)
            expected_subjects = set(np.unique(teacher_data.subjects).tolist()) - {fold}
            if set(int(s) - 1 for s in manifest.get('train_subjects', [])) != expected_subjects:
                raise ValueError('teacher training subject IDs differ from standard LOSO')
            expected_session = teacher_data.input_manifest.get('selected_session')
            if manifest.get('selected_session') != expected_session:
                raise ValueError('teacher selected session differs from canonical input recipe')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'completed MIRepNet checkpoint is missing; refusing implicit retraining: {checkpoint}')
    actual_sha = loso_source.sha256_file(checkpoint)
    if not expected_sha or actual_sha != str(expected_sha):
        raise ValueError(f'completed MIRepNet checkpoint hash differs from provenance: {checkpoint}')
    pretrain, pretrain_sha = _pretrain_record(manifest)
    model_config.update(pretrain=str(pretrain), dataset_name=dataset,
                        in_channels=45, samples=1000, skip_preprocess=True)
    train, test = _check_partition_counts(manifest, teacher_data, int(fold))
    return {
        'dataset': dataset, 'seed': int(seed), 'fold': int(fold),
        'checkpoint': checkpoint, 'checkpoint_sha256': actual_sha,
        'manifest_path': Path(manifest_path),
        'manifest_sha256': loso_source.sha256_file(manifest_path), 'manifest': manifest,
        'model_config': model_config, 'archived_model_config': archived_config,
        'pretrained_checkpoint': str(pretrain), 'pretrained_checkpoint_sha256': pretrain_sha,
        'source_manifest_sha256': teacher_data.source_manifest_sha256,
        'input_manifest_sha256': teacher_data.input_manifest_sha256,
        'train_indices': train, 'test_indices': test,
        'train_uid_sha256': _array_sha(np.asarray(teacher_data.trial_uid[train]).astype(str)),
        'test_uid_sha256': _array_sha(np.asarray(teacher_data.trial_uid[test]).astype(str)),
        'train_subjects': sorted(set(teacher_data.subjects[train].astype(int).tolist())),
        'num_classes': len(teacher_data.class_mapping), 'class_mapping': teacher_data.class_mapping,
        'legacy_target_manifest': target_manifest_path,
        'training_protocol': 'all N-1 non-target subjects; no feedback subject',
    }


def build_teacher(record: dict, data, device):
    """Build from archived architecture/config and strict-load the verified state."""
    import torch
    from models import get_adapter

    checkpoint = Path(record['checkpoint'])
    if loso_source.sha256_file(checkpoint) != record['checkpoint_sha256']:
        raise ValueError('teacher checkpoint changed between audit and model construction')
    if tuple(data.x.shape[1:]) != (45, 1000):
        raise ValueError('MIRepNet teacher requires canonical prepared [N,45,1000] input')
    adapter = get_adapter('mirepnet', device=device, **record['model_config'])
    model = adapter.build(int(record['num_classes']))
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = payload.get('model', payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f'expected a state_dict checkpoint: {checkpoint}')
    model.load_state_dict(state, strict=True)
    model.eval()
    return adapter, model
