#!/usr/bin/env python
"""Export reusable broadband MOABB sources for BNCI2014004 and BNCI2015001.

The existing flat NPY caches stay untouched.  This writes a versioned,
all-session source into each shared dataset directory, verifies every exported
trial against its legacy NPY row, and keeps model-specific transformations out
of the shared source.

Run with the pinned MOABB overlay used by the other source exporters:

  PYTHONPATH=/tmp/loso_alignment_deps_moabb \
    /home/lixinli/anaconda3/envs/cbramod/bin/python \
    experiments/finetune/export_bnci14004_15001_all_sessions.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.storage import DATA_CACHE_ROOT, require_external_output, resolve_local_file

DATA_ROOT = Path('/data1/llx')
OLD_RAW_ROOT = DATA_CACHE_ROOT / 'eegfm_alignment_v2/mne_data/MNE-bnci-data/database/data-sets'
MOABB_BASE = Path('MNE-bnci-data/database/data-sets')

DATASETS = {
    'BNCI2014004': {
        'moabb_class': 'BNCI2014_004',
        'raw_folder': '004-2014',
        'raw_source_folder': '004-2014',
        'source_filter_hz': [0.0, 120.0],
        'fs_hz': 250,
        'channels': 3,
        'expected_trials': 6520,
        'label_map': {'left_hand': 0, 'right_hand': 1},
        'reference_label_map': {'left_hand': 0, 'right_hand': 1},
        'source_variant': 'broadband_0_120hz',
        'expected_subjects': 9,
    },
    'BNCI2015001': {
        'moabb_class': 'BNCI2015_001',
        'raw_folder': '001-2015',
        'raw_source_folder': '001-2015',
        'source_filter_hz': [0.1, 75.0],
        'fs_hz': 512,
        'channels': 13,
        'expected_trials': 5600,
        # Preserve the task labels used by the existing LOSO results.
        'label_map': {'feet': 0, 'right_hand': 1},
        'reference_label_map': {'right_hand': 0, 'feet': 1},
        'source_variant': 'broadband_0p1_75hz',
        'expected_subjects': 12,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def subject_id(value: object) -> int:
    match = re.search(r'(\d+)$', str(value))
    if match is None:
        raise ValueError(f'cannot parse subject ID: {value!r}')
    return int(match.group(1))


def session_index(dataset_name: str, value: object, legacy: bool) -> int:
    text = str(value)
    if legacy:
        if dataset_name == 'BNCI2014004':
            match = re.fullmatch(r'session_(\d+)', text)
        else:
            match = re.fullmatch(r'session_([ABC])', text)
            if match:
                return ord(match.group(1)) - ord('A')
    else:
        if dataset_name == 'BNCI2014004':
            match = re.match(r'^(\d+)(?:train|test)$', text)
        else:
            match = re.match(r'^(\d+)[ABC]$', text)
    if match is None:
        raise ValueError(f'{dataset_name}: cannot normalize session {text!r} legacy={legacy}')
    return int(match.group(1))


def run_index(value: object) -> str:
    return re.sub(r'^run[_-]?', '', str(value))


def source_session_role(dataset_name: str, session: str) -> str:
    if dataset_name == 'BNCI2014004':
        return 'T' if session.endswith('train') else 'E'
    match = re.search(r'([ABC])$', session)
    if match is None:
        raise ValueError(f'cannot map raw session role: {session}')
    return match.group(1)


def raw_candidates(dataset_name: str, subject: int) -> dict[str, Path]:
    cfg = DATASETS[dataset_name]
    source_dir = OLD_RAW_ROOT / cfg['raw_source_folder']
    if not source_dir.is_dir():
        raise FileNotFoundError(f'local MOABB raw directory not found: {source_dir}')
    prefix = f'B{subject:02d}' if dataset_name == 'BNCI2014004' else f'S{subject:02d}'
    out = {}
    for role in (('T', 'E') if dataset_name == 'BNCI2014004' else ('A', 'B', 'C')):
        old_path = source_dir / f'{prefix}{role}.mat'
        if old_path.exists():
            out[role] = resolve_local_file(old_path)
    if not out:
        raise FileNotFoundError(f'{dataset_name} subject {subject}: no local raw MAT files')
    return out


def hydrate_raw(dataset_name: str, output_dir: Path) -> tuple[Path, list[dict]]:
    cfg = DATASETS[dataset_name]
    raw_root = require_external_output(output_dir / 'raw/mne_data')
    raw_folder = raw_root / MOABB_BASE / cfg['raw_folder']
    raw_folder.mkdir(parents=True, exist_ok=True)
    records = []
    for subject in range(1, cfg['expected_subjects'] + 1):
        for role, resolved in raw_candidates(dataset_name, subject).items():
            name = (f'B{subject:02d}{role}.mat' if dataset_name == 'BNCI2014004'
                    else f'S{subject:02d}{role}.mat')
            target = raw_folder / name
            digest = sha256(resolved)
            if target.exists():
                if sha256(target) != digest:
                    raise RuntimeError(f'refusing to overwrite changed raw file: {target}')
            else:
                partial = target.with_name(target.name + '.partial')
                shutil.copyfile(resolved, partial)
                os.replace(partial, target)
            records.append({
                'subject': subject, 'session_role': role,
                'path': str(target), 'sha256': digest,
                'bytes': target.stat().st_size,
                'reused_local_object': str(resolved),
            })
    return raw_root, records


def verify_and_map(dataset_name: str, source_meta: pd.DataFrame,
                   source_labels: np.ndarray) -> tuple[pd.DataFrame, dict]:
    cfg = DATASETS[dataset_name]
    legacy_dir = DATA_ROOT / dataset_name
    legacy_meta_name = 'meta004.csv' if dataset_name == 'BNCI2014004' else 'meta.csv'
    legacy_meta = pd.read_csv(resolve_local_file(legacy_dir / legacy_meta_name))
    legacy_labels = np.load(resolve_local_file(legacy_dir / 'labels.npy'), allow_pickle=True).astype(str)
    legacy_x = np.load(resolve_local_file(legacy_dir / 'X.npy'), mmap_mode='r')
    if len(legacy_meta) != cfg['expected_trials'] or len(legacy_labels) != len(legacy_meta):
        raise RuntimeError(f'{dataset_name}: legacy metadata/labels have unexpected row counts')
    if len(source_meta) != cfg['expected_trials'] or len(source_labels) != len(source_meta):
        raise RuntimeError(f'{dataset_name}: MOABB source has unexpected trial count')
    legacy_meta = legacy_meta.copy()
    legacy_meta['legacy_raw_row'] = np.arange(len(legacy_meta), dtype=np.int64)
    legacy_meta['class_name'] = legacy_labels
    legacy_meta['session_index'] = legacy_meta.session.map(
        lambda value: session_index(dataset_name, value, True))
    legacy_meta['run_key'] = legacy_meta.run.map(run_index)
    legacy_meta['event_ordinal'] = legacy_meta.groupby(
        ['subject', 'session', 'run'], sort=False).cumcount()
    new_meta = source_meta.copy()
    new_meta['source_row'] = np.arange(len(new_meta), dtype=np.int64)
    new_meta['class_name'] = np.asarray(source_labels, dtype=str)
    new_meta['subject_id'] = new_meta.subject.map(subject_id)
    new_meta['session_index'] = new_meta.session.map(
        lambda value: session_index(dataset_name, value, False))
    new_meta['run_key'] = new_meta.run.map(run_index)
    new_meta['event_ordinal'] = new_meta.groupby(
        ['subject', 'session', 'run'], sort=False).cumcount()

    def make_key(row, source=False):
        subject = int(row.subject_id if source else row.subject)
        return (subject, int(row.session_index), str(row.run_key), int(row.event_ordinal))

    legacy = {}
    for row in legacy_meta.itertuples(index=False):
        key = make_key(row)
        if key in legacy:
            raise RuntimeError(f'{dataset_name}: duplicate legacy trial key {key}')
        legacy[key] = (int(row.legacy_raw_row), str(row.class_name))
    rebuilt = {}
    for row in new_meta.itertuples(index=False):
        key = make_key(row, source=True)
        if key in rebuilt:
            raise RuntimeError(f'{dataset_name}: duplicate MOABB trial key {key}')
        rebuilt[key] = (int(row.source_row), str(row.class_name), str(row.session))

    mismatches = []
    mapping = []
    for key, (legacy_row, legacy_label) in legacy.items():
        item = rebuilt.get(key)
        if item is None:
            mismatches.append({'key': key, 'reason': 'missing_moabb_trial'})
            continue
        source_row, source_label, source_session = item
        if source_label != legacy_label:
            mismatches.append({'key': key, 'reason': 'label_mismatch',
                               'legacy': legacy_label, 'moabb': source_label})
        mapping.append({
            'dataset': dataset_name, 'legacy_raw_row': legacy_row,
            'source_row': source_row, 'subject': key[0],
            'legacy_session_index': key[1], 'session_key': source_session,
            'run_key': key[2], 'event_ordinal': key[3],
            'class_name': legacy_label,
            'label_id': int(cfg['label_map'][legacy_label]),
            'paired_verified': source_label == legacy_label,
        })
    for key in sorted(set(rebuilt) - set(legacy)):
        mismatches.append({'key': key, 'reason': 'unexpected_moabb_trial'})
    report = {
        'legacy_trials': len(legacy), 'moabb_trials': len(rebuilt),
        'mapped_trials': len(mapping), 'mismatch_count': len(mismatches),
        'one_to_one': len(legacy) == len(rebuilt) == len(mapping),
        'all_labels_match': not any(row['reason'] == 'label_mismatch' for row in mismatches),
        'mismatches_preview': mismatches[:20],
        'legacy_array_shape': list(legacy_x.shape),
    }
    if not report['one_to_one'] or report['mismatch_count'] or not report['all_labels_match']:
        raise RuntimeError(f'{dataset_name}: legacy/MOABB mapping failed: {json.dumps(report, sort_keys=True)}')
    return pd.DataFrame(mapping).sort_values('source_row').reset_index(drop=True), report


def build(dataset_name: str) -> None:
    cfg = DATASETS[dataset_name]
    variant_dir = require_external_output(DATA_ROOT / dataset_name / cfg['source_variant'])
    variant_dir.mkdir(parents=True, exist_ok=True)
    raw_root, raw_files = hydrate_raw(dataset_name, variant_dir)
    os.environ['MNE_DATA'] = str(raw_root)
    os.environ['MNE_DATASETS_BNCI_PATH'] = str(raw_root)

    import mne
    import moabb
    from moabb.paradigms import MotorImagery
    from moabb import datasets

    if moabb.__version__ != '1.2.0' or mne.__version__ != '1.11.0':
        raise RuntimeError(
            f'Export requires MOABB 1.2.0 / MNE 1.11.0; got '
            f'{moabb.__version__} / {mne.__version__}')
    moabb.set_log_level('ERROR')
    mne.set_log_level('ERROR')
    dataset = getattr(datasets, cfg['moabb_class'])()
    subjects = list(dataset.subject_list)
    if len(subjects) != cfg['expected_subjects']:
        raise RuntimeError(f'{dataset_name}: unexpected subject list {subjects}')
    for subject in subjects:
        paths = [Path(path) for path in dataset.data_path(subject=subject)]
        if not paths or any(not path.is_file() for path in paths):
            raise RuntimeError(f'{dataset_name} subject {subject}: MOABB raw files unavailable: {paths}')

    paradigm = MotorImagery(fmin=cfg['source_filter_hz'][0],
                            fmax=cfg['source_filter_hz'][1],
                            tmin=0.0, tmax=None, baseline=None, resample=None)
    x_all, labels_all, meta_all = paradigm.get_data(dataset=dataset, subjects=subjects)
    meta = meta_all.reset_index(drop=True).copy()
    labels = np.asarray(labels_all, dtype=str)
    x = np.asarray(x_all, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != cfg['channels'] or len(x) != cfg['expected_trials']:
        raise RuntimeError(f'{dataset_name}: unexpected exported arrays {x.shape}')
    if len(labels) != len(meta):
        raise RuntimeError(f'{dataset_name}: MOABB array/metadata mismatch')
    unknown = sorted(set(labels) - set(cfg['label_map']))
    if unknown:
        raise RuntimeError(f'{dataset_name}: unexpected classes {unknown}')
    y = np.asarray([cfg['label_map'][label] for label in labels], dtype=np.int64)
    mapping, pairing_report = verify_and_map(dataset_name, meta, labels)
    legacy_row_by_source = dict(zip(
        mapping.source_row.astype(int), mapping.legacy_raw_row.astype(int)))

    # Build stable raw-file identities into every UID.
    subject_raw = {}
    for subject in subjects:
        subject_no = subject_id(subject)
        subject_raw[subject_no] = {
            record['session_role']: record
            for record in raw_files if record['subject'] == subject_no
        }
    meta['source_row'] = np.arange(len(meta), dtype=np.int64)
    meta['class_name'] = labels
    meta['subject_id'] = meta.subject.map(subject_id)
    meta['event_ordinal'] = meta.groupby(['subject', 'session', 'run'], sort=False).cumcount()
    meta['label_id'] = y
    trials = []
    for row in meta.itertuples(index=False):
        role = source_session_role(dataset_name, str(row.session))
        raw = subject_raw[int(row.subject_id)].get(role)
        if raw is None:
            raise RuntimeError(f'{dataset_name} S{row.subject_id} session {row.session}: raw role {role} missing')
        run = run_index(row.run)
        uid = (f'{dataset_name}|{raw["sha256"]}|sub-{int(row.subject_id):02d}'
               f'|session-{row.session}|run-{run}|event-{int(row.event_ordinal):04d}'
               f'|class-{row.class_name}')
        trials.append({
            'dataset': dataset_name, 'source_row': int(row.source_row),
            'trial_uid': uid, 'subject': int(row.subject_id),
            'session': str(row.session), 'session_index': session_index(dataset_name, row.session, False),
            'run': str(row.run), 'run_key': run,
            'event_ordinal': int(row.event_ordinal), 'class_name': str(row.class_name),
            'label_id': int(row.label_id), 'native_fs_hz': cfg['fs_hz'],
            'n_channels': cfg['channels'], 'n_samples': int(x.shape[-1]),
            'epoch_tmin_s': float(paradigm.tmin),
            'epoch_tmax_s': None if paradigm.tmax is None else float(paradigm.tmax),
            'source_filter_low_hz': cfg['source_filter_hz'][0],
            'source_filter_high_hz': cfg['source_filter_hz'][1],
            'raw_file': raw['path'], 'raw_file_sha256': raw['sha256'],
            'legacy_raw_row': legacy_row_by_source[int(row.source_row)],
        })
    trials_df = pd.DataFrame(trials)
    if trials_df.trial_uid.duplicated().any():
        raise RuntimeError(f'{dataset_name}: generated trial_uid collision')
    if not np.array_equal(trials_df.label_id.to_numpy(), y):
        raise RuntimeError(f'{dataset_name}: metadata label IDs differ from y array')

    manifest_path = variant_dir / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        old_hashes = old.get('files', {})
        if old.get('source_filter_hz') != cfg['source_filter_hz'] or not old.get('all_sessions'):
            raise RuntimeError(f'refusing to overwrite an incompatible source variant: {variant_dir}')
        for name, record in old_hashes.items():
            path = variant_dir / name
            if not path.is_file() or sha256(path) != record['sha256']:
                raise RuntimeError(f'existing source manifest is invalid: {path}')
        print(f'[skip-valid] {dataset_name} {variant_dir}', flush=True)
        return

    meta['trial_uid'] = trials_df.trial_uid
    meta['source_row'] = np.arange(len(meta), dtype=np.int64)
    meta['native_fs_hz'] = cfg['fs_hz']
    meta['label_id'] = y
    x_path = variant_dir / 'X.npy.partial'
    with x_path.open('wb') as stream:
        np.save(stream, x, allow_pickle=False)
    os.replace(x_path, variant_dir / 'X.npy')
    np.save(variant_dir / 'labels.npy', labels.astype('U32'), allow_pickle=False)
    np.save(variant_dir / 'y.npy', y, allow_pickle=False)
    meta.to_csv(variant_dir / 'meta.csv', index=False)
    trials_df.to_csv(variant_dir / 'trials.csv', index=False)
    mapping.to_csv(variant_dir / 'legacy_row_mapping.csv', index=False)
    manifest = {
        'dataset': dataset_name, 'source_variant': cfg['source_variant'],
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'all_sessions': True, 'source_filter_hz': cfg['source_filter_hz'],
        'source_filter_description': 'MOABB MotorImagery epoch-level filter',
        'moabb_version': moabb.__version__,
        'mne_version': __import__('mne').__version__,
        'paradigm': 'MotorImagery', 'paradigm_tmin_s': float(paradigm.tmin),
        'paradigm_tmax_s': None if paradigm.tmax is None else float(paradigm.tmax),
        'native_fs_hz': cfg['fs_hz'], 'channels': cfg['channels'],
        'shape': list(x.shape), 'dtype': str(x.dtype),
        'unit': 'MOABB MotorImagery output as returned; expected microvolts',
        'class_map': cfg['label_map'],
        'reference_class_map': cfg['reference_label_map'],
        'sessions': sorted(meta.session.astype(str).unique().tolist()),
        'subject_trial_counts': meta.groupby('subject_id').size().astype(int).to_dict(),
        'raw_files': raw_files, 'legacy_pairing': pairing_report,
        'export_script_sha256': sha256(Path(__file__)),
        'files': {},
    }
    for name in ('X.npy', 'labels.npy', 'y.npy', 'meta.csv', 'trials.csv', 'legacy_row_mapping.csv'):
        file_path = variant_dir / name
        manifest['files'][name] = {
            'bytes': file_path.stat().st_size, 'sha256': sha256(file_path),
        }
    tmp = manifest_path.with_suffix('.json.partial')
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, manifest_path)
    print(f'[built] {dataset_name} shape={x.shape} sessions={manifest["sessions"]} '
          f'paired={pairing_report["mapped_trials"]}/{pairing_report["legacy_trials"]} '
          f'mismatches={pairing_report["mismatch_count"]}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=tuple(DATASETS),
                        default=list(DATASETS))
    args = parser.parse_args()
    for name in args.datasets:
        build(name)


if __name__ == '__main__':
    main()
