#!/usr/bin/env python
"""Rebuild the three EEG-FM-Benchmark MI source caches with MOABB.

Run with the isolated MOABB 1.2 overlay and a writable MNE_DATA directory:

  PYTHONPATH=/tmp/loso_alignment_deps_moabb \
  MNE_DATA=/data1/llx/BigSmallcollab/cache/eegfm_alignment_v2/mne_data \
  MNE_DATASETS_BNCI_PATH=/data1/llx/BigSmallcollab/cache/eegfm_alignment_v2/mne_data \
  conda run -n cbramod python experiments/finetune/build_eegfm_reference_source.py

This script only builds reference-source arrays and a verified row mapping. It
does not train models or alter the legacy .npy files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.storage import (DATA_CACHE_ROOT, RESULTS_ROOT,
                                 require_external_output, resolve_local_file)
DATA_ROOT = Path('/data1/llx')
OUTPUT_ROOT = DATA_CACHE_ROOT / 'eegfm_alignment_v2/rebuilt'
MNE_ROOT = DATA_CACHE_ROOT / 'eegfm_alignment_v2/mne_data'

DATASETS = {
    'BNCI2014001-4': {
        'source_name': 'BNCI2014001', 'moabb_name': 'BNCI2014_001',
        'session': '0train', 'legacy_session': 'session_T',
        'fmin': 0.1, 'fmax': 75.0, 'fs': 250, 'channels': 22,
        'label_map': {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3},
        'expected_subjects': 9, 'expected_trials': 2592,
        'expected_per_subject': 288, 'source_samples': 1000,
    },
    'BNCI2014004': {
        'source_name': 'BNCI2014004', 'moabb_name': 'BNCI2014_004',
        'session': '3test', 'legacy_session': 'session_3',
        'fmin': 0.0, 'fmax': 120.0, 'fs': 250, 'channels': 3,
        'label_map': {'left_hand': 0, 'right_hand': 1},
        'expected_subjects': 9, 'expected_trials': 1400,
        'expected_per_subject': None, 'source_samples': None,
    },
    'BNCI2015001': {
        'source_name': 'BNCI2015001', 'moabb_name': 'BNCI2015_001',
        'session': '0A', 'legacy_session': 'session_A',
        'fmin': 0.1, 'fmax': 75.0, 'fs': 512, 'channels': 13,
        'label_map': {'right_hand': 0, 'feet': 1},
        'expected_subjects': 12, 'expected_trials': 2400,
        'expected_per_subject': 200, 'source_samples': None,
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _canonical_session(value: object, legacy: bool) -> str:
    value = str(value)
    if legacy:
        return {'session_T': 'train', 'session_A': 'train',
                'session_3': 'test'}.get(value, value)
    if value.endswith(('train', 'A')):
        return 'train'
    if value.endswith(('test', 'B', 'C')):
        return 'test'
    return value


def _canonical_run(value: object) -> str:
    return re.sub(r'^run[_-]?', '', str(value))


def _raw_identity(path_list: list[str], subject: int, session: str,
                  dataset_name: str) -> tuple[str, str]:
    if dataset_name == 'BNCI2015001':
        # BNCI2015-001 stores its two sessions as A/B files rather than T/E.
        role = 'A' if session.endswith(('train', 'A')) else 'B'
    else:
        role = 'T' if session.endswith('train') else 'E'
    candidates = [Path(p) for p in path_list if Path(p).stem.upper().endswith(role)]
    if len(candidates) != 1:
        raise RuntimeError(f'subject={subject} session={session}: cannot map to one raw {role} file: {path_list}')
    path = candidates[0]
    return str(path), sha256(path)


def _get_moabb_dataset(name: str):
    from moabb import datasets
    return getattr(datasets, name)()


def _verify_legacy_mapping(dataset_name: str, new_meta: pd.DataFrame,
                           trial_uids: list[str], source_cfg: dict) -> tuple[pd.DataFrame, dict]:
    source = source_cfg['source_name']
    old_dir = DATA_ROOT / source
    meta_file = 'meta004.csv' if source == 'BNCI2014004' else 'meta.csv'
    old_meta = pd.read_csv(old_dir / meta_file)
    old_labels = np.load(old_dir / 'labels.npy', allow_pickle=True).astype(str)
    if len(old_meta) != len(old_labels):
        raise RuntimeError(f'{source}: legacy metadata/label count differs')
    old_meta = old_meta.copy()
    old_meta['raw_row'] = (old_meta['Unnamed: 0'].astype(int)
                           if 'Unnamed: 0' in old_meta else np.arange(len(old_meta)))
    old_meta['class_name'] = old_labels
    old_meta['canon_session'] = old_meta['session'].map(lambda v: _canonical_session(v, True))
    old_meta['canon_run'] = old_meta['run'].map(_canonical_run)
    selected = old_meta['session'].astype(str).eq(source_cfg['legacy_session'])
    old_meta = old_meta.loc[selected].copy()
    old_meta['event_ordinal'] = old_meta.groupby(
        ['subject', 'session', 'run'], sort=False).cumcount()

    new_meta = new_meta.copy()
    new_meta['canon_session'] = new_meta['session'].map(lambda v: _canonical_session(v, False))
    new_meta['canon_run'] = new_meta['run'].map(_canonical_run)
    new_meta['event_ordinal'] = new_meta.groupby(
        ['subject', 'session', 'run'], sort=False).cumcount()

    key_fields = ['subject', 'canon_session', 'canon_run', 'event_ordinal']
    old_rows = {}
    for row in old_meta.itertuples(index=False):
        key = (int(row.subject), str(row.canon_session), str(row.canon_run), int(row.event_ordinal))
        old_rows[key] = (int(row.raw_row), str(row.class_name))
    new_rows = {}
    for idx, row in enumerate(new_meta.itertuples(index=False)):
        key = (int(row.subject), str(row.canon_session), str(row.canon_run), int(row.event_ordinal))
        new_rows[key] = (idx, str(row.class_name), trial_uids[idx])

    mapped = []
    mismatches = []
    for key, (old_row, old_label) in old_rows.items():
        item = new_rows.get(key)
        if item is None:
            mismatches.append({'key': key, 'reason': 'missing_rebuilt_event'})
            continue
        new_row, new_label, uid = item
        label_ok = old_label == new_label
        if not label_ok:
            mismatches.append({'key': key, 'reason': 'class_mismatch',
                               'legacy': old_label, 'rebuilt': new_label})
        mapped.append({'dataset': dataset_name, 'legacy_raw_row': old_row,
                       'rebuilt_row': new_row, 'trial_uid': uid,
                       'subject': key[0], 'session_key': key[1],
                       'run_key': key[2], 'event_ordinal': key[3],
                       'class_name': old_label, 'paired_verified': label_ok})
    extra_keys = set(new_rows) - set(old_rows)
    for key in sorted(extra_keys):
        mismatches.append({'key': key, 'reason': 'unexpected_rebuilt_event'})
    report = {
        'legacy_selected_trials': len(old_rows), 'rebuilt_selected_trials': len(new_rows),
        'mapped_trials': len(mapped), 'mismatch_count': len(mismatches),
        'one_to_one': len(old_rows) == len(new_rows) == len(mapped),
        'all_labels_match': not any(m['reason'] == 'class_mismatch' for m in mismatches),
        'mismatches_preview': mismatches[:25],
    }
    return pd.DataFrame(mapped), report


def build_one(dataset_name: str, cfg: dict, download_retries: int = 5) -> None:
    import moabb
    from moabb.paradigms import MotorImagery

    dataset = _get_moabb_dataset(cfg['moabb_name'])
    subjects = list(dataset.subject_list)
    if len(subjects) != cfg['expected_subjects']:
        raise RuntimeError(f'{dataset_name}: expected {cfg["expected_subjects"]} subjects, got {subjects}')

    raw_by_subject: dict[int, list[str]] = {}
    raw_manifest = []
    for subject in subjects:
        paths = None
        for attempt in range(download_retries + 1):
            try:
                paths = [str(Path(p).resolve()) for p in dataset.data_path(subject=subject)]
                break
            except Exception as exc:
                if attempt >= download_retries:
                    raise
                wait_s = min(60, 3 * (2 ** attempt))
                print(f'[download-retry] {dataset_name} subject={subject} '
                      f'attempt={attempt + 1}/{download_retries} '
                      f'error={type(exc).__name__}: {exc}; sleeping={wait_s}s', flush=True)
                time.sleep(wait_s)
        if paths is None:
            raise RuntimeError(f'{dataset_name} subject {subject}: download retries exhausted')
        if not paths or any(not Path(p).is_file() for p in paths):
            raise RuntimeError(f'{dataset_name} subject {subject}: raw files are not present: {paths}')
        raw_by_subject[int(subject)] = paths
        for path in paths:
            raw_manifest.append({'subject': int(subject), 'path': path,
                                 'bytes': Path(path).stat().st_size,
                                 'sha256': sha256(Path(path))})
        print(f'[raw-ready] {dataset_name} subject={subject} files={len(paths)}', flush=True)

    paradigm = MotorImagery(fmin=cfg['fmin'], fmax=cfg['fmax'])
    x_all, labels_all, meta_all = paradigm.get_data(dataset, subjects=subjects)
    meta_all = meta_all.reset_index(drop=True).copy()
    meta_all['class_name'] = np.asarray(labels_all, dtype=str)
    meta_all['run'] = meta_all['run'].astype(str)
    meta_all['session'] = meta_all['session'].astype(str)
    meta_all['event_ordinal'] = meta_all.groupby(
        ['subject', 'session', 'run'], sort=False).cumcount()

    keep = meta_all['session'].eq(cfg['session']).to_numpy()
    chosen_meta = meta_all.loc[keep].reset_index(drop=True)
    x = np.asarray(x_all[keep], dtype=np.float64)
    class_names = chosen_meta['class_name'].to_numpy(dtype=str)
    unknown = sorted(set(class_names) - set(cfg['label_map']))
    if unknown:
        raise RuntimeError(f'{dataset_name}: labels missing from fixed map: {unknown}')
    y = np.asarray([cfg['label_map'][label] for label in class_names], dtype=np.int64)
    if dataset_name == 'BNCI2014001-4':
        x = x[:, :, :cfg['source_samples']]
    if x.ndim != 3 or x.shape[1] != cfg['channels']:
        raise RuntimeError(f'{dataset_name}: unexpected shape {x.shape}')
    if len(x) != cfg['expected_trials']:
        raise RuntimeError(f'{dataset_name}: expected {cfg["expected_trials"]} selected trials, got {len(x)}')
    if dataset_name == 'BNCI2014001-4' and x.shape[-1] != cfg['source_samples']:
        raise RuntimeError(f'{dataset_name}: expected first {cfg["source_samples"]} points, got {x.shape}')
    if dataset_name == 'BNCI2014004':
        counts = chosen_meta.groupby('subject').size().to_dict()
        expected = {s: (120 if s == 2 else 160) for s in subjects}
        if counts != expected:
            raise RuntimeError(f'{dataset_name}: unexpected per-subject counts {counts}')
    if dataset_name == 'BNCI2015001':
        counts = chosen_meta.groupby('subject').size().to_dict()
        if any(counts.get(s, 0) != 200 for s in subjects):
            raise RuntimeError(f'{dataset_name}: unexpected per-subject counts {counts}')

    raw_sha_by_key: dict[tuple[int, str], tuple[str, str]] = {}
    uids = []
    records = []
    for row in chosen_meta.itertuples(index=False):
        subject = int(row.subject)
        session = str(row.session)
        key = (subject, session)
        if key not in raw_sha_by_key:
            raw_sha_by_key[key] = _raw_identity(
                raw_by_subject[subject], subject, session, dataset_name)
        raw_path, raw_sha = raw_sha_by_key[key]
        run = _canonical_run(row.run)
        ordinal = int(row.event_ordinal)
        label = str(row.class_name)
        uid = (f'{dataset_name}|{raw_sha}|sub-{subject:02d}|session-{session}'
               f'|run-{run}|event-{ordinal:04d}|class-{label}')
        uids.append(uid)
        record = {str(k): (v.item() if isinstance(v, np.generic) else v)
                  for k, v in row._asdict().items()}
        record.update({'dataset': dataset_name, 'source_row': len(records),
                       'native_fs_hz': cfg['fs'], 'channels': cfg['channels'],
                       'raw_file': raw_path, 'raw_file_sha256': raw_sha,
                       'trial_uid': uid, 'label_id': int(cfg['label_map'][label])})
        records.append(record)

    mapping, pair_report = _verify_legacy_mapping(dataset_name, chosen_meta, uids, cfg)
    if (not pair_report['one_to_one'] or pair_report['mismatch_count'] != 0
            or not pair_report['all_labels_match']):
        raise RuntimeError(
            f'{dataset_name}: legacy/rebuilt trial mapping failed: '
            f'{json.dumps(pair_report, sort_keys=True)}')
    dest = OUTPUT_ROOT / dataset_name
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        'dataset': dataset_name, 'source_dataset': cfg['source_name'],
        'moabb_version': moabb.__version__, 'moabb_fmin_hz': cfg['fmin'],
        'moabb_fmax_hz': cfg['fmax'], 'moabb_tmin_s': float(paradigm.tmin),
        'moabb_tmax_s': None if paradigm.tmax is None else float(paradigm.tmax),
        'dataset_event_interval_s': list(dataset.interval),
        'paradigm_resample_hz': paradigm.resample,
        'effective_native_fs_hz': cfg['fs'], 'source_shape': list(x.shape),
        'source_dtype': str(x.dtype), 'source_unit': 'MOABB MotorImagery output; recorded as returned',
        'selected_session': cfg['session'], 'selected_trials': len(x),
        'class_map': cfg['label_map'], 'raw_files': raw_manifest,
        'pairing_report': pair_report,
        'dataset_preprocess_code_sha256': sha256(
            ROOT / 'experiments/finetune/build_eegfm_reference_source.py'),
    }
    tmp = dest / 'X.npy.partial'
    with tmp.open('wb') as f:
        np.save(f, x, allow_pickle=False)
    os.replace(tmp, dest / 'X.npy')
    np.save(dest / 'y.npy', y, allow_pickle=False)
    pd.DataFrame(records).to_csv(dest / 'trials.csv', index=False)
    mapping.to_csv(dest / 'legacy_row_mapping.csv', index=False)
    manifest['files'] = {
        name: {'bytes': (dest / name).stat().st_size,
               'sha256': sha256(dest / name)}
        for name in ('X.npy', 'y.npy', 'trials.csv', 'legacy_row_mapping.csv')
    }
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(f'[built] {dataset_name} shape={x.shape} paired={pair_report["mapped_trials"]}'
          f'/{pair_report["legacy_selected_trials"]} mismatches={pair_report["mismatch_count"]}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=tuple(DATASETS),
                        default=list(DATASETS))
    parser.add_argument('--download-retries', type=int, default=5)
    args = parser.parse_args()
    os.environ['MNE_DATA'] = str(require_external_output(os.environ.get('MNE_DATA', MNE_ROOT)))
    os.environ['MNE_DATASETS_BNCI_PATH'] = str(require_external_output(
        os.environ.get('MNE_DATASETS_BNCI_PATH', MNE_ROOT)))
    if not Path(os.environ['MNE_DATA']).exists():
        raise RuntimeError(f'MNE_DATA does not exist: {os.environ["MNE_DATA"]}')
    for name in args.datasets:
        build_one(name, DATASETS[name], download_retries=args.download_retries)


if __name__ == '__main__':
    main()
