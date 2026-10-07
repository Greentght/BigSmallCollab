#!/usr/bin/env python
"""Prepare reference-aligned model inputs from the existing NPY source cache.

This control intentionally reuses the verified trial mapping and the same
model-side transforms as the MOABB-source run. The NPY values are used as
stored; the MOABB source-level filter is not applied a second time because the
legacy NPY filter history is not recorded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

from experiments.finetune import prepare_loso_alignment_inputs as prep


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.storage import (DATA_CACHE_ROOT, RESULTS_ROOT,
                                 require_external_output, resolve_local_file)
OLD_DATA = Path('/data1/llx')
SOURCE_ROOT = DATA_CACHE_ROOT / 'eegfm_alignment_v2/rebuilt'
OUTPUT_ROOT = DATA_CACHE_ROOT / 'eegfm_alignment_v2/model_inputs'
SPEC_PATH = ROOT / 'configs/reproductions/loso_config_alignment_v2.yaml'

DATASET = 'BNCI2014001-4'
SOURCE_DATASET = 'BNCI2014001'
MODEL = 'cbramod'
PROFILE = 'reference_aligned_npy'
VARIANT = 'npy_source'
SOURCE_FS = 250
SOURCE_CROP = 1000


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    old_dir = OLD_DATA / SOURCE_DATASET
    old_x = np.load(old_dir / 'X.npy', mmap_mode='r')
    old_labels = np.load(old_dir / 'labels.npy', mmap_mode='r', allow_pickle=True).astype(str)
    old_meta = pd.read_csv(old_dir / 'meta.csv')
    mapping_path = SOURCE_ROOT / DATASET / 'legacy_row_mapping.csv'
    mapping = pd.read_csv(mapping_path).sort_values('rebuilt_row').reset_index(drop=True)

    if len(mapping) != 2592 or not mapping['paired_verified'].all():
        raise RuntimeError('Expected 2592 verified 001-4 NPY↔MOABB trial mappings')
    if not np.array_equal(mapping['rebuilt_row'].to_numpy(), np.arange(len(mapping))):
        raise RuntimeError('Rebuilt row order is not contiguous; refusing silent reordering')

    legacy_rows = mapping['legacy_raw_row'].to_numpy(dtype=np.int64)
    if legacy_rows.min() < 0 or legacy_rows.max() >= len(old_x):
        raise RuntimeError('Legacy NPY row mapping exceeds X.npy bounds')
    selected_meta = old_meta.iloc[legacy_rows].reset_index(drop=True)
    class_names = mapping['class_name'].astype(str).to_numpy()
    actual_labels = old_labels[legacy_rows]
    if not np.array_equal(actual_labels, class_names):
        raise RuntimeError('NPY labels do not match the verified trial mapping')
    if not np.all(selected_meta['session'].astype(str).to_numpy() == 'session_T'):
        raise RuntimeError('Mapped NPY rows contain trials outside session_T')
    if not np.array_equal(selected_meta['subject'].to_numpy(dtype=np.int64),
                          mapping['subject'].to_numpy(dtype=np.int64)):
        raise RuntimeError('NPY subject IDs differ from mapped source trials')
    if 'Unnamed: 0' in selected_meta and not np.array_equal(
            selected_meta['Unnamed: 0'].to_numpy(dtype=np.int64), legacy_rows):
        raise RuntimeError('NPY metadata row IDs do not match mapped array rows')

    run = selected_meta['run'].astype(str).str.replace(r'^run[_-]?', '', regex=True)
    if not np.array_equal(run.to_numpy(), mapping['run_key'].astype(str).to_numpy()):
        raise RuntimeError('NPY run IDs differ from mapped trial identities')
    event_ord = selected_meta.groupby(['subject', 'session', 'run'], sort=False).cumcount()
    if not np.array_equal(event_ord.to_numpy(dtype=np.int64),
                          mapping['event_ordinal'].to_numpy(dtype=np.int64)):
        raise RuntimeError('NPY event ordinals differ from mapped trial identities')

    reference_map = prep.LABELS_REFERENCE[DATASET]
    y = np.asarray([reference_map[name] for name in class_names], dtype=np.int64)
    subjects = mapping['subject'].to_numpy(dtype=np.int64) - 1
    uids = mapping['trial_uid'].astype(str).tolist()

    old_files = {
        'X.npy': {'path': str(old_dir / 'X.npy'), 'sha256': file_sha256(old_dir / 'X.npy')},
        'labels.npy': {'path': str(old_dir / 'labels.npy'), 'sha256': file_sha256(old_dir / 'labels.npy')},
        'meta.csv': {'path': str(old_dir / 'meta.csv'), 'sha256': file_sha256(old_dir / 'meta.csv')},
    }
    source_manifest = {
        'source_kind': 'existing_npy_cache', 'source_dataset': SOURCE_DATASET,
        'source_root': str(old_dir), 'selected_legacy_session': 'session_T',
        'raw_array_shape': list(old_x.shape), 'raw_array_dtype': str(old_x.dtype),
        'selected_trials': len(mapping), 'source_sample_crop': SOURCE_CROP,
        'native_fs_hz': SOURCE_FS,
        'unit': 'not recorded in legacy NPY; values preserved as stored',
        'filter_history': 'not recorded in legacy NPY; no MOABB source filter reapplied',
        'files': old_files,
        'mapping_sha256': file_sha256(mapping_path),
        'reference_source_manifest_sha256': file_sha256(SOURCE_ROOT / DATASET / 'manifest.json'),
        'mapping_report': {
            'one_to_one': True, 'paired_verified_count': int(mapping['paired_verified'].sum()),
            'trial_count': len(mapping), 'labels_match': True,
            'selected_session_match': True, 'subject_run_event_order_match': True,
        },
    }
    return old_x, legacy_rows, y, subjects, uids, source_manifest


def prepare() -> Path:
    spec = yaml.safe_load(SPEC_PATH.read_text())
    x_source, source_indices, y, subjects, uids, source_manifest = source_rows()
    cfg = prep.get_profile('reference_aligned', MODEL, DATASET, spec)
    cfg['source_cast'] = 'float64_existing_npy'
    output = require_external_output(OUTPUT_ROOT / PROFILE / DATASET / MODEL / VARIANT)

    # Ensure this input can be compared trial-for-trial with the finished
    # MOABB-source run, including order and class encoding.
    ref_trials = pd.read_csv(
        OUTPUT_ROOT / 'reference_aligned' / DATASET / MODEL / 'rebuilt_source' / 'trials.csv')
    if ref_trials['trial_uid'].astype(str).tolist() != uids:
        raise RuntimeError('NPY trial UID order differs from the completed MOABB-source run')
    if not np.array_equal(ref_trials['label_id'].to_numpy(dtype=np.int64), y):
        raise RuntimeError('NPY class IDs differ from the completed MOABB-source run')
    ref_x_manifest = json.loads((OUTPUT_ROOT / 'reference_aligned' / DATASET / MODEL
                                 / 'rebuilt_source' / 'manifest.json').read_text())
    if ref_x_manifest['input_shape'] != [22, 4, 200]:
        raise RuntimeError(f"Unexpected paired MOABB model input shape: {ref_x_manifest['input_shape']}")

    output.mkdir(parents=True, exist_ok=True)
    first = np.asarray(x_source[source_indices[:1], :, :SOURCE_CROP], dtype=np.float64)
    sample, _ = prep.transform_chunk(first, SOURCE_FS, cfg, MODEL)
    x_path = output / 'X.npy'
    partial = output / 'X.npy.partial'
    x_out = np.lib.format.open_memmap(
        partial, mode='w+', dtype=np.float32, shape=(len(y),) + sample.shape[1:])
    after_resample = set()
    for start in range(0, len(y), 96):
        stop = min(start + 96, len(y))
        chunk = np.asarray(x_source[source_indices[start:stop], :, :SOURCE_CROP], dtype=np.float64)
        transformed, n_after = prep.transform_chunk(chunk, SOURCE_FS, cfg, MODEL)
        if transformed.shape[1:] != sample.shape[1:]:
            raise RuntimeError(f'Inconsistent transformed chunk shape: {transformed.shape}')
        x_out[start:stop] = transformed
        after_resample.add(int(n_after))
        print(f'[prepare-npy] {stop}/{len(y)} trials', flush=True)
    x_out.flush()
    del x_out
    partial.replace(x_path)
    np.save(output / 'y.npy', y, allow_pickle=False)
    np.save(output / 'subjects.npy', subjects, allow_pickle=False)
    pd.DataFrame({'trial_uid': uids, 'subject_zero_based': subjects,
                  'label_id': y}).to_csv(output / 'trials.csv', index=False)

    manifest = {
        'profile': PROFILE, 'dataset': DATASET, 'model': MODEL, 'variant': VARIANT,
        'source_fs_hz': SOURCE_FS, 'source_manifest_or_hashes': source_manifest,
        'resolved_profile': cfg,
        'preprocessing_order': ['first_1000_npy_samples', 'resample_to_model_fs',
                                'trim_or_repeat_pad', 'filter', 'notch',
                                'normalization', 'float32'],
        'after_resample_sample_lengths': sorted(after_resample),
        'duration_target_samples': int(cfg['duration'] * cfg['target_fs']),
        'input_shape': list(sample.shape[1:]), 'input_dtype': 'float32',
        'class_mapping': prep.LABELS_REFERENCE[DATASET], 'trial_count': len(y),
        'subjects': sorted(np.unique(subjects).tolist()), 'files': {},
        'paired_moabb_trial_table_sha256': file_sha256(
            OUTPUT_ROOT / 'reference_aligned' / DATASET / MODEL / 'rebuilt_source' / 'trials.csv'),
    }
    for name in ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv'):
        path = output / name
        manifest['files'][name] = {'bytes': path.stat().st_size, 'sha256': file_sha256(path)}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return output


if __name__ == '__main__':
    print(prepare(), flush=True)
