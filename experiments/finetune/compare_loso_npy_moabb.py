#!/usr/bin/env python
"""Quantify raw NPY-vs-MOABB differences over verified paired trials."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import welch

from data.preproc import bandpass


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.storage import (DATA_CACHE_ROOT, RESULTS_ROOT,
                                 require_external_output, resolve_local_file)
OLD = Path('/data1/llx')
REBUILT = DATA_CACHE_ROOT / 'eegfm_alignment_v2/rebuilt'
OUT = RESULTS_ROOT / 'reproductions/loso_config_alignment_v2/npy_vs_moabb_raw_comparison.json'
DATASETS = [
    ('BNCI2014001-4', 'BNCI2014001', 'meta.csv', 250),
    ('BNCI2014004', 'BNCI2014004', 'meta004.csv', 250),
    ('BNCI2015001', 'BNCI2015001', 'meta.csv', 512),
]


def compare(dataset: str, source_name: str, metadata_name: str, fs: int) -> dict:
    old_dir, new_dir = OLD / source_name, REBUILT / dataset
    mapping = pd.read_csv(new_dir / 'legacy_row_mapping.csv').sort_values('rebuilt_row')
    old = np.load(old_dir / 'X.npy', mmap_mode='r')
    new = np.load(new_dir / 'X.npy', mmap_mode='r')
    if not mapping['paired_verified'].all():
        raise RuntimeError(f'{dataset}: mapping includes unverified rows')
    old_rows = mapping['legacy_raw_row'].to_numpy(dtype=np.int64)
    new_rows = mapping['rebuilt_row'].to_numpy(dtype=np.int64)
    old_labels = np.load(old_dir / 'labels.npy', mmap_mode='r', allow_pickle=True).astype(str)
    if not np.array_equal(old_labels[old_rows], mapping['class_name'].astype(str).to_numpy()):
        raise RuntimeError(f'{dataset}: NPY labels do not match paired trial mapping')
    old_meta = pd.read_csv(old_dir / metadata_name)
    if dataset == 'BNCI2014001-4':
        selected_session = old_meta.iloc[old_rows]['session'].astype(str).to_numpy()
        if not np.all(selected_session == 'session_T'):
            raise RuntimeError('001-4 selected NPY rows are not all session_T')

    common = min(old.shape[-1], new.shape[-1])
    per_trial = {key: [] for key in ('old_std', 'moabb_std', 'old_abs_p99',
                                     'moabb_abs_p99', 'difference_rms',
                                     'relative_rms', 'correlation')}
    reconstructed = {key: [] for key in ('correlation', 'relative_rms', 'std_ratio')}
    for start in range(0, len(mapping), 16):
        stop = min(start + 16, len(mapping))
        x_old = np.asarray(old[old_rows[start:stop], :, :common], dtype=np.float64)
        x_new = np.asarray(new[new_rows[start:stop], :, :common], dtype=np.float64)
        x_new_8_30 = bandpass(x_new, fs, 8.0, 30.0)
        for a, b in zip(x_old, x_new):
            av, bv = a.ravel(), b.ravel()
            delta = av - bv
            rms = float(np.sqrt(np.mean(delta * delta)))
            per_trial['old_std'].append(float(av.std()))
            per_trial['moabb_std'].append(float(bv.std()))
            per_trial['old_abs_p99'].append(float(np.quantile(np.abs(av), .99)))
            per_trial['moabb_abs_p99'].append(float(np.quantile(np.abs(bv), .99)))
            per_trial['difference_rms'].append(rms)
            per_trial['relative_rms'].append(rms / (float(np.sqrt(np.mean(bv * bv))) + 1e-30))
            per_trial['correlation'].append(float(np.corrcoef(av, bv)[0, 1]))
        for a, b in zip(x_old, x_new_8_30):
            av, bv = a.ravel(), b.ravel()
            reconstructed['correlation'].append(float(np.corrcoef(av, bv)[0, 1]))
            reconstructed['relative_rms'].append(
                float(np.sqrt(np.mean((av - bv) ** 2)))
                / (float(np.sqrt(np.mean(bv * bv))) + 1e-30))
            reconstructed['std_ratio'].append(float(av.std()) / (float(bv.std()) + 1e-30))

    band_edges = [(0.1, 1), (1, 4), (4, 8), (8, 13), (13, 30),
                  (30, 50), (50, 75), (75, 120)]
    pick = np.unique(np.linspace(0, len(mapping) - 1, min(64, len(mapping)), dtype=int))
    band_power = {}
    for name, source in (('legacy_npy', old), ('moabb', new)):
        trials = np.asarray(source[old_rows[pick] if name == 'legacy_npy' else new_rows[pick],
                                  :, :common], dtype=np.float64)
        freqs, power = welch(trials, fs=fs, nperseg=min(512, common), axis=-1)
        pooled = power.mean(axis=(0, 1))
        total = float(np.trapz(pooled, freqs))
        band_power[name] = {}
        for lo, hi in band_edges:
            mask = (freqs >= lo) & (freqs < hi)
            if not np.any(mask):
                continue
            band_power[name][f'{lo:g}-{hi:g}Hz'] = float(
                np.trapz(pooled[mask], freqs[mask]) / (total + 1e-30))

    def summary(key: str) -> dict:
        values = np.asarray(per_trial[key], dtype=np.float64)
        return {'trial_mean': float(values.mean()), 'trial_median': float(np.median(values)),
                'trial_p05': float(np.quantile(values, .05)),
                'trial_p95': float(np.quantile(values, .95))}

    new_manifest = json.loads((new_dir / 'manifest.json').read_text())
    old_meta_rows = old_meta.iloc[old_rows]
    session_counts = old_meta_rows['session'].astype(str).value_counts().to_dict()
    return {
        'dataset': dataset, 'paired_trials': len(mapping),
        'mapping_one_to_one_verified': True, 'labels_match': True,
        'npy_shape': list(old.shape), 'moabb_selected_shape': list(new.shape),
        'common_samples_compared': int(common),
        'legacy_extra_samples': int(old.shape[-1] - common),
        'moabb_extra_samples': int(new.shape[-1] - common),
        'legacy_selected_session_counts': session_counts,
        'moabb_selected_session': new_manifest['selected_session'],
        'moabb_version': new_manifest['moabb_version'],
        'moabb_source_band_hz': [new_manifest['moabb_fmin_hz'], new_manifest['moabb_fmax_hz']],
        'relative_band_power_sampled_trials': band_power,
        'legacy_unit_and_filter_history': 'not recorded in legacy NPY metadata',
        'legacy_npy_vs_moabb_after_local_8_30_filter': {
            'interpretation': 'close waveform match supports that legacy NPY values already contain an 8-30 Hz filter; exact cache-generation code remains unlocated',
            'per_trial_mean_correlation': float(np.mean(reconstructed['correlation'])),
            'per_trial_mean_relative_rms': float(np.mean(reconstructed['relative_rms'])),
            'per_trial_mean_std_ratio_npy_over_filtered_moabb': float(np.mean(reconstructed['std_ratio'])),
        },
        'statistics_across_trialwise_values': {k: summary(k) for k in per_trial},
    }


def main() -> None:
    report = {
        'comparison': 'paired raw trial values before model-side transforms',
        'method': 'verified legacy_raw_row↔rebuilt_row mapping; compare shared time samples',
        'unit_warning': 'legacy NPY does not encode physical unit or filter history; amplitude differences alone do not prove a unit error',
        'datasets': [compare(*record) for record in DATASETS],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f'saved: {OUT}', flush=True)


if __name__ == '__main__':
    main()
