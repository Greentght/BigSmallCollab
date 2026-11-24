"""Export a verified AlexMI broadband source from the original Zenodo FIFs.

The export contains both task and rest trials at their native 512 Hz. The
source signal is filtered continuously before event-locked 3-second epochs.
The historical NPY version is read only to check event/label identity.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import mne
import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from data.channels import AlexMI_chn_names

ROOT = Path('/data1/llx/AlexMI')
OUTPUT = ROOT / 'wide_fre_band'
RAW_ROOT = OUTPUT / 'raw'
EVENTS = {2: 'right_hand', 3: 'feet', 4: 'rest'}
EXPECTED_MD5 = {
    1: 'ca425544294db7374bd1f6c06e0f94fa',
    2: 'a9f0392dd55b90917fe4cec4845cd77b',
    3: 'f0f9ddeb4b52fabbfdd4f406ea8bf6f4',
    4: '5abb108ac7aede30004ae602a98ce38e',
    5: 'd63e8b7870bf503b005b192e15f2eb4a',
    6: '0777e6829b6db87782a85afc25873cc8',
    7: '3e41580135f9024c70c77fd4a1eaa030',
    8: 'a3f0ae9848cdf9a62d7dae2124686908',
}


def file_hash(path: Path, algorithm: str = 'sha256') -> str:
    digest = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def correlate_legacy(old: np.ndarray, rebuilt: np.ndarray) -> dict:
    # A shared analysis band allows identity matching despite unknown historical
    # preprocessing. It is used only for verification, not as the source data.
    sos = butter(4, [8, 30], fs=512, btype='bandpass', output='sos')
    left = sosfiltfilt(sos, np.asarray(old, dtype=np.float64), axis=-1).reshape(60, -1)
    right = sosfiltfilt(sos, np.asarray(rebuilt, dtype=np.float64), axis=-1).reshape(60, -1)
    for item in (left, right):
        item -= item.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(item, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError('Non-finite or zero norm during trial identity verification')
        item /= norms
    correlations = left @ right.T
    diagonal = np.diag(correlations)
    if not np.array_equal(correlations.argmax(axis=1), np.arange(60)):
        raise ValueError('Legacy trial order does not match original event order')
    if float(diagonal.min()) < 0.90:
        raise ValueError(f'Legacy and original trial correlation below 0.90: {diagonal.min()}')
    return {'trials': 60, 'same_order_best_matches': 60,
            'diagonal_correlation_min': float(diagonal.min()),
            'diagonal_correlation_mean': float(diagonal.mean())}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = ('X.npy', 'labels.npy', 'meta.csv', 'trials.csv', 'verification.json',
               'legacy_row_mapping.csv', 'manifest.json')
    with (OUTPUT / '.export.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        present = [name for name in outputs if (OUTPUT / name).exists()]
        if present:
            raise FileExistsError(f'Refusing to replace existing broadband export files: {present}')
        legacy_x = np.load(ROOT / 'X.npy', mmap_mode='r', allow_pickle=False)
        legacy_y = np.load(ROOT / 'labels.npy', allow_pickle=False).astype(str)
        legacy_meta = pd.read_csv(ROOT / 'meta.csv')
        if legacy_x.shape != (480, 16, 1537) or len(legacy_y) != 480 or len(legacy_meta) != 480:
            raise ValueError('Unexpected historical AlexMI array dimensions')
        if legacy_meta['subject'].tolist() != np.repeat(np.arange(1, 9), 60).tolist():
            raise ValueError('Historical subject order is not the expected eight blocks')
        if not (legacy_meta['session'].astype(str).eq('0').all() and
                legacy_meta['run'].astype(str).eq('0').all()):
            raise ValueError('Historical session or run differs from expected AlexMI identity')

        chunks, all_labels, rows, raw_files, subject_checks = [], [], [], [], []
        for subject in range(1, 9):
            path = RAW_ROOT / f'subject{subject}.raw.fif'
            md5 = file_hash(path, 'md5')
            if md5 != EXPECTED_MD5[subject]:
                raise ValueError(f'Zenodo MD5 mismatch for {path}')
            sha = file_hash(path)
            raw = mne.io.read_raw_fif(path, preload=True, verbose='ERROR')
            if raw.info['sfreq'] != 512.0:
                raise ValueError(f'Unexpected sampling rate for subject {subject}')
            eeg_names = [name.upper() for name, kind in zip(raw.ch_names, raw.get_channel_types())
                         if kind == 'eeg']
            if eeg_names != list(AlexMI_chn_names):
                raise ValueError(f'Unexpected EEG channel order for subject {subject}: {eeg_names}')
            events = mne.find_events(raw, verbose='ERROR')
            events = events[np.isin(events[:, 2], list(EVENTS))]
            if len(events) != 60 or any(np.sum(events[:, 2] == code) != 20 for code in EVENTS):
                raise ValueError(f'Unexpected event counts for subject {subject}')
            labels = np.array([EVENTS[int(event[2])] for event in events])
            base = (subject - 1) * 60
            if not np.array_equal(labels, legacy_y[base:base + 60]):
                raise ValueError(f'Legacy labels differ from raw events for subject {subject}')
            raw.filter(l_freq=0.1, h_freq=75., picks='eeg', method='iir',
                       iir_params={'order': 4, 'ftype': 'butter', 'output': 'sos'},
                       phase='zero', verbose='ERROR')
            event_ids = {name: code for code, name in EVENTS.items()}
            epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=0, tmax=3,
                                baseline=None, picks='eeg', preload=True,
                                reject_by_annotation=False, verbose='ERROR')
            chunk = np.asarray(epochs.get_data(), dtype=np.float64) * 1e6
            if chunk.shape != (60, 16, 1537) or not np.isfinite(chunk).all():
                raise ValueError(f'Invalid epochs for subject {subject}: {chunk.shape}')
            check = correlate_legacy(legacy_x[base:base + 60], chunk)
            check['subject'] = subject
            subject_checks.append(check)
            chunks.append(chunk)
            all_labels.extend(labels.tolist())
            for ordinal, (event, label) in enumerate(zip(events, labels)):
                row = base + ordinal
                uid = f'AlexMI|{sha}|sub-{subject:02d}|session-0|run-0|event-{ordinal:04d}|class-{label}'
                rows.append({'source_row': row, 'subject': subject, 'session': 0,
                             'run': 0, 'event_ordinal': ordinal,
                             'event_sample': int(event[0]), 'event_code': int(event[2]),
                             'class_name': label, 'trial_uid': uid,
                             'raw_file_sha256': sha})
            raw_files.append({'subject': subject, 'path': str(path), 'md5': md5,
                              'sha256': sha, 'bytes': path.stat().st_size,
                              'download_url': f'https://zenodo.org/records/806023/files/subject{subject}.raw.fif'})
            print(f'[verified] subject={subject} trials=60 correlation_min={check["diagonal_correlation_min"]:.4f}', flush=True)

        x = np.concatenate(chunks, axis=0)
        labels = np.asarray(all_labels)
        trials = pd.DataFrame(rows)
        if len(set(trials['trial_uid'])) != 480:
            raise ValueError('Duplicate physical trial identities')
        verification = {
            'legacy_mapping': {'one_to_one': True, 'mapped_trials': 480, 'all_labels_match': True,
                               'waveform_check_band_hz': [8, 30],
                               'waveform_check_description': 'per-trial normalized correlation after identical 8-30 Hz filtering; each legacy trial selects the matching raw event as its best match',
                               'subjects': subject_checks,
                               'legacy_files': {name: {'sha256': file_hash(ROOT / name),
                                                       'path': str(ROOT / name)}
                                                for name in ('X.npy', 'labels.npy', 'meta.csv')}},
            'raw_checksum_source': 'https://zenodo.org/api/records/806023',
        }
        mapping = trials[['source_row', 'subject', 'session', 'run', 'event_ordinal',
                          'class_name', 'trial_uid']].copy()
        mapping.insert(0, 'legacy_raw_row', np.arange(480, dtype=np.int64))
        mapping['paired_verified'] = True
        with tempfile.TemporaryDirectory(prefix='.export-', dir=OUTPUT) as temp:
            stage = Path(temp)
            np.save(stage / 'X.npy', x, allow_pickle=False)
            np.save(stage / 'labels.npy', labels, allow_pickle=False)
            trials.to_csv(stage / 'meta.csv', index=False)
            trials.to_csv(stage / 'trials.csv', index=False)
            mapping.to_csv(stage / 'legacy_row_mapping.csv', index=False)
            (stage / 'verification.json').write_text(json.dumps(verification, indent=2, sort_keys=True) + '\n')
            files = {name: {'sha256': file_hash(stage / name), 'bytes': (stage / name).stat().st_size}
                     for name in outputs if name != 'manifest.json'}
            manifest = {'dataset': 'AlexMI', 'source_variant': 'wide_fre_band',
                        'created_utc': datetime.now(timezone.utc).isoformat(),
                        'shape': list(x.shape), 'dtype': str(x.dtype), 'unit': 'microvolts',
                        'native_fs_hz': 512, 'duration_seconds': 3,
                        'channels': list(AlexMI_chn_names), 'events': EVENTS,
                        'preprocessing_order': ['continuous_raw_0.1_to_75_Hz_order4_butterworth_iir_zero_phase',
                                                'event_epoch_0_to_3s_at_native_512Hz_inclusive',
                                                'EEG_channel_selection', 'volts_to_microvolts'],
                        'raw_files': raw_files, 'files': files,
                        'software': {'mne': mne.__version__, 'numpy': np.__version__,
                                     'scipy': scipy.__version__, 'pandas': pd.__version__},
                        'export_script_sha256': file_hash(Path(__file__).resolve())}
            (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
            for name in outputs:
                os.replace(stage / name, OUTPUT / name)
        print(f'[complete] source={OUTPUT} shape={x.shape} manifest_sha256={file_hash(OUTPUT / "manifest.json")}', flush=True)


if __name__ == '__main__':
    main()
