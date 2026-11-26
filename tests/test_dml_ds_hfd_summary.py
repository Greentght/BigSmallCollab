"""Matched reports must reject changed baselines and use paired test trials."""

from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from data.loso_source import sha256_file
from experiments.bidir import summarize_dml_ds_hfd_fewshot as report
from experiments.bidir.run_dml_ds_hfd_fewshot import uid_hash


def test_summary_matches_baseline_and_detects_changed_predictions():
    with TemporaryDirectory(prefix='dml-ds-hfd-report-') as directory:
        root = Path(directory)
        config_path = root / 'config.yaml'
        config_path.write_text('synthetic: true\n')
        cfg = {'dataset': 'BNCI2014004', 'pair': 'cbramod_eegnet', 'condition': 'dml_ds_hfd',
               'subjects': [0, 1], 'seeds': [666, 667], 'baseline_root': str(root/'baseline'),
               'output_root': str(root/'addon'), 'report_root': 'unused'}
        for subject, seed in product(cfg['subjects'], cfg['seeds']):
            baseline = Path(cfg['baseline_root']) / f'subject_{subject}' / f'seed_{seed}'
            addon = Path(cfg['output_root']) / 'runs' / f'subject_{subject}' / f'seed_{seed}'
            baseline.mkdir(parents=True); addon.mkdir(parents=True)
            uids = np.array([f's{subject}_r{seed}_0', f's{subject}_r{seed}_1'])
            truth = np.array([0, 1]); before = np.array([0, 0]); after = np.array([0, 1])
            np.savez_compressed(baseline/'predictions.npz', uids=uids, y=truth, big=before, small=before)
            np.savez_compressed(addon/'predictions.npz', uids=uids, y=truth, big=after, small=after)
            common = {'dataset': cfg['dataset'], 'pair': cfg['pair'], 'subject': subject, 'seed': seed,
                      'protocol': 'fewshot_stratified_random_30_70', 'smoke': False,
                      'source_manifest_sha256': 'src', 'train_uid_sha256': f'train-{subject}-{seed}',
                      'test_uid_sha256': uid_hash(uids.tolist()), 'big_training': {'lr': .1},
                      'small_training': {'lr': .2}, 'pretrained_sha256': 'weight', 'epochs': 20, 'batch_size': 16}
            bi = dict(common, condition='dml_ds')
            br = {'identity': bi, 'big_accuracy': 50.0, 'small_accuracy': 50.0,
                  'train_count': 2, 'test_count': 2}
            (baseline/'evaluation.json').write_text(json.dumps(br))
            ai = dict(common, condition='dml_ds_hfd', train_count=2, test_count=2, final_epoch_only=True,
                      implementation_sha256={'unit': 'hash'}, config_sha256=sha256_file(config_path),
                      baseline_result_sha256=sha256_file(baseline/'evaluation.json'),
                      baseline_predictions_sha256=sha256_file(baseline/'predictions.npz'))
            ar = {'identity': ai, 'evaluation_split': 'test',
                  'predictions_sha256': sha256_file(addon/'predictions.npz'),
                  'big': {'accuracy': 100.0, 'balanced_accuracy': 100.0, 'selected_epoch': 20},
                  'small': {'accuracy': 100.0, 'balanced_accuracy': 100.0, 'selected_epoch': 20}}
            (addon/'evaluation.json').write_text(json.dumps(ar))
        outputs = {}
        with patch.object(report, 'implementation_hashes', return_value={'unit': 'hash'}), \
                patch.object(report, 'write_csv', side_effect=lambda p, rows: outputs.update({p.name: rows})):
            report.summarize(cfg, config_path)
            assert len(outputs['subject_seed_results.csv']) == 16
            assert all(row['mean_delta_pp'] == 50 for row in outputs['paired_contrasts.csv'])
            next(Path(cfg['baseline_root']).rglob('predictions.npz')).write_bytes(b'changed')
            try: report.summarize(cfg, config_path)
            except ValueError: pass
            else: raise AssertionError('Changed baseline predictions must be rejected')
