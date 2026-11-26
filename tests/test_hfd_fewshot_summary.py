"""Verify few-shot reports use paired subjects and validate retained predictions."""

from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

from data.loso_source import sha256_file
from experiments.bidir import summarize_hfd_fewshot as report
from experiments.bidir.run_hfd_fewshot import uid_hash


def test_summary_pairs_subjects_and_rejects_changed_predictions():
    with TemporaryDirectory(prefix='hfd-fewshot-report-') as directory:
        root = Path(directory)
        config_path = root / 'config.yaml'
        config_path.write_text('synthetic: true\n')
        cfg = {'experiment_id': 'unit', 'dataset': 'BNCI2014004',
               'pairs': {'mirepnet_ifnet': {'big': {'model': 'mirepnet'}, 'small': {'model': 'ifnet'}},
                         'cbramod_eegnet': {'big': {'model': 'cbramod'}, 'small': {'model': 'eegnet'}}},
               'conditions': ['ce', 'ce_hfd', 'dml_kl', 'dml_kl_hfd'],
               'subjects': [0, 1], 'seeds': [666, 667], 'epochs': 100,
               'output_root': str(root / 'artifacts'), 'report_root': 'unused'}
        for pair, condition, subject, seed in product(
                cfg['pairs'], cfg['conditions'], cfg['subjects'], cfg['seeds']):
            folder = Path(cfg['output_root']) / 'runs' / pair / condition / f'subject_{subject}' / f'seed_{seed}'
            folder.mkdir(parents=True)
            uid = np.array([f's{subject}_r{seed}_0', f's{subject}_r{seed}_1'])
            truth = np.array([0, 1])
            predicted = np.array([0, 1] if condition.endswith('hfd') else [0, 0])
            logits = np.eye(2, dtype=np.float32)[predicted]
            np.savez_compressed(folder / 'predictions.npz', uid=uid, y=truth,
                                big_logits=logits, small_logits=logits)
            metric = {'accuracy': float(np.mean(predicted == truth) * 100),
                      'balanced_accuracy': float(balanced_accuracy_score(truth, predicted) * 100),
                      'kappa': float(cohen_kappa_score(truth, predicted)), 'selected_epoch': 100}
            identity = {'experiment_id': 'unit', 'dataset': cfg['dataset'], 'pair': pair,
                        'condition': condition, 'subject': subject, 'seed': seed,
                        'protocol': 'fewshot_stratified_random_30_70', 'checkpoint_selection': 'final_epoch',
                        'smoke': False, 'config_sha256': sha256_file(config_path),
                        'implementation_sha256': {'unit': 'code'}, 'source_manifest_sha256': 'source',
                        'input_manifest_sha256': [pair, 'input'], 'pretrained_sha256': pair,
                        'mirepnet_ea_policy': 'transductive' if pair.startswith('mirepnet') else None,
                        'initial_state_sha256': [pair, str(seed)], 'calibration_uid_sha256': f'cal-{subject}-{seed}',
                        'test_uid_sha256': uid_hash(uid.tolist()), 'calibration_count': 2, 'test_count': 2}
            result = {'identity': identity, 'big': metric, 'small': metric, 'evaluation_split': 'test',
                      'predictions_sha256': sha256_file(folder / 'predictions.npz')}
            (folder / 'evaluation.json').write_text(json.dumps(result))
        outputs = {}
        with patch.object(report, 'implementation_hashes', return_value={'unit': 'code'}), \
                patch.object(report, 'write_csv', side_effect=lambda path, rows: outputs.update({path.name: rows})):
            report.summarize(cfg, config_path)
            assert len(outputs['subject_seed_results.csv']) == 64
            assert all(row['mean_delta_pp'] == 50.0 for row in outputs['paired_contrasts.csv'])
            path = next(Path(cfg['output_root']).rglob('predictions.npz'))
            path.write_bytes(b'changed')
            try:
                report.summarize(cfg, config_path)
            except ValueError:
                pass
            else:
                raise AssertionError('modified prediction artifact must be rejected')
