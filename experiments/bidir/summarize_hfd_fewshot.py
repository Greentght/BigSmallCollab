"""Validate and report complete paired subject-wise HFD few-shot results."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
import yaml

from data.loso_source import sha256_file
from experiments.bidir.run_hfd_fewshot import CONFIG, implementation_hashes, uid_hash
from experiments.bidir.summarize_hfd_loso import write_csv


def summarize(cfg, config_path):
    root = Path(cfg['output_root']) / 'runs'
    config_hash = sha256_file(config_path)
    records, sources, initial, splits, evaluation = {}, {}, {}, {}, {}
    rows = []
    for pair, condition, subject, seed in product(
            cfg['pairs'], cfg['conditions'], cfg['subjects'], cfg['seeds']):
        folder = root / pair / condition / f'subject_{subject}' / f'seed_{seed}'
        result_path, prediction_path = folder / 'evaluation.json', folder / 'predictions.npz'
        result = json.loads(result_path.read_text())
        identity = result['identity']
        expected = (cfg['experiment_id'], cfg['dataset'], pair, condition, subject, seed,
                    'fewshot_stratified_random_30_70', 'final_epoch', False, config_hash)
        actual = tuple(identity[k] for k in ('experiment_id', 'dataset', 'pair', 'condition',
                      'subject', 'seed', 'protocol', 'checkpoint_selection', 'smoke', 'config_sha256'))
        if (actual != expected or identity['implementation_sha256'] != implementation_hashes(pair)
                or result.get('evaluation_split') != 'test'):
            raise ValueError(f'{result_path}: few-shot run or code identity mismatch')
        if not prediction_path.is_file() or sha256_file(prediction_path) != result['predictions_sha256']:
            raise ValueError(f'{result_path}: missing or changed predictions')
        provenance = (identity['source_manifest_sha256'], tuple(identity['input_manifest_sha256']),
                      identity['pretrained_sha256'], identity['mirepnet_ea_policy'])
        if sources.setdefault(pair, provenance) != provenance:
            raise ValueError(f'{result_path}: input or pretrained identity differs')
        weight_id = tuple(identity['initial_state_sha256'])
        if initial.setdefault((pair, seed), weight_id) != weight_id:
            raise ValueError(f'{result_path}: initial model weights differ')
        split = (identity['calibration_uid_sha256'], identity['test_uid_sha256'],
                 identity['calibration_count'], identity['test_count'])
        if splits.setdefault((subject, seed), split) != split:
            raise ValueError(f'{result_path}: subject calibration/test split differs')
        with np.load(prediction_path, allow_pickle=False) as pred:
            uid, truth = pred['uid'], pred['y']
            if (len(uid) != identity['test_count'] or len(set(uid.tolist())) != len(uid) or
                    uid_hash(uid.tolist()) != identity['test_uid_sha256']):
                raise ValueError(f'{result_path}: invalid test UIDs')
            paired = (tuple(uid.tolist()), tuple(truth.tolist()))
            if evaluation.setdefault((subject, seed), paired) != paired:
                raise ValueError(f'{result_path}: paired test data differ')
            for role in ('big', 'small'):
                logits = pred[f'{role}_logits']
                if logits.shape != (len(truth), 2) or not np.isfinite(logits).all():
                    raise ValueError(f'{result_path}: invalid logits')
                predicted = logits.argmax(axis=1)
                computed = {'accuracy': float(np.mean(predicted == truth) * 100),
                            'balanced_accuracy': float(balanced_accuracy_score(truth, predicted) * 100),
                            'kappa': float(cohen_kappa_score(truth, predicted))}
                rec = result[role]
                if any(not np.isclose(score, rec[name], atol=1e-9) for name, score in computed.items()):
                    raise ValueError(f'{result_path}: metrics do not match predictions')
                if rec['selected_epoch'] != cfg['epochs']:
                    raise ValueError(f'{result_path}: expected final epoch selection')
                row = {'pair': pair, 'model': cfg['pairs'][pair][role]['model'], 'role': role,
                       'condition': condition, 'subject': subject, 'seed': seed,
                       'accuracy_percent': rec['accuracy'],
                       'balanced_accuracy_percent': rec['balanced_accuracy'], 'kappa': rec['kappa'],
                       'selected_epoch': rec['selected_epoch'], 'calibration_trials': split[2],
                       'test_trials': split[3], 'source': str(result_path)}
                rows.append(row)
                records[(pair, role, condition, subject, seed)] = row
    means, differences, contrasts = [], [], []
    rng = np.random.default_rng(20261126)
    for pair, role, condition, metric in product(
            cfg['pairs'], ('big', 'small'), cfg['conditions'],
            ('accuracy_percent', 'balanced_accuracy_percent')):
        seed_means = [np.mean([records[(pair, role, condition, subject, seed)][metric]
                               for subject in cfg['subjects']]) for seed in cfg['seeds']]
        means.append({'pair': pair, 'model': cfg['pairs'][pair][role]['model'], 'role': role,
                      'condition': condition, 'metric': metric, 'mean_percent': float(np.mean(seed_means)),
                      **{f'seed_{seed}': float(value) for seed, value in zip(cfg['seeds'], seed_means)}})
    for pair, role, metric, (method, baseline) in product(
            cfg['pairs'], ('big', 'small'), ('accuracy_percent', 'balanced_accuracy_percent'),
            (('ce_hfd', 'ce'), ('dml_kl_hfd', 'dml_kl'))):
        delta = np.array([
            np.mean([records[(pair, role, method, subject, seed)][metric] -
                     records[(pair, role, baseline, subject, seed)][metric] for seed in cfg['seeds']])
            for subject in cfg['subjects']])
        draws = rng.integers(0, len(delta), size=(10000, len(delta)))
        low, high = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])
        for subject, value in zip(cfg['subjects'], delta):
            differences.append({'pair': pair, 'model': cfg['pairs'][pair][role]['model'],
                                'role': role, 'metric': metric, 'method': method,
                                'baseline': baseline, 'subject': subject, 'delta_pp': float(value)})
        contrasts.append({'pair': pair, 'model': cfg['pairs'][pair][role]['model'],
                          'role': role, 'metric': metric, 'method': method, 'baseline': baseline,
                          'mean_delta_pp': float(delta.mean()), 'ci95_low_pp': float(low),
                          'ci95_high_pp': float(high), 'wins': int((delta > 1e-8).sum()),
                          'ties': int((np.abs(delta) <= 1e-8).sum()), 'losses': int((delta < -1e-8).sum()),
                          'n_subjects': len(delta), 'bootstrap_draws': 10000})
    output = Path(cfg['report_root'])
    write_csv(output / 'subject_seed_results.csv', rows)
    write_csv(output / 'summary.csv', means)
    write_csv(output / 'paired_subject_differences.csv', differences)
    write_csv(output / 'paired_contrasts.csv', contrasts)
    print(f'Validated {len(rows)} few-shot model-condition-subject-seed records and wrote {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    summarize(yaml.safe_load(args.config.read_text()), args.config.resolve())


if __name__ == '__main__':
    main()
