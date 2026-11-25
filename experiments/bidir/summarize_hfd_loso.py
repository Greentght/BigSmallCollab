"""Create complete, identity-checked paired HFD LOSO reports."""

import argparse
import csv
import json
import os
from itertools import product
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

from data.loso_source import sha256_file
from experiments.bidir.run_hfd_loso import CONFIG, implementation_hashes
from experiments.storage import require_report_output


def write_csv(path, rows):
    destination = require_report_output(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def summarize(cfg, config_path):
    root = Path(cfg['output_root']) / 'runs'
    config_hash = sha256_file(config_path)
    records = {}
    data_identities = {}
    initial_states = {}
    prediction_identities = {}
    folds = []
    for pair, condition, fold, seed in product(cfg['pairs'], cfg['conditions'], cfg['folds'], cfg['seeds']):
        folder = root / pair / condition / f'fold_{fold}' / f'seed_{seed}'
        result_path = folder / 'evaluation.json'
        predictions_path = folder / 'predictions.npz'
        result = json.loads(result_path.read_text())
        identity = result['identity']
        expected = (cfg['experiment_id'], cfg['dataset'], pair, condition, fold, seed, False, config_hash)
        actual = tuple(identity[k] for k in ('experiment_id', 'dataset', 'pair', 'condition',
                                               'fold', 'seed', 'smoke', 'config_sha256'))
        if (actual != expected or identity['implementation_sha256'] != implementation_hashes(pair)
                or identity['validation_subject'] != (fold + 1) % 9
                or result.get('evaluation_split') != 'test'):
            raise ValueError(f'{result_path}: run or code identity mismatch')
        if not predictions_path.is_file() or result['predictions_sha256'] != sha256_file(predictions_path):
            raise ValueError(f'{result_path}: missing or changed predictions')
        provenance = (identity['source_manifest_sha256'], tuple(identity['input_manifest_sha256']),
                      identity['pretrained_sha256'])
        previous = data_identities.setdefault(pair, provenance)
        if provenance != previous:
            raise ValueError(f'{result_path}: data or pretrained identity differs')
        initial = tuple(identity['initial_state_sha256'])
        if initial_states.setdefault((pair, seed), initial) != initial:
            raise ValueError(f'{result_path}: initial weights differ across conditions or folds')
        with np.load(predictions_path, allow_pickle=False) as predictions:
            uid = predictions['uid']
            truth = predictions['y']
            if len(uid) != len(set(uid.tolist())):
                raise ValueError(f'{result_path}: duplicated prediction UIDs')
            pred_identity = (tuple(uid.tolist()), tuple(truth.tolist()))
            if prediction_identities.setdefault((fold,), pred_identity) != pred_identity:
                raise ValueError(f'{result_path}: paired evaluation UIDs differ')
            for role in ('big', 'small'):
                logits = predictions[f'{role}_logits']
                if logits.shape != (len(truth), 2) or not np.isfinite(logits).all():
                    raise ValueError(f'{result_path}: invalid logits')
                predicted = logits.argmax(axis=1)
                metrics = {'accuracy': float(np.mean(predicted == truth) * 100),
                           'balanced_accuracy': float(balanced_accuracy_score(truth, predicted) * 100),
                           'kappa': float(cohen_kappa_score(truth, predicted))}
                if any(not np.isclose(score, result[role][name], atol=1e-9)
                       for name, score in metrics.items()):
                    raise ValueError(f'{result_path}: metrics do not match predictions')
                rec = result[role]
                row = {'pair': pair, 'model': cfg['pairs'][pair][role]['model'],
                       'role': role, 'condition': condition, 'fold': fold, 'seed': seed,
                       'accuracy_percent': rec['accuracy'], 'balanced_accuracy_percent': rec['balanced_accuracy'],
                       'kappa': rec['kappa'], 'selected_epoch': rec['selected_epoch'],
                       'test_trials': len(truth), 'source': str(result_path)}
                folds.append(row)
                records[(pair, role, condition, fold, seed)] = row
    means, differences, contrasts = [], [], []
    rng = np.random.default_rng(20261125)
    for pair, role, condition in product(cfg['pairs'], ('big', 'small'), cfg['conditions']):
        for metric in ('accuracy_percent', 'balanced_accuracy_percent'):
            seed_means = [np.mean([records[(pair, role, condition, fold, seed)][metric]
                                  for fold in cfg['folds']]) for seed in cfg['seeds']]
            means.append({'pair': pair, 'model': cfg['pairs'][pair][role]['model'],
                          'role': role, 'condition': condition, 'metric': metric,
                          'mean_percent': float(np.mean(seed_means)),
                          'seed_666': seed_means[0], 'seed_667': seed_means[1], 'seed_668': seed_means[2]})
    for pair, role, metric, (method, baseline) in product(
            cfg['pairs'], ('big', 'small'), ('accuracy_percent', 'balanced_accuracy_percent'),
            (('ce_hfd', 'ce'), ('dml_kl_hfd', 'dml_kl'))):
        delta = np.array([
            np.mean([records[(pair, role, method, fold, seed)][metric] -
                     records[(pair, role, baseline, fold, seed)][metric] for seed in cfg['seeds']])
            for fold in cfg['folds']
        ])
        draws = rng.integers(0, len(delta), size=(10000, len(delta)))
        low, high = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])
        for fold, value in zip(cfg['folds'], delta):
            differences.append({'pair': pair, 'model': cfg['pairs'][pair][role]['model'],
                                'role': role, 'metric': metric, 'method': method, 'baseline': baseline,
                                'fold': fold, 'delta_pp': float(value)})
        contrasts.append({'pair': pair, 'model': cfg['pairs'][pair][role]['model'],
                          'role': role, 'metric': metric, 'method': method, 'baseline': baseline,
                          'mean_delta_pp': float(delta.mean()), 'ci95_low_pp': float(low),
                          'ci95_high_pp': float(high), 'wins': int((delta > 1e-8).sum()),
                          'ties': int((np.abs(delta) <= 1e-8).sum()), 'losses': int((delta < -1e-8).sum()),
                          'n_subjects': len(delta), 'bootstrap_draws': 10000})
    output = Path(cfg['report_root'])
    write_csv(output / 'fold_results.csv', folds)
    write_csv(output / 'summary.csv', means)
    write_csv(output / 'paired_subject_differences.csv', differences)
    write_csv(output / 'paired_contrasts.csv', contrasts)
    print(f'Validated {len(folds)} model-condition-fold-seed records and wrote {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    summarize(yaml.safe_load(args.config.read_text()), args.config.resolve())


if __name__ == '__main__':
    main()
