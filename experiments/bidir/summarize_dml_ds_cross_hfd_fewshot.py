"""Pair completed DML-DS+HFD runs with their existing DML-DS baselines."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
import yaml

from data.loso_source import sha256_file
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import CONFIG, implementation_hashes, uid_hash
from experiments.bidir.summarize_hfd_loso import write_csv


def summarize(cfg, config_path):
    rows, results, subject_differences, contrasts = [], {}, [], []
    implementation = implementation_hashes()
    for subject, seed in product(cfg['subjects'], cfg['seeds']):
        baseline = Path(cfg['baseline_root']) / f'subject_{subject}' / f'seed_{seed}'
        addon = Path(cfg['output_root']) / 'runs' / f'subject_{subject}' / f'seed_{seed}'
        bpath, apath = baseline / 'evaluation.json', addon / 'evaluation.json'
        br, ar = json.loads(bpath.read_text()), json.loads(apath.read_text())
        bi, ai = br['identity'], ar['identity']
        if ((bi['dataset'], bi['pair'], bi['condition'], bi['subject'], bi['seed'], bi['smoke']) !=
                (cfg['dataset'], cfg['pair'], 'dml_ds', subject, seed, False) or
                (ai['dataset'], ai['pair'], ai['condition'], ai['subject'], ai['seed'], ai['smoke']) !=
                (cfg['dataset'], cfg['pair'], cfg['condition'], subject, seed, False) or
                bi['protocol'] != 'fewshot_stratified_random_30_70' or
                ai['protocol'] != 'fewshot_stratified_random_30_70' or
                not ai['final_epoch_only'] or ar['evaluation_split'] != 'test' or
                ai['implementation_sha256'] != implementation or
                ai['config_sha256'] != sha256_file(config_path) or
                ai['baseline_result_sha256'] != sha256_file(bpath) or
                ai['baseline_predictions_sha256'] != sha256_file(baseline / 'predictions.npz') or
                ar['predictions_sha256'] != sha256_file(addon / 'predictions.npz')):
            raise ValueError(f'Unmatched result identities: subject={subject}, seed={seed}')
        for key in ('source_manifest_sha256', 'train_uid_sha256', 'test_uid_sha256',
                    'big_training', 'small_training', 'pretrained_sha256', 'epochs', 'batch_size'):
            if bi[key] != ai[key]: raise ValueError(f'Baseline differs in {key}: {bpath}')
        with np.load(baseline / 'predictions.npz', allow_pickle=False) as bp, \
             np.load(addon / 'predictions.npz', allow_pickle=False) as ap:
            if (not np.array_equal(bp['uids'], ap['uids']) or not np.array_equal(bp['y'], ap['y']) or
                    uid_hash(bp['uids'].tolist()) != ai['test_uid_sha256'] or
                    len(set(bp['uids'].tolist())) != len(bp['uids']) or
                    len(bp['uids']) != ai['test_count'] or ai['test_count'] != br['test_count'] or
                    ai['train_count'] != br['train_count']):
                raise ValueError(f'Unmatched test trials: subject={subject}, seed={seed}')
            truth = bp['y']
            for role, name in (('big', 'CBraMod'), ('small', 'EEGNet')):
                for method, pred, record in (('dml_ds', bp, br), ('dml_ds_cross_hfd', ap, ar)):
                    predicted = pred[role]
                    if predicted.shape != truth.shape or not np.isin(predicted, [0, 1]).all():
                        raise ValueError(f'Invalid predictions: subject={subject}, seed={seed}')
                    accuracy = float(np.mean(predicted == truth) * 100)
                    balanced = float(balanced_accuracy_score(truth, predicted) * 100)
                    stored = record[f'{role}_accuracy'] if method == 'dml_ds' else record[role]['accuracy']
                    if not np.isclose(accuracy, stored, rtol=0, atol=1e-9):
                        raise ValueError(f'Prediction metric mismatch: subject={subject}, seed={seed}')
                    if method == 'dml_ds_cross_hfd' and not np.isclose(
                            balanced, record[role]['balanced_accuracy'], rtol=0, atol=1e-9):
                        raise ValueError(f'Balanced accuracy mismatch: subject={subject}, seed={seed}')
                    if method == 'dml_ds_cross_hfd' and record[role]['selected_epoch'] != ai['epochs']:
                        raise ValueError(f'Wrong evaluation epoch: subject={subject}, seed={seed}')
                    row = {'dataset': cfg['dataset'], 'pair': cfg['pair'], 'model': name,
                           'role': role, 'method': method, 'subject': subject, 'seed': seed,
                           'accuracy_percent': accuracy, 'balanced_accuracy_percent': balanced,
                           'train_trials': ai['train_count'], 'test_trials': ai['test_count'],
                           'source': str(bpath if method == 'dml_ds' else apath)}
                    rows.append(row)
                    results[(role, method, subject, seed)] = row
    means = []
    rng = np.random.default_rng(20261126)
    for role, model, metric in product(('big', 'small'), ('CBraMod', 'EEGNet'),
                                        ('accuracy_percent', 'balanced_accuracy_percent')):
        if (role, model) not in (('big', 'CBraMod'), ('small', 'EEGNet')): continue
        for method in ('dml_ds', 'dml_ds_cross_hfd'):
            seed_means = [np.mean([results[(role, method, subject, seed)][metric]
                                   for subject in cfg['subjects']]) for seed in cfg['seeds']]
            means.append({'dataset': cfg['dataset'], 'pair': cfg['pair'], 'model': model,
                          'role': role, 'method': method, 'metric': metric,
                          'mean_percent': float(np.mean(seed_means)),
                          **{f'seed_{seed}': float(value) for seed, value in zip(cfg['seeds'], seed_means)}})
        delta = np.array([
            np.mean([results[(role, 'dml_ds_cross_hfd', subject, seed)][metric] -
                     results[(role, 'dml_ds', subject, seed)][metric] for seed in cfg['seeds']])
            for subject in cfg['subjects']])
        low, high = np.percentile(delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1),
                                  [2.5, 97.5])
        for subject, value in zip(cfg['subjects'], delta):
            subject_differences.append({'dataset': cfg['dataset'], 'pair': cfg['pair'],
                'model': model, 'role': role, 'metric': metric, 'subject': subject, 'delta_pp': float(value)})
        contrasts.append({'dataset': cfg['dataset'], 'pair': cfg['pair'], 'model': model, 'role': role,
                          'metric': metric, 'mean_delta_pp': float(delta.mean()),
                          'ci95_low_pp': float(low), 'ci95_high_pp': float(high),
                          'wins': int((delta > 1e-8).sum()), 'ties': int((np.abs(delta) <= 1e-8).sum()),
                          'losses': int((delta < -1e-8).sum()), 'n_subjects': len(delta),
                          'bootstrap_draws': 10000})
    output = Path(cfg['report_root'])
    write_csv(output / 'subject_seed_results.csv', rows)
    write_csv(output / 'summary.csv', means)
    write_csv(output / 'paired_subject_differences.csv', subject_differences)
    write_csv(output / 'paired_contrasts.csv', contrasts)
    print(f'Validated {len(rows)} baseline/add-on model records; wrote {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    summarize(yaml.safe_load(args.config.read_text()), args.config.resolve())


if __name__ == '__main__':
    main()
