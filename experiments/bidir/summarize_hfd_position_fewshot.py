"""Export complete subject/seed matched HFD-position comparisons."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['BIGSMALLCOLLAB_CPU_THREADS'] = '1'

import argparse
import csv
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import yaml
from sklearn.metrics import balanced_accuracy_score

from data.loso_source import sha256_file
from experiments.bidir.run_dml_ds_cross_hfd_all import baseline_dir
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import uid_hash
from experiments.bidir.run_hfd_position_baselines import BASE, DATASETS, OUTPUT, PAIRS, ROOT
from experiments.bidir.summarize_dml_ds_cross_hfd_all import addon_dir, write_csv
from experiments.storage import require_report_output

REPORT = ROOT / 'results/hfd_position_fewshot_v1'
SEEDS = (666, 667, 668)
METHODS = ('CE20', 'CE100', 'C1', 'C0', 'C2', 'C3', 'C4', 'C5', 'C6')


def find_folder(dataset, pair, subject, seed, method, cfg, old_cfg):
    big = pair.split('_')[0]
    normal_epochs = 100 if big == 'neurogpt' else 20
    if method in ('CE20', 'CE100'):
        epochs = int(method[2:])
        if big != 'neurogpt' and epochs == normal_epochs:
            dml = baseline_dir(dataset, pair, subject, seed, cfg)
            return dml.parents[2] / 'none' / f'subject_{subject}' / f'seed_{seed}'
        return OUTPUT / 'runs' / dataset / pair / 'ce' / str(epochs) / f'subject_{subject}' / f'seed_{seed}'
    if method == 'C1':
        return OUTPUT / 'runs' / dataset / pair / 'ds' / str(normal_epochs) / f'subject_{subject}' / f'seed_{seed}'
    if method == 'C0' and big != 'neurogpt':
        return baseline_dir(dataset, pair, subject, seed, cfg)
    if method == 'C2' and big != 'neurogpt':
        return addon_dir(old_cfg, dataset, pair, subject, seed)[0]
    return OUTPUT / 'runs' / dataset / pair / method / f'subject_{subject}' / f'seed_{seed}'


def read_result(folder, expected):
    result_path, pred_path = folder / 'evaluation.json', folder / 'predictions.npz'
    if not result_path.is_file() or not pred_path.is_file():
        return None
    result = json.loads(result_path.read_text())
    ident = result['identity']
    dataset, pair, subject, seed = expected
    if (ident['dataset'], ident['pair'], ident['subject'], ident['seed']) != expected:
        raise ValueError(f'Wrong result identity: {folder}')
    if result.get('predictions_sha256') and sha256_file(pred_path) != result['predictions_sha256']:
        raise ValueError(f'Prediction digest differs: {folder}')
    with np.load(pred_path, allow_pickle=False) as pred:
        uids = pred['uids'].astype(str)
        truth = pred['y']
        if uid_hash(uids.tolist()) != ident['test_uid_sha256'] or len(set(uids.tolist())) != len(uids):
            raise ValueError(f'Test UID identity differs: {folder}')
        metrics = {role: float(balanced_accuracy_score(truth, pred[role]) * 100)
                   for role in ('big', 'small')}
        return {'identity': ident, 'uids': uids, 'truth': truth, 'metrics': metrics,
                'path': str(result_path)}


def write_xlsx(rows, ranking, differences, budgets, output):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)
    for name, entries in (('被试种子结果', rows), ('位置排名', ranking),
                          ('配对增益', differences), ('独立学生预算', budgets)):
        ws = wb.create_sheet(name)
        ws.sheet_view.showGridLines = False
        if not entries:
            ws.append(['尚无完整数据'])
            continue
        keys = list(entries[0])
        ws.append(keys)
        for entry in entries:
            ws.append([entry[k] for k in keys])
        for cell in ws[1]:
            cell.fill = PatternFill('solid', fgColor='1F4E78')
            cell.font = Font(bold=True, color='FFFFFF')
            cell.alignment = Alignment(horizontal='center')
        ws.freeze_panes = 'D2'
        for col, key in enumerate(keys, 1):
            ws.column_dimensions[get_column_letter(col)].width = min(36, max(14, len(key) + 2))
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(horizontal='center')
                if isinstance(cell.value, float):
                    cell.number_format = '0.00'
    target = require_report_output(output)
    temp = target.with_name('.' + target.name + f'.{os.getpid()}.tmp.xlsx')
    try:
        wb.save(temp)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def summarize():
    cfg = yaml.safe_load(BASE.read_text())
    old_cfg = yaml.safe_load((ROOT / 'configs/experiments/fewshot_dml_ds_cross_hfd_all.yaml').read_text())
    rows, ranking, differences, subject_differences, budgets, statuses = [], [], [], [], [], []
    for dataset, pair in product(DATASETS, PAIRS):
        cells = {}
        for subject, seed in product(range(9), SEEDS):
            expected = (dataset, pair, subject, seed)
            available = {method: read_result(find_folder(dataset, pair, subject, seed, method, cfg, old_cfg), expected)
                         for method in METHODS}
            reference = available['C0']
            if reference:
                for method, record in available.items():
                    if record is None:
                        continue
                    if (not np.array_equal(reference['uids'], record['uids']) or
                            not np.array_equal(reference['truth'], record['truth']) or
                            reference['identity']['train_uid_sha256'] != record['identity']['train_uid_sha256'] or
                            reference['identity']['source_manifest_sha256'] != record['identity']['source_manifest_sha256']):
                        raise ValueError(f'Unmatched trial identity: {dataset}/{pair}/{subject}/{seed}/{method}')
                    for role in ('big', 'small'):
                        rows.append({'dataset': dataset, 'pair': pair, 'subject': subject + 1, 'seed': seed,
                                     'role': role, 'condition': method,
                                     'balanced_accuracy_percent': record['metrics'][role],
                                     'source': record['path']})
            cells[(subject, seed)] = available
        counts = {method: sum(v[method] is not None for v in cells.values()) for method in METHODS}
        statuses.append({'dataset': dataset, 'pair': pair, **counts, 'expected_per_condition': 27})
        rng = np.random.default_rng(20261127)
        bootstrap = rng.integers(0, 9, size=(10000, 9))
        for role in ('big', 'small'):
            for method, ref in [(m, 'C0') for m in ('C2', 'C3', 'C4', 'C5', 'C6')] + [
                    ('C1', 'CE100' if pair.startswith('neurogpt_') else 'CE20'), ('C0', 'C1')]:
                if any(cells[(s, z)][method] is None or cells[(s, z)][ref] is None
                       for s, z in product(range(9), SEEDS)):
                    continue
                delta = np.array([[cells[(s, z)][method]['metrics'][role] -
                                   cells[(s, z)][ref]['metrics'][role] for z in SEEDS] for s in range(9)])
                subject_means = delta.mean(axis=1)
                low, high = np.percentile(subject_means[bootstrap].mean(axis=1), [2.5, 97.5])
                for subject, value in enumerate(subject_means):
                    subject_differences.append({
                        'dataset': dataset, 'pair': pair, 'role': role, 'subject': subject + 1,
                        'condition': method, 'reference': ref, 'mean_delta_pp': float(value),
                        **{f'seed_{seed}_delta_pp': float(delta[subject, index])
                           for index, seed in enumerate(SEEDS)},
                    })
                differences.append({'dataset': dataset, 'pair': pair, 'role': role, 'condition': method,
                                    'reference': ref, 'mean_delta_pp': float(delta.mean()),
                                    'seed_sd_pp': float(delta.mean(axis=0).std(ddof=1)),
                                    'ci_low_pp': float(low), 'ci_high_pp': float(high),
                                    'ci_half_width_pp': float((high - low) / 2),
                                    'subjects_improved': int((subject_means > 1e-8).sum()),
                                    'subjects_equal': int((abs(subject_means) <= 1e-8).sum()),
                                    'subjects_declined': int((subject_means < -1e-8).sum())})
            position_rows = [r for r in differences if r['dataset'] == dataset and r['pair'] == pair and
                             r['role'] == role and r['condition'] in ('C2', 'C3', 'C4', 'C5')]
            if len(position_rows) == 4:
                sorted_rows = sorted(position_rows, key=lambda r: -r['mean_delta_pp'])
                spread = sorted_rows[0]['mean_delta_pp'] - sorted_rows[-1]['mean_delta_pp']
                for rank, row in enumerate(sorted_rows, 1):
                    tied_rank = 1 + sum(other['mean_delta_pp'] > row['mean_delta_pp'] + 1e-8
                                        for other in sorted_rows)
                    ranking.append({'dataset': dataset, 'pair': pair, 'role': role, 'rank': tied_rank,
                                    'condition': row['condition'], 'mean_delta_pp': row['mean_delta_pp'],
                                    'position_spread_pp': spread, 'seed_sd_pp': row['seed_sd_pp'],
                                    'ci_half_width_pp': row['ci_half_width_pp']})
        if all(cells[(s, z)]['CE20'] and cells[(s, z)]['CE100'] for s, z in product(range(9), SEEDS)):
            for method in ('CE20', 'CE100'):
                values = np.array([[cells[(s, z)][method]['metrics']['small'] for z in SEEDS] for s in range(9)])
                budgets.append({'dataset': dataset, 'pair': pair, 'condition': method,
                                'student_balanced_accuracy_percent': float(values.mean()),
                                'seed_sd_pp': float(values.mean(axis=0).std(ddof=1))})
    consistency = []
    for dataset, role in product(DATASETS, ('big', 'small')):
        winners = {pair: [r['condition'] for r in ranking if r['dataset'] == dataset and
                          r['pair'] == pair and r['role'] == role and r['rank'] == 1] for pair in PAIRS}
        if all(winners.values()):
            common = set(winners[PAIRS[0]]).intersection(*(set(winners[pair]) for pair in PAIRS[1:]))
            consistency.append({'dataset': dataset, 'role': role,
                                **{pair: ','.join(winners[pair]) for pair in PAIRS},
                                'common_best_positions': ','.join(sorted(common)) or 'none'})
    REPORT.mkdir(parents=True, exist_ok=True)
    for name, entries, fields in (
            ('subject_seed_results.csv', rows, ['dataset', 'pair', 'subject', 'seed', 'role', 'condition',
                                                'balanced_accuracy_percent', 'source']),
            ('paired_differences.csv', differences, ['dataset', 'pair', 'role', 'condition', 'reference',
                'mean_delta_pp', 'seed_sd_pp', 'ci_low_pp', 'ci_high_pp', 'ci_half_width_pp',
                'subjects_improved', 'subjects_equal', 'subjects_declined']),
            ('subject_paired_differences.csv', subject_differences, ['dataset', 'pair', 'role', 'subject',
                'condition', 'reference', 'mean_delta_pp', 'seed_666_delta_pp', 'seed_667_delta_pp',
                'seed_668_delta_pp']),
            ('position_ranking.csv', ranking, ['dataset', 'pair', 'role', 'rank', 'condition', 'mean_delta_pp',
                                                'position_spread_pp', 'seed_sd_pp', 'ci_half_width_pp']),
            ('position_consistency.csv', consistency, ['dataset', 'role', *PAIRS, 'common_best_positions']),
            ('student_budgets.csv', budgets, ['dataset', 'pair', 'condition',
                                              'student_balanced_accuracy_percent', 'seed_sd_pp']),
            ('group_status.csv', statuses, ['dataset', 'pair', *METHODS, 'expected_per_condition'])):
        write_csv(REPORT / name, entries, fields)
    write_xlsx(rows, ranking, differences, budgets, REPORT / 'hfd_position_summary.xlsx')
    print(f'Exported {len(rows)} subject-seed results, {len(differences)} contrasts', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    with (REPORT / 'summary.lock').open('a') as lock:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            summarize()
            if not args.watch:
                break
            queues = [REPORT / 'baseline_queue.json', REPORT / 'position_queue.json']
            if all(p.is_file() and json.loads(p.read_text())['stage'] in ('completed', 'error') for p in queues):
                break
            time.sleep(60)


if __name__ == '__main__':
    main()
