"""Validate matched few-shot results and export complete groups only."""
from __future__ import annotations

import argparse
import csv
from itertools import product
import json
import os
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import balanced_accuracy_score

from data.loso_source import sha256_file
from experiments.bidir.run_dml_ds_cross_hfd_all import (
    CONFIG, ROOT, baseline_dir, check_reused_training_config, implementation_hashes, uid_hash,
)
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import implementation_hashes as previous_hashes
from experiments.storage import require_report_output


def addon_dir(cfg, dataset, pair, subject, seed):
    if (dataset, pair) == ('BNCI2014004', 'cbramod_eegnet'):
        return (Path(cfg['existing_004_cbramod_eegnet']) / 'runs' /
                f'subject_{subject}' / f'seed_{seed}'), True
    return (Path(cfg['output_root']) / 'runs' / dataset / pair /
            f'subject_{subject}' / f'seed_{seed}'), False


def write_csv(path, rows, fields):
    path = require_report_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    try:
        with tmp.open('w', encoding='utf8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def validated_pair(cfg, base, config_path, dataset, pair, subject, seed, hashes, old_hashes):
    baseline = baseline_dir(dataset, pair, subject, seed, base)
    addon, previous = addon_dir(cfg, dataset, pair, subject, seed)
    bpath, apath = baseline / 'evaluation.json', addon / 'evaluation.json'
    if not (bpath.is_file() and (baseline / 'predictions.npz').is_file() and
            apath.is_file() and (addon / 'predictions.npz').is_file() and (addon / 'training.json').is_file()):
        return None
    br, ar = json.loads(bpath.read_text()), json.loads(apath.read_text())
    bi, ai = br['identity'], ar['identity']
    expected = (dataset, pair, subject, seed, 'fewshot_stratified_random_30_70', False)
    keys = ('dataset', 'pair', 'subject', 'seed', 'protocol', 'smoke')
    if tuple(bi[k] for k in keys) != expected or tuple(ai[k] for k in keys) != expected:
        raise ValueError(f'Pair identity mismatch: {dataset}/{pair}/{subject}/{seed}')
    expected_hashes = old_hashes if previous else hashes
    expected_config = ROOT / 'configs/experiments/fewshot_dml_ds_cross_hfd_004.yaml' if previous else config_path
    if (bi['condition'] != 'dml_ds' or ai['condition'] != cfg['condition'] or
            ai.get('hfd_method') != 'paper_eq2_native_peer_operators_eeg_adaptation' or
            ai['implementation_sha256'] != expected_hashes or ai['config_sha256'] != sha256_file(expected_config) or
            ai['baseline_result_sha256'] != sha256_file(bpath) or
            ai['baseline_predictions_sha256'] != sha256_file(baseline / 'predictions.npz') or
            ar['predictions_sha256'] != sha256_file(addon / 'predictions.npz') or
            ar['evaluation_split'] != 'test' or not ai['final_epoch_only']):
        raise ValueError(f'Pair provenance mismatch: {dataset}/{pair}/{subject}/{seed}')
    for key in ('source_manifest_sha256', 'train_uid_sha256', 'test_uid_sha256',
                'pretrained_sha256', 'epochs', 'batch_size'):
        if bi[key] != ai[key]:
            raise ValueError(f'Baseline differs in {key}: {dataset}/{pair}/{subject}/{seed}')
    big, small = pair.split('_')
    if ai['big_training'] != base['teacher_training'][big] or ai['small_training'] != base['student_training'][small]:
        raise ValueError(f'Add-on training differs: {dataset}/{pair}/{subject}/{seed}')
    if pair in base['reuse_pairs']:
        check_reused_training_config(dataset, pair, base)
    elif bi['big_training'] != ai['big_training'] or bi['small_training'] != ai['small_training']:
        raise ValueError(f'Baseline training differs: {dataset}/{pair}/{subject}/{seed}')
    history = json.loads((addon / 'training.json').read_text())
    if history.get('identity') != ai or len(history.get('history', [])) != ai['epochs']:
        raise ValueError(f'Training history mismatch: {addon}')
    with np.load(baseline / 'predictions.npz', allow_pickle=False) as bp, \
         np.load(addon / 'predictions.npz', allow_pickle=False) as ap:
        if (not np.array_equal(bp['uids'], ap['uids']) or not np.array_equal(bp['y'], ap['y']) or
                uid_hash(bp['uids'].tolist()) != ai['test_uid_sha256'] or
                len(set(bp['uids'].tolist())) != len(bp['uids']) or len(bp['uids']) != ai['test_count'] or
                ai['train_count'] != br['train_count'] or ai['test_count'] != br['test_count']):
            raise ValueError(f'Test UID or labels mismatch: {dataset}/{pair}/{subject}/{seed}')
        truth = bp['y']
        records = {}
        for role in ('big', 'small'):
            for method, pred, record in (('dml_ds', bp, br), ('dml_ds_cross_hfd', ap, ar)):
                predicted = pred[role]
                if predicted.shape != truth.shape:
                    raise ValueError(f'Prediction shape mismatch: {dataset}/{pair}/{subject}/{seed}')
                acc = float(np.mean(predicted == truth) * 100)
                ba = float(balanced_accuracy_score(truth, predicted) * 100)
                stored = record[role + '_accuracy'] if method == 'dml_ds' else record[role]['accuracy']
                if not np.isclose(acc, stored, rtol=0, atol=1e-9):
                    raise ValueError(f'Prediction metric mismatch: {dataset}/{pair}/{subject}/{seed}')
                if method != 'dml_ds' and (not np.isclose(ba, record[role]['balanced_accuracy'], rtol=0, atol=1e-9)
                                           or record[role]['selected_epoch'] != ai['epochs']):
                    raise ValueError(f'Add-on evaluation mismatch: {dataset}/{pair}/{subject}/{seed}')
                records[(role, method)] = {'accuracy_percent': acc, 'balanced_accuracy_percent': ba}
    return records, str(bpath), str(apath), previous


def supervised_mean(dataset, pair, count, base, seeds):
    values = []
    for subject, seed in product(range(count), seeds):
        dml = baseline_dir(dataset, pair, subject, seed, base)
        baseline = dml.parents[2] / 'none' / f'subject_{subject}' / f'seed_{seed}'
        path = baseline / 'evaluation.json'
        if not path.is_file() or not (baseline / 'predictions.npz').is_file():
            return None
        record = json.loads(path.read_text())
        i = record['identity']
        if ((i['dataset'], i['pair'], i['condition'], i['subject'], i['seed'], i['protocol'], i['smoke']) !=
                (dataset, pair, 'none', subject, seed, 'fewshot_stratified_random_30_70', False)):
            raise ValueError(f'Supervised baseline identity mismatch: {path}')
        match = json.loads((dml / 'evaluation.json').read_text())['identity']
        for key in ('source_manifest_sha256', 'train_uid_sha256', 'test_uid_sha256',
                    'pretrained_sha256', 'epochs', 'batch_size'):
            if i[key] != match[key]:
                raise ValueError(f'Supervised baseline differs in {key}: {path}')
        with np.load(baseline / 'predictions.npz', allow_pickle=False) as pred:
            acc = float(np.mean(pred['small'] == pred['y']) * 100)
            if uid_hash(pred['uids'].tolist()) != i['test_uid_sha256'] or not np.isclose(
                    acc, record['small_accuracy'], rtol=0, atol=1e-9):
                raise ValueError(f'Supervised predictions mismatch: {path}')
            values.append(acc)
    return float(np.mean(values))


def make_workbook(cfg, means, supervised, output):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)
    navy, white = '1F4E78', 'FFFFFF'
    orange, blue = 'FAC090', '93CDDD'
    red, green, yellow = 'FFC7CE', 'C6EFCE', 'FFF2CC'
    datasets = [('004', 'BNCI2014004'), ('001', 'BNCI2014001'), ('001-4', 'BNCI2014001-4'),
                ('15001', 'BNCI2015001'), ('AlexMI', 'AlexMI')]
    for big, small in product(cfg['teachers'], cfg['students']):
        pair = big + '_' + small
        ws = wb.create_sheet(pair)
        ws.sheet_view.showGridLines = False
        ws.append(['student', 'method'] + [name for name, _ in datasets])
        rows = [(big, 'dml_ds', 'big', 'dml_ds'), (big, 'dml_ds+hfd', 'big', 'dml_ds_cross_hfd'),
                (small, 'none', 'small', 'none'), (small, 'dml_ds', 'small', 'dml_ds'),
                (small, 'dml_ds+hfd', 'small', 'dml_ds_cross_hfd')]
        for model, label, role, method in rows:
            ws.append([model, label] + [supervised.get((dataset, pair)) if method == 'none' else
                       means.get((dataset, pair, role, method, 'accuracy_percent')) for _, dataset in datasets])
        for cell in ws[1]:
            cell.fill = PatternFill('solid', fgColor=navy)
            cell.font = Font(bold=True, color=white)
        for row in ws.iter_rows(min_row=2, max_row=6):
            for cell in row:
                cell.alignment = Alignment(horizontal='center', vertical='center')
            for cell in row[2:]:
                cell.number_format = '0.00'
            if row[0].row in (2, 3):
                for cell in row: cell.fill = PatternFill('solid', fgColor=orange)
            elif row[0].row == 4:
                for cell in row: cell.fill = PatternFill('solid', fgColor=blue)
        for col, (_, dataset) in enumerate(datasets, start=3):
            baseline = supervised.get((dataset, pair))
            for rownum, method in ((5, 'dml_ds'), (6, 'dml_ds_cross_hfd')):
                score = means.get((dataset, pair, 'small', method, 'accuracy_percent'))
                teacher = means.get((dataset, pair, 'big', method, 'accuracy_percent'))
                if score is None or teacher is None or baseline is None:
                    continue
                over_big, over_small = score > teacher, score > baseline
                color = red if over_big and over_small else green if over_big else yellow if over_small else None
                if color: ws.cell(rownum, col).fill = PatternFill('solid', fgColor=color)
        ws['A8'], ws['B8'] = '颜色说明', '大于小模型'
        ws['B8'].fill = PatternFill('solid', fgColor=yellow)
        ws['C8'] = '大于大模型'
        ws['C8'].fill = PatternFill('solid', fgColor=green)
        ws['D8'] = '大于二者'
        ws['D8'].fill = PatternFill('solid', fgColor=red)
        ws.merge_cells('A10:G10')
        ws['A10'] = ('Accuracy (%)；每个种子先对被试等权平均，再对 666、667、668 等权平均。'
                     '004/001/001-4 为 9 人，15001 为 12 人，AlexMI 为 8 人。')
        ws.merge_cells('A11:G11')
        ws['A11'] = ('同一被试按 30% 校准、70% 测试；none 为同配对监督基线。'
                     'HFD 权重 0.1，第 6–10 轮渐进开启；空白表示对应组尚未全部完成。')
        for number in (10, 11):
            ws.cell(number, 1).alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
            ws.row_dimensions[number].height = 30
        for col in range(1, 8):
            ws.column_dimensions[get_column_letter(col)].width = 17 if col <= 2 else 15
        ws.freeze_panes = 'C2'
    target = require_report_output(output)
    tmp = target.with_name('.' + target.name + f'.{os.getpid()}.tmp.xlsx')
    try:
        wb.save(tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def summarize(cfg, base, config_path, allow_partial=False):
    hashes, old_hashes = implementation_hashes(), previous_hashes()
    rows, group_status, means_rows, differences, contrasts = [], [], [], [], []
    means, supervised = {}, {}
    for dataset, (big, small) in product(cfg['datasets'], product(cfg['teachers'], cfg['students'])):
        pair = big + '_' + small
        n = cfg['datasets'][dataset]
        collected = {}
        for subject, seed in product(range(n), cfg['seeds']):
            value = validated_pair(cfg, base, config_path, dataset, pair, subject, seed, hashes, old_hashes)
            if value is not None: collected[(subject, seed)] = value
        group_status.append({'dataset': dataset, 'pair': pair, 'completed': len(collected), 'expected': n * len(cfg['seeds'])})
        if len(collected) != n * len(cfg['seeds']):
            if allow_partial: continue
            raise ValueError(f'Incomplete group {dataset}/{pair}: {len(collected)}/{n * len(cfg["seeds"])}')
        supervised[(dataset, pair)] = supervised_mean(dataset, pair, n, base, cfg['seeds'])
        for (subject, seed), (records, bpath, apath, reused) in collected.items():
            for role, method in product(('big', 'small'), ('dml_ds', 'dml_ds_cross_hfd')):
                rows.append({'dataset': dataset, 'pair': pair, 'model': big if role == 'big' else small,
                             'role': role, 'method': method, 'subject': subject, 'seed': seed,
                             **records[(role, method)], 'reused_prior_hfd': reused and method != 'dml_ds',
                             'source': bpath if method == 'dml_ds' else apath})
        rng = np.random.default_rng(20261126)
        for role, metric in product(('big', 'small'), ('accuracy_percent', 'balanced_accuracy_percent')):
            for method in ('dml_ds', 'dml_ds_cross_hfd'):
                seed_means = [np.mean([collected[(s, seed)][0][(role, method)][metric] for s in range(n)])
                              for seed in cfg['seeds']]
                score = float(np.mean(seed_means))
                means[(dataset, pair, role, method, metric)] = score
                means_rows.append({'dataset': dataset, 'pair': pair, 'model': big if role == 'big' else small,
                                   'role': role, 'method': method, 'metric': metric, 'mean_percent': score,
                                   **{f'seed_{seed}': float(x) for seed, x in zip(cfg['seeds'], seed_means)}})
            delta = np.asarray([np.mean([collected[(s, seed)][0][(role, 'dml_ds_cross_hfd')][metric] -
                                         collected[(s, seed)][0][(role, 'dml_ds')][metric]
                                         for seed in cfg['seeds']]) for s in range(n)])
            low, high = np.percentile(delta[rng.integers(0, n, size=(10000, n))].mean(axis=1), [2.5, 97.5])
            differences.extend({'dataset': dataset, 'pair': pair, 'role': role, 'metric': metric,
                                'subject': s, 'delta_pp': float(x)} for s, x in enumerate(delta))
            contrasts.append({'dataset': dataset, 'pair': pair, 'model': big if role == 'big' else small,
                              'role': role, 'metric': metric, 'mean_delta_pp': float(delta.mean()),
                              'ci95_low_pp': float(low), 'ci95_high_pp': float(high),
                              'wins': int((delta > 1e-8).sum()), 'ties': int((np.abs(delta) <= 1e-8).sum()),
                              'losses': int((delta < -1e-8).sum()), 'n_subjects': n, 'bootstrap_draws': 10000})
    output = Path(cfg['report_root'])
    write_csv(output / 'group_status.csv', group_status, ['dataset', 'pair', 'completed', 'expected'])
    write_csv(output / 'subject_seed_results.csv', rows, ['dataset', 'pair', 'model', 'role', 'method', 'subject',
              'seed', 'accuracy_percent', 'balanced_accuracy_percent', 'reused_prior_hfd', 'source'])
    write_csv(output / 'summary.csv', means_rows, ['dataset', 'pair', 'model', 'role', 'method', 'metric',
              'mean_percent', 'seed_666', 'seed_667', 'seed_668'])
    write_csv(output / 'paired_subject_differences.csv', differences,
              ['dataset', 'pair', 'role', 'metric', 'subject', 'delta_pp'])
    write_csv(output / 'paired_contrasts.csv', contrasts,
              ['dataset', 'pair', 'model', 'role', 'metric', 'mean_delta_pp', 'ci95_low_pp', 'ci95_high_pp',
               'wins', 'ties', 'losses', 'n_subjects', 'bootstrap_draws'])
    make_workbook(cfg, means, supervised, output / 'dml_ds_hfd_all_pairs.xlsx')
    print(f'Validated {sum(x["completed"] for x in group_status)} paired units; reports at {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    args.config = args.config.resolve()
    cfg = yaml.safe_load(args.config.read_text())
    base = yaml.safe_load((ROOT / cfg['baseline_config']).read_text())
    summarize(cfg, base, args.config, args.allow_partial)


if __name__ == '__main__':
    main()
