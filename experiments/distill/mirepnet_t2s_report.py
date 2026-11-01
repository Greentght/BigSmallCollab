"""Complete-registry reporting for the MIRepNet T2S all-dataset experiment.

This reader consumes retained metrics only.  It never opens model weights or
starts training, and it refuses an incomplete three-seed/fold registry.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np

from experiments.storage import require_report_output


DATASETS = ('BNCI2014004', 'BNCI2014001', 'BNCI2014001-4', 'BNCI2015001', 'AlexMI')
DATASET_LABELS = dict(zip(DATASETS, ('004', '001', '001-4', '15001', 'AlexMI')))
SUBJECT_COUNTS = dict(zip(DATASETS, (9, 9, 9, 12, 8)))
SEEDS = (666, 667, 668)
STUDENTS = ('ifnet', 'eegnet', 'adfcnn')
STUDENT_LABELS = {'ifnet': 'IFNet', 'eegnet': 'EEGNet', 'adfcnn': 'ADFCNN',
                  'mirepnet': 'MIRepNet'}
METHODS = ('none', 'CE_T2S_FEATURE', 'MCC_T2S_FEATURE', 'CE_T2S_DIFFKD', 'MCC_T2S_DIFFKD')
TEACHERS = ('T0', 'T_CE', 'T_MCC')
METRICS = ('accuracy', 'balanced_accuracy', 'kappa', 'cross_entropy')
PERCENT_METRICS = ('accuracy', 'balanced_accuracy')


def _dataset(value) -> str:
    name = str(value).strip()
    aliases = {key.lower(): key for key in DATASETS}
    aliases.update({label.lower(): name for name, label in DATASET_LABELS.items()})
    if name.lower() not in aliases:
        raise ValueError(f'unknown report dataset {value!r}')
    return aliases[name.lower()]


def _int(value, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{field} must be an integer, not bool')
    integer = int(value)
    if float(value) != integer:
        raise ValueError(f'{field} must be an integer: {value!r}')
    return integer


def _rows_order():
    return [('mirepnet', teacher) for teacher in TEACHERS] + [
        (student, method) for student in STUDENTS for method in METHODS]


def _validate(config: dict, records: list[dict]) -> dict:
    if ({_dataset(value) for value in config.get('datasets', [])} != set(DATASETS)
            or len(config.get('datasets', [])) != len(DATASETS)):
        raise ValueError('final report requires the exact five configured datasets')
    if tuple(sorted(_int(value, 'seed') for value in config.get('seeds', []))) != SEEDS:
        raise ValueError('final report requires seeds 666,667,668 exactly')
    if ({str(value).lower() for value in config.get('students', [])} != set(STUDENTS)
            or len(config.get('students', [])) != len(STUDENTS)):
        raise ValueError('final report requires IFNet, EEGNet and ADFCNN exactly')
    if tuple(config.get('conditions', METHODS)) != METHODS:
        raise ValueError('final report conditions differ from the fixed V1 registry')
    base = _int(config.get('report', {}).get('subject_index_base', 1), 'subject_index_base')
    if base not in (0, 1):
        raise ValueError('subject_index_base must be 0 or 1')
    registry = {}
    for index, source in enumerate(records):
        dataset = _dataset(source['dataset'])
        student = str(source['student']).strip().lower()
        method = str(source['method']).strip()
        if student == 'mirepnet':
            if method not in TEACHERS:
                raise ValueError(f'unknown teacher role {method!r}')
        elif student not in STUDENTS or method not in METHODS:
            raise ValueError(f'unknown report student/method: {student}/{method}')
        seed = _int(source['seed'], 'seed')
        subject = _int(source['subject'], 'subject') + (1 if base == 0 else 0)
        if seed not in SEEDS or subject not in range(1, SUBJECT_COUNTS[dataset] + 1):
            raise ValueError(f'invalid seed/subject for {dataset}: {seed}/{subject}')
        row = {'dataset': dataset, 'student': student, 'method': method,
               'seed': seed, 'subject': subject,
               'n_trials': _int(source['n_trials'], 'n_trials')}
        if row['n_trials'] < 1:
            raise ValueError(f'row {index} has no target trials')
        for metric in METRICS:
            value = float(source[metric])
            if not math.isfinite(value):
                raise ValueError(f'row {index} has nonfinite {metric}')
            if metric in PERCENT_METRICS and not 0.0 <= value <= 1.0:
                raise ValueError(f'input {metric} must be on scale 0–1; row {index} has {value}')
            if metric == 'cross_entropy' and value < 0:
                raise ValueError(f'row {index} has negative cross_entropy')
            row[metric] = value
        key = dataset, student, method, seed, subject
        if key in registry:
            raise ValueError(f'duplicate report fold: {key}')
        registry[key] = row
    expected = {(dataset, student, method, seed, subject)
                for dataset in DATASETS for student, method in _rows_order()
                for seed in SEEDS for subject in range(1, SUBJECT_COUNTS[dataset] + 1)}
    missing = expected - set(registry)
    extra = set(registry) - expected
    if missing or extra:
        example = sorted(missing)[:8]
        raise ValueError(f'incomplete final report: {len(missing)} missing fold records, '
                         f'{len(extra)} unexpected; first missing={example}')
    # Every condition must be evaluated on the same number of target trials.
    for dataset in DATASETS:
        for seed in SEEDS:
            for subject in range(1, SUBJECT_COUNTS[dataset] + 1):
                counts = {registry[(dataset, student, method, seed, subject)]['n_trials']
                          for student, method in _rows_order()}
                if len(counts) != 1:
                    raise ValueError(f'target trial count mismatch for {dataset}/{seed}/{subject}')
    return registry


def _aggregate(registry: dict):
    summary, per_seed, per_subject = [], [], []
    for dataset in DATASETS:
        count = SUBJECT_COUNTS[dataset]
        for student, method in _rows_order():
            seed_scores = {metric: [] for metric in METRICS}
            for seed in SEEDS:
                rows = [registry[(dataset, student, method, seed, subject)]
                        for subject in range(1, count + 1)]
                seed_row = {'dataset': dataset, 'student': STUDENT_LABELS[student],
                            'method': method, 'seed': seed, 'n_subject': count}
                for metric in METRICS:
                    value = float(np.mean([row[metric] for row in rows]))
                    scale = 100.0 if metric in PERCENT_METRICS else 1.0
                    seed_scores[metric].append(value * scale)
                    seed_row[_mean_field(metric)] = value * scale
                per_seed.append(seed_row)
                for row in rows:
                    per_subject.append({'dataset': dataset,
                        'student': STUDENT_LABELS[student], 'method': method,
                        'seed': seed, 'subject': row['subject'], 'n_trials': row['n_trials'],
                        'accuracy_pct': 100.0 * row['accuracy'],
                        'balanced_accuracy_pct': 100.0 * row['balanced_accuracy'],
                        'kappa': row['kappa'], 'cross_entropy': row['cross_entropy']})
            item = {'dataset': dataset, 'dataset_label': DATASET_LABELS[dataset],
                    'student': STUDENT_LABELS[student], 'method': method,
                    'n_seed': 3, 'seed_set': '666,667,668', 'n_subject_per_seed': count,
                    'n_fold_records': 3 * count}
            for metric, values in seed_scores.items():
                item[_mean_field(metric)] = float(statistics.mean(values))
                item[_sd_field(metric)] = float(statistics.stdev(values))
            summary.append(item)
    return summary, per_seed, per_subject


def _mean_field(metric):
    return f'{metric}_mean_pct' if metric in PERCENT_METRICS else f'{metric}_mean'


def _sd_field(metric):
    return f'{metric}_seed_sd_pp' if metric in PERCENT_METRICS else f'{metric}_seed_sd'


def _paired(registry: dict, replicates: int, bootstrap_seed: int):
    comparisons = [(method, 'none') for method in METHODS if method != 'none'] + [
        ('MCC_T2S_FEATURE', 'CE_T2S_FEATURE'),
        ('MCC_T2S_DIFFKD', 'CE_T2S_DIFFKD'),
        ('CE_T2S_DIFFKD', 'CE_T2S_FEATURE'),
        ('MCC_T2S_DIFFKD', 'MCC_T2S_FEATURE')]
    summaries, subject_rows = [], []
    for dataset in DATASETS:
        count = SUBJECT_COUNTS[dataset]
        # Same subject bootstrap table for every comparison/metric in a dataset.
        indices = np.random.default_rng(bootstrap_seed).integers(
            0, count, size=(replicates, count))
        for student in STUDENTS:
            for left, right in comparisons:
                for metric in PERCENT_METRICS:
                    subject_delta = []
                    for subject in range(1, count + 1):
                        differences = [100.0 * (
                            registry[(dataset, student, left, seed, subject)][metric]
                            - registry[(dataset, student, right, seed, subject)][metric])
                            for seed in SEEDS]
                        delta = float(np.mean(differences))
                        subject_delta.append(delta)
                        subject_rows.append({'dataset': dataset,
                            'student': STUDENT_LABELS[student], 'left_method': left,
                            'right_method': right, 'metric': metric, 'subject': subject,
                            'n_seed': 3, 'seed_mean_difference_pp': delta,
                            **{f'seed_{seed}_difference_pp': differences[i]
                               for i, seed in enumerate(SEEDS)}})
                    delta = np.asarray(subject_delta, dtype=np.float64)
                    bootstrap = delta[indices].mean(axis=1)
                    summaries.append({'dataset': dataset,
                        'student': STUDENT_LABELS[student], 'left_method': left,
                        'right_method': right, 'comparison': f'{left} - {right}',
                        'metric': metric, 'unit': 'pp', 'n_seed': 3,
                        'n_subject': count, 'paired_difference_pp': float(delta.mean()),
                        'bootstrap_95ci_low_pp': float(np.quantile(bootstrap, .025)),
                        'bootstrap_95ci_high_pp': float(np.quantile(bootstrap, .975)),
                        'wins': int(np.sum(delta > 1e-6)),
                        'ties': int(np.sum(np.abs(delta) <= 1e-6)),
                        'losses': int(np.sum(delta < -1e-6)),
                        'bootstrap_replicates': replicates,
                        'bootstrap_seed': bootstrap_seed})
    return summaries, subject_rows


def _atomic_csv(path: Path, rows: list[dict]):
    target = require_report_output(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + '.tmp')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, target)


def _atomic_json(path: Path, content: dict):
    target = require_report_output(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + '.tmp')
    temp.write_text(json.dumps(content, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, target)


def _excel(path: Path, summary: list[dict]):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    lookup = {(row['dataset'], row['student'], row['method']): row for row in summary}
    workbook = Workbook()
    workbook.remove(workbook.active)
    for metric, title in (('accuracy', 'Accuracy'), ('balanced_accuracy', 'BA')):
        sheet = workbook.create_sheet(title)
        sheet.sheet_view.showGridLines = False
        sheet.append(['student', 'method', *[DATASET_LABELS[name] for name in DATASETS]])
        for cell in sheet[1]:
            cell.fill = PatternFill('solid', fgColor='1F4E78')
            cell.font = Font(bold=True, color='FFFFFF')
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        for row_index, (student, method) in enumerate(_rows_order(), start=2):
            student_label = STUDENT_LABELS[student]
            values = []
            for dataset in DATASETS:
                item = lookup[(dataset, student_label, method)]
                values.append(f'{item[_mean_field(metric)]:.2f} ± {item[_sd_field(metric)]:.2f}\nn_seed=3')
            sheet.append([student_label, method, *values])
            sheet.row_dimensions[row_index].height = 34
            for cell in sheet[row_index]:
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            if student == 'mirepnet' or method == 'none':
                color = 'FAC090' if student == 'mirepnet' else '93CDDD'
                for cell in sheet[row_index]:
                    cell.fill = PatternFill('solid', fgColor=color)
                continue
            matched_teacher = 'T_MCC' if method.startswith('MCC_') else 'T_CE'
            for column, dataset in enumerate(DATASETS, start=3):
                value = lookup[(dataset, student_label, method)][_mean_field(metric)]
                teacher = lookup[(dataset, 'MIRepNet', matched_teacher)][_mean_field(metric)]
                baseline = lookup[(dataset, student_label, 'none')][_mean_field(metric)]
                above_teacher, above_student = value > teacher, value > baseline
                color = ('FFC7CE' if above_teacher and above_student else
                         'C6EFCE' if above_teacher else 'FFF2CC' if above_student else None)
                if color:
                    sheet.cell(row_index, column).fill = PatternFill('solid', fgColor=color)
        sheet.freeze_panes = 'C2'
        sheet.column_dimensions['A'].width = 18
        sheet.column_dimensions['B'].width = 27
        for column in range(3, 8):
            sheet.column_dimensions[get_column_letter(column)].width = 25
        sheet.row_dimensions[1].height = 26
        legend_row = len(_rows_order()) + 4
        for offset, (label, color) in enumerate((('大于小模型', 'FFF2CC'),
                ('大于大模型', 'C6EFCE'), ('大于二者', 'FFC7CE'),
                ('教师参考', 'FAC090'), ('学生 none 基线', '93CDDD'))):
            cell = sheet.cell(legend_row + offset, 1, label)
            cell.fill = PatternFill('solid', fgColor=color)
            cell.font = Font(bold=True)
        metric_title = 'Accuracy (%)' if metric == 'accuracy' else 'Balanced Accuracy (%)'
        notes = [
            f'指标：{metric_title}；单元格为 mean ± sample SD，百分比 0–100、保留两位小数；所有单元格 n_seed=3。',
            '聚合：每 seed 内等权平均全部 LOSO 目标被试，再对 seed 666/667/668 等权平均；SD 为三个 seed 均值的样本 SD。',
            '各 seed 被试数：004=9，001=9，001-4=9，15001=12，AlexMI=8；001-4 为四分类，其余为二分类。',
            '同一学生组内 none 为本次匹配 CE 参考。CE_* 方法比较同格 T_CE，MCC_* 方法比较同格 T_MCC；着色使用未四舍五入均值的严格 >。',
            '教师为 MIRepNet；冻结教师的原生分类头输入投影到学生原生分类维度，再做无仿射 LayerNorm。扩维 P 满足 PᵀP=I，降维 P 满足 PPᵀ=I。',
            'FEATURE：CE + 0.5 同 UID 特征 MSE；DIFFKD：CE + 0.5 冻结五步去噪后同 UID 特征 MSE；从 epoch1 开启。',
            '教师 T0/T_CE/T_MCC 指标独立列出。当前标准 LOSO 与历史 7/1/1 pilot 协议不同，旧成绩不构成严格配对参考。',
            '配对 bootstrap：先对每个被试的差值跨 seed 平均，再重采样被试 10000 次；95% CI 为探索性描述，不把 seed×subject 当独立被试。',
        ]
        start = legend_row + 6
        for offset, note in enumerate(notes):
            sheet.merge_cells(start_row=start + offset, start_column=1,
                              end_row=start + offset, end_column=7)
            cell = sheet.cell(start + offset, 1, note)
            cell.alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
            sheet.row_dimensions[start + offset].height = 31
    target = require_report_output(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + '.tmp.xlsx')
    workbook.save(temporary)
    os.replace(temporary, target)


def _conclusions(path: Path, summary: list[dict], paired: list[dict]):
    lines = ['MIRepNet T2S：五数据集 × 三学生 × 三 seed；每格完成全部 LOSO 被试。',
             'Accuracy/BA 分别报告；均值按被试等权后按 seed 等权，±为三 seed 均值的样本 SD。']
    for student in STUDENTS:
        label = STUDENT_LABELS[student]
        deltas = [row['paired_difference_pp'] for row in paired
                  if row['student'] == label and row['metric'] == 'balanced_accuracy'
                  and row['right_method'] == 'none']
        lines.append(f'{label}：20 个 BA 方法−CE 比较中 {sum(d > 1e-6 for d in deltas)} 个正差值，'
                     f'{sum(d < -1e-6 for d in deltas)} 个负差值；各格 n_seed=3。')
    for left, right, label in (
            ('MCC_T2S_FEATURE', 'CE_T2S_FEATURE', 'MCC 在 FEATURE 路径的贡献'),
            ('MCC_T2S_DIFFKD', 'CE_T2S_DIFFKD', 'MCC 在 DIFFKD 路径的贡献'),
            ('CE_T2S_DIFFKD', 'CE_T2S_FEATURE', 'CE 教师下去噪流程的贡献'),
            ('MCC_T2S_DIFFKD', 'MCC_T2S_FEATURE', 'MCC 教师下去噪流程的贡献')):
        values = [row['paired_difference_pp'] for row in paired
                  if row['left_method'] == left and row['right_method'] == right
                  and row['metric'] == 'balanced_accuracy']
        lines.append(f'{label}：15 个数据集×学生比较中 {sum(v > 1e-6 for v in values)} 个正差值、'
                     f'{sum(v < -1e-6 for v in values)} 个负差值；逐格差值与被试 bootstrap CI 见 CSV。')
    lines.append('单格 CI 与胜/平/负按真实被试计算；跨任务正差值计数不构成通用有效性的统计检验。')
    target = require_report_output(path)
    target.write_text('\n'.join(lines[:10]) + '\n', encoding='utf-8')


def report(config: dict, records: list[dict], output_dir: Path) -> dict:
    """Validate all 2,538 fold metrics, then write self-contained final reports."""
    registry = _validate(config, records)
    summary, seed_means, subject_metrics = _aggregate(registry)
    replicates = _int(config.get('report', {}).get('bootstrap_replicates', 10000), 'bootstrap_replicates')
    bootstrap_seed = _int(config.get('report', {}).get('bootstrap_seed', 20261101), 'bootstrap_seed')
    if replicates != 10000 or bootstrap_seed != 20261101:
        raise ValueError('V1 report requires 10000 bootstrap samples with seed 20261101')
    paired, paired_subjects = _paired(registry, replicates, bootstrap_seed)
    folder = Path(output_dir)
    outputs = {
        'metrics': folder / 'metrics.csv', 'seed_means': folder / 'seed_means.csv',
        'subject_metrics': folder / 'subject_metrics.csv',
        'paired_differences': folder / 'paired_differences.csv',
        'paired_subject_differences': folder / 'paired_subject_differences.csv',
        'summary_json': folder / 'summary.json', 'summary_excel': folder / 'summary.xlsx',
        'conclusions': folder / 'conclusions.md'}
    # Validate every destination before writing any report file.
    outputs = {name: require_report_output(path) for name, path in outputs.items()}
    outputs['summary_json'].parent.mkdir(parents=True, exist_ok=True)
    for key, rows in (('metrics', summary), ('seed_means', seed_means),
                      ('subject_metrics', subject_metrics), ('paired_differences', paired),
                      ('paired_subject_differences', paired_subjects)):
        _atomic_csv(outputs[key], rows)
    _excel(outputs['summary_excel'], summary)
    _conclusions(outputs['conclusions'], summary, paired)
    result = {'status': 'complete', 'experiment_id': config.get('experiment_id'),
        'datasets': list(DATASETS), 'students': [STUDENT_LABELS[name] for name in STUDENTS],
        'seeds': list(SEEDS), 'subject_counts': SUBJECT_COUNTS,
        'fold_registry_count': len(registry), 'metric_cells': len(summary),
        'aggregation': 'equal_subject_mean_within_seed_then_equal_seed_mean',
        'sd': 'sample_SD_of_three_seed_means', 'percentage_scale': '0–100',
        'paired_unit': 'subject_after_equal_mean_over_seeds',
        'bootstrap_replicates': replicates, 'bootstrap_seed': bootstrap_seed,
        'metrics': summary, 'paired_differences': paired,
        'files': {name: str(path) for name, path in outputs.items()}}
    _atomic_json(outputs['summary_json'], result)
    return result
