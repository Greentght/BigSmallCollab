"""Validate retained predictions and export complete-group comparisons."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from scipy.stats import t
from sklearn.metrics import balanced_accuracy_score
import yaml

from data.loso_source import sha256_file
from experiments.bidir.run_cross_domain_features_fewshot import CONFIG, ROOT, fingerprint, run_folder
from experiments.bidir.run_dml_ds_cross_hfd_all import baseline_dir
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import atomic_json, uid_hash
from experiments.storage import REPORTS_ROOT, require_report_output

NAMES = {'none': 'none', 'dml_ds': 'DML-DS', 'semckd': 'DML-DS + SemCKD',
         'semckd_uniform': 'DML-DS + SemCKD 等权', 'at': 'DML-DS + AT'}
METHODS = ('none', 'dml_ds', 'semckd_uniform', 'semckd', 'at')


def read_result(folder, dataset, method, subject, seed, cfg, base, impl, config_sha):
    path = folder / 'evaluation.json'
    if not path.is_file(): return []
    doc = json.loads(path.read_text())
    ident = doc['identity']
    bt, st = base['teacher_training']['cbramod'], base['student_training']['eegnet']
    if (ident['dataset'] != dataset or ident['pair'] != cfg['pair'] or
            ident['subject'] != subject or ident['seed'] != seed or ident['smoke'] or
            ident['protocol'] != 'fewshot_stratified_random_30_70' or
            ident['epochs'] != bt['epochs'] or ident['batch_size'] != bt['batch_size'] or
            ident['big_training'] != bt or ident['small_training'] != st):
        raise ValueError(f'Unmatched result: {folder}')
    if method in cfg['methods']:
        if (ident['method'] != method or ident['implementation_sha256'] != impl or
                ident['config_sha256'] != config_sha or doc['evaluation_split'] != 'test' or
                doc['predictions_sha256'] != sha256_file(folder / 'predictions.npz')):
            raise ValueError(f'Invalid new result: {folder}')
        baseline = ident['baseline']
        if (sha256_file(baseline['result_path']) != baseline['result_sha256'] or
                sha256_file(Path(baseline['result_path']).parent / 'predictions.npz') != baseline['predictions_sha256']):
            raise ValueError('Matched baseline artifacts changed')
    elif ident['condition'] != method:
        raise ValueError('Incorrect baseline condition')
    with np.load(folder / 'predictions.npz', allow_pickle=False) as p:
        y, uids = p['y'], p['uids'].tolist()
        if (len(set(uids)) != len(y) or uid_hash(uids) != ident['test_uid_sha256'] or
                len(y) != doc.get('test_count', ident.get('test_count', len(y)))):
            raise ValueError('Invalid trial identities')
        records = []
        for role in ('big', 'small'):
            acc = float(np.mean(p[role] == y) * 100)
            ba = float(balanced_accuracy_score(y, p[role]) * 100)
            reported = doc[role]['accuracy'] if role in doc else doc[role + '_accuracy']
            if not np.isclose(acc, reported, atol=1e-9, rtol=0):
                raise ValueError('Predictions and accuracy disagree')
            if role in doc and not np.isclose(ba, doc[role]['balanced_accuracy'], atol=1e-9, rtol=0):
                raise ValueError('Predictions and balanced accuracy disagree')
            records.append({'dataset': dataset, 'method': method, 'subject': subject, 'seed': seed,
                'role': role, 'accuracy': acc, 'balanced_accuracy': ba, 'source': str(path),
                'source_manifest_sha256': ident['source_manifest_sha256'],
                'train_uid_sha256': ident['train_uid_sha256'], 'test_uid_sha256': ident['test_uid_sha256']})
        return records


def write_csv(path, rows):
    if not rows: return
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def style(ws):
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = 'C2'
    for cell in ws[1]:
        cell.fill = PatternFill('solid', fgColor='1F4E78')
        cell.font = Font(bold=True, color='FFFFFF')
    for row in ws:
        for cell in row:
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    for column in ws.columns:
        letter = column[0].column_letter
        ws.column_dimensions[letter].width = min(65, max(18, max(len(str(c.value or '')) for c in column) + 3))
    ws.auto_filter.ref = ws.dimensions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    base = yaml.safe_load((ROOT / cfg['baseline_config']).read_text())
    report = require_report_output(REPORTS_ROOT / cfg['report_name'] / 'summary.json').parent
    report.mkdir(parents=True, exist_ok=True)
    impl = fingerprint()
    rows, associations = [], []
    for d, count in cfg['datasets'].items():
        for subject in range(count):
            for seed in cfg['seeds']:
                for method in METHODS:
                    if method in ('none', 'dml_ds'):
                        folder = baseline_dir(d, cfg['pair'], subject, seed, base)
                        if method == 'none': folder = folder.parent.parent.parent / 'none' / folder.parent.name / folder.name
                    else:
                        folder = run_folder(cfg, d, method, subject, seed)
                    found = read_result(folder, d, method, subject, seed, cfg, base, impl, sha256_file(args.config))
                    rows.extend(found)
                    if found and method in ('semckd', 'semckd_uniform'):
                        history = json.loads((folder / 'training.json').read_text())['history']
                        last = history[-1]
                        for role, details in enumerate(last['diagnostics']):
                            source_layers = ('patch_embedding', 'encoder_block4', 'encoder_block8') if role == 0 else ('batchnorm1', 'pooling2', 'pooling3')
                            target_layers = ('batchnorm1', 'pooling2', 'pooling3') if role == 0 else ('patch_embedding', 'encoder_block4', 'encoder_block8')
                            for a, source_layer in enumerate(source_layers):
                                for b, target_layer in enumerate(target_layers):
                                    associations.append({'dataset': d, 'method': method, 'subject': subject,
                                        'seed': seed, 'epoch': last['epoch'], 'receiver': 'CBraMod' if role == 0 else 'EEGNet',
                                        'source_layer': source_layer, 'target_layer': target_layer,
                                        'mean_association': details['association'][a][b],
                                        'mean_pair_mse': details['pair_mse'][a][b]})
    index = {(r['dataset'], r['method'], r['subject'], r['seed'], r['role']): r for r in rows}
    if len(index) != len(rows): raise ValueError('Duplicate subject/seed/role records')
    for r in rows:
        if r['method'] == 'dml_ds': continue
        ref = index.get((r['dataset'], 'dml_ds', r['subject'], r['seed'], r['role']))
        if ref is None or any(r[k] != ref[k] for k in (
                'source_manifest_sha256', 'train_uid_sha256', 'test_uid_sha256')):
            raise ValueError('Source or trial pairing differs from DML-DS')
    progress, scores, contrasts = [], {}, []
    for d, count in cfg['datasets'].items():
        expected = count * len(cfg['seeds'])
        for method in METHODS:
            group = [r for r in rows if r['dataset'] == d and r['method'] == method and r['role'] == 'small']
            complete = len(group) == expected
            progress.append({'dataset': d, 'method': method, 'completed': len(group),
                             'expected': expected, 'status': 'complete' if complete else 'pending'})
            if not complete: continue
            for role in ('big', 'small'):
                for metric in ('accuracy', 'balanced_accuracy'):
                    values = [np.mean([index[d, method, s, z, role][metric] for s in range(count)])
                              for z in cfg['seeds']]
                    scores[d, method, role, metric] = float(np.mean(values))
                    if method == 'dml_ds': continue
                    differences = [np.mean([index[d, method, s, z, role][metric] -
                                           index[d, 'dml_ds', s, z, role][metric] for z in cfg['seeds']])
                                   for s in range(count)]
                    mean = float(np.mean(differences))
                    half = float(t.ppf(.975, count - 1) * np.std(differences, ddof=1) / np.sqrt(count))
                    contrasts.append({'dataset': d, 'method': method, 'model': 'CBraMod' if role == 'big' else 'EEGNet',
                        'metric': metric, 'DMLDS': scores.get((d, 'dml_ds', role, metric), float(np.mean([
                            index[d, 'dml_ds', s, z, role][metric] for s in range(count) for z in cfg['seeds']]))),
                        'score': scores[d, method, role, metric], 'delta_pp': mean,
                        'ci95_low_pp': mean - half, 'ci95_high_pp': mean + half, 'subjects': count})
    selection_contrasts = []
    for d, count in cfg['datasets'].items():
        for role in ('big', 'small'):
            for metric in ('accuracy', 'balanced_accuracy'):
                learned = scores.get((d, 'semckd', role, metric))
                uniform = scores.get((d, 'semckd_uniform', role, metric))
                if learned is None or uniform is None: continue
                differences = [np.mean([index[d, 'semckd', s, z, role][metric] -
                                       index[d, 'semckd_uniform', s, z, role][metric] for z in cfg['seeds']])
                               for s in range(count)]
                mean = float(np.mean(differences))
                half = float(t.ppf(.975, count - 1) * np.std(differences, ddof=1) / np.sqrt(count))
                selection_contrasts.append({'dataset': d, 'model': 'CBraMod' if role == 'big' else 'EEGNet',
                    'metric': metric, 'SemCKD': learned, 'uniform': uniform, 'delta_pp': mean,
                    'ci95_low_pp': mean - half, 'ci95_high_pp': mean + half, 'subjects': count})
    write_csv(report / 'subject_seed_results.csv', rows)
    write_csv(report / 'group_status.csv', progress)
    write_csv(report / 'paired_differences.csv', contrasts)
    write_csv(report / 'layer_associations.csv', associations)
    write_csv(report / 'association_contrasts.csv', selection_contrasts)
    wb = Workbook()
    wb.remove(wb.active)
    for role, label in (('small', '小模型'), ('big', '大模型')):
        for metric, title in (('balanced_accuracy', 'BA'), ('accuracy', 'Accuracy')):
            ws = wb.create_sheet(label + title)
            ws.append(['student', 'method'] + [d.replace('BNCI2014', '') for d in cfg['datasets']])
            table = [('CBraMod', 'none', 'big'), ('EEGNet', 'none', 'small')] + [
                ('EEGNet' if role == 'small' else 'CBraMod', method, role) for method in METHODS[1:]]
            for model, method, r in table:
                ws.append([model, NAMES[method]] + [scores.get((d, method, r, metric)) for d in cfg['datasets']])
                for col, d in enumerate(cfg['datasets'], 3):
                    cell = ws.cell(ws.max_row, col)
                    cell.number_format = '0.00'
                    color = None
                    if method == 'none': color = 'FAC090' if r == 'big' else '93CDDD'
                    elif cell.value is not None:
                        teacher, student = scores.get((d, 'none', 'big', metric)), scores.get((d, 'none', 'small', metric))
                        if teacher is not None and student is not None:
                            a, b = cell.value > teacher, cell.value > student
                            color = 'FFC7CE' if a and b else 'C6EFCE' if a else 'FFF2CC' if b else None
                    if color: cell.fill = PatternFill('solid', fgColor=color)
            ws.append([])
            for color, note in (('FFF2CC', '大于小模型独立 CE'), ('C6EFCE', '大于大模型独立 CE'),
                                ('FFC7CE', '大于二者')):
                ws.append([note]); ws.cell(ws.max_row, 1).fill = PatternFill('solid', fgColor=color)
            ws.append([('Balanced Accuracy (%)' if metric == 'balanced_accuracy' else 'Accuracy (%)') +
                       '；每 seed 被试等权平均，再平均 666、667、668。整组未完成时留空。'])
            ws.merge_cells(start_row=ws.max_row, start_column=1, end_row=ws.max_row, end_column=2 + len(cfg['datasets']))
            style(ws)
            ws.column_dimensions['A'].width = 23
            ws.column_dimensions['B'].width = 29
            ws.cell(ws.max_row, 1).alignment = Alignment(horizontal='left', wrap_text=True)
    ws = wb.create_sheet('相对DMLDS')
    ws.append(['数据集', '模型', '方法', '指标', 'DML-DS (%)', '方法 (%)', '差值 (百分点)', '95%CI 下限', '95%CI 上限'])
    for r in contrasts:
        ws.append([r[k] for k in ('dataset', 'model', 'method', 'metric', 'DMLDS', 'score', 'delta_pp', 'ci95_low_pp', 'ci95_high_pp')])
        for cell in ws[ws.max_row][4:]: cell.number_format = '0.00'
        for cell in ws[ws.max_row][6:]:
            cell.number_format = '+0.00;-0.00;0.00'
            cell.font = Font(color='008000' if cell.value > 0 else 'C00000' if cell.value < 0 else '000000')
    style(ws)
    ws = wb.create_sheet('学层与等权比较')
    ws.append(['数据集', '模型', '指标', 'SemCKD (%)', '等权 (%)', '差值 (百分点)', '95%CI 下限', '95%CI 上限'])
    for r in selection_contrasts:
        ws.append([r[k] for k in ('dataset', 'model', 'metric', 'SemCKD', 'uniform', 'delta_pp', 'ci95_low_pp', 'ci95_high_pp')])
        for cell in ws[ws.max_row][3:]: cell.number_format = '0.00'
        for cell in ws[ws.max_row][5:]:
            cell.number_format = '+0.00;-0.00;0.00'
            cell.font = Font(color='008000' if cell.value > 0 else 'C00000' if cell.value < 0 else '000000')
    style(ws)
    for name, records in (('完成进度', progress), ('逐被试种子', rows), ('层对应', associations)):
        ws = wb.create_sheet(name)
        if records:
            ws.append(list(records[0]))
            for r in records: ws.append(list(r.values()))
            style(ws)
    ws = wb.create_sheet('方法说明')
    ws.append(['项目', '说明'])
    notes = [
        ('任务', '004、001，CBraMod–EEGNet，完整4秒，30%校准/70%测试，种子666–668，每被试独立适配。'),
        ('对照', '复用同源、同划分、同初始化配方、20 epoch、batch 16 的 DML-DS；独立CE为none。'),
        ('SemCKD', '实例Gram→两层MLP→归一化query/key→softmax层对应；全部3×3层对使用1×1/3×3/1×1卷积回归和加权MSE。'),
        ('等权消融', '同样9条回归路径、相同初始回归参数；每个接收层到三个对方层的权重固定1/3。'),
        ('AT', '三个预定深度层对，通道平方能量平均→空间L2归一化→MSE，各层相加。'),
        ('适配', '两方向各自detach对方特征；图像方形池化推广到EEG矩形；SemCKD短batch的Gram列补0，保留全部CE样本。'),
        ('特征坐标', 'CBraMod为[特征维,电极,原生1秒patch]；EEGNet为[卷积通道,电极或混合轴,时间]。按两轴较小长度平均池化。'),
        ('参数', 'SemCKD及等权权重0.1，AT权重1.0；这是本轮EEG固定参数。前5轮不加特征损失，第6–10轮线性升权。'),
        ('解释限制', '层对应表为最终训练轮、校准样本加权平均的学习权重；仅说明训练关联，不等同于因果解释或最优选层证明。'),
        ('统计', '分组完整才显示均值；种子先在被试内平均差值，再按被试计算t分布95%配对区间。'),
        ('推理', '双方各自原分类头独立推理，完整4秒；最终第20轮一次测试，测试集不选层、选参数或选checkpoint。'),
        ('论文', 'https://arxiv.org/abs/2012.03236 ; https://arxiv.org/abs/1612.03928'),
    ]
    for row in notes: ws.append(row)
    style(ws)
    ws.column_dimensions['B'].width = 115
    for row in ws.iter_rows(min_row=2):
        row[1].alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
        ws.row_dimensions[row[0].row].height = 42
    temporary = report / f'.comparison.{os.getpid()}.tmp.xlsx'
    wb.save(temporary)
    os.replace(temporary, report / 'SemCKD_AT_DMLDS_对照.xlsx')
    completed = sum(p['completed'] for p in progress if p['method'] in cfg['methods'])
    atomic_json(report / 'summary.json', {'experiment_id': cfg['experiment_id'], 'new_completed': completed,
        'new_expected': sum(cfg['datasets'].values()) * len(cfg['seeds']) * len(cfg['methods']),
        'groups': progress, 'config_sha256': sha256_file(args.config), 'implementation_sha256': impl})
    print(f'Report exported: {report}, new runs completed={completed}', flush=True)


if __name__ == '__main__':
    main()
