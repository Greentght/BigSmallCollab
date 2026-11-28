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
from experiments.bidir.run_terminal_hfd_fewshot import CONFIG, ROOT, fingerprint, run_folder
from experiments.bidir.run_dml_ds_cross_hfd_all import baseline_dir
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import atomic_json, uid_hash
from experiments.storage import REPORTS_ROOT, require_report_output

NAMES = {'none': 'none', 'dml_ds': 'DML-DS',
         'terminal_hfd_w01': 'DML-DS + 末层 HFD (0.1)',
         'terminal_hfd_w1': 'DML-DS + 末层 HFD (1.0)'}
METHODS = ('none', 'dml_ds', 'terminal_hfd_w01', 'terminal_hfd_w1')


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
    methods = tuple(m for m in METHODS if m in ('none', 'dml_ds') or m in cfg['methods'])
    report = require_report_output(REPORTS_ROOT / cfg['report_name'] / 'summary.json').parent
    report.mkdir(parents=True, exist_ok=True)
    impl = fingerprint()
    rows = []
    for d, count in cfg['datasets'].items():
        for subject in range(count):
            for seed in cfg['seeds']:
                for method in methods:
                    if method in ('none', 'dml_ds'):
                        folder = baseline_dir(d, cfg['pair'], subject, seed, base)
                        if method == 'none': folder = folder.parent.parent.parent / 'none' / folder.parent.name / folder.name
                    else:
                        folder = run_folder(cfg, d, method, subject, seed)
                    found = read_result(folder, d, method, subject, seed, cfg, base, impl, sha256_file(args.config))
                    rows.extend(found)
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
        for method in methods:
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
    write_csv(report / 'subject_seed_results.csv', rows)
    write_csv(report / 'group_status.csv', progress)
    write_csv(report / 'paired_differences.csv', contrasts)
    wb = Workbook()
    wb.remove(wb.active)
    for role, label in (('small', '小模型'), ('big', '大模型')):
        for metric, title in (('balanced_accuracy', 'BA'), ('accuracy', 'Accuracy')):
            ws = wb.create_sheet(label + title)
            ws.append(['student', 'method'] + [d.replace('BNCI2014', '') for d in cfg['datasets']])
            table = [('CBraMod', 'none', 'big'), ('EEGNet', 'none', 'small')] + [
                ('EEGNet' if role == 'small' else 'CBraMod', method, role) for method in methods[1:]]
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
    for name, records in (('完成进度', progress), ('逐被试种子', rows)):
        ws = wb.create_sheet(name)
        if records:
            ws.append(list(records[0]))
            for r in records: ws.append(list(r.values()))
            style(ws)
    ws = wb.create_sheet('方法说明')
    ws.append(['项目', '说明'])
    notes = [
        ('任务', 'CBraMod–EEGNet，004、001，完整4秒，30%校准/70%测试，9被试、种子666–668。'),
        ('保留的主路径', '双方自己的原生最终特征→自己的原分类头→原生预测双向KL；同时保留主CE与两个DS头CE。'),
        ('末层HFD', 'CBraMod block11输出→投影→EEGNet末层CNN算子→匹配原生pooling3；反向由pooling2→投影→冻结CBraMod完整block12→匹配其原生输出。'),
        ('距离', '与现有正确HFD一致，按各目标特征通道计算1−cosine，平均batch、时间与电极位置；双方目标detach，算子参数冻结且BN缓冲区克隆。'),
        ('适配范围', '原HFD浅层算子扩展为完整最终特征算子，以目标直接落到最终分类表示。不是原论文的浅层位置复现。'),
        ('维度适配', '投影以32个学习得到的时间节点连接4个原生patch和CNN时间轴；这些节点属于适配器，没有额外分类标签或新试次。'),
        ('新损失', '每方：主CE + 0.5×两DS头CE均值 + 0.25×T²KL(对方原预测||自己原预测) + r×w×末层HFD；T=2，w=0.1或1。'),
        ('时间顺序', '对齐与KL在同一batch联合反传，逐步学出末层表示；没有把特征替换成对方特征，也没有分成两个独立训练阶段。'),
        ('预算', '20 epoch，batch16；前5轮HFD为0，第6–10轮线性升权。两个条件均报告，测试不选权重、不选epoch。'),
        ('随机数与状态', '额外交叉算子用独立CUDA随机数流并随断点保存，恢复后逐项匹配；不消耗原生分类前向的随机数流，不更新对方BN缓冲区。'),
        ('统计', '完整27次才显示均值；先在每个被试内平均三个种子的配对差，再用9名被试算t分布95%区间。'),
    ]
    for row in notes: ws.append(row)
    style(ws)
    ws.column_dimensions['B'].width = 115
    for row in ws.iter_rows(min_row=2):
        row[1].alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
        ws.row_dimensions[row[0].row].height = 42
    temporary = report / f'.comparison.{os.getpid()}.tmp.xlsx'
    wb.save(temporary)
    os.replace(temporary, require_report_output(report / cfg.get('workbook_name', 'SemCKD_AT_DMLDS_对照.xlsx')))
    completed = sum(p['completed'] for p in progress if p['method'] in cfg['methods'])
    atomic_json(report / 'summary.json', {'experiment_id': cfg['experiment_id'], 'new_completed': completed,
        'new_expected': sum(cfg['datasets'].values()) * len(cfg['seeds']) * len(cfg['methods']),
        'groups': progress, 'config_sha256': sha256_file(args.config), 'implementation_sha256': impl})
    print(f'Report exported: {report}, new runs completed={completed}', flush=True)


if __name__ == '__main__':
    main()
