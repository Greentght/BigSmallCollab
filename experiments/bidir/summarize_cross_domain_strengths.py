"""One complete, colored comparison for the pilot and both published strengths."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import csv
import json
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from scipy.stats import t
import yaml

from data.loso_source import sha256_file
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import atomic_json
from experiments.bidir.summarize_cross_domain_features import style, write_csv
from experiments.storage import PROJECT_ROOT, REPORTS_ROOT, require_report_output


def main():
    names = ('fewshot_cross_domain_features_v1.yaml', 'fewshot_semckd_weight50_v1.yaml',
             'fewshot_semckd_weight400_v1.yaml')
    records, baseline, index, configs = [], {}, {}, []
    for name in names:
        path = PROJECT_ROOT / 'configs/experiments' / name
        cfg = yaml.safe_load(path.read_text())
        report = REPORTS_ROOT / cfg['report_name']
        summary = json.loads((report / 'summary.json').read_text())
        controller = json.loads((report / 'controller.json').read_text())
        if (summary['config_sha256'] != sha256_file(path) or controller['stage'] != 'completed' or
                summary['new_completed'] != summary['new_expected'] or controller['failures']):
            raise ValueError(f'Complete validated results required: {report}')
        configs.append({'config': str(path), 'sha256': sha256_file(path), 'report': str(report)})
        with (report / 'subject_seed_results.csv').open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                method = row['method']
                key = (row['dataset'], row['subject'], row['seed'], row['role'])
                if method in ('none', 'dml_ds'):
                    if (method, *key) in baseline:
                        if baseline[(method, *key)] != row:
                            raise ValueError('Baselines differ across strengths')
                        continue
                    baseline[(method, *key)] = dict(row)
                    weight = None
                else:
                    weight = float(cfg['methods'][method]['weight'])
                record = {**row, 'weight': weight}
                full_key = (method, weight, *key)
                if full_key in index: raise ValueError('Duplicate experiment record')
                index[full_key] = record
                records.append(record)
    conditions = [('none', None), ('dml_ds', None), ('semckd', .1), ('semckd_uniform', .1),
                  ('semckd', 50.), ('semckd_uniform', 50.), ('semckd', 400.), ('semckd_uniform', 400.), ('at', 1.)]
    datasets, seeds = ('BNCI2014004', 'BNCI2014001'), ('666', '667', '668')
    scores, contrasts = {}, []
    for dataset in datasets:
        for method, weight in conditions:
            for role in ('big', 'small'):
                for metric in ('accuracy', 'balanced_accuracy'):
                    own = [[float(index[method, weight, dataset, str(s), z, role][metric]) for z in seeds]
                           for s in range(9)]
                    scores[dataset, method, weight, role, metric] = float(np.mean(own))
                    if method == 'dml_ds': continue
                    ref = [[float(index['dml_ds', None, dataset, str(s), z, role][metric]) for z in seeds]
                           for s in range(9)]
                    diff = np.mean(np.asarray(own) - np.asarray(ref), axis=1)
                    mean = float(np.mean(diff))
                    half = float(t.ppf(.975, 8) * np.std(diff, ddof=1) / 3)
                    contrasts.append({'dataset': dataset, 'method': method, 'weight': weight,
                        'model': 'CBraMod' if role == 'big' else 'EEGNet', 'metric': metric,
                        'score': float(np.mean(own)), 'DMLDS': float(np.mean(ref)), 'delta_pp': mean,
                        'ci95_low_pp': mean - half, 'ci95_high_pp': mean + half})
    report = require_report_output(REPORTS_ROOT / 'semckd_strengths_fewshot_v1/comparison.json').parent
    report.mkdir(parents=True, exist_ok=True)
    write_csv(report / 'subject_seed_results.csv', records)
    write_csv(report / 'paired_differences.csv', contrasts)
    wb = Workbook()
    wb.remove(wb.active)
    labels = {'none': 'none', 'dml_ds': 'DML-DS', 'semckd': 'DML-DS + SemCKD',
              'semckd_uniform': 'DML-DS + SemCKD 等权', 'at': 'DML-DS + AT'}
    for role, title in (('small', '小模型'), ('big', '大模型')):
        for metric, suffix in (('balanced_accuracy', 'BA'), ('accuracy', 'Accuracy')):
            ws = wb.create_sheet(title + suffix)
            ws.append(['student', 'method', '004 (%)', '004 ΔDMLDS (pp)', '001 (%)', '001 ΔDMLDS (pp)'])
            table = [('CBraMod', 'none', None, 'big'), ('EEGNet', 'none', None, 'small')]
            table += [('EEGNet' if role == 'small' else 'CBraMod', m, w, role) for m, w in conditions[1:]]
            for model, method, weight, r in table:
                cells = [model, labels[method] + (f' (λ={weight:g})' if weight is not None else '')]
                for dataset in datasets:
                    score = scores[dataset, method, weight, r, metric]
                    diff = score - scores[dataset, 'dml_ds', None, r, metric]
                    cells += [score, None if method == 'none' and r != role else diff]
                ws.append(cells)
                for col, dataset in ((3, datasets[0]), (5, datasets[1])):
                    cell = ws.cell(ws.max_row, col)
                    cell.number_format = '0.00'
                    teacher = scores[dataset, 'none', None, 'big', metric]
                    student = scores[dataset, 'none', None, 'small', metric]
                    a, b = cell.value > teacher, cell.value > student
                    color = ('FAC090' if r == 'big' else '93CDDD') if method == 'none' else (
                        'FFC7CE' if a and b else 'C6EFCE' if a else 'FFF2CC' if b else None)
                    if color: cell.fill = PatternFill('solid', fgColor=color)
                    delta = ws.cell(ws.max_row, col + 1)
                    delta.number_format = '+0.00;-0.00;0.00'
                    if delta.value is not None:
                        delta.font = Font(color='008000' if delta.value > 0 else 'C00000' if delta.value < 0 else '000000')
            ws.append([])
            for color, note in (('FFF2CC', '仅大于小模型CE'), ('C6EFCE', '仅大于大模型CE'), ('FFC7CE', '大于二者')):
                ws.append([note]);ws.cell(ws.max_row, 1).fill = PatternFill('solid', fgColor=color)
            ws.append(['指标：' + ('Balanced Accuracy (%)' if metric == 'balanced_accuracy' else 'Accuracy (%)') +
                       '。Δ为相对DML-DS的百分点差，绿字为正，红字为负；9被试、种子666–668等权平均。'])
            ws.merge_cells(start_row=ws.max_row, start_column=1, end_row=ws.max_row, end_column=6)
            style(ws)
            ws.column_dimensions['A'].width = 18
            ws.column_dimensions['B'].width = 40
            for c in ('C', 'D', 'E', 'F'): ws.column_dimensions[c].width = 22
            ws.cell(ws.max_row, 1).alignment = Alignment(horizontal='left', wrap_text=True)
    for title, rows in (('被试配对区间', contrasts), ('逐被试种子', records)):
        ws = wb.create_sheet(title)
        ws.append(list(rows[0]))
        for row in rows: ws.append(list(row.values()))
        style(ws)
    ws = wb.create_sheet('范围与说明')
    ws.append(['项目', '说明'])
    for row in (
        ('设置', 'CBraMod–EEGNet；004、001；30%校准/70%测试；完整4秒；20 epoch、batch16；种子666–668。'),
        ('运行规模', '7个新增条件×2数据集×9被试×3种子=378次双模型训练；复用54次DML-DS及已有独立CE对照。'),
        ('权重', 'SemCKD和等权各0.1、50、400；AT为1。官方公开50用于ImageNet、400用于CIFAR。本项目保留EEG双向DML-DS配方。'),
        ('公平性', '只有辅助损失权重改变；同源、同UID划分、相同模型初始化和回归器初始化；所有条件均报告，不按测试选最优。'),
        ('启用时间', '前5轮只用原DML-DS；第6–10轮线性升到辅助损失预设权重；第20轮固定最终模型各自独立测试。'),
        ('机制', 'SemCKD保留实例Gram、自注意力层对应、九个非线性回归器及加权MSE；等权固定1/3。AT保留平方能量图归一化与MSE。'),
        ('适配', '原论文固定教师改为在线双向、对方特征detach；EEG矩形特征按轴平均池化；短batch仅Gram列补0，保留所有训练样本。'),
        ('区间', '每被试先平均三个种子的配对差，再按9名被试算t分布95%区间。权重补充属于探索性分析，不把27次种子结果当独立被试。'),
        ('梯度', '初轮0.1在EEGNet conv1上的辅助/原损失梯度范数比中位数约0.00086，因此补充两个官方系数；全程不按测试选择权重。'),
        ('论文', 'https://arxiv.org/abs/2012.03236 ; https://arxiv.org/abs/1612.03928'),
    ): ws.append(row)
    style(ws);ws.column_dimensions['B'].width = 110
    for row in ws.iter_rows(min_row=2):
        row[1].alignment = Alignment(horizontal='left', wrap_text=True)
        ws.row_dimensions[row[0].row].height = 42
    target = require_report_output(report / 'SemCKD_AT_全部权重对照.xlsx')
    tmp = target.with_name(f'.comparison.{os.getpid()}.xlsx')
    wb.save(tmp);os.replace(tmp, target)
    atomic_json(report / 'comparison.json', {'completed_new_units': 378, 'matched_DMLDS_units': 54,
        'sources': configs, 'record_count': len(records), 'workbook': str(target)})
    print('Complete strength comparison exported:', target, flush=True)


if __name__ == '__main__':
    main()
