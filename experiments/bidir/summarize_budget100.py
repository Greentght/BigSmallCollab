"""Export only verified 100-epoch results; incomplete subject/seed groups stay blank."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from sklearn.metrics import balanced_accuracy_score

from experiments.bidir.budget100 import REPORT, epoch_count, sha, verify_completion, write_json


def records_for(path, family, job=None):
    doc = json.loads(path.read_text()); i = doc['identity']
    if epoch_count(doc) != 100 or i.get('smoke'): raise ValueError('Only formal 100-epoch results are exportable')
    dataset = i['dataset']; pair = i.get('pair') or ('mirepnet_' + i['student'] if i.get('student') else 'mirepnet')
    method = i.get('condition', i.get('method', i.get('kind')))
    roles = pair.split('_')
    rows = []
    with np.load(path.parent / 'predictions.npz', allow_pickle=False) as p:
        y = p['y']; uids = p['uids'].tolist()
        if len(set(uids)) != len(y): raise ValueError('Invalid trial identities')
        for key in p.files:
            if key in ('uids', 'y'): continue
            model = roles[0 if key == 'big' else 1] if key in ('big', 'small') else key
            rows.append({'family': family, 'dataset': dataset, 'pair': pair, 'model': model,
                'method': method, 'subject': i['subject'], 'seed': i['seed'], 'epochs': 100,
                'accuracy': float(np.mean(p[key] == y) * 100),
                'balanced_accuracy': float(balanced_accuracy_score(y, p[key]) * 100),
                'profile': job['profile'] if job else i.get('normalization', 'retained_original_100epoch_protocol'),
                'origin': 'rerun_20_to_100' if job else 'verified_existing_100',
                'source': str(path), 'source_sha256': sha(path)})
    return rows


def csv_file(path, rows, fields=None):
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def main():
    manifest = json.loads((REPORT / 'manifest.json').read_text())
    rows, progress = [], []
    done = Counter()
    for job in manifest['jobs']:
        if verify_completion(job):
            done[job['family']] += 1
            rows.extend(records_for(Path(job['output']) / 'evaluation.json', job['family'], job))
    seen = set()
    for reused in manifest['reused_100epoch_records']:
        path = Path(reused['reuse100'])
        if str(path) in seen: continue
        seen.add(str(path))
        rows.extend(records_for(path, reused['family']))
    for family, expected in manifest['family_counts'].items():
        progress.append({'family': family, 'epochs': 100, 'completed': done[family], 'expected': expected,
                         'status': 'complete' if done[family] == expected else 'running_or_pending'})
    csv_file(REPORT / '100epoch_subject_seed_results.csv', rows,
             ['family','dataset','pair','model','method','subject','seed','epochs','accuracy',
              'balanced_accuracy','profile','origin','source','source_sha256'])
    csv_file(REPORT / '100epoch_group_status.csv', progress)
    wb = Workbook(); status = wb.active; status.title = '100轮重跑进度'
    status.append(list(progress[0]))
    for r in progress: status.append(list(r.values()))
    grouped = defaultdict(list)
    for r in rows: grouped[r['family'], r['pair'], r['profile']].append(r)
    datasets = ('BNCI2014004','BNCI2014001','BNCI2014001-4','BNCI2015001','AlexMI')
    expected_subjects = dict(zip(datasets, (9,9,9,12,8)))
    for group, data in sorted(grouped.items()):
        family,pair,profile = group
        for metric, suffix in (('accuracy','Acc'),('balanced_accuracy','BA')):
            tag = __import__('hashlib').sha256('|'.join(group).encode()).hexdigest()[:5]
            ws = wb.create_sheet((pair[:18]+'_'+tag+'_'+suffix)[:31])
            ws.append(['student','method', '004','001','001-4','15001','AlexMI'])
            teacher=pair.split('_')[0]
            names = sorted({(r['model'],r['method']) for r in data}, key=lambda x:(x[0]!=teacher,x[0],x[1] not in ('none','ce','teacher','student'),x[1]))
            table={}
            for model,method in names:
                values = []
                for dataset in datasets:
                    selected = [r for r in data if r['dataset']==dataset and r['model']==model and r['method']==method]
                    ids = {(int(r['subject']),int(r['seed'])) for r in selected}
                    required = {(s,z) for s in range(expected_subjects[dataset]) for z in (666,667,668)}
                    if ids != required or len(selected) != len(required):
                        values.append(None)
                    else:
                        values.append(float(np.mean([np.mean([r[metric] for r in selected if int(r['seed'])==z]) for z in (666,667,668)])))
                ws.append([model,method]+values)
                table[model,method]=values
            for index,(model,method) in enumerate(names,2):
                for col in range(3,8):
                    cell=ws.cell(index,col)
                    if method in ('none','ce','teacher','student'):
                        cell.fill=PatternFill('solid',fgColor='FAC090' if model==teacher else '93CDDD')
                        continue
                    big=next((table[teacher,m][col-3] for m in ('none','ce','teacher') if (teacher,m) in table),None)
                    small=next((table[model,m][col-3] for m in ('none','ce','student') if (model,m) in table),None)
                    if cell.value is None or big is None or small is None:continue
                    a,b=cell.value>big,cell.value>small
                    color='FFC7CE' if a and b else 'C6EFCE' if a else 'FFF2CC' if b else None
                    if color:cell.fill=PatternFill('solid',fgColor=color)
            ws.append([])
            for color,note in (('FFF2CC','大于匹配小模型CE，未大于教师CE'),('C6EFCE','大于教师CE，未大于匹配小模型CE'),('FFC7CE','大于二者')):
                ws.append([note]);ws.cell(ws.max_row,1).fill=PatternFill('solid',fgColor=color)
            for note in (f'Experiment family: {family}', f'Input/normalization profile: {profile}',
                         ('Accuracy (%)' if metric=='accuracy' else 'Balanced Accuracy (%)')+'; all records have exactly 100 epochs.',
                         'Only complete subject sets and seeds 666,667,668 are averaged. Blank cells are pending.',
                         'Different experiment families and preprocessing profiles are kept in separate sheets.'):
                ws.append([note]);ws.merge_cells(start_row=ws.max_row,start_column=1,end_row=ws.max_row,end_column=7)
                ws.cell(ws.max_row,1).alignment=Alignment(horizontal='left',wrap_text=True)
    raw=wb.create_sheet('100轮逐被试记录')
    if rows:
        raw.append(list(rows[0]))
        for r in rows: raw.append(list(r.values()))
    notes=wb.create_sheet('口径与来源')
    notes.append(['项目','说明'])
    for row in [('训练预算','本文件只接受实际完成100轮的结果。20轮历史数据完全不进入成绩单元格。'),
                ('重跑','从相同随机初始化重新训练，余弦调度的周期为100轮。历史20轮训练状态不用于续接。'),
                ('模型输入','为只改变训练预算，历史输入分支分别重跑并分表；MIRepNet 8–30Hz修正版与旧宽带输入对照不合并。'),
                ('对照','原有100轮结果保留来源。不同配对、输入、归一化和方法家族分表，未匹配的教师/独立CE不做颜色比较。'),
                ('存储','训练状态在外部存储，CSV/XLSX在真实results目录，历史结果保留。')]:notes.append(row)
    for ws in wb:
        ws.sheet_view.showGridLines=False;ws.freeze_panes='C2'
        for c in ws[1]:c.fill=PatternFill('solid',fgColor='1F4E78');c.font=Font(bold=True,color='FFFFFF')
        for row in ws.iter_rows(min_row=2):
            for c in row:
                if c.alignment.horizontal!='left':c.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
                if isinstance(c.value,float):c.number_format='0.00'
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width=25 if col[0].column<3 else 20
        if ws==notes:ws.column_dimensions['B'].width=105
    target=REPORT/'100epoch_统一预算重跑进度与结果.xlsx'
    tmp=REPORT/f'.summary.{os.getpid()}.xlsx';wb.save(tmp);tmp.replace(target)
    write_json(REPORT/'summary.json',{'epochs':100,'completed_new_units':sum(done.values()),
        'expected_new_units':len(manifest['jobs']),'unique_reused_100epoch_records':len(seen),
        'all_records_100epoch':True,'families':progress,'workbook':str(target)})
    print('100-epoch report exported; new completions:',sum(done.values()),flush=True)


if __name__ == '__main__':main()
