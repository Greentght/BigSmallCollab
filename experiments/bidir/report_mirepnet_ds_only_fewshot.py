"""Validate and report DS-only against the completed MIRepNet DML-DS run."""
import csv
from fractions import Fraction
import json
import os
from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import yaml

from experiments.bidir.queue_mirepnet_ds_only_fewshot import (
    BASE_CONFIG, CONFIG, REPORT, assert_matched, base_folder, folder_for, jobs, validate_config)
from experiments.bidir.run_mirepnet_ds_only_fewshot import code_hashes, uid_hash
from experiments.bidir.report_mirepnet_fewshot_8_30 import read_scores as read_base_scores
from experiments.storage import require_report_output

DATASETS=[('BNCI2014004','004'),('BNCI2014001','001'),('BNCI2014001-4','001-4'),
          ('BNCI2015001','15001'),('AlexMI','AlexMI')]
DISPLAY={'ifnet':'IFNet','eegnet':'EEGNet','adfcnn':'ADFCNN'}


def read_ds(cfg, base):
    root=Path(cfg['output_root']); base_root=Path(base['output_root'])
    values={}; splits={}; subject_rows=[]
    for job in jobs(cfg):
        dataset,subject,seed,student=job; folder=folder_for(root,job)
        result=json.loads((folder/'evaluation.json').read_text()); ident=result['identity']
        if (ident['experiment'],ident['dataset'],ident['subject'],ident['seed'],ident['kind'],ident['student'],ident['smoke']) != (cfg['experiment_id'],dataset,subject,seed,'ds_only',student,False):
            raise ValueError(f'DS-only result identity mismatch: {folder}')
        if ident['code_sha256']!=code_hashes(dataset,['mirepnet',student]): raise ValueError(f'Code mismatch: {folder}')
        if ident['protocol']!='fewshot_stratified_random_30_70': raise ValueError(f'Protocol mismatch: {folder}')
        previous={kind:json.loads((base_folder(base_root,job,kind)/'evaluation.json').read_text())['identity']
                  for kind in ('teacher','student','dml_ds')}
        assert_matched(ident,previous['teacher'],previous['student'],previous['dml_ds'],student)
        with np.load(folder/'predictions.npz',allow_pickle=False) as ds, \
             np.load(base_folder(base_root,job,'dml_ds')/'predictions.npz',allow_pickle=False) as dml:
            uids=ds['uids'].tolist(); y=ds['y']
            if (len(uids)!=result['test_count'] or len(set(uids))!=len(uids) or uid_hash(uids)!=ident['test_uid_sha256'] or
                not np.array_equal(ds['uids'],dml['uids']) or not np.array_equal(y,dml['y'])):
                raise ValueError(f'Prediction trial/label mismatch: {folder}')
            for model in ('mirepnet',student):
                if len(ds[model])!=len(y): raise ValueError(f'Prediction count mismatch: {folder}')
                score=Fraction(int(np.count_nonzero(ds[model]==y))*100,len(y))
                if abs(float(score)-result['accuracy_percent'][model])>1e-9: raise ValueError(f'Accuracy mismatch: {folder}')
                dml_score=Fraction(int(np.count_nonzero(dml[model]==y))*100,len(y))
                values[dataset,subject,seed,student,model]=score
                subject_rows.append([dataset,subject,seed,student,model,float(score),float(dml_score),float(dml_score-score),str(folder/'evaluation.json')])
            splits[dataset,subject,seed]=ident['test_uid_sha256']
    scores={}
    for dataset,count in cfg['datasets'].items():
        for student in cfg['students']:
            for model in ('mirepnet',student):
                scores[dataset,student,model]=sum(
                    sum(values[dataset,subject,seed,student,model] for subject in range(count))/count
                    for seed in cfg['seeds'])/len(cfg['seeds'])
    return scores,subject_rows,len(splits)


def comparison_color(v,t,s):
    if v>t and v>s:return 'FFFFC7CE'
    if v>t:return 'FFC6EFCE'
    if v>s:return 'FFFFF2CC'
    return None


def style_sheet(sheet):
    sheet.sheet_view.showGridLines=False; sheet.freeze_panes='C3'
    sheet.column_dimensions['A'].width=30; sheet.column_dimensions['B'].width=16
    for c in range(3,8): sheet.column_dimensions[get_column_letter(c)].width=15
    sheet.column_dimensions['H'].width=3; sheet.column_dimensions['I'].width=42
    border=Border(left=Side(style='thin',color='FFD9E2F3'),right=Side(style='thin',color='FFD9E2F3'),
                  top=Side(style='thin',color='FFD9E2F3'),bottom=Side(style='thin',color='FFD9E2F3'))
    for col,name in enumerate(['student','method']+[label for _,label in DATASETS],1):
        cell=sheet.cell(1,col,name);cell.fill=PatternFill('solid',fgColor='FF1F4E78');cell.border=border
        cell.font=Font(name='Calibri',size=11,bold=True,color='FFFFFFFF')
        cell.alignment=Alignment(horizontal='center',vertical='center')
    sheet.row_dimensions[1].height=25
    return border


def add_row(sheet,row,name,method,border,fill=None):
    sheet.cell(row,1,name);sheet.cell(row,2,method);sheet.row_dimensions[row].height=23
    for col in range(1,8):
        c=sheet.cell(row,col);c.border=border;c.font=Font(name='Calibri',size=11,bold=bool(fill))
        c.alignment=Alignment(horizontal='center',vertical='center')
        if fill:c.fill=PatternFill('solid',fgColor=fill)


def notes(sheet,start,lines):
    for row,line in enumerate(lines,start):
        sheet.merge_cells(start_row=row,start_column=1,end_row=row,end_column=7)
        cell=sheet.cell(row,1,line);cell.font=Font(name='Calibri',size=10,color='FF404040')
        cell.alignment=Alignment(horizontal='left',vertical='center',wrap_text=True)
        sheet.row_dimensions[row].height=31


def make_result_sheet(workbook,base_scores,ds_scores,role):
    sheet=workbook.create_sheet('学生结果' if role=='student' else '教师结果');border=style_sheet(sheet)
    add_row(sheet,2,'MIRepNet','none',border,'FFFAC090')
    for col,(dataset,_) in enumerate(DATASETS,3):
        sheet.cell(2,col,float(base_scores[dataset,'teacher',None,'mirepnet'])).number_format='0.00'
    for i,student in enumerate(DISPLAY):
        first=3+3*i;add_row(sheet,first,DISPLAY[student],'none',border,'FF93CDDD')
        model=student if role=='student' else 'mirepnet'
        label=DISPLAY[student] if role=='student' else f'MIRepNet（配对 {DISPLAY[student]}）'
        for offset,method in ((1,'ds_only'),(2,'dml_ds')):add_row(sheet,first+offset,label,method,border)
        for col,(dataset,_) in enumerate(DATASETS,3):
            teacher=base_scores[dataset,'teacher',None,'mirepnet']; small=base_scores[dataset,'student',student,student]
            sheet.cell(first,col,float(small)).number_format='0.00'
            for offset,value in ((1,ds_scores[dataset,student,model]),(2,base_scores[dataset,'dml_ds',student,model])):
                cell=sheet.cell(first+offset,col,float(value));cell.number_format='0.00'
                fill=comparison_color(value,teacher,small)
                if fill:cell.fill=PatternFill('solid',fgColor=fill)
    for row,color,text in [(2,'FFFFF2CC','大于小模型基线，未大于教师基线'),
                           (3,'FFC6EFCE','大于教师基线，未大于小模型基线'),
                           (4,'FFFFC7CE','大于教师与小模型两者的基线')]:
        cell=sheet.cell(row,9,text);cell.fill=PatternFill('solid',fgColor=color);cell.border=border
        cell.font=Font(name='Calibri',size=10);cell.alignment=Alignment(horizontal='left',vertical='center',wrap_text=True)
        sheet.row_dimensions[row].height=25
    notes(sheet,14,[
        '指标：Accuracy (%)，每种子对受试者等权平均，再对 666、667、668 三个种子等权平均。',
        '协议：各受试者 30% 校准训练、70% 测试；按类别分层，固定最终 epoch。',
        ('ds_only、dml_ds 行是学生准确率。' if role=='student' else 'ds_only、dml_ds 行是相应配对中教师的准确率；蓝色 none 为配对学生基线。'),
        '严格比较同一数据集的统一 MIRepNet none 和对应学生 none；颜色判断使用未四舍五入的结果。',
        'ds_only：双方主头 CE + 0.5 × 两个辅助头 CE 的均值；无 KL。dml_ds：在相同深监督上加入双向主头 KL，T=2、权重 0.25。',
        'MIRepNet 输入：宽带源 → 8–30 Hz 四阶双向滤波 → 校准集 EA → 45 通道映射；训练预算均为 20 epoch、batch 8。',
        '004、001、001-4 各 9 人，15001 为 12 人，AlexMI 为 8 人；001-4 四分类，其余二分类。',
        '判断 DML 在深监督基础上的增量请查看“DML增量”表；按同一受试者和种子进行配对。',
    ])


def make_delta_sheet(workbook,base_scores,ds_scores):
    sheet=workbook.create_sheet('DML增量');border=style_sheet(sheet)
    for i,student in enumerate(DISPLAY):
        for offset,(role,model) in enumerate((('学生',student),('教师','mirepnet'))):
            row=2+i*2+offset;add_row(sheet,row,DISPLAY[student],f'{role}增量',border)
            for col,(dataset,_) in enumerate(DATASETS,3):
                delta=base_scores[dataset,'dml_ds',student,model]-ds_scores[dataset,student,model]
                sheet.cell(row,col,float(delta)).number_format='0.00;[Red]-0.00;0.00'
    notes(sheet,10,['数值为同一配对的 dml_ds − ds_only，单位为百分点；正数表示加 DML 后准确率上升。',
                    '每个条件先按受试者和三个种子等权平均，再做差；数值依据未四舍五入的结果。',
                    '此表为准确率变化而非准确率，未使用红、绿、黄基线比较填色。'])


def create_report(cfg,base):
    validate_config(cfg,base);base_scores=read_base_scores(base);ds_scores,rows,split_count=read_ds(cfg,base)
    REPORT.mkdir(parents=True,exist_ok=True)
    target=require_report_output(REPORT/'subject_accuracy.csv');temp=target.with_suffix('.tmp')
    with temp.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','subject_zero_based','seed','student','model','ds_only_accuracy_percent','dml_ds_accuracy_percent','dml_delta_pp','source']);w.writerows(rows)
    os.replace(temp,target)
    target=require_report_output(REPORT/'summary.csv');temp=target.with_suffix('.tmp')
    with temp.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','student','model','none_percent','ds_only_percent','dml_ds_percent','dml_delta_pp'])
        for dataset,_ in DATASETS:
            for student in DISPLAY:
                for model in ('mirepnet',student):
                    baseline=base_scores[dataset,'teacher',None,'mirepnet'] if model=='mirepnet' else base_scores[dataset,'student',student,student]
                    ds=ds_scores[dataset,student,model];dml=base_scores[dataset,'dml_ds',student,model]
                    w.writerow([dataset,student,model,float(baseline),float(ds),float(dml),float(dml-ds)])
    os.replace(temp,target)
    book=openpyxl.Workbook();book.remove(book.active)
    make_result_sheet(book,base_scores,ds_scores,'student');make_result_sheet(book,base_scores,ds_scores,'teacher')
    make_delta_sheet(book,base_scores,ds_scores)
    target=require_report_output(REPORT/'mirepnet_ds_only_vs_dml_ds_3seed.xlsx');book.save(target)
    check=openpyxl.load_workbook(target)
    for student_index,student in enumerate(DISPLAY):
        for col,(dataset,_) in enumerate(DATASETS,3):
            actual=check['DML增量'].cell(2+2*student_index,col).value
            expected=base_scores[dataset,'dml_ds',student,student]-ds_scores[dataset,student,student]
            if abs(actual-float(expected))>1e-10:raise ValueError('Workbook delta mismatch')
    print(f'Validated 423 DS-only units and {split_count} aligned splits; wrote {target}',flush=True)


def main():create_report(yaml.safe_load(CONFIG.read_text()),yaml.safe_load(BASE_CONFIG.read_text()))
if __name__=='__main__':main()
