"""Validate per-trial predictions and export the MIRepNet few-shot workbook."""
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import yaml

from experiments.bidir.run_mirepnet_fewshot_8_30 import CONFIG, code_hashes, uid_hash
from experiments.bidir.queue_mirepnet_fewshot_8_30 import folder_for, jobs
from experiments.storage import REPORTS_ROOT, require_report_output

DATASETS = [('BNCI2014004', '004'), ('BNCI2014001', '001'),
            ('BNCI2014001-4', '001-4'), ('BNCI2015001', '15001'), ('AlexMI', 'AlexMI')]
DISPLAY = {'ifnet': 'IFNet', 'eegnet': 'EEGNet', 'adfcnn': 'ADFCNN'}
REPORT = REPORTS_ROOT / 'dml_ds_mirepnet_fewshot_8_30_v1'


def read_scores(cfg):
    root = Path(cfg['output_root']); results = {}; trial_sets = {}; source_by_dataset = {}
    for job in jobs(cfg):
        dataset, subject, seed, kind, student = job
        folder = folder_for(root, job); result_path = folder/'evaluation.json'
        result = json.loads(result_path.read_text()); identity = result['identity']
        if (identity['dataset'],identity['subject'],identity['seed'],identity['kind'],identity['student'],identity['smoke']) != (dataset,subject,seed,kind,student,False):
            raise ValueError(f'Wrong result identity: {result_path}')
        names = ['mirepnet'] if kind == 'teacher' else [student] if kind == 'student' else ['mirepnet',student]
        if identity['code_sha256'] != code_hashes(dataset,names) or identity['protocol'] != 'fewshot_stratified_random_30_70':
            raise ValueError(f'Unexpected code or protocol: {result_path}')
        if identity['epochs'] != 20 or identity['batch_size'] != 8:
            raise ValueError(f'Unexpected budget: {result_path}')
        if 'mirepnet' in names and identity['mirepnet_preprocessing'] != 'broadband_250Hz_8_30Hz_order4_filtfilt_calibration_EA_pad45':
            raise ValueError(f'Incorrect MIRepNet input: {result_path}')
        if dataset in source_by_dataset and source_by_dataset[dataset] != identity['source_manifest_sha256']:
            raise ValueError(f'Different dataset source: {result_path}')
        source_by_dataset[dataset] = identity['source_manifest_sha256']
        with np.load(folder/'predictions.npz',allow_pickle=False) as archive:
            uids=archive['uids'].tolist(); labels=archive['y']
            if uid_hash(uids)!=identity['test_uid_sha256'] or len(uids)!=result['test_count'] or len(set(uids))!=len(uids) or len(labels)!=len(uids):
                raise ValueError(f'Incorrect test data: {result_path}')
            split_key=(dataset,subject,seed)
            split=(identity['train_uid_sha256'],identity['test_uid_sha256'],tuple(labels.tolist()))
            if split_key in trial_sets and trial_sets[split_key] != split:
                raise ValueError(f'Different trial or label alignment: {result_path}')
            trial_sets[split_key]=split
            for model in names:
                prediction=archive[model]
                if len(prediction)!=len(labels): raise ValueError(f'Wrong prediction count: {result_path}')
                value=Fraction(int(np.count_nonzero(prediction==labels))*100,len(labels))
                if abs(float(value)-result['accuracy_percent'][model])>1e-9: raise ValueError(f'Accuracy mismatch: {result_path}')
                results[dataset,subject,seed,kind,student,model]=(value,identity)
    scores={}
    for dataset,count in cfg['datasets'].items():
        combinations=[('teacher',None,'mirepnet')]+[('student',s,s) for s in cfg['students']]+[( 'dml_ds',s,m) for s in cfg['students'] for m in ('mirepnet',s)]
        for kind,student,model in combinations:
            seed_means=[]
            for seed in cfg['seeds']:
                vals=[results[dataset,subject,seed,kind,student,model][0] for subject in range(count)]
                seed_means.append(sum(vals)/count)
            scores[dataset,kind,student,model]=sum(seed_means)/len(seed_means)
        for subject in range(count):
            for seed in cfg['seeds']:
                teacher=results[dataset,subject,seed,'teacher',None,'mirepnet'][1]
                for student in cfg['students']:
                    small=results[dataset,subject,seed,'student',student,student][1]
                    pair=results[dataset,subject,seed,'dml_ds',student,'mirepnet'][1]
                    if (teacher['initial_hashes']['mirepnet'] != pair['initial_hashes']['mirepnet'] or
                        small['initial_hashes'][student] != pair['initial_hashes'][student] or
                        teacher['mirepnet_ea_sha256'] != pair['mirepnet_ea_sha256']):
                        raise ValueError(f'Initialization/preprocessing mismatch: {dataset}/{subject}/{seed}/{student}')
    return scores


def color_for(value, teacher, student):
    if value > teacher and value > student: return 'FFFFC7CE'
    if value > teacher: return 'FFC6EFCE'
    if value > student: return 'FFFFF2CC'
    return None


def create_sheet(wb, scores, role):
    sheet=wb.create_sheet('学生结果' if role=='student' else '教师结果')
    sheet.sheet_view.showGridLines=False; sheet.freeze_panes='C3'
    sheet.column_dimensions['A'].width=27; sheet.column_dimensions['B'].width=16
    for c in range(3,8): sheet.column_dimensions[get_column_letter(c)].width=15
    sheet.column_dimensions['H'].width=3; sheet.column_dimensions['I'].width=40
    border=Border(left=Side(style='thin',color='FFD9E2F3'),right=Side(style='thin',color='FFD9E2F3'),
                  top=Side(style='thin',color='FFD9E2F3'),bottom=Side(style='thin',color='FFD9E2F3'))
    for c,v in enumerate(['student','method']+[label for _,label in DATASETS],1):
        cell=sheet.cell(1,c,v); cell.fill=PatternFill('solid',fgColor='FF1F4E78')
        cell.font=Font(name='Calibri',size=11,bold=True,color='FFFFFFFF'); cell.border=border
        cell.alignment=Alignment(horizontal='center',vertical='center')
    def row(r,name,method,fill=None):
        sheet.cell(r,1,name); sheet.cell(r,2,method); sheet.row_dimensions[r].height=23
        for c in range(1,8):
            cell=sheet.cell(r,c); cell.border=border; cell.font=Font(name='Calibri',size=11,bold=bool(fill))
            cell.alignment=Alignment(horizontal='center',vertical='center')
            if fill: cell.fill=PatternFill('solid',fgColor=fill)
    row(2,'MIRepNet','none','FFFAC090')
    for col,(dataset,_) in enumerate(DATASETS,3):
        cell=sheet.cell(2,col,float(scores[dataset,'teacher',None,'mirepnet'])); cell.number_format='0.00'
    for i,student in enumerate(DISPLAY):
        baseline=3+2*i; method=baseline+1
        row(baseline,DISPLAY[student],'none','FF93CDDD')
        row(method,DISPLAY[student] if role=='student' else f'MIRepNet（配对 {DISPLAY[student]}）','dml_ds')
        for col,(dataset,_) in enumerate(DATASETS,3):
            t=scores[dataset,'teacher',None,'mirepnet']; s=scores[dataset,'student',student,student]
            v=scores[dataset,'dml_ds',student,student if role=='student' else 'mirepnet']
            sheet.cell(baseline,col,float(s)).number_format='0.00'
            cell=sheet.cell(method,col,float(v)); cell.number_format='0.00'
            fill=color_for(v,t,s)
            if fill: cell.fill=PatternFill('solid',fgColor=fill)
    for r,color,label in [(2,'FFFFF2CC','大于小模型基线，未大于教师基线'),
                          (3,'FFC6EFCE','大于教师基线，未大于小模型基线'),
                          (4,'FFFFC7CE','大于教师与小模型两者的基线')]:
        cell=sheet.cell(r,9,label); cell.fill=PatternFill('solid',fgColor=color); cell.border=border
        cell.alignment=Alignment(horizontal='left',vertical='center',wrap_text=True)
        cell.font=Font(name='Calibri',size=10); sheet.row_dimensions[r].height=25
    notes=[
        '指标：Accuracy (%)，范围 0–100。每个种子内对受试者等权平均，再对 666、667、668 三个种子等权平均。',
        '协议：逐受试者按类别分层随机划分 30% 校准训练、70% 测试；使用最终 epoch 主分类头。',
        ('本表 dml_ds 为学生准确率。' if role=='student' else '本表 dml_ds 为对应配对中 MIRepNet 的准确率；蓝色行为学生 none 基线。'),
        '着色：与相同数据集的统一 MIRepNet none 及对应学生 none 比较，严格大于才计入；比较使用显示前精度。',
        'MIRepNet：宽带源 → 8–30 Hz 四阶双向滤波 → 校准集拟合 EA 并用于测试 → 45 通道映射。',
        '独立教师与学生均为 20 epoch、batch 8；DML-DS 使用 T=2、辅助监督权重 0.5、蒸馏权重 0.25。',
        '数据集人数：004、001、001-4 各 9 人，15001 为 12 人，AlexMI 为 8 人。001-4 为四分类，其余为二分类。',
        '教师独立预训练微调；学生随机初始化。每个受试者、种子和模型在单训与配对训练时使用相同初始化。',
        'AlexMI 信号按已有项目协议重采样后从 3 秒补接前 1 秒。完整来源见 subject_accuracy.csv。',
    ]
    for r,note in enumerate(notes,11):
        sheet.merge_cells(start_row=r,start_column=1,end_row=r,end_column=7)
        cell=sheet.cell(r,1,note); cell.font=Font(name='Calibri',size=10,color='FF404040')
        cell.alignment=Alignment(horizontal='left',vertical='center',wrap_text=True); sheet.row_dimensions[r].height=31
    sheet.print_area='A1:I19'; sheet.page_setup.fitToWidth=1
    return sheet


def main():
    cfg=yaml.safe_load(CONFIG.read_text()); scores=read_scores(cfg)
    wb=openpyxl.Workbook(); wb.remove(wb.active)
    create_sheet(wb,scores,'student'); create_sheet(wb,scores,'teacher')
    target=require_report_output(REPORT/'mirepnet_dml_ds_fewshot_3seed.xlsx')
    wb.save(target)
    check=openpyxl.load_workbook(target)
    for role,title in [('student','学生结果'),('teacher','教师结果')]:
        s=check[title]
        for i,student in enumerate(DISPLAY):
            for col,(dataset,_) in enumerate(DATASETS,3):
                teacher=scores[dataset,'teacher',None,'mirepnet']; baseline=scores[dataset,'student',student,student]
                value=scores[dataset,'dml_ds',student,student if role=='student' else 'mirepnet']
                cell=s.cell(4+2*i,col); fill=cell.fill.fgColor.rgb if cell.fill.patternType=='solid' else None
                if abs(cell.value-float(value))>1e-10 or fill!=color_for(value,teacher,baseline):
                    raise ValueError(f'Workbook verification failed: {title}/{cell.coordinate}')
    print(f'Validated 987 experiment units, 141 trial splits and all 30 method fills; wrote {target}')


if __name__=='__main__': main()
