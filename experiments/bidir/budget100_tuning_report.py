"""Audited final reports and subject-level paired tuning comparisons."""
import csv
from fractions import Fraction
import json

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy.stats import wilcoxon

from experiments.bidir.budget100_tuning_protocol import load_result, task_path
from experiments.storage import require_report_output

DISPLAY={'mirepnet':'MIRepNet','cbramod':'CBraMod','neurogpt':'NeuroGPT',
         'ifnet':'IFNet','eegnet':'EEGNet','adfcnn':'ADFCNN'}
LABELS={'BNCI2014004':'004','BNCI2014001':'001','BNCI2014001-4':'001-4','BNCI2015001':'15001','AlexMI':'AlexMI'}
METHODS=['none','ds_only','none_matched','ds_only_matched','none_default','ds_only_default','dml_ds_default','dml_ds_selected']


def csv_write(path,header,rows):
    path=require_report_output(path);tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(header);writer.writerows(rows)
    tmp.replace(path)


def audit(cfg,root,aliases):
    values={};records=[];splits={};initial={};policies={};seen={}
    for alias,spec in aliases.items():
        result=load_result(root,spec);folder=task_path(root,spec)
        key=str(folder)
        if key not in seen:
            ident=result['identity']
            if result['evaluation_scope']!='test' or spec['phase']!='final':raise ValueError('Report requires final tests')
            with np.load(folder/'predictions.npz',allow_pickle=False) as archive:
                from experiments.bidir.run_mirepnet_fewshot_8_30 import uid_hash
                uids=archive['uids'].tolist();y=archive['y']
                if len(set(uids))!=len(uids) or len(uids)!=result['evaluation_count'] or uid_hash(uids)!=ident['evaluation_uid_sha256']:
                    raise ValueError('Prediction UID mismatch')
                split_key=(spec['dataset'],spec['subject'],spec['seed'])
                identity=(ident['source_manifest_sha256'],ident['train_uid_sha256'],ident['evaluation_uid_sha256'],tuple(y.tolist()))
                if split_key in splits and splits[split_key]!=identity:raise ValueError('Different report trial sets')
                splits[split_key]=identity
                seen[key]={}
                for model,correct in result['correct'].items():
                    if len(archive[model])!=len(y) or int(np.count_nonzero(archive[model]==y))!=correct:
                        raise ValueError('Prediction accuracy mismatch')
                    model_key=split_key+(model,)
                    init=ident['initial_hashes'][model];policy=ident['preprocessing'][model]
                    if model_key in initial and initial[model_key]!=init:raise ValueError('Unequal initial states')
                    if model_key in policies and policies[model_key]!=policy:raise ValueError('Unequal model preprocessing')
                    initial[model_key]=init;policies[model_key]=policy
                    seen[key][model]=Fraction(correct*100,len(y))
        for model,value in seen[key].items():
            values[alias,model]=value
            records.append([alias,spec['dataset'],spec['subject'],spec['seed'],model,float(value),str(folder/'evaluation.json')])
    return values,records


def score(cfg,values,dataset,student,method,model):
    if student=='teacher':prefix=f'{dataset}/{method}'
    else:prefix=f'{dataset}/{student}/{method}'
    return sum(sum(values[f'{prefix}/{subject}/{seed}',model] for subject in range(cfg['datasets'][dataset]))/
               cfg['datasets'][dataset] for seed in cfg['seeds'])/len(cfg['seeds'])


def fill_for(value,teacher,student):
    if value>teacher and value>student:return 'FFFFC7CE'
    if value>teacher:return 'FFC6EFCE'
    if value>student:return 'FFFFF2CC'
    return None


def workbook(cfg,values,teacher):
    book=openpyxl.Workbook();book.remove(book.active)
    for role in ('student','teacher'):
        sheet=book.create_sheet('学生结果' if role=='student' else '教师结果')
        sheet.sheet_view.showGridLines=False;sheet.freeze_panes='C3'
        sheet.column_dimensions['A'].width=29;sheet.column_dimensions['B'].width=25
        for c in range(3,8):sheet.column_dimensions[get_column_letter(c)].width=15
        sheet.column_dimensions['H'].width=3;sheet.column_dimensions['I'].width=40
        for c,label in enumerate(['student','method']+[LABELS[d] for d in cfg['datasets']],1):
            cell=sheet.cell(1,c,label);cell.fill=PatternFill('solid',fgColor='FF1F4E78')
            cell.font=Font(bold=True,color='FFFFFFFF');cell.alignment=Alignment(horizontal='center')
        def row(r,name,method,fill=None):
            sheet.cell(r,1,name);sheet.cell(r,2,method);sheet.row_dimensions[r].height=23
            for c in range(1,8):
                cell=sheet.cell(r,c);cell.alignment=Alignment(horizontal='center',vertical='center')
                if fill:cell.fill=PatternFill('solid',fgColor=fill)
        row(2,DISPLAY[teacher],'none','FFFAC090')
        for col,dataset in enumerate(cfg['datasets'],3):
            sheet.cell(2,col,float(score(cfg,values,dataset,'teacher','teacher_none',teacher))).number_format='0.00'
        methods=METHODS if role=='student' else ['none','ds_only','dml_ds_default','dml_ds_selected']
        r=3
        for student in cfg['students']:
            for method in methods:
                label=DISPLAY[student] if role=='student' or method=='none' else f'{DISPLAY[teacher]}（{DISPLAY[student]}）'
                baseline=method.startswith('none')
                row(r,label,method,'FF93CDDD' if baseline else None)
                for col,dataset in enumerate(cfg['datasets'],3):
                    teacher_score=score(cfg,values,dataset,'teacher','teacher_none',teacher)
                    student_score=score(cfg,values,dataset,student,'none',student)
                    if role=='student' or method=='none':v=score(cfg,values,dataset,student,method,student)
                    elif method=='ds_only':v=score(cfg,values,dataset,'teacher','teacher_ds_only',teacher)
                    else:v=score(cfg,values,dataset,student,method,teacher)
                    cell=sheet.cell(r,col,float(v));cell.number_format='0.00'
                    fill=fill_for(v,teacher_score,student_score)
                    if not baseline and fill:cell.fill=PatternFill('solid',fgColor=fill)
                r+=1
        for rr,color,text in [(2,'FFFFF2CC','大于学生 none，未大于教师 none'),(3,'FFC6EFCE','大于教师 none，未大于学生 none'),(4,'FFFFC7CE','大于教师与学生两者的 none')]:
            cell=sheet.cell(rr,9,text);cell.fill=PatternFill('solid',fgColor=color)
            cell.alignment=Alignment(horizontal='left',wrap_text=True)
        notes=[
            'Accuracy (%)：每种子内受试者等权平均，再对种子 666、667、668 等权平均。',
            'none、ds_only 分别为各方法通过 A0–A4 验证选参后的结果；matched 使用最终 DML 所选 A*。',
            'default 使用 A0+B0；selected 使用内部验证集选择的 A*+B*。超参数不使用测试准确率选择。',
            '固定 micro-batch=4，教师与学生按各自有效 batch 累积梯度；尾批按实际样本数归一化。',
            '搜索仅评估校准集内部的 20% 验证数据；最终从头训练完整 30% 校准集并评估 70% 测试集。',
            '教师 none 为唯一监督参考；方法颜色严格比较教师 none 与该学生独立调优的 none，使用显示前精度。',
            'MIRepNet 使用 8–30 Hz 滤波及仅在当前训练集拟合的 EA；其他模型沿用各自输入要求。',
            '这些测试数据此前已被查看，结果属于开发性调参；教师训练轮数 MIRepNet/CBraMod=20，NeuroGPT=100。',
        ]
        for rr,note in enumerate(notes,r+2):
            sheet.merge_cells(start_row=rr,start_column=1,end_row=rr,end_column=7)
            cell=sheet.cell(rr,1,note);cell.alignment=Alignment(horizontal='left',vertical='center',wrap_text=True)
            sheet.row_dimensions[rr].height=31
    return book


def export(cfg,root,report,teacher,aliases):
    values,records=audit(cfg,root,aliases)
    csv_write(report/f'{teacher}_subjects.csv',['alias','dataset','subject','seed','model','accuracy_percent','source'],records)
    rows=[]
    for dataset in cfg['datasets']:
        rows.append([dataset,'teacher','none',teacher,float(score(cfg,values,dataset,'teacher','teacher_none',teacher))])
        for student in cfg['students']:
            for method in METHODS:
                rows.append([dataset,student,method,student,float(score(cfg,values,dataset,student,method,student))])
                if method.startswith('dml_ds'):
                    rows.append([dataset,student,method,teacher,float(score(cfg,values,dataset,student,method,teacher))])
    csv_write(report/f'{teacher}_summary.csv',['dataset','student','method','model','accuracy_percent'],rows)
    target=require_report_output(report/f'{teacher}_tuning_3seed.xlsx');workbook(cfg,values,teacher).save(target)


def statistics(cfg,root,report):
    output=[]
    for teacher in cfg['teachers']:
        manifest=json.loads((root/f'final_manifest_{teacher}.json').read_text())
        values,_=audit(cfg,root,manifest['aliases'])
        for dataset,n in cfg['datasets'].items():
            for student in cfg['students']:
                for baseline in ('none','ds_only','ds_only_matched','dml_ds_default'):
                    differences=[]
                    for subject in range(n):
                        differences.append(float(sum(values[f'{dataset}/{student}/dml_ds_selected/{subject}/{seed}',student]-
                            values[f'{dataset}/{student}/{baseline}/{subject}/{seed}',student] for seed in cfg['seeds'])/len(cfg['seeds'])))
                    d=np.asarray(differences);rng=np.random.default_rng(81173)
                    means=d[rng.integers(0,n,size=(10000,n))].mean(axis=1)
                    low,high=np.quantile(means,[0.025,0.975]);p=1.0 if np.all(d==0) else float(wilcoxon(d).pvalue)
                    output.append({'teacher':teacher,'dataset':dataset,'student':student,'baseline':baseline,
                        'mean_delta_pp':float(d.mean()),'ci95_low_pp':float(low),'ci95_high_pp':float(high),
                        'wins':int((d>0).sum()),'ties':int((d==0).sum()),'losses':int((d<0).sum()),'n_subjects':n,'p_value':p})
    for baseline in ('none','ds_only','ds_only_matched','dml_ds_default'):
        family=sorted([r for r in output if r['baseline']==baseline],key=lambda r:r['p_value']);previous=0.0
        for i,row in enumerate(family):
            previous=max(previous,min(1.0,(len(family)-i)*row['p_value']));row['p_holm']=previous
    columns=list(output[0]);csv_write(report/'paired_statistics.csv',columns,[[row[c] for c in columns] for row in output])
