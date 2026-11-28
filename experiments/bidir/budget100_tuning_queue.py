"""Resume-safe A/B search and final refits on process-count-limited GPUs."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'

import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import config
from data.loso_source import sha256_file
from experiments.bidir.queue_dcm_loso import gpu_processes, job_is_running
from experiments.bidir.budget100_tuning_protocol import (
    ROOT, cfg_load, choose, digest, fingerprint, group_specs, install_task, load_result,
    make_task, mean_score, report_json, roots, task_path, write_json)

MODULE='experiments.bidir.budget100_tuning_worker'


def active_workers():
    workers={}
    for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
        parts=line.split()
        if MODULE in parts and '--gpu' in parts:
            workers[int(parts[0])]=int(parts[parts.index('--gpu')+1])
    return workers


def free_slots(cfg):
    from experiments.distill.queue import query_gpus
    active=active_workers(); slots=[]
    for gpu,info in sorted(query_gpus().items()):
        if info['total_mb']-info['used_mb'] < 4096: continue
        reserved=sum(g==gpu and pid not in info['pids'] for pid,g in active.items())
        slots.extend([gpu]*max(0,cfg['gpu_process_limit']-len(info['pids'])-reserved))
    return slots[:max(0,cfg['max_workers']-len(active))]


def prior_20epoch_tasks(cfg, root, report, frozen_hash, teacher):
    """Replay every completed old 20-epoch candidate before the new search.

    Search specs coincide with the new 100-epoch search cache. Old final
    selections are evaluated as fixed settings, never called 100-epoch optima.
    The later validation-only stages select and refit fresh 100-epoch optima.
    """
    import copy
    old=Path('/data1/llx/BigSmallcollab/results/bidir/fewshot_dml_ds_tuning_v1/tasks')
    specs=[];sources=[]
    for path in sorted(old.glob('*/evaluation.json')):
        doc=json.loads(path.read_text()); original=doc.get('task',{})
        if (original.get('teacher')!=teacher or original.get('smoke') or
                original.get('teacher_training',{}).get('epochs')!=20): continue
        spec=copy.deepcopy(original)
        spec['teacher_training']['epochs']=100
        spec['experiment']=cfg['experiment_id'];spec['frozen_hash']=frozen_hash
        if spec['phase']=='final':spec['phase']='replay_fixed_final'
        specs.append(spec)
        sources.append({'source20':str(path),'source_sha256':sha256_file(path),
                        'new_task':digest(spec),'phase':spec['phase']})
    report_json(report,f'replay_20epoch_sources_{teacher}.json',sources)
    return specs


def spawn(folder,gpu,stop=None):
    cmd=[sys.executable,'-u','-B','-m',MODULE,'--folder',str(folder),'--gpu',str(gpu)]
    if stop:cmd+=['--stop-after',str(stop)]
    log=(folder/'worker.log').open('a')
    try:p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
    except Exception:log.close();raise
    return p,log


def run_batch(cfg,root,report,frozen,specs,label):
    unique={digest(s):s for s in specs};pending=[];done=0
    for key,spec in unique.items():
        folder=install_task(root,spec)
        if (folder/'evaluation.json').exists():load_result(root,spec);done+=1
        else:pending.append(spec)
    processes={};retries={};failures=[];last_check=0
    while pending or processes:
        for pid,(process,log,spec) in list(processes.items()):
            code=process.poll()
            if code is None:continue
            log.close();del processes[pid];key=digest(spec)
            if code:
                retries[key]=retries.get(key,0)+1
                if retries[key]<=2:pending.append(spec)
                else:failures.append({'task':key,'exit_code':code,'log':str(task_path(root,spec)/'worker.log')})
            else:load_result(root,spec);done+=1
        if time.monotonic()-last_check>30:
            if fingerprint()!=frozen['code']:raise RuntimeError('Frozen experiment source changed; restart only after resolving provenance')
            last_check=time.monotonic()
        # Recover completed workers that survived a previous controller.
        remaining=[]
        for spec in pending:
            folder=task_path(root,spec)
            if (folder/'evaluation.json').exists():load_result(root,spec);done+=1
            else:remaining.append(spec)
        pending=remaining
        try:slots=free_slots(cfg)
        except subprocess.SubprocessError:slots=[]
        for gpu in slots:
            index=next((i for i,s in enumerate(pending) if not job_is_running(task_path(root,s))),None)
            if index is None:break
            spec=pending.pop(index);folder=task_path(root,spec);process,log=spawn(folder,gpu)
            processes[process.pid]=(process,log,spec)
            print(f"started {label} {spec['teacher']} {spec['dataset']} {spec['student']} {spec['method']} "
                  f"subject={spec['subject']} seed={spec['seed']} gpu={gpu} task={digest(spec)[:10]}",flush=True)
        report_json(report,'status.json',{'stage':label,'completed_in_stage':done,'total_in_stage':len(unique),
            'pending':len(pending),'active':active_workers(),'failed_tasks':failures,'updated_at_unix':time.time()})
        if pending or processes:time.sleep(5)
    if failures:raise RuntimeError(f'{len(failures)} tasks failed after retries; see status.json')


def preflight(cfg,root,report,frozen,teacher):
    mark=report/f'preflight_{teacher}.json'
    if mark.exists():
        if json.loads(mark.read_text())['frozen_hash']!=digest(frozen):raise ValueError('Preflight source mismatch')
        return
    h=digest(frozen);dataset='BNCI2014001-4';cases=[]
    for student in cfg['students']:
        cases.append(make_task(cfg,h,teacher,dataset,0,666,'pair','dml_ds',student,smoke=True))
    cases.append(make_task(cfg,h,teacher,dataset,0,666,'pair','dml_ds','ifnet','A4',smoke=True))
    cases.append(make_task(cfg,h,teacher,dataset,0,666,'teacher','none',smoke=True))
    cases.append(make_task(cfg,h,teacher,dataset,0,666,'student','ds_only','ifnet',smoke=True))
    cases.append(make_task(cfg,h,teacher,dataset,0,666,'teacher','ds_only',smoke=True))
    cases.append(make_task(cfg,h,teacher,dataset,0,666,'pair','ds_only','ifnet',smoke=True))
    for spec in cases:
        folder=install_task(root,spec)
        if (folder/'evaluation.json').exists():continue
        while True:
            slots=free_slots(cfg)
            if slots:break
            report_json(report,'status.json',{'stage':'preflight_waiting_for_gpu','teacher':teacher});time.sleep(10)
        process,log=spawn(folder,slots[0]);report_json(report,'status.json',{'stage':'preflight',
            'teacher':teacher,'student':spec['student'],'method':spec['method'],'pid':process.pid,'gpu':slots[0]})
        code=process.wait();log.close()
        if code:raise RuntimeError(f'Preflight failed: {folder}/worker.log')
    control=cases[3];folder=install_task(root,control,'resume')
    for stop in (1,None):
        if (folder/'evaluation.json').exists():break
        while not (slots:=free_slots(cfg)):time.sleep(10)
        process,log=spawn(folder,slots[0],stop);code=process.wait();log.close()
        if code:raise RuntimeError(f'Resume preflight failed: {folder}')
    a=load_result(root,control);b=json.loads((folder/'evaluation.json').read_text())
    if a!=b:raise RuntimeError('Resume result or final state mismatch')
    with np.load(task_path(root,control)/'predictions.npz',allow_pickle=False) as x, np.load(folder/'predictions.npz',allow_pickle=False) as y:
        if any(not np.array_equal(x[k],y[k]) for k in x.files):raise RuntimeError('Resume predictions mismatch')
    teacher_only=load_result(root,cases[4]);ds_student=load_result(root,cases[5])
    for spec in cases[:4]:
        result=load_result(root,spec)
        if result['identity']['initial_hashes'][teacher]!=teacher_only['identity']['initial_hashes'][teacher]:
            raise RuntimeError('Teacher initialization depends on student or candidate')
        if any(result['identity'][k]!=teacher_only['identity'][k] for k in ('train_uid_sha256','evaluation_uid_sha256')):
            raise RuntimeError('Trial alignment mismatch')
        if spec['student']=='ifnet' and result['identity']['initial_hashes']['ifnet']!=ds_student['identity']['initial_hashes']['ifnet']:
            raise RuntimeError('Student initialization mismatch')
    standalone=load_result(root,cases[6]);paired=load_result(root,cases[7])
    if standalone['final_state_hashes'][teacher]!=paired['final_state_hashes'][teacher]:
        raise RuntimeError('Independent teacher DS training changed with paired student')
    report_json(report,f'preflight_{teacher}.json',{'status':'passed','teacher':teacher,'frozen_hash':h,
        'checks':['real gradients/logits','independent accumulation and tail updates','same trial and initialization',
                  'teacher DS independence','exact interrupted resume','validation-only evaluation']})


def stage_a_tasks(cfg,h,teacher):
    specs=[]
    for dataset,count in cfg['datasets'].items():
        for subject in range(count):
            for seed in cfg['seeds']:
                for method in ('none','ds_only'):
                    specs.append(make_task(cfg,h,teacher,dataset,subject,seed,'teacher',method))
        for student in cfg['students']:
            for a in cfg['stage_a']:
                for method in ('none','ds_only','dml_ds'):
                    specs.extend(group_specs(cfg,h,teacher,dataset,student,method,a))
    return specs


def select_a(cfg,root,report,h,teacher):
    selection={}
    for dataset in cfg['datasets']:
        for student in cfg['students']:
            key=f'{dataset}/{student}';selection[key]={}
            for method in ('none','ds_only','dml_ds'):
                scores={a:mean_score(cfg,root,group_specs(cfg,h,teacher,dataset,student,method,a),student) for a in cfg['stage_a']}
                selection[key][method]={'best':choose(list(cfg['stage_a']),scores),
                    'validation':{a:{'numerator':v.numerator,'denominator':v.denominator,'accuracy_percent':float(v)} for a,v in scores.items()}}
    report_json(report,f'selection_A_{teacher}.json',selection)
    return selection


def stage_b_tasks(cfg,h,teacher,selected_a):
    return [spec for dataset in cfg['datasets'] for student in cfg['students']
            for b in cfg['stage_b'] for spec in group_specs(cfg,h,teacher,dataset,student,'dml_ds',
                selected_a[f'{dataset}/{student}']['dml_ds']['best'],b)]


def select_b(cfg,root,report,h,teacher,selected_a):
    selection={}
    for dataset in cfg['datasets']:
        for student in cfg['students']:
            key=f'{dataset}/{student}';a=selected_a[key]['dml_ds']['best']
            scores={b:mean_score(cfg,root,group_specs(cfg,h,teacher,dataset,student,'dml_ds',a,b),student) for b in cfg['stage_b']}
            selection[key]={'a':a,'b':choose(list(cfg['stage_b']),scores),
                'none_a':selected_a[key]['none']['best'],'ds_a':selected_a[key]['ds_only']['best'],
                'validation':{b:{'numerator':v.numerator,'denominator':v.denominator,'accuracy_percent':float(v)} for b,v in scores.items()}}
    report_json(report,f'selection_B_{teacher}.json',selection)
    return selection


def final_tasks(cfg,h,teacher,selection):
    specs=[];aliases={};lock=digest(selection)
    for dataset,count in cfg['datasets'].items():
        for subject in range(count):
            for seed in cfg['seeds']:
                for method in ('none','ds_only'):
                    spec=make_task(cfg,h,teacher,dataset,subject,seed,'teacher',method,phase='final',selection_hash=lock)
                    specs.append(spec);aliases[f'{dataset}/teacher_{method}/{subject}/{seed}']=spec
        for student in cfg['students']:
            choice=selection[f'{dataset}/{student}']
            options=[('none','none',choice['none_a'],'B0'),('ds_only','ds_only',choice['ds_a'],'B0'),
                     ('none_matched','none',choice['a'],'B0'),('ds_only_matched','ds_only',choice['a'],'B0'),
                     ('none_default','none','A0','B0'),('ds_only_default','ds_only','A0','B0'),
                     ('dml_ds_default','dml_ds','A0','B0'),('dml_ds_selected','dml_ds',choice['a'],choice['b'])]
            for alias,method,a,b in options:
                group=group_specs(cfg,h,teacher,dataset,student,method,a,b,'final',lock)
                specs.extend(group)
                for spec in group:aliases[f"{dataset}/{student}/{alias}/{spec['subject']}/{spec['seed']}"]=spec
    return specs,aliases


def main():
    cfg=cfg_load();root,report=roots(cfg);root.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            frozen_path=root/'frozen.json'
            if frozen_path.exists():
                frozen=json.loads(frozen_path.read_text())
                if frozen['code']!=fingerprint() or frozen['config']!=cfg:raise ValueError('Frozen run differs from source')
            else:
                weights={}
                for name in cfg['teachers']:
                    path=Path(config.weight_path(name));stat=path.stat()
                    weights[name]={'path':str(path),'sha256':sha256_file(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
                frozen={'code':fingerprint(),'config':cfg,'weights':weights};write_json(frozen_path,frozen)
            h=digest(frozen)
            report_json(report,'protocol.json',{'config':cfg,'frozen_hash':h,'selection':'student_validation_accuracy_only',
                'tie_break':'fixed_candidate_order','test_status':'previously_observed_development_data',
                'storage':str(root)})
            completed=[]
            for teacher in cfg['teachers']:
                preflight(cfg,root,report,frozen,teacher)
                legacy=prior_20epoch_tasks(cfg,root,report,h,teacher)
                if legacy:run_batch(cfg,root,report,frozen,legacy,f'{teacher}/replay_completed_20epoch_candidates')
                run_batch(cfg,root,report,frozen,stage_a_tasks(cfg,h,teacher),f'{teacher}/stage_A')
                selected_a=select_a(cfg,root,report,h,teacher)
                run_batch(cfg,root,report,frozen,stage_b_tasks(cfg,h,teacher,selected_a),f'{teacher}/stage_B')
                selection=select_b(cfg,root,report,h,teacher,selected_a)
                specs,aliases=final_tasks(cfg,h,teacher,selection)
                write_json(root/f'final_manifest_{teacher}.json',{'selection':selection,'aliases':aliases,'frozen_hash':h})
                run_batch(cfg,root,report,frozen,specs,f'{teacher}/final_refits')
                from experiments.bidir.budget100_tuning_report import export
                export(cfg,root,report,teacher,aliases);completed.append(teacher)
                report_json(report,'completed_teachers.json',completed)
            from experiments.bidir.budget100_tuning_report import statistics
            statistics(cfg,root,report)
            report_json(report,'status.json',{'stage':'completed','teachers':completed,'frozen_hash':h})
        except Exception as exc:
            report_json(report,'error.json',{'error':f'{type(exc).__name__}: {exc}','time_unix':time.time()})
            report_json(report,'status.json',{'stage':'failed','error':f'{type(exc).__name__}: {exc}',
                'active':active_workers(),'updated_at_unix':time.time()})
            raise


if __name__=='__main__':main()
